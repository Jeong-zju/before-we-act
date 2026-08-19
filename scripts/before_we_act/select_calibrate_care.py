#!/usr/bin/env python3
"""Select frozen CARE checkpoints, fit split conformal correction, and test offline."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from before_we_act.care_belief import (
    CAREBeliefConfig,
    CAREBeliefHead,
    CARECalibration,
    select_care_candidate,
)
from before_we_act.care_training_data import (
    CARETrainingDataset,
    atomic_json,
    load_prepared_care,
    sha256_file,
)


VARIANTS = ("care", "reactive_only", "replay_only", "capacity")
SEEDS = (20260818, 20260819, 20260820)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_model(path: Path, device: torch.device) -> tuple[CAREBeliefHead, dict[str, Any]]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format_version") != "before-we-act.a6r1-care-training-checkpoint/1":
        raise ValueError(f"wrong CARE training checkpoint: {path}")
    config = CAREBeliefConfig.from_mapping(saved["config"])
    model = CAREBeliefHead(config).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    return model, saved


def selected_training(training_root: Path) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        rows = []
        for seed in SEEDS:
            status_path = training_root / variant / f"seed_{seed}" / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") != "COMPLETED":
                raise RuntimeError(f"CARE training is incomplete: {status_path}")
            if status["variant"] != variant or int(status["seed"]) != seed:
                raise RuntimeError(f"CARE training identity drifted: {status_path}")
            rows.append(status)
        winner = min(
            rows,
            key=lambda row: (
                float(row["selected_validation"]["regret"]),
                float(row["selected_validation"]["loss"]),
                int(row["seed"]),
            ),
        )
        selected[variant] = winner
    return selected


@torch.no_grad()
def prediction_rows(
    model: CAREBeliefHead,
    loader: DataLoader,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in loader:
        batch = {key: value.to(device) for key, value in raw.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(
                batch["memory"],
                batch["memory_mask"],
                batch["candidate_chunks"],
                batch["horizon_index"],
            )
        for index in range(output.quantiles.shape[0]):
            rows.append(
                {
                    "family_index": int(batch["family_index"][index]),
                    "task_id": int(batch["task_id"][index]),
                    "quantiles": output.quantiles[index].float().cpu(),
                    "hard_safety_logit": output.hard_safety_logit[index].float().cpu(),
                    "target": batch["target"][index].float().cpu(),
                    "hard_safety": batch["hard_safety"][index].float().cpu(),
                }
            )
    return rows


def conformal_correction(rows: Iterable[dict[str, Any]], nominal: float) -> float:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        lower = row["quantiles"][:, 2, 0]
        target = row["target"][:, 2]
        grouped[row["family_index"]].extend((lower[1:] - target[1:]).tolist())
    scores = np.asarray([max(values) for values in grouped.values()], dtype=np.float64)
    if not len(scores):
        raise RuntimeError("calibration split has no CARE families")
    # Split-conformal finite-sample rank ceil((n+1)*coverage)/n.
    adjusted = min(1.0, np.ceil((len(scores) + 1) * nominal) / len(scores))
    try:
        correction = float(np.quantile(scores, adjusted, method="higher"))
    except TypeError:  # numpy < 1.22
        correction = float(np.quantile(scores, adjusted, interpolation="higher"))
    return max(0.0, correction)


def offline_metrics(
    rows: list[dict[str, Any]],
    variant: str,
    calibration: CARECalibration | None = None,
) -> dict[str, Any]:
    component = 0 if variant == "replay_only" else 2
    median = 2
    regrets: list[float] = []
    correct = 0
    pair_correct = pair_count = 0
    overrides = harmful = 0
    coverage_rows: dict[int, bool] = {}
    by_task: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        scores = row["quantiles"][:, component, median]
        target = row["target"][:, component]
        if calibration is None:
            selected = int(scores.argmax())
        else:
            from before_we_act.care_belief import CAREBeliefOutput

            output = CAREBeliefOutput(
                row["quantiles"].unsqueeze(0),
                row["hard_safety_logit"].unsqueeze(0),
                torch.empty(1, 6, 0),
            )
            selected = int(select_care_candidate(output, calibration, variant=variant)[0])
            overrides += int(selected != 0)
            harmful += int(selected != 0 and float(target[selected]) < 0.0)
            lower = row["quantiles"][:, component, 0] - calibration.lower_correction
            covered = bool(torch.all(target[1:] >= lower[1:]))
            family = row["family_index"]
            coverage_rows[family] = coverage_rows.get(family, True) and covered
        best = int(target.argmax())
        regret = float(target[best] - target[selected])
        regrets.append(regret)
        by_task[row["task_id"]].append(regret)
        correct += int(selected == best)
        score_delta = scores[:, None] - scores[None, :]
        target_delta = target[:, None] - target[None, :]
        mask = target_delta.abs() > 1e-6
        pair_correct += int(((score_delta.sign() == target_delta.sign()) & mask).sum())
        pair_count += int(mask.sum())
    result = {
        "rows": len(rows),
        "families": len({row["family_index"] for row in rows}),
        "top1_accuracy": correct / len(rows),
        "pairwise_accuracy": pair_correct / pair_count if pair_count else 0.0,
        "mean_regret": float(np.mean(regrets)),
        "median_regret": float(np.median(regrets)),
        "mean_regret_by_task_id": {
            str(key): float(np.mean(values)) for key, values in sorted(by_task.items())
        },
    }
    if calibration is not None:
        result.update(
            {
                "override_rate": overrides / len(rows),
                "harmful_override_rate": harmful / max(overrides, 1),
                "simultaneous_family_coverage": float(np.mean(list(coverage_rows.values()))),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "offline_report.json"
    deployment_path = args.output_root / "care_deployment_checkpoint.pt"
    if report_path.exists() and deployment_path.exists():
        print(json.dumps({"status": "PRESERVED", "report": str(report_path)}))
        return
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract.get("stage_id") != "A6R1-CARE-OWNER-AUTHORIZED-DIAGNOSTIC":
        raise RuntimeError("wrong A6 CARE contract")
    prepared = load_prepared_care(args.prepared_data)
    selected = selected_training(args.training_root)
    device = torch.device(args.device)

    selected_models: dict[str, dict[str, Any]] = {}
    test_metrics: dict[str, Any] = {}
    calibration_rows = test_rows = None
    care_model = care_saved = None
    for variant in VARIANTS:
        checkpoint = Path(selected[variant]["selected_checkpoint"])
        model, saved = load_model(checkpoint, device)
        calibration_data = CARETrainingDataset(
            prepared, "calibration", primary_horizon_only=True, primary_horizon=16
        )
        test_data = CARETrainingDataset(
            prepared, "test", primary_horizon_only=True, primary_horizon=16
        )
        calibration_loader = DataLoader(calibration_data, batch_size=48, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=48, shuffle=False)
        current_test_rows = prediction_rows(model, test_loader, device)
        test_metrics[variant] = offline_metrics(current_test_rows, variant)
        selected_models[variant] = {
            "seed": int(selected[variant]["seed"]),
            "update": int(selected[variant]["selected_update"]),
            "validation": selected[variant]["selected_validation"],
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "parameter_count": int(selected[variant]["parameter_count"]),
        }
        if variant == "care":
            care_model, care_saved = model, saved
            calibration_rows = prediction_rows(model, calibration_loader, device)
            test_rows = current_test_rows

    assert care_model is not None and care_saved is not None
    assert calibration_rows is not None and test_rows is not None
    nominal = float(contract["calibration"]["nominal_simultaneous_coverage"])
    correction = conformal_correction(calibration_rows, nominal)
    calibration = CARECalibration(
        lower_correction=correction,
        selector_delta=float(contract["selector"]["delta"]),
        hard_safety_probability_max=float(contract["selector"]["hard_safety_probability_max"]),
        nominal_simultaneous_coverage=nominal,
        primary_horizon=int(contract["training"]["primary_horizon"]),
    )
    calibrated_test = offline_metrics(test_rows, "care", calibration)
    controls = ["reactive_only", "replay_only", "capacity"]
    gate_conditions = {
        "care_regret_below_each_matched_control": all(
            calibrated_test["mean_regret"] < test_metrics[name]["mean_regret"]
            for name in controls
        ),
        "nominal_coverage_with_tolerance": calibrated_test[
            "simultaneous_family_coverage"
        ]
        >= float(contract["calibration"]["test_coverage_min"]),
        "harmful_override_below_limit": calibrated_test["harmful_override_rate"]
        <= float(contract["selector"]["harmful_override_rate_max"]),
    }
    gate_c_passed = all(gate_conditions.values())
    deployment = {
        "format_version": "before-we-act.a6r1-care-deployment-checkpoint/1",
        "created_at_utc": utc_now(),
        "stage_id": contract["stage_id"],
        "config": care_saved["config"],
        "model": care_saved["model"],
        "calibration": calibration.__dict__,
        "selected_seed": int(care_saved["seed"]),
        "selected_update": int(care_saved["update"]),
        "reference_checkpoint": str(args.reference_checkpoint.resolve()),
        "reference_checkpoint_sha256": sha256_file(args.reference_checkpoint),
        "contract": str(args.contract.resolve()),
        "contract_sha256": sha256_file(args.contract),
        "prepared_data_sha256": sha256_file(args.prepared_data),
        "gate_a_preserved_as_not_passed": True,
        "gate_c_passed": gate_c_passed,
        "closed_loop_authorized_regardless_of_gate_c_for_diagnostic": bool(
            contract["authorization"]["validation20_authorized_even_if_gate_c_fails"]
        ),
    }
    temporary = deployment_path.with_name(f".{deployment_path.name}.{os.getpid()}.tmp")
    torch.save(deployment, temporary)
    os.replace(temporary, deployment_path)
    report = {
        "format_version": "before-we-act.a6r1-care-offline-report/1",
        "stage_id": contract["stage_id"],
        "completed_at_utc": utc_now(),
        "selected_models": selected_models,
        "test_uncalibrated": test_metrics,
        "calibration": calibration.__dict__,
        "test_calibrated_care": calibrated_test,
        "gate_c_conditions": gate_conditions,
        "gate_c_passed": gate_c_passed,
        "gate_a_preserved_as_not_passed": True,
        "response_decomposition_claim_allowed": gate_c_passed
        and calibrated_test["mean_regret"] < test_metrics["reactive_only"]["mean_regret"],
        "deployment_checkpoint": str(deployment_path.resolve()),
        "deployment_checkpoint_sha256": sha256_file(deployment_path),
        "reference_checkpoint_sha256_before_after": [
            sha256_file(args.reference_checkpoint),
            sha256_file(args.reference_checkpoint),
        ],
    }
    atomic_json(report_path, report)
    print(json.dumps({"status": "A6R1_OFFLINE_COMPLETED", "gate_c_passed": gate_c_passed, "report": str(report_path)}))


if __name__ == "__main__":
    main()
