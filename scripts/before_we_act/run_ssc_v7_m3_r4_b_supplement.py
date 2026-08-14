#!/usr/bin/env python3
"""Run matched R4-B controls under the frozen owner signal-first amendment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts.before_we_act import run_ssc_v7_m3 as m3  # noqa: E402
from scripts.before_we_act import run_ssc_v7_m3_r4 as r4  # noqa: E402
from scripts.before_we_act import run_ssc_v7_m3_r4_b as r4b  # noqa: E402
from scripts.before_we_act import run_ssc_v7_m3_r4_successor as successor  # noqa: E402


STAGE_ID = "SSC-V7-M3-R4-B-SUPPLEMENT"
FROZEN_STATUS = "FROZEN_BEFORE_SUPPLEMENT_CONTROL_METRICS"
CONDITIONS = (
    "hc_hidden_only_direct",
    "phase_matched_row_shuffle_direct",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train-branch", "aggregate"))
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed-index", type=int, choices=(0, 1, 2))
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_gate(path: Path) -> dict[str, Any]:
    gate = read_json(path)
    if gate.get("stage_id") != STAGE_ID or gate.get("status") != FROZEN_STATUS:
        raise RuntimeError("supplement gate identity/status is not frozen")
    unsigned = {key: value for key, value in gate.items() if key != "integrity"}
    expected = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    if expected != str(gate["integrity"]["payload_sha256"]):
        raise RuntimeError("supplement gate payload hash mismatch")
    gate["_runtime_gate_sha256"] = sha256_file(path)
    return gate


def load_source(gate: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source = gate["source"]
    original_gate_path = Path(str(source["r4_b_gate"]))
    if sha256_file(original_gate_path) != str(source["r4_b_gate_sha256"]):
        raise RuntimeError("source R4-B gate hash mismatch")
    original_gate = r4b.load_gate(original_gate_path)
    r4b.preflight(original_gate)
    receipt_path = Path(str(source["r4_b_receipt"]))
    if sha256_file(receipt_path) != str(source["r4_b_receipt_sha256"]):
        raise RuntimeError("source R4-B receipt hash mismatch")
    receipt = read_json(receipt_path)
    if not receipt.get("r4_b_completed") or int(receipt.get("test_paths_opened", -1)) != 0:
        raise RuntimeError("source R4-B receipt is not a sealed completed source")
    return original_gate, receipt


def preflight(gate: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    implementation = gate["implementation"]
    checks = {
        "script": sha256_file(REPOSITORY / str(implementation["script"]))
        == str(implementation["script_sha256"]),
        "test": sha256_file(REPOSITORY / str(implementation["test"]))
        == str(implementation["test_sha256"]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"supplement implementation preflight failed: {checks}")
    return load_source(gate)


def phase_bins(data: Any) -> np.ndarray:
    ratios = np.asarray(data.time[:, 0], dtype=np.float32)
    return np.minimum(3, np.floor(np.clip(ratios, 0.0, 1.0) * 4.0).astype(int))


def phase_matched_shuffle(values: np.ndarray, data: Any, seed: int) -> np.ndarray:
    """Derange rows within task and the frozen four-bin phase."""
    result = np.empty_like(values)
    bins = phase_bins(data)
    effective_bins = bins.copy()
    for task in r4b.TASKS:
        task_mask = data.tasks == task
        counts = np.asarray(
            [np.sum(task_mask & (effective_bins == phase)) for phase in range(4)]
        )
        for phase in np.flatnonzero(counts == 1):
            destinations = np.flatnonzero(counts >= 2)
            if len(destinations) == 0:
                raise RuntimeError(f"no phase-matched destination: {task}/{phase}")
            destination = int(destinations[np.argmin(np.abs(destinations - phase))])
            effective_bins[task_mask & (effective_bins == phase)] = destination
            counts[destination] += 1
            counts[phase] = 0
    group_count = 0
    for task_index, task in enumerate(r4b.TASKS):
        for phase in range(4):
            indices = np.flatnonzero((data.tasks == task) & (effective_bins == phase))
            if len(indices) == 0:
                continue
            if len(indices) == 1:
                raise RuntimeError(
                    f"phase-matched shuffle group too small: {task}/{phase}"
                )
            shuffled = successor.deranged_indices(
                indices, seed + task_index * 7919 + phase * 37
            )
            result[indices] = values[shuffled]
            group_count += 1
    if group_count == 0:
        raise RuntimeError("phase-matched shuffle produced no groups")
    return result


def condition_features(
    condition: str,
    cached: Mapping[str, np.ndarray],
    bundle: successor.CachedBundle,
    data: r4b.StageData,
    gate: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if condition == "hc_hidden_only_direct":
        features = np.zeros(
            (len(bundle), successor.TOKEN_COUNT, successor.TOKEN_WIDTH),
            dtype=np.float32,
        )
        reliability = np.ones((len(bundle), 1), dtype=np.float32)
        return features, reliability
    if condition == "phase_matched_row_shuffle_direct":
        seed = int(gate["seeds"]["phase_matched_row_shuffle"])
        raw = phase_matched_shuffle(cached["candidate"], bundle.base, seed)
        reliability = phase_matched_shuffle(
            cached["candidate_reliability"], bundle.base, seed
        )
        normalized = (
            raw.reshape(-1, successor.TOKEN_COUNT, successor.TOKEN_WIDTH)
            - data.arb_mean
        ) / data.arb_std
        return normalized.astype(np.float32), reliability.astype(np.float32)
    raise ValueError(condition)


def train_branch(args: argparse.Namespace, gate: Mapping[str, Any]) -> None:
    if args.condition is None or args.seed_index is None:
        raise ValueError("train-branch requires --condition and --seed-index")
    output = (
        args.output_root
        / "formal"
        / "branches"
        / args.condition
        / f"seed_{args.seed_index}"
    )
    if output.exists():
        raise FileExistsError(f"fresh supplement branch output required: {output}")
    output.mkdir(parents=True)

    original_gate, _ = load_source(gate)
    source_root = Path(str(gate["source"]["r4_b_root"]))
    data = r4b.load_stage_data(original_gate)
    train_cache, confirmation_cache, predictor_receipt = r4b.load_predictions(
        source_root, original_gate
    )
    hc_receipt, hc_payload, _ = r4b.load_source_hc(original_gate)
    train_features, train_reliability = condition_features(
        args.condition, train_cache, data.train, data, gate
    )
    confirmation_features, confirmation_reliability = condition_features(
        args.condition, confirmation_cache, data.confirmation, data, gate
    )
    hc_seed = int(original_gate["source"]["hc_noise_seed"])
    train_x = np.concatenate(
        (
            r4.hc_input(data.train, hc_seed),
            train_features.reshape(len(data.train), -1),
            train_reliability,
        ),
        axis=1,
    ).astype(np.float32)
    confirmation_x = np.concatenate(
        (
            r4.hc_input(data.confirmation, hc_seed),
            confirmation_features.reshape(len(data.confirmation), -1),
            confirmation_reliability,
        ),
        axis=1,
    ).astype(np.float32)

    init_seed = int(original_gate["seeds"]["residual_initialization"][args.seed_index])
    sampler_seed = int(original_gate["seeds"]["residual_sampler"][args.seed_index])
    model = successor.DirectResidualFactory.create(hc_payload, init_seed)
    config = original_gate["action_training"]
    model, training = r4.train_residual(
        model,
        train_x,
        data.train.normalized_target,
        data.train.base.target_mask,
        confirmation_x,
        data.confirmation.base,
        data.confirmation.normalized_target,
        args.device,
        float(config["learning_rate"]),
        sampler_seed,
        int(config["max_epochs"]),
        int(config["patience"]),
    )
    checkpoint = output / "action_residual.pt"
    m3.save_torch_checkpoint(
        checkpoint,
        {
            "stage_id": STAGE_ID,
            "condition": args.condition,
            "seed_index": args.seed_index,
            "state_dict": model.state_dict(),
            "input_width": int(train_x.shape[1]),
        },
    )
    metrics = m3.evaluate_model(
        model,
        confirmation_x,
        data.confirmation.base,
        data.confirmation.normalized_target,
        args.device,
    )
    metrics_path = output / "confirmation_metrics.json"
    write_json(metrics_path, metrics)
    receipt = {
        "format_version": "ssc-v7.m3_r4_b.supplement_branch/1",
        "stage_id": STAGE_ID,
        "completed_at_utc": utc_now(),
        "gate_sha256": gate["_runtime_gate_sha256"],
        "source_r4_b_gate_sha256": original_gate["_runtime_gate_sha256"],
        "condition": args.condition,
        "seed_index": args.seed_index,
        "initialization_seed": init_seed,
        "sampler_seed": sampler_seed,
        "hc_checkpoint_sha256": hc_receipt["checkpoint_sha256"],
        "predictor_receipt_sha256": sha256_file(
            Path(str(gate["source"]["r4_b_root"]))
            / "predictors"
            / "predictor_receipt.json"
        ),
        "training": training,
        "strict_gate_eligible": bool(training["converged_by_patience"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "metrics": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "test_paths_opened": 0,
        "r4_c_started": False,
    }
    write_json(output / "branch_receipt.json", receipt)
    print(f"SSC_V7_M3_R4_B_SUPPLEMENT_COMPLETE {args.condition} seed={args.seed_index}")


def load_supplement_results(
    root: Path, gate: Mapping[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    metrics: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    receipts: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for seed_index in range(3):
            branch = root / "formal" / "branches" / condition / f"seed_{seed_index}"
            receipt = read_json(branch / "branch_receipt.json")
            if receipt.get("gate_sha256") != gate["_runtime_gate_sha256"]:
                raise RuntimeError("supplement branch gate hash mismatch")
            path = Path(str(receipt["metrics"]))
            if sha256_file(path) != str(receipt["metrics_sha256"]):
                raise RuntimeError("supplement branch metrics hash mismatch")
            receipts.append(receipt)
            metrics[condition].append(read_json(path))
    return metrics, receipts


def decision_checks(
    source_receipt: Mapping[str, Any],
    supplement_converged: bool,
    supplement_test_paths_opened: int = 0,
) -> dict[str, bool]:
    source_checks = source_receipt["strict_checks"]
    return {
        "source_r4_b_completed": bool(source_receipt["r4_b_completed"]),
        "arb_hat_ci_lower_positive": bool(source_checks["arb_hat_ci_lower_positive"]),
        "at_least_two_positive_tasks": bool(source_checks["at_least_two_positive_tasks"]),
        "at_least_two_of_three_seeds_positive": bool(
            source_checks["at_least_two_of_three_seeds_positive"]
        ),
        "no_stable_task_harm_at_3pct": bool(
            source_checks["no_stable_task_harm_at_3pct"]
        ),
        "no_seed_stably_harmed_at_3pct": bool(
            source_checks["no_seed_stably_harmed_at_3pct"]
        ),
        "retains_at_least_half_oracle_direct_gain": bool(
            source_checks["retains_at_least_half_oracle_direct_gain"]
        ),
        "all_active_heads_beat_constant_brier": bool(
            source_checks["all_active_heads_beat_constant_brier"]
        ),
        "reliability_bins_directional": bool(
            source_checks["reliability_bins_directional"]
        ),
        "gate_off_exactly_returns_hc": bool(
            source_checks["gate_off_exactly_returns_hc"]
        ),
        "source_action_branches_converged": bool(
            source_checks["all_action_branches_converged"]
        ),
        "supplement_action_branches_converged": supplement_converged,
        "sealed_test_untouched": (
            int(source_receipt["test_paths_opened"]) == 0
            and supplement_test_paths_opened == 0
        ),
    }


def attribution_code(summary: Mapping[str, Any]) -> str:
    if float(summary["ci95"][0]) > 0.0:
        return "SUPPORTED_ARB_HAT_INCREMENT_BEYOND_CONTROL"
    if float(summary["macro_gain"]) > 0.0:
        return "DIRECTIONAL_ARB_HAT_INCREMENT_NOT_CI_CONFIRMED"
    return "ARB_HAT_SEMANTIC_INCREMENT_NOT_ISOLATED_FROM_CONTROL"


def aggregate(args: argparse.Namespace, gate: Mapping[str, Any]) -> None:
    output = args.output_root / "formal" / "r4_b_supplement_receipt.json"
    if output.exists():
        raise FileExistsError(f"fresh supplement aggregate required: {output}")
    original_gate, source_receipt = load_source(gate)
    source_root = Path(str(gate["source"]["r4_b_root"]))
    source_metrics, _ = r4b.load_branch_results(source_root, original_gate)
    supplement_metrics, supplement_receipts = load_supplement_results(
        args.output_root, gate
    )
    candidate = r4.median_metrics(source_metrics["arb_hat_direct"])
    medians = {
        condition: r4.median_metrics(values)
        for condition, values in supplement_metrics.items()
    }
    statistics_seed = int(gate["seeds"]["statistics"])
    comparisons = {
        condition: m3.summarize_gain(
            medians[condition], candidate, statistics_seed + offset
        )
        for offset, condition in enumerate(CONDITIONS, start=1)
    }
    _, _, hc_metrics = r4b.load_source_hc(original_gate)
    controls_vs_hc = {
        condition: m3.summarize_gain(
            hc_metrics, medians[condition], statistics_seed + 20 + offset
        )
        for offset, condition in enumerate(CONDITIONS, start=1)
    }
    converged = all(bool(receipt["strict_gate_eligible"]) for receipt in supplement_receipts)
    supplement_test_paths_opened = sum(
        int(receipt.get("test_paths_opened", -1))
        for receipt in supplement_receipts
    )
    checks = decision_checks(
        source_receipt, converged, supplement_test_paths_opened
    )
    passed = all(checks.values())
    decision = (
        "PASSED_M3_R4_B_SIGNAL_FIRST_OWNER_AMENDMENT"
        if passed
        else "FAILED_M3_R4_B_SIGNAL_FIRST_OWNER_AMENDMENT"
    )
    receipt = {
        "format_version": "ssc-v7.m3_r4_b.supplement_receipt/1",
        "stage_id": STAGE_ID,
        "completed_at_utc": utc_now(),
        "gate_sha256": gate["_runtime_gate_sha256"],
        "source_r4_b_receipt_sha256": gate["source"]["r4_b_receipt_sha256"],
        "owner_amendment_decision_code": decision,
        "r4_c_authorized": passed,
        "arb_hat_vs_supplement_controls": comparisons,
        "supplement_controls_vs_hc": controls_vs_hc,
        "attribution": {
            condition: attribution_code(summary)
            for condition, summary in comparisons.items()
        },
        "decision_checks": checks,
        "diagnostic_controls_are_non_blocking": True,
        "interpretation_policy": gate["interpretation_policy"],
        "test_paths_opened": 0,
        "sealed_test_generated": False,
        "r4_c_started": False,
        "m4_authorized": False,
        "b_core_authorized": False,
    }
    write_json(output, receipt)
    print(decision)


def main() -> None:
    args = parse_args()
    gate = load_gate(args.gate)
    preflight(gate)
    if args.command == "train-branch":
        train_branch(args, gate)
    elif args.command == "aggregate":
        aggregate(args, gate)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
