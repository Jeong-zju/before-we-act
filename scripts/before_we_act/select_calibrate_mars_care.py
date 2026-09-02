#!/usr/bin/env python3
"""Select and calibrate official CARE scorer checkpoints for MARS."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from before_we_act.care_belief import CAREBeliefConfig, CAREBeliefHead, CARECalibration
from before_we_act.care_training_data import CARETrainingDataset, atomic_json, load_prepared_care, sha256_file
from scripts.before_we_act.select_calibrate_care import conformal_correction, offline_metrics, prediction_rows


VARIANTS = ("care", "reactive_only", "replay_only", "capacity")
SEEDS = (20260818, 20260819, 20260820)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_model(path: Path, device: torch.device) -> tuple[CAREBeliefHead, dict[str, Any]]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format_version") != "before-we-act.care-mars-training-checkpoint/1":
        raise ValueError(f"wrong MARS CARE training checkpoint: {path}")
    model = CAREBeliefHead(CAREBeliefConfig.from_mapping(saved["config"])).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    return model, saved


def select_training(training_root: Path) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        rows = []
        for seed in SEEDS:
            path = training_root / variant / f"seed_{seed}" / "status.json"
            row = json.loads(path.read_text())
            if row.get("status") != "COMPLETED" or row.get("variant") != variant or int(row.get("seed", -1)) != seed:
                raise RuntimeError(f"MARS CARE training incomplete: {path}")
            rows.append(row)
        selected[variant] = min(
            rows,
            key=lambda row: (
                float(row["selected_validation"]["regret"]),
                float(row["selected_validation"]["loss"]),
                int(row["seed"]),
            ),
        )
    return selected


def atomic_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report_path = args.output_root / "offline_report.json"
    deployment_path = args.output_root / "care_deployment_checkpoint.pt"
    if report_path.exists() and deployment_path.exists():
        print(json.dumps({"status": "PRESERVED", "report": str(report_path)}))
        return
    settings = json.loads(args.settings.read_text())
    recipe = settings["care_recipe"]
    if tuple(recipe["care_variants"]) != VARIANTS or tuple(recipe["care_seeds"]) != SEEDS:
        raise ValueError("MARS CARE variant/seed contract drift")
    prepared = load_prepared_care(args.prepared_data)
    selected = select_training(args.training_root)
    device = torch.device(args.device)
    diagnostic = CARETrainingDataset(prepared, "all", primary_horizon_only=True, primary_horizon=16)
    loader = DataLoader(diagnostic, batch_size=48, shuffle=False, num_workers=0)
    selected_models: dict[str, Any] = {}
    uncalibrated: dict[str, Any] = {}
    care_model = care_saved = None
    care_rows = None
    for variant in VARIANTS:
        checkpoint = Path(selected[variant]["selected_checkpoint"])
        model, saved = load_model(checkpoint, device)
        rows = prediction_rows(model, loader, device)
        uncalibrated[variant] = offline_metrics(rows, variant)
        selected_models[variant] = {
            "seed": int(selected[variant]["seed"]),
            "update": int(selected[variant]["selected_update"]),
            "diagnostic": selected[variant]["selected_validation"],
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "parameter_count": int(selected[variant]["parameter_count"]),
        }
        if variant == "care":
            care_model, care_saved, care_rows = model, saved, rows
    assert care_model is not None and care_saved is not None and care_rows is not None
    nominal = 0.9
    correction = conformal_correction(care_rows, nominal)
    calibration = CARECalibration(
        lower_correction=correction,
        selector_delta=0.0,
        hard_safety_probability_max=0.25,
        nominal_simultaneous_coverage=nominal,
        primary_horizon=16,
    )
    calibrated = offline_metrics(care_rows, "care", calibration)
    deployment = {
        "format_version": "before-we-act.care-mars-deployment-checkpoint/1",
        "created_at_utc": utc_now(),
        "config": care_saved["config"],
        "model": care_saved["model"],
        "calibration": calibration.__dict__,
        "selected_seed": int(care_saved["seed"]),
        "selected_update": int(care_saved["update"]),
        "reference_checkpoint": str(args.reference_checkpoint.resolve()),
        "reference_checkpoint_sha256": sha256_file(args.reference_checkpoint),
        "settings": str(args.settings.resolve()),
        "settings_sha256": sha256_file(args.settings),
        "prepared_data_sha256": sha256_file(args.prepared_data),
        "strict_local": True,
        "benchmark_adapter": "MARS-Control",
    }
    atomic_save(deployment, deployment_path)
    report = {
        "format_version": "before-we-act.care-mars-offline-report/1",
        "completed_at_utc": utc_now(),
        "selected_models": selected_models,
        "diagnostic_uncalibrated": uncalibrated,
        "calibration": calibration.__dict__,
        "diagnostic_calibrated_care": calibrated,
        "all_family_training": True,
        "same_family_diagnostics_and_calibration": True,
        "held_out_measurement": "four-task Validation20 only",
        "deployment_checkpoint": str(deployment_path.resolve()),
        "deployment_checkpoint_sha256": sha256_file(deployment_path),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(report_path, report)
    print(json.dumps({"status": "CARE_MARS_OFFLINE_READY", "report": str(report_path)}))


if __name__ == "__main__":
    main()
