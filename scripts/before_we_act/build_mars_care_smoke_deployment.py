#!/usr/bin/env python3
"""Build a runtime-only deployment artifact from the resumed CARE smoke scorer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from before_we_act.care_belief import CARECalibration
from before_we_act.care_training_data import CARETrainingDataset, load_prepared_care, sha256_file
from scripts.before_we_act.select_calibrate_care import conformal_correction, prediction_rows
from scripts.before_we_act.select_calibrate_mars_care import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--training-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        return
    device = torch.device(args.device)
    prepared = load_prepared_care(args.prepared_data)
    model, saved = load_model(args.training_checkpoint, device)
    if int(saved["update"]) < 4 or saved.get("variant") != "care":
        raise RuntimeError("CARE smoke scorer must be resumed through update 4")
    data = CARETrainingDataset(
        prepared, "all", primary_horizon_only=True, primary_horizon=16
    )
    rows = prediction_rows(
        model, DataLoader(data, batch_size=48, shuffle=False), device
    )
    calibration = CARECalibration(
        lower_correction=conformal_correction(rows, 0.9),
        selector_delta=0.0,
        hard_safety_probability_max=0.25,
        nominal_simultaneous_coverage=0.9,
        primary_horizon=16,
    )
    payload = {
        "format_version": "before-we-act.care-mars-deployment-checkpoint/1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": saved["config"],
        "model": saved["model"],
        "calibration": calibration.__dict__,
        "selected_seed": int(saved["seed"]),
        "selected_update": int(saved["update"]),
        "reference_checkpoint": str(args.reference_checkpoint.resolve()),
        "reference_checkpoint_sha256": sha256_file(args.reference_checkpoint),
        "prepared_data_sha256": sha256_file(args.prepared_data),
        "strict_local": True,
        "benchmark_adapter": "MARS-Control",
        "stage": "smoke",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
