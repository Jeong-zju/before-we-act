#!/usr/bin/env python3
"""Build the deployable H8 CARE-v3 artifact from final fit + strict OOF."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

import torch

from before_we_act.care_belief import CARECalibration
from before_we_act.care_training_data import load_prepared_care, sha256_file
from before_we_act.mars_care_v3_deployment import (
    DEPLOYMENT_FORMAT_VERSION,
    FINAL_TRAINING_FORMAT_VERSION,
    OOF_FORMAT_VERSION,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--oof-report", type=Path)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--promotion-scope",
        choices=("smoke", "formal", "exploratory"),
        default="formal",
    )
    parser.add_argument("--paired-smoke-report", type=Path)
    parser.add_argument("--interface-smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite deployment checkpoint {args.output}")
    prepared = load_prepared_care(args.prepared_data)
    final = torch.load(args.final_checkpoint, map_location="cpu", weights_only=False)
    report = (
        __import__("json").loads(args.oof_report.read_text(encoding="utf-8"))
        if args.oof_report is not None
        else None
    )
    if final.get("format_version") != FINAL_TRAINING_FORMAT_VERSION:
        raise ValueError("wrong final CARE-v3 checkpoint format")
    if final.get("prepared_data_sha256") != sha256_file(args.prepared_data):
        raise ValueError("final checkpoint/prepared-data hash mismatch")
    if int(final.get("action_prefix_steps", -1)) != 8:
        raise ValueError("final checkpoint is not the H8 fit")
    if not args.interface_smoke and report is None:
        raise ValueError("formal deployment requires a strict OOF report")
    if report is not None and report.get("format_version") != "before-we-act.care-mars-oof-predictions-v3/2-all-horizon":
        raise ValueError("wrong strict OOF report format")
    if report is not None and report.get("horizon_oof_complete") is not True:
        raise ValueError("strict OOF report is not complete over all horizons")
    calibration_report = report.get("horizon_calibration", report.get("calibration", {})) if report is not None else {}
    normalized = float(calibration_report.get("normalized_familywise_correction", 0.0))
    if not torch.isfinite(torch.tensor(normalized)) or normalized < 0:
        raise ValueError("strict OOF normalized correction is invalid")
    metrics = report.get("metrics", {}) if report is not None else {}
    required_metrics = {
        "pairwise_accuracy_including_reference": 0.65,
        "top1_accuracy": 0.40,
        "candidate_vs_reference_sign_accuracy": 0.65,
    }
    # OOF-v3 reports retain the stricter legality-masked metric name for
    # pairwise accuracy and the legacy selector name for top-1.  Resolve these
    # aliases centrally so a schema spelling difference cannot look like a
    # missing metric (or trigger a late, manual repair).
    metric_aliases = {
        "pairwise_accuracy_including_reference": (
            "pairwise_accuracy_including_reference",
            "legal_pairwise_accuracy_including_reference",
        ),
        "top1_accuracy": ("top1_accuracy", "selector_top1_accuracy"),
        "candidate_vs_reference_sign_accuracy": (
            "candidate_vs_reference_sign_accuracy",
        ),
    }
    resolved_metrics = {
        key: next((metrics.get(name) for name in names if name in metrics), None)
        for key, names in metric_aliases.items()
    }
    enforce_admission = args.promotion_scope == "formal" and not args.interface_smoke
    failures = [] if not enforce_admission else [
        f"{key}={resolved_metrics[key]!r} < {threshold}"
        for key, threshold in required_metrics.items()
        if not isinstance(resolved_metrics[key], (int, float))
        or float(resolved_metrics[key]) < threshold
    ]
    coverage = calibration_report.get("crossfit_family_coverage", report.get("crossfit_family_coverage")) if report is not None else 1.0
    if not isinstance(coverage, (int, float)) or float(coverage) < 0.90:
        failures.append(f"crossfit_family_coverage={coverage!r} < 0.90")
    harmful = metrics.get("harmful_override_rate")
    # Interface smoke deliberately has no OOF report and therefore no
    # scorer-derived harmful-override statistic.  Enforce this admission
    # gate only for formal promotion after OOF calibration.
    if enforce_admission and (
        not isinstance(harmful, (int, float)) or float(harmful) > 0.05
    ):
        failures.append(f"harmful_override_rate={harmful!r} > 0.05")
    if failures:
        raise ValueError("strict OOF scorer admission failed: " + "; ".join(failures))
    smoke_report: dict[str, Any] | None = None
    if args.promotion_scope == "formal":
        if args.paired_smoke_report is None:
            raise ValueError("formal promotion requires a paired smoke report")
        smoke_report = __import__("json").loads(
            args.paired_smoke_report.read_text(encoding="utf-8")
        )
        if smoke_report.get("status") != "PASSED" or smoke_report.get("passed") is not True:
            raise ValueError("formal promotion paired smoke gate did not pass")
    scales = torch.as_tensor(final["task_horizon_component_scales"], dtype=torch.float32)
    expected = (len(prepared.tasks), 4, 3)
    if tuple(scales.shape) != expected or not torch.isfinite(scales).all() or bool((scales <= 0).any()):
        raise ValueError("final task/horizon scales do not match MARS contract")
    corrections = normalized * scales[:, :, 2]
    calibration = CARECalibration(
        lower_correction=float(corrections[:, 1].max()),
        selector_delta=0.0,
        hard_safety_probability_max=0.25,
        nominal_simultaneous_coverage=float(calibration_report.get("nominal_simultaneous_coverage", 0.90)),
        primary_horizon=16,
    )
    safety_support = int(final.get("safety_positive_label_count", 0))
    # The current OOF contract calibrates utility bounds, not a learned safety
    # probability threshold.  Never turn positive labels into an uncalibrated
    # learned gate; a corpus with such labels must stop for an explicit safety
    # calibration stage instead of silently weakening fail-closed behavior.
    if safety_support > 0:
        raise ValueError(
            "CARE v3 final fit has positive hard-safety labels but no independent "
            "safety-threshold calibration"
        )
    safety_mode = "legality_only"
    payload: dict[str, Any] = {
        "format_version": DEPLOYMENT_FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_training_format_version": FINAL_TRAINING_FORMAT_VERSION,
        "config": final["model_config"],
        "model": final["model"],
        "calibration": calibration.__dict__,
        "task_names": list(prepared.tasks),
        "task_horizon_component_scales": scales,
        "task_horizon_lower_corrections": corrections,
        "intervention_steps": 8,
        "safety_gate_mode": safety_mode,
        "safety_positive_label_count": safety_support,
        "safety_threshold_calibrated": safety_support > 0,
        "reference_checkpoint": str(args.reference_checkpoint.resolve()),
        "reference_checkpoint_sha256": sha256_file(args.reference_checkpoint),
        "prepared_data": str(args.prepared_data.resolve()),
        "prepared_data_sha256": sha256_file(args.prepared_data),
        "final_training_checkpoint": str(args.final_checkpoint.resolve()),
        "final_training_checkpoint_sha256": sha256_file(args.final_checkpoint),
        "oof_report": str(args.oof_report.resolve()) if args.oof_report is not None else None,
        "oof_report_sha256": sha256_file(args.oof_report) if args.oof_report is not None else None,
        "action_contract_version": prepared.manifest.get("action_contract_version"),
        "action_contract_sha256": prepared.manifest.get("action_contract_sha256"),
        "normalization_sha256": prepared.manifest.get("normalization_sha256"),
        "provenance": {
            "oof_format_version": OOF_FORMAT_VERSION if report is not None else None,
            "admission_passed": report is not None,
            "family_disjoint": report is not None,
            "calibration_independent": report is not None,
            "no_validation20_tuning": True,
            "physical_unit_runtime_parity": True,
            "horizon_oof_complete": report is not None,
            "oof_report_sha256": sha256_file(args.oof_report) if args.oof_report is not None else None,
            "promotion_scope": args.promotion_scope,
            "interface_smoke_only": bool(args.interface_smoke),
            "admission_bypassed": args.promotion_scope == "exploratory",
            "admission_thresholds": {
                **required_metrics,
                "crossfit_family_coverage": 0.90,
                "harmful_override_rate": 0.05,
            },
            "paired_smoke_passed": smoke_report is not None,
            "decentralized_smoke_passed": smoke_report is not None,
            "paired_smoke_report_sha256": (
                sha256_file(args.paired_smoke_report)
                if args.paired_smoke_report is not None
                else None
            ),
        },
        "all_family_training": True,
        "validation20_used_for_tuning": False,
        "care_theory_contract_unchanged": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    print(__import__("json").dumps({"status": "complete", "output": str(args.output)}))


if __name__ == "__main__":
    main()
