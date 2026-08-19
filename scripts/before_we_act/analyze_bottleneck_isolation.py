#!/usr/bin/env python3
"""Judge the staged bounded-BT then bottleneck pilot without closed-loop data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from before_we_act.temporal_history_data import sha256_file


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(root: Path, variant: str) -> dict:
    path = root / "training" / variant / "seed_20260815" / "status.json"
    status = read(path)
    validation = status["selected_validation"]
    pairing = validation["action_pairing"]
    relative = validation["base_relative"]
    return {
        "status_path": str(path.resolve()),
        "status_sha256": sha256_file(path),
        "status": status["status"],
        "action_mse": float(validation["macro"]["b_core"]),
        "shuffle_to_matched_ratio": float(validation["macro"]["b_shuffle"])
        / max(float(validation["macro"]["b_core"]), 1e-12),
        "output_to_target_sensitivity": float(
            pairing["output_to_target_sensitivity"]
        ),
        "output_to_residual_energy": float(
            pairing["output_to_residual_energy"]
        ),
        "conditional_kl_nats": float(relative["conditional_kl_nats"]),
        "nuisance_residual_relative_mse": float(
            relative["nuisance_proxy"]["residual_relative_mse"]
        ),
        "guard": status["sensitivity_guard"],
    }


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    control = metrics(args.run_root, "a4_no_bottleneck")
    full = metrics(args.run_root, "a4_full")
    bt_checks = {
        "bounded_bt_no_bottleneck_pilot_passed": control["status"]
        == "PASSED_LOSS_SCALE_PILOT",
        "bounded_bt_no_bottleneck_sensitivity_guard_passed": bool(
            control["guard"]["passed"]
        ),
    }
    bottleneck_checks = {
        "full_pilot_passed": full["status"] == "PASSED_LOSS_SCALE_PILOT",
        "full_sensitivity_guard_passed": bool(full["guard"]["passed"]),
        "action_mse_not_over_two_percent_worse": full["action_mse"]
        <= 1.02 * control["action_mse"],
        "target_sensitivity_not_amplified_over_twenty_percent": full[
            "output_to_target_sensitivity"
        ]
        <= 1.20 * control["output_to_target_sensitivity"],
        "residual_energy_ratio_not_amplified_over_twenty_percent": full[
            "output_to_residual_energy"
        ]
        <= 1.20 * control["output_to_residual_energy"],
        "nuisance_sensitivity_not_amplified_over_twenty_five_percent": full[
            "nuisance_residual_relative_mse"
        ]
        <= 1.25 * control["nuisance_residual_relative_mse"],
    }
    compression_checks = {
        "conditional_kl_at_least_five_percent_lower": full[
            "conditional_kl_nats"
        ]
        <= 0.95 * control["conditional_kl_nats"],
    }
    bt_passed = all(bt_checks.values())
    bottleneck_safe = bt_passed and all(bottleneck_checks.values())
    if not bt_passed:
        status = "FAILED_BOUNDED_BT_ISOLATION"
    elif not bottleneck_safe:
        status = "FAILED_BOTTLENECK_SAFETY_ISOLATION"
    elif all(compression_checks.values()):
        status = "PASSED_5K_BOTTLENECK_ISOLATION"
    else:
        status = "PASSED_BT_BOTTLENECK_BENEFIT_NOT_ESTABLISHED"
    atomic_json(
        args.output,
        {
            "format_version": "before-we-act.a4-repair-isolation/1",
            "status": status,
            "contract_sha256": sha256_file(args.contract),
            "bounded_bt_checks": bt_checks,
            "bottleneck_safety_checks": bottleneck_checks,
            "bottleneck_compression_checks": compression_checks,
            "a4_no_bottleneck": control,
            "a4_full": full,
            "claim_boundary": (
                "5k single-seed safety screen only; no closed-loop efficacy claim"
            ),
        },
    )


if __name__ == "__main__":
    main()
