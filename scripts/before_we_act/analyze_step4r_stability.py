#!/usr/bin/env python3
"""Judge the frozen bounded-BT beta=0 Step-4R 25k stability pilot."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from before_we_act.temporal_history_data import sha256_file


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def candidate_metrics(row: dict) -> dict[str, float]:
    validation = row["validation"]
    pairing = validation["action_pairing"]
    relative = validation["base_relative"]
    return {
        "action_mse": float(validation["macro"]["b_core"]),
        "output_to_residual_energy": float(
            pairing["output_to_residual_energy"]
        ),
        "output_to_target_sensitivity": float(
            pairing["output_to_target_sensitivity"]
        ),
        "nuisance_residual_relative_mse": float(
            relative["nuisance_proxy"]["residual_relative_mse"]
        ),
        "belief_off_max_abs": float(validation["belief_off_max_abs"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stability-contract", type=Path, required=True)
    parser.add_argument("--a4-contract", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = read_json(args.stability_contract)
    if contract.get("status") != "FROZEN_BEFORE_25K":
        raise RuntimeError("Step-4R stability contract is not frozen")
    if sha256_file(args.a4_contract) != contract["a4_contract"]["sha256"]:
        raise RuntimeError("Step-4R A4 contract hash differs")
    status_path = args.candidate_root / "status.json"
    evaluations_path = args.candidate_root / "evaluations.jsonl"
    status = read_json(status_path)
    expected = contract["candidate"]
    if int(status.get("seed", -1)) != int(expected["seed"]):
        raise RuntimeError("Step-4R candidate seed differs")
    if status.get("variant") != expected["variant"]:
        raise RuntimeError("Step-4R candidate variant differs")
    rows = {int(row["update"]): row for row in read_jsonl(evaluations_path)}
    required = [int(value) for value in expected["evaluation_updates"]]
    gates = contract["hard_gates"]
    baseline = contract["bcore_same_budget_baseline"]["mean_by_update"]
    nuisance_anchor = float(
        contract["bounded_bt_beta0_5k_anchor"][
            "nuisance_residual_relative_mse"
        ]
    )

    points = {}
    point_checks = {}
    for update in required:
        if update not in rows:
            continue
        current = candidate_metrics(rows[update])
        reference = baseline[str(update)]
        checks = {
            "all_metrics_finite": all(math.isfinite(value) for value in current.values()),
            "action_mse_within_two_percent_of_bcore_mean": current["action_mse"]
            <= float(gates["action_mse_vs_bcore_mean_max_ratio"])
            * float(reference["action_mse"]),
            "residual_energy_within_twenty_five_percent_of_bcore_mean": current[
                "output_to_residual_energy"
            ]
            <= float(
                gates["output_to_residual_energy_vs_bcore_mean_max_ratio"]
            )
            * float(reference["output_to_residual_energy"]),
            "target_sensitivity_within_twenty_five_percent_of_bcore_mean": current[
                "output_to_target_sensitivity"
            ]
            <= float(
                gates["output_to_target_sensitivity_vs_bcore_mean_max_ratio"]
            )
            * float(reference["output_to_target_sensitivity"]),
            "nuisance_within_twenty_five_percent_of_frozen_5k_anchor": current[
                "nuisance_residual_relative_mse"
            ]
            <= float(gates["nuisance_vs_bounded_bt_5k_max_ratio"])
            * nuisance_anchor,
            "belief_off_exact_base": current["belief_off_max_abs"]
            <= float(gates["belief_off_max_abs"]),
        }
        points[str(update)] = {
            "candidate": current,
            "bcore_three_seed_mean": reference,
            "ratios": {
                "action_mse": current["action_mse"]
                / max(float(reference["action_mse"]), 1e-12),
                "output_to_residual_energy": current["output_to_residual_energy"]
                / max(float(reference["output_to_residual_energy"]), 1e-12),
                "output_to_target_sensitivity": current[
                    "output_to_target_sensitivity"
                ]
                / max(float(reference["output_to_target_sensitivity"]), 1e-12),
                "nuisance_vs_bounded_bt_5k": current[
                    "nuisance_residual_relative_mse"
                ]
                / max(nuisance_anchor, 1e-12),
            },
        }
        point_checks[str(update)] = checks

    global_checks = {
        "trainer_completed_25k": int(status.get("update", -1)) == 25_000,
        "trainer_internal_sensitivity_guard_passed": bool(
            status.get("sensitivity_guard", {}).get("passed", False)
        ),
        "all_five_frozen_evaluations_present": set(required).issubset(rows),
        "every_frozen_point_passed": len(point_checks) == len(required)
        and all(all(checks.values()) for checks in point_checks.values()),
    }
    passed = all(global_checks.values())
    atomic_json(
        args.output,
        {
            "format_version": "before-we-act.step4r-stability-report/1",
            "status": (
                "PASSED_STEP4R_25K_STABILITY"
                if passed
                else "FAILED_STEP4R_25K_STABILITY"
            ),
            "stability_contract_sha256": sha256_file(args.stability_contract),
            "a4_contract_sha256": sha256_file(args.a4_contract),
            "candidate_status": {
                "path": str(status_path.resolve()),
                "sha256": sha256_file(status_path),
                "trainer_status": status.get("status"),
            },
            "evaluations": {
                "path": str(evaluations_path.resolve()),
                "sha256": sha256_file(evaluations_path),
            },
            "global_checks": global_checks,
            "point_checks": point_checks,
            "points": points,
            "claim_boundary": contract["claim_boundary"],
        },
    )


if __name__ == "__main__":
    main()
