#!/usr/bin/env python3
"""Freeze the evidence and gates for the bounded-BT beta=0 Step-4R pilot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean

from before_we_act.temporal_history_data import sha256_file


UPDATES = (5_000, 10_000, 15_000, 20_000, 25_000)
BCORE_SEEDS = (20260815, 20260816, 20260817)


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


def metrics(row: dict) -> dict[str, float]:
    validation = row["validation"]
    pairing = validation["action_pairing"]
    return {
        "action_mse": float(validation["macro"]["b_core"]),
        "output_to_residual_energy": float(
            pairing["output_to_residual_energy"]
        ),
        "output_to_target_sensitivity": float(
            pairing["output_to_target_sensitivity"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a4-contract", type=Path, required=True)
    parser.add_argument("--bcore-training-root", type=Path, required=True)
    parser.add_argument("--bounded-bt-5k-status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    a4 = read_json(args.a4_contract)
    pilot = a4["training"]["loss_scale_pilot"]
    if a4.get("status") != "FROZEN_PILOT":
        raise RuntimeError("Step-4R requires a frozen pilot contract")
    if a4["variants"]["a4_no_bottleneck"]["beta_b"] != 0.0:
        raise RuntimeError("Step-4R control arm must freeze beta_b=0")
    if int(pilot["updates"]) != 25_000 or int(
        pilot["evaluation_interval"]
    ) != 5_000:
        raise RuntimeError("Step-4R is frozen at 25k with 5k evaluation points")

    baseline_files = []
    by_update: dict[int, list[dict[str, float]]] = {
        update: [] for update in UPDATES
    }
    for seed in BCORE_SEEDS:
        path = args.bcore_training_root / f"seed_{seed}" / "evaluations.jsonl"
        rows = {int(row["update"]): row for row in read_jsonl(path)}
        missing = [update for update in UPDATES if update not in rows]
        if missing:
            raise RuntimeError(f"B-core seed {seed} misses updates {missing}")
        baseline_files.append(
            {"seed": seed, "path": str(path.resolve()), "sha256": sha256_file(path)}
        )
        for update in UPDATES:
            by_update[update].append(metrics(rows[update]))

    baseline_means = {
        str(update): {
            key: mean(row[key] for row in by_update[update])
            for key in by_update[update][0]
        }
        for update in UPDATES
    }
    bounded_status = read_json(args.bounded_bt_5k_status)
    if (
        bounded_status.get("status") != "PASSED_LOSS_SCALE_PILOT"
        or bounded_status.get("variant") != "a4_no_bottleneck"
        or int(bounded_status.get("update", -1)) != 5_000
    ):
        raise RuntimeError("the frozen bounded-BT beta=0 5k anchor is invalid")
    bounded_validation = bounded_status["selected_validation"]
    bounded_nuisance = float(
        bounded_validation["base_relative"]["nuisance_proxy"][
            "residual_relative_mse"
        ]
    )

    atomic_json(
        args.output,
        {
            "format_version": "before-we-act.step4r-stability-contract/1",
            "status": "FROZEN_BEFORE_25K",
            "candidate": {
                "variant": "a4_no_bottleneck",
                "seed": int(pilot["seed"]),
                "updates": 25_000,
                "evaluation_updates": list(UPDATES),
                "closed_loop_forbidden": True,
            },
            "a4_contract": {
                "path": str(args.a4_contract.resolve()),
                "sha256": sha256_file(args.a4_contract),
            },
            "bcore_same_budget_baseline": {
                "seeds": list(BCORE_SEEDS),
                "files": baseline_files,
                "mean_by_update": baseline_means,
            },
            "bounded_bt_beta0_5k_anchor": {
                "path": str(args.bounded_bt_5k_status.resolve()),
                "sha256": sha256_file(args.bounded_bt_5k_status),
                "nuisance_residual_relative_mse": bounded_nuisance,
            },
            "hard_gates": {
                "action_mse_vs_bcore_mean_max_ratio": 1.02,
                "output_to_residual_energy_vs_bcore_mean_max_ratio": 1.25,
                "output_to_target_sensitivity_vs_bcore_mean_max_ratio": 1.25,
                "nuisance_vs_bounded_bt_5k_max_ratio": 1.25,
                "belief_off_max_abs": 0.0,
                "all_five_evaluations_required": True,
            },
            "claim_boundary": (
                "single fresh-seed 25k offline stability test only; no closed-loop "
                "efficacy or final method claim"
            ),
        },
    )


if __name__ == "__main__":
    main()
