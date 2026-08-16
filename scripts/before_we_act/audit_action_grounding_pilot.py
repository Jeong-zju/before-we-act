#!/usr/bin/env python3
"""Create a read-only diagnostic receipt for the frozen R1-3 rollouts."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from before_we_act.temporal_history_data import SIX_TASKS, sha256_file


MODES = ("normal", "delay_freeze", "timing_early_or_late", "wrong_role")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--conclusion", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    conclusion = json.loads(args.conclusion.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in args.rollouts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if sha256_file(args.contract) != conclusion["contract_sha256"]:
        raise RuntimeError("pilot contract hash differs")
    if sha256_file(args.rollouts) != conclusion["rollouts_sha256"]:
        raise RuntimeError("pilot rollout hash differs")
    if len(rows) != int(contract["design"]["rollouts"]):
        raise RuntimeError("pilot rollout count differs")

    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["task"], int(row["state_index"]), row["mode"])].append(row)
    nonexact: list[dict] = []
    for key, group in sorted(grouped.items()):
        signatures = {
            (
                round(float(row["cumulative_dense_reward"]), 8),
                bool(row["success"]),
                bool(row["terminal"]),
                round(
                    float(row["shared_state_change"]["max_object_displacement"]),
                    8,
                ),
            )
            for row in group
        }
        displacement = [
            float(row["shared_state_change"]["max_object_displacement"])
            for row in group
        ]
        if len(signatures) != 1:
            nonexact.append(
                {
                    "task": key[0],
                    "state_index": key[1],
                    "mode": key[2],
                    "max_object_displacement_range": max(displacement)
                    - min(displacement),
                }
            )
    secondary: dict[str, dict] = {}
    for task in SIX_TASKS:
        secondary[task] = {}
        for mode in MODES:
            selected = [
                row for row in rows if row["task"] == task and row["mode"] == mode
            ]
            secondary[task][mode] = {
                "max_object_displacement_mean": float(
                    np.mean(
                        [
                            row["shared_state_change"]["max_object_displacement"]
                            for row in selected
                        ]
                    )
                ),
                "contact_changed_count": sum(
                    bool(row["shared_state_change"]["contact_changed"])
                    for row in selected
                ),
                "grasp_changed_count": sum(
                    bool(row["shared_state_change"]["grasp_changed"])
                    for row in selected
                ),
            }
    rewards = [float(row["cumulative_dense_reward"]) for row in rows]
    result = {
        "format_version": "before-we-act.b3-n1-r1-pilot-diagnostic/1",
        "stage": "R1-3-COUNTERFACTUAL-PILOT-DIAGNOSTIC-ONLY",
        "completed_at_utc": utc_now(),
        "contract_sha256": sha256_file(args.contract),
        "conclusion_sha256": sha256_file(args.conclusion),
        "rollouts_sha256": sha256_file(args.rollouts),
        "rollouts": len(rows),
        "groups": len(grouped),
        "rewards": {
            "minimum": min(rewards),
            "maximum": max(rewards),
            "all_zero": all(value == 0.0 for value in rewards),
        },
        "success_count": sum(bool(row["success"]) for row in rows),
        "terminal_count": sum(bool(row["terminal"]) for row in rows),
        "exact_repeat_groups": len(grouped) - len(nonexact),
        "nonexact_repeat_groups": len(nonexact),
        "maximum_nonexact_displacement_range": max(
            (
                row["max_object_displacement_range"]
                for row in nonexact
            ),
            default=0.0,
        ),
        "nonexact_groups": nonexact,
        "secondary_outcomes_not_used_to_change_the_frozen_gate": secondary,
        "interpretation": (
            "The frozen primary reward is degenerate and restore reproducibility failed. "
            "Secondary object/contact changes remain diagnostic only and cannot replace "
            "the preregistered primary outcome after results are visible."
        ),
    }
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "all_rewards_zero": result["rewards"]["all_zero"],
                "nonexact_repeat_groups": len(nonexact),
                "groups": len(grouped),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
