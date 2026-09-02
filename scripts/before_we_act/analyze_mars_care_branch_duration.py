#!/usr/bin/env python3
"""Audit fixed 1/4/8/16-step MARS CARE branch-duration smoke corpora."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


FORMAT_VERSION = "before-we-act.care-mars-branch-duration-audit/1"
DURATIONS = (1, 4, 8, 16)
HORIZONS = (8, 16, 32, 64)
TASKS = (
    "place_cube_in_cup",
    "strike_cube_hard",
    "three_robots_place_shoes",
    "four_robots_stack_cube",
)
SIGNAL_EPSILON = 0.01


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _branch_map(family: Mapping[str, Any]) -> dict[tuple[int, str, int], Mapping[str, Any]]:
    result = {}
    for branch in family["branches"]:
        identity = (
            int(branch["repeat_id"]),
            str(branch["regime"]),
            int(branch["candidate_id"]),
        )
        if identity in result:
            raise ValueError(f"duplicate branch identity: {identity}")
        result[identity] = branch
    return result


def audit_family(path: Path, duration: int) -> dict[str, Any]:
    family = json.loads(path.read_text())
    if int(family.get("intervention_steps", -1)) != int(duration):
        raise ValueError(f"duration metadata differs: {path}")
    branches = _branch_map(family)
    if len(branches) != 24:
        raise ValueError(f"branch support is incomplete: {path}")
    effective = 0
    direct_signal = 0
    response_signal = 0
    total_signal = 0
    total_pairs = 0
    direct_abs: list[float] = []
    response_abs: list[float] = []
    total_abs: list[float] = []
    final_stage_change = 0
    hard_safety = 0
    action_exposure_l2: list[float] = []
    focal_key = f"panda-{int(family['focal_agent'])}"

    for repeat in (0, 1):
        reactive_reference = branches[(repeat, "reactive", 0)]
        replay_reference = branches[(repeat, "replay", 0)]
        for candidate in range(1, 6):
            reactive = branches[(repeat, "reactive", candidate)]
            replay = branches[(repeat, "replay", candidate)]
            reference_actions = reactive_reference["executed_actions"]
            candidate_actions = reactive["executed_actions"]
            for step in range(min(duration, len(reference_actions), len(candidate_actions))):
                left = np.asarray(candidate_actions[step][focal_key], dtype=np.float64)
                right = np.asarray(reference_actions[step][focal_key], dtype=np.float64)
                action_exposure_l2.append(float(np.linalg.norm(left - right)))
            for horizon in HORIZONS:
                key = str(horizon)
                r0 = reactive_reference["outcomes"][key]
                p0 = replay_reference["outcomes"][key]
                r1 = reactive["outcomes"][key]
                p1 = replay["outcomes"][key]
                direct = float(p1["utility_main"]) - float(p0["utility_main"])
                total = float(r1["utility_main"]) - float(r0["utility_main"])
                response = total - direct
                components = (abs(direct), abs(response), abs(total))
                direct_abs.append(components[0])
                response_abs.append(components[1])
                total_abs.append(components[2])
                total_pairs += 1
                effective += int(max(components) >= SIGNAL_EPSILON)
                direct_signal += int(components[0] >= SIGNAL_EPSILON)
                response_signal += int(components[1] >= SIGNAL_EPSILON)
                total_signal += int(components[2] >= SIGNAL_EPSILON)
                final_stage_change += int(
                    str(r1["final_stage_id"]) != str(r0["final_stage_id"])
                    or str(p1["final_stage_id"]) != str(p0["final_stage_id"])
                )
                hard_safety += int(
                    bool(r1["hard_safety_violation"])
                    or bool(p1["hard_safety_violation"])
                )

    def stats(values: list[float]) -> dict[str, float]:
        return {
            "median": float(np.quantile(values, 0.5)),
            "p95": float(np.quantile(values, 0.95)),
            "max": max(values),
        }

    return {
        "path": str(path.resolve()),
        "task": str(family["task"]),
        "snapshot_id": str(family["snapshot_id"]),
        "duration": int(duration),
        "branch_count": len(branches),
        "all_candidates_legal": all(
            bool(row["valid"]) for row in family["candidate_legality"]
        ),
        "all_branch_support_complete": all(
            branch.get("candidate_valid") is True
            and all(str(horizon) in branch.get("outcomes", {}) for horizon in HORIZONS)
            for branch in branches.values()
        ),
        "maximum_restore_error": max(float(row["restore_observation_max_abs_error"]) for row in branches.values()),
        "maximum_restore_rerender_diagnostic_error": max(
            float(
                row.get(
                    "restore_rerender_diagnostic_max_abs_error",
                    row["restore_observation_max_abs_error"],
                )
            )
            for row in branches.values()
        ),
        "maximum_replay_teammate_action_error": max(float(row["replay_teammate_action_max_abs_error"]) for row in branches.values()),
        "maximum_candidate0_reference_action_error": max(
            float(row["candidate0_reference_action_max_abs_error"])
            for row in branches.values()
            if int(row["candidate_id"]) == 0
        ),
        "effective_pair_count": effective,
        "pair_count": total_pairs,
        "signal_density": effective / max(total_pairs, 1),
        "direct_signal_pair_count": direct_signal,
        "direct_signal_density": direct_signal / max(total_pairs, 1),
        "response_signal_pair_count": response_signal,
        "response_signal_density": response_signal / max(total_pairs, 1),
        "total_signal_pair_count": total_signal,
        "total_signal_density": total_signal / max(total_pairs, 1),
        "stage_change_pair_count": final_stage_change,
        "hard_safety_pair_count": hard_safety,
        "direct_absolute": stats(direct_abs),
        "response_absolute": stats(response_abs),
        "total_absolute": stats(total_abs),
        "action_exposure_l2": stats(action_exposure_l2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for duration in DURATIONS:
        family_root = args.root / f"duration_{duration}" / "families"
        for task in TASKS:
            paths = sorted((family_root / task).glob("*.json"))
            if len(paths) != 1:
                raise RuntimeError(f"expected one {task}/H{duration} family, got {paths}")
            families[str(duration)].append(audit_family(paths[0], duration))

    aggregate = {}
    for duration in DURATIONS:
        rows = families[str(duration)]
        pair_count = sum(row["pair_count"] for row in rows)
        aggregate[str(duration)] = {
            "family_count": len(rows),
            "effective_pair_count": sum(row["effective_pair_count"] for row in rows),
            "pair_count": pair_count,
            "signal_density": sum(row["effective_pair_count"] for row in rows)
            / max(pair_count, 1),
            "direct_signal_pair_count": sum(
                row["direct_signal_pair_count"] for row in rows
            ),
            "direct_signal_density": sum(
                row["direct_signal_pair_count"] for row in rows
            )
            / max(pair_count, 1),
            "response_signal_pair_count": sum(
                row["response_signal_pair_count"] for row in rows
            ),
            "response_signal_density": sum(
                row["response_signal_pair_count"] for row in rows
            )
            / max(pair_count, 1),
            "total_signal_pair_count": sum(
                row["total_signal_pair_count"] for row in rows
            ),
            "total_signal_density": sum(
                row["total_signal_pair_count"] for row in rows
            )
            / max(pair_count, 1),
            "stage_change_pair_count": sum(row["stage_change_pair_count"] for row in rows),
            "hard_safety_pair_count": sum(row["hard_safety_pair_count"] for row in rows),
            "support_complete": all(row["all_branch_support_complete"] for row in rows),
            "all_candidates_legal": all(row["all_candidates_legal"] for row in rows),
            "maximum_restore_error": max(row["maximum_restore_error"] for row in rows),
            "maximum_restore_rerender_diagnostic_error": max(
                row["maximum_restore_rerender_diagnostic_error"] for row in rows
            ),
            "maximum_replay_teammate_action_error": max(row["maximum_replay_teammate_action_error"] for row in rows),
            "maximum_candidate0_reference_action_error": max(row["maximum_candidate0_reference_action_error"] for row in rows),
        }
    baseline = aggregate["1"]["signal_density"]
    for duration in DURATIONS:
        row = aggregate[str(duration)]
        row["signal_density_gain_over_h1"] = row["signal_density"] - baseline
        row["signal_density_ratio_over_h1"] = (
            row["signal_density"] / baseline if baseline > 0 else None
        )
        row["eligible_for_scorer_smoke"] = bool(
            duration != 1
            and row["support_complete"]
            and row["all_candidates_legal"]
            and row["maximum_restore_error"] <= 1e-6
            and row["maximum_replay_teammate_action_error"] <= 1e-6
            and row["maximum_candidate0_reference_action_error"] <= 1e-6
            and row["hard_safety_pair_count"] == 0
            and row["signal_density_gain_over_h1"] >= 0.10
            and (row["signal_density_ratio_over_h1"] or 0.0) >= 1.5
        )
    result = {
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "protocol": "same fixed-stratified first family per task; duration-only ablation",
        "signal_epsilon": SIGNAL_EPSILON,
        "validation20_used_for_tuning": False,
        "main_protocol_unchanged": True,
        "families": dict(families),
        "aggregate": aggregate,
        "eligible_durations_for_scorer_smoke": [
            duration
            for duration in DURATIONS
            if aggregate[str(duration)]["eligible_for_scorer_smoke"]
        ],
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
