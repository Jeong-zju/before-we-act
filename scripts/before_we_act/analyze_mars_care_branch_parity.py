#!/usr/bin/env python3
"""Audit corrected H8 restore parity against the immutable duration-v1 H8 corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.before_we_act.analyze_mars_care_branch_duration import (
    audit_family,
)


TASKS = (
    "place_cube_in_cup",
    "strike_cube_hard",
    "three_robots_place_shoes",
    "four_robots_stack_cube",
)
FORMAT_VERSION = "before-we-act.care-mars-branch-parity-audit/2"


def _branch_key(branch: Mapping[str, Any]) -> tuple[int, str, int]:
    return (
        int(branch["repeat_id"]),
        str(branch["regime"]),
        int(branch["candidate_id"]),
    )


def _branch_map(family: Mapping[str, Any]) -> dict[tuple[int, str, int], Mapping[str, Any]]:
    return {_branch_key(branch): branch for branch in family["branches"]}


def _same_execution(baseline: Mapping[str, Any], corrected: Mapping[str, Any]) -> bool:
    """Compare all physical execution evidence, excluding restore diagnostics."""

    ignored = {
        "format_version",
        "checkpoint_sha256",
        "checkpoint",
        "wall_seconds",
        "json_sha256",
        "npz_sha256",
    }
    if any(baseline.get(key) != corrected.get(key) for key in ("snapshot_id", "task", "intervention_steps", "branch_count")):
        return False
    for key in ("candidate_legality", "snapshot_metrics", "prebranch_diagnostics"):
        if baseline.get(key) != corrected.get(key):
            return False
    left = _branch_map(baseline)
    right = _branch_map(corrected)
    if set(left) != set(right):
        return False
    branch_ignored = {
        "restore_observation_max_abs_error",
        "restore_rerender_diagnostic_max_abs_error",
        "restore_observation_source",
        "wall_seconds",
    }
    for key in sorted(left):
        a = {name: value for name, value in left[key].items() if name not in branch_ignored}
        b = {name: value for name, value in right[key].items() if name not in branch_ignored}
        if a != b:
            return False
    return True


def audit_task(
    baseline_path: Path,
    corrected_path: Path,
) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    corrected = json.loads(corrected_path.read_text(encoding="utf-8"))
    audit = audit_family(corrected_path, 8)
    execution_parity = _same_execution(baseline, corrected)
    passes_parity_and_safety = bool(
        execution_parity
        and audit["all_branch_support_complete"]
        and audit["all_candidates_legal"]
        and audit["maximum_restore_error"] <= 1e-6
        and audit["maximum_replay_teammate_action_error"] <= 1e-6
        and audit["maximum_candidate0_reference_action_error"] <= 1e-6
        and audit["hard_safety_pair_count"] == 0
    )
    return {
        "task": audit["task"],
        "baseline_path": str(baseline_path.resolve()),
        "corrected_path": str(corrected_path.resolve()),
        "execution_parity": execution_parity,
        "signal_density": audit["signal_density"],
        "direct_signal_pair_count": audit["direct_signal_pair_count"],
        "direct_signal_density": audit["direct_signal_density"],
        "response_signal_pair_count": audit["response_signal_pair_count"],
        "response_signal_density": audit["response_signal_density"],
        "total_signal_pair_count": audit["total_signal_pair_count"],
        "total_signal_density": audit["total_signal_density"],
        "effective_pair_count": audit["effective_pair_count"],
        "pair_count": audit["pair_count"],
        "all_candidates_legal": audit["all_candidates_legal"],
        "all_branch_support_complete": audit["all_branch_support_complete"],
        "maximum_restore_error": audit["maximum_restore_error"],
        "maximum_restore_rerender_diagnostic_error": audit[
            "maximum_restore_rerender_diagnostic_error"
        ],
        "maximum_replay_teammate_action_error": audit[
            "maximum_replay_teammate_action_error"
        ],
        "maximum_candidate0_reference_action_error": audit[
            "maximum_candidate0_reference_action_error"
        ],
        "hard_safety_pair_count": audit["hard_safety_pair_count"],
        # Signal promotion is intentionally not decided per task.  The
        # pre-registered duration rule is a four-task aggregate gate, so a
        # sparse task (for example hammer) must not veto a duration whose
        # aggregate signal gain clears the fixed threshold.
        "passes_parity_and_safety": passes_parity_and_safety,
    }


def aggregate_rows(
    rows: list[Mapping[str, Any]], *, baseline_h1_density: float
) -> dict[str, Any]:
    """Apply the pre-registered H8 gate once over all four task families."""

    aggregate_pairs = sum(int(row["pair_count"]) for row in rows)
    aggregate_effective = sum(int(row["effective_pair_count"]) for row in rows)
    signal_density = aggregate_effective / max(aggregate_pairs, 1)
    density_gain = signal_density - float(baseline_h1_density)
    density_ratio = (
        signal_density / float(baseline_h1_density)
        if baseline_h1_density > 0.0
        else None
    )
    aggregate = {
        "family_count": len(rows),
        "pair_count": aggregate_pairs,
        "effective_pair_count": aggregate_effective,
        "signal_density": signal_density,
        "baseline_signal_density_h1": float(baseline_h1_density),
        "signal_density_gain_over_h1": density_gain,
        "signal_density_ratio_over_h1": density_ratio,
        "direct_signal_pair_count": sum(
            int(row["direct_signal_pair_count"]) for row in rows
        ),
        "response_signal_pair_count": sum(
            int(row["response_signal_pair_count"]) for row in rows
        ),
        "total_signal_pair_count": sum(
            int(row["total_signal_pair_count"]) for row in rows
        ),
        "execution_parity": all(bool(row["execution_parity"]) for row in rows),
        "support_complete": all(
            bool(row["all_branch_support_complete"]) for row in rows
        ),
        "all_candidates_legal": all(
            bool(row["all_candidates_legal"]) for row in rows
        ),
        "maximum_restore_error": max(row["maximum_restore_error"] for row in rows),
        "maximum_restore_rerender_diagnostic_error": max(
            row["maximum_restore_rerender_diagnostic_error"] for row in rows
        ),
        "maximum_replay_teammate_action_error": max(
            row["maximum_replay_teammate_action_error"] for row in rows
        ),
        "maximum_candidate0_reference_action_error": max(
            row["maximum_candidate0_reference_action_error"] for row in rows
        ),
        "hard_safety_pair_count": sum(
            int(row["hard_safety_pair_count"]) for row in rows
        ),
    }
    aggregate["direct_signal_density"] = (
        aggregate["direct_signal_pair_count"] / max(aggregate_pairs, 1)
    )
    aggregate["response_signal_density"] = (
        aggregate["response_signal_pair_count"] / max(aggregate_pairs, 1)
    )
    aggregate["total_signal_density"] = (
        aggregate["total_signal_pair_count"] / max(aggregate_pairs, 1)
    )
    aggregate["eligible_for_scorer_smoke"] = bool(
        len(rows) == len(TASKS)
        and all(bool(row["passes_parity_and_safety"]) for row in rows)
        and density_gain >= 0.10
        and (density_ratio or 0.0) >= 1.5
    )
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--corrected-root", type=Path, required=True)
    parser.add_argument("--baseline-h1-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    h1 = json.loads(args.baseline_h1_audit.read_text(encoding="utf-8"))
    h1_density = float(h1["aggregate"]["1"]["signal_density"])
    rows = []
    for task in TASKS:
        baseline_paths = sorted((args.baseline_root / task).glob("*.json"))
        corrected_paths = sorted((args.corrected_root / task).glob("*.json"))
        if len(baseline_paths) != 1 or len(corrected_paths) != 1:
            raise RuntimeError(
                f"expected one baseline/corrected family for {task}, "
                f"got {baseline_paths} / {corrected_paths}"
            )
        rows.append(
            audit_task(baseline_paths[0], corrected_paths[0])
        )
    aggregate = aggregate_rows(rows, baseline_h1_density=h1_density)
    result = {
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "duration": 8,
        "protocol": "corrected restore parity; immutable v1 H8 execution baseline",
        "validation20_used_for_tuning": False,
        "main_protocol_unchanged": True,
        "families": rows,
        "aggregate": aggregate,
        "eligible_for_scorer_smoke": aggregate["eligible_for_scorer_smoke"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
