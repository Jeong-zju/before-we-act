#!/usr/bin/env python3
"""Verify the frozen 120-family MARS CARE H8 branch corpus."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from before_we_act.care_branch_collector import OUTCOME_HORIZONS, sha256_file
from before_we_act.mars_care_branch_collector import BRANCH_COUNT, FORMAT_VERSION
from scripts.before_we_act.analyze_mars_care_branch_duration import atomic_json


TASKS = (
    "place_cube_in_cup",
    "strike_cube_hard",
    "three_robots_place_shoes",
    "four_robots_stack_cube",
)
RECEIPT_VERSION = "before-we-act.care-mars-h8-corpus-receipt/1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _branch_key(row: Mapping[str, Any]) -> tuple[int, str, int]:
    return int(row["repeat_id"]), str(row["regime"]), int(row["candidate_id"])


def verify_corpus(
    manifest_path: Path,
    family_root: Path,
    reference_checkpoint: Path,
    *,
    intervention_steps: int = 8,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != "before-we-act.care-mars-family-manifest/1":
        raise ValueError("wrong frozen family manifest format")
    expected_rows = list(manifest.get("families", ()))
    if len(expected_rows) != 120:
        raise ValueError(f"H8 formal corpus requires 120 families, got {len(expected_rows)}")
    expected = {str(row["snapshot_id"]): row for row in expected_rows}
    if len(expected) != len(expected_rows):
        raise ValueError("frozen family manifest contains duplicate snapshot ids")
    task_counts = Counter(str(row["task"]) for row in expected_rows)
    if task_counts != Counter({task: 30 for task in TASKS}):
        raise ValueError(f"frozen family task support drifted: {dict(task_counts)}")

    json_paths = sorted(family_root.glob("*/*.json"))
    npz_paths = sorted(family_root.glob("*/*.npz"))
    json_ids = {path.stem for path in json_paths}
    npz_ids = {path.stem for path in npz_paths}
    if len(json_paths) != 120 or json_ids != set(expected):
        raise ValueError("H8 JSON family coverage is incomplete or contains extras")
    if len(npz_paths) != 120 or npz_ids != set(expected):
        raise ValueError("H8 NPZ family coverage is incomplete or contains extras")

    checkpoint_sha256 = sha256_file(reference_checkpoint)
    maximum_restore = 0.0
    maximum_rerender = 0.0
    maximum_replay = 0.0
    maximum_candidate0 = 0.0
    hard_safety = 0
    rows = []
    for path in json_paths:
        snapshot_id = path.stem
        source = expected[snapshot_id]
        family = json.loads(path.read_text(encoding="utf-8"))
        if family.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"branch format drift: {path}")
        if str(family.get("snapshot_id")) != snapshot_id:
            raise ValueError(f"snapshot id drift: {path}")
        if str(family.get("task")) != str(source["task"]):
            raise ValueError(f"task drift: {path}")
        if int(family.get("intervention_steps", -1)) != int(intervention_steps):
            raise ValueError(f"intervention duration drift: {path}")
        if family.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError(f"reference checkpoint drift: {path}")
        branches = list(family.get("branches", ()))
        if int(family.get("branch_count", -1)) != BRANCH_COUNT or len(branches) != BRANCH_COUNT:
            raise ValueError(f"branch support is incomplete: {path}")
        keys = {_branch_key(branch) for branch in branches}
        expected_keys = {
            (repeat, regime, candidate)
            for repeat in (0, 1)
            for regime in ("reactive", "replay")
            for candidate in range(6)
        }
        if keys != expected_keys:
            raise ValueError(f"paired branch identities drifted: {path}")
        if not all(bool(row.get("valid")) for row in family.get("candidate_legality", ())):
            raise ValueError(f"candidate legality failed: {path}")
        if not all(
            branch.get("candidate_valid") is True
            and all(str(horizon) in branch.get("outcomes", {}) for horizon in OUTCOME_HORIZONS)
            for branch in branches
        ):
            raise ValueError(f"outcome horizon support is incomplete: {path}")

        restore = max(float(row["restore_observation_max_abs_error"]) for row in branches)
        rerender = max(
            float(row.get("restore_rerender_diagnostic_max_abs_error", 0.0))
            for row in branches
        )
        replay = max(float(row["replay_teammate_action_max_abs_error"]) for row in branches)
        candidate0 = max(
            float(row["candidate0_reference_action_max_abs_error"])
            for row in branches
            if int(row["candidate_id"]) == 0
        )
        if restore > 1e-6 or replay > 1e-6 or candidate0 > 1e-6:
            raise ValueError(f"physical restore/replay/reference parity failed: {path}")
        maximum_restore = max(maximum_restore, restore)
        maximum_rerender = max(maximum_rerender, rerender)
        maximum_replay = max(maximum_replay, replay)
        maximum_candidate0 = max(maximum_candidate0, candidate0)
        hard_safety += sum(
            int(bool(outcome.get("hard_safety_violation")))
            for branch in branches
            for outcome in branch["outcomes"].values()
        )

        npz_path = path.with_suffix(".npz")
        with np.load(npz_path, allow_pickle=False) as values:
            shapes = {name: tuple(values[name].shape) for name in values.files}
        expected_shapes = {
            "memory": (20, 384),
            "memory_mask": (20,),
            "candidate_chunks": (6, 100, 8),
            "focal_agent": (1,),
        }
        if shapes != expected_shapes:
            raise ValueError(f"H8 training tensor contract drifted: {npz_path}: {shapes}")
        rows.append(
            {
                "snapshot_id": snapshot_id,
                "task": str(family["task"]),
                "json": str(path.resolve()),
                "json_sha256": sha256_file(path),
                "npz": str(npz_path.resolve()),
                "npz_sha256": sha256_file(npz_path),
                "wall_seconds": float(family.get("wall_seconds", 0.0)),
            }
        )

    return {
        "format_version": RECEIPT_VERSION,
        "status": "PASSED",
        "created_at_utc": utc_now(),
        "protocol": "frozen fixed-stratified 20 critical + 10 uniform per task",
        "intervention_steps": int(intervention_steps),
        "family_count": len(rows),
        "branches_per_family": BRANCH_COUNT,
        "tasks": {task: task_counts[task] for task in TASKS},
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "reference_checkpoint": str(reference_checkpoint.resolve()),
        "reference_checkpoint_sha256": checkpoint_sha256,
        "all_candidates_legal": True,
        "all_branch_support_complete": True,
        "maximum_restore_error": maximum_restore,
        "maximum_restore_rerender_diagnostic_error": maximum_rerender,
        "maximum_replay_teammate_action_error": maximum_replay,
        "maximum_candidate0_reference_action_error": maximum_candidate0,
        "hard_safety_outcome_count": hard_safety,
        "validation20_used_for_tuning": False,
        "legacy_h1_corpus_unchanged": True,
        "automatic_retry": False,
        "globally_serial_vulkan": True,
        "total_collection_wall_seconds": sum(row["wall_seconds"] for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--intervention-steps", type=int, choices=(8,), default=8)
    args = parser.parse_args()
    result = verify_corpus(
        args.manifest,
        args.family_root,
        args.reference_checkpoint,
        intervention_steps=args.intervention_steps,
    )
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
