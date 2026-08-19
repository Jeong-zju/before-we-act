#!/usr/bin/env python3
"""在读取修订候选结果前冻结 A5R2 状态恢复/资源试跑清单。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from before_we_act.care_branch_collector import atomic_json, sha256_file
from two_three_task_manifest import get_task


TASKS = (
    "lift_barrier",
    "camera_alignment",
    "long_pipeline_delivery",
    "take_photo",
    "pass_shoe",
    "place_food",
)


def integer(namespace: str, modulo: int) -> int:
    return int.from_bytes(hashlib.sha256(namespace.encode("utf-8")).digest()[:8], "big") % modulo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--families-per-task", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite frozen pilot manifest: {args.output}")
    if not 1 <= args.families_per_task <= 40:
        raise ValueError("six-task resource pilot must contain 1..40 families per task")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract.get("stage_id") != "A4R2-CARE-CONTRACT":
        raise RuntimeError("wrong CARE contract stage")
    parent = contract.get("parent_contract", {})
    parent_path = Path(str(parent.get("path", "")))
    if not parent_path.is_absolute():
        parent_path = Path.cwd() / parent_path
    if not parent_path.is_file() or sha256_file(parent_path) != parent.get("sha256"):
        raise RuntimeError("A4R2 parent contract is missing or drifted")
    if sha256_file(args.checkpoint) != contract["reference_policy"]["checkpoint_sha256"]:
        raise RuntimeError("B-core checkpoint hash differs from A4")
    families = []
    used_seeds: set[int] = set()
    for task in TASKS:
        arms = tuple(int(value) for value in get_task(task)["agents"])
        for index in range(args.families_per_task):
            snapshot_id = f"pilot-r2-{task}-{index:03d}"
            seed = integer(
                f"A5R2-CARE-BRANCHES|pilot-v3|seed|{task}|{index}",
                2_147_483_646,
            ) + 1
            while seed in used_seeds:
                seed = seed % 2_147_483_646 + 1
            used_seeds.add(seed)
            anchor = 24 + integer(
                f"A5R2-CARE-BRANCHES|pilot-v3|anchor|{task}|{index}", 36
            )
            focal = arms[
                integer(snapshot_id + "|focal-v1", len(arms))
            ]
            families.append(
                {
                    "snapshot_id": snapshot_id,
                    "task": task,
                    "scenario_group_id": f"pilot-seed-{seed}",
                    "episode_seed": seed,
                    "anchor_step": anchor,
                    "focal_agent": focal,
                    "selection": (
                        "uniform hash in the preregistered resource-pilot window "
                        "steps 24..59 before any branch outcome"
                    ),
                }
            )
    value = {
        "format_version": "before-we-act.a5r2-care-resource-pilot-manifest/3",
        "stage_id": "A5R2-CARE-BRANCHES-PILOT",
        "status": "FROZEN_BEFORE_OUTCOME",
        "resource_only": True,
        "contract": str(args.contract.resolve()),
        "contract_sha256": sha256_file(args.contract),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "families_per_task": args.families_per_task,
        "family_count": len(families),
        "planned_branch_count": len(families) * 6 * 2 * 2,
        "families": families,
        "forbidden_uses": [
            "CARE training",
            "Gate A candidate headroom",
            "Gate B team-response signal",
            "changing the A4R2 candidate family after inspecting outcomes",
        ],
        "allowed_measurements": [
            "restore validity",
            "reactive/replay mechanism fidelity",
            "candidate legality rate",
            "reference repeat noise",
            "wall time, GPU memory, and artifact bytes",
        ],
    }
    atomic_json(args.output, value)
    print(json.dumps(value | {"families": "frozen"}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
