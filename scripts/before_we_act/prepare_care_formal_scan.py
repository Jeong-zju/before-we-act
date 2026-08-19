#!/usr/bin/env python3
"""在任何正式分支结果产生前冻结 CARE 正式候选状态扫描清单。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from before_we_act.formal_care_sampling import (
    TASKS,
    formal_targets,
    hash_integer,
    pool_size,
    split_name,
)
from two_three_task_manifest import get_task


CONTRACT_STAGE = "A4R4-CARE-FORMAL-COLLECTION"
SCAN_STAGE = "A5R4-CARE-FORMAL-PREBRANCH-SCAN"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def next_group(task: str, split: str, stratum: str, serial: int) -> tuple[str, int]:
    probe = serial
    while True:
        group = f"formal-v1-{stratum}-{task}-{split}-{probe:08d}"
        if split_name(task, group) == split:
            return group, probe + 1
        probe += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"拒绝覆盖已冻结扫描清单：{args.output}")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract.get("stage_id") != CONTRACT_STAGE:
        raise RuntimeError("正式采集合同阶段错误")
    parent = contract["parent_contract"]
    parent_path = Path(parent["path"])
    if not parent_path.is_absolute():
        parent_path = Path.cwd() / parent_path
    if sha256_file(parent_path) != parent["sha256"]:
        raise RuntimeError("A4R4 父合同缺失或发生漂移")
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != contract["reference_policy"]["checkpoint_sha256"]:
        raise RuntimeError("B-core 部署权重与正式合同不一致")

    targets = formal_targets()
    candidates = []
    used_seeds: set[int] = set()
    serials = {(task, split, stratum): 0 for task in TASKS for split in ("train", "validation", "calibration", "test") for stratum in ("uniform", "critical")}
    for (split, stratum, task), target_count in sorted(targets.items()):
        arms = tuple(int(value) for value in get_task(task)["agents"])
        size = pool_size(target_count, stratum)
        key = (task, split, stratum)
        for pool_index in range(size):
            group, serials[key] = next_group(task, split, stratum, serials[key])
            candidate_id = f"scan-{stratum}-{split}-{task}-{pool_index:05d}"
            seed = hash_integer(
                f"A5R4|formal-scan|episode-seed|{group}", 2_147_483_646
            ) + 1
            while seed in used_seeds:
                seed = seed % 2_147_483_646 + 1
            used_seeds.add(seed)
            anchor = 24 + hash_integer(f"A5R4|formal-scan|anchor|{group}", 36)
            focal = arms[hash_integer(candidate_id + "|focal-v1", len(arms))]
            candidates.append(
                {
                    "scan_id": candidate_id,
                    "task": task,
                    "target_split": split,
                    "pool_stratum": stratum,
                    "target_count": target_count,
                    "scenario_group_id": group,
                    "split_hash_bucket": hash_integer(
                        f"A4-CARE-CONTRACT|split-v1|{task}|{group}", 100
                    ),
                    "episode_seed": seed,
                    "anchor_step": anchor,
                    "focal_agent": focal,
                    "selection_rank": hashlib.sha256(
                        f"A5R4|formal-scan|selection-rank|{candidate_id}".encode()
                    ).hexdigest(),
                }
            )
    value = {
        "format_version": "before-we-act.a5r4-care-formal-scan-manifest/1",
        "stage_id": SCAN_STAGE,
        "status": "FROZEN_BEFORE_FORMAL_BRANCH_OUTCOME",
        "contract": str(args.contract.resolve()),
        "contract_sha256": sha256_file(args.contract),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "anchor_window_steps_inclusive": [24, 59],
        "formal_targets": [
            {"split": key[0], "stratum": key[1], "task": key[2], "count": count}
            for key, count in sorted(targets.items())
        ],
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selection_inputs": [
            "B-core residual norm",
            "belief entropy",
            "evidence reliability",
            "partner segmentation visibility",
            "contact or immediate phase change",
            "paired inactivity",
        ],
        "forbidden_selection_inputs": contract["split_and_sampling"]["forbidden_selection_inputs"],
    }
    atomic_json(args.output, value)
    print(json.dumps({"output": str(args.output), "candidate_count": len(candidates)}, sort_keys=True))


if __name__ == "__main__":
    main()
