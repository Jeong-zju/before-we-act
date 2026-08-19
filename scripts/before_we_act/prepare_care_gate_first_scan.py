#!/usr/bin/env python3
"""在分支结果产生前冻结每任务 150 状态的 CARE 门槛扫描清单。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from before_we_act.formal_care_sampling import (
    TASKS,
    compact_gate_targets,
    gate_first_targets,
    hash_integer,
    pool_size,
    split_name,
)
from two_three_task_manifest import get_task


SUPPORTED_CONTRACTS = {
    "A4R5-CARE-GATE-FIRST-COLLECTION": {
        "scan_stage": "A5R5-CARE-GATE-FIRST-PREBRANCH-SCAN",
        "scan_format": "before-we-act.a5r5-care-gate-first-scan-manifest/1",
        "namespace": "A5R5|gate-first-scan",
        "group_prefix": "gate-first-v1",
        "reference_count_per_task": 200,
        "targets": gate_first_targets,
    },
    "A4R6-CARE-COMPACT-COLLECTION": {
        "scan_stage": "A5R6-CARE-COMPACT-PREBRANCH-SCAN",
        "scan_format": "before-we-act.a5r6-care-compact-scan-manifest/1",
        "namespace": "A5R6|compact-scan",
        "group_prefix": "compact-v1",
        "reference_count_per_task": 50,
        "targets": compact_gate_targets,
    },
}


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


def next_group(
    task: str, split: str, stratum: str, serial: int, group_prefix: str
) -> tuple[str, int]:
    probe = serial
    while True:
        group = f"{group_prefix}-{stratum}-{task}-{split}-{probe:08d}"
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
    stage = SUPPORTED_CONTRACTS.get(str(contract.get("stage_id")))
    if stage is None:
        raise RuntimeError("不支持的 CARE 紧凑采集合同阶段")
    parent = contract["parent_contract"]
    parent_path = Path(parent["path"])
    if not parent_path.is_absolute():
        parent_path = Path.cwd() / parent_path
    if sha256_file(parent_path) != parent["sha256"]:
        raise RuntimeError("A4R5 父合同缺失或发生漂移")
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != contract["reference_policy"]["checkpoint_sha256"]:
        raise RuntimeError("B-core 部署权重与 A4R5 合同不一致")

    targets = stage["targets"]()
    reference_count_per_task = int(stage["reference_count_per_task"])
    specifications = []
    for task in TASKS:
        specifications.append(
            {
                "task": task,
                "split": "train",
                "stratum": "reference_uniform",
                "target_count": reference_count_per_task,
                "pool_count": math.ceil(reference_count_per_task / 0.95),
                "collect_branches": False,
            }
        )
        for stratum in ("critical", "uniform"):
            target = targets[("test", stratum, task)]
            specifications.append(
                {
                    "task": task,
                    "split": "test",
                    "stratum": stratum,
                    "target_count": target,
                    "pool_count": pool_size(target, stratum),
                    "collect_branches": True,
                }
            )

    candidates = []
    used_seeds: set[int] = set()
    for specification in specifications:
        task = str(specification["task"])
        split = str(specification["split"])
        stratum = str(specification["stratum"])
        arms = tuple(int(value) for value in get_task(task)["agents"])
        serial = 0
        for pool_index in range(int(specification["pool_count"])):
            group, serial = next_group(
                task, split, stratum, serial, str(stage["group_prefix"])
            )
            candidate_id = f"scan-{stratum}-{split}-{task}-{pool_index:05d}"
            seed = hash_integer(
                f"{stage['namespace']}|episode-seed|{group}", 2_147_483_646
            ) + 1
            while seed in used_seeds:
                seed = seed % 2_147_483_646 + 1
            used_seeds.add(seed)
            anchor = 24 + hash_integer(
                f"{stage['namespace']}|anchor|{group}", 36
            )
            focal = arms[hash_integer(candidate_id + "|focal-v1", len(arms))]
            candidates.append(
                {
                    "scan_id": candidate_id,
                    "task": task,
                    "target_split": split,
                    "pool_stratum": stratum,
                    "target_count": int(specification["target_count"]),
                    "collect_branches": bool(specification["collect_branches"]),
                    "scenario_group_id": group,
                    "split_hash_bucket": hash_integer(
                        f"A4-CARE-CONTRACT|split-v1|{task}|{group}", 100
                    ),
                    "episode_seed": seed,
                    "anchor_step": anchor,
                    "focal_agent": focal,
                    "selection_rank": hashlib.sha256(
                        f"{stage['namespace']}|selection-rank|{candidate_id}".encode()
                    ).hexdigest(),
                }
            )

    value = {
        "format_version": stage["scan_format"],
        "stage_id": stage["scan_stage"],
        "status": "FROZEN_BEFORE_GATE_BRANCH_OUTCOME",
        "contract": str(args.contract.resolve()),
        "contract_sha256": sha256_file(args.contract),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "anchor_window_steps_inclusive": [24, 59],
        "branch_targets": [
            {"split": key[0], "stratum": key[1], "task": key[2], "count": count}
            for key, count in sorted(targets.items())
        ],
        "percentile_reference_selected_per_task": reference_count_per_task,
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
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidate_count": len(candidates),
                "branch_family_target": sum(targets.values()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
