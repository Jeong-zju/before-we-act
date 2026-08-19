#!/usr/bin/env python3
"""只用分支前扫描信号生成冻结规模的 CARE 门槛清单。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from before_we_act.formal_care_sampling import (
    TASKS,
    compact_gate_targets,
    fitted_references,
    gate_first_targets,
    score_critical,
    split_name,
)


SUPPORTED_SCAN_STAGES = {
    "A5R5-CARE-GATE-FIRST-PREBRANCH-SCAN": {
        "contract_stage": "A4R5-CARE-GATE-FIRST-COLLECTION",
        "branch_stage": "A5R5-CARE-GATE-FIRST-BRANCHES",
        "manifest_format": "before-we-act.a5r5-care-gate-first-manifest/1",
        "snapshot_prefix": "gate-r5",
        "targets": gate_first_targets,
    },
    "A5R6-CARE-COMPACT-PREBRANCH-SCAN": {
        "contract_stage": "A4R6-CARE-COMPACT-COLLECTION",
        "branch_stage": "A5R6-CARE-COMPACT-BRANCHES",
        "manifest_format": "before-we-act.a5r6-care-compact-manifest/1",
        "snapshot_prefix": "compact-r6",
        "targets": compact_gate_targets,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-manifest", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"拒绝覆盖已冻结门槛清单：{args.output}")
    scan_manifest = json.loads(args.scan_manifest.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    scan_stage = str(scan_manifest.get("stage_id"))
    stage = SUPPORTED_SCAN_STAGES.get(scan_stage)
    if stage is None or contract.get("stage_id") != stage["contract_stage"]:
        raise RuntimeError("CARE 门槛扫描清单或合同阶段错误")
    scan_manifest_sha = sha256_file(args.scan_manifest)
    contract_sha = sha256_file(args.contract)
    if scan_manifest["contract_sha256"] != contract_sha:
        raise RuntimeError("CARE 门槛扫描清单的合同发生漂移")

    rows = []
    missing = []
    for candidate in scan_manifest["candidates"]:
        path = args.scan_root / candidate["task"] / f"{candidate['scan_id']}.json"
        if not path.is_file():
            missing.append(candidate["scan_id"])
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("stage_id") != scan_stage
            or row.get("scan_manifest_sha256") != scan_manifest_sha
            or row.get("contract_sha256") != contract_sha
        ):
            raise RuntimeError(f"CARE 门槛扫描结果来源不一致：{path}")
        rows.append(row)
    if missing:
        raise RuntimeError(
            f"CARE 门槛分支前扫描尚缺 {len(missing)} 项；首项：{missing[0]}"
        )

    valid = [row for row in rows if bool(row["valid"])]
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in valid:
        candidate = row["candidate"]
        key = (
            str(candidate["target_split"]),
            str(candidate["pool_stratum"]),
            str(candidate["task"]),
        )
        grouped[key].append(row)

    reference_count = int(scan_manifest["percentile_reference_selected_per_task"])
    reference_selected = {}
    for task in TASKS:
        key = ("train", "reference_uniform", task)
        choices = sorted(grouped[key], key=lambda row: row["candidate"]["selection_rank"])
        if len(choices) < reference_count:
            raise RuntimeError(
                f"{task} 的分位数参考状态不足：{len(choices)} < {reference_count}"
            )
        reference_selected[task] = choices[:reference_count]
    references = {
        task: fitted_references(reference_selected[task]) for task in TASKS
    }

    targets = stage["targets"]()
    selected_by_key: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    scored_by_scan_id: dict[str, tuple[float, dict[str, float]]] = {}
    critical_pool_counts = {}
    for key, target in targets.items():
        split, stratum, task = key
        choices = grouped[key]
        if stratum == "uniform":
            ranked = sorted(choices, key=lambda row: row["candidate"]["selection_rank"])
            if len(ranked) < target:
                raise RuntimeError(f"{key} 有效普通状态不足：{len(ranked)} < {target}")
            selected_by_key[key] = ranked[:target]
            continue
        scored = []
        for row in choices:
            score, percentiles = score_critical(row, references[task])
            scored_by_scan_id[str(row["scan_id"])] = (score, percentiles)
            scored.append((score, str(row["candidate"]["selection_rank"]), row))
        scored.sort(key=lambda item: (-item[0], item[1]))
        eligible_count = math.floor(0.30 * len(scored))
        if eligible_count < target:
            raise RuntimeError(f"{key} 协作困难状态不足：top30%={eligible_count} < {target}")
        eligible = scored[:eligible_count]
        chosen = sorted(eligible, key=lambda item: item[1])[:target]
        selected_by_key[key] = [item[2] for item in chosen]
        critical_pool_counts["|".join(key)] = {
            "valid_candidates": len(scored),
            "top30_candidates": eligible_count,
            "selected": target,
        }

    families = []
    for key in sorted(targets):
        split, stratum, task = key
        for index, row in enumerate(selected_by_key[key]):
            candidate = row["candidate"]
            if not bool(candidate["collect_branches"]):
                raise AssertionError("分位数参考状态不得进入候选分支清单")
            family = {
                "snapshot_id": (
                    f"{stage['snapshot_prefix']}-{split}-{stratum}-{task}-{index:04d}"
                ),
                "task": task,
                "split": split,
                "sampling_stratum": stratum,
                "scenario_group_id": candidate["scenario_group_id"],
                "split_hash_bucket": int(candidate["split_hash_bucket"]),
                "episode_seed": int(candidate["episode_seed"]),
                "anchor_step": int(candidate["anchor_step"]),
                "focal_agent": int(candidate["focal_agent"]),
                "source_scan_id": row["scan_id"],
                "source_scan_content_sha256": hashlib.sha256(
                    json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "selection": (
                    "uniform frozen hash rank"
                    if stratum == "uniform"
                    else "frozen hash rank inside top-30-percent prebranch critical pool"
                ),
            }
            if stratum == "critical":
                score, percentiles = scored_by_scan_id[str(row["scan_id"])]
                family["critical_score"] = score
                family["critical_feature_percentiles"] = percentiles
                family["critical_features"] = row["features"]
            if split_name(task, str(candidate["scenario_group_id"])) != "test":
                raise AssertionError("CARE 门槛状态不在冻结测试桶")
            families.append(family)

    counts = Counter(
        (row["split"], row["sampling_stratum"], row["task"]) for row in families
    )
    if any(counts[key] != value for key, value in targets.items()):
        raise AssertionError("CARE 门槛清单的任务与分层配额发生漂移")
    per_task = Counter(row["task"] for row in families)
    expected_per_task = {
        task: sum(
            count
            for (_split, _stratum, row_task), count in targets.items()
            if row_task == task
        )
        for task in TASKS
    }
    if dict(per_task) != expected_per_task:
        raise AssertionError("CARE 门槛清单的每任务状态配额发生漂移")
    value = {
        "format_version": stage["manifest_format"],
        "stage_id": stage["branch_stage"],
        "status": "FROZEN_BEFORE_GATE_BRANCH_OUTCOME",
        "resource_only": False,
        "gate_only": True,
        "contract": str(args.contract.resolve()),
        "contract_sha256": contract_sha,
        "checkpoint": scan_manifest["checkpoint"],
        "checkpoint_sha256": scan_manifest["checkpoint_sha256"],
        "source_scan_manifest": str(args.scan_manifest.resolve()),
        "source_scan_manifest_sha256": scan_manifest_sha,
        "source_scan_candidate_count": len(rows),
        "source_scan_invalid_count": len(rows) - len(valid),
        "percentile_reference_rows_per_task": reference_count,
        "family_count": len(families),
        "families_per_task": dict(sorted(per_task.items())),
        "planned_branch_count": len(families) * 24,
        "allocation": [
            {"split": key[0], "stratum": key[1], "task": key[2], "count": count}
            for key, count in sorted(targets.items())
        ],
        "critical_pool_counts": critical_pool_counts,
        "families": families,
        "authorized_uses_after_collection_receipt": ["Gate A", "Gate B"],
        "forbidden_uses": ["CARE training", "validation", "calibration"],
    }
    atomic_json(args.output, value)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "family_count": len(families),
                "families_per_task": dict(sorted(per_task.items())),
                "branch_count": len(families) * 24,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
