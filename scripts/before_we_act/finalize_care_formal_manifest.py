#!/usr/bin/env python3
"""只用冻结的分支前扫描信号生成 14,800 状态正式清单。"""
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
    fitted_references,
    formal_targets,
    score_critical,
    split_name,
)


SCAN_STAGE = "A5R4-CARE-FORMAL-PREBRANCH-SCAN"
FORMAL_STAGE = "A5R4-CARE-BRANCHES-FORMAL"
CONTRACT_STAGE = "A4R4-CARE-FORMAL-COLLECTION"


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
        raise RuntimeError(f"拒绝覆盖已冻结正式清单：{args.output}")
    scan_manifest = json.loads(args.scan_manifest.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if scan_manifest.get("stage_id") != SCAN_STAGE or contract.get("stage_id") != CONTRACT_STAGE:
        raise RuntimeError("扫描清单或正式合同阶段错误")
    scan_manifest_sha = sha256_file(args.scan_manifest)
    contract_sha = sha256_file(args.contract)
    if scan_manifest["contract_sha256"] != contract_sha:
        raise RuntimeError("扫描清单的正式合同发生漂移")

    rows = []
    missing = []
    for candidate in scan_manifest["candidates"]:
        path = args.scan_root / candidate["task"] / f"{candidate['scan_id']}.json"
        if not path.is_file():
            missing.append(candidate["scan_id"])
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("stage_id") != SCAN_STAGE
            or row.get("scan_manifest_sha256") != scan_manifest_sha
            or row.get("contract_sha256") != contract_sha
        ):
            raise RuntimeError(f"扫描结果来源不一致：{path}")
        rows.append(row)
    if missing:
        raise RuntimeError(f"正式分支前扫描尚缺 {len(missing)} 项；首项：{missing[0]}")

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
    targets = formal_targets()

    uniform_selected: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for key, target in targets.items():
        if key[1] != "uniform":
            continue
        choices = sorted(grouped[key], key=lambda row: row["candidate"]["selection_rank"])
        if len(choices) < target:
            raise RuntimeError(f"{key} 有效均匀状态不足：{len(choices)} < {target}")
        uniform_selected[key] = choices[:target]

    references = {
        task: fitted_references(uniform_selected[("train", "uniform", task)])
        for task in TASKS
    }
    critical_selected: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    scored_by_scan_id: dict[str, tuple[float, dict[str, float]]] = {}
    critical_pool_counts = {}
    for key, target in targets.items():
        if key[1] != "critical":
            continue
        task = key[2]
        scored = []
        for row in grouped[key]:
            score, percentiles = score_critical(row, references[task])
            scored_by_scan_id[str(row["scan_id"])] = (score, percentiles)
            scored.append((score, str(row["candidate"]["selection_rank"]), row))
        scored.sort(key=lambda item: (-item[0], item[1]))
        eligible_count = math.floor(0.30 * len(scored))
        if eligible_count < target:
            raise RuntimeError(
                f"{key} 协调临界池不足：top30%={eligible_count} < {target}"
            )
        eligible = scored[:eligible_count]
        chosen = sorted(eligible, key=lambda item: item[1])[:target]
        critical_selected[key] = [item[2] for item in chosen]
        critical_pool_counts["|".join(key)] = {
            "valid_candidates": len(scored),
            "top30_candidates": eligible_count,
            "selected": target,
        }

    families = []
    for key in sorted(targets):
        split, stratum, task = key
        selected = (
            uniform_selected[key] if stratum == "uniform" else critical_selected[key]
        )
        for index, row in enumerate(selected):
            candidate = row["candidate"]
            snapshot_id = f"formal-r4-{split}-{stratum}-{task}-{index:05d}"
            family = {
                "snapshot_id": snapshot_id,
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
                "selection": "uniform frozen hash rank" if stratum == "uniform" else "frozen hash rank inside top-30-percent prebranch critical pool",
            }
            if stratum == "critical":
                score, percentiles = scored_by_scan_id[str(row["scan_id"])]
                family["critical_score"] = score
                family["critical_feature_percentiles"] = percentiles
                family["critical_features"] = row["features"]
            if split_name(task, str(candidate["scenario_group_id"])) != split:
                raise AssertionError("scenario group split drift")
            families.append(family)

    counts = Counter((row["split"], row["sampling_stratum"], row["task"]) for row in families)
    if any(counts[key] != value for key, value in targets.items()):
        raise AssertionError("formal family allocation drift")
    if len({row["scenario_group_id"] for row in families}) != len(families):
        raise AssertionError("formal scenario groups are not unique")
    value = {
        "format_version": "before-we-act.a5r4-care-formal-manifest/1",
        "stage_id": FORMAL_STAGE,
        "status": "FROZEN_BEFORE_FORMAL_BRANCH_OUTCOME",
        "resource_only": False,
        "contract": str(args.contract.resolve()),
        "contract_sha256": contract_sha,
        "checkpoint": scan_manifest["checkpoint"],
        "checkpoint_sha256": scan_manifest["checkpoint_sha256"],
        "source_scan_manifest": str(args.scan_manifest.resolve()),
        "source_scan_manifest_sha256": scan_manifest_sha,
        "source_scan_candidate_count": len(rows),
        "source_scan_invalid_count": len(rows) - len(valid),
        "family_count": len(families),
        "planned_branch_count": len(families) * 24,
        "allocation": [
            {"split": key[0], "stratum": key[1], "task": key[2], "count": value}
            for key, value in sorted(targets.items())
        ],
        "critical_pool_counts": critical_pool_counts,
        "families": families,
        "authorized_uses_after_collection_receipt": ["Gate A", "Gate B"],
        "forbidden_uses_before_gate_a_and_gate_b_pass": ["CARE training"],
    }
    atomic_json(args.output, value)
    print(json.dumps({"output": str(args.output), "family_count": len(families), "branch_count": len(families) * 24}, sort_keys=True))


if __name__ == "__main__":
    main()
