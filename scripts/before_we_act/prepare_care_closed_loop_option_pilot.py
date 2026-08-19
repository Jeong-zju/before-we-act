#!/usr/bin/env python3
"""Freeze the outcome-blind 30-family A4R9 closed-loop option pilot."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


CONTRACT_STAGE = "A4R9-CARE-CLOSED-LOOP-OPTION-PILOT"
SOURCE_STAGE = "A5R7-CARE-COMMON-SUPPORT-BRANCHES"
QUALITY_STAGE = "A5R7Q1-CARE-SIMULATOR-QUALITY-LABELS"
TARGET_STAGE = "A5R8-CARE-CLOSED-LOOP-OPTION-PILOT"
EXPECTED_TASKS = (
    "camera_alignment",
    "lift_barrier",
    "long_pipeline_delivery",
    "pass_shoe",
    "place_food",
    "take_photo",
)


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
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--quality-summary", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite frozen pilot manifest: {args.output}")

    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    quality_summary = json.loads(args.quality_summary.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract.get("stage_id") != CONTRACT_STAGE or contract.get("status") != "FROZEN_BEFORE_OPTION_OUTCOMES":
        raise RuntimeError("A4R9 pilot contract is not frozen")
    if source.get("stage_id") != SOURCE_STAGE or source.get("status") != "FROZEN_BEFORE_GATE_BRANCH_OUTCOME":
        raise RuntimeError("A5R7 source manifest is not frozen")
    if quality_summary.get("stage_id") != QUALITY_STAGE:
        raise RuntimeError("wrong A5R7Q1 quality stage")

    frozen = contract["frozen_source"]
    source_sha = sha256_file(args.source_manifest)
    quality_summary_sha = sha256_file(args.quality_summary)
    if source_sha != frozen["manifest_sha256"]:
        raise RuntimeError("A5R7 source manifest drifted")
    if quality_summary_sha != frozen["quality_summary_sha256"]:
        raise RuntimeError("A5R7Q1 quality summary drifted")
    if source.get("checkpoint_sha256") != frozen["reference_policy_checkpoint_sha256"]:
        raise RuntimeError("A5R7 reference checkpoint drifted")

    selection = contract["sample_selection"]
    primary_horizon = str(int(selection["primary_horizon_steps"]))
    per_task = int(selection["families_per_task"])
    salt = str(selection["rank_salt"])
    quality_root = args.quality_summary.parent
    source_root = args.source_manifest.parent.parent
    ranked: dict[str, list[tuple[str, dict[str, Any], Path, Path]]] = defaultdict(list)
    for source_family in source["families"]:
        if source_family.get("split") != "test" or source_family.get("sampling_stratum") != "critical":
            continue
        task = str(source_family["task"])
        snapshot_id = str(source_family["snapshot_id"])
        quality_path = quality_root / task / f"{snapshot_id}.quality.json"
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        if quality.get("stage_id") != QUALITY_STAGE or quality.get("snapshot_id") != snapshot_id:
            raise RuntimeError(f"quality sidecar identity mismatch: {quality_path}")
        horizon = quality["horizons"][primary_horizon]
        if not bool(horizon["use_for_gate_analysis"]):
            continue
        source_family_path = source_root / "families" / task / f"{snapshot_id}.json"
        if not source_family_path.is_file():
            raise RuntimeError(f"missing A5R7 family: {source_family_path}")
        rank = hashlib.sha256(f"{salt}{snapshot_id}".encode("utf-8")).hexdigest()
        ranked[task].append((rank, deepcopy(source_family), quality_path, source_family_path))

    if set(ranked) != set(EXPECTED_TASKS):
        raise RuntimeError("pilot eligible population does not cover all six tasks")
    families = []
    eligible_counts = {}
    for task in EXPECTED_TASKS:
        rows = sorted(ranked[task], key=lambda row: (row[0], row[1]["snapshot_id"]))
        eligible_counts[task] = len(rows)
        if len(rows) < per_task:
            raise RuntimeError(f"task {task} has fewer than {per_task} eligible families")
        for rank, source_family, quality_path, source_family_path in rows[:per_task]:
            family = deepcopy(source_family)
            family.update(
                {
                    "selection_rank_sha256": rank,
                    "source_a5r7_snapshot_id": source_family["snapshot_id"],
                    "source_a5r7_family": str(source_family_path.resolve()),
                    "source_a5r7_family_sha256": sha256_file(source_family_path),
                    "source_a5r7_quality": str(quality_path.resolve()),
                    "source_a5r7_quality_sha256": sha256_file(quality_path),
                }
            )
            family["snapshot_id"] = str(source_family["snapshot_id"]).replace(
                "compact-r7-", "option-r9-", 1
            )
            families.append(family)

    if len(families) != int(selection["family_count"]):
        raise RuntimeError("pilot family count drifted")
    allocation = Counter(row["task"] for row in families)
    if any(allocation[task] != per_task for task in EXPECTED_TASKS):
        raise RuntimeError("pilot per-task allocation drifted")

    branch_protocol = contract["branch_protocol"]
    value = {
        "format_version": "before-we-act.a5r8-care-closed-loop-option-pilot-manifest/1",
        "stage_id": TARGET_STAGE,
        "status": "FROZEN_BEFORE_OPTION_OUTCOMES",
        "resource_only": True,
        "contract": str(args.contract.resolve()),
        "contract_sha256": sha256_file(args.contract),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": source_sha,
        "quality_summary": str(args.quality_summary.resolve()),
        "quality_summary_sha256": quality_summary_sha,
        "checkpoint": frozen["reference_policy_checkpoint"],
        "checkpoint_sha256": frozen["reference_policy_checkpoint_sha256"],
        "selection_did_not_read_nonreference_outcomes": True,
        "eligible_critical_use_families_by_task": eligible_counts,
        "families_per_task": per_task,
        "family_count": len(families),
        "new_branches_per_family": int(branch_protocol["new_branches_per_family"]),
        "planned_new_branch_count": int(branch_protocol["new_branch_count"]),
        "families": families,
        "forbidden_uses": contract["output"]["forbidden_uses"],
    }
    atomic_json(args.output, value)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "contract_sha256": value["contract_sha256"],
                "family_count": len(families),
                "planned_new_branch_count": value["planned_new_branch_count"],
                "eligible_counts": eligible_counts,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
