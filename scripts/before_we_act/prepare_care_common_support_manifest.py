#!/usr/bin/env python3
"""Freeze A4R7 on the exact pre-outcome A4R6 family selection."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SOURCE_STAGE = "A5R6-CARE-COMPACT-BRANCHES"
TARGET_STAGE = "A5R7-CARE-COMMON-SUPPORT-BRANCHES"
CONTRACT_STAGE = "A4R7-CARE-COMMON-SUPPORT-COLLECTION"


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
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite frozen manifest: {args.output}")

    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    source_sha = sha256_file(args.source_manifest)
    contract_sha = sha256_file(args.contract)
    if source.get("stage_id") != SOURCE_STAGE or source.get("status") != "FROZEN_BEFORE_GATE_BRANCH_OUTCOME":
        raise RuntimeError("A4R6 source selection manifest is not frozen")
    if contract.get("stage_id") != CONTRACT_STAGE:
        raise RuntimeError("wrong A4R7 contract stage")
    reuse = contract.get("selection_reuse", {})
    if source_sha != reuse.get("source_manifest_sha256"):
        raise RuntimeError("A4R6 source selection manifest drifted")
    parent = contract.get("parent_contract", {})
    if source.get("contract_sha256") != parent.get("sha256"):
        raise RuntimeError("A4R6 source selection does not match the parent contract")
    if int(source.get("family_count", -1)) != 180 or int(source.get("planned_branch_count", -1)) != 4320:
        raise RuntimeError("A4R6 source selection scale drifted")

    families = []
    for source_family in source["families"]:
        family = deepcopy(source_family)
        source_id = str(family["snapshot_id"])
        if not source_id.startswith("compact-r6-"):
            raise RuntimeError(f"unexpected A4R6 snapshot id: {source_id}")
        family["source_a5r6_snapshot_id"] = source_id
        family["snapshot_id"] = source_id.replace("compact-r6-", "compact-r7-", 1)
        families.append(family)
    if len({row["snapshot_id"] for row in families}) != len(families):
        raise RuntimeError("A4R7 snapshot ids are not unique")

    allocation = Counter(
        (row["split"], row["sampling_stratum"], row["task"])
        for row in families
    )
    expected_allocation = {
        (row["split"], row["stratum"], row["task"]): int(row["count"])
        for row in source["allocation"]
    }
    if dict(allocation) != expected_allocation:
        raise RuntimeError("A4R7 family allocation drifted")

    value = {
        "format_version": "before-we-act.a5r7-care-common-support-manifest/1",
        "stage_id": TARGET_STAGE,
        "status": "FROZEN_BEFORE_GATE_BRANCH_OUTCOME",
        "resource_only": False,
        "gate_only": True,
        "contract": str(args.contract.resolve()),
        "contract_sha256": contract_sha,
        "checkpoint": source["checkpoint"],
        "checkpoint_sha256": source["checkpoint_sha256"],
        "source_selection_manifest": str(args.source_manifest.resolve()),
        "source_selection_manifest_sha256": source_sha,
        "source_selection_frozen_before_a5r6_outcomes": True,
        "family_count": len(families),
        "families_per_task": source["families_per_task"],
        "planned_branch_count": len(families) * 24,
        "allocation": source["allocation"],
        "families": families,
        "authorized_uses_after_collection_receipt": ["supported-horizon Gate A", "supported-horizon Gate B"],
        "forbidden_uses": ["CARE training", "validation", "calibration", "unsupported-horizon imputation"],
    }
    atomic_json(args.output, value)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_manifest_sha256": source_sha,
                "family_count": len(families),
                "branch_count": len(families) * 24,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
