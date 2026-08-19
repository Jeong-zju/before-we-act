#!/usr/bin/env python3
"""Freeze A4R10 on the exact outcome-blind A5R8 pilot selection."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


CONTRACT_STAGE = "A4R10-CARE-COMMON-SNAPSHOT-OPTION-PILOT"
SOURCE_STAGE = "A5R8-CARE-CLOSED-LOOP-OPTION-PILOT"
TARGET_STAGE = "A5R9-CARE-COMMON-SNAPSHOT-OPTION-PILOT"


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
    parser.add_argument("--source-selection-manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite frozen A5R9 manifest: {args.output}")

    source = json.loads(args.source_selection_manifest.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    source_sha = sha256_file(args.source_selection_manifest)
    if source.get("stage_id") != SOURCE_STAGE or source.get("status") != "FROZEN_BEFORE_OPTION_OUTCOMES":
        raise RuntimeError("A5R8 selection manifest is not frozen")
    if contract.get("stage_id") != CONTRACT_STAGE or contract.get("status") != "FROZEN_BEFORE_OPTION_OUTCOMES":
        raise RuntimeError("A4R10 contract is not frozen")
    if source_sha != contract["frozen_source"]["selection_manifest_sha256"]:
        raise RuntimeError("A5R8 outcome-blind selection manifest drifted")
    if int(source.get("family_count", -1)) != 30:
        raise RuntimeError("A5R8 selection family count drifted")

    families = []
    for row in source["families"]:
        family = deepcopy(row)
        old_id = str(family["snapshot_id"])
        if not old_id.startswith("option-r9-"):
            raise RuntimeError(f"unexpected A5R8 snapshot id: {old_id}")
        family["source_a5r8_selection_snapshot_id"] = old_id
        family["snapshot_id"] = old_id.replace("option-r9-", "option-r10-", 1)
        families.append(family)
    allocation = Counter(row["task"] for row in families)
    if len(allocation) != 6 or any(value != 5 for value in allocation.values()):
        raise RuntimeError("A5R9 per-task allocation drifted")

    protocol = contract["branch_protocol"]
    value = {
        "format_version": "before-we-act.a5r9-care-common-snapshot-option-pilot-manifest/1",
        "stage_id": TARGET_STAGE,
        "status": "FROZEN_BEFORE_OPTION_OUTCOMES",
        "resource_only": True,
        "contract": str(args.contract.resolve()),
        "contract_sha256": sha256_file(args.contract),
        "source_selection_manifest": str(args.source_selection_manifest.resolve()),
        "source_selection_manifest_sha256": source_sha,
        "selection_reused_without_reranking": True,
        "checkpoint": contract["frozen_source"]["reference_policy_checkpoint"],
        "checkpoint_sha256": contract["frozen_source"]["reference_policy_checkpoint_sha256"],
        "families_per_task": 5,
        "family_count": len(families),
        "new_branches_per_family": int(protocol["new_branches_per_family"]),
        "planned_new_branch_count": int(protocol["new_branch_count"]),
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
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
