#!/usr/bin/env python3
"""Strict lightweight validator for an R10 implementation card."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml


REQUIRED = {
    "schema_version",
    "round",
    "candidate_id",
    "branch",
    "parent",
    "allowed_files",
    "public_symbols",
    "tensor_contracts",
    "losses",
    "config_keys",
    "required_tests",
    "papers",
    "acceptance",
}
PARENT_CHECKPOINT = "061b7a4acea8fa10f146779e7a1206822179920dfe573db536d237df81eb541d"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("card")
    parser.add_argument("--expected-parent", default="")
    args = parser.parse_args()
    path = Path(args.card).resolve()
    raw = path.read_bytes()
    card = yaml.safe_load(raw)
    failures = []
    if not isinstance(card, dict):
        failures.append("card must be a mapping")
        card = {}
    missing = REQUIRED - set(card)
    extra = set(card) - REQUIRED
    if missing:
        failures.append(f"missing keys: {sorted(missing)}")
    if extra:
        failures.append(f"unknown keys: {sorted(extra)}")
    candidate = card.get("candidate_id")
    if card.get("schema_version") != 1 or card.get("round") != "R10":
        failures.append("schema_version=1 and round=R10 are required")
    if candidate not in {"p0", "p1", "p2", "p3"}:
        failures.append("candidate_id must be p0..p3")
    if not re.fullmatch(r"bwa/r10-p[0-3]-[a-z0-9-]+", str(card.get("branch", ""))):
        failures.append("invalid R10 branch")
    parent = card.get("parent") if isinstance(card.get("parent"), dict) else {}
    if set(parent) != {"branch", "commit", "checkpoint_sha256"}:
        failures.append("parent keys must be branch/commit/checkpoint_sha256")
    if parent.get("branch") != "bwa/r9-core-native":
        failures.append("wrong parent branch")
    if not re.fullmatch(r"[0-9a-f]{40}", str(parent.get("commit", ""))):
        failures.append("parent commit must be a full hash")
    if args.expected_parent and parent.get("commit") != args.expected_parent:
        failures.append("parent commit differs from frozen round parent")
    if parent.get("checkpoint_sha256") != PARENT_CHECKPOINT:
        failures.append("wrong S10 parent checkpoint")
    for key in (
        "allowed_files", "public_symbols", "tensor_contracts", "losses",
        "config_keys", "required_tests", "papers", "acceptance",
    ):
        value = card.get(key)
        if not isinstance(value, list) or not value:
            failures.append(f"{key} must be a non-empty list")
    if len(card.get("papers", [])) < 2:
        failures.append("at least two paper mappings are required")
    for paper in card.get("papers", []):
        if not isinstance(paper, dict) or set(paper) != {"title", "url", "mechanism", "non_claim"}:
            failures.append("each paper requires title/url/mechanism/non_claim only")
    if len(card.get("acceptance", [])) != 5:
        failures.append("R10 has exactly five hard acceptance items")
    result = {
        "card": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "candidate_id": candidate,
        "parent_commit": parent.get("commit"),
        "failures": failures,
        "passed": not failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
