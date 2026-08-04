#!/usr/bin/env python3
"""Ensure an R10 branch changed only files frozen in its implementation card."""
from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()
    card = yaml.safe_load(Path(args.card).read_text(encoding="utf-8"))
    output = subprocess.run(
        ["git", "diff", "--name-only", f"{args.parent}...{args.head}"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    changed = [line for line in output.splitlines() if line]
    allowed = list(card["allowed_files"])
    unexpected = [
        path for path in changed if not any(fnmatch.fnmatch(path, rule) for rule in allowed)
    ]
    result = {
        "candidate_id": card["candidate_id"],
        "parent": args.parent,
        "head": args.head,
        "allowed_files": allowed,
        "changed_files": changed,
        "unexpected_files": unexpected,
        "passed": not unexpected,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
