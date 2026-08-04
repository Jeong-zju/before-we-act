#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    lock = yaml.safe_load(Path(args.lock).read_text(encoding="utf-8"))
    license_path = project / lock["local_license_file"]
    passed = (
        license_path.is_file()
        and sha(license_path) == lock["license_sha256"]
        and lock["code_weight_data_license"]["code"] not in (None, "unknown")
    )
    result = {
        "schema_version": 1,
        "candidate_id": lock["candidate_id"],
        "passed": passed,
        "license_path": str(license_path),
        "license_sha256": sha(license_path) if license_path.is_file() else None,
        "declared": lock["code_weight_data_license"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
