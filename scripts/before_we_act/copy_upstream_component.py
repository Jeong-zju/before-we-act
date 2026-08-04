#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil

import yaml


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()
    lock_path = Path(args.lock)
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    upstream = Path(args.upstream).resolve()
    project = Path(args.project_root).resolve()
    destination = (project / lock["local_destination"]).resolve()
    if project not in destination.parents:
        raise ValueError("component destination escapes the project")
    files = list(lock["copied_upstream_files"])
    if not files or len(files) != len(set(files)):
        raise ValueError("component file allowlist is empty or duplicated")
    destination.mkdir(parents=True, exist_ok=True)
    source_map = {
        "schema_version": 1,
        "candidate_id": lock["candidate_id"],
        "official_repo": lock["official_repo"],
        "upstream_commit_sha": lock["upstream_commit_sha"],
        "license": lock["code_weight_data_license"],
        "files": [],
    }
    for relative in files:
        source = (upstream / relative).resolve()
        if upstream not in source.parents or not source.is_file():
            raise ValueError(f"missing allowlisted upstream file: {relative}")
        target = destination / relative
        if target.exists():
            raise ValueError(f"refusing to overwrite copied component file: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_map["files"].append(
            {
                "upstream_path": relative,
                "local_path": str(target.relative_to(project)),
                "sha256_before_adaptation": digest(source),
                "sha256_after_copy": digest(target),
            }
        )
    map_path = project / lock["source_map"]
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text(yaml.safe_dump(source_map, sort_keys=False), encoding="utf-8")
    print(map_path)


if __name__ == "__main__":
    main()
