#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import yaml


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(("git", *args), cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()


def normalize(value: str) -> str:
    return value.rstrip("/").removesuffix(".git")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    lock = yaml.safe_load(Path(args.lock).read_text(encoding="utf-8"))
    upstream = Path(args.upstream).resolve()
    resolved = git(upstream, "rev-parse", "HEAD")
    origin = git(upstream, "remote", "get-url", "origin")
    clean = not git(upstream, "status", "--porcelain")
    passed = (
        resolved == lock["upstream_commit_sha"]
        and normalize(origin) == normalize(lock["official_repo"])
        and clean
        and all((upstream / path).is_file() for path in lock["copied_upstream_files"])
    )
    result = {
        "schema_version": 1,
        "round": lock.get("round", "R11"),
        "candidate_id": lock["candidate_id"],
        "official_repo": lock["official_repo"],
        "declared_commit": lock["upstream_commit_sha"],
        "resolved_commit": resolved,
        "origin": origin,
        "clean": clean,
        "passed": passed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
