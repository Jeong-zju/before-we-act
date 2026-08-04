#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess


def run(*command: str, cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        raise ValueError("upstream commit must be a full lowercase SHA")
    destination = Path(args.destination)
    if destination.exists():
        if not (destination / ".git").exists():
            raise ValueError("existing upstream cache is not a Git checkout")
        origin = run("git", "remote", "get-url", "origin", cwd=destination)
        if origin.rstrip("/").removesuffix(".git") != args.repo.rstrip("/").removesuffix(".git"):
            raise ValueError("upstream cache origin differs")
        run("git", "fetch", "--no-tags", "origin", args.commit, cwd=destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--filter=blob:none", "--no-checkout", args.repo, str(destination))
        run("git", "fetch", "--no-tags", "origin", args.commit, cwd=destination)
    run("git", "checkout", "--detach", args.commit, cwd=destination)
    if run("git", "rev-parse", "HEAD", cwd=destination) != args.commit:
        raise RuntimeError("upstream checkout commit differs")
    if run("git", "status", "--porcelain", cwd=destination):
        raise RuntimeError("upstream checkout is not clean")
    print(json.dumps({"repo": args.repo, "commit": args.commit, "path": str(destination.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
