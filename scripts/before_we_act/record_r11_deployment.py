#!/usr/bin/env python3
"""Create an immutable per-commit remote deployment receipt."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--candidate", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--tmux", required=True)
    parser.add_argument("--launcher-commit", required=True)
    args = parser.parse_args()
    payload = {
        "format_version": "before-we-act.r11.deployment/1",
        "immutable": True,
        "candidate": args.candidate,
        "branch": args.branch,
        "commit": args.commit,
        "base_commit": args.base_commit,
        "upstream_commit": args.upstream_commit,
        "worktree": str(args.worktree.resolve()),
        "gpu": args.gpu,
        "tmux": args.tmux,
        "launcher_commit": args.launcher_commit,
        "created_at_epoch": time.time(),
    }
    root = args.run_root / "deployments"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{args.candidate}-{args.commit}.json"
    if destination.exists():
        current = json.loads(destination.read_text())
        comparable = dict(current)
        comparable["created_at_epoch"] = payload["created_at_epoch"]
        if canonical(comparable) != canonical(payload):
            raise ValueError("existing immutable deployment receipt differs")
        print(destination)
        return
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=root, prefix=f".{destination.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, destination)
    destination.chmod(0o444)
    print(destination)


if __name__ == "__main__":
    main()
