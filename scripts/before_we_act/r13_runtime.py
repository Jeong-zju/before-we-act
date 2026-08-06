#!/usr/bin/env python3
"""Atomic status, actual heartbeat and unified R13 four-candidate monitor."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


_BASE_PATH = Path(__file__).with_name("r11_runtime.py")
_SPEC = importlib.util.spec_from_file_location("_bwa_r11_runtime", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load shared runtime implementation")
base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base)

CANDIDATES = base.CANDIDATES
GPU_MAP = base.GPU_MAP


def init(args) -> None:
    manifest = args.run_root / "run_manifest.json"
    if manifest.exists():
        current = base.read_json(manifest)
        if current.get("run_id") != args.run_id or current.get("parent_commit") != args.parent_commit:
            raise ValueError("existing R13 run manifest identity differs")
        return
    worktrees, branches, commits = {}, {}, {}
    for item in args.worktree:
        candidate, branch, commit, path = item.split("=", 3)
        if candidate not in CANDIDATES or candidate in worktrees:
            raise ValueError(f"invalid worktree mapping {item}")
        worktrees[candidate], branches[candidate], commits[candidate] = path, branch, commit
    if set(worktrees) != set(CANDIDATES):
        raise ValueError("R13 manifest requires all four immutable worktree identities")
    base.atomic_json(
        manifest,
        {
            "schema_version": 1,
            "round": "R13",
            "run_id": args.run_id,
            "run_root": str(args.run_root.resolve()),
            "created_at": base.now(),
            "parent_commit": args.parent_commit,
            "belief_checkpoint": str(Path(args.belief_checkpoint).resolve()),
            "action_checkpoint": str(Path(args.action_checkpoint).resolve()),
            "world_cache": str(Path(args.cache).resolve()),
            "worktrees": worktrees,
            "branches": branches,
            "commits": commits,
            "gpu_assignment": GPU_MAP,
            "tmux_sessions": {candidate: f"bwa-r13-{candidate}" for candidate in CANDIDATES},
            "shared_data": args.index,
            "shared_hf_cache": "/workspace/.cache/huggingface",
            "heartbeat_seconds": 20,
            "stale_after_seconds": 75,
            "selection_rule": "highest pre-frozen world_screen_score among valid candidates",
            "acceptance_rules": [
                "official source commit and preserved license are hash-pinned",
                "minimal copied component passes numerical upstream parity and patch audit",
                "no full upstream runtime dependency",
                "strictly off-path: planner and rerank remain disabled",
                "future targets are rejected by the model input contract",
                "two-update save/strict-restore and formal 10000 updates complete",
                "frozen W12 proposal and checkpoint hashes remain bit-exact; Gate20 is N/A",
                "no quality threshold; highest frozen world screen score wins among valid candidates",
            ],
        },
    )
    for candidate in CANDIDATES:
        root = base.root_for(args.run_root, candidate)
        root.mkdir(parents=True, exist_ok=True)
        if not (root / "status.json").exists():
            base.atomic_json(
                root / "status.json",
                {
                    "schema_version": 1,
                    "candidate": candidate,
                    "state": "NOT_STARTED",
                    "stage": "not_started",
                    "program": "-",
                    "detail": "candidate has not been launched",
                    "created_at": base.now(),
                    "updated_at": base.now(),
                },
            )


def add_status(parser) -> None:
    base.add_status_arguments(parser)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    init_parser = sub.add_parser("init")
    init_parser.add_argument("--run-root", type=Path, required=True)
    init_parser.add_argument("--run-id", required=True)
    init_parser.add_argument("--parent-commit", required=True)
    init_parser.add_argument("--belief-checkpoint", required=True)
    init_parser.add_argument("--action-checkpoint", required=True)
    init_parser.add_argument("--cache", required=True)
    init_parser.add_argument("--index", required=True)
    init_parser.add_argument("--worktree", action="append", default=[])
    status_parser = sub.add_parser("status")
    add_status(status_parser)
    beat_parser = sub.add_parser("heartbeat")
    beat_parser.add_argument("--run-root", type=Path, required=True)
    beat_parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    beat_parser.add_argument("--pid", type=int)
    beat_parser.add_argument("--child-pid", type=int)
    monitor_parser = sub.add_parser("monitor")
    monitor_parser.add_argument("--run-root", type=Path, required=True)
    monitor_parser.add_argument("--candidate", choices=("all",) + CANDIDATES, default="all")
    monitor_parser.add_argument("--once", action="store_true")
    monitor_parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    if args.command == "init":
        init(args)
    elif args.command == "status":
        base.update_status(args)
    elif args.command == "heartbeat":
        base.heartbeat(args.run_root, args.candidate, args.pid, args.child_pid)
    else:
        selected = CANDIDATES if args.candidate == "all" else (args.candidate,)
        while True:
            if not args.once:
                sys.stdout.write("\033[2J\033[H")
            rendered = base.render(args.run_root, selected)
            rendered = rendered.replace("BWA R11 monitor", "BWA R13 monitor")
            rendered = rendered.replace("representation_screen_score=", "world_screen_score=")
            print(rendered, flush=True)
            if args.once:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
