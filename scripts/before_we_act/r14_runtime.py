#!/usr/bin/env python3
"""Atomic R14 status, producer heartbeat and unified Gate20 monitor."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time


_BASE_PATH = Path(__file__).with_name("r12_runtime.py")
_SPEC = importlib.util.spec_from_file_location("_bwa_r12_runtime", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load shared runtime implementation")
base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base)
CANDIDATES = base.CANDIDATES
GPU_MAP = base.GPU_MAP


def init(args) -> None:
    manifest = args.run_root / "run_manifest.json"
    worktrees, branches, commits, components = {}, {}, {}, {}
    import yaml
    for item in args.worktree:
        candidate, branch, commit, path = item.split("=", 3)
        if candidate not in CANDIDATES or candidate in worktrees:
            raise ValueError(f"invalid R14 worktree mapping {item}")
        worktrees[candidate], branches[candidate], commits[candidate] = path, branch, commit
        lock = Path(path) / f"experiments/before_we_act/r14/{candidate}/component_lock.yaml"
        payload = yaml.safe_load(lock.read_text(encoding="utf-8"))
        components[candidate] = {
            "name": payload["component_name"],
            "repo": payload["official_repo"],
            "commit": payload["upstream_commit_sha"],
            "license": payload["code_weight_data_license"]["code"],
            "copied_files": payload["copied_upstream_files"],
            "fallback": "bit-exact W12 base on exception/NaN/range/deadline/low utility",
        }
    if set(worktrees) != set(CANDIDATES):
        raise ValueError("R14 manifest requires four immutable worktrees")
    if manifest.exists():
        current = base.read_json(manifest)
        if current.get("run_id") != args.run_id or current.get("commits") != commits:
            raise ValueError("existing R14 run identity differs")
        return
    base.atomic_json(manifest, {
        "schema_version": 1,
        "round": "R14",
        "run_id": args.run_id,
        "run_root": str(args.run_root.resolve()),
        "created_at": base.now(),
        "parent_commit": args.parent_commit,
        "belief_checkpoint": str(Path(args.belief_checkpoint).resolve()),
        "action_checkpoint": str(Path(args.action_checkpoint).resolve()),
        "world_checkpoint": str(Path(args.world_checkpoint).resolve()),
        "worktrees": worktrees,
        "branches": branches,
        "commits": commits,
        "components": components,
        "gpu_assignment": GPU_MAP,
        "tmux_sessions": {candidate: f"bwa-r14-{candidate}" for candidate in CANDIDATES},
        "shared_data": "/workspace/datasets/robofactory_multitask",
        "shared_hf_cache": "/workspace/.cache/huggingface",
        "shared_spatial_cache": "/workspace/bwa_runs/shared/r12r4_native_full_cache_v2",
        "protocol_variant": "W12 protected exact fallback plus R14 Stack decision; frozen same-seed Gate20",
        "formal_updates": {candidate: 0 for candidate in CANDIDATES},
        "heartbeat_seconds": 20,
        "stale_after_seconds": 75,
        "action_affecting": True,
        "baseline": "W12=77/100",
        "selection_rule": "complete 100 episodes; >77; then successes, paired wins, Camera+Stack, worst task, p95, GPU-hours, ID",
        "acceptance_rules": [
            "official source commit and preserved license are hash-pinned",
            "copied upstream decision algorithm passes numerical/control-flow parity",
            "algorithmic lines changed=0 and no full upstream runtime dependency",
            "candidate is classified action-affecting and requires Gate20",
            "synthetic preflight is finite, shape-safe, effective and inside trust region",
            "runtime has no CoRE import/checkpoint and fail-closed returns W12 base",
            "four protected tasks are exact W12 per-seed materializations",
            "Stack runs the R14 planner on exactly 20 frozen paired seeds",
            "all five tasks complete exactly 20 episodes with identical seeds/cadence/ensemble",
            "frozen W12 baseline is exactly 77/100",
            "candidate total successes are strictly greater than 77/100",
        ],
    })
    for candidate in CANDIDATES:
        root = base.root_for(args.run_root, candidate)
        root.mkdir(parents=True, exist_ok=True)
        base.atomic_json(root / "status.json", {
            "schema_version": 1, "candidate": candidate, "state": "NOT_STARTED",
            "stage": "not_started", "program": "-", "detail": "candidate has not been launched",
            "created_at": base.now(), "updated_at": base.now(),
        })


def heartbeat(run_root: Path, candidate: str, pid=None, child_pid=None) -> None:
    base.heartbeat(run_root, candidate, pid, child_pid)
    progress = base.read_json(
        base.root_for(run_root, candidate) / "validation/gate20_progress.json"
    )
    if not progress:
        return
    status_path = base.root_for(run_root, candidate) / "status.json"
    status = base.read_json(status_path)
    status.update(
        epoch=progress.get("episode_index"),
        step=progress.get("step"),
        total_steps=progress.get("max_steps"),
        acceptance_progress=(
            f"Stack episode {progress.get('episode_index', '?')}/"
            f"{progress.get('episodes_total', '?')} step "
            f"{progress.get('step', '?')}/{progress.get('max_steps', '?')}"
        ),
        updated_at=base.now(),
    )
    base.atomic_json(status_path, status)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    init_parser = sub.add_parser("init")
    init_parser.add_argument("--run-root", type=Path, required=True)
    init_parser.add_argument("--run-id", required=True)
    init_parser.add_argument("--parent-commit", required=True)
    init_parser.add_argument("--belief-checkpoint", required=True)
    init_parser.add_argument("--action-checkpoint", required=True)
    init_parser.add_argument("--world-checkpoint", required=True)
    init_parser.add_argument("--worktree", action="append", default=[])
    status_parser = sub.add_parser("status"); base.add_status_arguments(status_parser)
    beat_parser = sub.add_parser("heartbeat")
    beat_parser.add_argument("--run-root", type=Path, required=True)
    beat_parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    beat_parser.add_argument("--pid", type=int); beat_parser.add_argument("--child-pid", type=int)
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
        heartbeat(args.run_root, args.candidate, args.pid, args.child_pid)
    else:
        selected = CANDIDATES if args.candidate == "all" else (args.candidate,)
        while True:
            if not args.once:
                sys.stdout.write("\033[2J\033[H")
            rendered = base.render(args.run_root, selected)
            rendered = rendered.replace("W10=74/100", "W12=77/100")
            rendered = rendered.replace("core_free=true", "core_free=true action_affecting=true")
            rendered = rendered.replace("update=-/-", "update=N/A(no training)")
            print(rendered, flush=True)
            if args.once:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
