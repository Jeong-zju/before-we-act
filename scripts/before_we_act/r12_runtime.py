#!/usr/bin/env python3
"""Atomic status, real heartbeat and unified four-route R12 monitor."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


CANDIDATES = ("p0", "p1", "p2", "p3")
TASKS = ("lift_barrier", "camera_alignment", "three_robots_stack_cube", "long_pipeline_delivery", "take_photo")
GPU_MAP = {candidate: int(candidate[1:]) for candidate in CANDIDATES}
TERMINAL = {"PASSED", "FAILED", "STOPPED"}
ALLOWED = {
    "NOT_STARTED", "PREPARING", "DOWNLOADING", "TRAINING", "VALIDATING",
    "ACCEPTING", "PASSED", "FAILED", "STOPPED", "STALE", "UNKNOWN",
}


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def root_for(run_root: Path, candidate: str):
    return run_root / "candidates" / candidate


def heartbeat(run_root: Path, candidate: str, pid=None, child_pid=None):
    root = root_for(run_root, candidate)
    path = root / "heartbeat.json"
    current = read_json(path)
    payload = {**current, "schema_version": 1, "candidate": candidate, "updated_at": now()}
    if pid is not None:
        payload["pid"] = pid
    if child_pid is not None:
        payload["child_pid"] = child_pid
    atomic_json(path, payload)
    status_path = root / "status.json"
    status = read_json(status_path)
    progress = sorted((root / "train").glob("**/progress.jsonl"), key=lambda item: item.stat().st_mtime)
    if progress:
        try:
            row = json.loads(progress[-1].read_text(errors="replace").splitlines()[-1])
        except (IndexError, json.JSONDecodeError, OSError):
            row = {}
        if row:
            status.update(
                update=row.get("update", status.get("update")),
                total_updates=row.get("target_updates", status.get("total_updates")),
                loss=row.get("loss", status.get("loss")),
                eta_hours=row.get("eta_hours"),
                updates_per_hour=row.get("updates_per_hour"),
                updated_at=now(),
            )
            atomic_json(status_path, status)


def update_status(args):
    if args.state not in ALLOWED:
        raise ValueError(f"unsupported R12 state {args.state}")
    path = root_for(args.run_root, args.candidate) / "status.json"
    current = read_json(path)
    payload = {
        **current,
        "schema_version": 1,
        "candidate": args.candidate,
        "state": args.state,
        "stage": args.stage,
        "program": args.program,
        "detail": args.detail,
        "updated_at": now(),
        "created_at": current.get("created_at", now()),
    }
    for name in (
        "pid", "child_pid", "update", "total_updates", "epoch", "step",
        "total_steps", "loss", "validation_metric", "checkpoint", "best_checkpoint",
        "log", "exit_code", "acceptance_progress",
    ):
        value = getattr(args, name)
        if value is not None:
            payload[name] = value
    atomic_json(path, payload)
    heartbeat(args.run_root, args.candidate, args.pid, args.child_pid)


def init(args):
    manifest = args.run_root / "run_manifest.json"
    if manifest.exists():
        current = read_json(manifest)
        if current.get("run_id") != args.run_id or current.get("parent_commit") != args.parent_commit:
            raise ValueError("existing R12 run manifest identity differs")
        return
    worktrees, branches, commits, components = {}, {}, {}, {}
    for item in args.worktree:
        candidate, branch, commit, path = item.split("=", 3)
        if candidate not in CANDIDATES or candidate in worktrees:
            raise ValueError(f"invalid worktree mapping {item}")
        worktrees[candidate], branches[candidate], commits[candidate] = path, branch, commit
        lock = Path(path) / f"experiments/before_we_act/r12/{candidate}/component_lock.yaml"
        if not lock.is_file():
            raise FileNotFoundError(lock)
        import yaml
        payload = yaml.safe_load(lock.read_text(encoding="utf-8"))
        components[candidate] = {
            "name": payload["component_name"],
            "repo": payload["official_repo"],
            "commit": payload["upstream_commit_sha"],
            "license": payload["code_weight_data_license"]["code"],
            "copied_files": payload["copied_upstream_files"],
            "fallback": payload.get("fallback"),
        }
    if set(worktrees) != set(CANDIDATES):
        raise ValueError("all four R12 worktrees are required")
    atomic_json(
        manifest,
        {
            "schema_version": 1,
            "round": "R12",
            "run_id": args.run_id,
            "run_root": str(args.run_root.resolve()),
            "created_at": now(),
            "parent_commit": args.parent_commit,
            "belief_checkpoint": str(Path(args.belief_checkpoint).resolve()),
            "belief_checkpoint_sha256": args.belief_checkpoint_sha256,
            "normalization_checkpoint": str(Path(args.normalization_checkpoint).resolve()),
            "worktrees": worktrees,
            "branches": branches,
            "commits": commits,
            "components": components,
            "gpu_assignment": GPU_MAP,
            "tmux_sessions": {candidate: f"bwa-r12r2-{candidate}" for candidate in CANDIDATES},
            "shared_data": "/workspace/datasets/robofactory_multitask",
            "shared_hf_cache": "/workspace/.cache/huggingface",
            "shared_action_cache": "/workspace/bwa_runs/shared/r12_dense_causal_history_action_cache_v2.pt",
            "protocol_variant": "causal_lag1_coldstart_dense_v2",
            "formal_updates": {"p0": 120_000, "p1": 60_000, "p2": 120_000, "p3": 120_000},
            "heartbeat_seconds": 20,
            "stale_after_seconds": 75,
            "action_affecting": True,
            "acceptance_rules": [
                "official commit/license/source map and unmodified algorithm parity",
                "minimal component closure; no complete upstream runtime dependency",
                "two-update train/save/strict-restore; normalization/finite/range/mask smoke",
                "dense per-episode causal lag-1 cache with explicit t=0/1/2 cold starts",
                "deterministic clean/zero/noise/lag action-history robustification",
                "runtime contains no Stereo-CoRE/PAIR/ARCA/forced-role implementation or checkpoint",
                "candidate-specific preregistered 60k/120k budget complete; offline control-cycle smoke is finite",
                "mandatory five tasks x exactly 20 paired episodes using frozen W10 seeds",
                "candidate qualifies only when valid and total successes strictly exceed W10 74/100",
            ],
        },
    )
    for candidate in CANDIDATES:
        root_for(args.run_root, candidate).mkdir(parents=True, exist_ok=True)


def parse_time(value):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (AttributeError, ValueError):
        return None


def pid_alive(pid):
    try:
        pid = int(pid)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def gpu_rows():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    rows = {}
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            parts = [value.strip() for value in line.split(",")]
            if len(parts) == 6:
                rows[int(parts[0])] = parts[1:]
    return rows


def tail(path, lines=4):
    try:
        return Path(path).read_text(errors="replace").splitlines()[-lines:] if path else []
    except OSError:
        return []


def latest_checkpoint(root: Path):
    values = list((root / "train").glob("**/checkpoints/checkpoint_*.pt"))
    values.extend((root / "preflight/checkpoints").glob("checkpoint_*.pt"))
    return str(max(values, key=lambda item: item.stat().st_mtime)) if values else "-"


def gate20_progress(root: Path):
    reports, complete, successes, episodes, per_task, p95 = {}, 0, 0, 0, {}, []
    for task in TASKS:
        path = root / "validation/gate20" / f"{task}.json"
        payload = read_json(path)
        reports[task] = payload
        if payload.get("episodes") == 20:
            complete += 1
        task_episodes = int(payload.get("episodes", 0))
        task_successes = int(payload.get("successes", 0))
        episodes += task_episodes
        successes += task_successes
        per_task[task] = f"{task_successes}/{task_episodes}"
        value = payload.get("latency_ms", {}).get("p95")
        if value is not None:
            p95.append(float(value))
    return {"complete_tasks": complete, "episodes": episodes, "successes": successes, "per_task": per_task, "p95": max(p95) if p95 else None}


def runtime_alerts(state, alive, beat_age, stale_after, recent):
    alerts = []
    if state not in TERMINAL and not alive:
        alerts.append("PROCESS_MISSING")
    if state not in TERMINAL and beat_age is not None and beat_age > stale_after:
        alerts.append("NO_HEARTBEAT")
    text = "\n".join(recent).lower()
    for pattern, alert in (("out of memory", "OOM"), ("traceback", "TRACEBACK"), ("nan", "NAN"), ("killed", "KILLED")):
        if pattern in text:
            alerts.append(alert)
    return sorted(set(alerts))


def render(run_root: Path, selected):
    manifest = read_json(run_root / "run_manifest.json")
    gpus, current_epoch = gpu_rows(), time.time()
    output = [f"BWA R12 monitor | run={manifest.get('run_id', run_root.name)} | {now()}", f"root={run_root} action_affecting=true W10=74/100"]
    round_complete = True
    for candidate in selected:
        root = root_for(run_root, candidate)
        status = read_json(root / "status.json")
        beat = read_json(root / "heartbeat.json")
        acceptance = read_json(root / "acceptance.json")
        parity = read_json(root / "receipts/parity.json")
        gate = gate20_progress(root)
        beat_epoch = parse_time(beat.get("updated_at"))
        beat_age = current_epoch - beat_epoch if beat_epoch else None
        created = parse_time(status.get("created_at"))
        ended = parse_time(status.get("updated_at")) if status.get("state") in TERMINAL else current_epoch
        duration = max(0.0, ended - created) if created and ended else None
        state = status.get("state", "NOT_STARTED")
        alive = pid_alive(status.get("pid")) or pid_alive(status.get("child_pid"))
        if state not in TERMINAL and beat_age is not None and beat_age > manifest.get("stale_after_seconds", 75):
            state = "STALE"
        round_complete = round_complete and state in TERMINAL
        gpu = gpus.get(GPU_MAP[candidate], ["?", "?", "?", "?", "?"])
        recent = tail(status.get("log"), 20)
        alerts = runtime_alerts(state, alive, beat_age, manifest.get("stale_after_seconds", 75), recent)
        component = manifest.get("components", {}).get(candidate, {})
        progress = f"update={status.get('update', '-')}/{status.get('total_updates', '-')}"
        output.extend([
            "",
            f"{candidate.upper()} | state={state} stage={status.get('stage', '-')} program={status.get('program', '-')} {progress}",
            f"  branch={manifest.get('branches', {}).get(candidate, '?')} commit={manifest.get('commits', {}).get(candidate, '?')} gpu={GPU_MAP[candidate]} tmux={manifest.get('tmux_sessions', {}).get(candidate, '?')}",
            f"  component={component.get('name', '?')} upstream={component.get('repo', '?')}@{component.get('commit', '?')} license={component.get('license', '?')} fallback={component.get('fallback') or 'none'}",
            f"  copied_files={component.get('copied_files', [])} parity={parity.get('passed', 'PENDING')} core_free=true",
            (
                f"  pid={status.get('pid', '-')} child={status.get('child_pid', '-')} alive={alive} started={status.get('created_at', '-')} duration={duration / 3600:.2f}h heartbeat={beat.get('updated_at', '-')} age={beat_age:.1f}s"
                if duration is not None and beat_age is not None else "  heartbeat=missing"
            ),
            f"  GPU util={gpu[0]}% memory={gpu[1]}/{gpu[2]}MiB temp={gpu[3]}C power={gpu[4]}W",
            f"  epoch={status.get('epoch', 'N/A(update-based)')} step={status.get('step', '-')} loss={status.get('loss', '-')} updates/h={status.get('updates_per_hour', '-')} eta={status.get('eta_hours', '-')}h",
            f"  checkpoint={status.get('checkpoint') or latest_checkpoint(root)} best={status.get('best_checkpoint', '-')}",
            f"  validation_metric={status.get('validation_metric', '-')} entered_validation={status.get('stage') in ('offline_validation','gate20','acceptance','complete')}",
            f"  Gate20 tasks={gate['complete_tasks']}/5 episodes={gate['episodes']}/100 successes={gate['successes']} per_task={gate['per_task']} p95={gate['p95']}",
            f"  entered_acceptance={status.get('stage') in ('acceptance','complete')} acceptance={acceptance.get('status', 'PENDING')} progress={len(acceptance.get('acceptance', []))}/11 failed={[row['id'] for row in acceptance.get('acceptance', []) if not row.get('passed')]}",
            f"  log={status.get('log', '-')} detail={status.get('detail', '-')} alerts={alerts or ['NONE']}",
        ])
        if recent:
            output.append("  recent:")
            output.extend(f"    {line[-220:]}" for line in recent[-4:])
    output.extend(["", f"round_terminal={round_complete}", "acceptance rules:"])
    output.extend(f"  {index}. {rule}" for index, rule in enumerate(manifest.get("acceptance_rules", []), 1))
    return "\n".join(output)


def add_status_arguments(parser):
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument("--detail", default="")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--child-pid", type=int)
    parser.add_argument("--update", type=int)
    parser.add_argument("--total-updates", type=int)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--step", type=int)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--loss", type=float)
    parser.add_argument("--validation-metric", type=float)
    parser.add_argument("--checkpoint")
    parser.add_argument("--best-checkpoint")
    parser.add_argument("--log")
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--acceptance-progress")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    init_parser = sub.add_parser("init")
    init_parser.add_argument("--run-root", type=Path, required=True)
    init_parser.add_argument("--run-id", required=True)
    init_parser.add_argument("--parent-commit", required=True)
    init_parser.add_argument("--belief-checkpoint", required=True)
    init_parser.add_argument("--belief-checkpoint-sha256", required=True)
    init_parser.add_argument("--normalization-checkpoint", required=True)
    init_parser.add_argument("--worktree", action="append", default=[])
    status_parser = sub.add_parser("status"); add_status_arguments(status_parser)
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
        update_status(args)
    elif args.command == "heartbeat":
        heartbeat(args.run_root, args.candidate, args.pid, args.child_pid)
    else:
        selected = CANDIDATES if args.candidate == "all" else (args.candidate,)
        while True:
            if not args.once:
                sys.stdout.write("\033[2J\033[H")
            print(render(args.run_root, selected), flush=True)
            if args.once:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
