#!/usr/bin/env python3
"""Atomic status, real heartbeat and unified four-candidate R10 monitor."""
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
GPU_MAP = {"p0": 0, "p1": 1, "p2": 2, "p3": 3}
TERMINAL = {"PASSED", "FAILED", "STOPPED"}
ALLOWED = {
    "NOT_STARTED", "PREPARING", "DOWNLOADING", "TRAINING", "VALIDATING",
    "ACCEPTING", "PASSED", "FAILED", "STOPPED", "STALE", "UNKNOWN",
}


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def root_for(run_root: Path, candidate: str):
    return run_root / "candidates" / candidate


def update_status(args):
    if args.state not in ALLOWED:
        raise ValueError(f"unsupported state {args.state}")
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


def heartbeat(run_root: Path, candidate: str, pid=None, child_pid=None):
    path = root_for(run_root, candidate) / "heartbeat.json"
    current = read_json(path)
    payload = {**current, "schema_version": 1, "candidate": candidate, "updated_at": now()}
    if pid is not None:
        payload["pid"] = pid
    if child_pid is not None:
        payload["child_pid"] = child_pid
    atomic_json(path, payload)
    status_path = root_for(run_root, candidate) / "status.json"
    status = read_json(status_path)
    progress_paths = sorted(
        (root_for(run_root, candidate) / "train").glob("*/progress.jsonl"),
        key=lambda item: item.stat().st_mtime,
    )
    if progress_paths:
        try:
            row = json.loads(progress_paths[-1].read_text(errors="replace").splitlines()[-1])
        except (IndexError, json.JSONDecodeError, OSError):
            row = {}
        if row:
            status.update(
                update=row.get("update", status.get("update")),
                total_updates=row.get("target_updates", status.get("total_updates")),
                loss=row.get("loss", status.get("loss")),
                eta_hours=row.get("eta_hours"),
                updated_at=now(),
            )
            atomic_json(status_path, status)


def init(args):
    manifest = args.run_root / "run_manifest.json"
    if manifest.exists():
        current = read_json(manifest)
        if current.get("run_id") != args.run_id or current.get("parent_commit") != args.parent_commit:
            raise ValueError("existing run manifest identity differs")
        return
    worktrees, branches, commits = {}, {}, {}
    for item in args.worktree:
        candidate, branch, commit, path = item.split("=", 3)
        if candidate not in CANDIDATES or candidate in worktrees:
            raise ValueError(f"invalid worktree mapping {item}")
        worktrees[candidate], branches[candidate], commits[candidate] = path, branch, commit
    if set(worktrees) != set(CANDIDATES):
        raise ValueError("all four worktrees are required")
    atomic_json(
        manifest,
        {
            "schema_version": 1,
            "round": "R10",
            "run_id": args.run_id,
            "run_root": str(args.run_root.resolve()),
            "created_at": now(),
            "parent_commit": args.parent_commit,
            "parent_checkpoint": str(Path(args.parent_checkpoint).resolve()),
            "worktrees": worktrees,
            "branches": branches,
            "commits": commits,
            "gpu_assignment": GPU_MAP,
            "tmux_sessions": {candidate: f"bwa-r10-{candidate}" for candidate in CANDIDATES},
            "shared_data": "/workspace/datasets/robofactory_multitask",
            "shared_hf_cache": "/workspace/.cache/huggingface",
            "heartbeat_seconds": 20,
            "stale_after_seconds": 75,
            "acceptance_rules": [
                "gate-zero exact base/forced/routes/temporal",
                "paired Gate20 macro>B9 and each task delta>=-1/20",
                "Camera+Stack >=+4/40 and other three total nondecline",
                "causal episode-bootstrap 95% lower>0",
                "P95 latency<=1.15x and no privileged input",
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
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            timeout=5,
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
    if not path:
        return []
    try:
        values = Path(path).read_text(errors="replace").splitlines()
        return values[-lines:]
    except OSError:
        return []


def latest_checkpoint(root: Path):
    candidates = list((root / "train").glob("**/checkpoints/checkpoint_*.pt"))
    candidates.extend((root / "preflight/checkpoints").glob("checkpoint_*.pt"))
    if not candidates:
        return "-"
    return str(max(candidates, key=lambda item: item.stat().st_mtime))


def acceptance_result(root: Path):
    """Return the authoritative completed acceptance, or a fail-closed gate result."""
    for relative in (
        "acceptance.json",
        "validation/formal/acceptance.json",
        "validation/screen/acceptance.json",
    ):
        path = root / relative
        payload = read_json(path)
        if payload:
            payload["_monitor_source"] = str(path)
            return payload

    # A hard gate-zero/latency failure intentionally prevents Gate20 from running.
    # Preserve that official decision without pretending the three skipped checks ran.
    for relative in (
        "validation/formal/gate_zero_latency.json",
        "validation/screen/gate_zero_latency.json",
    ):
        path = root / relative
        gate = read_json(path)
        if not gate:
            continue
        exact = bool(gate.get("gate_zero_passed"))
        latency_inputs = bool(gate.get("latency_passed")) and not bool(
            gate.get("privileged_inputs")
        )
        return {
            "passed": False,
            "status": "FAILED" if not gate.get("passed") else "PENDING",
            "acceptance": [
                {"id": "gate_zero_exact", "passed": exact},
                {"id": "latency_and_inputs", "passed": latency_inputs},
            ],
            "not_evaluated": [
                "paired_gate20",
                "camera_stack_and_other_tasks",
                "causal_intervention",
            ],
            "_monitor_source": str(path),
        }
    return {}


def acceptance_summary(acceptance, terminal_state):
    checks = acceptance.get("acceptance", [])
    failed = [item["id"] for item in checks if not item.get("passed")]
    passed_count = sum(bool(item.get("passed")) for item in checks)
    if acceptance.get("passed"):
        status = "PASSED"
    elif acceptance:
        status = acceptance.get("status") or (
            "FAILED" if terminal_state in TERMINAL else "PENDING"
        )
    else:
        status = "FAILED_NO_RESULT" if terminal_state == "FAILED" else "PENDING"
    return {
        "status": status,
        "progress": f"{len(checks)}/5",
        "passed_count": passed_count,
        "failed": failed,
        "not_evaluated": acceptance.get("not_evaluated", []),
        "source": acceptance.get("_monitor_source", "-"),
    }


def runtime_alerts(state, alive, beat_age, stale_after, recent):
    alerts = []
    if state not in TERMINAL and not alive:
        alerts.append("PROCESS_MISSING")
    if state not in TERMINAL and beat_age is not None and beat_age > stale_after:
        alerts.append("NO_HEARTBEAT")
    recent_text = "\n".join(recent).lower()
    patterns = {
        "CUDA out of memory": "OOM",
        "traceback (most recent call last)": "TRACEBACK",
        "nan": "NAN",
        "killed": "KILLED",
    }
    for pattern, alert in patterns.items():
        if pattern.lower() in recent_text:
            alerts.append(alert)
    return sorted(set(alerts))


def render(run_root: Path, selected):
    manifest = read_json(run_root / "run_manifest.json")
    gpus = gpu_rows()
    output = [f"BWA R10 monitor | run={manifest.get('run_id', run_root.name)} | {now()}", f"root={run_root}"]
    current_epoch = time.time()
    for candidate in selected:
        root = root_for(run_root, candidate)
        status = read_json(root / "status.json")
        beat = read_json(root / "heartbeat.json")
        acceptance = acceptance_result(root)
        beat_epoch = parse_time(beat.get("updated_at"))
        beat_age = current_epoch - beat_epoch if beat_epoch else None
        started_epoch = parse_time(status.get("created_at"))
        duration = current_epoch - started_epoch if started_epoch else None
        state = status.get("state", "NOT_STARTED")
        alive = pid_alive(status.get("pid")) or pid_alive(status.get("child_pid"))
        stale_after = manifest.get("stale_after_seconds", 75)
        if state not in TERMINAL and beat_age is not None and beat_age > stale_after:
            state = "STALE"
        gpu = gpus.get(GPU_MAP[candidate], ["?", "?", "?", "?", "?"])
        recent = tail(status.get("log"), lines=20)
        alerts = runtime_alerts(state, alive, beat_age, stale_after, recent)
        acceptance_view = acceptance_summary(acceptance, state)
        progress = "-"
        if status.get("update") is not None:
            progress = f"update={status['update']}/{status.get('total_updates', '?')}"
        elif status.get("step") is not None:
            progress = f"step={status['step']}/{status.get('total_steps', '?')}"
        output.extend(
            [
                "",
                (
                    f"{candidate.upper()} | state={state} stage={status.get('stage', '-')} "
                    f"program={status.get('program', '-')} {progress}"
                ),
                (
                    f"  branch={manifest.get('branches', {}).get(candidate, '?')} "
                    f"commit={manifest.get('commits', {}).get(candidate, '?')} "
                    f"gpu={GPU_MAP[candidate]} tmux={manifest.get('tmux_sessions', {}).get(candidate, '?')}"
                ),
                (
                    f"  pid={status.get('pid', '-')} child={status.get('child_pid', '-')} "
                    f"alive={alive} started={status.get('created_at', '-')} heartbeat={beat.get('updated_at', '-')} "
                    f"age={beat_age:.1f}s duration={duration / 3600:.2f}h"
                    if beat_age is not None and duration is not None else "  heartbeat=missing"
                ),
                (
                    f"  GPU util={gpu[0]}% memory={gpu[1]}/{gpu[2]}MiB temp={gpu[3]}C power={gpu[4]}W"
                ),
                (
                    f"  loss={status.get('loss', '-')} validation={status.get('validation_metric', '-')} "
                    f"eta={status.get('eta_hours', '-')}h epoch={status.get('epoch', 'N/A(update-based)')}"
                ),
                f"  checkpoint={status.get('checkpoint') or latest_checkpoint(root)} best={status.get('best_checkpoint', '-')}",
                f"  log={status.get('log', '-')}",
                f"  detail={status.get('detail', '-')}",
                f"  alerts={alerts or ['NONE']}",
                (
                    f"  acceptance={acceptance_view['status']} "
                    f"progress={acceptance_view['progress']} "
                    f"passed={acceptance_view['passed_count']} "
                    f"reasons={acceptance_view['failed']} "
                    f"not_evaluated={acceptance_view['not_evaluated']} "
                    f"source={acceptance_view['source']}"
                ),
            ]
        )
        if recent:
            output.append("  recent:")
            output.extend(f"    {line[-220:]}" for line in recent[-4:])
    output.append("")
    output.append("acceptance rules:")
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
    init_parser.add_argument("--parent-checkpoint", required=True)
    init_parser.add_argument("--worktree", action="append", default=[])
    status_parser = sub.add_parser("status")
    add_status_arguments(status_parser)
    beat_parser = sub.add_parser("heartbeat")
    beat_parser.add_argument("--run-root", type=Path, required=True)
    beat_parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    beat_parser.add_argument("--pid", type=int)
    beat_parser.add_argument("--child-pid", type=int)
    monitor_parser = sub.add_parser("monitor")
    monitor_parser.add_argument("--run-root", type=Path, required=True)
    monitor_parser.add_argument("--candidate", choices=("all",) + CANDIDATES, default="all")
    monitor_parser.add_argument("--once", action="store_true")
    monitor_parser.add_argument("--interval", type=float, default=5)
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
