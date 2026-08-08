#!/usr/bin/env python3
"""Atomic state, paired screen acceptance, and monitor for R15 evolution."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


CANDIDATES = ("p0", "p1", "p2", "p3")
TERMINAL = {"REFERENCE", "PASSED", "FAILED", "STOPPED"}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def candidate_root(run_root: Path, candidate: str) -> Path:
    return run_root / "candidates" / candidate


def register(args) -> None:
    formal = bool(getattr(args, "formal", False))
    protected_successes = int(getattr(args, "protected_successes", 0))
    baseline_total = int(getattr(args, "baseline_total", 0))
    args.run_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.run_root / ".manifest.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = args.run_root / "run_manifest.json"
        manifest = read_json(path)
        if manifest and (
            manifest.get("round") != "R15-Evolution"
            or manifest.get("run_id") != args.run_id
            or manifest.get("split") != args.split
            or manifest.get("seed_file_sha256") != args.seed_file_sha256
        ):
            raise ValueError("existing R15 run identity differs")
        if not manifest:
            manifest = {
                "schema_version": 1,
                "round": "R15-Evolution",
                "run_id": args.run_id,
                "run_root": str(args.run_root.resolve()),
                "created_at": now(),
                "split": args.split,
                "seed_file": str(Path(args.seed_file).resolve()),
                "seed_file_sha256": args.seed_file_sha256,
                "screen_only": not formal,
                "control": "p0",
                "acceptance_rule": (
                    "on the original R14 Gate20 seeds, protected tasks remain exact "
                    "and the candidate total must be strictly greater than W12"
                    if formal
                    else "on identical discovery/validation seeds, candidate successes "
                    "must be strictly greater than W12 control; this is not formal promotion"
                ),
                "protected_successes": protected_successes,
                "baseline_total": baseline_total,
                "heartbeat_seconds": 20,
                "stale_after_seconds": 75,
                "shared_data": "/workspace/datasets/robofactory_multitask",
                "shared_hf_cache": "/workspace/.cache/huggingface",
                "candidates": {},
            }
        session = str(getattr(args, "session", "") or f"bwa-r15s-{args.candidate}")
        if not re.fullmatch(r"bwa-r15s-[A-Za-z0-9_.-]+", session):
            raise ValueError("R15 tmux session identity is invalid")
        identity = {
            "label": args.label,
            "gpu": args.gpu,
            "worktree": str(Path(args.worktree).resolve()),
            "branch": args.branch,
            "commit": args.commit,
            "config": str(Path(args.config).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "session": session,
            "reference": args.reference,
        }
        current = manifest["candidates"].get(args.candidate)
        if current and current != identity:
            raise ValueError(f"existing R15 {args.candidate} identity differs")
        manifest["candidates"][args.candidate] = identity
        atomic_json(path, manifest)
        root = candidate_root(args.run_root, args.candidate)
        root.mkdir(parents=True, exist_ok=True)
        status_path = root / "status.json"
        if not status_path.exists():
            atomic_json(
                status_path,
                {
                    "schema_version": 1,
                    "candidate": args.candidate,
                    "state": "NOT_STARTED",
                    "stage": "not_started",
                    "program": "-",
                    "detail": "registered but not launched",
                    "pid": 0,
                    "child_pid": 0,
                    "created_at": now(),
                    "updated_at": now(),
                },
            )


def update_status(args) -> None:
    path = candidate_root(args.run_root, args.candidate) / "status.json"
    current = read_json(path)
    created = current.get("created_at", now())
    payload = {
        **current,
        "schema_version": 1,
        "candidate": args.candidate,
        "state": args.state,
        "stage": args.stage,
        "program": args.program,
        "detail": args.detail,
        "pid": args.pid,
        "child_pid": args.child_pid,
        "log": args.log,
        "created_at": created,
        "updated_at": now(),
    }
    if args.exit_code is not None:
        payload["exit_code"] = args.exit_code
    atomic_json(path, payload)


def heartbeat(args) -> None:
    root = candidate_root(args.run_root, args.candidate)
    payload = {
        "schema_version": 1,
        "candidate": args.candidate,
        "producer": "run_r15_stack_screen.sh",
        "pid": args.pid,
        "child_pid": args.child_pid,
        "updated_at": now(),
    }
    atomic_json(root / "heartbeat.json", payload)
    status = read_json(root / "status.json")
    if status and status.get("state") not in TERMINAL:
        status.update(
            child_pid=args.child_pid,
            updated_at=payload["updated_at"],
        )
        atomic_json(root / "status.json", status)


def result_rows(path: Path) -> tuple[dict, dict[int, dict]]:
    payload = read_json(path)
    rows = payload.get("rows", [])
    mapped = {
        int(row["seed"]): row
        for row in rows
        if isinstance(row.get("success"), bool)
    }
    return payload, mapped


def logged_rows(path: Path) -> dict[int, dict]:
    """Recover completed episodes from the append-only evaluator log."""

    try:
        lines = path.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return {}
    rows = {}
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(row.get("seed"), int)
            and isinstance(row.get("success"), bool)
            and isinstance(row.get("steps"), int)
        ):
            rows[int(row["seed"])] = row
    return rows


def last_jsonl(path: Path) -> dict:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return {}
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {}


def accept(args) -> int:
    root = candidate_root(args.run_root, args.candidate)
    manifest = read_json(args.run_root / "run_manifest.json")
    split = manifest["split"]
    candidate_payload, candidate_rows = result_rows(
        root / "validation" / f"{split}.json"
    )
    if manifest["candidates"][args.candidate]["reference"]:
        successes = sum(bool(row["success"]) for row in candidate_rows.values())
        result = {
            "schema_version": 1,
            "round": "R15-Evolution",
            "candidate": args.candidate,
            "status": "REFERENCE",
            "screen_only": bool(manifest.get("screen_only", True)),
            "split": split,
            "episodes": len(candidate_rows),
            "successes": successes,
            "formal_total_successes": (
                int(manifest.get("protected_successes", 0)) + successes
                if not manifest.get("screen_only", True)
                else None
            ),
            "rule": "W12 control reference; no promotion decision",
        }
        if len(candidate_rows) != 20:
            raise ValueError("R15 reference screen is incomplete")
        atomic_json(root / "acceptance.json", result)
        return 0

    control_root = candidate_root(args.run_root, manifest["control"])
    control_payload, control_rows = result_rows(
        control_root / "validation" / f"{split}.json"
    )
    if len(candidate_rows) != 20 or set(candidate_rows) != set(control_rows):
        raise ValueError("R15 paired screen is incomplete or seed-mismatched")
    candidate_successes = sum(bool(row["success"]) for row in candidate_rows.values())
    control_successes = sum(bool(row["success"]) for row in control_rows.values())
    wins = sum(
        bool(candidate_rows[seed]["success"]) and not bool(control_rows[seed]["success"])
        for seed in candidate_rows
    )
    losses = sum(
        bool(control_rows[seed]["success"]) and not bool(candidate_rows[seed]["success"])
        for seed in candidate_rows
    )
    formal = not bool(manifest.get("screen_only", True))
    protected = int(manifest.get("protected_successes", 0))
    candidate_total = protected + candidate_successes
    control_total = protected + control_successes
    if formal and control_total != int(manifest.get("baseline_total", -1)):
        raise ValueError("R15 formal W12 baseline total differs")
    passed = candidate_total > control_total if formal else candidate_successes > control_successes
    result = {
        "schema_version": 1,
        "round": "R15-Evolution",
        "candidate": args.candidate,
        "status": "PASSED" if passed else "FAILED",
        "screen_only": not formal,
        "split": split,
        "episodes": 20,
        "candidate_successes": candidate_successes,
        "control_successes": control_successes,
        "delta_successes": candidate_successes - control_successes,
        "protected_successes": protected if formal else None,
        "candidate_total_successes": candidate_total if formal else None,
        "control_total_successes": control_total if formal else None,
        "paired_wins": wins,
        "paired_losses": losses,
        "rule": manifest["acceptance_rule"],
        "candidate_route": candidate_payload.get("route"),
        "control_route": control_payload.get("route"),
    }
    atomic_json(root / "acceptance.json", result)
    return 0 if passed else 10


def process_alive(pid) -> bool:
    try:
        return int(pid) > 0 and Path(f"/proc/{int(pid)}").exists()
    except (TypeError, ValueError):
        return False


def gpu_rows() -> dict[int, str]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    return {int(row.split(",", 1)[0]): row for row in output.splitlines() if row.strip()}


def tail_alerts(path: Path) -> tuple[list[str], list[str]]:
    try:
        lines = path.read_text(errors="replace").splitlines()[-12:]
    except FileNotFoundError:
        return [], []
    joined = "\n".join(lines)
    patterns = {
        "OOM": r"out of memory|CUDA error: out of memory",
        "NaN": r"\bnan\b|non-finite",
        "TRACEBACK": r"Traceback \(most recent call last\)",
        "KILLED": r"\bKilled\b",
    }
    alerts = [name for name, pattern in patterns.items() if re.search(pattern, joined, re.I)]
    return alerts, lines[-4:]


def render(run_root: Path, selected: tuple[str, ...]) -> str:
    manifest = read_json(run_root / "run_manifest.json")
    timestamp = dt.datetime.now(dt.timezone.utc)
    gpus = gpu_rows()
    lines = [
        f"BWA R15-Evolution screen | run={manifest.get('run_id', '?')} | {now()}",
        f"split={manifest.get('split', '?')} screen_only={str(manifest.get('screen_only', True)).lower()} control={manifest.get('control', '?')}",
        f"rule={manifest.get('acceptance_rule', '?')}",
        "",
    ]
    for candidate in selected:
        identity = manifest.get("candidates", {}).get(candidate)
        if not identity:
            lines.extend([f"{candidate.upper()} | state=NOT_STARTED detail=not registered", ""])
            continue
        root = candidate_root(run_root, candidate)
        status = read_json(root / "status.json")
        beat = read_json(root / "heartbeat.json")
        result = read_json(root / "validation" / f"{manifest['split']}.json")
        acceptance = read_json(root / "acceptance.json")
        heartbeat_at = parse_time(beat.get("updated_at"))
        age = (timestamp - heartbeat_at).total_seconds() if heartbeat_at else None
        state = status.get("state", "UNKNOWN")
        if state not in TERMINAL and age is not None and age > manifest.get("stale_after_seconds", 75):
            state = "STALE"
        pid = status.get("pid", 0)
        child = status.get("child_pid", 0)
        if state not in TERMINAL and pid and not process_alive(pid):
            state = "FAILED"
        eval_log = root / "logs" / f"{manifest['split']}.log"
        alerts, tail = tail_alerts(eval_log)
        live_rows = {
            int(row["seed"]): row for row in result.get("rows", [])
        } if result else logged_rows(eval_log)
        completed = len(live_rows)
        successes = sum(bool(row["success"]) for row in live_rows.values())
        stack_stages = {
            "cubeB_placed": sum(
                bool(row.get("terminal_info", {}).get("cubeB_placed"))
                for row in live_rows.values()
            ),
            "A_on_B": sum(
                bool(row.get("terminal_info", {}).get("is_cubeA_on_cubeB"))
                for row in live_rows.values()
            ),
            "C_on_A": sum(
                bool(row.get("terminal_info", {}).get("is_cubeC_on_cubeA"))
                for row in live_rows.values()
            ),
        }
        progress = read_json(root / "validation" / "closed_loop_progress.json")
        training = last_jsonl(root / "train" / "aligned_world" / "progress.jsonl")
        if not training:
            training = last_jsonl(root / "train" / "stack_expert" / "progress.jsonl")
        cache = read_json(root / "cache_heartbeat.json")
        lines.extend(
            [
                f"{candidate.upper()} {identity['label']} | state={state} stage={status.get('stage', '-')} program={status.get('program', '-')}",
                f"  branch={identity['branch']} commit={identity['commit']} gpu={identity['gpu']} tmux={identity['session']}",
                f"  pid={pid} child={child} alive={process_alive(pid)} started={status.get('created_at', '-')} heartbeat={beat.get('updated_at', '-')} age={age if age is not None else '-'}s",
                f"  GPU={gpus.get(identity['gpu'], 'unavailable')} episodes={completed}/20 successes={successes} current_episode={progress.get('episode_index', '-')} step={progress.get('step', '-')}/{progress.get('max_steps', '-')}",
                f"  stack_stages={stack_stages} planner=interventions:{progress.get('interventions', '-')},fallbacks:{progress.get('fallbacks', '-')},timeouts:{progress.get('planner_timeouts', '-')},exceptions:{progress.get('planner_exceptions', '-')}",
                f"  train_update={training.get('update', training.get('fine_tune_update', '-'))} loss={training.get('loss', '-')} eta={training.get('eta_hours', '-')}h cache={cache.get('split', '-')}:{cache.get('rows', '-')}/{cache.get('total_rows', '-')}",
                f"  checkpoint={identity['checkpoint']}",
                f"  result={root / 'validation' / (manifest['split'] + '.json')} acceptance={acceptance.get('status', 'PENDING')}",
                f"  log={eval_log} alerts={alerts or ['NONE']} detail={status.get('detail', '-')}",
            ]
        )
        if acceptance:
            lines.append(
                "  paired="
                f"candidate={acceptance.get('candidate_successes', acceptance.get('successes', '-'))} "
                f"control={acceptance.get('control_successes', '-')} "
                f"wins/losses={acceptance.get('paired_wins', '-')}/{acceptance.get('paired_losses', '-')}"
            )
        for row in tail:
            lines.append(f"    {row[-240:]}")
        lines.append("")
    return "\n".join(lines)


def add_common_status(parser) -> None:
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument("--detail", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--child-pid", type=int, required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--exit-code", type=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    register_parser = sub.add_parser("register")
    register_parser.add_argument("--run-root", type=Path, required=True)
    register_parser.add_argument("--run-id", required=True)
    register_parser.add_argument("--split", required=True)
    register_parser.add_argument("--seed-file", required=True)
    register_parser.add_argument("--seed-file-sha256", required=True)
    register_parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    register_parser.add_argument("--label", required=True)
    register_parser.add_argument("--gpu", type=int, choices=range(4), required=True)
    register_parser.add_argument("--worktree", required=True)
    register_parser.add_argument("--branch", required=True)
    register_parser.add_argument("--commit", required=True)
    register_parser.add_argument("--config", required=True)
    register_parser.add_argument("--checkpoint", required=True)
    register_parser.add_argument("--session", default="")
    register_parser.add_argument("--reference", action="store_true")
    register_parser.add_argument("--formal", action="store_true")
    register_parser.add_argument("--protected-successes", type=int, default=0)
    register_parser.add_argument("--baseline-total", type=int, default=0)
    status_parser = sub.add_parser("status")
    add_common_status(status_parser)
    beat_parser = sub.add_parser("heartbeat")
    beat_parser.add_argument("--run-root", type=Path, required=True)
    beat_parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    beat_parser.add_argument("--pid", type=int, required=True)
    beat_parser.add_argument("--child-pid", type=int, required=True)
    accept_parser = sub.add_parser("accept")
    accept_parser.add_argument("--run-root", type=Path, required=True)
    accept_parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    monitor_parser = sub.add_parser("monitor")
    monitor_parser.add_argument("--run-root", type=Path, required=True)
    monitor_parser.add_argument("--candidate", choices=("all",) + CANDIDATES, default="all")
    monitor_parser.add_argument("--once", action="store_true")
    monitor_parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    if args.command == "register":
        register(args)
    elif args.command == "status":
        update_status(args)
    elif args.command == "heartbeat":
        heartbeat(args)
    elif args.command == "accept":
        raise SystemExit(accept(args))
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
