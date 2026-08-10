#!/usr/bin/env python3
"""Atomic R11 status, PID-bound watchdog heartbeat, and four-way monitor."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping


CANDIDATES = ("A", "B", "C", "D")
GPU_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}
ALLOWED_STATES = {
    "NOT_STARTED",
    "PREPARING",
    "DOWNLOADING",
    "PREFLIGHT",
    "TRAINING",
    "VALIDATING",
    "ACCEPTING",
    "PASSED",
    "FAILED",
    "FAILED_FIT",
    "STOPPED",
    "STALE",
    "UNKNOWN",
}
TERMINAL = {"PASSED", "FAILED", "FAILED_FIT", "STOPPED"}
STAGE_QUEUE = (
    "F0",
    "F1",
    "Discovery",
    "Validation5",
    "Selection",
    "Formal",
    "Validation20",
    "Acceptance",
    "Confirmation50",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def candidate_root(run_root: Path, candidate: str) -> Path:
    return run_root / candidate


def process_start_ticks(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{int(pid)}/stat").read_text().split()[21])
    except (FileNotFoundError, IndexError, PermissionError, ProcessLookupError, ValueError):
        return None


def pid_identity_alive(pid: Any, expected_ticks: Any) -> bool:
    try:
        pid_value, ticks_value = int(pid), int(expected_ticks)
    except (TypeError, ValueError):
        return False
    return pid_value > 0 and process_start_ticks(pid_value) == ticks_value


def update_status(args: argparse.Namespace) -> None:
    if args.state not in ALLOWED_STATES:
        raise ValueError(f"unsupported state {args.state}")
    path = candidate_root(args.run_root, args.candidate) / "status/runtime.json"
    current = read_json(path)
    payload = {
        **current,
        "format_version": "before-we-act.r11.runtime_status/1",
        "candidate": args.candidate,
        "state": args.state,
        "stage": args.stage,
        "program": args.program,
        "detail": args.detail,
        "branch": args.branch,
        "commit": args.commit,
        "upstream_commit": args.upstream_commit,
        "pid": args.pid,
        "pid_start_time_ticks": args.pid_start_time_ticks,
        "child_pid": args.child_pid,
        "child_pid_start_time_ticks": args.child_pid_start_time_ticks,
        "log": args.log,
        "exit_code": args.exit_code,
        "created_at_epoch": current.get("created_at_epoch", time.time()),
        "updated_at_epoch": time.time(),
    }
    atomic_json(path, payload)


def watchdog(args: argparse.Namespace) -> None:
    if not pid_identity_alive(args.pid, args.pid_start_time_ticks):
        raise ProcessLookupError("refusing heartbeat: PID/start-time identity is not alive")
    path = candidate_root(args.run_root, args.candidate) / "status/pipeline_heartbeat.json"
    atomic_json(
        path,
        {
            "format_version": "before-we-act.r11.watchdog_heartbeat/1",
            "candidate": args.candidate,
            "stage": args.stage,
            "pid": args.pid,
            "pid_start_time_ticks": args.pid_start_time_ticks,
            "worker_identity_alive": True,
            "updated_at_epoch": time.time(),
        },
    )


def _last_jsonl(path: Path) -> dict[str, Any]:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return {}
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def latest_progress(root: Path) -> dict[str, Any]:
    paths = list((root / "train").glob("*/progress.jsonl"))
    if not paths:
        return {}
    return _last_jsonl(max(paths, key=lambda path: path.stat().st_mtime_ns))


def latest_checkpoint(root: Path) -> str:
    paths = list((root / "train").glob("*/checkpoints/checkpoint_*.pt"))
    paths.extend((root / "preflight").glob("**/checkpoints/checkpoint_*.pt"))
    return str(max(paths, key=lambda path: path.stat().st_mtime_ns)) if paths else "-"


def _heartbeat(root: Path) -> dict[str, Any]:
    choices = [
        root / "status/worker.json",
        root / "status/pipeline_heartbeat.json",
    ]
    existing = [path for path in choices if path.is_file()]
    if not existing:
        return {}
    return read_json(max(existing, key=lambda path: path.stat().st_mtime_ns))


def _acceptance(root: Path) -> dict[str, Any]:
    for path in (
        root / "acceptance.json",
        root / "validation/formal/acceptance.json",
    ):
        value = read_json(path)
        if value:
            value["_source"] = str(path)
            return value
    return {}


def _validation(root: Path) -> dict[str, Any]:
    for path in (
        root / "validation/formal/summary.json",
        root / "validation/discovery/summary.json",
    ):
        value = read_json(path)
        if value:
            return value
    return {}


def _causal(root: Path) -> dict[str, Any]:
    for path in (
        root / "causal/formal/summary.json",
        root / "causal/selection/summary.json",
        root / "causal/discovery/summary.json",
    ):
        value = read_json(path)
        if value:
            return value
    return {}


def _gpu_rows() -> dict[int, list[str]]:
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
    rows: dict[int, list[str]] = {}
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 6:
                rows[int(fields[0])] = fields[1:]
    return rows


def _tmux_alive(session: str) -> bool:
    try:
        return subprocess.run(
            ["tmux", "has-session", "-t", session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _tail(path: Any, lines: int = 20) -> list[str]:
    if not path:
        return []
    try:
        return Path(path).read_text(errors="replace").splitlines()[-lines:]
    except OSError:
        return []


def _bar(current: int, total: int, width: int = 24) -> str:
    fraction = min(max(current / max(total, 1), 0.0), 1.0)
    filled = int(fraction * width)
    return "[" + "#" * filled + "." * (width - filled) + f"] {fraction:6.2%}"


def _authoritative_state(raw: str, acceptance: Mapping[str, Any]) -> tuple[str, list[str]]:
    alerts = []
    if acceptance:
        checks = acceptance.get("checks", acceptance.get("acceptance", []))
        complete = bool(acceptance.get("complete")) and len(checks) >= 7
        if complete:
            if raw == "FAILED_FIT" and not acceptance.get("passed"):
                return "FAILED_FIT", alerts
            return ("PASSED" if acceptance.get("passed") else "FAILED"), alerts
    if raw in {"PASSED", "FAILED"}:
        alerts.append("TERMINAL_WITHOUT_COMPLETE_ACCEPTANCE")
        return "UNKNOWN", alerts
    return raw, alerts


def _alerts(
    state: str,
    alive: bool,
    heartbeat_age: float | None,
    recent: list[str],
) -> list[str]:
    alerts = []
    if state not in TERMINAL and state != "NOT_STARTED" and not alive:
        alerts.append("PROCESS_IDENTITY_MISSING")
    if state not in TERMINAL and heartbeat_age is not None and heartbeat_age > 75:
        alerts.append("STALE_HEARTBEAT")
    text = "\n".join(recent).lower()
    for pattern, name in (
        ("cuda out of memory", "OOM"),
        ("traceback (most recent call last)", "TRACEBACK"),
        ("nan", "NAN"),
        ("killed", "KILLED"),
    ):
        if pattern in text:
            alerts.append(name)
    return sorted(set(alerts))


def render(run_root: Path, selected: tuple[str, ...]) -> str:
    manifest = read_json(run_root / "run_manifest.json")
    gpus = _gpu_rows()
    current_time = time.time()
    lines = [
        f"BWA R11 four-way monitor | {now_iso()}",
        f"run_root={run_root} base={manifest.get('base', {}).get('commit', '?')}",
    ]
    for candidate in selected:
        root = candidate_root(run_root, candidate)
        runtime = read_json(root / "status/runtime.json")
        worker = read_json(root / "status/worker.json")
        progress = latest_progress(root)
        heartbeat = _heartbeat(root)
        acceptance = _acceptance(root)
        validation = _validation(root)
        causal = _causal(root)
        raw_state = runtime.get("state", "NOT_STARTED")
        state, authority_alerts = _authoritative_state(raw_state, acceptance)
        heartbeat_epoch = heartbeat.get("updated_at_epoch")
        heartbeat_age = (
            current_time - float(heartbeat_epoch) if heartbeat_epoch is not None else None
        )
        alive = pid_identity_alive(
            heartbeat.get("pid"), heartbeat.get("pid_start_time_ticks")
        )
        if (
            state not in TERMINAL
            and state != "NOT_STARTED"
            and heartbeat_age is not None
            and heartbeat_age > 75
        ):
            state = "STALE"
        meta = manifest.get("candidates", {}).get(candidate, {})
        session = meta.get("tmux", "?")
        gpu = gpus.get(GPU_MAP[candidate], ["?", "?", "?", "?", "?"])
        update = int(progress.get("update", worker.get("update", 0)) or 0)
        total = int(progress.get("protocol_updates", 120000) or 120000)
        recent = _tail(runtime.get("log"), 20)
        alerts = authority_alerts + _alerts(state, alive, heartbeat_age, recent)
        losses = {
            name: progress.get(name, worker.get(name, "-"))
            for name in ("loss", "action_loss", "world_loss", "value_loss")
        }
        checks = acceptance.get("checks", acceptance.get("acceptance", []))
        passed_checks = sum(bool(item.get("passed")) for item in checks)
        failed_checks = [item.get("id") for item in checks if not item.get("passed")]
        tasks = validation.get("tasks", {})
        validation_text = (
            f"{validation.get('successes')}/{validation.get('episodes')} "
            + " ".join(
                f"{task}={row.get('successes', '?')}"
                for task, row in tasks.items()
            )
            if validation
            else "-"
        )
        duration = current_time - float(runtime.get("created_at_epoch", current_time))
        lines.extend(
            [
                "",
                (
                    f"{candidate} {meta.get('model', '?')} | {state} "
                    f"stage={runtime.get('stage', '-')} program={runtime.get('program', '-')}"
                ),
                f"  {_bar(update, total)} update={update}/{total} ETA={progress.get('eta_hours', '-')}h",
                (
                    f"  branch={manifest.get('branches', {}).get(candidate, '?')} "
                    f"commit={runtime.get('commit', '?')} upstream={meta.get('upstream_commit', '?')}"
                ),
                (
                    f"  GPU={GPU_MAP[candidate]} util={gpu[0]}% mem={gpu[1]}/{gpu[2]}MiB "
                    f"temp={gpu[3]}C power={gpu[4]}W tmux={session}:{_tmux_alive(session)}"
                ),
                (
                    f"  pid={heartbeat.get('pid', '-')} start_ticks={heartbeat.get('pid_start_time_ticks', '-')} "
                    f"alive={alive} heartbeat_age={heartbeat_age:.1f}s duration={duration/3600:.2f}h"
                    if heartbeat_age is not None
                    else f"  pid=- alive=False heartbeat=missing duration={duration/3600:.2f}h"
                ),
                (
                    f"  loss={losses['loss']} action={losses['action_loss']} "
                    f"world={losses['world_loss']} value={losses['value_loss']} "
                    f"pred_gain={causal.get('macro_prediction_gain', '-')}"
                ),
                f"  validation={validation_text}",
                (
                    f"  checkpoint={worker.get('checkpoint') or latest_checkpoint(root)} "
                    f"best={runtime.get('best_checkpoint', '-')}"
                ),
                f"  log={runtime.get('log', '-')} alerts={sorted(set(alerts)) or ['NONE']}",
                (
                    f"  modes=normal/prediction_off/prediction_shuffled "
                    f"causal={causal.get('status', 'PENDING')} acceptance={passed_checks}/7 "
                    f"failed={failed_checks} source={acceptance.get('_source', '-')}"
                ),
                f"  queue={' -> '.join(STAGE_QUEUE)}",
            ]
        )
        if recent:
            lines.append("  recent:")
            lines.extend(f"    {line[-220:]}" for line in recent[-3:])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--run-root", type=Path, required=True)
    status.add_argument("--candidate", choices=CANDIDATES, required=True)
    status.add_argument("--state", choices=sorted(ALLOWED_STATES), required=True)
    status.add_argument("--stage", required=True)
    status.add_argument("--program", required=True)
    status.add_argument("--detail", default="")
    status.add_argument("--branch", required=True)
    status.add_argument("--commit", required=True)
    status.add_argument("--upstream-commit", required=True)
    status.add_argument("--pid", type=int, required=True)
    status.add_argument("--pid-start-time-ticks", type=int, required=True)
    status.add_argument("--child-pid", type=int, default=0)
    status.add_argument("--child-pid-start-time-ticks", type=int, default=0)
    status.add_argument("--log", required=True)
    status.add_argument("--exit-code", type=int)

    beat = subparsers.add_parser("watchdog")
    beat.add_argument("--run-root", type=Path, required=True)
    beat.add_argument("--candidate", choices=CANDIDATES, required=True)
    beat.add_argument("--stage", required=True)
    beat.add_argument("--pid", type=int, required=True)
    beat.add_argument("--pid-start-time-ticks", type=int, required=True)

    monitor = subparsers.add_parser("monitor")
    monitor.add_argument("--run-root", type=Path, required=True)
    monitor.add_argument("--candidate", choices=("all",) + CANDIDATES, default="all")
    monitor.add_argument("--once", action="store_true")
    monitor.add_argument("--interval", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "status":
        update_status(args)
    elif args.command == "watchdog":
        watchdog(args)
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
