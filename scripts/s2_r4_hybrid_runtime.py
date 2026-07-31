#!/usr/bin/env python3
"""Initialize, heartbeat, and monitor one zero-training S2-R4 hybrid run."""

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
from typing import Any, Mapping, Sequence


FORMAT_VERSION = "wam.robofactory.s2_r4.hybrid_runtime/1"
HEARTBEAT_TIMEOUT_SECONDS = 75.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init")
    initialize.add_argument("--run-root", type=Path, required=True)
    initialize.add_argument("--run-id", required=True)
    initialize.add_argument("--session", required=True)
    initialize.add_argument("--window-prefix", required=True)
    initialize.add_argument("--monitor-window", required=True)
    initialize.add_argument("--repo", type=Path, required=True)
    initialize.add_argument("--own-source", type=Path, required=True)
    initialize.add_argument("--team-source", type=Path, required=True)
    status = commands.add_parser("status")
    status.add_argument("--run-root", type=Path, required=True)
    status.add_argument("--phase", required=True)
    status.add_argument("--program", required=True)
    status.add_argument("--detail", default="")
    status.add_argument("--exit-code", type=int)
    heartbeat = commands.add_parser("heartbeat")
    heartbeat.add_argument("--run-root", type=Path, required=True)
    monitor = commands.add_parser("monitor")
    monitor.add_argument("--run-root", type=Path, required=True)
    monitor.add_argument("--interval", type=float, default=5.0)
    monitor.add_argument("--once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        initialize_run(
            args.run_root,
            run_id=args.run_id,
            session=args.session,
            window_prefix=args.window_prefix,
            monitor_window=args.monitor_window,
            repo=args.repo,
            own_source=args.own_source,
            team_source=args.team_source,
        )
    elif args.command == "status":
        update_status(
            args.run_root,
            phase=args.phase,
            program=args.program,
            detail=args.detail,
            exit_code=args.exit_code,
        )
    elif args.command == "heartbeat":
        _atomic_json(
            args.run_root.expanduser().resolve() / "heartbeat.json",
            {"updated_at": _now(), "pid": os.getppid()},
        )
    elif args.command == "monitor":
        monitor_run(args.run_root, interval=args.interval, once=args.once)
    return 0


def initialize_run(
    run_root: Path,
    *,
    run_id: str,
    session: str,
    window_prefix: str,
    monitor_window: str,
    repo: Path,
    own_source: Path,
    team_source: Path,
) -> None:
    root = run_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "run_manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    repository = repo.expanduser().resolve(strict=True)
    _atomic_json(
        manifest,
        {
            "format_version": FORMAT_VERSION,
            "round_id": "s2-r4-hybrid",
            "run_id": run_id,
            "created_at": _now(),
            "tmux_session": session,
            "tmux_mode": "shared_existing_session",
            "tmux_window_prefix": window_prefix,
            "tmux_monitor_window": monitor_window,
            "repo": str(repository),
            "shared_data": str(repository / "datasets/robofactory_multitask"),
            "shared_artifacts": str(repository / "artifacts"),
            "sources": {
                "protected_own": str(own_source.expanduser().resolve(strict=True)),
                "team": str(team_source.expanduser().resolve(strict=True)),
            },
            "training_allowed": False,
            "gpu_count": 1,
        },
    )
    update_status(
        root,
        phase="initialized",
        program="s2_r4_hybrid_runtime.py",
        detail="zero-training run initialized",
        exit_code=None,
    )


def update_status(
    run_root: Path,
    *,
    phase: str,
    program: str,
    detail: str,
    exit_code: int | None,
) -> None:
    if not phase or any(character.isspace() for character in phase):
        raise ValueError("phase must be one token")
    if not program:
        raise ValueError("program cannot be empty")
    root = run_root.expanduser().resolve()
    path = root / "status.json"
    current = _maybe_json(path)
    payload = {
        **current,
        "format_version": FORMAT_VERSION,
        "phase": phase,
        "program": program,
        "detail": detail,
        "created_at": current.get("created_at", _now()),
        "updated_at": _now(),
    }
    if exit_code is not None:
        payload["exit_code"] = exit_code
    _atomic_json(path, payload)


def monitor_run(run_root: Path, *, interval: float, once: bool) -> None:
    root = run_root.expanduser().resolve(strict=True)
    if interval <= 0:
        raise ValueError("--interval must be positive")
    try:
        while True:
            if not once:
                sys.stdout.write("\033[2J\033[H")
            print(render_monitor(root), flush=True)
            if once:
                return
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped; evaluation and permanent tmux remain active.")


def render_monitor(run_root: Path) -> str:
    manifest = _maybe_json(run_root / "run_manifest.json")
    status = _maybe_json(run_root / "status.json")
    heartbeat = _heartbeat(_maybe_json(run_root / "heartbeat.json").get("updated_at"))
    phase = str(status.get("phase", "pending"))
    if phase in {"complete", "failed"}:
        heartbeat_text = "finished"
    elif heartbeat["stale"]:
        heartbeat_text = f"STALE age={heartbeat['age']:.0f}s"
    else:
        heartbeat_text = f"alive age={heartbeat['age']:.0f}s"
    progress = _latest_jsonl(run_root / "evaluation_progress.jsonl")
    diagnostic = _maybe_json(run_root / "hybrid_diagnostic.json")
    lines = [
        (
            "WAM S2-R4 HYBRID monitor | "
            f"run={manifest.get('run_id', run_root.name)} | "
            f"tmux={manifest.get('tmux_session', '?')} | {_now()}"
        ),
        f"artifacts: {run_root}",
        "mode: EVALUATE ONLY | training_allowed=false | GPU0 only",
        (
            f"status={phase} heartbeat={heartbeat_text} "
            f"program={status.get('program', '-')}"
        ),
        f"detail={_compact(status.get('detail', ''), 150)}",
        (
            "source own="
            f"{_compact(_mapping_or_empty(manifest, 'sources').get('protected_own', '-'), 80)}"
        ),
        (
            "source team="
            f"{_compact(_mapping_or_empty(manifest, 'sources').get('team', '-'), 80)}"
        ),
        "",
        _progress_line(progress),
        "",
    ]
    if diagnostic:
        lines.extend(_diagnostic_lines(diagnostic))
    else:
        lines.append("diagnostic: pending")
    lines.extend(("", *_gpu_lines()))
    if heartbeat["stale"] and phase not in {"complete", "failed"}:
        lines.extend(
            (
                "",
                "STALE DETAIL:",
                f"  current program: {status.get('program', '-')}",
                f"  last heartbeat age: {heartbeat['age']:.0f}s",
                f"  log: {run_root / 'evaluate.log'}",
                f"  GPU PID: {_gpu_pid_text()}",
            )
        )
    lines.append(
        "Permanent tmux stays alive; monitor window: "
        f"{manifest.get('tmux_session', '<session>')}:"
        f"{manifest.get('tmux_monitor_window', '<window>')}"
    )
    return "\n".join(lines)


def _progress_line(progress: Mapping[str, Any]) -> str:
    if not progress:
        return "progress: waiting for compose/evaluate"
    fraction = 100.0 * float(progress.get("completed_fraction", 0.0))
    return (
        f"progress={progress.get('completed_batches', 0)}/"
        f"{progress.get('total_batches', '?')} ({fraction:.1f}%) "
        f"task={progress.get('task_id', '-')} "
        f"batch={progress.get('batch', '-')}/{progress.get('batches', '-')} "
        f"windows={progress.get('windows', '-')}\n"
        f"  own max_abs_diff={_number(progress.get('own_max_abs_diff'))} | "
        f"peer/shared={_number(progress.get('peer_shared_loss'))} | "
        f"persistence={_number(progress.get('persistence_loss'))} | "
        f"peer-shuffle delta={_number(progress.get('peer_shuffle_delta'))}"
    )


def _diagnostic_lines(payload: Mapping[str, Any]) -> list[str]:
    diagnostic = _mapping_or_empty(payload, "diagnostic")
    lines = [
        "SPECIAL R4 ACCEPTANCE:",
        (
            f"  final={'PASS' if diagnostic.get('passed') else 'FAIL'} "
            f"conclusion={diagnostic.get('conclusion', '-')}"
        ),
        f"  next={diagnostic.get('next_action', '-')}",
    ]
    for task_id, metrics_value in sorted(
        _mapping_or_empty(payload, "per_task").items()
    ):
        if not isinstance(metrics_value, Mapping):
            continue
        protected = _mapping_or_empty(metrics_value, "protected_own")
        peer = _mapping_or_empty(metrics_value, "peer_shared")
        ci = _mapping_or_empty(peer, "shuffle_delta_bootstrap_95")
        lines.append(
            f"  {task_id}: own_exact={protected.get('max_abs_diff') == 0.0} "
            f"maxdiff={_number(protected.get('max_abs_diff'))} "
            f"peer/shared={_number(peer.get('normal_composite_future_loss'))} "
            f"persistence={_number(peer.get('persistence_composite_future_loss'))} "
            f"shuffle={_number(peer.get('shuffle_delta'))} "
            f"ci95_lower={_number(ci.get('lower'))}"
        )
    return lines


def _gpu_lines() -> list[str]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ["GPU: nvidia-smi unavailable"]
    lines = ["GPU:"]
    for row in completed.stdout.splitlines():
        fields = [field.strip() for field in row.split(",")]
        if len(fields) == 5:
            lines.append(
                f"  GPU{fields[0]} {fields[1]} util={fields[2]}% "
                f"memory={fields[3]}/{fields[4]} MiB"
            )
    return lines


def _gpu_pid_text() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return _compact(completed.stdout.strip() or "none", 120)


def _heartbeat(value: object) -> dict[str, float | bool]:
    if not isinstance(value, str):
        return {"age": float("inf"), "stale": True}
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return {"age": float("inf"), "stale": True}
    age = max((datetime.now(timezone.utc) - timestamp).total_seconds(), 0.0)
    return {"age": age, "stale": age > HEARTBEAT_TIMEOUT_SECONDS}


def _latest_jsonl(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    return {}


def _maybe_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_or_empty(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    nested = value.get(key)
    return nested if isinstance(nested, Mapping) else {}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _compact(value: object, width: int) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= width else text[: width - 3] + "..."


def _number(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "-"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
