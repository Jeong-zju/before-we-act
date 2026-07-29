#!/usr/bin/env python3
"""Create, heartbeat, monitor, and summarize one two-GPU S2-R3 run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


FORMAT_VERSION = "wam.robofactory.s2_r3.runtime/1"
CANDIDATES = ("W0", "W1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init")
    initialize.add_argument("--run-root", type=Path, required=True)
    initialize.add_argument("--run-id", required=True)
    initialize.add_argument("--session", required=True)
    initialize.add_argument("--window-prefix", required=True)
    initialize.add_argument("--monitor-window", required=True)
    initialize.add_argument("--base-repo", type=Path, required=True)
    initialize.add_argument("--worktree", action="append", default=[])
    status = commands.add_parser("status")
    status.add_argument("--run-root", type=Path, required=True)
    status.add_argument("--candidate", choices=CANDIDATES, required=True)
    status.add_argument("--phase", required=True)
    status.add_argument("--program", required=True)
    status.add_argument("--detail", default="")
    status.add_argument("--gpu-index", type=int)
    status.add_argument("--total-updates", type=int)
    status.add_argument("--exit-code", type=int)
    heartbeat = commands.add_parser("heartbeat")
    heartbeat.add_argument("--run-root", type=Path, required=True)
    heartbeat.add_argument("--candidate", choices=CANDIDATES)
    heartbeat.add_argument("--shared", action="store_true")
    shared_status = commands.add_parser("shared-status")
    shared_status.add_argument("--run-root", type=Path, required=True)
    shared_status.add_argument("--phase", required=True)
    shared_status.add_argument("--program", required=True)
    shared_status.add_argument("--detail", default="")
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
            base_repo=args.base_repo,
            worktrees=args.worktree,
            window_prefix=args.window_prefix,
            monitor_window=args.monitor_window,
        )
    elif args.command == "status":
        update_status(
            args.run_root,
            candidate=args.candidate,
            phase=args.phase,
            program=args.program,
            detail=args.detail,
            gpu_index=args.gpu_index,
            total_updates=args.total_updates,
            exit_code=args.exit_code,
        )
    elif args.command == "heartbeat":
        if args.shared == (args.candidate is not None):
            raise ValueError("heartbeat requires exactly one of --shared/--candidate")
        root = args.run_root.expanduser().resolve()
        path = (
            root / "shared_heartbeat.json"
            if args.shared
            else root / "candidates" / args.candidate.lower() / "heartbeat.json"
        )
        _atomic_json(path, {"updated_at": _now(), "pid": os.getppid()})
    elif args.command == "shared-status":
        _atomic_json(
            args.run_root.expanduser().resolve() / "shared_status.json",
            {
                "format_version": FORMAT_VERSION,
                "phase": args.phase,
                "program": args.program,
                "detail": args.detail,
                "updated_at": _now(),
            },
        )
    elif args.command == "monitor":
        monitor_run(args.run_root, interval=args.interval, once=args.once)
    return 0


def initialize_run(
    run_root: Path,
    *,
    run_id: str,
    session: str,
    base_repo: Path,
    worktrees: Sequence[str],
    window_prefix: str,
    monitor_window: str,
) -> None:
    root = run_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "run_manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    parsed: dict[str, str] = {}
    for item in worktrees:
        candidate, separator, path = item.partition("=")
        if separator != "=" or candidate not in CANDIDATES or not path:
            raise ValueError(f"invalid --worktree: {item!r}")
        parsed[candidate] = str(Path(path).expanduser().resolve())
    if set(parsed) != set(CANDIDATES):
        raise ValueError("init requires W0 and W1 worktrees")
    base = base_repo.expanduser().resolve()
    _atomic_json(
        manifest,
        {
            "format_version": FORMAT_VERSION,
            "round_id": "s2-r3",
            "run_id": run_id,
            "created_at": _now(),
            "tmux_session": session,
            "tmux_mode": "shared_existing_session",
            "tmux_window_prefix": window_prefix,
            "tmux_monitor_window": monitor_window,
            "base_repo": str(base),
            "worktrees": parsed,
            "shared_data": str(base / "datasets/robofactory_multitask"),
            "shared_artifacts": str(base / "artifacts"),
        },
    )
    for candidate in CANDIDATES:
        (root / "candidates" / candidate.lower()).mkdir(parents=True)


def update_status(
    run_root: Path,
    *,
    candidate: str,
    phase: str,
    program: str,
    detail: str,
    gpu_index: int | None,
    total_updates: int | None,
    exit_code: int | None,
) -> None:
    if not phase or any(character.isspace() for character in phase):
        raise ValueError("phase must be one token")
    if not program:
        raise ValueError("program cannot be empty")
    path = (
        run_root.expanduser().resolve()
        / "candidates"
        / candidate.lower()
        / "status.json"
    )
    current = _load_json(path) if path.exists() else {}
    payload = {
        **current,
        "format_version": FORMAT_VERSION,
        "candidate": candidate,
        "phase": phase,
        "program": program,
        "detail": detail,
        "created_at": current.get("created_at", _now()),
        "updated_at": _now(),
    }
    if gpu_index is not None:
        payload["gpu_index"] = gpu_index
    if total_updates is not None:
        payload["total_updates"] = total_updates
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
        print("\nMonitor stopped; training and permanent tmux remain active.")


def render_monitor(run_root: Path) -> str:
    manifest = _maybe_json(run_root / "run_manifest.json")
    shared = _maybe_json(run_root / "shared_status.json")
    shared_hb = _heartbeat_text(
        _maybe_json(run_root / "shared_heartbeat.json").get("updated_at")
    )
    lines = [
        (
            f"WAM S2-R3 monitor | run={manifest.get('run_id', run_root.name)} | "
            f"tmux={manifest.get('tmux_session', '?')} | {_now()}"
        ),
        f"artifacts: {run_root}",
        (
            "shared prepare | "
            f"status={shared.get('phase', 'pending')} heartbeat={shared_hb} "
            f"program={shared.get('program', '-')} "
            f"detail={_compact(shared.get('detail', ''), 90)}"
        ),
        "",
        "ID GPU STATUS      HEARTBEAT  PROGRAM                               TRAIN/VALIDATE",
        "-- --- ----------- ---------- ------------------------------------- ----------------------------------------------",
    ]
    for candidate in CANDIDATES:
        value = collect_candidate(run_root, candidate)
        lines.append(
            f"{candidate:<3}{value['gpu']:<4}{value['phase']:<12}"
            f"{value['heartbeat']:<11}{value['program']:<38}{value['progress']}"
        )
        if value["detail"]:
            lines.append(f"   detail: {value['detail']}")
    acceptance = _maybe_json(run_root / "acceptance.json")
    lines.extend(("", *_acceptance_lines(acceptance), "", *_gpu_lines()))
    lines.append(
        "Permanent tmux stays alive; monitor window: "
        f"{manifest.get('tmux_session', '<session>')}:"
        f"{manifest.get('tmux_monitor_window', '<window>')}"
    )
    return "\n".join(lines)


def collect_candidate(run_root: Path, candidate: str) -> dict[str, str]:
    root = run_root / "candidates" / candidate.lower()
    status = _maybe_json(root / "status.json")
    heartbeat = _heartbeat_text(
        _maybe_json(root / "heartbeat.json").get("updated_at")
    )
    train = read_latest_jsonl(root / "train" / "progress.jsonl")
    stages = read_latest_jsonl(root / "train" / "stages.jsonl")
    validation = read_latest_jsonl(root / "validation" / "progress.jsonl")
    evaluation = _maybe_json(root / "validation" / "evaluation.json")
    phase = str(status.get("phase", "pending"))
    program = str(status.get("program", "-"))
    detail = _compact(status.get("detail", ""), 110)
    progress = "not started"
    if train:
        update = _integer(train.get("update"))
        total = _integer(train.get("updates")) or _integer(
            status.get("total_updates")
        )
        loss = _number(train.get("loss"))
        if update is not None and total:
            progress = (
                f"train {update}/{total} {100.0 * update / total:5.1f}% "
                f"loss={loss:.5g}" if loss is not None
                else f"train {update}/{total}"
            )
    elif stages:
        progress = (
            f"startup {stages.get('stage', '?')} "
            f"({_age_text(stages.get('created_at'))})"
        )
    if validation:
        progress = (
            f"validate task={validation.get('task_id', '?')} "
            f"batch={validation.get('batch', '?')}/{validation.get('batches', '?')} "
            f"delta={_number_text(validation.get('shuffle_delta'))}"
        )
    if evaluation:
        per_task = evaluation.get("per_task", {})
        if isinstance(per_task, Mapping):
            completed = len(per_task)
            positive = sum(
                float(value.get("shuffle_delta_bootstrap_95", {}).get("lower", 0))
                > 0
                for value in per_task.values()
                if isinstance(value, Mapping)
            )
            progress = f"evaluation {completed}/5 tasks; shuffle-CI+={positive}/5"
    return {
        "gpu": str(status.get("gpu_index", "-")),
        "phase": phase[:11],
        "heartbeat": heartbeat[:10],
        "program": _compact(program, 37),
        "progress": _compact(progress, 70),
        "detail": detail,
    }


def _acceptance_lines(value: Mapping[str, Any]) -> list[str]:
    if not value:
        return [
            "S2-R3 special acceptance: pending (wait for both five-task evaluations)"
        ]
    decision = "PASS -> enter R4" if value.get("passed") else "FAIL -> stop before R4"
    lines = [f"S2-R3 special acceptance: {decision}"]
    checks = value.get("checks")
    if isinstance(checks, Mapping):
        lines.append(
            "checks: "
            + " | ".join(
                f"{name}={'PASS' if passed else 'FAIL'}"
                for name, passed in checks.items()
            )
        )
    per_task = value.get("per_task")
    if isinstance(per_task, Mapping):
        lines.append("task                         W0 loss   W1 loss   W0-W1      shuffle Δ   CI95 lower")
        for task_id, row in sorted(per_task.items()):
            if not isinstance(row, Mapping) or not row.get("available"):
                lines.append(f"{task_id:<28} missing")
                continue
            lines.append(
                f"{task_id:<28}"
                f"{float(row['w0_normal_composite_future_loss']):<10.5g}"
                f"{float(row['w1_normal_composite_future_loss']):<10.5g}"
                f"{float(row['w0_minus_w1']):<11.5g}"
                f"{float(row['w1_shuffle_delta']):<12.5g}"
                f"{float(row['w1_shuffle_bootstrap_95_lower']):.5g}"
            )
    return lines


def read_latest_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open("rb") as stream:
        stream.seek(max(path.stat().st_size - 256 * 1024, 0))
        data = stream.read().decode("utf-8", errors="replace")
    for line in reversed(data.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _heartbeat_text(value: Any) -> str:
    age = _age_seconds(value)
    if age is None:
        return "missing"
    return f"{'alive' if age <= 75 else 'STALE'}:{_duration(age)}"


def _age_text(value: Any) -> str:
    age = _age_seconds(value)
    return "unknown" if age is None else _duration(age)


def _age_seconds(value: Any) -> int | None:
    try:
        observed = datetime.fromisoformat(str(value))
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        return max(
            int(
                (
                    datetime.now(timezone.utc)
                    - observed.astimezone(timezone.utc)
                ).total_seconds()
            ),
            0,
        )
    except (TypeError, ValueError):
        return None


def _duration(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _gpu_lines() -> list[str]:
    if shutil.which("nvidia-smi") is None:
        return ["GPU: nvidia-smi unavailable"]
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = ["GPU index | model | util% | memory MiB"]
    lines.extend(f"  {line}" for line in gpu.stdout.splitlines())
    lines.append("GPU processes (uuid,pid,program,memory MiB):")
    lines.extend(
        [f"  {line}" for line in processes.stdout.splitlines()]
        or ["  none"]
    )
    return lines


def _maybe_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and abs(parsed) != float("inf") else None


def _number_text(value: Any) -> str:
    parsed = _number(value)
    return "?" if parsed is None else f"{parsed:.5g}"


def _compact(value: Any, limit: int) -> str:
    return " ".join(str(value).split())[:limit]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
