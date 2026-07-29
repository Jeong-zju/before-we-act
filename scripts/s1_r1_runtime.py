#!/usr/bin/env python3
"""Create, update, and monitor one two-candidate S1-R1 run."""

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


FORMAT_VERSION = "wam.robofactory.s1_r1.runtime/1"
CANDIDATES = ("F0", "F1")


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
    initialize.add_argument(
        "--worktree",
        action="append",
        default=[],
        metavar="CANDIDATE=PATH",
    )

    status = commands.add_parser("status")
    status.add_argument("--run-root", type=Path, required=True)
    status.add_argument("--candidate", choices=CANDIDATES, required=True)
    status.add_argument("--phase", required=True)
    status.add_argument("--detail", default="")
    status.add_argument("--gpu-index", type=int)
    status.add_argument("--total-updates", type=int)
    status.add_argument("--exit-code", type=int)

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
            detail=args.detail,
            gpu_index=args.gpu_index,
            total_updates=args.total_updates,
            exit_code=args.exit_code,
        )
    elif args.command == "monitor":
        if args.interval <= 0:
            raise ValueError("--interval must be positive")
        monitor_run(args.run_root, interval=args.interval, once=args.once)
    else:
        raise AssertionError(args.command)
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
        raise FileExistsError(f"run manifest already exists: {manifest}")
    parsed: dict[str, str] = {}
    for item in worktrees:
        candidate, separator, path = item.partition("=")
        if separator != "=" or candidate not in CANDIDATES or not path:
            raise ValueError(f"invalid --worktree value: {item!r}")
        parsed[candidate] = str(Path(path).expanduser().resolve())
    if set(parsed) != set(CANDIDATES):
        raise ValueError("init requires exactly one F0 and one F1 worktree")
    _atomic_json(
        manifest,
        {
            "format_version": FORMAT_VERSION,
            "run_id": run_id,
            "round_id": "s1-r1",
            "tmux_session": session,
            "tmux_mode": "shared_existing_session",
            "tmux_window_prefix": window_prefix,
            "tmux_monitor_window": monitor_window,
            "created_at": _now(),
            "base_repo": str(base_repo.expanduser().resolve()),
            "worktrees": parsed,
            "shared_data": str(
                (base_repo.expanduser().resolve() / "datasets").resolve()
            ),
        },
    )
    for candidate in CANDIDATES:
        (root / "candidates" / candidate.lower()).mkdir(parents=True)


def update_status(
    run_root: Path,
    *,
    candidate: str,
    phase: str,
    detail: str,
    gpu_index: int | None,
    total_updates: int | None,
    exit_code: int | None,
) -> None:
    if not phase or any(character.isspace() for character in phase):
        raise ValueError("phase must be one non-empty token")
    path = (
        run_root.expanduser().resolve()
        / "candidates"
        / candidate.lower()
        / "status.json"
    )
    current = _load_json(path) if path.exists() else {}
    now = _now()
    payload = {
        **current,
        "format_version": FORMAT_VERSION,
        "candidate": candidate,
        "phase": phase,
        "detail": detail,
        "created_at": current.get("created_at", now),
        "updated_at": now,
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
    try:
        while True:
            if not once:
                sys.stdout.write("\033[2J\033[H")
            print(render_monitor(root), flush=True)
            if once:
                return
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped; training windows and tmux session remain active.")


def render_monitor(run_root: Path) -> str:
    manifest_path = run_root / "run_manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.exists() else {}
    lines = [
        (
            f"WAM S1-R1 monitor | run={manifest.get('run_id', run_root.name)} | "
            f"tmux={manifest.get('tmux_session', '?')} | {_now()}"
        ),
        f"artifacts: {run_root}",
        f"shared data: {manifest.get('shared_data', '?')}",
        "",
        (
            "ID  GPU  PHASE       TRAIN                         "
            "VALIDATION                         DETAIL"
        ),
        (
            "--  ---  ----------  ----------------------------  "
            "---------------------------------  ----------------"
        ),
    ]
    for candidate in CANDIDATES:
        value = collect_candidate(run_root, candidate)
        lines.append(
            f"{candidate:<3} {value['gpu']:<4} {value['phase']:<11} "
            f"{value['training']:<29} {value['validation']:<34} "
            f"{value['detail']}"
        )
    lines.extend(("", *_gpu_lines()))
    lines.append(
        "The permanent tmux session remains active. Switch monitor: "
        f"tmux select-window -t "
        f"{manifest.get('tmux_session', '<session>')}:"
        f"{manifest.get('tmux_monitor_window', '<monitor-window>')}"
    )
    return "\n".join(lines)


def collect_candidate(run_root: Path, candidate: str) -> dict[str, str]:
    candidate_root = run_root / "candidates" / candidate.lower()
    status_path = candidate_root / "status.json"
    status = _load_json(status_path) if status_path.exists() else {}
    progress = read_latest_jsonl(candidate_root / "train" / "progress.jsonl")
    startup = read_latest_jsonl(candidate_root / "train" / "stages.jsonl")
    completed = _progress_value(progress)
    total = _integer(status.get("total_updates")) or _progress_total(progress)
    has_optimizer_step = _has_optimizer_step(progress)
    phase = str(status.get("phase", "pending"))
    detail = " ".join(str(status.get("detail", "")).split())
    training = "not started"
    if startup is not None and phase in {"setup", "startup", "training"}:
        stage = " ".join(str(startup.get("stage", "startup")).split())
        stage_detail = " ".join(str(startup.get("detail", "")).split())
        age = _age_text(startup.get("created_at"))
        if not has_optimizer_step:
            phase = "startup"
            training = f"{stage} ({age})"
            detail = f"{stage_detail}; stage alive for {age}"
        elif completed is None:
            training = f"optimizer active ({age})"
    elif completed is None:
        training = "not started"
    if completed is not None and has_optimizer_step:
        phase = "training" if phase == "startup" else phase
    if completed is not None and total and has_optimizer_step:
        percent = min(max(100.0 * completed / total, 0.0), 100.0)
        loss = _number(progress.get("loss")) if progress else None
        suffix = f" loss={loss:.4g}" if loss is not None else ""
        training = f"{completed}/{total} {percent:5.1f}%{suffix}"
    elif completed is not None and has_optimizer_step:
        training = f"step={completed}"

    validation_parts = []
    validation_root = candidate_root / "validation"
    for task_slug, label in (
        ("lift_barrier", "lift"),
        ("long_pipeline_delivery", "lpd"),
    ):
        episodes = sorted(validation_root.rglob(f"{task_slug}/rollout_episodes.jsonl"))
        if not episodes:
            validation_parts.append(f"{label}=0")
            continue
        records = _read_jsonl(episodes[-1])
        successes = sum(bool(record.get("success")) for record in records)
        validation_parts.append(f"{label}={successes}/{len(records)}")
    summaries = sorted(validation_root.rglob("gate_summary.json"))
    if summaries:
        gate = _load_json(summaries[-1])
        validation_parts.append(f"gate={'pass' if gate.get('passed') else 'done'}")
    return {
        "gpu": str(status.get("gpu_index", "-")),
        "phase": phase[:11],
        "training": " ".join(training.split())[:29],
        "validation": " ".join(validation_parts)[:34],
        "detail": detail[:60],
    }


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def _progress_value(value: Mapping[str, Any] | None) -> int | None:
    if value is None:
        return None
    for key in ("update", "global_step", "step", "completed"):
        parsed = _integer(value.get(key))
        if parsed is not None:
            return parsed
    return None


def _progress_total(value: Mapping[str, Any] | None) -> int | None:
    if value is None:
        return None
    for key in ("updates", "total_steps", "total"):
        parsed = _integer(value.get(key))
        if parsed is not None:
            return parsed
    return None


def _has_optimizer_step(value: Mapping[str, Any] | None) -> bool:
    if value is None:
        return False
    if value.get("event") == "optimizer_step":
        return True
    return _integer(value.get("update")) is not None or (
        (_integer(value.get("global_step")) or 0) > 0
    )


def _age_text(value: Any) -> str:
    try:
        started = datetime.fromisoformat(str(value))
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        seconds = max(
            int(
                (
                    datetime.now(timezone.utc) - started.astimezone(timezone.utc)
                ).total_seconds()
            ),
            0,
        )
    except (TypeError, ValueError):
        return "unknown"
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
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ["GPU: " + (" ".join(result.stderr.split()) or "driver query failed")]
    return [
        "GPU index | model | util% | memory MiB",
        *(f"  {line}" for line in result.stdout.splitlines()),
    ]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
