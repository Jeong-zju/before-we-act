#!/usr/bin/env python3
"""Persistent status, heartbeat and special-rule monitor for S3-R6."""

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


FORMAT_VERSION = "wam.robofactory.s3_r6.runtime/1"
CANDIDATES = ("R6L-P0", "R6L-P1", "R6J-P0", "R6J-P1")
S3_TASKS = (
    "lift_barrier",
    "long_pipeline_delivery",
    "take_photo",
    "three_robots_stack_cube",
    "camera_alignment",
)
EPISODES_PER_TASK = 20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--run-root", type=Path, required=True)
    init.add_argument("--run-id", required=True)
    init.add_argument("--session", required=True)
    init.add_argument("--window-prefix", required=True)
    init.add_argument("--monitor-window", required=True)
    init.add_argument("--base-repo", type=Path, required=True)
    init.add_argument("--worktree", action="append", default=[])
    status = commands.add_parser("status")
    status.add_argument("--run-root", type=Path, required=True)
    status.add_argument("--candidate", choices=CANDIDATES, required=True)
    status.add_argument("--phase", required=True)
    status.add_argument("--program", required=True)
    status.add_argument("--detail", default="")
    status.add_argument("--gpu-index", type=int)
    status.add_argument("--total-updates", type=int)
    status.add_argument("--exit-code", type=int)
    shared = commands.add_parser("shared-status")
    shared.add_argument("--run-root", type=Path, required=True)
    shared.add_argument("--phase", required=True)
    shared.add_argument("--program", required=True)
    shared.add_argument("--detail", default="")
    heartbeat = commands.add_parser("heartbeat")
    heartbeat.add_argument("--run-root", type=Path, required=True)
    heartbeat.add_argument("--candidate", choices=CANDIDATES)
    heartbeat.add_argument("--shared", action="store_true")
    monitor = commands.add_parser("monitor")
    monitor.add_argument("--run-root", type=Path, required=True)
    monitor.add_argument("--interval", type=float, default=5.0)
    monitor.add_argument("--once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.run_root.expanduser().resolve()
    if args.command == "init":
        initialize(
            root,
            run_id=args.run_id,
            session=args.session,
            window_prefix=args.window_prefix,
            monitor_window=args.monitor_window,
            base_repo=args.base_repo,
            worktrees=args.worktree,
        )
    elif args.command == "status":
        update_status(
            root,
            candidate=args.candidate,
            phase=args.phase,
            program=args.program,
            detail=args.detail,
            gpu_index=args.gpu_index,
            total_updates=args.total_updates,
            exit_code=args.exit_code,
        )
    elif args.command == "shared-status":
        _atomic_json(
            root / "shared_status.json",
            {
                "format_version": FORMAT_VERSION,
                "phase": args.phase,
                "program": args.program,
                "detail": args.detail,
                "updated_at": _now(),
            },
        )
    elif args.command == "heartbeat":
        if args.shared == (args.candidate is not None):
            raise ValueError("heartbeat requires exactly one of --shared/--candidate")
        path = (
            root / "shared_heartbeat.json"
            if args.shared
            else root / "candidates" / _slug(args.candidate) / "heartbeat.json"
        )
        _atomic_json(path, {"updated_at": _now(), "pid": os.getppid()})
    else:
        monitor(root, interval=args.interval, once=args.once)
    return 0


def initialize(
    root: Path,
    *,
    run_id: str,
    session: str,
    window_prefix: str,
    monitor_window: str,
    base_repo: Path,
    worktrees: Sequence[str],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "run_manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    parsed: dict[str, str] = {}
    for item in worktrees:
        candidate, separator, path = item.partition("=")
        if separator != "=" or candidate not in CANDIDATES or not path:
            raise ValueError(f"invalid --worktree {item!r}")
        parsed[candidate] = str(Path(path).expanduser().resolve())
    if set(parsed) != set(CANDIDATES):
        raise ValueError("init requires all four S3-R6 candidate worktrees")
    base = base_repo.expanduser().resolve()
    _atomic_json(
        manifest,
        {
            "format_version": FORMAT_VERSION,
            "round_id": "s3-r6",
            "run_id": run_id,
            "created_at": _now(),
            "tmux_session": session,
            "tmux_mode": "shared_existing_session",
            "tmux_window_prefix": window_prefix,
            "tmux_monitor_window": monitor_window,
            "base_repo": str(base),
            "worktrees": parsed,
            "gpu_schedule": [
                {"phase": 1, "gpu0": "R6L-P0", "gpu1": "R6L-P1"},
                {"phase": 2, "gpu0": "R6J-P0", "gpu1": "R6J-P1"},
            ],
            "shared_data": str(base / "datasets/robofactory_multitask"),
            "shared_artifacts": str(base / "artifacts"),
            "training_scope": (
                "fresh candidate-specific five-task Flow for all four candidates; "
                "P1 then trains adapter/gate"
            ),
        },
    )
    for candidate in CANDIDATES:
        (root / "candidates" / _slug(candidate)).mkdir(parents=True)


def update_status(
    root: Path,
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
    path = root / "candidates" / _slug(candidate) / "status.json"
    current = _maybe_json(path)
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


def monitor(root: Path, *, interval: float, once: bool) -> None:
    if interval <= 0:
        raise ValueError("--interval must be positive")
    root.resolve(strict=True)
    try:
        while True:
            if not once:
                sys.stdout.write("\033[2J\033[H")
            print(render_monitor(root), flush=True)
            if once:
                return
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped; candidate jobs and permanent tmux remain active.")


def render_monitor(root: Path) -> str:
    manifest = _maybe_json(root / "run_manifest.json")
    shared = _maybe_json(root / "shared_status.json")
    shared_heartbeat = (
        "finished"
        if shared.get("phase") == "complete"
        else _heartbeat(_maybe_json(root / "shared_heartbeat.json").get("updated_at"))
    )
    lines = [
        (
            f"WAM S3-R6 monitor | run={manifest.get('run_id', root.name)} | "
            f"tmux={manifest.get('tmux_session', '?')} | {_now()}"
        ),
        f"artifacts: {root}",
        (
            "shared | "
            f"status={shared.get('phase', 'pending')} heartbeat={shared_heartbeat} "
            f"program={shared.get('program', '-')} "
            f"detail={_compact(shared.get('detail', ''), 96)}"
        ),
        (
            "schedule | phase1 GPU0=R6L-P0 GPU1=R6L-P1; "
            "then phase2 GPU0=R6J-P0 GPU1=R6J-P1"
        ),
        "",
        "ID     GPU STATUS      HEARTBEAT  PROGRAM                              PROGRESS",
        "------ --- ----------- ---------- ------------------------------------ -----------------------------------------------",
    ]
    for candidate in CANDIDATES:
        value = collect_candidate(root, candidate)
        lines.append(
            f"{candidate:<7}{value['gpu']:<4}{value['phase']:<12}"
            f"{value['heartbeat']:<11}{value['program']:<37}{value['progress']}"
        )
        if value["detail"]:
            lines.append(f"       detail: {value['detail']}")
    lines.extend(("", *_acceptance_lines(root), "", *_gpu_lines()))
    lines.append(
        "Permanent tmux remains alive; monitor="
        f"{manifest.get('tmux_session', '<session>')}:"
        f"{manifest.get('tmux_monitor_window', '<window>')}"
    )
    return "\n".join(lines)


def collect_candidate(root: Path, candidate: str) -> dict[str, str]:
    candidate_root = root / "candidates" / _slug(candidate)
    status = _maybe_json(candidate_root / "status.json")
    phase = str(status.get("phase", "pending"))
    heartbeat = (
        "finished"
        if phase in {"complete", "failed"}
        else _heartbeat(
            _maybe_json(candidate_root / "heartbeat.json").get("updated_at")
        )
    )
    progress = "not started"
    training = _latest_jsonl(candidate_root / "train" / "progress.jsonl")
    stages = _latest_jsonl(candidate_root / "train" / "stages.jsonl")
    gate = _find_gate(candidate_root)
    if training:
        update = _integer(training.get("update"))
        updates = _integer(training.get("updates"))
        loss = _number(training.get("loss"))
        gate_value = _number(training.get("gate"))
        if update is not None and updates:
            progress = (
                f"train {update}/{updates} {100.0 * update / updates:5.1f}% "
                f"loss={_number_text(loss)} gate={_number_text(gate_value)}"
            )
    elif stages:
        progress = (
            f"startup {stages.get('stage', '?')} "
            f"age={_age_text(stages.get('created_at'))}"
        )
    validation = _rollout_progress(candidate_root)
    if validation:
        progress = validation
    if phase == "failed":
        interrupted = _interrupted_rollout_progress(candidate_root)
        if interrupted:
            progress = interrupted
    if gate:
        macro = _number(gate.get("macro_average_success_rate"))
        tasks = gate.get("task_order")
        task_count = len(tasks) if isinstance(tasks, list) else "?"
        progress = (
            f"Gate complete tasks={task_count}/5 "
            f"macro={_number_text(macro)}"
        )
    detail = _compact(status.get("detail", ""), 116)
    if heartbeat.startswith("STALE"):
        detail = _compact(
            f"{detail} STALE log={candidate_root / 'logs' / 'candidate.log'}",
            116,
        )
    return {
        "gpu": str(status.get("gpu_index", "-")),
        "phase": phase[:11],
        "heartbeat": heartbeat[:10],
        "program": _compact(status.get("program", "-"), 36),
        "progress": _compact(progress, 104),
        "detail": detail,
    }


def _rollout_progress(candidate_root: Path) -> str:
    validation = candidate_root / "validation"
    if not validation.is_dir():
        return ""
    statuses = list(validation.rglob("rollout_status.json"))
    if statuses:
        path = max(statuses, key=lambda value: value.stat().st_mtime_ns)
        status = _maybe_json(path)
        return (
            f"validate task={status.get('task_id', path.parent.name)} "
            f"episode={status.get('episode_current', 0)}/"
            f"{status.get('episodes_total', '?')} "
            f"step={status.get('step', 0)}/{status.get('max_steps', '?')} "
            f"success={status.get('successes', 0)} stage={status.get('stage', '?')}"
        )
    candidates = sorted(validation.rglob("rollout_episodes.jsonl"))
    if not candidates:
        task_dirs = [
            path
            for path in validation.rglob("*")
            if path.is_dir()
            and path.name
            in {
                "lift_barrier",
                "long_pipeline_delivery",
                "take_photo",
                "three_robots_stack_cube",
                "camera_alignment",
            }
        ]
        if not task_dirs:
            return ""
        current = max(task_dirs, key=lambda value: value.stat().st_mtime_ns)
        return f"validate task={current.name} episode=0 success=0 stage=starting"
    path = candidates[-1]
    records = _read_jsonl(path)
    successes = sum(bool(value.get("success")) for value in records)
    return f"validate task={path.parent.name} episode={len(records)} success={successes}"


def _acceptance_lines(root: Path) -> list[str]:
    reports = {
        name: _maybe_json(root / "pairs" / f"{name.lower()}_acceptance.json")
        for name in ("R6L", "R6J")
    }
    lines = [
        "S3 acceptance (five-task macro-average P1>=P0; ties pass; per-task is report-only):"
    ]
    for name, report in reports.items():
        if not report:
            early_stop = _early_stop_bound(root, name)
            if not early_stop:
                lines.append(f"{name}: pending")
                continue
            lines.append(
                f"{name}: EARLY-STOP FAIL retain P0 | "
                f"observed={early_stop['observed_successes']}/"
                f"{early_stop['total_episodes']} "
                f"max={_number_text(early_stop['max_success_rate'])} < "
                f"P0={_number_text(early_stop['p0_success_rate'])}"
            )
            lines.append(
                "  macro-average upper bound (hard gate) "
                f"P1<={_number_text(early_stop['max_success_rate'])} "
                f"P0={_number_text(early_stop['p0_success_rate'])} FAIL; "
                f"remaining={early_stop['remaining_episodes']} episodes"
            )
            for task, row in _mapping_or_empty(early_stop.get("tasks")).items():
                values = _mapping_or_empty(row)
                lines.append(
                    f"  {task:<24} P0={values.get('p0_successes', '?')}/"
                    f"{EPISODES_PER_TASK} P1={values.get('p1_successes', '?')}/"
                    f"{values.get('episodes_completed', '?')} "
                    f"remaining={values.get('remaining_episodes', '?')} report-only"
                )
            continue
        lines.append(
            f"{name}: {'PASS P1' if report.get('passed') else 'FAIL retain P0'} | "
            + " | ".join(
                f"{invariant}={'PASS' if passed else 'FAIL'}"
                for invariant, passed in _mapping_or_empty(
                    report.get("structural_invariants")
                ).items()
            )
        )
        macro = _mapping_or_empty(report.get("macro_average"))
        if macro:
            lines.append(
                "  macro-average (hard gate) "
                f"P0={_number_text(_number(macro.get('p0_success_rate')))} "
                f"P1={_number_text(_number(macro.get('p1_success_rate')))} "
                f"delta={_number_text(_number(macro.get('delta_success_rate')))} "
                f"{'PASS' if macro.get('passed_no_regression') else 'FAIL'}"
            )
        for task, row in _mapping_or_empty(report.get("tasks")).items():
            values = _mapping_or_empty(row)
            lines.append(
                f"  {task:<24} P0={values.get('p0_successes', '?')}/"
                f"{values.get('episodes', '?')} P1={values.get('p1_successes', '?')}/"
                f"{values.get('episodes', '?')} delta={values.get('delta_successes', '?')} "
                "report-only"
            )
    final = _maybe_json(root / "acceptance.json")
    if final:
        lines.append(f"FINAL: {final.get('decision', '?')}")
    elif reports.get("R6L") and _early_stop_bound(root, "R6J"):
        lines.append("FINAL: R6L pass P1; R6J early-stop fail retain P0")
    else:
        lines.append("FINAL: pending both micro-rounds")
    return lines


def _interrupted_rollout_progress(candidate_root: Path) -> str:
    summaries = list(
        (candidate_root / "validation").rglob("rollout_summary.json")
    )
    incomplete = []
    for path in summaries:
        summary = _maybe_json(path)
        if summary.get("completed") is False:
            incomplete.append((path, summary))
    if not incomplete:
        return ""
    path, summary = max(
        incomplete, key=lambda value: value[0].stat().st_mtime_ns
    )
    error = _mapping_or_empty(summary.get("fatal_error"))
    reason = str(error.get("type", "interrupted"))
    return (
        f"early-stop task={path.parent.name} "
        f"episode={summary.get('episodes_completed', '?')}/"
        f"{summary.get('episodes_requested', '?')} "
        f"success={summary.get('successes', '?')} reason={reason}"
    )


def _early_stop_bound(root: Path, micro_round: str) -> dict[str, Any]:
    if micro_round not in {"R6L", "R6J"}:
        return {}
    p0_root = root / "candidates" / _slug(f"{micro_round}-P0")
    p1_root = root / "candidates" / _slug(f"{micro_round}-P1")
    p1_status = _maybe_json(p1_root / "status.json")
    if p1_status.get("phase") != "failed" or p1_status.get("exit_code") != 130:
        return {}
    p0_gate = _find_gate(p0_root)
    if tuple(p0_gate.get("task_order", ())) != S3_TASKS:
        return {}

    p0_successes: dict[str, int] = {}
    for task in S3_TASKS:
        value = _integer(_mapping_or_empty(p0_gate.get(task)).get("successes"))
        if value is None or not 0 <= value <= EPISODES_PER_TASK:
            return {}
        p0_successes[task] = value
    total_episodes = len(S3_TASKS) * EPISODES_PER_TASK
    p0_success_rate = sum(p0_successes.values()) / total_episodes

    tasks: dict[str, dict[str, int]] = {}
    observed_successes = 0
    max_successes = 0
    interrupted = False
    for task in S3_TASKS:
        paths = sorted(
            (p1_root / "validation").rglob(f"{task}/rollout_summary.json")
        )
        if paths:
            summary = _maybe_json(paths[-1])
            requested = _integer(summary.get("episodes_requested"))
            completed = _integer(summary.get("episodes_completed"))
            successes = _integer(summary.get("successes"))
            if (
                requested != EPISODES_PER_TASK
                or completed is None
                or successes is None
                or not 0 <= successes <= completed <= EPISODES_PER_TASK
            ):
                return {}
            error = _mapping_or_empty(summary.get("fatal_error"))
            interrupted = interrupted or (
                summary.get("completed") is False
                and error.get("type") == "KeyboardInterrupt"
            )
        else:
            completed = 0
            successes = 0
        remaining = EPISODES_PER_TASK - completed
        observed_successes += successes
        max_successes += successes + remaining
        tasks[task] = {
            "p0_successes": p0_successes[task],
            "p1_successes": successes,
            "episodes_completed": completed,
            "remaining_episodes": remaining,
        }
    max_success_rate = max_successes / total_episodes
    if not interrupted or max_success_rate >= p0_success_rate:
        return {}
    return {
        "p0_success_rate": p0_success_rate,
        "observed_successes": observed_successes,
        "max_successes": max_successes,
        "max_success_rate": max_success_rate,
        "remaining_episodes": sum(
            value["remaining_episodes"] for value in tasks.values()
        ),
        "total_episodes": total_episodes,
        "tasks": tasks,
    }


def _find_gate(candidate_root: Path) -> Mapping[str, Any]:
    paths = sorted((candidate_root / "validation").rglob("gate_summary.json"))
    return _maybe_json(paths[-1]) if paths else {}


def _heartbeat(value: Any) -> str:
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
            int((datetime.now(timezone.utc) - observed.astimezone(timezone.utc)).total_seconds()),
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
    lines = ["GPU index/name/util/memory:"]
    lines.extend(f"  {line}" for line in gpu.stdout.splitlines() if line.strip())
    process_rows = [line for line in processes.stdout.splitlines() if line.strip()]
    lines.append("GPU processes/PID: " + (" | ".join(process_rows) if process_rows else "none"))
    return lines


def _latest_jsonl(path: Path) -> Mapping[str, Any]:
    records = _read_jsonl(path, tail_only=True)
    return records[-1] if records else {}


def _read_jsonl(path: Path, *, tail_only: bool = False) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    data = path.read_bytes()
    if tail_only:
        data = data[-256 * 1024 :]
    result = []
    for line in data.decode("utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def _maybe_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _number_text(value: Any) -> str:
    number = _number(value)
    return "?" if number is None else f"{number:.5g}"


def _compact(value: Any, width: int) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _slug(candidate: str | None) -> str:
    if candidate not in CANDIDATES:
        raise ValueError(f"invalid candidate {candidate!r}")
    return candidate.lower().replace("-", "_")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
