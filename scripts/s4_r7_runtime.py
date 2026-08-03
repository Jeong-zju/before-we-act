#!/usr/bin/env python3
"""Persistent status, heartbeat and special-gate monitor for S4-R7.

Candidate and prepare wrappers use the small CLI in this module to publish
atomic status files.  The monitor deliberately derives the R7 structural,
causal and utility gates from their individual metrics; a generic ``passed``
field is never sufficient to report acceptance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


FORMAT_VERSION = "wam.robofactory.s4_r7.runtime/1"
CANDIDATES = ("P0", "P1")
TOTAL_UPDATES = 30_000
EFFECTIVE_TEAM_BATCH = 12
TARGET_AGENT_WINDOWS = 1_152_000
FLOW_UNFREEZE_UPDATE = 6_400
HEARTBEAT_SECONDS = 20
STALE_SECONDS = 75
# The latest completed five-task Gate20 reference on the same server/model
# family provides a task-aware initial wall-time estimate.  These are the sums
# of 20 episode ``duration_seconds`` values per task.  The monitor labels this
# as historical and replaces its scale with this run's live rollout durations.
HISTORICAL_GATE20_TASK_SECONDS = {
    "lift_barrier": 1_400.88,
    "long_pipeline_delivery": 4_097.85,
    "take_photo": 7_248.68,
    "three_robots_stack_cube": 4_115.55,
    "camera_alignment": 4_298.28,
}
HISTORICAL_GATE20_CONDITION_SECONDS = round(
    sum(HISTORICAL_GATE20_TASK_SECONDS.values())
)
BEIJING_TIMEZONE = timezone(timedelta(hours=8), name="UTC+08:00")
MILESTONES = (5_000, 10_000, 15_000, 20_000, 25_000, 30_000)
TERMINAL_PHASES = {"complete", "failed", "stopped"}
TASKS = (
    "lift_barrier",
    "long_pipeline_delivery",
    "take_photo",
    "three_robots_stack_cube",
    "camera_alignment",
)
CORE_CONDITIONS = (
    "normal",
    "legacy_reference",
    "world_evidence_gate_zero",
    "shuffle_all",
)
DIAGNOSTIC_CONDITIONS = (
    "all_world_gates_zero",
    "shuffle_own",
    "shuffle_peer",
    "shuffle_shared",
)
VALIDATION_ORDER = (
    "normal",
    "legacy_reference",
    "world_evidence_gate_zero",
    "shuffle_all",
    *DIAGNOSTIC_CONDITIONS,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="create an immutable run manifest")
    init.add_argument("--run-root", type=Path, required=True)
    init.add_argument("--run-id", required=True)
    init.add_argument("--session", required=True)
    init.add_argument("--window-prefix", required=True)
    init.add_argument("--monitor-window", required=True)
    init.add_argument("--base-repo", type=Path, required=True)
    init.add_argument("--parent-commit", required=True)
    init.add_argument("--worktree", action="append", default=[])

    status = commands.add_parser("status", help="publish candidate status and heartbeat")
    status.add_argument("--run-root", type=Path, required=True)
    status.add_argument("--candidate", choices=CANDIDATES, required=True)
    _add_status_arguments(status, include_gpu=True)

    shared = commands.add_parser("shared-status", help="publish prepare status and heartbeat")
    shared.add_argument("--run-root", type=Path, required=True)
    _add_status_arguments(shared, include_gpu=False)

    heartbeat = commands.add_parser("heartbeat", help="refresh one heartbeat atomically")
    heartbeat.add_argument("--run-root", type=Path, required=True)
    heartbeat.add_argument("--candidate", choices=CANDIDATES)
    heartbeat.add_argument("--shared", action="store_true")
    heartbeat.add_argument("--pid", type=int)
    heartbeat.add_argument("--child-pid", type=int)
    heartbeat.add_argument("--gpu-pid", type=int)

    monitor_parser = commands.add_parser("monitor", help="render the persistent monitor")
    monitor_parser.add_argument("--run-root", type=Path, required=True)
    monitor_parser.add_argument("--interval", type=float, default=300.0)
    monitor_parser.add_argument("--once", action="store_true")
    return parser


def _add_status_arguments(parser: argparse.ArgumentParser, *, include_gpu: bool) -> None:
    parser.add_argument("--phase", required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument("--detail", default="")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--child-pid", type=int)
    if include_gpu:
        parser.add_argument("--gpu-index", type=int, choices=(0, 1))
        parser.add_argument("--gpu-pid", type=int)
    parser.add_argument("--condition")
    parser.add_argument("--task")
    parser.add_argument("--episode", type=int)
    parser.add_argument("--episodes-total", type=int)
    parser.add_argument("--step", type=int)
    parser.add_argument("--steps-total", type=int)
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--effective-batch", type=int)
    parser.add_argument("--update", type=int)
    parser.add_argument("--total-updates", type=int)
    parser.add_argument("--team-windows-seen", type=int)
    parser.add_argument("--agent-windows-seen", type=int)
    parser.add_argument("--milestone")
    parser.add_argument("--flow-unfreeze-state")
    parser.add_argument("--loss", type=float)
    parser.add_argument("--grad-norm", type=float)
    parser.add_argument("--learning-rate")
    parser.add_argument("--preflight")
    parser.add_argument("--exit-code", type=int)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.run_root.expanduser().resolve()
    if args.command == "init":
        initialize_run(
            root,
            run_id=args.run_id,
            session=args.session,
            window_prefix=args.window_prefix,
            monitor_window=args.monitor_window,
            base_repo=args.base_repo,
            parent_commit=args.parent_commit,
            worktrees=args.worktree,
        )
    elif args.command == "status":
        update_status(root, candidate=args.candidate, **_status_kwargs(args, True))
    elif args.command == "shared-status":
        update_shared_status(root, **_status_kwargs(args, False))
    elif args.command == "heartbeat":
        if args.shared == (args.candidate is not None):
            raise ValueError("heartbeat requires exactly one of --shared/--candidate")
        target = (
            root / "shared_heartbeat.json"
            if args.shared
            else _candidate_root(root, args.candidate) / "heartbeat.json"
        )
        _write_heartbeat(
            target,
            pid=args.pid,
            child_pid=args.child_pid,
            gpu_pid=args.gpu_pid,
        )
    else:
        monitor(root, interval=args.interval, once=args.once)
    return 0


def _status_kwargs(args: argparse.Namespace, include_gpu: bool) -> dict[str, Any]:
    names = (
        "phase",
        "program",
        "detail",
        "pid",
        "child_pid",
        "condition",
        "task",
        "episode",
        "episodes_total",
        "step",
        "steps_total",
        "micro_batch",
        "gradient_accumulation",
        "effective_batch",
        "update",
        "total_updates",
        "team_windows_seen",
        "agent_windows_seen",
        "milestone",
        "flow_unfreeze_state",
        "loss",
        "grad_norm",
        "learning_rate",
        "preflight",
        "exit_code",
    )
    if include_gpu:
        names += ("gpu_index", "gpu_pid")
    return {name: getattr(args, name) for name in names}


def initialize_run(
    root: Path,
    *,
    run_id: str,
    session: str,
    window_prefix: str,
    monitor_window: str,
    base_repo: Path,
    parent_commit: str,
    worktrees: Sequence[str],
) -> None:
    if not _safe_identifier(run_id):
        raise ValueError(f"invalid run id {run_id!r}")
    if not _safe_identifier(window_prefix):
        raise ValueError(f"invalid window prefix {window_prefix!r}")
    if monitor_window != f"{window_prefix}-monitor":
        raise ValueError("monitor window must equal <window-prefix>-monitor")
    parsed: dict[str, str] = {}
    for item in worktrees:
        candidate, separator, value = item.partition("=")
        if separator != "=" or candidate not in CANDIDATES or not value:
            raise ValueError(f"invalid --worktree {item!r}")
        parsed[candidate] = str(Path(value).expanduser().resolve())
    if set(parsed) != set(CANDIDATES):
        raise ValueError("init requires exactly P0 and P1 worktrees")

    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "run_manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    base = base_repo.expanduser().resolve()
    _atomic_json(
        manifest,
        {
            "format_version": FORMAT_VERSION,
            "round_id": "s4-r7",
            "run_id": run_id,
            "run_root": str(root),
            "created_at": _now(),
            "parent_commit": parent_commit,
            "tmux_session": session,
            "tmux_mode": "reuse_existing_permanent_session",
            "tmux_window_prefix": window_prefix,
            "tmux_monitor_window": monitor_window,
            "tmux_windows": {
                "prepare": f"{window_prefix}-prepare",
                "P0": f"{window_prefix}-p0",
                "P1": f"{window_prefix}-p1",
                "monitor": monitor_window,
            },
            "base_repo": str(base),
            "worktrees": parsed,
            "gpu_assignment": {"P0": 0, "P1": 1},
            "branches": {
                "P0": "s4/r7-p0-token-preserving-evidence",
                "P1": "s4/r7-p1-world-utility-coupling",
            },
            "candidate_axis": {"name": "utility_coupling_weight", "P0": 0.0, "P1": 0.05},
            "shared_data": str(base / "datasets/robofactory_multitask"),
            "shared_artifacts": str(base / "artifacts"),
            "heartbeat_seconds": HEARTBEAT_SECONDS,
            "stale_after_seconds": STALE_SECONDS,
            "training": {
                "budget_mode": "fast_selection_30k",
                "updates": TOTAL_UPDATES,
                "micro_team_batch": 4,
                "gradient_accumulation": 3,
                "effective_team_batch": EFFECTIVE_TEAM_BATCH,
                "target_agent_windows": TARGET_AGENT_WINDOWS,
                "flow_unfreeze_update": FLOW_UNFREEZE_UPDATE,
                "min_updates_per_second": 0.75,
                "milestones": list(MILESTONES),
            },
            "validation_order": [
                "normal",
                "legacy_reference",
                "world_evidence_gate_zero",
                "shuffle_all",
                "all_world_gates_zero",
                "shuffle_own",
                "shuffle_peer",
                "shuffle_shared",
            ],
        },
    )
    for candidate in CANDIDATES:
        _candidate_root(root, candidate).mkdir(parents=True, exist_ok=True)


def update_status(root: Path, *, candidate: str, **values: Any) -> None:
    candidate_root = _candidate_root(root, candidate)
    _update_status_file(candidate_root / "status.json", candidate=candidate, **values)
    _write_heartbeat(
        candidate_root / "heartbeat.json",
        pid=values.get("pid"),
        child_pid=values.get("child_pid"),
        gpu_pid=values.get("gpu_pid"),
    )


def update_shared_status(root: Path, **values: Any) -> None:
    _update_status_file(root / "shared_status.json", candidate=None, **values)
    _write_heartbeat(
        root / "shared_heartbeat.json",
        pid=values.get("pid"),
        child_pid=values.get("child_pid"),
        gpu_pid=None,
    )


def _update_status_file(path: Path, *, candidate: str | None, **values: Any) -> None:
    phase = str(values.get("phase", ""))
    program = str(values.get("program", ""))
    if not phase or any(character.isspace() for character in phase):
        raise ValueError("phase must be one non-empty token")
    if not program:
        raise ValueError("program must be non-empty")
    current = _maybe_json(path)
    payload: dict[str, Any] = {
        **current,
        "format_version": FORMAT_VERSION,
        "phase": phase,
        "program": program,
        "detail": str(values.get("detail") or ""),
        "created_at": current.get("created_at", _now()),
        "updated_at": _now(),
    }
    if candidate is not None:
        payload["candidate"] = candidate
    for key, value in values.items():
        if key not in {"phase", "program", "detail"} and value is not None:
            payload[key] = value
    payload.setdefault("pid", os.getppid())
    _atomic_json(path, payload)


def _write_heartbeat(
    path: Path,
    *,
    pid: int | None,
    child_pid: int | None,
    gpu_pid: int | None,
) -> None:
    current = _maybe_json(path)
    payload: dict[str, Any] = {
        **current,
        "format_version": FORMAT_VERSION,
        "updated_at": _now(),
        "pid": pid if pid is not None else os.getppid(),
    }
    if child_pid is not None:
        payload["child_pid"] = child_pid
    if gpu_pid is not None:
        payload["gpu_pid"] = gpu_pid
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
        print("\nMonitor stopped; training and the permanent tmux session remain active.")


def render_monitor(root: Path) -> str:
    manifest = _maybe_json(root / "run_manifest.json")
    shared = _maybe_json(root / "shared_status.json")
    data_receipt = _maybe_json(root / "shared_hdf5_verification_receipt.json")
    data_receipt_sha256 = _digest_text(
        root / "shared_hdf5_verification_receipt.json.sha256"
    )
    future_cache = _maybe_json(root / "shared_future_feature_cache.json")
    shared_heartbeat = _terminal_heartbeat(
        str(shared.get("phase", "pending")),
        _maybe_json(root / "shared_heartbeat.json").get("updated_at"),
    )
    lines = [
        (
            f"WAM S4-R7 monitor | run={manifest.get('run_id', root.name)} | "
            f"tmux={manifest.get('tmux_session', '?')} | {_now()}"
        ),
        *_beijing_timeline_lines(root),
        f"artifacts: {root}",
        (
            "shared | "
            f"phase={shared.get('phase', 'pending')} heartbeat={shared_heartbeat} "
            f"program={shared.get('program', '-')} "
            f"pid={shared.get('pid', '-')} child={shared.get('child_pid', '-')} "
            f"detail={_compact(shared.get('detail', ''), 88)}"
        ),
        (
            "shared data | verification=accepted-checkpoint/stat-bound-receipt "
            f"sha256={data_receipt_sha256} "
            f"manifests={len(data_receipt.get('manifests', ())) if data_receipt else 0}/5 "
            f"hdf5={len(data_receipt.get('files', ())) if data_receipt else 0}/750"
        ),
        (
            "future target cache | mode=shared-float32-DINO-PCA "
            f"sha256={str(future_cache.get('features_sha256', 'pending'))[:12]} "
            f"root={future_cache.get('root', 'pending')}"
        ),
        *_future_cache_progress_lines(root),
        (
            "schedule | GPU0=P0 token-preserving/no-WUC; "
            "GPU1=P1 token-preserving/WUC; one candidate per GPU (no DDP)"
        ),
        (
            f"heartbeat contract | producer every {HEARTBEAT_SECONDS}s; "
            f"STALE strictly after {STALE_SECONDS}s"
        ),
        "",
    ]
    gpu_processes = _gpu_processes_by_index()
    for candidate in CANDIDATES:
        lines.extend(_candidate_lines(root, candidate, gpu_processes))
    lines.extend(("", *_special_acceptance_lines(root), "", *_gpu_lines()))
    lines.append(
        "Permanent tmux remains alive; monitor="
        f"{manifest.get('tmux_session', '<session>')}:"
        f"{manifest.get('tmux_monitor_window', '<window>')}"
    )
    return "\n".join(lines)


def _beijing_timeline_lines(root: Path) -> list[str]:
    """Render current Beijing time and conservative, provenance-labelled ETAs."""

    now = datetime.now(timezone.utc)
    candidate_etas: dict[str, datetime] = {}
    candidate_rates: dict[str, float] = {}
    candidate_complete: dict[str, bool] = {}
    for candidate in CANDIDATES:
        candidate_root = _candidate_root(root, candidate)
        status = _maybe_json(candidate_root / "status.json")
        progress = _latest_jsonl(candidate_root / "train" / "progress.jsonl")
        update = _integer(progress.get("update")) or 0
        total = (
            _integer(progress.get("updates"))
            or _integer(progress.get("total_updates"))
            or TOTAL_UPDATES
        )
        rate = _number(progress.get("updates_per_second"))
        complete = update >= total or str(status.get("phase", "")) in {
            "validating",
            "waiting_peer_report",
            "accepting",
            "complete",
        }
        candidate_complete[candidate] = complete
        if complete:
            candidate_etas[candidate] = now
        elif rate is not None and rate > 0 and update > 0:
            candidate_rates[candidate] = rate
            candidate_etas[candidate] = now + timedelta(
                seconds=max(total - update, 0) / rate
            )

    current_text = _beijing_datetime(now)
    rate_parts = []
    for candidate in CANDIDATES:
        if candidate_complete.get(candidate):
            rate_parts.append(f"{candidate}-train=complete")
        elif candidate in candidate_etas:
            rate_parts.append(
                f"{candidate}-train={_beijing_datetime(candidate_etas[candidate])} "
                f"({_number_text(candidate_rates.get(candidate))} update/s)"
            )
        else:
            rate_parts.append(f"{candidate}-train=pending")

    lines = [
        f"Beijing time | current={current_text}",
        "Beijing ETA | " + "; ".join(rate_parts),
    ]
    if len(candidate_etas) != len(CANDIDATES):
        lines.append(
            "Beijing ETA | paired-train=pending normal=pending core4=pending "
            "full-R7=pending"
        )
        return lines

    paired_train = max(candidate_etas.values())
    validation, completed, samples, scales = _validation_eta(
        root,
        now=now,
        train_etas=candidate_etas,
    )
    normal = max(validation[candidate]["normal"] for candidate in CANDIDATES)
    core = max(validation[candidate]["shuffle_all"] for candidate in CANDIDATES)
    full = max(validation[candidate]["shuffle_shared"] for candidate in CANDIDATES)
    normal_text = (
        "complete"
        if all("normal" in completed[candidate] for candidate in CANDIDATES)
        else f"≈{_beijing_datetime(normal)}"
    )
    core_text = (
        "complete"
        if all(
            set(CORE_CONDITIONS).issubset(completed[candidate])
            for candidate in CANDIDATES
        )
        else f"≈{_beijing_datetime(core)}"
    )
    full_text = (
        "complete"
        if all(
            set(VALIDATION_ORDER).issubset(completed[candidate])
            for candidate in CANDIDATES
        )
        else f"≈{_beijing_datetime(full)}"
    )
    lines.extend(
        (
            "Beijing ETA | "
            f"paired-train={_beijing_datetime(paired_train)} "
            f"normal={normal_text} core4={core_text} full-R7={full_text}",
            "ETA basis | training=live cumulative update/s; "
            + (
                "validation=live Gate20 episode durations with historical "
                "task baselines for pending episodes; "
                + " ".join(
                    f"{candidate}:samples={samples[candidate]},"
                    f"scale={scales[candidate]:.3f}"
                    for candidate in CANDIDATES
                )
                if sum(samples.values())
                else "validation=historical S3-R6 five-task Gate20 "
                f"{_duration(HISTORICAL_GATE20_CONDITION_SECONDS)}/condition; "
                "recalibrate from this run after normal starts"
            ),
        )
    )
    return lines


def _beijing_datetime(value: datetime) -> str:
    return value.astimezone(BEIJING_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S UTC+08:00")


def _validation_eta(
    root: Path,
    *,
    now: datetime,
    train_etas: Mapping[str, datetime],
) -> tuple[
    dict[str, dict[str, datetime]],
    dict[str, set[str]],
    dict[str, int],
    dict[str, float],
]:
    schedules: dict[str, dict[str, datetime]] = {}
    completed_by_candidate: dict[str, set[str]] = {}
    sample_counts: dict[str, int] = {}
    scales: dict[str, float] = {}
    for candidate in CANDIDATES:
        candidate_root = _candidate_root(root, candidate)
        completed = set(_gate20_from_disk(candidate_root))
        completed_by_candidate[candidate] = completed
        observed_seconds = 0.0
        historical_seconds = 0.0
        observed_count = 0
        for condition in VALIDATION_ORDER:
            for task in TASKS:
                rows = _rollout_duration_rows(
                    candidate_root
                    / "validation"
                    / "gate20"
                    / condition
                    / task
                    / "rollout_episodes.jsonl"
                )
                observed_seconds += sum(rows.values())
                observed_count += len(rows)
                historical_seconds += len(rows) * (
                    HISTORICAL_GATE20_TASK_SECONDS[task] / 20.0
                )
        # A few early episodes can terminate unusually quickly or slowly.  Blend
        # their ratio with the historical scale until one full task (20 seeds)
        # has been observed, then let this run's measured durations dominate.
        raw_scale = (
            observed_seconds / historical_seconds
            if historical_seconds > 0
            else 1.0
        )
        weight = min(observed_count / 20.0, 1.0)
        scale = min(max(1.0 + weight * (raw_scale - 1.0), 0.25), 4.0)
        sample_counts[candidate] = observed_count
        scales[candidate] = scale

        cursor = max(now, train_etas[candidate])
        schedule: dict[str, datetime] = {}
        for condition in VALIDATION_ORDER:
            if condition not in completed:
                remaining = 0.0
                for task in TASKS:
                    rows = _rollout_duration_rows(
                        candidate_root
                        / "validation"
                        / "gate20"
                        / condition
                        / task
                        / "rollout_episodes.jsonl"
                    )
                    remaining_episodes = max(20 - len(rows), 0)
                    remaining += (
                        remaining_episodes
                        * HISTORICAL_GATE20_TASK_SECONDS[task]
                        / 20.0
                        * scale
                    )
                cursor += timedelta(seconds=remaining)
            schedule[condition] = cursor
        schedules[candidate] = schedule
    return schedules, completed_by_candidate, sample_counts, scales


def _rollout_duration_rows(path: Path) -> dict[int, float]:
    """Return one finite non-negative duration per episode, last record wins."""

    if not path.is_file():
        return {}
    rows: dict[int, float] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, Mapping):
            continue
        episode = _integer(value.get("episode_index"))
        duration = _number(value.get("duration_seconds"))
        if episode is not None and 0 <= episode < 20 and duration is not None and duration >= 0:
            rows[episode] = duration
    return rows


def _future_cache_progress_lines(root: Path) -> list[str]:
    """Expose the latest progress event from each two-GPU cache worker."""

    path = root / "prepare.log"
    if not path.is_file():
        return ["future cache workers | pending"]
    try:
        data = path.read_bytes()[-256 * 1024 :]
    except OSError:
        return ["future cache workers | unavailable"]
    latest: dict[int, Mapping[str, Any]] = {}
    for line in reversed(data.decode("utf-8", errors="replace").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, Mapping) or value.get("event") != "future_cache_progress":
            continue
        worker = _integer(value.get("worker"))
        if worker in {0, 1} and worker not in latest:
            latest[worker] = value
        if len(latest) == 2:
            break
    if not latest:
        return ["future cache workers | starting; shared heartbeat remains authoritative"]
    parts = []
    for worker in (0, 1):
        value = latest.get(worker)
        if value is None:
            parts.append(f"GPU{worker}=starting")
            continue
        age = _age_seconds(value.get("created_at"))
        updated = "?" if age is None else _duration(age)
        parts.append(
            f"GPU{value.get('gpu', worker)}={value.get('episode', '?')}/"
            f"{value.get('episodes', '?')} task={value.get('task_id', '-')} "
            f"source_episode={value.get('episode_index', '?')} updated={updated}-ago"
        )
    return ["future cache workers | " + "; ".join(parts)]


def _digest_text(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return "pending"
    if len(value) == 64 and all(character in "0123456789abcdef" for character in value):
        return value[:12]
    return "INVALID"


def _candidate_lines(
    root: Path,
    candidate: str,
    gpu_processes: Mapping[int, Sequence[int]],
) -> list[str]:
    candidate_root = _candidate_root(root, candidate)
    status = _maybe_json(candidate_root / "status.json")
    heartbeat_data = _maybe_json(candidate_root / "heartbeat.json")
    phase = str(status.get("phase", "pending"))
    heartbeat = _terminal_heartbeat(phase, heartbeat_data.get("updated_at"))
    progress = _merged_progress(candidate_root, status)
    gpu_index = _integer(progress.get("gpu_index"))
    pid = progress.get("pid", heartbeat_data.get("pid", "-"))
    child = progress.get("child_pid", heartbeat_data.get("child_pid"))
    if child is None:
        child = _first_child_pid(_integer(pid))
    gpu_pid = progress.get("gpu_pid", heartbeat_data.get("gpu_pid"))
    # The shell status bridge uses 0 until the CUDA child is discoverable.  Treat
    # that sentinel as missing so a later monitor call can resolve the live PID
    # from nvidia-smi instead of permanently rendering ``gpu_pid=-``.
    if _integer(gpu_pid) in {None, 0} and gpu_index is not None:
        values = gpu_processes.get(gpu_index, ())
        gpu_pid = ",".join(str(value) for value in values) or "-"

    update = _integer(progress.get("update")) or 0
    total = _integer(progress.get("total_updates")) or TOTAL_UPDATES
    micro = progress.get("micro_batch", "?")
    accumulation = progress.get("gradient_accumulation", "?")
    effective = progress.get("effective_batch", EFFECTIVE_TEAM_BATCH)
    team_windows = _integer(progress.get("team_windows_seen"))
    if team_windows is None and update:
        team_windows = update * (_integer(effective) or EFFECTIVE_TEAM_BATCH)
    agent_windows = _integer(progress.get("agent_windows_seen"))
    flow_state = progress.get("flow_unfreeze_state") or (
        "unfrozen" if update >= FLOW_UNFREEZE_UPDATE else f"frozen->{FLOW_UNFREEZE_UPDATE}"
    )
    milestone = progress.get("milestone") or _milestone_text(update)
    condition = progress.get("condition", "-")
    task = progress.get("task", "-")
    episode = _fraction(progress.get("episode"), progress.get("episodes_total"))
    step = _fraction(progress.get("step"), progress.get("steps_total"))
    preflight = _preflight_text(candidate_root, progress.get("preflight"))

    lines = [
        (
            f"{candidate} | GPU={gpu_index if gpu_index is not None else '-'} "
            f"phase={phase} heartbeat={heartbeat} program={progress.get('program', '-')}"
        ),
        (
            f"  process | pid={pid} child_pid={child or '-'} gpu_pid={gpu_pid or '-'} "
            f"detail={_compact(progress.get('detail', ''), 92)}"
        ),
        (
            f"  validation | condition={condition} task={task} "
            f"episode={episode} step={step}"
        ),
        (
            f"  training | micro/accum/effective={micro}/{accumulation}/{effective} "
            f"update={update}/{total} ({_percent(update, total)}) "
            f"team_windows={team_windows if team_windows is not None else '?'} "
            f"agent_windows={agent_windows if agent_windows is not None else '?'}/"
            f"{TARGET_AGENT_WINDOWS} ({_percent(agent_windows, TARGET_AGENT_WINDOWS)})"
        ),
        (
            f"  schedule | milestone={milestone} flow={flow_state} preflight={preflight}"
        ),
        (
            f"  optimizer | loss={_number_text(progress.get('loss'))} "
            f"grad={_number_text(progress.get('grad_norm'))} "
            f"lr={_lr_text(progress.get('learning_rate'))}"
        ),
    ]
    if heartbeat.startswith("STALE"):
        lines.append(
            f"  STALE | last_program={progress.get('program', '-')} "
            f"log={candidate_root / 'logs' / 'candidate.log'} gpu_pid={gpu_pid or '-'}"
        )
    return lines


def _merged_progress(candidate_root: Path, status: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(status)
    for path in (
        candidate_root / "preflight" / "progress.jsonl",
        candidate_root / "train" / "progress.jsonl",
        candidate_root / "validation" / "progress.jsonl",
    ):
        record = _latest_jsonl(path)
        if record:
            for key, value in record.items():
                if value is not None:
                    merged[key] = value
    return merged


def _special_acceptance_lines(root: Path) -> list[str]:
    lines = [
        "R7 special acceptance (derived gates; a generic passed=true is insufficient):"
    ]
    final = _first_json(root / "acceptance.json", root / "pairs" / "acceptance.json")
    pair_exact = _first_json(
        root / "pair_exact.json", root / "pairs" / "pair_exact.json"
    ) or _mapping_or_empty(final.get("pair_exact"))
    # pair_exact embeds complete preflight reports, including expected-false
    # observations such as ``oom=false`` and ``formal_budget_complete=false``.
    # Only the validator's named equality/invariant checks form this gate.
    pair_checks = _mapping_or_empty(pair_exact.get("checks"))
    structural = _boolean_gate(pair_checks or pair_exact)
    lines.append(f"  pair structure | {structural} | artifact={_presence(pair_exact)}")
    final_candidates = _mapping_or_empty(final.get("candidates"))

    for candidate in CANDIDATES:
        candidate_root = _candidate_root(root, candidate)
        accepted = _mapping_or_empty(final_candidates.get(candidate))
        gradient = _first_json(
            candidate_root / "parameter_gradient_audit.json",
            candidate_root / "train" / "parameter_gradient_audit.json",
            candidate_root / "audit" / "parameter_gradient_audit.json",
        )
        causal_report = _first_json(
            candidate_root / "legacy_scaled_zero_shuffle_gate20.json",
            candidate_root / "validation" / "legacy_scaled_zero_shuffle_gate20.json",
        )
        source_report = _first_json(
            candidate_root / "source_shuffle_gate20.json",
            candidate_root / "validation" / "source_shuffle_gate20.json",
        )
        special = _first_json(
            candidate_root / "special_acceptance.json",
            candidate_root / "validation" / "special_acceptance.json",
        )
        candidate_report = _first_json(
            candidate_root / "causal_candidate_report.json",
            candidate_root / "validation" / "causal_candidate_report.json",
            candidate_root / "validation" / "candidate_report.json",
        )
        gate20 = _mapping_or_empty(accepted.get("gate20")) or _gate20_from_disk(
            candidate_root
        )
        causal = _mapping_or_empty(special.get("causal")) or causal_report
        normal = _first_number(
            _condition_macro(gate20, "normal"),
            _metric(causal, "normal_macro", "normal_macro_average", "normal"),
        )
        legacy = _first_number(
            _condition_macro(gate20, "legacy_reference"),
            _metric(causal, "legacy_macro", "legacy_reference_macro", "legacy"),
        )
        zero = _first_number(
            _condition_macro(gate20, "world_evidence_gate_zero"),
            _metric(
                causal,
                "world_evidence_gate_zero_macro",
                "new_gate_zero_macro",
                "gate_zero_macro",
            ),
        )
        shuffled = _first_number(
            _condition_macro(gate20, "shuffle_all"),
            _metric(
                causal,
                "joint_shuffle_macro",
                "shuffled_future_macro",
                "shuffle_macro",
            ),
        )
        causal_gate = _causal_gate(normal, legacy, zero, shuffled)
        core_complete = sum(condition in gate20 for condition in CORE_CONDITIONS)
        diagnostic_complete = sum(
            condition in gate20 for condition in DIAGNOSTIC_CONDITIONS
        )
        lines.append(
            f"  {candidate} validation priority | core={core_complete}/4 "
            f"diagnostic={diagnostic_complete}/4; normal is completed first"
        )
        lines.append(
            f"  {candidate} causal | {causal_gate} normal={_number_text(normal)} "
            f"legacy={_number_text(legacy)} new-gate-zero={_number_text(zero)} "
            f"shuffle={_number_text(shuffled)}"
        )
        lines.append(
            f"  {candidate} gradient | {_explicit_pass_gate(gradient)} "
            f"artifact={_presence(gradient)}"
        )
        candidate_structure = _mapping_or_empty(accepted.get("structural_invariants"))
        if candidate_structure:
            lines.append(
                f"  {candidate} structure | {_boolean_gate(candidate_structure)}"
            )
        gaps = (
            _mapping_or_empty(accepted.get("source_shuffle_gaps"))
            or _mapping_or_empty(special.get("source_gaps"))
            or source_report
        )
        lines.append(
            f"  {candidate} source gaps (claim/report) | "
            f"own={_number_text(_metric(gaps, 'own', 'own_gap', 'shuffle_own_gap'))} "
            f"peer={_number_text(_metric(gaps, 'peer', 'peer_gap', 'shuffle_peer_gap'))} "
            f"shared={_number_text(_metric(gaps, 'shared', 'shared_gap', 'shuffle_shared_gap'))}"
        )
        normal_tasks = _mapping_or_empty(
            _mapping_or_empty(gate20.get("normal")).get("tasks")
        )
        if normal_tasks:
            lines.append(
                f"  {candidate} normal Gate20 by task | "
                + " ".join(
                    f"{task}={_mapping_or_empty(row).get('successes', '?')}/"
                    f"{_mapping_or_empty(row).get('episodes', 20)}"
                    for task, row in normal_tasks.items()
                )
            )
        if candidate == "P1":
            utility = _first_json(
                candidate_root / "router_utility_spearman.json",
                candidate_root / "validation" / "router_utility_spearman.json",
                candidate_root / "audit" / "router_utility_spearman.json",
            )
            utility_values = (
                _mapping_or_empty(candidate_report.get("utility_calibration"))
                or _mapping_or_empty(special.get("utility"))
                or utility
            )
            coefficient = _metric(
                utility_values, "spearman", "spearman_rho", "coefficient"
            )
            lower = _metric(
                utility_values,
                "bootstrap_ci95_lower",
                "ci95_lower",
                "episode_bootstrap_lower",
                "episode_bootstrap_95_lower",
            )
            utility_checks = _mapping_or_empty(accepted.get("utility_checks"))
            utility_gate = _boolean_gate(utility_checks)
            if utility_gate == "pending":
                utility_gate = (
                    "PASS"
                    if coefficient is not None
                    and lower is not None
                    and coefficient > 0
                    and lower > 0
                    else "FAIL"
                    if coefficient is not None and lower is not None
                    else "pending"
                )
            lines.append(
                f"  P1 utility calibration | {utility_gate} "
                f"spearman={_number_text(coefficient)} ci95_lower={_number_text(lower)}"
            )

        exposure = _first_json(
            candidate_root / "module_exposure.json",
            candidate_root / "train" / "module_exposure.json",
            candidate_root / "audit" / "module_exposure.json",
        )
        forced = any(candidate_root.rglob("forced_evidence_errors.npz"))
        lines.append(
            f"  {candidate} required artifacts | exposure={_presence(exposure)} "
            f"forced-evidence={'present' if forced else 'missing'}"
        )
        required_reports = _mapping_or_empty(accepted.get("required_reports"))
        if required_reports:
            lines.append(
                f"  {candidate} acceptance report references | "
                f"{_boolean_gate(required_reports)}"
            )

    lines.append(
        "  FINAL | "
        + (
            f"decision={final.get('decision', '?')} winner={final.get('winner', '?')} "
            f"eligible={final.get('eligible_candidates', '?')} "
            f"r8_may_start={final.get('r8_may_start', '?')} "
            "(metrics above remain authoritative)"
            if final
            else "pending pair decision"
        )
    )
    return lines


def _condition_macro(gate20: Mapping[str, Any], condition: str) -> float | None:
    row = _mapping_or_empty(gate20.get(condition))
    return _metric(row, "macro_success_rate", "macro_average_success_rate")


def _gate20_from_disk(candidate_root: Path) -> Mapping[str, Any]:
    """Expose completed condition summaries before the final report exists."""

    observed: dict[str, Any] = {}
    for condition in (*CORE_CONDITIONS, *DIAGNOSTIC_CONDITIONS):
        summary = _first_json(
            candidate_root
            / "validation"
            / "gate20"
            / condition
            / "gate_summary.json"
        )
        if tuple(summary.get("task_order", ())) != TASKS:
            continue
        tasks: dict[str, Any] = {}
        for task in TASKS:
            row = _mapping_or_empty(summary.get(task))
            episodes = row.get("episodes")
            if not isinstance(episodes, list) or len(episodes) != 20:
                tasks = {}
                break
            successes = sum(
                1
                for episode in episodes
                if isinstance(episode, Mapping) and episode.get("success") is True
            )
            tasks[task] = {
                "successes": successes,
                "episodes": 20,
                "success_rate": successes / 20.0,
            }
        if not tasks:
            continue
        observed[condition] = {
            "tasks": tasks,
            "macro_success_rate": sum(
                float(row["success_rate"]) for row in tasks.values()
            )
            / len(TASKS),
        }
    return observed


def _first_number(*values: float | None) -> float | None:
    return next((value for value in values if value is not None), None)


def _causal_gate(
    normal: float | None,
    legacy: float | None,
    zero: float | None,
    shuffled: float | None,
) -> str:
    if None in {normal, legacy, zero, shuffled}:
        return "pending"
    assert normal is not None and legacy is not None and zero is not None and shuffled is not None
    return "PASS" if normal >= legacy and normal > zero and normal > shuffled else "FAIL"


def _boolean_gate(report: Mapping[str, Any]) -> str:
    values = list(_boolean_values(report))
    if not values:
        return "pending"
    passed = sum(values)
    return f"{'PASS' if passed == len(values) else 'FAIL'} {passed}/{len(values)}"


def _explicit_pass_gate(report: Mapping[str, Any]) -> str:
    passed = report.get("passed")
    if passed is True:
        return "PASS"
    if passed is False:
        return "FAIL"
    return _boolean_gate(report)


def _boolean_values(value: Any) -> list[bool]:
    if isinstance(value, bool):
        return [value]
    if isinstance(value, Mapping):
        result: list[bool] = []
        for key, child in value.items():
            if key in {"passed", "complete"}:
                continue
            result.extend(_boolean_values(child))
        return result
    if isinstance(value, list):
        result = []
        for child in value:
            result.extend(_boolean_values(child))
        return result
    return []


def _metric(report: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in report:
            value = report[key]
            if isinstance(value, Mapping):
                for nested in ("macro", "value", "success_rate", "mean"):
                    if nested in value:
                        number = _number(value[nested])
                        if number is not None:
                            return number
            number = _number(value)
            if number is not None:
                return number
    for child in report.values():
        if isinstance(child, Mapping):
            result = _metric(child, *keys)
            if result is not None:
                return result
    return None


def _preflight_text(candidate_root: Path, explicit: Any) -> str:
    if explicit not in {None, ""}:
        return str(explicit)
    report = _first_json(
        candidate_root / "preflight.json",
        candidate_root / "preflight" / "preflight.json",
    )
    values = list(_boolean_values(report))
    if not report:
        return "pending"
    if values:
        return "PASS" if all(values) else "FAIL"
    return str(report.get("status", "present"))


def _milestone_text(update: int) -> str:
    for milestone in MILESTONES:
        if update < milestone:
            return f"next={milestone}"
    return "30k-complete"


def _terminal_heartbeat(phase: str, updated_at: Any) -> str:
    if phase in TERMINAL_PHASES:
        return "finished"
    return _heartbeat(updated_at)


def _heartbeat(value: Any) -> str:
    age = _age_seconds(value)
    if age is None:
        return "missing"
    state = "alive" if age <= STALE_SECONDS else "STALE"
    return f"{state}:{_duration(age)}"


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


def _gpu_processes_by_index() -> dict[int, list[int]]:
    if shutil.which("nvidia-smi") is None:
        return {}
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    uuid_to_index: dict[str, int] = {}
    for row in gpu.stdout.splitlines():
        fields = [field.strip() for field in row.split(",", 1)]
        if len(fields) == 2 and fields[0].isdigit():
            uuid_to_index[fields[1]] = int(fields[0])
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    result: dict[int, list[int]] = {}
    for row in processes.stdout.splitlines():
        fields = [field.strip() for field in row.split(",", 1)]
        if len(fields) != 2 or not fields[1].isdigit():
            continue
        index = uuid_to_index.get(fields[0])
        if index is not None:
            result.setdefault(index, []).append(int(fields[1]))
    return result


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
    lines = ["GPU index/name/util/memory:"]
    rows = [row for row in result.stdout.splitlines() if row.strip()]
    lines.extend(f"  {row}" for row in rows)
    if not rows:
        lines.append("  unavailable")
    return lines


def _first_child_pid(pid: int | None) -> int | None:
    if pid is None:
        return None
    try:
        values = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
    except OSError:
        return None
    return int(values[0]) if values and values[0].isdigit() else None


def _latest_jsonl(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    data = path.read_bytes()[-256 * 1024 :]
    for line in reversed(data.decode("utf-8", errors="replace").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _first_json(*paths: Path) -> dict[str, Any]:
    for path in paths:
        value = _maybe_json(path)
        if value:
            return value
    return {}


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
    return result if math.isfinite(result) else None


def _number_text(value: Any) -> str:
    number = _number(value)
    return "?" if number is None else f"{number:.6g}"


def _lr_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return ",".join(f"{key}={_number_text(item)}" for key, item in value.items())
    return str(value) if value not in {None, ""} else "?"


def _fraction(value: Any, total: Any) -> str:
    left = "?" if value is None else str(value)
    right = "?" if total is None else str(total)
    return f"{left}/{right}"


def _percent(value: int | None, total: int | None) -> str:
    if value is None or total in {None, 0}:
        return "?"
    return f"{100.0 * value / total:.1f}%"


def _compact(value: Any, width: int) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _presence(report: Mapping[str, Any]) -> str:
    return "present" if report else "missing"


def _candidate_root(root: Path, candidate: str | None) -> Path:
    if candidate not in CANDIDATES:
        raise ValueError(f"invalid candidate {candidate!r}")
    return root / "candidates" / candidate.lower()


def _safe_identifier(value: str) -> bool:
    return bool(value) and value[0].isalnum() and all(
        character.isalnum() or character in "_.-" for character in value
    )


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
