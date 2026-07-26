#!/usr/bin/env python3
"""Select safe CPU seed-parallelism for RoboFactory data collection."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys


MIB = 1024 * 1024
TASK_MEMORY_MIB = {
    # Native sensor RGB is 640x480.  LongPipeline retains five camera streams
    # for up to ~800 steps, so one worker can transiently own several GiB.
    "lift_barrier": 4096,
    "long_pipeline_delivery": 6144,
}


@dataclass(frozen=True)
class WorkerPlan:
    workers: int
    threads_per_worker: int
    effective_cpus: int
    available_memory_bytes: int
    cpu_limit: int
    memory_limit: int
    automatic_limit: int
    manually_overridden: bool


def choose_worker_plan(
    *,
    task: str,
    trajectories: int,
    effective_cpus: int,
    available_memory_bytes: int,
    requested: str,
    max_workers: int,
    cpu_threads_per_worker: int,
    memory_fraction: float,
) -> WorkerPlan:
    if task not in TASK_MEMORY_MIB:
        raise ValueError(f"unknown collection task {task!r}")
    if trajectories <= 0 or effective_cpus <= 0 or available_memory_bytes <= 0:
        raise ValueError("trajectory/compute capacities must be positive")
    if max_workers <= 0 or cpu_threads_per_worker <= 0:
        raise ValueError("worker limits must be positive")
    if not 0.1 <= memory_fraction <= 0.95:
        raise ValueError("memory_fraction must lie in [0.1,0.95]")

    reserved_cpus = min(2, max(0, effective_cpus - 1))
    usable_cpus = max(1, effective_cpus - reserved_cpus)
    cpu_limit = max(1, usable_cpus // cpu_threads_per_worker)
    task_memory_bytes = TASK_MEMORY_MIB[task] * MIB
    memory_budget = int(available_memory_bytes * memory_fraction)
    memory_limit = max(1, memory_budget // task_memory_bytes)
    automatic_limit = max(
        1,
        min(cpu_limit, memory_limit, max_workers, trajectories),
    )

    normalized = requested.strip().lower()
    manually_overridden = normalized != "auto"
    if manually_overridden:
        try:
            workers = int(normalized)
        except ValueError as exc:
            raise ValueError("requested workers must be 'auto' or an integer") from exc
        if not 1 <= workers <= trajectories:
            raise ValueError("requested workers must lie in [1, trajectories]")
    else:
        workers = automatic_limit
    threads_per_worker = max(1, effective_cpus // workers)
    return WorkerPlan(
        workers=workers,
        threads_per_worker=threads_per_worker,
        effective_cpus=effective_cpus,
        available_memory_bytes=available_memory_bytes,
        cpu_limit=cpu_limit,
        memory_limit=memory_limit,
        automatic_limit=automatic_limit,
        manually_overridden=manually_overridden,
    )


def effective_cpu_count() -> int:
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = os.cpu_count() or 1
    limits = [float(affinity)]
    for directory in _cgroup_directories():
        path = directory / "cpu.max"
        if not path.is_file():
            continue
        fields = path.read_text(encoding="utf-8").strip().split()
        if len(fields) != 2 or fields[0] == "max":
            continue
        quota, period = map(int, fields)
        if quota > 0 and period > 0:
            limits.append(quota / period)
    return max(1, int(math.floor(min(limits))))


def available_memory_bytes() -> int:
    host_available = None
    with Path("/proc/meminfo").open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                host_available = int(line.split()[1]) * 1024
                break
    if host_available is None or host_available <= 0:
        raise RuntimeError("/proc/meminfo lacks MemAvailable")
    limits = [host_available]
    for directory in _cgroup_directories():
        maximum_path = directory / "memory.max"
        current_path = directory / "memory.current"
        if not maximum_path.is_file() or not current_path.is_file():
            continue
        maximum_text = maximum_path.read_text(encoding="utf-8").strip()
        if maximum_text == "max":
            continue
        maximum = int(maximum_text)
        current = int(current_path.read_text(encoding="utf-8").strip())
        if maximum > current:
            limits.append(maximum - current)
    return max(1, min(limits))


def _cgroup_directories() -> tuple[Path, ...]:
    root = Path("/sys/fs/cgroup")
    try:
        unified = next(
            line.split("::", 1)[1].strip()
            for line in Path("/proc/self/cgroup").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.startswith("0::")
        )
    except (OSError, StopIteration):
        return ()
    current = root / unified.lstrip("/")
    values: list[Path] = []
    while current == root or root in current.parents:
        values.append(current)
        if current == root:
            break
        current = current.parent
    return tuple(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=tuple(TASK_MEMORY_MIB),
        required=True,
    )
    parser.add_argument("--trajectories", type=int, default=150)
    parser.add_argument("--requested", default="auto")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=2)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plan = choose_worker_plan(
        task=args.task,
        trajectories=args.trajectories,
        effective_cpus=effective_cpu_count(),
        available_memory_bytes=available_memory_bytes(),
        requested=args.requested,
        max_workers=args.max_workers,
        cpu_threads_per_worker=args.cpu_threads_per_worker,
        memory_fraction=args.memory_fraction,
    )
    mode = "manual" if plan.manually_overridden else "auto"
    warning = (
        " WARNING: manual value exceeds the automatic safety limit."
        if plan.manually_overridden and plan.workers > plan.automatic_limit
        else ""
    )
    print(
        "[collection-workers] "
        f"task={args.task} mode={mode} cpus={plan.effective_cpus} "
        f"available_memory_gib={plan.available_memory_bytes / (1024**3):.1f} "
        f"cpu_limit={plan.cpu_limit} memory_limit={plan.memory_limit} "
        f"cap={args.max_workers} workers={plan.workers} "
        f"threads_per_worker={plan.threads_per_worker}.{warning}",
        file=sys.stderr,
        flush=True,
    )
    print(plan.workers, plan.threads_per_worker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
