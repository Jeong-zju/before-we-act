#!/usr/bin/env python3
"""Read-only live progress display for sharded RoboFactory collection."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import re
import sys
import time


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_record_dir = (
        project_root.parent
        / "RoboFactory"
        / "data"
        / "m2_raw"
        / "LongPipelineDelivery-rf"
        / "motionplanning"
    )
    parser = argparse.ArgumentParser(
        description="Display live RoboFactory HDF5/JSON collection progress."
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=default_record_dir,
        help="Directory containing the trajectory JSON/HDF5 files.",
    )
    parser.add_argument(
        "--traj-name",
        default="LongPipelineDelivery-rf_m2_multiview_150",
        help="Trajectory basename before .<worker>.json.",
    )
    parser.add_argument("--expected", type=int, default=150)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Expected worker count; zero discovers it from shard filenames.",
    )
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.expected <= 0:
        parser.error("--expected must be positive")
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    if args.interval <= 0:
        parser.error("--interval must be positive")
    return args


def episode_count(path: Path) -> int | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        episodes = payload["episodes"]
        if not isinstance(episodes, list):
            return None
        return len(episodes)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        # A worker may be replacing metadata while it is sampled.
        return None


def discover_shards(
    record_dir: Path, traj_name: str
) -> dict[int, tuple[Path, int | None]]:
    pattern = re.compile(rf"^{re.escape(traj_name)}\.(\d+)\.json$")
    shards: dict[int, tuple[Path, int | None]] = {}
    try:
        paths = tuple(record_dir.iterdir())
    except FileNotFoundError:
        return shards
    for path in paths:
        match = pattern.match(path.name)
        if match is not None:
            shards[int(match.group(1))] = (path, episode_count(path))
    return shards


def worker_targets(expected: int, workers: int) -> tuple[int, ...]:
    quotient, remainder = divmod(expected, workers)
    return tuple(quotient + int(index < remainder) for index in range(workers))


def progress_bar(done: int, expected: int, width: int = 40) -> str:
    ratio = min(max(done / expected, 0.0), 1.0)
    filled = min(width, int(ratio * width))
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--"
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    return f"{minutes:d}m{seconds:02d}s"


def render(
    args: argparse.Namespace,
    history: deque[tuple[float, int]],
) -> tuple[bool, str]:
    now = time.time()
    final_json = args.record_dir / f"{args.traj_name}.json"
    final_count = episode_count(final_json) if final_json.exists() else None
    shards = discover_shards(args.record_dir, args.traj_name)
    discovered_workers = max(shards, default=-1) + 1
    workers = args.workers or discovered_workers

    if final_count is not None and final_count >= args.expected:
        total = final_count
        workers = 0
        complete = True
    else:
        total = sum(count or 0 for _, count in shards.values())
        complete = False

    history.append((now, total))
    while len(history) > 1 and now - history[0][0] > 120:
        history.popleft()
    elapsed = history[-1][0] - history[0][0]
    delta = history[-1][1] - history[0][1]
    rate_per_minute = delta / elapsed * 60 if elapsed > 0 and delta > 0 else None
    eta = (
        (args.expected - total) / rate_per_minute * 60
        if rate_per_minute is not None
        else None
    )

    data_paths = [
        args.record_dir / f"{args.traj_name}.{index}.h5"
        for index in shards
    ]
    if final_json.exists():
        data_paths.append(args.record_dir / f"{args.traj_name}.h5")
    total_bytes = sum(
        path.stat().st_size for path in data_paths if path.exists()
    )
    mtimes = [
        path.stat().st_mtime
        for path, _ in shards.values()
        if path.exists()
    ]
    latest_age = now - max(mtimes) if mtimes else None

    if complete:
        state = "COMPLETE"
    elif total >= args.expected:
        state = "MERGING"
    elif shards and (latest_age is None or latest_age < 300):
        state = "COLLECTING"
    elif shards:
        state = "NO RECENT METADATA UPDATE"
    else:
        state = "WAITING FOR WORKERS"

    percent = min(total / args.expected * 100, 100.0)
    lines = [
        f"{args.traj_name}  {state}",
        (
            f"{progress_bar(total, args.expected)} "
            f"{total}/{args.expected}  {percent:5.1f}%"
        ),
        (
            f"workers={len(shards)}/{workers or '-'}  "
            f"rate={rate_per_minute:.2f} episodes/min  "
            f"ETA={format_duration(eta)}"
            if rate_per_minute is not None
            else (
                f"workers={len(shards)}/{workers or '-'}  "
                "rate=sampling...  ETA=--"
            )
        ),
        (
            f"shard_size={total_bytes / (1024 ** 3):.2f} GiB  "
            f"latest_metadata={format_duration(latest_age)} ago"
        ),
    ]

    if workers:
        targets = worker_targets(args.expected, workers)
        lines.extend(["", "worker   done/target   progress   metadata_age"])
        for index, target in enumerate(targets):
            entry = shards.get(index)
            count = entry[1] if entry is not None and entry[1] is not None else 0
            age = now - entry[0].stat().st_mtime if entry is not None else None
            lines.append(
                f"{index:>4}   {count:>3}/{target:<3}      "
                f"{count / target * 100:>6.1f}%   {format_duration(age):>8}"
            )
    lines.extend(
        ["", f"updated {time.strftime('%Y-%m-%d %H:%M:%S')}  Ctrl-C to stop"]
    )
    return complete, "\n".join(lines)


def main() -> int:
    args = parse_args()
    history: deque[tuple[float, int]] = deque(maxlen=32)
    try:
        while True:
            complete, display = render(args, history)
            if sys.stdout.isatty() and not args.once:
                print("\033[2J\033[H", end="")
            print(display, flush=True)
            if args.once or complete:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nProgress monitor stopped; collection was not interrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
