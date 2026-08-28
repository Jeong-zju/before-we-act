"""Fault-tolerant official MARS expert collection.

Each motion-planning attempt runs in a fresh process. Successful episodes are
kept as individual HDF5 parts and atomically merged only when a shard is full.
This bounds pathological planner calls and makes collection resumable.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import deque

import h5py
import numpy as np
import yaml


def trajectory_names(handle: h5py.File) -> list[str]:
    return sorted(
        (key for key in handle if key.startswith("traj_")),
        key=lambda key: int(key.split("_")[-1]),
    )


def successful_part(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as handle:
            names = trajectory_names(handle)
            return len(names) == 1 and bool(np.asarray(handle[names[0]]["success"])[-1])
    except Exception:
        return False


def seed_from_path(path: Path) -> int | None:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return None


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def copy_episode(source: h5py.File, source_name: str, target: Path) -> None:
    temporary = target.with_suffix(".h5.tmp")
    with h5py.File(temporary, "w") as output:
        source.copy(source_name, output, name="traj_0")
    os.replace(temporary, target)


def adopt_partial(target: Path, parts: Path) -> None:
    """Salvage successful episodes written by the old non-resumable collector."""
    if not target.exists():
        return
    metadata_path = target.with_suffix(".json")
    metadata = {}
    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception:
        pass
    episodes = metadata.get("episodes", [])
    adopted = 0
    try:
        with h5py.File(target, "r") as source:
            for index, name in enumerate(trajectory_names(source)):
                if not bool(np.asarray(source[name]["success"])[-1]):
                    continue
                episode = episodes[index] if index < len(episodes) else {}
                seed = int(episode.get("episode_seed", -(index + 1)))
                part = parts / f"seed_{seed:012d}.h5"
                if not successful_part(part):
                    copy_episode(source, name, part)
                    part_metadata = {key: value for key, value in metadata.items() if key != "episodes"}
                    part_metadata["episodes"] = [{**episode, "episode_id": 0}]
                    atomic_json(part.with_suffix(".json"), part_metadata)
                adopted += 1
    except Exception as error:
        print({"event": "partial_adoption_failed", "path": str(target), "error": repr(error)}, flush=True)
        return
    timestamp = int(time.time())
    os.replace(target, target.with_name(f"{target.name}.incomplete.{timestamp}"))
    if metadata_path.exists():
        os.replace(metadata_path, metadata_path.with_name(f"{metadata_path.name}.incomplete.{timestamp}"))
    print({"event": "partial_adopted", "episodes": adopted, "parts": str(parts)}, flush=True)


def valid_parts(parts: Path) -> list[Path]:
    return sorted(
        (path for path in parts.glob("seed_*.h5") if successful_part(path)),
        key=lambda path: (seed_from_path(path) is None, seed_from_path(path) or 0),
    )


def merge_parts(paths: list[Path], target: Path) -> None:
    temporary = target.with_suffix(".h5.tmp")
    combined_metadata: dict = {}
    combined_episodes: list[dict] = []
    with h5py.File(temporary, "w") as output:
        for episode_id, path in enumerate(paths):
            with h5py.File(path, "r") as source:
                names = trajectory_names(source)
                if len(names) != 1 or not bool(np.asarray(source[names[0]]["success"])[-1]):
                    raise RuntimeError(f"invalid successful part: {path}")
                source.copy(names[0], output, name=f"traj_{episode_id}")
            try:
                metadata = json.loads(path.with_suffix(".json").read_text())
                if not combined_metadata:
                    combined_metadata = {key: value for key, value in metadata.items() if key != "episodes"}
                episode = metadata.get("episodes", [{}])[0]
            except Exception:
                episode = {}
            combined_episodes.append({**episode, "episode_id": episode_id})
    combined_metadata["episodes"] = combined_episodes
    os.replace(temporary, target)
    atomic_json(target.with_suffix(".json"), combined_metadata)


def clean_attempt(parts: Path, attempt_name: str) -> None:
    for suffix in (".h5", ".json", ".h5.tmp", ".json.tmp", ".timeout.json", ".failed.json"):
        try:
            (parts / f"{attempt_name}{suffix}").unlink()
        except FileNotFoundError:
            pass


def mark_attempt(parts: Path, seed: int, outcome: str, elapsed: float, returncode: int | None) -> None:
    atomic_json(
        parts / f"seed_{seed:012d}.{outcome}.json",
        {"seed": seed, "outcome": outcome, "elapsed_seconds": elapsed,
         "returncode": returncode, "collector_version": 2},
    )


def attempted_seeds(parts: Path) -> list[int]:
    values: list[int] = []
    for path in parts.glob("seed_*.*"):
        value = seed_from_path(Path(path.name.split(".")[0]))
        if value is not None:
            values.append(value)
    return values


def legacy_timeout_seeds(parts: Path) -> list[int]:
    """Old recorder timeouts are eligible once under the optimized recorder."""
    successful = {seed_from_path(path) for path in valid_parts(parts)}
    values: list[int] = []
    for path in parts.glob("seed_*.timeout.json"):
        seed = seed_from_path(Path(path.name.split(".")[0]))
        if seed is None or seed in successful:
            continue
        try:
            if int(json.loads(path.read_text()).get("collector_version", 1)) >= 2:
                continue
        except Exception:
            pass
        values.append(seed)
    return sorted(set(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--render-device", choices=("cuda:0", "cuda:1"), required=True)
    parser.add_argument("--attempt-timeout", type=float, default=float(os.environ.get("MARS_EXPERT_ATTEMPT_TIMEOUT", "180")))
    parser.add_argument("--max-attempts", type=int, default=500)
    args = parser.parse_args()

    with open(args.config) as stream:
        env_id = yaml.safe_load(stream)["task_name"] + "-rf"
    directory = args.record_dir / env_id / "motionplanning"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{args.name}.h5"
    parts = directory / f".{args.name}.parts"
    parts.mkdir(parents=True, exist_ok=True)
    adopt_partial(target, parts)

    previous = attempted_seeds(parts)
    seed = max([args.seed - 1, *previous]) + 1
    retry_seeds = deque(legacy_timeout_seeds(parts))
    if retry_seeds:
        print({"event": "legacy_timeout_retry_queue", "count": len(retry_seeds)}, flush=True)
    attempts = 0
    while len(valid_parts(parts)) < args.episodes:
        if attempts >= args.max_attempts:
            raise RuntimeError(f"exhausted {args.max_attempts} attempts for {args.name}")
        attempt_seed = retry_seeds.popleft() if retry_seeds else seed
        if attempt_seed == seed:
            seed += 1
        attempt_name = f"seed_{attempt_seed:012d}"
        clean_attempt(parts, attempt_name)
        argv = [
            sys.executable,
            "-m",
            "deployment.mars_care.expert_attempt",
            "--config", args.config,
            "--seed", str(attempt_seed),
            "--output-dir", str(parts),
            "--name", attempt_name,
            "--render-device", args.render_device,
        ]
        started = time.monotonic()
        process = subprocess.Popen(argv)
        outcome = "failed"
        try:
            returncode = process.wait(timeout=args.attempt_timeout)
            if returncode == 0 and successful_part(parts / f"{attempt_name}.h5"):
                outcome = "success"
            else:
                clean_attempt(parts, attempt_name)
                mark_attempt(parts, attempt_seed, "failed", time.monotonic() - started, returncode)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            returncode = process.returncode
            clean_attempt(parts, attempt_name)
            mark_attempt(parts, attempt_seed, "timeout", time.monotonic() - started, returncode)
            outcome = "timeout"
        attempts += 1
        saved = len(valid_parts(parts))
        print(
            {"saved": saved, "target": args.episodes, "seed": attempt_seed, "outcome": outcome,
             "elapsed_seconds": round(time.monotonic() - started, 1), "timeout_seconds": args.attempt_timeout},
            flush=True,
        )

    selected = valid_parts(parts)[:args.episodes]
    merge_parts(selected, target)
    print({"event": "shard_complete", "target": str(target), "episodes": len(selected)}, flush=True)


if __name__ == "__main__":
    main()
