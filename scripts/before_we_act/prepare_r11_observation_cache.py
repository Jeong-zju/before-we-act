#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from before_we_act.data.raw_team_windows import TASKS, manifest_receipt  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def patch_means(image: np.ndarray, grid: int) -> np.ndarray:
    height, width, channels = image.shape
    if channels != 3 or height % grid or width % grid:
        raise ValueError("R11 images must be divisible into the frozen RGB grid")
    value = image.reshape(grid, height // grid, grid, width // grid, 3)
    return value.mean(axis=(1, 3), dtype=np.float32).reshape(grid * grid, 3) / 127.5 - 1.0


def choose_examples(manifests: dict, split: str, count: int, seed: int):
    rng = random.Random(seed + (0 if split == "train" else 10_000))
    pools = {}
    for task in TASKS:
        episodes = [row for row in manifests[task]["episodes"] if row["split"] == split]
        if not episodes:
            raise ValueError(f"no {split} episodes for {task}")
        pools[task] = episodes
    examples = []
    for index in range(count):
        task = TASKS[index % len(TASKS)]
        episode = rng.choice(pools[task])
        steps = int(episode["steps"])
        if steps < 5:
            raise ValueError("episode is too short for the frozen R11 history")
        current = rng.randint(2, steps - 2)
        examples.append((task, episode, current))
    rng.shuffle(examples)
    return examples


def read_example(data_root: Path, item, grid: int = 4):
    task, episode, current = item
    path = data_root / task / episode["hdf5_path"]
    history_indices = list(range(current - 2, current + 1))
    with h5py.File(path, "r") as handle:
        agent_names = sorted(handle["data/observation/agents"].keys())
        view_names = ["global"] + [f"agent_{index}" for index in range(len(agent_names))]
        visual = np.zeros((3, grid * grid, 15), dtype=np.float16)
        view_mask = np.zeros((3, 5), dtype=np.float16)
        qpos = np.zeros((3, 4, 9), dtype=np.float32)
        actions = np.zeros((3, 4, 8), dtype=np.float32)
        for time_index, row in enumerate(history_indices):
            for view_index, view in enumerate(view_names):
                rgb = handle[f"data/observation/images/{view}"][row]
                visual[time_index, :, view_index * 3 : (view_index + 1) * 3] = patch_means(rgb, grid)
                view_mask[time_index, view_index] = 1
            for agent_index, agent in enumerate(agent_names):
                qpos[time_index, agent_index] = handle[f"data/observation/agents/{agent}/qpos"][row]
                actions[time_index, agent_index] = handle[f"data/action/agents/{agent}/executed"][row]
        future_visual = np.zeros((grid * grid, 15), dtype=np.float16)
        for view_index, view in enumerate(view_names):
            rgb = handle[f"data/next_observation/images/{view}"][current]
            future_visual[:, view_index * 3 : (view_index + 1) * 3] = patch_means(rgb, grid)
        partner_action = np.zeros((4, 8), dtype=np.float32)
        for agent_index, agent in enumerate(agent_names):
            partner_action[agent_index] = handle[f"data/action/agents/{agent}/executed"][current + 1]
    agent_mask = np.zeros(4, dtype=np.bool_)
    agent_mask[: len(agent_names)] = True
    return {
        "visual": visual,
        "view_mask": view_mask,
        "qpos": qpos,
        "actions": actions,
        "agent_mask": agent_mask,
        "future_visual": future_visual,
        "partner_action": partner_action,
        "shared_progress": np.float32((current + 1) / max(1, int(episode["steps"]) - 1)),
        "task_index": np.int64(TASKS.index(task)),
    }


def stack(rows: list[dict]) -> dict[str, torch.Tensor]:
    result = {}
    for key in rows[0]:
        value = np.stack([row[key] for row in rows])
        result[key] = torch.from_numpy(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/workspace/datasets/robofactory_multitask")
    parser.add_argument("--output", default="/workspace/bwa_runs/shared/r11_observation_cache.pt")
    parser.add_argument("--state", default="/workspace/bwa_runs/shared/r11_observation_cache_state.json")
    parser.add_argument("--heartbeat", default="/workspace/bwa_runs/shared/r11_observation_cache_heartbeat.json")
    parser.add_argument("--train-windows", type=int, default=4096)
    parser.add_argument("--validation-windows", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    if args.train_windows != 4096 or args.validation_windows != 1024:
        raise ValueError("R11 cache size is frozen at 4096/1024")
    data_root, output = Path(args.data_root), Path(args.output)
    state, heartbeat = Path(args.state), Path(args.heartbeat)
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = manifest_receipt(data_root)
    if output.exists():
        payload = torch.load(output, map_location="cpu", weights_only=False)
        if payload.get("metadata", {}).get("manifest_sha256") == receipt:
            atomic_json(state, {"state": "PASSED", "detail": "verified existing cache", "updated_at": now(), "path": str(output)})
            print(f"verified existing R11 observation cache: {output}")
            return
        raise ValueError("existing R11 observation cache has a different manifest receipt")
    manifests = {
        task: json.loads((data_root / task / "training_manifest.json").read_text(encoding="utf-8"))
        for task in TASKS
    }
    started = time.monotonic()
    atomic_json(state, {"state": "PREPARING", "detail": "reading legal fixed-view windows", "updated_at": now(), "pid": os.getpid()})
    splits = {}
    total = args.train_windows + args.validation_windows
    completed = 0
    for split, count in (("train", args.train_windows), ("validation", args.validation_windows)):
        rows = []
        for item in choose_examples(manifests, split, count, args.seed):
            rows.append(read_example(data_root, item))
            completed += 1
            if completed % 20 == 0 or completed == total:
                progress = {
                    "state": "PREPARING",
                    "stage": "observation_cache",
                    "completed": completed,
                    "total": total,
                    "updated_at": now(),
                    "pid": os.getpid(),
                    "elapsed_seconds": time.monotonic() - started,
                }
                atomic_json(heartbeat, progress)
                atomic_json(state, progress)
        splits[split] = stack(rows)
    payload = {
        "schema_version": 1,
        "metadata": {
            "created_at": now(),
            "data_root": str(data_root.resolve()),
            "manifest_sha256": receipt,
            "tasks": TASKS,
            "history_steps": 3,
            "grid_size": 4,
            "train_windows": args.train_windows,
            "validation_windows": args.validation_windows,
            "seed": args.seed,
            "legal_inputs": ["fixed_view_rgb", "qpos", "executed_action_history", "agent_view_masks"],
            "forbidden_inputs": ["task_id", "robot_id", "role_id", "simulator_state", "core_hidden", "core_router", "forced_role"],
        },
        **splits,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    atomic_json(state, {"state": "PASSED", "detail": "observation cache complete", "updated_at": now(), "path": str(output), "sha256": digest, "windows": total})
    print(json.dumps({"path": str(output), "sha256": digest, "windows": total}, sort_keys=True))


if __name__ == "__main__":
    main()
