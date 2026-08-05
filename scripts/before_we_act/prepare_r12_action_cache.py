#!/usr/bin/env python3
"""Build causal R12 histories, cold starts, and future joint-action targets."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time

import h5py
import numpy as np
import torch

from before_we_act.data.raw_team_windows import TASKS, manifest_receipt


EXPECTED_PARENT_SHA256 = "061b7a4acea8fa10f146779e7a1206822179920dfe573db536d237df81eb541d"
PROTOCOL_VARIANT = "causal_lag1_coldstart_dense_v2"
COLD_START_STEPS = (0, 1, 2)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_examples(manifests: dict, split: str, per_episode: int, seed: int):
    """Return dense, unique interior windows plus every episode's cold-start prefix."""
    rng = random.Random(seed + (0 if split == "train" else 10_000))
    pools = {}
    for task in TASKS:
        rows = [row for row in manifests[task]["episodes"] if row["split"] == split]
        if not rows:
            raise ValueError(f"no {split} episodes for {task}")
        pools[task] = rows
    examples = []
    for task in TASKS:
        for episode in pools[task]:
            steps = int(episode["steps"])
            if steps < 6:
                raise ValueError("episode is too short for causal R12 history and target")
            available = list(range(3, steps - 1))
            if len(available) >= per_episode:
                selected = rng.sample(available, per_episode)
            else:
                selected = [available[index % len(available)] for index in range(per_episode)]
                rng.shuffle(selected)
            examples.extend((task, episode, current) for current in selected)
    for task in TASKS:
        for episode in pools[task]:
            steps = int(episode["steps"])
            examples.extend(
                (task, episode, current)
                for current in COLD_START_STEPS
                if current < steps
            )
    rng.shuffle(examples)
    return examples


def causal_history_indices(current: int, history: int = 3) -> list[int]:
    if current < 0 or history < 1:
        raise ValueError("invalid causal history request")
    return [max(0, current - offset) for offset in range(history - 1, -1, -1)]


def previous_action_index(observation_index: int) -> int | None:
    if observation_index < 0:
        raise ValueError("observation index cannot be negative")
    return observation_index - 1 if observation_index else None


def patch_means(image: np.ndarray, grid: int = 4) -> np.ndarray:
    height, width, channels = image.shape
    if channels != 3 or height % grid or width % grid:
        raise ValueError("R12 images must be divisible into the frozen RGB grid")
    value = image.reshape(grid, height // grid, grid, width // grid, 3)
    return value.mean(axis=(1, 3), dtype=np.float32).reshape(grid * grid, 3) / 127.5 - 1.0


def read_causal_example(data_root: Path, item, stats: dict, horizon: int = 100):
    task, episode, current = item
    path = data_root / task / episode["hdf5_path"]
    history = causal_history_indices(current)
    visual = np.zeros((3, 16, 15), dtype=np.float16)
    view_mask = np.zeros((3, 5), dtype=np.float16)
    qpos = np.zeros((3, 4, 9), dtype=np.float32)
    action_history = np.zeros((3, 4, 8), dtype=np.float32)
    actions = np.zeros((horizon, 4, 8), dtype=np.float32)
    step_mask = np.zeros(horizon, dtype=np.bool_)
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        agents = sorted(data["observation/agents"].keys())
        views = ["global"] + [f"agent_{index}" for index in range(len(agents))]
        for time_index, observation_index in enumerate(history):
            for view_index, view in enumerate(views):
                visual[time_index, :, view_index * 3 : (view_index + 1) * 3] = patch_means(
                    data[f"observation/images/{view}"][observation_index]
                )
                view_mask[time_index, view_index] = 1
            prior = previous_action_index(observation_index)
            for agent_index, agent in enumerate(agents):
                qpos[time_index, agent_index] = data[
                    f"observation/agents/{agent}/qpos"
                ][observation_index]
                if prior is not None:
                    action_history[time_index, agent_index] = data[
                        f"action/agents/{agent}/executed"
                    ][prior]
        end = min(current + horizon, int(episode["steps"]))
        valid = end - current
        step_mask[:valid] = True
        for agent_index, agent in enumerate(agents):
            command = np.asarray(
                data[f"action/agents/{agent}/commanded"][current:end],
                dtype=np.float32,
            )
            if not len(command):
                raise ValueError(f"empty action suffix: {path}:{current}")
            actions[:valid, agent_index] = command
            actions[valid:, agent_index] = command[-1]
    actions = (actions - stats["a_mean"][None, None]) / stats["a_std"][None, None]
    agent_mask = np.zeros(4, dtype=np.bool_)
    agent_mask[: len(agents)] = True
    return {
        "visual": torch.from_numpy(visual),
        "view_mask": torch.from_numpy(view_mask),
        "qpos": torch.from_numpy(qpos),
        "actions": torch.from_numpy(action_history),
        "agent_mask": torch.from_numpy(agent_mask),
        "task_index": torch.tensor(TASKS.index(task), dtype=torch.long),
        "joint_actions": torch.from_numpy(actions),
        "action_step_mask": torch.from_numpy(step_mask),
    }


def read_causal_episode_group(
    data_root: Path,
    task: str,
    episode: dict,
    currents: list[int],
    stats: dict,
    horizon: int = 100,
) -> dict[int, dict[str, torch.Tensor]]:
    """Read one episode once and reproduce ``read_causal_example`` exactly.

    Image frames shared by adjacent history windows are decoded once in bounded
    batches.  This is an artifact-construction optimization only; selected
    windows, legal inputs, targets, normalization, and output order are unchanged.
    """
    path = data_root / task / episode["hdf5_path"]
    histories = {current: causal_history_indices(current) for current in currents}
    observation_indices = sorted({index for values in histories.values() for index in values})
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        agents = sorted(data["observation/agents"].keys())
        views = ["global"] + [f"agent_{index}" for index in range(len(agents))]
        visual_cache: dict[tuple[str, int], np.ndarray] = {}
        for view in views:
            dataset = data[f"observation/images/{view}"]
            for index in observation_indices:
                visual_cache[(view, index)] = patch_means(np.asarray(dataset[index]))
        qpos_by_agent = {
            agent: np.asarray(data[f"observation/agents/{agent}/qpos"], dtype=np.float32)
            for agent in agents
        }
        executed_by_agent = {
            agent: np.asarray(data[f"action/agents/{agent}/executed"], dtype=np.float32)
            for agent in agents
        }
        commanded_by_agent = {
            agent: np.asarray(data[f"action/agents/{agent}/commanded"], dtype=np.float32)
            for agent in agents
        }
    rows = {}
    for current in currents:
        history = histories[current]
        visual = np.zeros((3, 16, 15), dtype=np.float16)
        view_mask = np.zeros((3, 5), dtype=np.float16)
        qpos = np.zeros((3, 4, 9), dtype=np.float32)
        action_history = np.zeros((3, 4, 8), dtype=np.float32)
        actions = np.zeros((horizon, 4, 8), dtype=np.float32)
        step_mask = np.zeros(horizon, dtype=np.bool_)
        for time_index, observation_index in enumerate(history):
            for view_index, view in enumerate(views):
                visual[time_index, :, view_index * 3 : (view_index + 1) * 3] = visual_cache[
                    (view, observation_index)
                ]
                view_mask[time_index, view_index] = 1
            prior = previous_action_index(observation_index)
            for agent_index, agent in enumerate(agents):
                qpos[time_index, agent_index] = qpos_by_agent[agent][observation_index]
                if prior is not None:
                    action_history[time_index, agent_index] = executed_by_agent[agent][prior]
        end = min(current + horizon, int(episode["steps"]))
        valid = end - current
        step_mask[:valid] = True
        for agent_index, agent in enumerate(agents):
            command = commanded_by_agent[agent][current:end]
            if not len(command):
                raise ValueError(f"empty action suffix: {path}:{current}")
            actions[:valid, agent_index] = command
            actions[valid:, agent_index] = command[-1]
        actions = (actions - stats["a_mean"][None, None]) / stats["a_std"][None, None]
        agent_mask = np.zeros(4, dtype=np.bool_)
        agent_mask[: len(agents)] = True
        rows[current] = {
            "visual": torch.from_numpy(visual),
            "view_mask": torch.from_numpy(view_mask),
            "qpos": torch.from_numpy(qpos),
            "actions": torch.from_numpy(action_history),
            "agent_mask": torch.from_numpy(agent_mask),
            "task_index": torch.tensor(TASKS.index(task), dtype=torch.long),
            "joint_actions": torch.from_numpy(actions),
            "action_step_mask": torch.from_numpy(step_mask),
        }
    return rows


def read_causal_episode_group_job(arguments):
    key, data_root, group, stats = arguments
    unique_currents = list(dict.fromkeys(group["currents"]))
    rows = read_causal_episode_group(
        Path(data_root), group["task"], group["episode"], unique_currents, stats
    )
    # Avoid torch.multiprocessing file-descriptor sharing after a short-lived
    # ProcessPool worker exits. NumPy arrays use ordinary pickle transport and
    # are converted back in the parent without changing values or dtypes.
    return key, {
        current: {name: value.numpy() for name, value in row.items()}
        for current, row in rows.items()
    }


def stack(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r11-cache", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--heartbeat", required=True)
    parser.add_argument("--train-interior-per-episode", type=int, default=100)
    parser.add_argument("--validation-interior-per-episode", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if (
        args.train_interior_per_episode < 1
        or args.validation_interior_per_episode < 1
        or args.workers < 1
    ):
        raise ValueError("dense R12 interior windows per episode must be positive")
    output = Path(args.output)
    if output.exists():
        payload = torch.load(output, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata", {})
        if (
            payload.get("round") != "R12"
            or metadata.get("protocol_variant") != PROTOCOL_VARIANT
            or metadata.get("action_history_lag") != 1
            or metadata.get("cold_start_steps") != list(COLD_START_STEPS)
            or metadata.get("parent_normalization_checkpoint_sha256")
            != EXPECTED_PARENT_SHA256
            or metadata.get("train_interior_per_episode") != args.train_interior_per_episode
            or metadata.get("validation_interior_per_episode")
            != args.validation_interior_per_episode
            or metadata.get("history_robustification")
            != "deterministic_clean_zero_noise_lag_mixture_in_trainer"
        ):
            raise ValueError("existing action cache identity differs")
        print(json.dumps({"reused": str(output), "sha256": sha256(output)}))
        return
    state, heartbeat = Path(args.state), Path(args.heartbeat)
    atomic_json(state, {"state": "PREPARING", "stage": "action_targets", "updated_at": now()})
    atomic_json(heartbeat, {"producer": "prepare_r12_action_cache", "updated_at": now()})
    parent_path = Path(args.parent_checkpoint).resolve(strict=True)
    if sha256(parent_path) != EXPECTED_PARENT_SHA256:
        raise ValueError("W10 normalization source checkpoint hash differs")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in parent["stats"].items()}
    r11_path = Path(args.r11_cache).resolve(strict=True)
    r11 = torch.load(r11_path, map_location="cpu", weights_only=False)
    metadata = r11["metadata"]
    if metadata["seed"] != 20260805 or metadata["history_steps"] != 3:
        raise ValueError("R11 cache selection protocol differs")
    data_root = Path(args.data_root).resolve(strict=True)
    receipt = manifest_receipt(data_root)
    if metadata.get("manifest_sha256") != receipt:
        raise ValueError("R11 cache and R12 dataset manifests differ")
    manifests = {
        task: json.loads((data_root / task / "training_manifest.json").read_text(encoding="utf-8"))
        for task in TASKS
    }
    result_splits = {}
    last_beat = time.monotonic()
    per_episode_by_split = {
        "train": args.train_interior_per_episode,
        "validation": args.validation_interior_per_episode,
    }
    for split in ("train", "validation"):
        examples = choose_examples(
            manifests, split, per_episode_by_split[split], int(metadata["seed"])
        )
        groups = {}
        for task, episode, current in examples:
            key = (task, episode["hdf5_path"])
            groups.setdefault(key, {"task": task, "episode": episode, "currents": []})[
                "currents"
            ].append(current)
        prepared = {}
        completed = 0
        jobs = [(key, str(data_root), group, stats) for key, group in groups.items()]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for key, result in executor.map(read_causal_episode_group_job, jobs):
                prepared[key] = {
                    current: {
                        name: torch.from_numpy(value) for name, value in row.items()
                    }
                    for current, row in result.items()
                }
                completed += len(groups[key]["currents"])
                if time.monotonic() - last_beat >= 20:
                    atomic_json(
                        heartbeat,
                        {"producer": "prepare_r12_action_cache", "split": split, "row": completed, "total": len(examples), "workers": args.workers, "updated_at": now()},
                    )
                    last_beat = time.monotonic()
        rows = [
            prepared[(task, episode["hdf5_path"])][current]
            for task, episode, current in examples
        ]
        result_splits[split] = stack(rows)
    payload = {
        "schema_version": 1,
        "round": "R12",
        "metadata": {
            "created_at": now(),
            "protocol_variant": PROTOCOL_VARIANT,
            "action_history_lag": 1,
            "target_starts_at_current_step": True,
            "cold_start_steps": list(COLD_START_STEPS),
            "cold_start_padding": "repeat_first_observation_and_zero_previous_action",
            "r11_cache": str(r11_path),
            "r11_cache_sha256": sha256(r11_path),
            "parent_normalization_checkpoint": str(parent_path),
            "parent_normalization_checkpoint_sha256": EXPECTED_PARENT_SHA256,
            "data_root": str(data_root),
            "manifest_sha256": receipt,
            "tasks": TASKS,
            "seed": 20260805,
            "history_steps": 3,
            "horizon": 100,
            "max_agents": 4,
            "action_dim": 8,
            "train_interior_per_episode": args.train_interior_per_episode,
            "validation_interior_per_episode": args.validation_interior_per_episode,
            "dense_train_interior_windows": sum(
                1
                for task in TASKS
                for episode in manifests[task]["episodes"]
                if episode["split"] == "train"
                for _ in range(args.train_interior_per_episode)
            ),
            "dense_validation_interior_windows": sum(
                1
                for task in TASKS
                for episode in manifests[task]["episodes"]
                if episode["split"] == "validation"
                for _ in range(args.validation_interior_per_episode)
            ),
            "history_robustification": "deterministic_clean_zero_noise_lag_mixture_in_trainer",
            "train_windows": len(result_splits["train"]["visual"]),
            "validation_windows": len(result_splits["validation"]["visual"]),
            "legal_inputs": metadata["legal_inputs"],
            "forbidden_inputs": metadata["forbidden_inputs"],
        },
        "stats": {key: torch.from_numpy(value) for key, value in stats.items()},
        **result_splits,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    receipt = {"state": "PASSED", "stage": "complete", "output": str(output), "sha256": sha256(output), "updated_at": now()}
    atomic_json(state, receipt)
    atomic_json(heartbeat, {"producer": "prepare_r12_action_cache", "updated_at": now(), "complete": True})
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
