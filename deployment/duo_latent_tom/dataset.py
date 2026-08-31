from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .common import DATASET_REVISION, TASKS


class DuoLatentToMDataset(Dataset):
    """Every causal local-arm decision in all 550 DuoBench demonstrations."""

    def __init__(self, root: str | Path, *, obs_steps: int = 2, horizon: int = 40):
        self.root = Path(root)
        self.obs_steps = int(obs_steps)
        self.horizon = int(horizon)
        if self.obs_steps != 2 or self.horizon <= 0:
            raise ValueError("DuoBench LatentToM requires two observations and a positive horizon")
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("dataset_revision") != DATASET_REVISION:
            raise ValueError("prepared DuoBench dataset revision drift")
        if self.manifest.get("recording_alignment", {}).get("action_lag_rows") != 1:
            raise ValueError("DuoBench LatentToM requires causal lag-1 targets")
        norm = self.manifest["normalization"]
        self.stats = {
            "schema": "duobench.latent-tom.normalization.v1",
            "qpos": {
                "mean": list(norm["qpos_mean"]), "std": list(norm["qpos_std"]),
            },
            "action": {
                "mean": list(norm["action_mean"]), "std": list(norm["action_std"]),
            },
            "population": norm["population"],
            "action_lag_rows": 1,
        }
        self.task_data: list[dict[str, np.ndarray]] = []
        self.rows: list[tuple[int, int, int, int, int]] = []
        self.task_indices: list[list[int]] = [[] for _ in TASKS]
        self.streams = defaultdict(list)
        for task_id, task in enumerate(TASKS):
            task_root = self.root / task
            data = {
                key: np.load(task_root / f"{key}.npy", mmap_mode="r")
                for key in ("state", "action", "head", "left", "right", "episodes")
            }
            episodes = np.asarray(data["episodes"])
            starts = np.flatnonzero(np.r_[True, episodes[1:] != episodes[:-1]])
            ends = np.r_[starts[1:], len(episodes)]
            if len(starts) != 50:
                raise ValueError(f"{task}: expected 50 episodes, found {len(starts)}")
            self.task_data.append(data)
            for start, end in zip(starts, ends, strict=True):
                if int(end) - int(start) < 2:
                    raise ValueError(f"{task}: episode is too short for causal lag-1")
                for arm in (0, 1):
                    stream = []
                    for current in range(int(start), int(end) - 1):
                        index = len(self.rows)
                        self.rows.append((task_id, arm, current, int(start), int(end)))
                        self.task_indices[task_id].append(index)
                        stream.append(index)
                    self.streams[(task_id, int(episodes[start]), arm)] = stream
        expected = 2 * int(self.manifest["total_policy_samples"])
        if len(self.rows) != expected or expected != 570876:
            raise ValueError(f"indexed local causal sample count drift: {len(self.rows)} != {expected}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        task_id, arm, current, episode_start, episode_end = self.rows[int(index)]
        data = self.task_data[task_id]
        state = data["state"].reshape(-1, 2, 8)
        action = data["action"].reshape(-1, 2, 8)
        obs_start = max(episode_start, current - self.obs_steps + 1)
        qpos = np.asarray(state[obs_start:current + 1, arm], dtype=np.float32)
        head = np.asarray(data["head"][obs_start:current + 1], dtype=np.uint8)
        wrist_key = "left" if arm == 0 else "right"
        wrist = np.asarray(data[wrist_key][obs_start:current + 1], dtype=np.uint8)
        images = np.concatenate((head, wrist), axis=2)
        if len(images) < self.obs_steps:
            pad = self.obs_steps - len(images)
            images = np.concatenate((np.repeat(images[:1], pad, axis=0), images))
            qpos = np.concatenate((np.repeat(qpos[:1], pad, axis=0), qpos))

        target_start = current + 1
        target_end = min(episode_end, target_start + self.horizon)
        actions = np.asarray(action[target_start:target_end, arm], dtype=np.float32)
        valid = len(actions)
        if valid <= 0:
            raise RuntimeError("causal dataset exposed an empty action target")
        if valid < self.horizon:
            actions = np.concatenate((actions, np.repeat(actions[-1:], self.horizon - valid, axis=0)))
        mask = np.zeros((self.horizon,), dtype=np.float32)
        mask[:valid] = 1.0
        task = np.zeros((len(TASKS),), dtype=np.float32)
        task[task_id] = 1.0
        return {
            "image": torch.from_numpy(np.moveaxis(images, -1, 1).copy()),
            "qpos": torch.from_numpy(np.ascontiguousarray(qpos)),
            "task": torch.from_numpy(task),
            "action": torch.from_numpy(np.ascontiguousarray(actions)),
            "action_mask": torch.from_numpy(mask),
        }


class TaskBalancedBatchSampler(Sampler[list[int]]):
    """Uniform task sampling with replacement over each task's local decisions."""

    def __init__(self, rows: list[list[int]], batch_size: int, updates: int, seed: int):
        self.rows = rows
        self.batch_size = int(batch_size)
        self.updates = int(updates)
        self.seed = int(seed)
        if not self.rows or self.batch_size <= 0 or self.updates < 0:
            raise ValueError("invalid task-balanced sampler arguments")

    def __len__(self) -> int:
        return self.updates

    def __iter__(self):
        rng = random.Random(self.seed)
        task_count = len(self.rows)
        base, extra = divmod(self.batch_size, task_count)
        for _ in range(self.updates):
            batch = []
            offset = rng.randrange(task_count)
            for task_id, task_rows in enumerate(self.rows):
                count = base + int((task_id - offset) % task_count < extra)
                batch.extend(rng.choices(task_rows, k=count))
            rng.shuffle(batch)
            yield batch
