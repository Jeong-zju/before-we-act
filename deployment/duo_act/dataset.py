from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


TASKS = (
    "ball_maze", "bin_sort", "block_balance", "carry_pot", "hinge_chest", "join_blocks",
    "pour_marbles", "spring_door", "transfer_cube", "transfer_gate", "transfer_reorient",
)


class DuoACTDataset(Dataset):
    """Every frame of every demo, split into two independent local-arm streams."""

    def __init__(self, root: Path, horizon: int = 100, action_lag: int = 0):
        self.root = Path(root)
        self.horizon = int(horizon)
        self.action_lag = int(action_lag)
        if self.action_lag < 0:
            raise ValueError("action_lag must be non-negative")
        manifest = json.loads((self.root / "manifest.json").read_text())
        declared_lag = int(manifest.get("recording_alignment", {}).get("action_lag_rows", 0))
        if declared_lag != self.action_lag:
            raise ValueError(
                f"dataset declares action_lag={declared_lag}, requested {self.action_lag}"
            )
        norm = manifest["normalization"]
        self.q_mean = np.asarray(norm["qpos_mean"], np.float32)
        self.q_std = np.asarray(norm["qpos_std"], np.float32)
        self.a_mean = np.asarray(norm["action_mean"], np.float32)
        self.a_std = np.asarray(norm["action_std"], np.float32)
        self.task_data = []
        self.rows = []
        self.streams = defaultdict(list)
        self.by_task_stream = defaultdict(list)
        for task_id, task in enumerate(TASKS):
            task_root = self.root / task
            data = {
                key: np.load(task_root / f"{key}.npy", mmap_mode="r")
                for key in ("state", "action", "head", "left", "right", "episodes")
            }
            episodes = np.asarray(data["episodes"])
            changes = np.r_[True, episodes[1:] != episodes[:-1]]
            starts = np.flatnonzero(changes)
            ends = np.r_[starts[1:], len(episodes)]
            data["episode_start"] = np.repeat(starts, ends - starts)
            data["episode_end"] = np.repeat(ends, ends - starts)
            self.task_data.append(data)
            for start, end in zip(starts, ends, strict=True):
                for arm in (0, 1):
                    stream = []
                    sample_end = int(end) - self.action_lag
                    if sample_end <= int(start):
                        raise ValueError(
                            f"episode {int(episodes[start])} is too short for "
                            f"action_lag={self.action_lag}"
                        )
                    for step in range(int(start), sample_end):
                        index = len(self.rows)
                        self.rows.append((task_id, arm, step))
                        stream.append(index)
                    self.streams[(task_id, int(episodes[start]), arm)] = stream
                    self.by_task_stream[task_id].append(stream)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        task_id, arm, step = self.rows[index]
        data = self.task_data[task_id]
        state = data["state"].reshape(-1, 2, 8)
        action = data["action"].reshape(-1, 2, 8)
        qpos = np.asarray(state[step, arm], np.float32)
        episode_end = int(data["episode_end"][step])
        # RCS/LeRobot rows contain post-action observations: row i is the
        # observation after action i.  A positive lag trains the causal
        # policy contract (post-action observation -> next command) without
        # mutating the released source or changing episode lengths.
        target_start = step + self.action_lag
        end = min(episode_end, target_start + self.horizon)
        future = np.asarray(action[target_start:end, arm], np.float32)
        valid = len(future)
        padded = np.empty((self.horizon, 8), np.float32)
        padded[:valid] = future
        padded[valid:] = future[-1]
        mask = np.zeros(self.horizon, np.bool_)
        mask[:valid] = True
        # Shared head view plus only this arm's wrist view.
        image = np.concatenate(
            (np.asarray(data["head"][step]), np.asarray(data["left" if arm == 0 else "right"][step])), axis=1
        )
        return (
            torch.from_numpy(image.copy()).permute(2, 0, 1).contiguous(),
            torch.from_numpy((qpos - self.q_mean) / self.q_std),
            torch.tensor(task_id, dtype=torch.long),
            torch.from_numpy((padded - self.a_mean) / self.a_std),
            torch.from_numpy(mask),
        )


class TaskEpisodeBatchSampler(Sampler[list[int]]):
    """Equal task weighting, then uniform demo/arm/time sampling."""

    def __init__(self, dataset: DuoACTDataset, batch_size: int, updates: int, seed: int):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.updates = int(updates)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return self.updates

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.updates):
            batch = []
            for _ in range(self.batch_size):
                task = rng.randrange(len(TASKS))
                streams = self.dataset.by_task_stream[task]
                stream = streams[rng.randrange(len(streams))]
                batch.append(stream[rng.randrange(len(stream))])
            yield batch
