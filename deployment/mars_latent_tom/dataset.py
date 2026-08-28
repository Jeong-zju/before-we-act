from __future__ import annotations

import bisect
import glob
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .common import TASKS, atomic_json


def index_corpus(root: str | Path, stats_path: str | Path | None = None):
    root = Path(root)
    streams = [[] for _ in TASKS]
    moments = {key: None for key in ("q_sum", "q_sq", "a_sum", "a_sq")}
    q_min = q_max = a_min = a_max = None
    count = episodes = local_streams = timesteps = 0
    for task_id, spec in enumerate(TASKS):
        task_episodes = 0
        paths = sorted(glob.glob(str(root / spec.name / "motionplanning" / "*.shard*.h5")))
        if len(paths) != 10:
            raise RuntimeError(f"{spec.name}: expected 10 shards, found {len(paths)}")
        for path in paths:
            with h5py.File(path, "r") as handle:
                for trajectory in sorted(handle, key=lambda name: int(name.rsplit("_", 1)[-1])):
                    group = handle[trajectory]
                    if not bool(np.asarray(group["success"])[-1]):
                        raise RuntimeError(f"non-success trajectory: {path}:{trajectory}")
                    n = min(len(group[f"actions/panda-{arm}"]) for arm in range(spec.arms))
                    task_episodes += 1
                    episodes += 1
                    for arm in range(spec.arms):
                        qpos = np.asarray(group[f"obs/agent/panda-{arm}/qpos"][:n], np.float64)
                        action = np.asarray(group[f"actions/panda-{arm}"][:n], np.float64)
                        image = group[f"obs/sensor_data/head_camera_agent{arm}/rgb"]
                        if qpos.shape != (n, 9) or action.shape != (n, 8):
                            raise RuntimeError(f"local state/action contract mismatch: {path}:{trajectory}:arm{arm}")
                        if image.shape[0] < n or tuple(image.shape[1:]) != (240, 320, 3) or image.dtype != np.uint8:
                            raise RuntimeError(f"RGB contract mismatch: {path}:{trajectory}:arm{arm}")
                        streams[task_id].append((path, trajectory, arm, n))
                        local_streams += 1
                        timesteps += n
                        count += n
                        for key, value in (("q_sum", qpos.sum(0)), ("q_sq", np.square(qpos).sum(0)),
                                           ("a_sum", action.sum(0)), ("a_sq", np.square(action).sum(0))):
                            moments[key] = value if moments[key] is None else moments[key] + value
                        q_min = qpos.min(0) if q_min is None else np.minimum(q_min, qpos.min(0))
                        q_max = qpos.max(0) if q_max is None else np.maximum(q_max, qpos.max(0))
                        a_min = action.min(0) if a_min is None else np.minimum(a_min, action.min(0))
                        a_max = action.max(0) if a_max is None else np.maximum(a_max, action.max(0))
        if task_episodes != 150:
            raise RuntimeError(f"{spec.name}: expected 150 episodes, found {task_episodes}")
    q_mean = moments["q_sum"] / count
    a_mean = moments["a_sum"] / count
    q_std = np.sqrt(np.maximum(moments["q_sq"] / count - np.square(q_mean), 0.0)).clip(1e-4)
    a_std = np.sqrt(np.maximum(moments["a_sq"] / count - np.square(a_mean), 0.0)).clip(1e-4)
    stats = {
        "schema": "mars-control.latent-tom.normalization.v1",
        "status": "complete",
        "episodes": episodes,
        "local_streams": local_streams,
        "indexed_local_timesteps": timesteps,
        "all_data_no_split": True,
        "qpos": {"mean": q_mean.tolist(), "std": q_std.tolist(), "min": q_min.tolist(), "max": q_max.tolist()},
        "action": {"mean": a_mean.tolist(), "std": a_std.tolist(), "min": a_min.tolist(), "max": a_max.tolist()},
        "rgb": {"dtype": "uint8", "shape_hwc": [240, 320, 3], "train_transform": "float32_div_255"},
    }
    if episodes != 600 or local_streams != 1650 or timesteps != 1035318:
        raise RuntimeError(f"corpus count drift: episodes={episodes}, streams={local_streams}, timesteps={timesteps}")
    if stats_path:
        atomic_json(stats_path, stats)
    return streams, stats


class MarsLatentToMDataset(Dataset):
    """Every valid local arm/time pair, without any peer or task-id input."""

    def __init__(self, root: str | Path, stats_path: str | Path, obs_steps: int = 2, horizon: int = 40):
        self.obs_steps = int(obs_steps)
        self.horizon = int(horizon)
        self.handles = {}
        self.streams, self.stats = index_corpus(root, stats_path)
        self.entries = []
        self.task_indices = [[] for _ in TASKS]
        for task_id, task_streams in enumerate(self.streams):
            for stream in task_streams:
                for current in range(stream[3]):
                    index = len(self.entries)
                    self.entries.append((*stream, current))
                    self.task_indices[task_id].append(index)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["handles"] = {}
        return state

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path, trajectory, arm, n, current = self.entries[index]
        if path not in self.handles:
            self.handles[path] = h5py.File(path, "r", libver="latest", swmr=True)
        group = self.handles[path][trajectory]
        obs_start = max(0, current - self.obs_steps + 1)
        valid = max(0, min(self.horizon, n - current))
        images = np.asarray(group[f"obs/sensor_data/head_camera_agent{arm}/rgb"][obs_start:current + 1], np.uint8)
        qpos = np.asarray(group[f"obs/agent/panda-{arm}/qpos"][obs_start:current + 1], np.float32)
        if len(images) < self.obs_steps:
            images = np.concatenate([np.repeat(images[:1], self.obs_steps - len(images), axis=0), images])
            qpos = np.concatenate([np.repeat(qpos[:1], self.obs_steps - len(qpos), axis=0), qpos])
        action = np.asarray(group[f"actions/panda-{arm}"][current:current + valid], np.float32)
        if len(action) == 0:
            action = np.asarray(group[f"actions/panda-{arm}"][n - 1:n], np.float32)
        if len(action) < self.horizon:
            action = np.concatenate([action, np.repeat(action[-1:], self.horizon - len(action), axis=0)])
        mask = np.zeros((self.horizon,), np.float32)
        mask[:valid] = 1.0
        return {
            "image": torch.from_numpy(images).permute(0, 3, 1, 2),
            "qpos": torch.from_numpy(qpos),
            "action": torch.from_numpy(action),
            "action_mask": torch.from_numpy(mask),
        }


class TaskBalancedBatchSampler(Sampler):
    def __init__(self, task_indices, batch_size: int, updates: int, seed: int):
        if batch_size % len(task_indices):
            raise ValueError("effective batch must divide evenly across four tasks")
        self.rows = task_indices
        self.batch_size = int(batch_size)
        self.updates = int(updates)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return self.updates

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        per_task = self.batch_size // len(self.rows)
        for _ in range(self.updates):
            batch = []
            for rows in self.rows:
                batch.extend(rng.choices(rows, k=per_task))
            rng.shuffle(batch)
            yield batch
