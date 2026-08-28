"""Streaming decentralized RoboFactory HDF5 dataset for openpi.

The map index contains every timestep of every episode and every local Panda.
Each item returns one RGB image, its own 9-D qpos and its own 8-D commanded
joint-position action chunk; peer/global fields are never opened.
"""
from __future__ import annotations

import json
import os
import bisect
from pathlib import Path
from typing import SupportsIndex

import h5py
import numpy as np


class RoboFactoryDataset:
    def __init__(self, root: str | Path = "/workspace/datasets/robofactory_multitask", action_horizon: int = 16):
        self.root = Path(os.environ.get("OPENPI_ROBOFACTORY_ROOT", root))
        self.action_horizon = int(action_horizon)
        self.streams: list[tuple[str, int, int, str]] = []
        self.cumulative: list[int] = []
        self._lengths: dict[str, int] = {}
        self._cache: dict[tuple[str, int], h5py.File] = {}
        self._task_text: dict[str, str] = {}
        expected_tasks = 0
        expected_episodes = 0
        found_episodes = 0
        allow_incomplete = os.environ.get("OPENPI_ALLOW_INCOMPLETE_DATASET") == "1"
        for task_dir in sorted(self.root.iterdir()):
            manifest_path = task_dir / "training_manifest.json"
            if not manifest_path.is_file():
                continue
            expected_tasks += 1
            manifest = json.loads(manifest_path.read_text())
            # Split labels are intentionally ignored: all 150 are training.
            for ep in manifest.get("episodes", []):
                expected_episodes += 1
                path = task_dir / ep["hdf5_path"]
                if not path.is_file():
                    continue
                path_s = str(path)
                with h5py.File(path, "r") as h:
                    n = int(h["data/observation/agents/panda_0/qpos"].shape[0])
                    done = np.asarray(h["data/done"][:n], bool)
                    end = np.flatnonzero(done)
                    if len(end):
                        n = int(end[0] + 1)
                    agent_ids = sorted(int(k.rsplit("_", 1)[1]) for k in h["data/observation/agents"])
                    value = h["data/task/text"][0]
                    text = value.decode() if isinstance(value, bytes) else str(value)
                self._lengths[path_s] = n
                self._task_text[task_dir.name] = text
                found_episodes += 1
                for agent in agent_ids:
                    self.streams.append((path_s, agent, n, task_dir.name))
                    self.cumulative.append((self.cumulative[-1] if self.cumulative else 0) + n)
        if not self.streams:
            raise RuntimeError(f"No HDF5 items found under {self.root}")
        if not allow_incomplete and (expected_tasks != 6 or expected_episodes != 900 or found_episodes != 900):
            raise RuntimeError(
                f"Formal RoboFactory training requires 6 tasks x 150 episodes; "
                f"found {expected_tasks} tasks and {found_episodes}/{expected_episodes} episode files"
            )

    def __len__(self) -> int:
        return self.cumulative[-1]

    def _file(self, path: str) -> h5py.File:
        key = (path, os.getpid())
        if key not in self._cache:
            self._cache[key] = h5py.File(path, "r")
        return self._cache[key]

    def __getitem__(self, index: SupportsIndex) -> dict:
        flat_index = index.__index__()
        if flat_index < 0:
            flat_index += len(self)
        if flat_index < 0 or flat_index >= len(self):
            raise IndexError(flat_index)
        stream = bisect.bisect_right(self.cumulative, flat_index)
        start = self.cumulative[stream - 1] if stream else 0
        path, agent, _, task = self.streams[stream]
        t = flat_index - start
        h = self._file(path)
        n = self._lengths[path]
        image = np.asarray(h[f"data/observation/images/agent_{agent}"][t], np.uint8)
        state = np.asarray(h[f"data/observation/agents/panda_{agent}/qpos"][t], np.float32)
        actions = np.asarray(h[f"data/action/agents/panda_{agent}/commanded"][t:min(t + self.action_horizon, n)], np.float32)
        if len(actions) < self.action_horizon:
            actions = np.concatenate([actions, np.repeat(actions[-1:], self.action_horizon - len(actions), axis=0)], axis=0)
        return {
            "observation/image": image,
            "observation/state": state,
            "actions": actions,
            "prompt": self._task_text[task],
        }

    def close(self) -> None:
        for handle in self._cache.values():
            handle.close()
        self._cache.clear()
