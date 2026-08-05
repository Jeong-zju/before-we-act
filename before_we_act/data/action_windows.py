from __future__ import annotations

from collections import defaultdict
import random
from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import Dataset, Sampler

from .raw_team_windows import TASKS


class CachedActionWindows(Dataset):
    """Causal legal inputs plus normalized future joint-action supervision."""

    def __init__(self, cache_path: str | Path, split: str) -> None:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 1 or payload.get("round") != "R12":
            raise ValueError("unsupported R12 action cache")
        if split not in ("train", "validation"):
            raise ValueError("R12 cache split must be train or validation")
        self.metadata = payload["metadata"]
        if (
            self.metadata.get("protocol_variant") != "causal_lag1_coldstart_dense_v2"
            or self.metadata.get("action_history_lag") != 1
            or self.metadata.get("cold_start_steps") != [0, 1, 2]
        ):
            raise ValueError("R12 cache is not the dense causal cold-start protocol")
        self.stats: Mapping[str, torch.Tensor] = payload["stats"]
        self.data: Mapping[str, torch.Tensor] = payload[split]
        size = int(self.data["visual"].shape[0])
        if not size or any(int(value.shape[0]) != size for value in self.data.values()):
            raise ValueError("R12 cache tensors have inconsistent lengths")
        if self.data["joint_actions"].shape[1:] != (100, 4, 8):
            raise ValueError("R12 joint action cache shape differs")

    def __len__(self) -> int:
        return int(self.data["visual"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.data.items()}


class ExactFiveTaskWindowSampler(Sampler[list[int]]):
    """Deterministic one-sample-per-task batches shared by all four routes."""

    def __init__(
        self,
        task_indices: torch.Tensor,
        updates: int,
        seed: int,
        start_update: int = 0,
    ) -> None:
        self.updates = int(updates)
        self.seed = int(seed)
        self.start_update = int(start_update)
        self.by_task: dict[int, list[int]] = defaultdict(list)
        for index, task in enumerate(task_indices.tolist()):
            self.by_task[int(task)].append(index)
        if set(self.by_task) != set(range(len(TASKS))):
            raise ValueError("R12 action cache must contain all five task buckets")

    def __len__(self) -> int:
        return self.updates - self.start_update

    def __iter__(self):
        for update in range(self.start_update + 1, self.updates + 1):
            rng = random.Random(self.seed + 1_000_003 * update)
            batch = [rng.choice(self.by_task[task]) for task in range(len(TASKS))]
            rng.shuffle(batch)
            yield batch
