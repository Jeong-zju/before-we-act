"""Lazy full-episode windows for post-R12-R3 action-generator repair rounds.

The earlier dense cache sampled a fixed number of interior rows per episode.
This dataset instead exposes every legal timestep from immutable, per-episode
feature shards.  Targets are assembled lazily, so the spatial cache is stored
only once and no monolithic consolidation copy is required.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path
import random
from typing import Mapping, Sequence

import torch
from torch.utils.data import Dataset, Sampler

from .raw_team_windows import TASKS


FULL_EPISODE_PROTOCOL = "r12_full_episode_rectangular_dinov3_6x8_v1"


def _validate_episode(payload: Mapping[str, object], path: Path) -> int:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"full-episode shard has no metadata: {path}")
    if (
        payload.get("schema_version") != 1
        or payload.get("round") != "R12-R4"
        or metadata.get("protocol_variant") != FULL_EPISODE_PROTOCOL
    ):
        raise ValueError(f"unsupported full-episode shard: {path}")
    steps = int(metadata.get("steps", 0))
    if steps < 1:
        raise ValueError(f"full-episode shard has no steps: {path}")
    expected = {
        "visual": (steps, 16, 15),
        "view_mask": (steps, 5),
        "qpos": (steps, 4, 9),
        "executed_actions": (steps, 4, 8),
        "commanded_actions": (steps, 4, 8),
        "agent_mask": (4,),
        "spatial_tokens": (steps, 5, 48, 768),
        "spatial_view_mask": (steps, 5),
    }
    for key, shape in expected.items():
        value = payload.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(
                f"full-episode tensor {key} differs at {path}: "
                f"{None if not isinstance(value, torch.Tensor) else tuple(value.shape)} != {shape}"
            )
    if not torch.equal(
        payload["view_mask"].bool(), payload["spatial_view_mask"].bool()
    ):
        raise ValueError(f"coarse/spatial view masks differ at {path}")
    if not bool(torch.isfinite(payload["spatial_tokens"]).all()):
        raise ValueError(f"non-finite spatial tokens at {path}")
    return steps


class FullEpisodeActionWindows(Dataset):
    """Every causal action window from hash-audited per-episode shards.

    ``__getitem__`` accepts ``(episode_index, timestep)`` requests emitted by
    :class:`ExactFiveTaskFullEpisodeSampler`.  This avoids a large in-memory
    table of repeated paths and allows deterministic task-balanced sampling.
    """

    def __init__(
        self,
        episodes: Sequence[Mapping[str, object]],
        stats: Mapping[str, torch.Tensor],
        *,
        split: str,
        horizon: int = 100,
        history: int = 3,
        cache_episodes: int = 1,
    ) -> None:
        if split not in ("train", "validation"):
            raise ValueError("full-episode split must be train or validation")
        if horizon < 1 or history < 1 or cache_episodes < 1:
            raise ValueError("horizon/history/cache_episodes must be positive")
        self.split = split
        self.horizon = int(horizon)
        self.history = int(history)
        self.cache_episodes = int(cache_episodes)
        self.stats = {
            "a_mean": torch.as_tensor(stats["a_mean"], dtype=torch.float32),
            "a_std": torch.as_tensor(stats["a_std"], dtype=torch.float32),
        }
        if tuple(self.stats["a_mean"].shape) != (8,) or tuple(
            self.stats["a_std"].shape
        ) != (8,):
            raise ValueError("full-episode action statistics must be per-agent [8]")
        if not bool((self.stats["a_std"] > 0).all()):
            raise ValueError("full-episode action std must be positive")

        self.episodes: list[dict[str, object]] = []
        self.by_task: dict[int, list[int]] = defaultdict(list)
        self.requests_by_task: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self.total_steps = 0
        for episode in episodes:
            row = dict(episode)
            if row.get("split") != split:
                continue
            task = str(row.get("task"))
            if task not in TASKS:
                raise ValueError(f"unsupported full-episode task {task!r}")
            path = Path(str(row.get("path", ""))).resolve(strict=True)
            steps = int(row.get("steps", 0))
            if steps < 1:
                raise ValueError(f"invalid full-episode step count for {path}")
            index = len(self.episodes)
            row.update(path=str(path), task_index=TASKS.index(task), steps=steps)
            self.episodes.append(row)
            self.by_task[TASKS.index(task)].append(index)
            self.requests_by_task[TASKS.index(task)].extend(
                (index, timestep) for timestep in range(steps)
            )
            self.total_steps += steps
        if set(self.by_task) != set(range(len(TASKS))):
            raise ValueError("full-episode index must cover all five task buckets")
        self._cache: OrderedDict[int, Mapping[str, object]] = OrderedDict()

    def __len__(self) -> int:
        return self.total_steps

    def _load(self, episode_index: int) -> Mapping[str, object]:
        if episode_index in self._cache:
            payload = self._cache.pop(episode_index)
            self._cache[episode_index] = payload
            return payload
        row = self.episodes[episode_index]
        path = Path(str(row["path"]))
        payload = torch.load(path, map_location="cpu", weights_only=False)
        observed_steps = _validate_episode(payload, path)
        if observed_steps != int(row["steps"]):
            raise ValueError(f"full-episode index/shard step count differs at {path}")
        metadata = payload["metadata"]
        if (
            metadata.get("task") != row["task"]
            or metadata.get("split") != self.split
            or int(metadata.get("seed", -1)) != int(row["seed"])
            or metadata.get("hdf5_sha256") != row.get("hdf5_sha256")
        ):
            raise ValueError(f"full-episode index/shard identity differs at {path}")
        self._cache[episode_index] = payload
        while len(self._cache) > self.cache_episodes:
            self._cache.popitem(last=False)
        return payload

    def __getitem__(self, request) -> dict[str, torch.Tensor]:
        if not isinstance(request, (tuple, list)) or len(request) != 2:
            raise TypeError("full-episode request must be (episode_index, timestep)")
        episode_index, current = map(int, request)
        row = self.episodes[episode_index]
        steps = int(row["steps"])
        if not 0 <= current < steps:
            raise IndexError((episode_index, current))
        episode = self._load(episode_index)
        history_indices = [
            max(0, current - offset)
            for offset in range(self.history - 1, -1, -1)
        ]
        indices = torch.tensor(history_indices, dtype=torch.long)
        action_indices = torch.tensor(
            [max(0, value - 1) for value in history_indices], dtype=torch.long
        )
        actions = episode["executed_actions"].index_select(0, action_indices).clone()
        cold = torch.tensor([value == 0 for value in history_indices], dtype=torch.bool)
        actions[cold] = 0

        end = min(current + self.horizon, steps)
        valid = end - current
        target = torch.empty((self.horizon, 4, 8), dtype=torch.float32)
        commanded = episode["commanded_actions"][current:end].float()
        target[:valid] = commanded
        target[valid:] = commanded[-1]
        target = (target - self.stats["a_mean"][None, None]) / self.stats[
            "a_std"
        ][None, None]
        step_mask = torch.zeros(self.horizon, dtype=torch.bool)
        step_mask[:valid] = True
        agent_mask = episode["agent_mask"].bool()
        return {
            "visual": episode["visual"].index_select(0, indices).float(),
            "view_mask": episode["view_mask"].index_select(0, indices).float(),
            "qpos": episode["qpos"].index_select(0, indices).float(),
            "actions": actions.float(),
            "agent_mask": agent_mask,
            "task_index": torch.tensor(int(row["task_index"]), dtype=torch.long),
            "joint_actions": target,
            "action_step_mask": step_mask,
            "spatial_tokens": episode["spatial_tokens"][current].float(),
            "spatial_view_mask": episode["spatial_view_mask"][current].bool(),
            # Retain source identity in every collated batch.  Recovery rows use
            # one, so the trainer can exempt genuine student histories.
            "source_index": torch.tensor(0, dtype=torch.long),
        }


class ExactFiveTaskFullEpisodeSampler(Sampler[list[tuple[int, int]]]):
    """Deterministic balanced cycles that guarantee every timestep is seen."""

    def __init__(
        self,
        dataset: FullEpisodeActionWindows,
        *,
        updates: int,
        rows_per_task: int,
        seed: int,
        start_update: int = 0,
    ) -> None:
        if updates <= start_update or rows_per_task < 1:
            raise ValueError("invalid full-episode sampler budget")
        self.dataset = dataset
        self.updates = int(updates)
        self.rows_per_task = int(rows_per_task)
        self.seed = int(seed)
        self.start_update = int(start_update)

    def __len__(self) -> int:
        return self.updates - self.start_update

    def __iter__(self):
        permutations: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for update in range(self.start_update + 1, self.updates + 1):
            rng = random.Random(self.seed + 1_000_003 * update)
            batch: list[tuple[int, int]] = []
            for task_index in range(len(TASKS)):
                bucket = self.dataset.requests_by_task[task_index]
                for within_update in range(self.rows_per_task):
                    draw = (update - 1) * self.rows_per_task + within_update
                    epoch, offset = divmod(draw, len(bucket))
                    key = (task_index, epoch)
                    if key not in permutations:
                        permutation = list(bucket)
                        random.Random(
                            self.seed + 10_000_019 * task_index + 1_000_000_007 * epoch
                        ).shuffle(permutation)
                        permutations[key] = permutation
                    batch.append(permutations[key][offset])
            rng.shuffle(batch)
            yield batch


__all__ = [
    "ExactFiveTaskFullEpisodeSampler",
    "FULL_EPISODE_PROTOCOL",
    "FullEpisodeActionWindows",
]
