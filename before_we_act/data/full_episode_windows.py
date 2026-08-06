"""Lazy full-timestep windows from native-resolution visual feature shards.

Every cached spatial row is produced by first encoding the complete current
480x640 fixed-view RGB image with frozen DINOv3-B/16.  The native 30x40 patch
grid is compressed to 6x8 only *after* the visual backbone.  This keeps raw
high-resolution images as the deployed policy's primary observation while
making full-dataset training storage and I/O tractable.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
import json
from pathlib import Path
import random
from typing import Mapping, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .raw_team_windows import TASKS


FULL_EPISODE_PROTOCOL = (
    "r12_full_episode_native_480x640_dinov3_30x40_to_6x8_v2"
)


def _metadata(handle: h5py.File, path: Path) -> dict[str, object]:
    try:
        value = json.loads(str(handle.attrs["metadata_json"]))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"full-episode shard metadata is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"full-episode shard metadata is not an object: {path}")
    return value


class _HDF5Episode:
    """One bounded read-only feature-shard handle per DataLoader worker."""

    def __init__(
        self,
        path: Path,
        row: Mapping[str, object],
        *,
        split: str,
    ) -> None:
        self.path = path
        self.handle = h5py.File(path, "r")
        self.metadata = _metadata(self.handle, path)
        if (
            int(self.handle.attrs.get("schema_version", -1)) != 1
            or str(self.handle.attrs.get("round", "")) != "R12-R4"
            or self.metadata.get("protocol_variant") != FULL_EPISODE_PROTOCOL
        ):
            self.close()
            raise ValueError(f"unsupported full-episode shard: {path}")
        steps = int(self.metadata.get("steps", 0))
        expected_identity = {
            "task": row["task"],
            "split": split,
            "seed": int(row["seed"]),
            "hdf5_sha256": row.get("hdf5_sha256"),
        }
        observed_identity = {
            "task": self.metadata.get("task"),
            "split": self.metadata.get("split"),
            "seed": int(self.metadata.get("seed", -1)),
            "hdf5_sha256": self.metadata.get("hdf5_sha256"),
        }
        if steps != int(row["steps"]) or observed_identity != expected_identity:
            self.close()
            raise ValueError(f"full-episode index/shard identity differs: {path}")
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
            if key not in self.handle or tuple(self.handle[key].shape) != shape:
                observed = None if key not in self.handle else tuple(self.handle[key].shape)
                self.close()
                raise ValueError(
                    f"full-episode tensor {key} differs at {path}: "
                    f"{observed} != {shape}"
                )

    def close(self) -> None:
        if getattr(self, "handle", None) is not None:
            self.handle.close()

    def rows(self, key: str, indices: Sequence[int]) -> torch.Tensor:
        """Read possibly repeated causal rows without HDF5 fancy-index errors."""

        requested = np.asarray(tuple(int(value) for value in indices), dtype=np.int64)
        unique, inverse = np.unique(requested, return_inverse=True)
        values = np.asarray(self.handle[key][unique.tolist()])[inverse]
        return torch.from_numpy(values)

    def row(self, key: str, index: int) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.handle[key][int(index)]))


class FullEpisodeActionWindows(Dataset):
    """Every causal action window from hash-audited episode feature shards."""

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
            source_digest = str(row.get("hdf5_sha256", ""))
            if len(source_digest) < 8:
                raise ValueError(f"missing immutable source HDF5 digest for {path}")
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
        self._cache: OrderedDict[int, _HDF5Episode] = OrderedDict()

    def __len__(self) -> int:
        return self.total_steps

    def _load(self, episode_index: int) -> _HDF5Episode:
        if episode_index in self._cache:
            episode = self._cache.pop(episode_index)
            self._cache[episode_index] = episode
            return episode
        row = self.episodes[episode_index]
        episode = _HDF5Episode(
            Path(str(row["path"])), row, split=self.split
        )
        self._cache[episode_index] = episode
        while len(self._cache) > self.cache_episodes:
            _index, evicted = self._cache.popitem(last=False)
            evicted.close()
        return episode

    def __del__(self) -> None:
        for episode in getattr(self, "_cache", {}).values():
            episode.close()

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
        action_indices = [max(0, value - 1) for value in history_indices]
        actions = episode.rows("executed_actions", action_indices).float()
        cold = torch.tensor(
            [value == 0 for value in history_indices], dtype=torch.bool
        )
        actions[cold] = 0

        end = min(current + self.horizon, steps)
        valid = end - current
        target = torch.empty((self.horizon, 4, 8), dtype=torch.float32)
        commanded = torch.from_numpy(
            np.asarray(episode.handle["commanded_actions"][current:end])
        ).float()
        target[:valid] = commanded
        target[valid:] = commanded[-1]
        target = (target - self.stats["a_mean"][None, None]) / self.stats[
            "a_std"
        ][None, None]
        step_mask = torch.zeros(self.horizon, dtype=torch.bool)
        step_mask[:valid] = True

        view_mask = episode.rows("view_mask", history_indices).float()
        spatial_view_mask = episode.row("spatial_view_mask", current).bool()
        if not torch.equal(view_mask[-1].bool(), spatial_view_mask):
            raise ValueError(f"coarse/spatial view masks differ at {episode.path}")
        spatial_tokens = episode.row("spatial_tokens", current).float()
        if not bool(torch.isfinite(spatial_tokens).all()):
            raise ValueError(f"non-finite spatial tokens at {episode.path}:{current}")
        return {
            "visual": episode.rows("visual", history_indices).float(),
            "view_mask": view_mask,
            "qpos": episode.rows("qpos", history_indices).float(),
            "actions": actions,
            "agent_mask": episode.row("agent_mask", 0).bool()
            if episode.handle["agent_mask"].ndim > 1
            else torch.from_numpy(np.asarray(episode.handle["agent_mask"])).bool(),
            "task_index": torch.tensor(int(row["task_index"]), dtype=torch.long),
            "joint_actions": target,
            "action_step_mask": step_mask,
            "spatial_tokens": spatial_tokens,
            "spatial_view_mask": spatial_view_mask,
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
                        epoch_rng = random.Random(
                            self.seed
                            + 10_000_019 * task_index
                            + 1_000_000_007 * epoch
                        )
                        blocks = [
                            list(bucket[start : start + 16])
                            for start in range(0, len(bucket), 16)
                        ]
                        for block in blocks:
                            epoch_rng.shuffle(block)
                        epoch_rng.shuffle(blocks)
                        permutations[key] = [
                            request for block in blocks for request in block
                        ]
                    batch.append(permutations[key][offset])
            rng.shuffle(batch)
            yield batch


class SequentialFullEpisodeSampler(Sampler[list[tuple[int, int]]]):
    """Visit every indexed timestep exactly once for full validation."""

    def __init__(self, dataset: FullEpisodeActionWindows, batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError("full-episode validation batch size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        batch: list[tuple[int, int]] = []
        for episode_index, episode in enumerate(self.dataset.episodes):
            for timestep in range(int(episode["steps"])):
                batch.append((episode_index, timestep))
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch


__all__ = [
    "ExactFiveTaskFullEpisodeSampler",
    "FULL_EPISODE_PROTOCOL",
    "FullEpisodeActionWindows",
    "SequentialFullEpisodeSampler",
]
