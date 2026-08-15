"""Frozen data projection for Step 3-N1 raw-signal experiments.

The projection is built once from the audited Step-2 corpus.  Runtime inputs
contain only the legal 16-step history.  Synchronized teammate state and the
four future DINO anchors are returned as training targets, never as inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from before_we_act.step2_temporal_data import (
    EFFECTIVE_BATCH,
    HISTORY_STEPS,
    SAMPLES_PER_TASK,
    SIX_TASKS,
)


FUTURE_OFFSETS = (4, 8, 16, 32)
ACTION_PROBE_HORIZON = 16
CAPACITY_CANDIDATES = (4, 8, 16)


@dataclass(frozen=True)
class N1Episode:
    task: str
    task_index: int
    local_index: int
    offset: int
    length: int
    split: str
    episode_key: str
    hdf5_sha256: str


@dataclass(frozen=True)
class N1Request:
    episode_index: int
    arm: int
    time_index: int
    sample_key: str
    task: str


def load_n1_metadata(cache_root: str | Path) -> tuple[dict, list[N1Episode]]:
    root = Path(cache_root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != "before-we-act.b3-n1-cache/1":
        raise ValueError("unsupported 3-N1 cache format")
    episodes = [N1Episode(**row) for row in metadata["episodes"]]
    if len(episodes) != 720:
        raise ValueError("3-N1 cache must bind all 720 source episodes")
    return metadata, episodes


class N1RawSignalDataset(Dataset):
    """Memory-mapped N1 examples with a fail-closed input/target boundary."""

    RUNTIME_FIELDS = frozenset(
        {
            "history_visual",
            "history_qpos",
            "history_action",
            "history_mask",
            "action_history_mask",
            "task_index",
        }
    )
    TEACHER_TARGET_FIELDS = frozenset(
        {
            "future_visual",
            "current_target_visual",
            "future_mask",
            "teammate_qpos",
            "previous_teammate_qpos",
            "teammate_delta",
        }
    )
    PROBE_TARGET_FIELDS = frozenset({"action", "action_mask"})
    AUDIT_ONLY_FIELDS = frozenset(
        {
            "phase",
            "phase_bin",
            "episode_label",
            "task",
            "sample_key",
            "time_index",
            "agent_slot",
        }
    )

    def __init__(self, cache_root: str | Path) -> None:
        self.root = Path(cache_root)
        self.metadata, self.episodes = load_n1_metadata(self.root)
        stats = torch.load(self.root / "target_stats.pt", map_location="cpu", weights_only=False)
        self.visual_mean = torch.as_tensor(stats["visual_mean"], dtype=torch.float32)
        self.visual_std = torch.as_tensor(stats["visual_std"], dtype=torch.float32)
        self.q_mean = torch.as_tensor(stats["q_mean"], dtype=torch.float32)
        self.q_std = torch.as_tensor(stats["q_std"], dtype=torch.float32)
        self.a_mean = torch.as_tensor(stats["a_mean"], dtype=torch.float32)
        self.a_std = torch.as_tensor(stats["a_std"], dtype=torch.float32)
        if self.visual_mean.shape != (3, 768) or self.visual_std.shape != (3, 768):
            raise ValueError("3-N1 visual normalization contract differs")
        self._arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return sum(episode.length * 2 for episode in self.episodes)

    def _task_arrays(self, task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if task not in self._arrays:
            self._arrays[task] = (
                np.load(self.root / f"{task}_visual.npy", mmap_mode="r"),
                np.load(self.root / f"{task}_qpos.npy", mmap_mode="r"),
                np.load(self.root / f"{task}_action.npy", mmap_mode="r"),
            )
        return self._arrays[task]

    def __getitem__(self, request: N1Request | tuple) -> dict:
        if not isinstance(request, N1Request):
            request = N1Request(*request)
        episode = self.episodes[request.episode_index]
        if episode.task != request.task or request.arm not in (0, 1):
            raise ValueError("3-N1 request identity mismatch")
        if not 0 <= request.time_index < episode.length:
            raise IndexError(request.time_index)
        visual_np, qpos_np, action_np = self._task_arrays(episode.task)
        start = episode.offset
        t = request.time_index
        absolute_t = start + t
        observation_first = max(0, t - (HISTORY_STEPS - 1))
        observation_indices = np.arange(start + observation_first, absolute_t + 1)
        observation_offset = HISTORY_STEPS - len(observation_indices)
        action_first = max(0, t - HISTORY_STEPS)
        action_indices = np.arange(start + action_first, absolute_t)
        action_offset = HISTORY_STEPS - len(action_indices)
        teammate = 1 - request.arm

        history_visual = torch.zeros(HISTORY_STEPS, 2, 768, dtype=torch.float32)
        history_qpos = torch.zeros(HISTORY_STEPS, 9, dtype=torch.float32)
        history_action = torch.zeros(HISTORY_STEPS, 8, dtype=torch.float32)
        history_mask = torch.zeros(HISTORY_STEPS, dtype=torch.bool)
        action_history_mask = torch.zeros(HISTORY_STEPS, dtype=torch.bool)
        observed_visual = torch.from_numpy(
            np.array(visual_np[observation_indices][:, [0, request.arm + 1]], dtype=np.float32, copy=True)
        )
        view_indices = torch.tensor([0, request.arm + 1])
        history_visual[observation_offset:] = (
            observed_visual - self.visual_mean[view_indices]
        ) / self.visual_std[view_indices]
        observed_qpos = torch.from_numpy(
            np.array(qpos_np[observation_indices, request.arm], dtype=np.float32, copy=True)
        )
        history_qpos[observation_offset:] = (observed_qpos - self.q_mean) / self.q_std
        history_mask[observation_offset:] = True
        if len(action_indices):
            observed_action = torch.from_numpy(
                np.array(action_np[action_indices, request.arm], dtype=np.float32, copy=True)
            )
            history_action[action_offset:] = (observed_action - self.a_mean) / self.a_std
            action_history_mask[action_offset:] = True

        future_visual = torch.zeros(len(FUTURE_OFFSETS), 2, 768, dtype=torch.float32)
        teammate_delta = torch.zeros(len(FUTURE_OFFSETS), 9, dtype=torch.float32)
        future_mask = torch.zeros(len(FUTURE_OFFSETS), dtype=torch.bool)
        teammate_now = torch.from_numpy(
            np.array(qpos_np[absolute_t, teammate], dtype=np.float32, copy=True)
        )
        teammate_qpos = (teammate_now - self.q_mean) / self.q_std
        previous_absolute = start + max(0, t - 1)
        previous_teammate = torch.from_numpy(
            np.array(qpos_np[previous_absolute, teammate], dtype=np.float32, copy=True)
        )
        previous_teammate_qpos = (previous_teammate - self.q_mean) / self.q_std
        current_views = torch.from_numpy(
            np.array(visual_np[absolute_t, [0, teammate + 1]], dtype=np.float32, copy=True)
        )
        current_indices = torch.tensor([0, teammate + 1])
        current_target_visual = (
            current_views - self.visual_mean[current_indices]
        ) / self.visual_std[current_indices]
        for index, offset in enumerate(FUTURE_OFFSETS):
            target_t = t + offset
            if target_t >= episode.length:
                continue
            target_absolute = start + target_t
            target_views = torch.from_numpy(
                np.array(visual_np[target_absolute, [0, teammate + 1]], dtype=np.float32, copy=True)
            )
            target_indices = torch.tensor([0, teammate + 1])
            future_visual[index] = (
                target_views - self.visual_mean[target_indices]
            ) / self.visual_std[target_indices]
            future_teammate = torch.from_numpy(
                np.array(qpos_np[target_absolute, teammate], dtype=np.float32, copy=True)
            )
            teammate_delta[index] = (future_teammate - teammate_now) / self.q_std
            future_mask[index] = True

        action = torch.zeros(ACTION_PROBE_HORIZON, 8, dtype=torch.float32)
        action_mask = torch.zeros(ACTION_PROBE_HORIZON, dtype=torch.bool)
        action_end = min(t + ACTION_PROBE_HORIZON, episode.length)
        action_source = torch.from_numpy(
            np.array(action_np[absolute_t : start + action_end, request.arm], dtype=np.float32, copy=True)
        )
        valid_action = len(action_source)
        action[:valid_action] = (action_source - self.a_mean) / self.a_std
        action_mask[:valid_action] = True
        phase = float(t / max(episode.length - 1, 1))
        return {
            "history_visual": history_visual,
            "history_qpos": history_qpos,
            "history_action": history_action,
            "history_mask": history_mask,
            "action_history_mask": action_history_mask,
            "task_index": torch.tensor(episode.task_index, dtype=torch.long),
            "future_visual": future_visual,
            "current_target_visual": current_target_visual,
            "future_mask": future_mask,
            "teammate_qpos": teammate_qpos,
            "previous_teammate_qpos": previous_teammate_qpos,
            "teammate_delta": teammate_delta,
            "action": action,
            "action_mask": action_mask,
            "phase": torch.tensor(phase, dtype=torch.float32),
            "phase_bin": torch.tensor(min(3, int(phase * 4)), dtype=torch.long),
            "episode_label": torch.tensor(request.episode_index, dtype=torch.long),
            "task": episode.task,
            "sample_key": request.sample_key,
            "time_index": torch.tensor(t, dtype=torch.long),
            "agent_slot": torch.tensor(request.arm, dtype=torch.long),
        }


class N1BalancedBatchSampler(Sampler[list[N1Request]]):
    """Fixed six-task cursor over only the pre-registered train split."""

    def __init__(
        self,
        episodes: Sequence[N1Episode],
        *,
        updates: int,
        data_seed: int,
        start_update: int = 0,
    ) -> None:
        self.episodes = list(episodes)
        self.updates = int(updates)
        self.data_seed = int(data_seed)
        self.start_update = int(start_update)
        self.by_task = {
            task: [
                index
                for index, episode in enumerate(self.episodes)
                if episode.task == task and episode.split == "train"
            ]
            for task in SIX_TASKS
        }
        if any(len(value) != 100 for value in self.by_task.values()):
            raise ValueError("3-N1 split must contain 100 train episodes per task")

    def __len__(self) -> int:
        return self.updates - self.start_update

    def requests_for_update(self, update: int) -> list[N1Request]:
        rng = random.Random(self.data_seed + 1_000_003 * update)
        requests: list[N1Request] = []
        for task in SIX_TASKS:
            for _ in range(SAMPLES_PER_TASK):
                episode_index = rng.choice(self.by_task[task])
                episode = self.episodes[episode_index]
                arm = rng.randrange(2)
                time_index = rng.randrange(episode.length)
                sample_key = f"{episode.episode_key}:{arm}:{time_index}"
                requests.append(N1Request(episode_index, arm, time_index, sample_key, task))
        rng.shuffle(requests)
        if len(requests) != EFFECTIVE_BATCH:
            raise AssertionError("3-N1 effective batch drift")
        return requests

    def __iter__(self) -> Iterator[list[N1Request]]:
        for update in range(self.start_update + 1, self.updates + 1):
            yield self.requests_for_update(update)

    def cursor_receipt(self, update: int) -> dict:
        next_update = update + 1
        return {
            "format_version": "before-we-act.b3-n1-cursor/1",
            "data_seed": self.data_seed,
            "completed_update": update,
            "next_sample_keys": (
                [row.sample_key for row in self.requests_for_update(next_update)]
                if next_update <= self.updates
                else []
            ),
        }


def validation_requests(cache_root: str | Path) -> list[N1Request]:
    root = Path(cache_root)
    metadata, _ = load_n1_metadata(root)
    return [N1Request(**row) for row in metadata["validation_requests"]]


__all__ = [
    "ACTION_PROBE_HORIZON",
    "CAPACITY_CANDIDATES",
    "FUTURE_OFFSETS",
    "N1BalancedBatchSampler",
    "N1Episode",
    "N1RawSignalDataset",
    "N1Request",
    "load_n1_metadata",
    "validation_requests",
]
