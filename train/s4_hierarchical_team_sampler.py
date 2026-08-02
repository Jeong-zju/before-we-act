"""Resume-exact S4 ``task -> episode -> time`` team sampling.

Sampling always returns a complete synchronous team window.  Agent balancing
is a loss-reduction concern and is therefore tracked here as exposure, never
implemented by splitting a dataset item into per-agent samples.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from numbers import Integral
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Sampler

from train.s2_grouped_trajectory import S2GroupedTrajectoryDataset


S4ResumeKey = tuple[int, int, int, int]
S4_RESUME_KEY_FIELDS = (
    "seed",
    "optimizer_update",
    "accumulation_index",
    "item_index",
)
_MAX_KEY_VALUE = (1 << 63) - 1


def _key_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if not 0 <= normalized <= _MAX_KEY_VALUE:
        raise ValueError(f"{name} must be in [0, 2**63-1]")
    return normalized


def _positive_integer(value: object, *, name: str) -> int:
    normalized = _key_integer(value, name=name)
    if normalized == 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _item_rng_seed(key: S4ResumeKey) -> int:
    payload = "wam.s4.hierarchical-team/1:" + ":".join(map(str, key))
    digest = hashlib.sha256(payload.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & _MAX_KEY_VALUE


class S4HierarchicalTeamBatchSampler(Sampler[list[int]]):
    """Yield micro-batches with one independent RNG per item.

    A dataset index is a whole team window.  For every item, a CPU generator
    derived solely from ``(seed, optimizer_update, accumulation_index,
    item_index)`` draws a task, then an episode in that task, then a time in
    that episode.  Resume therefore does not depend on how many random values
    were consumed before a checkpoint or on DataLoader worker prefetch.

    ``first_update`` and ``final_update`` are inclusive.  A non-zero
    ``first_item_index`` yields the unconsumed suffix of the first micro-batch;
    normal optimizer-update checkpoints should use item index zero.
    """

    def __init__(
        self,
        dataset: S2GroupedTrajectoryDataset,
        *,
        micro_batch_size: int | None = None,
        batch_size: int | None = None,
        gradient_accumulation: int,
        first_update: int,
        final_update: int,
        seed: int,
        first_accumulation_index: int = 0,
        first_item_index: int = 0,
    ) -> None:
        if micro_batch_size is None and batch_size is None:
            raise TypeError("micro_batch_size is required")
        if micro_batch_size is not None and batch_size is not None:
            if int(micro_batch_size) != int(batch_size):
                raise ValueError("micro_batch_size and batch_size disagree")
        selected_batch_size = (
            batch_size if micro_batch_size is None else micro_batch_size
        )
        self.micro_batch_size = _positive_integer(
            selected_batch_size, name="micro_batch_size"
        )
        self.gradient_accumulation = _positive_integer(
            gradient_accumulation, name="gradient_accumulation"
        )
        self.seed = _key_integer(seed, name="seed")
        self.first_update = _key_integer(first_update, name="first_update")
        self.final_update = _key_integer(final_update, name="final_update")
        if self.final_update < self.first_update:
            raise ValueError("final_update cannot precede first_update")
        self.first_accumulation_index = _key_integer(
            first_accumulation_index, name="first_accumulation_index"
        )
        self.first_item_index = _key_integer(
            first_item_index, name="first_item_index"
        )
        self._validate_local_position(
            self.first_accumulation_index, self.first_item_index
        )
        self._hierarchy = self._normalize_hierarchy(
            dataset.hierarchical_indices()
        )

    @staticmethod
    def _normalize_hierarchy(
        value: Mapping[str, Mapping[int, Sequence[int]]],
    ) -> tuple[tuple[str, tuple[tuple[int, tuple[int, ...]], ...]], ...]:
        if not isinstance(value, Mapping) or not value:
            raise ValueError("hierarchical dataset indices must be non-empty")
        normalized_tasks: list[
            tuple[str, tuple[tuple[int, tuple[int, ...]], ...]]
        ] = []
        observed_indices: set[int] = set()
        for raw_task_id, raw_episodes in value.items():
            if not isinstance(raw_task_id, str) or not raw_task_id:
                raise ValueError("hierarchy task ids must be non-empty strings")
            task_id = raw_task_id
            if not isinstance(raw_episodes, Mapping):
                raise ValueError("hierarchy contains an invalid task entry")
            episodes: list[tuple[int, tuple[int, ...]]] = []
            for raw_episode_index in sorted(raw_episodes, key=int):
                episode_index = _key_integer(
                    raw_episode_index, name="episode_index"
                )
                raw_indices = raw_episodes[raw_episode_index]
                if isinstance(raw_indices, (str, bytes)):
                    raise ValueError("episode indices must be an integer sequence")
                indices = tuple(
                    _key_integer(index, name="dataset_index")
                    for index in raw_indices
                )
                if not indices:
                    raise ValueError(
                        f"task {task_id!r} episode {episode_index} has no times"
                    )
                if len(indices) != len(set(indices)):
                    raise ValueError("hierarchy repeats a time index within an episode")
                overlap = observed_indices.intersection(indices)
                if overlap:
                    raise ValueError(
                        f"hierarchy assigns dataset indices more than once: "
                        f"{sorted(overlap)!r}"
                    )
                observed_indices.update(indices)
                episodes.append((episode_index, indices))
            if not episodes:
                raise ValueError(f"task {task_id!r} has no episodes")
            normalized_tasks.append((task_id, tuple(episodes)))
        return tuple(normalized_tasks)

    def _validate_local_position(
        self, accumulation_index: int, item_index: int
    ) -> None:
        if accumulation_index >= self.gradient_accumulation:
            raise ValueError("accumulation_index lies outside the optimizer update")
        if item_index >= self.micro_batch_size:
            raise ValueError("item_index lies outside the micro-batch")

    def resume_key(
        self,
        optimizer_update: int,
        accumulation_index: int,
        item_index: int,
    ) -> S4ResumeKey:
        update = _key_integer(optimizer_update, name="optimizer_update")
        accumulation = _key_integer(
            accumulation_index, name="accumulation_index"
        )
        item = _key_integer(item_index, name="item_index")
        self._validate_local_position(accumulation, item)
        return self.seed, update, accumulation, item

    def sample_index(
        self,
        optimizer_update: int,
        accumulation_index: int,
        item_index: int,
    ) -> int:
        """Resolve exactly one resume-keyed team-window index."""

        key = self.resume_key(
            optimizer_update, accumulation_index, item_index
        )
        generator = torch.Generator().manual_seed(_item_rng_seed(key))
        task_position = int(
            torch.randint(len(self._hierarchy), (), generator=generator)
        )
        episodes = self._hierarchy[task_position][1]
        episode_position = int(
            torch.randint(len(episodes), (), generator=generator)
        )
        times = episodes[episode_position][1]
        time_position = int(torch.randint(len(times), (), generator=generator))
        return times[time_position]

    def sample_from_resume_key(self, key: Sequence[int]) -> int:
        if len(key) != len(S4_RESUME_KEY_FIELDS):
            raise ValueError("S4 resume key must contain exactly four integers")
        seed, update, accumulation, item = (
            _key_integer(value, name=name)
            for value, name in zip(key, S4_RESUME_KEY_FIELDS, strict=True)
        )
        if seed != self.seed:
            raise ValueError("resume key seed disagrees with sampler seed")
        return self.sample_index(update, accumulation, item)

    def set_resume_position(
        self,
        optimizer_update: int,
        *,
        accumulation_index: int = 0,
        item_index: int = 0,
    ) -> None:
        update = _key_integer(optimizer_update, name="optimizer_update")
        accumulation = _key_integer(
            accumulation_index, name="accumulation_index"
        )
        item = _key_integer(item_index, name="item_index")
        self._validate_local_position(accumulation, item)
        if update > self.final_update:
            raise ValueError("resume update lies after final_update")
        self.first_update = update
        self.first_accumulation_index = accumulation
        self.first_item_index = item

    def __len__(self) -> int:
        complete_later_updates = self.final_update - self.first_update
        return (
            complete_later_updates * self.gradient_accumulation
            + self.gradient_accumulation
            - self.first_accumulation_index
        )

    def __iter__(self) -> Iterator[list[int]]:
        for update in range(self.first_update, self.final_update + 1):
            first_accumulation = (
                self.first_accumulation_index
                if update == self.first_update
                else 0
            )
            for accumulation_index in range(
                first_accumulation, self.gradient_accumulation
            ):
                first_item = (
                    self.first_item_index
                    if update == self.first_update
                    and accumulation_index == first_accumulation
                    else 0
                )
                yield [
                    self.sample_index(update, accumulation_index, item_index)
                    for item_index in range(first_item, self.micro_batch_size)
                ]

    def summary(self) -> dict[str, Any]:
        episodes = sum(len(task_episodes) for _, task_episodes in self._hierarchy)
        windows = sum(
            len(indices)
            for _, task_episodes in self._hierarchy
            for _, indices in task_episodes
        )
        return {
            "strategy": "uniform_task_episode_time_complete_team",
            "hierarchy_order": ["task", "episode", "time", "all_valid_agent"],
            "resume_key_fields": list(S4_RESUME_KEY_FIELDS),
            "seed": self.seed,
            "micro_team_batch": self.micro_batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "effective_team_batch": (
                self.micro_batch_size * self.gradient_accumulation
            ),
            "first_update": self.first_update,
            "first_accumulation_index": self.first_accumulation_index,
            "first_item_index": self.first_item_index,
            "final_update": self.final_update,
            "tasks": len(self._hierarchy),
            "episodes": episodes,
            "team_windows": windows,
            "complete_team_items": True,
        }


@dataclass
class S4ExposureCounter:
    """Trainer-owned counters for team and valid-agent window exposure."""

    team_windows_seen: int = 0
    valid_agent_windows_seen: int = 0

    def __post_init__(self) -> None:
        self.team_windows_seen = _key_integer(
            self.team_windows_seen, name="team_windows_seen"
        )
        self.valid_agent_windows_seen = _key_integer(
            self.valid_agent_windows_seen, name="valid_agent_windows_seen"
        )

    def record_batch(self, valid_agent_mask: Tensor | Sequence[Sequence[bool]]) -> None:
        mask = torch.as_tensor(valid_agent_mask)
        if mask.ndim != 2 or mask.dtype != torch.bool:
            raise ValueError("valid_agent_mask must be bool [team,agent]")
        teams = int(mask.shape[0])
        valid_agents = int(mask.sum().item())
        self.record_counts(
            team_windows=teams, valid_agent_windows=valid_agents
        )

    def update(self, valid_agent_mask: Tensor | Sequence[Sequence[bool]]) -> None:
        """Compatibility spelling for trainer-side batch accounting."""

        self.record_batch(valid_agent_mask)

    def record_counts(
        self, *, team_windows: int, valid_agent_windows: int
    ) -> None:
        teams = _key_integer(team_windows, name="team_windows")
        agents = _key_integer(valid_agent_windows, name="valid_agent_windows")
        if teams == 0 and agents != 0:
            raise ValueError("zero team windows cannot contain valid agents")
        self.team_windows_seen += teams
        self.valid_agent_windows_seen += agents

    def state_dict(self) -> dict[str, int]:
        return {
            "team_windows_seen": self.team_windows_seen,
            "valid_agent_windows_seen": self.valid_agent_windows_seen,
        }

    def load_state_dict(self, value: Mapping[str, object]) -> None:
        expected = {"team_windows_seen", "valid_agent_windows_seen"}
        if set(value) != expected:
            raise ValueError("S4 exposure counter state keys drifted")
        teams = _key_integer(value["team_windows_seen"], name="team_windows_seen")
        agents = _key_integer(
            value["valid_agent_windows_seen"],
            name="valid_agent_windows_seen",
        )
        self.team_windows_seen = teams
        self.valid_agent_windows_seen = agents

    def summary(self, *, agent_window_budget: int | None = None) -> dict[str, Any]:
        value: dict[str, Any] = self.state_dict()
        value["mean_valid_agents_per_team"] = (
            0.0
            if self.team_windows_seen == 0
            else self.valid_agent_windows_seen / self.team_windows_seen
        )
        if agent_window_budget is not None:
            budget = _positive_integer(
                agent_window_budget, name="agent_window_budget"
            )
            value["agent_window_budget"] = budget
            value["agent_window_progress"] = (
                self.valid_agent_windows_seen / budget
            )
        return value


# Public compatibility names for callers that do not distinguish a PyTorch
# batch sampler from the sampling strategy itself.
S4HierarchicalTeamSampler = S4HierarchicalTeamBatchSampler
HierarchicalTeamBatchSampler = S4HierarchicalTeamBatchSampler
ExposureCounter = S4ExposureCounter


__all__ = [
    "ExposureCounter",
    "HierarchicalTeamBatchSampler",
    "S4ExposureCounter",
    "S4HierarchicalTeamBatchSampler",
    "S4HierarchicalTeamSampler",
    "S4ResumeKey",
    "S4_RESUME_KEY_FIELDS",
]
