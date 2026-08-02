from __future__ import annotations

from dataclasses import dataclass
import pickle
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from train.s2_grouped_trajectory import S2GroupedTrajectoryDataset
from train.s4_hierarchical_team_sampler import (
    S4ExposureCounter,
    S4HierarchicalTeamBatchSampler,
)


@dataclass(frozen=True)
class _Window:
    record_index: int
    decision_t: int


class _TaskDataset:
    def __init__(self, episodes: list[int], windows: list[_Window]) -> None:
        self.records = [
            SimpleNamespace(episode_index=episode, seed=episode + 100)
            for episode in episodes
        ]
        self._index = windows

    def __len__(self) -> int:
        return len(self._index)


class _Source:
    def __init__(self) -> None:
        self.datasets = (
            _TaskDataset(
                [20, 10],
                [
                    _Window(0, 8),
                    _Window(1, 4),
                    _Window(0, 2),
                    _Window(1, 1),
                ],
            ),
            _TaskDataset([30], [_Window(0, 7), _Window(0, 3)]),
        )
        self.contracts = (
            SimpleNamespace(task_id="task_b"),
            SimpleNamespace(task_id="task_a"),
        )
        self.task_vocabulary = ("task_b", "task_a")
        self._ranges = (range(0, 4), range(4, 6))

    def task_indices(self, task_index: int) -> range:
        return self._ranges[task_index]


def _grouped_dataset() -> S2GroupedTrajectoryDataset:
    dataset = S2GroupedTrajectoryDataset.__new__(S2GroupedTrajectoryDataset)
    dataset.source = _Source()
    dataset.contracts = dataset.source.contracts
    dataset.task_vocabulary = dataset.source.task_vocabulary
    dataset.split = "train"
    dataset._hierarchical_indices_cache = dataset._build_hierarchical_indices()
    dataset._restore_hierarchical_indices_view()
    return dataset


def test_grouped_dataset_caches_read_only_task_episode_time_hierarchy() -> None:
    dataset = _grouped_dataset()
    hierarchy = dataset.hierarchical_indices()

    assert list(hierarchy) == ["task_b", "task_a"]
    assert dict(hierarchy["task_b"]) == {10: (3, 1), 20: (2, 0)}
    assert dict(hierarchy["task_a"]) == {30: (5, 4)}
    with pytest.raises(TypeError):
        hierarchy["task_b"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        hierarchy["task_b"][10] = (99,)  # type: ignore[index]

    restored = pickle.loads(pickle.dumps(dataset))
    assert dict(restored.hierarchical_indices()["task_b"]) == {
        10: (3, 1),
        20: (2, 0),
    }


def test_hierarchical_sampler_resume_is_exact_at_accumulation_and_item() -> None:
    dataset = _grouped_dataset()
    complete_sampler = S4HierarchicalTeamBatchSampler(
        dataset,
        micro_batch_size=4,
        gradient_accumulation=3,
        first_update=5,
        final_update=7,
        seed=909,
    )
    complete = list(complete_sampler)
    resumed = S4HierarchicalTeamBatchSampler(
        dataset,
        micro_batch_size=4,
        gradient_accumulation=3,
        first_update=5,
        final_update=7,
        seed=909,
        first_accumulation_index=1,
        first_item_index=2,
    )

    assert list(resumed) == [complete[1][2:], *complete[2:]]
    key = complete_sampler.resume_key(6, 2, 3)
    assert key == (909, 6, 2, 3)
    assert complete_sampler.sample_from_resume_key(key) == (
        complete_sampler.sample_index(6, 2, 3)
    )
    replay = S4HierarchicalTeamBatchSampler(
        dataset,
        batch_size=4,
        gradient_accumulation=3,
        first_update=5,
        final_update=7,
        seed=909,
    )
    assert list(replay) == complete
    assert complete_sampler.summary()["effective_team_batch"] == 12


def test_hierarchical_sampler_balances_episode_before_time() -> None:
    hierarchy = MappingProxyType(
        {
            "task": MappingProxyType(
                {
                    0: (0,),
                    1: tuple(range(1, 101)),
                }
            )
        }
    )
    dataset = SimpleNamespace(hierarchical_indices=lambda: hierarchy)
    sampler = S4HierarchicalTeamBatchSampler(
        dataset,
        micro_batch_size=8,
        gradient_accumulation=4,
        first_update=0,
        final_update=124,
        seed=77,
    )
    sampled = [index for batch in sampler for index in batch]
    short_episode = sum(index == 0 for index in sampled)

    # The 100-times-long episode must not receive 100x the probability.
    assert 0.45 < short_episode / len(sampled) < 0.55


def test_exposure_counter_tracks_team_and_valid_agent_windows() -> None:
    counter = S4ExposureCounter()
    counter.record_batch(
        torch.tensor(
            [[True, True, False, False], [True, True, True, False]]
        )
    )
    counter.record_counts(team_windows=1, valid_agent_windows=4)
    assert counter.state_dict() == {
        "team_windows_seen": 3,
        "valid_agent_windows_seen": 9,
    }
    assert counter.summary(agent_window_budget=18) == {
        "team_windows_seen": 3,
        "valid_agent_windows_seen": 9,
        "mean_valid_agents_per_team": 3.0,
        "agent_window_budget": 18,
        "agent_window_progress": 0.5,
    }

    restored = S4ExposureCounter()
    restored.load_state_dict(counter.state_dict())
    assert restored == counter
    with pytest.raises(ValueError, match="bool"):
        counter.record_batch(torch.ones(2, 4))
