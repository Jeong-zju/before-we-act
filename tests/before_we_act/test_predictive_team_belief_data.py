from __future__ import annotations

from collections import Counter

import pytest

pytest.importorskip("torch")

from before_we_act.raw_team_signal_data import TeamEpisode
from before_we_act.predictive_team_belief_data import PairedSituationBatchSampler
from before_we_act.temporal_history_data import SIX_TASKS


def _episodes() -> tuple[list[TeamEpisode], dict[str, str]]:
    episodes = []
    split = {}
    offset = 0
    for task_index, task in enumerate(SIX_TASKS):
        for local_index in range(96):
            key = f"{task}-{local_index}"
            episodes.append(
                TeamEpisode(
                    task=task,
                    task_index=task_index,
                    local_index=local_index,
                    offset=offset,
                    length=1,
                    split="train",
                    episode_key=key,
                    hdf5_sha256=f"sha-{key}",
                )
            )
            split[key] = "train"
            offset += 1
    return episodes, split


def test_paired_sampler_never_reuses_a_situation_inside_one_batch() -> None:
    episodes, split = _episodes()
    sampler = PairedSituationBatchSampler(
        episodes,
        split,
        updates=1_000,
        data_seed=20260731,
    )

    for requests in sampler:
        pair_counts = Counter(
            (request.episode_index, request.time_index) for request in requests
        )
        assert set(pair_counts.values()) == {2}
        assert len(pair_counts) == 24
        assert all(
            {request.arm for request in requests if request.episode_index == episode_index}
            == {0, 1}
            for episode_index, _ in pair_counts
        )
