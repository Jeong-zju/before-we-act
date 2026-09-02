"""MARS must read the shared global camera, not a duplicated local view.

MARS records ``head_camera_global`` alongside each arm's own camera, but the
policy used to receive the local view in both dual-view slots, which left the
aligned two-view fusion comparing an image against itself.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from before_we_act.mars_temporal_data import (
    GLOBAL_VIEW_KEY,
    MarsTemporalEpisode,
    MarsVisualCache,
)


LENGTH = 4
ARMS = (0, 1)


def _write_cache(root: Path, episode: MarsTemporalEpisode) -> None:
    values = {
        GLOBAL_VIEW_KEY: np.full((LENGTH, 768), 0.5, dtype=np.float16),
        **{
            f"agent_{arm}": np.full((LENGTH, 768), float(arm + 1), dtype=np.float16)
            for arm in ARMS
        },
    }
    path = MarsVisualCache(root).path_for(episode)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def _episode(tmp_path: Path) -> MarsTemporalEpisode:
    return MarsTemporalEpisode(
        path=str(tmp_path / "shard.h5"),
        trajectory="traj_0",
        task="place_cube_in_cup",
        task_text="place the cube in the cup",
        arms=ARMS,
        length=LENGTH,
        episode_index=0,
        cache_key="shard-traj_0",
    )


def test_cache_load_returns_a_distinct_global_view(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    _write_cache(tmp_path / "cache", episode)
    loaded = MarsVisualCache(tmp_path / "cache").load(episode)

    assert GLOBAL_VIEW_KEY in loaded
    for arm in ARMS:
        assert not np.array_equal(loaded[GLOBAL_VIEW_KEY], loaded[f"agent_{arm}"])


def test_cache_rejects_a_stale_single_view_cache(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    path = MarsVisualCache(tmp_path / "cache").path_for(episode)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{
            f"agent_{arm}": np.zeros((LENGTH, 768), dtype=np.float16)
            for arm in ARMS
        },
    )

    with pytest.raises(ValueError, match="dual-view contract"):
        MarsVisualCache(tmp_path / "cache").load(episode)


def test_global_observation_reads_the_shared_camera() -> None:
    from deployment.mars_care.common import global_observation, local_observation

    shared = np.full((240, 320, 3), 7, dtype=np.uint8)
    observation = {
        "agent": {f"panda-{arm}": {"qpos": np.zeros(9, np.float32)} for arm in ARMS},
        "sensor_data": {
            "head_camera_global": {"rgb": shared},
            **{
                f"head_camera_agent{arm}": {
                    "rgb": np.full((240, 320, 3), arm, dtype=np.uint8)
                }
                for arm in ARMS
            },
        },
    }

    assert np.array_equal(global_observation(observation), shared)
    for arm in ARMS:
        local, _qpos = local_observation(observation, arm)
        assert not np.array_equal(local, global_observation(observation))
