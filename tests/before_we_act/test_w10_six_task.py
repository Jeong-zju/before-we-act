from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor/stereo-core/stereo_core"))

from before_we_act.train_w10_six_task import (  # noqa: E402
    SIX_TASKS,
    ExactSixTaskBatchSampler,
    NoWristFrameDataset,
)


def test_balanced_six_task_sampler() -> None:
    episodes = [
        {"task": task, "arms": (0,), "length": 5, "path": task}
        for task in SIX_TASKS
    ]
    batch = next(iter(ExactSixTaskBatchSampler(episodes, updates=1, seed=7)))
    counts = Counter(episodes[episode_index]["task"] for episode_index, _arm, _time in batch)
    assert len(batch) == 48
    assert counts == Counter({task: 8 for task in SIX_TASKS})


def test_missing_local_camera_reuses_global(tmp_path: Path) -> None:
    episode_path = tmp_path / "place_food.hdf5"
    image = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(1, 480, 640, 3)
    with h5py.File(episode_path, "w") as handle:
        data = handle.create_group("data")
        observation = data.create_group("observation")
        images = observation.create_group("images")
        images.create_dataset("global", data=image)
        agents = observation.create_group("agents")
        panda = agents.create_group("panda_0")
        panda.create_dataset("qpos", data=np.zeros((1, 9), dtype=np.float32))
        actions = data.create_group("action").create_group("agents")
        action = actions.create_group("panda_0")
        action.create_dataset("commanded", data=np.zeros((1, 8), dtype=np.float32))

    stats = {
        "q_mean": np.zeros(9, dtype=np.float32),
        "q_std": np.ones(9, dtype=np.float32),
        "a_mean": np.zeros(8, dtype=np.float32),
        "a_std": np.ones(8, dtype=np.float32),
    }
    dataset = NoWristFrameDataset(
        [
            {
                "path": str(episode_path),
                "task": "place_food",
                "arms": (0,),
                "length": 1,
            }
        ],
        horizon=2,
        stats=stats,
    )
    global_rgb, local_rgb, *_rest = dataset[(0, 0, 0)]
    assert np.array_equal(global_rgb.numpy(), local_rgb.numpy())
