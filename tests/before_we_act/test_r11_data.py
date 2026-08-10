import json
from pathlib import Path
from collections import Counter

import h5py
import numpy as np
import pytest

from before_we_act.r11_data import (
    EFFECTIVE_BATCH,
    ExactSixTaskAccumulationSampler,
    R11EpisodeDataset,
    SIX_TASKS,
    episode_receipt,
    load_r11_episodes,
)


def _write_task(root: Path, task: str, task_index: int) -> Path:
    task_root = root / task
    hdf5_root = task_root / "hdf5"
    hdf5_root.mkdir(parents=True)
    path = hdf5_root / "episode_000000.hdf5"
    length = 5
    arms = 2
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        observation = data.create_group("observation")
        images = observation.create_group("images")
        global_rgb = np.zeros((length, 6, 8, 3), dtype=np.uint8)
        for index in range(length):
            global_rgb[index] = task_index * 10 + index
        images.create_dataset("global", data=global_rgb)
        if task != "place_food":
            for arm in range(arms):
                images.create_dataset(f"agent_{arm}", data=global_rgb + arm + 1)
        agents = observation.create_group("agents")
        action_agents = data.create_group("action").create_group("agents")
        for arm in range(arms):
            agent = agents.create_group(f"panda_{arm}")
            agent.create_dataset("qpos", data=np.full((length, 9), arm, np.float32))
            action_agent = action_agents.create_group(f"panda_{arm}")
            action_agent.create_dataset(
                "commanded", data=np.arange(length * 8, dtype=np.float32).reshape(length, 8)
            )
    payload = path.read_bytes()
    import hashlib

    manifest = {
        "task": {"id": task},
        "action": {"dimension": arms * 8},
        "episodes": [
            {
                "split": "train",
                "hdf5_path": "hdf5/episode_000000.hdf5",
                "steps": length,
                "seed": 100 + task_index,
                "episode_index": 0,
                "hdf5_sha256": hashlib.sha256(payload).hexdigest(),
                "task_text": f"instruction for {task}",
            }
        ],
    }
    manifest_path = task_root / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


@pytest.fixture()
def episodes(tmp_path):
    manifests = [_write_task(tmp_path, task, index) for index, task in enumerate(SIX_TASKS)]
    return load_r11_episodes(manifests)


def test_episode_adapter_future_masks_text_and_place_food_fallback(episodes):
    stats = {
        "q_mean": np.zeros(9, np.float32),
        "q_std": np.ones(9, np.float32),
        "a_mean": np.zeros(8, np.float32),
        "a_std": np.ones(8, np.float32),
    }
    sampler = ExactSixTaskAccumulationSampler(
        episodes, updates=2, seed=7, micro_batch_size=2
    )
    place = next(item for item in sampler.requests_for_update(1) if item.task == "place_food")
    place = type(place)(place.episode_index, place.arm, 3, place.sample_key, place.task)
    sample = R11EpisodeDataset(episodes, stats, image_size=(3, 4))[place]
    assert sample["current_rgb"].shape == (2, 3, 3, 4)
    assert sample["future_rgb"].shape == (4, 2, 3, 3, 4)
    assert sample["future_qpos"].shape == (4, 9)
    assert sample["future_mask"].tolist() == [True, False, False, False]
    assert sample["future_qpos"][1:].count_nonzero().item() == 0
    assert sample["action_mask"].sum().item() == 2
    assert sample["task_text"] == "instruction for place_food"
    assert np.array_equal(sample["current_rgb"][0].numpy(), sample["current_rgb"][1].numpy())


def test_exact_accumulation_balance_and_resume_cursor(episodes):
    sampler = ExactSixTaskAccumulationSampler(
        episodes, updates=3, seed=20260809, micro_batch_size=2
    )
    assert sampler.accumulation_steps == 24
    assert len(sampler) == 72
    first = sampler.requests_for_update(2)
    assert len(first) == EFFECTIVE_BATCH
    assert Counter(item.task for item in first) == Counter({task: 8 for task in SIX_TASKS})
    assert all(item.time_index < episodes[item.episode_index].length - 1 for item in first)
    assert [item.sample_key for item in first] == [
        item.sample_key for item in sampler.requests_for_update(2)
    ]
    receipt = sampler.cursor_receipt(1)
    assert receipt["next_sample_keys"] == [item.sample_key for item in first]
    assert sampler.validate_resume_receipt(receipt) == 1
    broken = dict(receipt)
    broken["next_sample_keys"] = list(reversed(receipt["next_sample_keys"]))
    with pytest.raises(ValueError, match="next_sample_keys"):
        sampler.validate_resume_receipt(broken)
    projection = episode_receipt(episodes)
    assert projection["episodes"] == 6
    assert list(projection["task_texts"]) == list(SIX_TASKS)


def test_micro_batch_must_divide_effective_batch(episodes):
    with pytest.raises(ValueError, match="must divide"):
        ExactSixTaskAccumulationSampler(
            episodes, updates=1, seed=1, micro_batch_size=5
        )
