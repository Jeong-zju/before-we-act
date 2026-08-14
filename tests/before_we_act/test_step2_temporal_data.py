from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path

import h5py
import numpy as np
import torch

from before_we_act.step2_temporal_data import (
    ExactSixTaskDistributedBatchSampler,
    SIX_TASKS,
    Step2Episode,
    TeamTemporalDataset,
    TeamTemporalRequest,
)


def make_episode(tmp_path: Path, task: str = "lift_barrier", length: int = 20):
    path = tmp_path / f"{task}.hdf5"
    arms = (0, 1)
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        observation = data.create_group("observation")
        images = observation.create_group("images")
        image = np.zeros((length, 480, 640, 3), dtype=np.uint8)
        image[:, :, :, 0] = np.arange(length, dtype=np.uint8)[:, None, None]
        images.create_dataset("global", data=image)
        if task != "place_food":
            images.create_dataset("agent_0", data=image + 1)
            images.create_dataset("agent_1", data=image + 2)
        agents = observation.create_group("agents")
        action_agents = data.create_group("action").create_group("agents")
        for arm in arms:
            panda = agents.create_group(f"panda_{arm}")
            panda.create_dataset(
                "qpos",
                data=np.arange(length * 9, dtype=np.float32).reshape(length, 9)
                + arm,
            )
            action = action_agents.create_group(f"panda_{arm}")
            action.create_dataset(
                "commanded",
                data=np.arange(length * 8, dtype=np.float32).reshape(length, 8)
                + arm,
            )
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    episode = Step2Episode(
        path=str(path),
        relative_path=path.name,
        task=task,
        task_text={
            "lift_barrier": "Lift the barrier together",
            "place_food": "Place the food together",
        }[task],
        arms=arms,
        length=length,
        seed=1,
        episode_index=0,
        manifest_path="manifest.json",
        manifest_sha256="a" * 64,
        hdf5_sha256=source_hash,
    )
    cache_path = tmp_path / "cache" / task / f"{source_hash}.npz"
    cache_path.parent.mkdir(parents=True)
    values = {
        "view_global": np.ones((length, 768), dtype=np.float16),
        "view_agent_0": np.full((length, 768), 2, dtype=np.float16),
        "view_agent_1": np.full((length, 768), 3, dtype=np.float16),
    }
    np.savez(
        cache_path,
        source_hdf5_sha256=np.asarray(source_hash),
        **values,
    )
    return episode, tmp_path / "cache"


def stats():
    return {
        "q_mean": np.zeros(9, dtype=np.float32),
        "q_std": np.ones(9, dtype=np.float32),
        "a_mean": np.zeros(8, dtype=np.float32),
        "a_std": np.ones(8, dtype=np.float32),
    }


def request(time_index: int, task: str = "lift_barrier", arm: int = 0):
    return TeamTemporalRequest(0, arm, time_index, f"sample-{time_index}-{arm}", task)


def test_begin_middle_end_masks_are_past_causal(tmp_path: Path) -> None:
    episode, cache = make_episode(tmp_path)
    dataset = TeamTemporalDataset([episode], stats(), cache)
    beginning = dataset[request(0)]
    assert beginning["history_mask"].tolist() == [False] * 15 + [True]
    assert not beginning["action_history_mask"].any()
    assert beginning["episode_reset"]
    assert int(beginning["action_mask"].sum()) == 20

    middle = dataset[request(10)]
    assert int(middle["history_mask"].sum()) == 11
    assert int(middle["action_history_mask"].sum()) == 10
    assert torch.equal(middle["history_action"][-1], torch.arange(72, 80).float())
    assert torch.equal(middle["action"][0], torch.arange(80, 88).float())
    assert not middle["social_supervision_mask"]

    end = dataset[request(19)]
    assert int(end["history_mask"].sum()) == 16
    assert int(end["action_history_mask"].sum()) == 16
    assert int(end["action_mask"].sum()) == 1
    assert torch.equal(end["action"][0], end["action"][-1])


def test_place_food_missing_local_uses_original_global(tmp_path: Path) -> None:
    episode, cache = make_episode(tmp_path, task="place_food")
    dataset = TeamTemporalDataset([episode], stats(), cache)
    sample = dataset[request(5, task="place_food", arm=1)]
    assert torch.equal(sample["global_rgb"], sample["local_rgb"])
    assert torch.equal(
        sample["history_visual_raw"][-1, 0],
        sample["history_visual_raw"][-1, 1],
    )


def test_distributed_sampler_reconstructs_exact_global_batch() -> None:
    episodes = []
    for task in SIX_TASKS:
        for index in range(120):
            episodes.append(
                Step2Episode(
                    path=f"/{task}/{index}.hdf5",
                    relative_path=f"{index}.hdf5",
                    task=task,
                    task_text=task,
                    arms=(0, 1),
                    length=20,
                    seed=index,
                    episode_index=index,
                    manifest_path=f"/{task}/manifest.json",
                    manifest_sha256=task,
                    hdf5_sha256=f"{task}-{index}",
                )
            )
    global_sampler = ExactSixTaskDistributedBatchSampler(
        episodes, updates=4, seed=9
    )
    global_requests = global_sampler.requests_for_update(3)
    reconstructed = []
    per_rank = []
    for rank in range(4):
        sampler = ExactSixTaskDistributedBatchSampler(
            episodes, updates=4, seed=9, rank=rank, world_size=4
        )
        local = next(iter(sampler)) if rank == 0 else list(sampler)[2]
        # rank 0's first iterator row is update 1; use the direct frozen split
        # below for the update-3 reconstruction checked by every rank.
        direct = sampler.requests_for_update(3)[rank::4]
        per_rank.append(len(direct))
        reconstructed.extend((offset * 4 + rank, item) for offset, item in enumerate(direct))
        assert len(local) == 12
    reconstructed.sort(key=lambda value: value[0])
    assert [item.sample_key for _, item in reconstructed] == [
        item.sample_key for item in global_requests
    ]
    assert per_rank == [12, 12, 12, 12]
    assert Counter(item.task for item in global_requests) == Counter(
        {task: 8 for task in SIX_TASKS}
    )
    assert global_sampler.cursor_receipt(2)["next_sample_keys"] == [
        item.sample_key for item in global_requests
    ]
