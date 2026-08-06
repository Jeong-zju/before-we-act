from __future__ import annotations

from pathlib import Path

import torch

from before_we_act.data.full_episode_windows import (
    ExactFiveTaskFullEpisodeSampler,
    FULL_EPISODE_PROTOCOL,
    FullEpisodeActionWindows,
)
from before_we_act.data.raw_team_windows import TASKS
from before_we_act.contracts import TeamBeliefState


def _episode(path: Path, task: str, seed: int, steps: int = 4) -> dict:
    agents = 2 + TASKS.index(task) % 3
    view_mask = torch.zeros((steps, 5), dtype=torch.bool)
    view_mask[:, : agents + 1] = True
    payload = {
        "schema_version": 1,
        "round": "R12-R4",
        "metadata": {
            "protocol_variant": FULL_EPISODE_PROTOCOL,
            "task": task,
            "split": "train",
            "seed": seed,
            "steps": steps,
            "hdf5_sha256": f"sha-{task}",
        },
        "visual": torch.arange(steps * 16 * 15, dtype=torch.float16).reshape(
            steps, 16, 15
        ),
        "view_mask": view_mask,
        "qpos": torch.arange(steps * 4 * 9, dtype=torch.float32).reshape(
            steps, 4, 9
        ),
        "executed_actions": torch.arange(
            steps * 4 * 8, dtype=torch.float32
        ).reshape(steps, 4, 8),
        "commanded_actions": torch.arange(
            steps * 4 * 8, dtype=torch.float32
        ).reshape(steps, 4, 8),
        "agent_mask": torch.arange(4) < agents,
        "spatial_tokens": torch.zeros((steps, 5, 48, 768), dtype=torch.float16),
        "spatial_view_mask": view_mask.clone(),
    }
    torch.save(payload, path)
    return {
        "path": str(path),
        "task": task,
        "split": "train",
        "seed": seed,
        "steps": steps,
        "hdf5_sha256": f"sha-{task}",
    }


def test_full_episode_windows_are_causal_and_cover_terminal_padding(tmp_path):
    episodes = [
        _episode(tmp_path / f"{task}.pt", task, seed=100 + index)
        for index, task in enumerate(TASKS)
    ]
    dataset = FullEpisodeActionWindows(
        episodes,
        {"a_mean": torch.zeros(8), "a_std": torch.ones(8)},
        split="train",
        horizon=3,
    )
    cold = dataset[(0, 0)]
    assert torch.equal(cold["actions"], torch.zeros_like(cold["actions"]))
    assert cold["source_index"].item() == 0
    middle = dataset[(0, 2)]
    saved = torch.load(episodes[0]["path"], weights_only=False)
    assert torch.equal(middle["actions"][0], torch.zeros((4, 8)))
    assert torch.equal(middle["actions"][1], saved["executed_actions"][0])
    assert torch.equal(middle["actions"][2], saved["executed_actions"][1])
    terminal = dataset[(0, 3)]
    assert terminal["action_step_mask"].tolist() == [True, False, False]
    assert torch.equal(terminal["joint_actions"][1], terminal["joint_actions"][0])


def test_full_episode_sampler_is_balanced_and_resume_stable(tmp_path):
    episodes = [
        _episode(tmp_path / f"{task}.pt", task, seed=200 + index)
        for index, task in enumerate(TASKS)
    ]
    dataset = FullEpisodeActionWindows(
        episodes,
        {"a_mean": torch.zeros(8), "a_std": torch.ones(8)},
        split="train",
    )
    complete = list(
        ExactFiveTaskFullEpisodeSampler(
            dataset, updates=5, rows_per_task=2, seed=17
        )
    )
    resumed = list(
        ExactFiveTaskFullEpisodeSampler(
            dataset, updates=5, rows_per_task=2, seed=17, start_update=3
        )
    )
    assert resumed == complete[3:]
    for batch in complete:
        assert len(batch) == 10
        tasks = [dataset.episodes[episode]["task_index"] for episode, _ in batch]
        assert {task: tasks.count(task) for task in range(5)} == {
            task: 2 for task in range(5)
        }
    first_two = complete[:2]
    for task in range(5):
        observed = {
            request
            for batch in first_two
            for request in batch
            if dataset.episodes[request[0]]["task_index"] == task
        }
        assert observed == set(dataset.requests_by_task[task])


def test_spatial_query_bridge_has_direct_gradients_and_masks_absent_views():
    from before_we_act.action_generator.spatial_bridge import SpatialQueryBridge

    torch.manual_seed(5)
    bridge = SpatialQueryBridge()
    belief = TeamBeliefState(
        tokens=torch.randn(2, 16, 96),
        agent_tokens=torch.randn(2, 4, 96),
        consensus_token=torch.randn(2, 96),
        uncertainty=torch.zeros(2, 1),
        agent_mask=torch.tensor([[True, True, False, False], [True] * 4]),
    )
    spatial = torch.randn(2, 5, 48, 768)
    view_mask = torch.tensor(
        [[True, True, True, False, False], [True] * 5], dtype=torch.bool
    )
    tokens, mask = bridge(belief, spatial, view_mask)
    assert tokens.shape == (2, 37, 96)
    assert mask.shape == (2, 37)
    changed = spatial.clone()
    changed[0, 3:] = torch.randn_like(changed[0, 3:]) * 1000
    masked_tokens, _ = bridge(belief, changed, view_mask)
    assert torch.allclose(tokens[0], masked_tokens[0], atol=1e-6, rtol=1e-6)
    tokens.square().mean().backward()
    for name, parameter in bridge.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
