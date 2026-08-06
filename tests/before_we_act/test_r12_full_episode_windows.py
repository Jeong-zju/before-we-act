from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from before_we_act.data.full_episode_windows import (
    ExactFiveTaskFullEpisodeSampler,
    FULL_EPISODE_PROTOCOL,
    FullEpisodeActionWindows,
    SequentialFullEpisodeSampler,
    TaskWeightedFullEpisodeSampler,
)
from before_we_act.data.raw_team_windows import TASKS
from before_we_act.contracts import TeamBeliefState


def _episode(path: Path, task: str, seed: int, steps: int = 4) -> dict:
    agents = 2 + TASKS.index(task) % 3
    view_mask = np.zeros((steps, 5), dtype=np.bool_)
    view_mask[:, : agents + 1] = True
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = 1
        handle.attrs["round"] = "R12-R4"
        handle.attrs["metadata_json"] = json.dumps(
            {
                "protocol_variant": FULL_EPISODE_PROTOCOL,
                "task": task,
                "split": "train",
                "seed": seed,
                "steps": steps,
                "hdf5_sha256": "a" * 64,
            }
        )
        handle.create_dataset(
            "visual",
            data=np.arange(steps * 16 * 15, dtype=np.float16).reshape(
                steps, 16, 15
            ),
        )
        handle.create_dataset("view_mask", data=view_mask)
        handle.create_dataset(
            "qpos",
            data=np.arange(steps * 4 * 9, dtype=np.float32).reshape(
                steps, 4, 9
            ),
        )
        actions = np.arange(steps * 4 * 8, dtype=np.float32).reshape(
            steps, 4, 8
        )
        handle.create_dataset("executed_actions", data=actions)
        handle.create_dataset("commanded_actions", data=actions + 0.5)
        handle.create_dataset("agent_mask", data=np.arange(4) < agents)
        handle.create_dataset(
            "spatial_tokens",
            data=np.zeros((steps, 5, 48, 768), dtype=np.float16),
        )
        handle.create_dataset("spatial_view_mask", data=view_mask)
    return {
        "path": str(path),
        "task": task,
        "split": "train",
        "seed": seed,
        "steps": steps,
        "hdf5_sha256": "a" * 64,
    }


def test_full_episode_windows_are_causal_and_cover_terminal_padding(tmp_path):
    episodes = [
        _episode(tmp_path / f"{task}.hdf5", task, seed=100 + index)
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
    assert torch.equal(middle["actions"][0], torch.zeros((4, 8)))
    with h5py.File(episodes[0]["path"], "r") as saved:
        expected_0 = torch.from_numpy(np.asarray(saved["executed_actions"][0]))
        expected_1 = torch.from_numpy(np.asarray(saved["executed_actions"][1]))
    assert torch.equal(middle["actions"][1], expected_0)
    assert torch.equal(middle["actions"][2], expected_1)
    terminal = dataset[(0, 3)]
    assert terminal["action_step_mask"].tolist() == [True, False, False]
    assert torch.equal(terminal["joint_actions"][1], terminal["joint_actions"][0])


def test_full_episode_sampler_is_balanced_and_resume_stable(tmp_path):
    episodes = [
        _episode(tmp_path / f"{task}.hdf5", task, seed=200 + index)
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


def test_task_weighted_sampler_preserves_every_task_and_resume(tmp_path):
    episodes = [
        _episode(tmp_path / f"weighted_{task}.hdf5", task, seed=250 + index)
        for index, task in enumerate(TASKS)
    ]
    dataset = FullEpisodeActionWindows(
        episodes,
        {"a_mean": torch.zeros(8), "a_std": torch.ones(8)},
        split="train",
    )
    weights = {task: 1 for task in TASKS}
    weights["camera_alignment"] = 2
    weights["three_robots_stack_cube"] = 6
    complete = list(
        TaskWeightedFullEpisodeSampler(
            dataset, updates=5, rows_per_task=weights, seed=29
        )
    )
    resumed = list(
        TaskWeightedFullEpisodeSampler(
            dataset,
            updates=5,
            rows_per_task=weights,
            seed=29,
            start_update=3,
        )
    )
    assert resumed == complete[3:]
    for batch in complete:
        tasks = [dataset.episodes[episode]["task_index"] for episode, _ in batch]
        assert len(batch) == sum(weights.values())
        for task_index, task in enumerate(TASKS):
            assert tasks.count(task_index) == weights[task]


def test_spatial_query_bridge_has_direct_gradients_and_masks_absent_views():
    from before_we_act.action_generator.spatial_bridge import SpatialQueryBridge

    torch.manual_seed(5)
    bridge = SpatialQueryBridge(spatial_dim=8)
    belief = TeamBeliefState(
        tokens=torch.randn(2, 16, 96),
        agent_tokens=torch.randn(2, 4, 96),
        consensus_token=torch.randn(2, 96),
        uncertainty=torch.zeros(2, 1),
        agent_mask=torch.tensor([[True, True, False, False], [True] * 4]),
    )
    spatial = torch.randn(2, 5, 48, 8)
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


def test_full_episode_hdf5_reads_post_encoder_grid_and_cold_start_lazily(tmp_path):
    rows = []
    for index, task in enumerate(TASKS):
        hdf5 = tmp_path / f"{task}.hdf5"
        rows.append(_episode(hdf5, task, seed=300 + index))
    dataset = FullEpisodeActionWindows(
        rows,
        {"a_mean": torch.zeros(8), "a_std": torch.ones(8)},
        split="train",
        cache_episodes=2,
    )
    first = dataset[(0, 0)]
    assert first["visual"].shape == (3, 16, 15)
    assert torch.equal(first["actions"], torch.zeros_like(first["actions"]))
    assert first["spatial_tokens"].shape == (5, 48, 768)
    assert first["spatial_view_mask"].tolist() == [True, True, True, False, False]


class _DummyR4Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition_position = nn.Embedding(38, 96)
        self.head = nn.Linear(96, 32)

    def _prediction(self, tokens):
        pooled = tokens.mean(dim=1)
        return self.head(pooled)[:, None].expand(-1, 100, -1)

    def training_loss(self, tokens, token_mask, actions, mask):
        del token_mask
        prediction = self._prediction(tokens)
        loss = (prediction - actions).square().masked_select(mask).mean()
        return {"loss": loss, "mse": loss}

    def sample(self, tokens, token_mask, noise=None):
        del token_mask, noise
        return self._prediction(tokens)


def _r4_config():
    from before_we_act.action_generator.r4_base import R12R4Config
    from before_we_act.spatial_observation import locked_r12_full_episode_observation

    return R12R4Config(
        {
            "candidate_id": "p2",
            "component": {"kind": "act_action_chunk_transformer"},
            "observation": locked_r12_full_episode_observation(),
            "action": {
                "horizon": 100,
                "max_agents": 4,
                "action_dim": 8,
                "belief_dim": 96,
                "normalization": "W10_mean_std_copied_into_R12_checkpoint",
                "normalized_clip": 5.0,
                "num_proposals": 1,
                "condition_tokens": 37,
            },
            "training": {},
        }
    )


def test_r4_generator_bridge_stage_and_core_only_warm_start():
    from before_we_act.action_generator.r4_base import (
        R4JointActionGenerator,
        load_r3_core_warm_start,
    )

    core = _DummyR4Core()
    model = R4JointActionGenerator(_r4_config(), core=core)
    trainable = model.set_training_stage("bridge")
    assert any(name.startswith("bridge.") for name in trainable)
    assert "core.condition_position.weight" in trainable
    assert "core.head.weight" not in trainable
    source_core = _DummyR4Core()
    checkpoint = {
        "model": {
            f"core.{name}": value.clone()
            for name, value in source_core.state_dict().items()
        }
        | {"spatial_gate": torch.tensor(0.0)}
    }
    receipt = load_r3_core_warm_start(model, checkpoint)
    assert receipt["adapter_loaded"] is False
    assert "head.weight" in receipt["loaded_keys"]
    assert torch.equal(model.core.head.weight, source_core.head.weight)
    assert all(parameter.requires_grad for parameter in model.bridge.parameters())
    model.set_training_stage("joint")
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_r12e1_task_film_is_bounded_supplemental_conditioning():
    from before_we_act.action_generator.evolution import (
        R12EvolutionConfig,
        TaskConditionedActionGenerator,
    )
    from before_we_act.spatial_observation import locked_r12_full_episode_observation

    config = R12EvolutionConfig(
        {
            "candidate_id": "p2",
            "component": {"kind": "act_action_chunk_transformer"},
            "observation": locked_r12_full_episode_observation(),
            "action": _r4_config().action,
            "training": {
                "task_film_hidden_dim": 32,
                "task_film_scale": 0.25,
                "agent_slot_scale": 0.25,
            },
            "deployment": {},
        }
    )
    model = TaskConditionedActionGenerator(config, core=_DummyR4Core())
    trainable = model.set_training_stage("bridge")
    assert "task_embedding.weight" in trainable
    assert "task_film.3.weight" in trainable
    assert "agent_slot_embedding" in trainable
    assert "core.head.weight" not in trainable
    belief = TeamBeliefState(
        tokens=torch.randn(2, 16, 96),
        agent_tokens=torch.randn(2, 4, 96),
        consensus_token=torch.randn(2, 96),
        uncertainty=torch.zeros(2, 1),
        agent_mask=torch.tensor([[True, True, True, False], [True] * 4]),
    )
    spatial = torch.randn(2, 5, 48, 768)
    view_mask = torch.tensor(
        [[True, True, True, True, False], [True] * 5], dtype=torch.bool
    )
    tokens, mask = model.condition(
        belief, spatial, view_mask, torch.tensor([1, 2])
    )
    assert tokens.shape == (2, 37, 96)
    assert mask.shape == (2, 37)
    loss = tokens.square().mean()
    loss.backward()
    assert model.task_film[3].weight.grad is not None
    assert model.agent_slot_embedding.grad is not None
    assert model.agent_slot_embedding.grad.abs().sum() > 0
    with torch.no_grad():
        original_slot = model.agent_slot_embedding.clone()
        model.agent_slot_embedding.zero_()
        without_slot, _ = model.condition(
            belief, spatial, view_mask, torch.tensor([1, 2])
        )
        model.agent_slot_embedding.copy_(original_slot)
    # Slot identity changes present agent tokens, but never invents a token for
    # a masked/absent agent.
    slot_delta = tokens[:, 16:20] - without_slot[:, 16:20]
    assert slot_delta[0, :3].abs().sum() > 0
    assert torch.equal(slot_delta[0, 3], torch.zeros_like(slot_delta[0, 3]))
    proposals = model.sample(
        belief,
        spatial_tokens=spatial,
        spatial_view_mask=view_mask,
        task_index=torch.tensor([1, 2]),
    )
    assert proposals.actions.shape == (2, 1, 4, 100, 8)
    assert torch.equal(
        proposals.actions[0, :, 3], torch.zeros_like(proposals.actions[0, :, 3])
    )


def test_r4_warm_start_preserves_old_condition_positions_and_new_suffix():
    from before_we_act.action_generator.r4_base import (
        R4JointActionGenerator,
        load_r3_core_warm_start,
    )

    model = R4JointActionGenerator(_r4_config(), core=_DummyR4Core())
    original_suffix = model.core.condition_position.weight.detach()[22:].clone()
    old_positions = torch.arange(22 * 96, dtype=torch.float32).reshape(22, 96)
    source_head = _DummyR4Core().head.state_dict()
    checkpoint = {
        "model": {
            "core.condition_position.weight": old_positions,
            "core.head.weight": source_head["weight"],
            "core.head.bias": source_head["bias"],
        }
    }
    receipt = load_r3_core_warm_start(model, checkpoint)
    torch.testing.assert_close(
        model.core.condition_position.weight[:22], old_positions, rtol=0, atol=0
    )
    torch.testing.assert_close(
        model.core.condition_position.weight[22:], original_suffix, rtol=0, atol=0
    )
    partial = receipt["partial_position_prefix"]["condition_position.weight"]
    assert partial["copied_prefix_tokens"] == 22
    assert partial["new_suffix_tokens"] == 16
    assert "condition_position.weight" not in receipt["skipped_source_keys"]


def test_r4_history_augmentation_never_modifies_recovery_source():
    from before_we_act.train_action_generator_r4 import (
        robustify_source_aware_history,
    )

    batch = {
        "actions": torch.arange(4 * 3 * 4 * 8, dtype=torch.float32).reshape(
            4, 3, 4, 8
        ),
        "agent_mask": torch.ones(4, 4, dtype=torch.bool),
        "source_index": torch.tensor([0, 0, 1, 1]),
    }
    training = {
        "history_augmentation_probability": 1.0,
        "history_augmentation_ramp_updates": 1,
        "history_noise_scale": 0.25,
    }
    changed, metrics = robustify_source_aware_history(
        batch,
        {"a_std": torch.ones(8)},
        training,
        update=10,
        seed=3,
    )
    assert torch.equal(changed[2:], batch["actions"][2:])
    assert metrics["recovery_fraction"] == 0.5
    assert metrics["recovery_history_modified_fraction"] == 0.0


def test_r4_rng_checkpoint_restores_python_numpy_and_torch():
    import random

    import numpy as np

    from before_we_act.train_action_generator_r4 import (
        capture_rng_state,
        restore_rng_state,
    )

    random.seed(17)
    np.random.seed(18)
    torch.manual_seed(19)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    random.seed(117)
    np.random.seed(118)
    torch.manual_seed(119)
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_sequential_full_episode_sampler_visits_every_row_once(tmp_path):
    rows = [
        _episode(tmp_path / f"sequential_{task}.hdf5", task, seed=400 + index)
        for index, task in enumerate(TASKS)
    ]
    dataset = FullEpisodeActionWindows(
        rows,
        {"a_mean": torch.zeros(8), "a_std": torch.ones(8)},
        split="train",
    )
    batches = list(SequentialFullEpisodeSampler(dataset, batch_size=7))
    flattened = [request for batch in batches for request in batch]
    assert len(flattened) == len(dataset)
    assert len(set(flattened)) == len(dataset)
