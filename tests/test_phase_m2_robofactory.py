from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data.robofactory import _camera_alias
from models.wam_multimodal import (
    BlockCausalWAM,
    BlockCausalWAMConfig,
    build_block_causal_attention_mask,
)
from robofactory_rpc import (
    extract_robofactory_multiview_observation,
    extract_robofactory_observation,
    scalar_float,
    split_robofactory_action,
)
from scripts.serve_robofactory_m2_rollout import (
    TASK_MAX_EPISODE_STEPS,
    TASKS,
    _validate_client as _validate_m2_rollout_client,
    _validate_args as _validate_m2_rollout_args,
)
from scripts.run_robofactory_m2_inference import _validate_contract
from train.m2_checkpointing import load_m2_checkpoint, save_m2_checkpoint
from train.m2_training import (
    M2LossWeights,
    _active_agent_weights,
    _masked_mean,
    m2_batch_loss,
    shift_action_chunk,
)
from train.robofactory_multitask_dataset import (
    CoverageTemperatureDistributedSampler,
    RoboFactoryMultitaskDataset,
    encode_task_text,
)


def _config() -> BlockCausalWAMConfig:
    return BlockCausalWAMConfig(
        max_state_dim=72,
        max_action_dim=32,
        num_tasks=2,
        max_agents=4,
        max_cameras=5,
        history_steps=4,
        visual_history_steps=2,
        action_horizon=4,
        future_visual_horizons=(1, 2, 4),
        visual_feature_dim=16,
        max_text_tokens=16,
        d_model=32,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        text_layers=1,
        dropout=0.0,
    )


def _context(config: BlockCausalWAMConfig, *, partial: bool = False) -> dict[str, torch.Tensor]:
    batch = 2
    state_valid = torch.ones(batch, config.history_steps, dtype=torch.bool)
    past_valid = torch.ones(batch, config.history_steps - 1, dtype=torch.bool)
    if partial:
        state_valid[:, :-1] = False
        past_valid[:] = False
    image_valid = torch.ones(
        batch,
        config.visual_history_steps,
        config.max_cameras,
        dtype=torch.bool,
    )
    if partial:
        image_valid[:, :-1] = False
    return {
        "states": torch.randn(batch, config.history_steps, config.max_state_dim),
        "state_valid_mask": state_valid,
        "state_dimension_mask": torch.ones(
            batch, config.max_state_dim, dtype=torch.bool
        ),
        "past_actions": torch.randn(
            batch, config.history_steps - 1, config.max_action_dim
        ),
        "past_action_valid_mask": past_valid,
        "action_dimension_mask": torch.ones(
            batch, config.max_action_dim, dtype=torch.bool
        ),
        "visual_features": torch.randn(
            batch,
            config.visual_history_steps,
            config.max_cameras,
            config.visual_feature_dim,
        ),
        "image_valid_mask": image_valid,
        "camera_agent_index": torch.tensor(
            [[config.max_agents, 0, 1, 2, 3]] * batch,
            dtype=torch.long,
        ),
        "action_horizon_mask": torch.ones(
            batch, config.action_horizon, dtype=torch.bool
        ),
        "task_text_tokens": torch.stack(
            [encode_task_text("Lift the barrier", max_tokens=16)] * batch
        ),
        "task_index": torch.tensor([0, 1]),
        "embodiment_index": torch.tensor([2, 3]),
    }


def test_block_causal_mask_forbids_future_to_action() -> None:
    config = _config()
    mask = build_block_causal_attention_mask(config)
    history = config.history_steps * (3 + config.max_cameras)
    action_start = history
    future_start = history + config.action_horizon
    assert mask.shape == (
        history + 2 * config.action_horizon,
        history + 2 * config.action_horizon,
    )
    assert not bool(mask[action_start, history - 1])
    assert not bool(mask[action_start + 2, action_start + 1])
    assert bool(mask[action_start + 1, action_start + 2])
    assert bool(mask[action_start, future_start])
    assert not bool(mask[future_start, action_start + config.action_horizon - 1])


def test_spatial_visual_grid_uses_bounded_identity_preserving_tokens() -> None:
    config = BlockCausalWAMConfig(
        **{
            **_config().to_dict(),
            "future_visual_horizons": (1, 2, 4),
            "visual_grid_height": 2,
            "visual_grid_width": 3,
        }
    )
    model = BlockCausalWAM(config).eval()
    context = _context(config, partial=True)
    context["visual_features"] = torch.randn(
        2,
        config.visual_history_steps,
        config.max_cameras,
        6,
        config.visual_feature_dim,
    )
    expected_history = (
        config.history_steps * 3
        + config.visual_history_steps * config.max_cameras * 6
    )
    assert model.history_token_count == expected_history
    assert model.block_causal_mask.shape == (
        expected_history + 2 * config.action_horizon,
        expected_history + 2 * config.action_horizon,
    )
    generated = model.generate_actions(context, solver_steps=1)
    assert generated.shape == (2, config.action_horizon, config.max_action_dim)
    assert torch.isfinite(generated).all()


def test_action_is_invariant_to_future_branch_and_condition() -> None:
    torch.manual_seed(4)
    config = _config()
    model = BlockCausalWAM(config).eval()
    context = _context(config)
    actions = torch.randn(2, config.action_horizon, config.max_action_dim)
    flow_time = torch.tensor([0.2, 0.8])
    fast = model(
        **context,
        action_inputs=actions,
        flow_time=flow_time,
        include_future=False,
    ).action_velocity
    first = model(
        **context,
        action_inputs=actions,
        flow_time=flow_time,
        future_action_condition=torch.zeros_like(actions),
        include_future=True,
    ).action_velocity
    second = model(
        **context,
        action_inputs=actions,
        flow_time=flow_time,
        future_action_condition=torch.full_like(actions, 100.0),
        include_future=True,
    ).action_velocity
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    torch.testing.assert_close(fast, first, rtol=1e-5, atol=1e-6)


def test_multiview_tokens_are_not_reduced_by_cross_camera_average() -> None:
    torch.manual_seed(41)
    config = _config()
    model = BlockCausalWAM(config).eval()
    context = _context(config)
    context["visual_features"].zero_()
    contrast = torch.randn(
        2,
        config.visual_history_steps,
        config.visual_feature_dim,
    )
    context["visual_features"][:, :, 0] = contrast
    context["visual_features"][:, :, 1] = -contrast
    actions = torch.zeros(2, config.action_horizon, config.max_action_dim)
    flow_time = torch.full((2,), 0.5)
    multiview = model(
        **context,
        action_inputs=actions,
        flow_time=flow_time,
        include_future=False,
    ).action_velocity
    context["visual_features"].zero_()
    zero_views = model(
        **context,
        action_inputs=actions,
        flow_time=flow_time,
        include_future=False,
    ).action_velocity
    assert not torch.allclose(multiview, zero_views)


def test_reset_history_generation_is_finite() -> None:
    config = _config()
    model = BlockCausalWAM(config).eval()
    generated = model.generate_actions(_context(config, partial=True), solver_steps=2)
    assert generated.shape == (2, config.action_horizon, config.max_action_dim)
    assert torch.isfinite(generated).all()


def test_task_horizon_masks_generated_suffix() -> None:
    config = _config()
    model = BlockCausalWAM(config).eval()
    context = _context(config, partial=True)
    context["action_horizon_mask"][0, 2:] = False
    generated = model.generate_actions(context, solver_steps=1)
    assert bool(generated[0, :2].ne(0).any())
    assert bool(generated[0, 2:].eq(0).all())


def test_differentiable_solver_matches_deployed_generation() -> None:
    torch.manual_seed(8)
    config = _config()
    model = BlockCausalWAM(config).eval()
    context = _context(config, partial=True)
    differentiable = model.integrate_actions(
        context,
        solver_steps=2,
        solver="heun",
        normalized_clip=10.0,
    )
    deployed = model.generate_actions(
        context,
        solver_steps=2,
        solver="heun",
        normalized_clip=10.0,
    )
    # PyTorch selects an inference-only attention kernel under no_grad; the
    # shared solver contract should agree to floating-point roundoff.
    torch.testing.assert_close(differentiable, deployed, rtol=1e-5, atol=1e-6)
    differentiable.square().mean().backward()
    gradient = model.action_velocity_head.weight.grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.abs().sum()) > 0.0


def test_m2_endpoint_loss_uses_deployed_cold_solver() -> None:
    torch.manual_seed(9)
    config = _config()
    model = BlockCausalWAM(config).eval()
    context = _context(config)
    targets = torch.randn(
        2, config.action_horizon, config.max_action_dim
    ).clamp(-3.0, 3.0)
    target_valid = torch.ones(2, config.action_horizon, dtype=torch.bool)
    target_valid[1, 1:] = False
    targets[1, 1:] = 7.0
    batch = {
        **{key: value for key, value in context.items() if key != "visual_features"},
        "action_targets": targets,
        "action_target_valid_mask": target_valid,
        "future_states": torch.zeros(
            2, config.action_horizon, config.max_state_dim
        ),
        "future_state_valid_mask": torch.ones(
            2, config.action_horizon, dtype=torch.bool
        ),
        "future_image_novelty_mask": torch.zeros(
            2,
            len(config.future_visual_horizons),
            config.max_cameras,
            dtype=torch.bool,
        ),
        "future_visual_valid_mask": torch.ones(
            2,
            len(config.future_visual_horizons),
            config.max_cameras,
            dtype=torch.bool,
        ),
    }
    expected = model.integrate_actions(
        context,
        solver_steps=1,
        solver="euler",
        normalized_clip=10.0,
    )
    expected_mask = target_valid[:, :, None].expand_as(targets)
    error = (expected - targets).square()
    expected_loss = torch.stack(
        [
            error[index].masked_select(expected_mask[index]).mean()
            for index in range(error.shape[0])
        ]
    ).mean()
    loss = m2_batch_loss(
        model,
        batch,
        visual_features=context["visual_features"],
        future_visual_targets=torch.zeros(
            2, len(config.future_visual_horizons), 1, config.visual_feature_dim
        ).expand(
            -1,
            -1,
            config.max_cameras,
            -1,
        ),
        weights=M2LossWeights(
            flow_matching=0.0,
            action_endpoint=1.0,
            action_smoothness=0.0,
            future_state=0.0,
            future_visual_latent=0.0,
        ),
        warm_start_probability=0.0,
        warm_start_noise_std=0.0,
        execution_steps=2,
        executed_prefix_weight=1.0,
        solver_steps=1,
        solver="euler",
        normalized_action_clip=10.0,
        generator=torch.Generator().manual_seed(10),
    )
    torch.testing.assert_close(loss.action_endpoint, expected_loss)
    assert float(loss.action_endpoint_executed_prefix.detach()) > 0.0
    assert float(loss.action_endpoint_incomplete_horizon.detach()) > 0.0
    assert float(loss.incomplete_horizon_fraction.detach()) == 0.5
    loss.total.backward()
    assert model.action_velocity_head.weight.grad is not None


def test_utf8_task_text_is_deterministic_and_padded() -> None:
    first = encode_task_text("  共同 抬起 栏杆  ", max_tokens=32)
    second = encode_task_text("共同 抬起 栏杆", max_tokens=32)
    torch.testing.assert_close(first, second)
    assert first.dtype == torch.long
    assert bool(first.gt(0).any())
    assert bool(first.eq(0).any())


def test_masked_loss_is_invariant_to_valid_dimension_count() -> None:
    value = torch.tensor(
        [
            [[4.0, 4.0, 0.0, 0.0]],
            [[4.0, 4.0, 4.0, 4.0]],
        ]
    )
    mask = torch.tensor(
        [
            [[True, True, False, False]],
            [[True, True, True, True]],
        ]
    )
    assert _masked_mean(value, mask).item() == pytest.approx(4.0)


def test_active_agent_weighting_reallocates_without_agent_count_inflation() -> None:
    targets = torch.zeros(1, 4, 32)
    targets[:, 1:, 24] = torch.tensor([0.1, 0.2, 0.3])
    target_valid = torch.ones(1, 4, dtype=torch.bool)
    action_dimensions = torch.ones(1, 32, dtype=torch.bool)
    past_actions = torch.zeros(1, 3, 32)
    past_valid = torch.tensor([[False, False, True]])
    weights, fraction = _active_agent_weights(
        targets,
        target_valid=target_valid,
        action_dimension_mask=action_dimensions,
        past_actions=past_actions,
        past_action_valid_mask=past_valid,
        max_agents=4,
        active_weight=4.0,
        delta_threshold=0.005,
    )
    torch.testing.assert_close(weights.mean(dim=-1), torch.ones(1, 4))
    assert bool((weights[:, :, 3] > weights[:, :, :3].amax(dim=-1)).all())
    assert fraction.item() == pytest.approx(0.25)

    two_agent_dimensions = action_dimensions.clone()
    two_agent_dimensions[:, 16:] = False
    two_agent_targets = torch.zeros_like(targets)
    two_agent_targets[:, 1:, 8] = torch.tensor([0.1, 0.2, 0.3])
    two_agent_weights, _ = _active_agent_weights(
        two_agent_targets,
        target_valid=target_valid,
        action_dimension_mask=two_agent_dimensions,
        past_actions=past_actions,
        past_action_valid_mask=past_valid,
        max_agents=4,
        active_weight=4.0,
        delta_threshold=0.005,
    )
    torch.testing.assert_close(
        two_agent_weights[:, :, :2].mean(dim=-1),
        torch.ones(1, 4),
    )
    assert bool(two_agent_weights[:, :, 2:].eq(0.0).all())


def test_warm_start_shift_stays_inside_each_task_horizon() -> None:
    actions = torch.arange(2 * 6, dtype=torch.float32).reshape(2, 6, 1)
    valid = torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, True, True, True, True],
        ]
    )
    shifted = shift_action_chunk(
        actions,
        execution_steps=2,
        action_horizon_mask=valid,
    )
    assert shifted[:, :, 0].tolist() == [
        [2.0, 3.0, 3.0, 3.0, 0.0, 0.0],
        [8.0, 9.0, 10.0, 11.0, 11.0, 11.0],
    ]


def test_coverage_temperature_sampler_covers_once_then_uses_sqrt_mix() -> None:
    class FakeDataset:
        datasets = (range(4), range(16))
        _offsets = [0, 4, 20]
        contracts = (
            SimpleNamespace(task_id="small"),
            SimpleNamespace(task_id="large"),
        )

        def __len__(self) -> int:
            return 20

    fake = FakeDataset()
    sampler = CoverageTemperatureDistributedSampler(
        fake,  # type: ignore[arg-type]
        samples_per_epoch=24,
        coverage_epochs=1,
        temperature_alpha=0.5,
        seed=7,
    )
    first = list(iter(sampler))
    assert set(range(20)).issubset(first)
    summary = sampler.summary()
    assert summary["task_probabilities"]["small"] == pytest.approx(1.0 / 3.0)
    assert summary["task_probabilities"]["large"] == pytest.approx(2.0 / 3.0)
    sampler.set_epoch(1)
    second = list(iter(sampler))
    assert len(second) == 24
    assert all(0 <= index < 20 for index in second)


def test_single_and_multi_agent_workspace_cameras_share_canonical_name() -> None:
    assert _camera_alias("head_camera") == "global"
    assert _camera_alias("head_camera_global") == "global"


def test_four_agent_rpc_preserves_native_agent_order_and_camera() -> None:
    order = tuple(f"panda-{index}" for index in range(4))
    observation = {
        "agent": {
            name: {
                "qpos": np.full((1, 9), index, dtype=np.float32),
                "qvel": np.full((1, 9), index + 0.5, dtype=np.float32),
            }
            for index, name in enumerate(order)
        },
        "sensor_data": {
            "head_camera": {"rgb": np.zeros((1, 8, 12, 3), dtype=np.uint8)}
        },
    }
    state, rgb = extract_robofactory_observation(
        observation, agent_order=order, camera_name="head_camera"
    )
    assert state.shape == (72,)
    assert rgb.shape == (8, 12, 3)
    assert state[0] == 0.0 and state[18] == 1.0 and state[54] == 3.0
    split = split_robofactory_action(np.arange(32, dtype=np.float32), agent_order=order)
    assert list(split) == list(order)
    np.testing.assert_array_equal(split["panda-3"], np.arange(24, 32))
    assert scalar_float(np.asarray([1.25], dtype=np.float32), name="reward") == 1.25


def test_four_agent_rpc_retains_global_and_every_agent_camera() -> None:
    order = tuple(f"panda-{index}" for index in range(4))
    camera_sources = {
        "global": "head_camera_global",
        **{
            f"agent_{index}": f"head_camera_agent{index}"
            for index in range(4)
        },
    }
    observation = {
        "agent": {
            name: {
                "qpos": np.zeros((1, 9), dtype=np.float32),
                "qvel": np.zeros((1, 9), dtype=np.float32),
            }
            for name in order
        },
        "sensor_data": {
            source: {
                "rgb": np.full(
                    (1, 8, 12, 3),
                    fill_value=index,
                    dtype=np.uint8,
                )
            }
            for index, source in enumerate(camera_sources.values())
        },
    }
    state, images = extract_robofactory_multiview_observation(
        observation,
        agent_order=order,
        camera_names=camera_sources,
    )
    assert state.shape == (72,)
    assert list(images) == list(camera_sources)
    assert [int(value[0, 0, 0]) for value in images.values()] == list(range(5))


def test_m2_contract_accepts_json_sorted_camera_mappings() -> None:
    camera_order = ["global", "agent_0", "agent_1"]
    runtime = [
        {
            "task_id": "lift_barrier",
            "task_text": "Lift the barrier together",
            "state_dim": 36,
            "action_dim": 16,
            "agent_count": 2,
            "camera_order": camera_order,
            "action_codec": {
                "metadata": {"agent_order": ["panda-0", "panda-1"]}
            },
        }
    ]
    contract = {
        "task_id": "lift_barrier",
        "task_text": "Lift the barrier together",
        "state_dim": 36,
        "action_dim": 16,
        "agent_count": 2,
        "agent_order": ["panda-0", "panda-1"],
        "camera_order": camera_order,
        "camera_sources": {
            "agent_0": "head_camera_agent0",
            "agent_1": "head_camera_agent1",
            "global": "head_camera_global",
        },
        "camera_shapes": {
            "agent_0": [240, 320, 3],
            "agent_1": [240, 320, 3],
            "global": [240, 320, 3],
        },
        "control_mode": "pd_joint_pos",
    }
    assert _validate_contract(contract, runtime=runtime)["task_id"] == "lift_barrier"
    assert _validate_contract(
        contract,
        runtime=runtime,
        expected_image_shape=(240, 320, 3),
    )["task_id"] == "lift_barrier"
    with pytest.raises(ValueError, match="native camera resolution"):
        _validate_contract(
            contract,
            runtime=runtime,
            expected_image_shape=(480, 640, 3),
        )


def _m2_rollout_args(task: str, *, max_steps: int) -> argparse.Namespace:
    return argparse.Namespace(
        task=task,
        port=8872,
        episodes=1,
        seed_start=900,
        max_steps=max_steps,
        video_fps=20,
        socket_timeout=600.0,
        allow_remote=False,
        host="127.0.0.1",
    )


def test_m2_rollout_uses_native_task_episode_limits() -> None:
    assert set(TASK_MAX_EPISODE_STEPS) == set(TASKS)
    _validate_m2_rollout_args(
        _m2_rollout_args("LiftBarrier-rf", max_steps=500)
    )
    _validate_m2_rollout_args(
        _m2_rollout_args("LongPipelineDelivery-rf", max_steps=1000)
    )
    with pytest.raises(ValueError, match=r"LiftBarrier-rf.*\[1,500\]"):
        _validate_m2_rollout_args(
            _m2_rollout_args("LiftBarrier-rf", max_steps=501)
        )
    with pytest.raises(
        ValueError,
        match=r"LongPipelineDelivery-rf.*\[1,1500\]",
    ):
        _validate_m2_rollout_args(
            _m2_rollout_args("LongPipelineDelivery-rf", max_steps=1501)
        )


def test_m2_rollout_accepts_dense_act_and_rejects_format_source_mismatch() -> None:
    contract = {"task_id": "lift_barrier", "future_path": False}
    dense_client = {
        "checkpoint_format": "wam.robofactory.static_rgb_act_moe.checkpoint/1",
        "task_vocabulary": ["lift_barrier", "long_pipeline_delivery"],
        "future_path": False,
        "policy": {"action_source": "static_rgb_dino_act_dense"},
    }

    assert _validate_m2_rollout_client(
        dense_client,
        contract=contract,
    ) == dense_client
    with pytest.raises(RuntimeError, match="supported direct action source"):
        _validate_m2_rollout_client(
            {
                **dense_client,
                "checkpoint_format": "wam.robofactory.m2.checkpoint/5",
            },
            contract=contract,
        )


def test_dataset_rejects_non_robofactory_profile() -> None:
    fake = SimpleNamespace(
        schema_profile="custom_scene",
        schema_version="wam.multimodal/1.1",
        action_codec=object(),
        action_domain="canonical_unit_action",
    )
    with pytest.raises(ValueError, match="only native RoboFactory"):
        RoboFactoryMultitaskDataset._validate_manifest(fake)  # type: ignore[arg-type]


def test_dataset_rejects_unbound_native_source(tmp_path) -> None:
    fake = SimpleNamespace(
        raw_manifest={"source": {}},
        manifest_path=tmp_path / "training_manifest.json",
        state_dim=18,
        action_dim=8,
    )
    with pytest.raises(ValueError, match="conversion manifest path"):
        RoboFactoryMultitaskDataset._validate_native_source(  # type: ignore[arg-type]
            fake,
            "pick_meat",
            expected_camera_order=("global", "agent_0"),
        )


def test_m2_checkpoint_strict_round_trip(tmp_path) -> None:
    config = _config()
    model = BlockCausalWAM(config)
    runtime = [
        {
            "task_id": task_id,
            "task_text": task_id,
            "task_index": index,
            "state_dim": 36,
            "action_dim": 16,
            "action_horizon": 4,
            "agent_count": 2,
            "camera_order": ["global", "agent_0", "agent_1"],
            "camera_slot_indices": [0, 1, 2],
            "camera_agent_indices": [4, 0, 1],
            "state_mean": [0.0] * 36,
            "state_std": [1.0] * 36,
            "action_mean": [0.0] * 16,
            "action_std": [1.0] * 16,
            "action_codec": {},
        }
        for index, task_id in enumerate(("task_a", "task_b"))
    ]
    summary = save_m2_checkpoint(
        tmp_path / "checkpoint",
        model=model,
        task_runtime=runtime,
        vision_identity={"family": "unit_test", "output_dim": 16},
        action_generation={
            "solver_steps": 1,
            "solver": "euler",
            "normalized_action_clip": 10.0,
            "execution_steps": 2,
            "warm_start": False,
        },
        action_objective={
            "tail_windows": "repeat_last_with_validity_masks",
            "visual_prefix_windows": "left_zero_pad_with_validity_mask",
            "task_horizons": "max_tensor_with_task_validity_masks",
            "loss_reduction": "per_sample_valid_element_mean",
            "executed_prefix_weight": 4.0,
        },
        training={"smoke": True},
        metrics={"loss": 1.0},
    )
    restored, restored_runtime, schema = load_m2_checkpoint(
        summary["checkpoint"]
    )
    assert schema["format_version"] == "wam.robofactory.m2.checkpoint/5"
    assert schema["action_space"] == "per_task_zscore_canonical_unit_action"
    assert [value["task_id"] for value in restored_runtime] == ["task_a", "task_b"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name])
