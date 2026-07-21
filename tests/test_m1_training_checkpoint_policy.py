from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from models.wam import (
    NormalizationStats,
    RWMARConfig,
    RWMARWorldModel,
    StatefulActionFlow,
    StatefulActionFlowConfig,
)
from models.wam_multimodal import (
    FrozenResNet18Config,
    FrozenResNet18Encoder,
    LatentWAM,
    LatentWAMConfig,
    VisionEncoderOutput,
    default_resnet18_weights_path,
    sha256_file,
)
from policies.multimodal_joint_wam import (
    MultimodalJointWAMPolicy,
    MultimodalJointWAMPolicyConfig,
)
from train.m1_checkpointing import (
    checkpoint_tree_sha256,
    load_m1_checkpoint,
    save_m1_checkpoint,
)
from train.m1_training import (
    M1CausalPairWeights,
    M1FlowObjectiveConfig,
    M1LossWeights,
    M1StageConfig,
    M1StateCausalPairWeights,
    _causal_pair_visual_gradient_scope,
    _state_causal_pair_gradient_scope,
    action_chunk_required_keys,
    configure_stage,
    m1_batch_loss,
    m1_batch_required_keys,
    m1_causal_pair_loss,
    m1_state_causal_pair_loss,
)


TASKS = (
    "cooperative_stop",
    "visual_event_stop",
    "visual_target_select",
    "visual_obstacle_avoid",
)


def _stats(*, state_offset: float = 0.0) -> NormalizationStats:
    return NormalizationStats(
        state_mean=np.full(22, state_offset, dtype=np.float32),
        state_std=np.ones(22, dtype=np.float32),
        action_mean=np.zeros(8, dtype=np.float32),
        action_std=np.ones(8, dtype=np.float32),
        delta_mean=np.zeros(22, dtype=np.float32),
        delta_std=np.ones(22, dtype=np.float32),
        reward_mean=np.zeros(1, dtype=np.float32),
        reward_std=np.ones(1, dtype=np.float32),
    )


def _world(stats: NormalizationStats | None = None) -> RWMARWorldModel:
    return RWMARWorldModel(
        RWMARConfig(
            history_horizon=2,
            train_forecast_horizon=8,
            planning_horizon=8,
            encoder_hidden_dim=16,
            gru_hidden_dim=8,
            gru_layers=1,
        ),
        stats or _stats(),
    )


def _flow(
    world: RWMARWorldModel, stats: NormalizationStats | None = None
) -> StatefulActionFlow:
    return StatefulActionFlow(
        StatefulActionFlowConfig(
            feature_dim=world.planning_feature_dim,
            hidden_dim=16,
            hidden_layers=1,
            time_embedding_dim=4,
            anchor_hidden_dim=8,
            anchor_hidden_layers=1,
        ),
        stats or _stats(),
    )


class _TinyFrozenVision(nn.Module):
    output_dim = 512

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("basis", torch.linspace(0.1, 1.0, 512))
        self.forward_calls = 0

    def forward(self, images: torch.Tensor) -> VisionEncoderOutput:
        self.forward_calls += 1
        leading = images.shape[:-3]
        values = images.float().reshape(-1, 3, images.shape[-2], images.shape[-1])
        scale = values.mean(dim=(1, 2, 3), keepdim=False) / 255.0
        pooled = self.basis.unsqueeze(0) + scale.unsqueeze(1)
        pooled = pooled.reshape(*leading, 512).detach()
        return VisionEncoderOutput(
            spatial_tokens=pooled.unsqueeze(-2), pooled_latent=pooled
        )


def _latent(
    *,
    use_state: bool = True,
    use_vision: bool = True,
    capacity: str = "none",
    vision: nn.Module | None = None,
    stats: NormalizationStats | None = None,
) -> LatentWAM:
    return LatentWAM(
        LatentWAMConfig(
            task_vocabulary=TASKS,
            use_state=use_state,
            use_vision=use_vision,
            capacity_control=capacity,
            task_embedding_dim=8,
            fusion_hidden_dim=32,
            future_hidden_dim=32,
            future_action_hidden_dim=16,
        ),
        _world(stats),
        vision,
    )


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "states": torch.randn(batch_size, 2, 22),
        "state_valid_mask": torch.ones(batch_size, 2, dtype=torch.bool),
        "past_actions": torch.randn(batch_size, 1, 8),
        "images": torch.randint(
            0, 256, (batch_size, 2, 1, 3, 16, 16), dtype=torch.uint8
        ),
        "task_index": torch.ones(batch_size, dtype=torch.long),
        "action_targets": torch.randn(batch_size, 8, 8).clamp(-1.0, 1.0),
        "future_images": torch.randint(
            0, 256, (batch_size, 4, 1, 3, 16, 16), dtype=torch.uint8
        ),
        "future_image_novelty_mask": torch.ones(batch_size, 4, 1, dtype=torch.bool),
        "future_states": torch.randn(batch_size, 8, 22),
        "future_horizons": torch.tensor((1, 2, 4, 8)).repeat(batch_size, 1),
    }


def _causal_pair_batch(pair_count: int = 2) -> dict[str, torch.Tensor]:
    factual = _batch(batch_size=pair_count)
    pair = {
        name: torch.stack((value, value.clone()), dim=1)
        for name, value in factual.items()
        if name
        in {
            "states",
            "state_valid_mask",
            "past_actions",
            "images",
            "task_index",
            "action_targets",
        }
    }
    pair["images"][:, 0].zero_()
    pair["images"][:, 1].fill_(255)
    pair["image_valid_mask"] = torch.ones(pair_count, 2, 2, 1, dtype=torch.bool)
    pair["action_targets"].zero_()
    pair["action_targets"][:, 0, :, 0] = -0.75
    pair["action_targets"][:, 1, :, 0] = 0.75
    return pair


def _state_causal_pair_batch(pair_count: int = 2) -> dict[str, torch.Tensor]:
    states = torch.zeros(pair_count, 2, 4, 22)
    states[:, 0, -1, (0, 11)] = -0.1
    states[:, 1, -1, (0, 11)] = 0.1
    actions = torch.zeros(pair_count, 2, 8, 8)
    actions[:, 0, :, (0, 4)] = 0.05
    actions[:, 1, :, (0, 4)] = -0.05
    images = torch.randint(0, 256, (pair_count, 2, 1, 3, 16, 16), dtype=torch.uint8)
    return {
        "states": states,
        "state_valid_mask": torch.ones(pair_count, 2, 4, dtype=torch.bool),
        "past_actions": torch.zeros(pair_count, 2, 3, 8),
        "past_action_valid_mask": torch.ones(pair_count, 2, 3, dtype=torch.bool),
        "images": torch.stack((images, images.clone()), dim=1),
        "image_valid_mask": torch.ones(pair_count, 2, 2, 1, dtype=torch.bool),
        "task_index": torch.full((pair_count, 2), 2, dtype=torch.long),
        "action_targets": actions,
    }


@pytest.mark.parametrize(
    ("variant", "use_state", "use_vision", "capacity"),
    (
        ("state_only", True, False, "none"),
        ("vision_only", False, True, "none"),
        ("state_vision_no_future", True, True, "none"),
        ("state_vision_future", True, True, "future_head"),
        ("parameter_matched_mlp", True, True, "action_mlp"),
    ),
)
def test_m1_required_batch_keys_match_variant_and_stage_contract(
    variant: str,
    use_state: bool,
    use_vision: bool,
    capacity: str,
) -> None:
    model = _latent(
        use_state=use_state,
        use_vision=use_vision,
        capacity=capacity,
        vision=_TinyFrozenVision() if use_vision else None,
    )
    stages = (
        M1LossWeights(future_visual_latent=0.0, future_state=0.0),
        M1LossWeights(
            future_visual_latent=1.0 if variant == "state_vision_future" else 0.0,
            future_state=0.0,
        ),
        M1LossWeights(
            future_visual_latent=1.0 if variant == "state_vision_future" else 0.0,
            future_state=0.1,
        ),
    )
    common = {"task_index", "action_targets", "future_horizons"}
    state = {"states", "state_valid_mask", "past_actions"} if use_state else set()
    vision = {"images"} if use_vision else set()
    for stage_index, weights in enumerate(stages):
        expected = common | state | vision
        if use_state and stage_index == 2:
            expected.add("future_states")
        if variant == "state_vision_future" and stage_index >= 1:
            expected.update({"future_images", "future_image_novelty_mask"})
        assert m1_batch_required_keys(model, weights) == expected

    validation = {"task_index", "action_targets"} | state | vision
    assert action_chunk_required_keys(model) == validation


def test_zero_weight_future_targets_are_not_required_or_computed() -> None:
    state_model = _latent(use_state=True, use_vision=False)
    state_flow = _flow(state_model.world_model)
    state_batch = _batch()
    state_batch.pop("future_states")
    zero_future = M1LossWeights(future_visual_latent=0.0, future_state=0.0)
    state_loss = m1_batch_loss(
        state_model,
        state_flow,
        state_batch,
        weights=zero_future,
    )
    assert state_loss.future_state.item() == 0.0
    with pytest.raises(KeyError, match="future_states"):
        m1_batch_loss(
            state_model,
            state_flow,
            state_batch,
            weights=M1LossWeights(future_visual_latent=0.0, future_state=0.1),
        )

    vision = _TinyFrozenVision()
    future_model = _latent(
        use_state=False,
        use_vision=True,
        capacity="future_head",
        vision=vision,
    )
    future_flow = _flow(future_model.world_model)
    visual_batch = _batch()
    visual_batch.pop("future_images")
    visual_batch.pop("future_image_novelty_mask")
    visual_loss = m1_batch_loss(
        future_model,
        future_flow,
        visual_batch,
        weights=zero_future,
    )
    assert visual_loss.future_visual_latent.item() == 0.0
    assert vision.forward_calls == 1
    with pytest.raises(KeyError, match="future_image"):
        m1_batch_loss(
            future_model,
            future_flow,
            visual_batch,
            weights=M1LossWeights(future_visual_latent=1.0, future_state=0.0),
        )


def test_causal_pair_loss_uses_cold_deployed_solver_and_reaches_visual_path() -> None:
    model = _latent(vision=_TinyFrozenVision())
    flow = _flow(model.world_model)
    batch = _causal_pair_batch()
    calls = 0
    observed_tau: list[torch.Tensor] = []
    observed_initial: list[torch.Tensor] = []
    observed_warm: list[torch.Tensor] = []

    def count_solver_calls(
        _module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        _output: object,
    ) -> None:
        nonlocal calls
        calls += 1
        observed_tau.append(inputs[1].detach().clone())
        observed_initial.append(inputs[3].detach().clone())
        observed_warm.append(inputs[4].detach().clone())

    handle = flow.register_forward_hook(count_solver_calls)
    loss = m1_causal_pair_loss(
        model,
        flow,
        batch,
        weights=M1CausalPairWeights(),
    )
    handle.remove()
    assert calls == 4
    expected_tau = (0.0, 0.25, 0.5, 0.75)
    assert all(
        torch.equal(value, torch.full_like(value, expected))
        for value, expected in zip(observed_tau, expected_tau, strict=True)
    )
    assert all(torch.count_nonzero(value).item() == 0 for value in observed_initial)
    assert all(torch.count_nonzero(value).item() == 0 for value in observed_warm)
    assert loss.predicted_actions.shape == (2, 2, 8, 8)
    assert loss.target_delta_rms > 0.0
    assert torch.isfinite(loss.total)

    loss.total.backward()
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().any())
        for parameter in model.resampler.parameters()
    )
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().any())
        for parameter in model.fusion.parameters()
    )
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().any())
        for name, parameter in flow.named_parameters()
        if not name.startswith("anchor_prior.")
    )
    assert all(parameter.grad is None for parameter in flow.anchor_prior.parameters())


def test_causal_pair_loss_rejects_noncausal_pair_contracts() -> None:
    model = _latent(vision=_TinyFrozenVision())
    flow = _flow(model.world_model)
    state_leak = _causal_pair_batch()
    state_leak["states"][0, 1, -1, 0] += 1.0
    with pytest.raises(ValueError, match="states must be bitwise equal"):
        m1_causal_pair_loss(model, flow, state_leak)

    same_rgb = _causal_pair_batch()
    same_rgb["images"][:, 1].copy_(same_rgb["images"][:, 0])
    with pytest.raises(ValueError, match="differing valid RGB"):
        m1_causal_pair_loss(model, flow, same_rgb)

    same_action = _causal_pair_batch()
    same_action["action_targets"][:, 1].copy_(same_action["action_targets"][:, 0])
    with pytest.raises(ValueError, match="controlled action difference"):
        m1_causal_pair_loss(model, flow, same_action)


def test_training_pair_scope_preserves_state_and_flow_gradient_ownership() -> None:
    model = _latent(vision=_TinyFrozenVision())
    flow = _flow(model.world_model)
    with _causal_pair_visual_gradient_scope(model, flow):
        loss = m1_causal_pair_loss(model, flow, _causal_pair_batch())
    loss.total.backward()

    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().any())
        for parameter in model.resampler.parameters()
    )
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().any())
        for parameter in model.fusion.parameters()
    )
    assert all(parameter.grad is None for parameter in model.world_model.parameters())
    assert all(parameter.grad is None for parameter in flow.parameters())


def test_state_causal_pair_loss_uses_execute2_and_reaches_state_path() -> None:
    model = _latent(vision=_TinyFrozenVision())
    flow = _flow(model.world_model)
    batch = _state_causal_pair_batch()
    calls = 0

    def count_solver_calls(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        _output: object,
    ) -> None:
        nonlocal calls
        calls += 1

    handle = flow.register_forward_hook(count_solver_calls)
    with _state_causal_pair_gradient_scope(model):
        loss = m1_state_causal_pair_loss(
            model,
            flow,
            batch,
            weights=M1StateCausalPairWeights(),
        )
    handle.remove()
    assert calls == 4
    assert loss.predicted_actions.shape == (2, 2, 8, 8)
    assert loss.step0_target_delta_rms > 0.0
    assert torch.isfinite(loss.total)

    loss.total.backward()
    assert all(parameter.grad is None for parameter in model.parameters())
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().any())
        for name, parameter in flow.named_parameters()
        if not name.startswith("anchor_prior.")
    )
    assert all(parameter.grad is None for parameter in flow.anchor_prior.parameters())


def test_state_causal_pair_loss_rejects_shortcuts_and_nonfeedback() -> None:
    model = _latent(vision=_TinyFrozenVision())
    flow = _flow(model.world_model)

    past_leak = _state_causal_pair_batch()
    past_leak["past_actions"][0, 1, -1, 0] = 0.1
    with pytest.raises(ValueError, match="past_actions must be bitwise equal"):
        m1_state_causal_pair_loss(model, flow, past_leak)

    rgb_leak = _state_causal_pair_batch()
    rgb_leak["images"][0, 1, -1, 0, 0, 0, 0] ^= 1
    with pytest.raises(ValueError, match="RGB must be bitwise equal"):
        m1_state_causal_pair_loss(model, flow, rgb_leak)

    bad_feedback = _state_causal_pair_batch()
    bad_feedback["action_targets"][:, 0, :, (0, 4)] = -0.05
    bad_feedback["action_targets"][:, 1, :, (0, 4)] = 0.05
    with pytest.raises(ValueError, match="negative feedback"):
        m1_state_causal_pair_loss(model, flow, bad_feedback)

    vision_only = _latent(
        use_state=False,
        use_vision=True,
        vision=_TinyFrozenVision(),
    )
    with pytest.raises(ValueError, match="requires a state model"):
        m1_state_causal_pair_loss(
            vision_only,
            _flow(vision_only.world_model),
            _state_causal_pair_batch(),
        )


def test_training_ablation_inputs_are_structurally_isolated() -> None:
    state_model = _latent(use_vision=False, vision=None)
    state_batch = _batch()
    del state_batch["images"], state_batch["future_images"]
    del state_batch["future_image_novelty_mask"]
    state_loss = m1_batch_loss(
        state_model,
        _flow(state_model.world_model),
        state_batch,
        weights=M1LossWeights(future_visual_latent=0.0),
    )
    assert torch.isfinite(state_loss.total)

    vision_model = _latent(use_state=False, use_vision=True, vision=_TinyFrozenVision())
    vision_batch = _batch()
    for key in ("states", "state_valid_mask", "past_actions", "future_states"):
        del vision_batch[key]
    vision_loss = m1_batch_loss(
        vision_model,
        _flow(vision_model.world_model),
        vision_batch,
        weights=M1LossWeights(future_visual_latent=0.0, future_state=0.0),
    )
    assert torch.isfinite(vision_loss.total)


def test_future_rgb_is_teacher_only_and_horizon_order_is_fail_closed() -> None:
    model = _latent(capacity="future_head", vision=_TinyFrozenVision())
    flow = _flow(model.world_model)
    first = _batch()
    second = dict(first)
    second["future_images"] = torch.full_like(first["future_images"], 255)
    generator_a = torch.Generator().manual_seed(91)
    generator_b = torch.Generator().manual_seed(91)

    loss_a = m1_batch_loss(model, flow, first, generator=generator_a)
    loss_b = m1_batch_loss(model, flow, second, generator=generator_b)

    assert torch.equal(loss_a.endpoint_actions, loss_b.endpoint_actions)
    assert loss_a.future_target_detached and loss_b.future_target_detached
    assert not torch.equal(loss_a.teacher_future_latents, loss_b.teacher_future_latents)
    malformed = dict(first)
    malformed["future_horizons"] = torch.tensor((1, 4, 2, 8)).repeat(2, 1)
    with pytest.raises(ValueError, match="exactly match"):
        m1_batch_loss(model, flow, malformed)


def test_flow_endpoint_exactly_matches_deployed_cold_and_warm_solver() -> None:
    model = _latent(use_vision=False, vision=None)
    flow = _flow(model.world_model)
    batch = _batch(batch_size=4)
    weights = M1LossWeights(
        flow_matching=0.0,
        action_endpoint=1.0,
        action_smoothness=0.0,
        future_visual_latent=0.0,
        future_state=0.0,
    )
    features = model.encode(
        batch["states"],
        batch["past_actions"],
        batch["state_valid_mask"],
        None,
        batch["task_index"],
    ).planning_features

    calls = 0

    def count_solver_calls(
        _module: nn.Module, _inputs: object, _output: object
    ) -> None:
        nonlocal calls
        calls += 1

    handle = flow.register_forward_hook(count_solver_calls)
    cold = m1_batch_loss(
        model,
        flow,
        batch,
        weights=weights,
        flow_objective=M1FlowObjectiveConfig(
            warm_start_probability=0.0, warm_start_noise_std=0.0
        ),
        generator=torch.Generator().manual_seed(7),
    )
    handle.remove()
    # One flow-matching query plus the runtime's four Euler evaluations.
    assert calls == 5
    assert not bool(cold.warm_start_mask.any())
    assert torch.equal(
        cold.flow_initial_normalized,
        torch.zeros_like(cold.flow_initial_normalized),
    )
    torch.testing.assert_close(
        cold.endpoint_actions,
        flow.generate(features, solver_steps=4, solver="euler"),
    )

    warm = m1_batch_loss(
        model,
        flow,
        batch,
        weights=weights,
        flow_objective=M1FlowObjectiveConfig(
            execution_steps=2,
            warm_start_probability=1.0,
            warm_start_noise_std=0.0,
        ),
        generator=torch.Generator().manual_seed(9),
    )
    expected_warm = torch.cat(
        (
            batch["action_targets"][:, :6],
            batch["action_targets"][:, 5:6].expand(-1, 2, -1),
        ),
        dim=1,
    )
    assert bool(warm.warm_start_mask.all())
    torch.testing.assert_close(warm.warm_start_actions, expected_warm)
    torch.testing.assert_close(
        warm.flow_initial_normalized,
        flow.normalize_actions(expected_warm),
    )
    torch.testing.assert_close(
        warm.endpoint_actions,
        flow.generate(
            features,
            initial_actions=expected_warm,
            solver_steps=4,
            solver="euler",
        ),
    )

    warm.action_endpoint.backward()
    assert any(
        parameter.grad is not None and bool((parameter.grad != 0).any())
        for name, parameter in flow.named_parameters()
        if not name.startswith("anchor_prior.")
    )


def test_action_objective_masks_policy_fixed_axes_and_weights_executed_prefix() -> None:
    model = _latent(use_vision=False, vision=None)
    flow = _flow(model.world_model)
    first = _batch(batch_size=2)
    second = {name: value.clone() for name, value in first.items()}
    second["action_targets"][..., 3] = -first["action_targets"][..., 3]
    second["action_targets"][..., 7] = -first["action_targets"][..., 7]
    weights = M1LossWeights(
        flow_matching=0.0,
        action_endpoint=1.0,
        action_smoothness=0.0,
        future_visual_latent=0.0,
        future_state=0.0,
    )
    objective = M1FlowObjectiveConfig(
        warm_start_probability=0.0,
        warm_start_noise_std=0.0,
        executed_prefix_weight=3.0,
    )
    first_loss = m1_batch_loss(
        model,
        flow,
        first,
        weights=weights,
        flow_objective=objective,
        generator=torch.Generator().manual_seed(81),
    )
    second_loss = m1_batch_loss(
        model,
        flow,
        second,
        weights=weights,
        flow_objective=objective,
        generator=torch.Generator().manual_seed(81),
    )

    assert first_loss.supervised_action_mask.tolist() == [
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
    ]
    torch.testing.assert_close(first_loss.action_endpoint, second_loss.action_endpoint)
    error = (first_loss.endpoint_actions - first["action_targets"]).square()
    temporal = torch.tensor([3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    action = first_loss.supervised_action_mask.to(dtype=error.dtype)
    manual = (error * temporal[None, :, None] * action[None, None, :]).sum()
    manual = manual / (error.shape[0] * temporal.sum() * action.sum())
    torch.testing.assert_close(first_loss.action_endpoint, manual)


def test_flow_objective_rejects_non_deployable_controls() -> None:
    with pytest.raises(ValueError, match="warm_start_probability"):
        M1FlowObjectiveConfig(warm_start_probability=1.1)
    with pytest.raises(ValueError, match="solver"):
        M1FlowObjectiveConfig(solver="rk4")
    with pytest.raises(ValueError, match="unique non-negative"):
        M1FlowObjectiveConfig(policy_fixed_action_dims=(3, 3))
    with pytest.raises(ValueError, match="at least one"):
        M1FlowObjectiveConfig(executed_prefix_weight=0.5)

    model = _latent(use_vision=False, vision=None)
    with pytest.raises(ValueError, match="execution_steps"):
        m1_batch_loss(
            model,
            _flow(model.world_model),
            _batch(),
            weights=M1LossWeights(future_visual_latent=0.0, future_state=0.0),
            flow_objective=M1FlowObjectiveConfig(execution_steps=8),
        )
    with pytest.raises(ValueError, match="exceed"):
        m1_batch_loss(
            model,
            _flow(model.world_model),
            _batch(),
            weights=M1LossWeights(future_visual_latent=0.0, future_state=0.0),
            flow_objective=M1FlowObjectiveConfig(policy_fixed_action_dims=(3, 8)),
        )


def test_stage_freeze_plan_and_world_lr_ratio_are_auditable() -> None:
    model = _latent(capacity="future_head", vision=_TinyFrozenVision())
    flow = _flow(model.world_model)
    stage_one = M1StageConfig(
        name="adapter_fusion",
        steps=1,
        learning_rate=1e-3,
        train_future_head=False,
        train_action_flow=False,
        train_world_model=False,
    )
    configure_stage(model, flow, stage_one)
    assert any(parameter.requires_grad for parameter in model.resampler.parameters())
    assert any(parameter.requires_grad for parameter in model.fusion.parameters())
    assert all(
        not parameter.requires_grad for parameter in model.future_head.parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in model.world_model.parameters()
    )
    assert all(not parameter.requires_grad for parameter in flow.parameters())

    stage_three = M1StageConfig(
        name="low_lr_world",
        steps=1,
        learning_rate=1e-3,
        world_learning_rate=5e-5,
        train_future_head=True,
        train_action_flow=True,
        train_world_model=True,
    )
    optimizer = configure_stage(model, flow, stage_three)
    by_role = {group["role"]: group["lr"] for group in optimizer.param_groups}
    assert by_role == {"adapter_action": 1e-3, "legacy_world_low_lr": 5e-5}
    assert all(
        not parameter.requires_grad for parameter in flow.anchor_prior.parameters()
    )
    with pytest.raises(ValueError, match="10x to 20x"):
        replace(stage_three, world_learning_rate=1e-5)


def test_checkpoint_reload_is_self_contained_strict_and_m0_tree_compatible(
    tmp_path: Path,
) -> None:
    official = default_resnet18_weights_path()
    if not official.is_file():
        pytest.skip("cached official ResNet-18 weights are required")
    external = tmp_path / "external-resnet18.pth"
    shutil.copyfile(official, external)
    vision = FrozenResNet18Encoder(
        FrozenResNet18Config(
            external,
            expected_sha256=sha256_file(external),
            resize_shorter_side=32,
            crop_size=32,
        )
    )
    stats = _stats()
    model = _latent(capacity="none", vision=vision, stats=stats)
    flow = _flow(model.world_model, stats)
    legacy_world = _world(stats)
    legacy_flow = _flow(legacy_world, stats)
    destination = tmp_path / "checkpoint"
    save_m1_checkpoint(
        destination,
        model,
        flow,
        legacy_world,
        legacy_flow,
        stats,
        experiment_config={"note": "portable"},
        dataset_manifest={"manifest_sha256": "a" * 64},
        metrics={"loss": np.float32(0.25)},
        provenance={"source": "unit-test"},
        schema_version="wam.multimodal/1.1",
        train_seed=101,
        model_variant="state_vision_no_future",
    )
    external.unlink()

    reloaded, reloaded_flow, legacy, legacy_reloaded_flow, evidence = (
        load_m1_checkpoint(destination, expected_schema_version="wam.multimodal/1.1")
    )
    assert reloaded.vision_encoder.config.weights_path.parent == destination
    assert evidence["schema"]["runtime_inputs"] == [
        "task",
        "past_executed_actions",
        "images.fixed",
        "proprioception",
    ]
    assert torch.equal(flow.action_mean, reloaded_flow.action_mean)
    assert torch.equal(legacy.delta_mean, legacy_world.delta_mean)
    assert torch.equal(legacy_reloaded_flow.action_std, legacy_flow.action_std)

    tree = {
        str(path.relative_to(destination)): _file_sha(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    expected_tree = hashlib.sha256(
        json.dumps(
            dict(sorted(tree.items())), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    assert checkpoint_tree_sha256(destination) == expected_tree

    (destination / "metrics.json").write_text('{"loss": 9}', encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_m1_checkpoint(destination)


def test_checkpoint_rejects_stats_or_variant_misrepresentation(tmp_path: Path) -> None:
    model = _latent(use_vision=False, vision=None)
    flow = _flow(model.world_model)
    legacy_world = _world()
    legacy_flow = _flow(legacy_world)
    with pytest.raises(ValueError, match="variant"):
        save_m1_checkpoint(
            tmp_path / "bad-variant",
            model,
            flow,
            legacy_world,
            legacy_flow,
            _stats(),
            experiment_config={},
            dataset_manifest={},
            metrics={},
            provenance={},
            schema_version="wam.multimodal/1.1",
            train_seed=1,
            model_variant="vision_only",
        )
    with pytest.raises(ValueError, match="normalization"):
        save_m1_checkpoint(
            tmp_path / "bad-stats",
            model,
            flow,
            legacy_world,
            legacy_flow,
            _stats(state_offset=1.0),
            experiment_config={},
            dataset_manifest={},
            metrics={},
            provenance={},
            schema_version="wam.multimodal/1.1",
            train_seed=1,
            model_variant="state_only",
        )


class _PolicyWorld(nn.Module):
    def __init__(self, feature_dim: int = 12) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(history_horizon=2, state_dim=22, action_dim=8)
        self.planning_feature_dim = feature_dim


class _PolicyFlow(nn.Module):
    def __init__(self, feature_dim: int = 12) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(()))
        self.anchor_prior = nn.Linear(feature_dim, 8)
        self.config = SimpleNamespace(feature_dim=feature_dim, action_dim=8, horizon=8)
        self.initial_actions: list[torch.Tensor | None] = []

    def freeze_anchor(self) -> None:
        for parameter in self.anchor_prior.parameters():
            parameter.requires_grad_(False)

    def generate(
        self,
        features: torch.Tensor,
        *,
        initial_actions: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        self.initial_actions.append(
            None if initial_actions is None else initial_actions.detach().clone()
        )
        if initial_actions is not None:
            return initial_actions.clone()
        steps = torch.arange(8, dtype=features.dtype, device=features.device)
        return steps.view(1, 8, 1).expand(features.shape[0], 8, 8) / 10.0

    def anchor_action(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros(features.shape[0], 8, device=features.device)


class _PolicyModel(nn.Module):
    def __init__(
        self,
        *,
        use_state: bool,
        use_vision: bool,
        capacity_control: str = "none",
        feature_dim: int = 12,
    ) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            use_state=use_state,
            use_vision=use_vision,
            capacity_control=capacity_control,
            task_vocabulary=TASKS,
        )
        self.world_model = _PolicyWorld(feature_dim)
        self.planning_feature_dim = feature_dim
        self.vision_encoder = _TinyFrozenVision() if use_vision else None
        self.last_encode: tuple[object, ...] | None = None

    def task_indices(self, task_id: str, *, device: object = None) -> torch.Tensor:
        return torch.tensor([TASKS.index(task_id)], dtype=torch.long, device=device)

    def encode(self, *values: object) -> SimpleNamespace:
        self.last_encode = values
        states, _, _, images, _ = values
        assert (states is not None) is self.config.use_state
        assert (images is not None) is self.config.use_vision
        return SimpleNamespace(
            planning_features=torch.zeros(1, self.planning_feature_dim),
            state_planning_features=torch.zeros(1, self.planning_feature_dim),
        )


def _policy(
    *,
    use_state: bool,
    use_vision: bool,
    capacity: str = "none",
    clock=None,
) -> tuple[MultimodalJointWAMPolicy, _PolicyModel, _PolicyFlow]:
    model = _PolicyModel(
        use_state=use_state, use_vision=use_vision, capacity_control=capacity
    )
    flow = _PolicyFlow()
    return (
        MultimodalJointWAMPolicy(
            model,
            flow,
            _PolicyWorld(),
            _PolicyFlow(),
            **({} if clock is None else {"clock": clock}),
        ),
        model,
        flow,
    )


def _observation(
    *,
    frame: int,
    history: list[np.ndarray],
    include_state: bool = False,
    include_image: bool = True,
    task: str = "visual_event_stop",
) -> dict[str, object]:
    result: dict[str, object] = {
        "task": {"id": task, "text": "respond to visible cue"},
        "past_executed_actions": np.asarray(history, dtype=np.float32).reshape(-1, 8),
    }
    if include_state:
        result["proprioception"] = np.zeros(22, dtype=np.float32)
    if include_image:
        result["images"] = {"fixed": np.full((16, 16, 3), frame, dtype=np.uint8)}
        result["image_frame_indices"] = {"fixed": frame}
    return result


def test_policy_modalities_forbidden_inputs_and_canonical_action_source() -> None:
    state_policy, state_model, _ = _policy(use_state=True, use_vision=False)
    state_action = state_policy.act(
        _observation(frame=0, history=[], include_state=True, include_image=False)
    )
    assert state_action.shape == (8,)
    assert state_model.last_encode is not None
    assert state_policy.action_source == "m1_state_only"
    assert (
        "images.fixed"
        not in state_policy.last_diagnostics["consumed_observation_paths"]
    )
    assert state_policy.last_diagnostics["deadline_mode"] == "direct_state"
    assert state_policy.last_diagnostics["deadline_budget_ms"] == 50.0

    vision_policy, vision_model, _ = _policy(use_state=False, use_vision=True)
    vision_action = vision_policy.act(_observation(frame=0, history=[]))
    assert vision_action.shape == (8,)
    assert vision_model.last_encode is not None
    assert vision_model.last_encode[0] is None
    assert (
        "proprioception"
        not in vision_policy.last_diagnostics["consumed_observation_paths"]
    )
    assert vision_policy.last_diagnostics["deadline_mode"] == "decimated_visual"
    assert vision_policy.last_diagnostics["deadline_budget_ms"] == 100.0

    leaked = _observation(frame=0, history=[])
    leaked["future_images"] = np.zeros((1, 3, 4, 4), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="forbidden observation"):
        vision_policy.act(leaked)
    cue_leaked = _observation(frame=0, history=[])
    cue_leaked["metadata"] = {"cue_variant": {}}
    with pytest.raises(RuntimeError, match="cue_variant"):
        vision_policy.act(cue_leaked)

    matched, _, _ = _policy(use_state=True, use_vision=True, capacity="action_mlp")
    assert matched.canonical_variant == "parameter_matched_mlp"


def test_cooperative_bypass_reports_consumed_visual_frame_contract() -> None:
    policy, _, _ = _policy(use_state=True, use_vision=True)

    class _DirectLegacyPolicy:
        last_diagnostics: dict[str, object] = {}

        def act(self, observation: dict[str, object]) -> np.ndarray:
            assert set(observation) == {"proprioception"}
            self.last_diagnostics = {
                "direct_flow_executed": True,
                "fallback_enabled": False,
                "fallback_reason": "none",
            }
            return np.zeros(8, dtype=np.float32)

    policy._legacy_policy = _DirectLegacyPolicy()
    history: list[np.ndarray] = []

    first = policy.act(
        _observation(
            frame=0,
            history=history,
            include_state=True,
            task="cooperative_stop",
        )
    )
    first_diagnostics = dict(policy.last_diagnostics)
    history.append(first.copy())
    policy.act(
        _observation(
            frame=0,
            history=history,
            include_state=True,
            task="cooperative_stop",
        )
    )

    assert first_diagnostics["visual_frame_index"] == 0
    assert first_diagnostics["new_visual_frame"] is True
    assert policy.last_diagnostics["visual_frame_index"] == 0
    assert policy.last_diagnostics["new_visual_frame"] is False
    assert (
        "image_frame_indices.fixed"
        in policy.last_diagnostics["consumed_observation_paths"]
    )


def test_policy_visual_feature_cache_encodes_once_per_frame_and_reset_clears() -> None:
    policy, model, _ = _policy(use_state=False, use_vision=True)
    encoder = model.vision_encoder
    assert isinstance(encoder, _TinyFrozenVision)
    history: list[np.ndarray] = []

    first = policy.act(_observation(frame=0, history=history))
    history.append(first.copy())
    policy.act(_observation(frame=0, history=history))
    assert encoder.forward_calls == 1
    assert len(policy._vision_features) == len(policy._images) == 1

    policy.act(_observation(frame=1, history=history))
    assert encoder.forward_calls == 2
    assert len(policy._vision_features) == len(policy._images) == 2

    policy.reset()
    assert len(policy._vision_features) == 0
    assert len(policy._images) == 0
    assert len(policy._image_indices) == 0
    policy.act(_observation(frame=0, history=[]))
    assert encoder.forward_calls == 3


def test_policy_visual_feature_cache_rejects_changed_rgb_at_same_index() -> None:
    policy, model, _ = _policy(use_state=False, use_vision=True)
    encoder = model.vision_encoder
    assert isinstance(encoder, _TinyFrozenVision)
    first = policy.act(_observation(frame=0, history=[]))
    changed = _observation(frame=0, history=[first])
    changed["images"] = {"fixed": np.full((16, 16, 3), 255, dtype=np.uint8)}

    with pytest.raises(ValueError, match="changed without a new frame index"):
        policy.act(changed)

    assert encoder.forward_calls == 1


def test_policy_executes_two_then_replans_with_shifted_warm_chunk() -> None:
    policy, _, flow = _policy(use_state=False, use_vision=True)
    history: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for frame in (0, 0, 1, 1):
        action = policy.act(_observation(frame=frame, history=history))
        actions.append(action.copy())
        history.append(action.copy())

    assert len(flow.initial_actions) == 2
    assert flow.initial_actions[0] is None
    expected_shift = (
        torch.arange(2, 8, dtype=torch.float32).view(6, 1).expand(6, 8) / 10.0
    )
    expected_shift = torch.cat(
        (expected_shift, expected_shift[-1:].expand(2, 8)), dim=0
    )
    expected_shift[:, 3] = 1.0
    expected_shift[:, 7] = 1.0
    assert torch.equal(flow.initial_actions[1][0], expected_shift)
    assert actions[0][0] == pytest.approx(0.0)
    assert actions[1][0] == pytest.approx(0.1)
    assert actions[2][0] == pytest.approx(0.2)
    assert policy.last_diagnostics["warm_start_used"] is False


def test_policy_can_cold_start_every_execute_two_replan() -> None:
    model = _PolicyModel(use_state=True, use_vision=True)
    flow = _PolicyFlow()
    policy = MultimodalJointWAMPolicy(
        model,
        flow,
        _PolicyWorld(),
        _PolicyFlow(),
        config=MultimodalJointWAMPolicyConfig(replan_warm_start_enabled=False),
    )
    history: list[np.ndarray] = []
    for step in range(3):
        action = policy.act(
            _observation(
                frame=step // 2,
                history=history,
                include_state=True,
                task="visual_target_select",
            )
        )
        history.append(action.copy())

    assert len(flow.initial_actions) == 2
    assert flow.initial_actions == [None, None]
    assert policy.last_diagnostics["warm_start_used"] is False
    assert policy.last_diagnostics["replan_warm_start_enabled"] is False


def test_policy_deadline_uses_50ms_direct_and_100ms_only_with_vision() -> None:
    state_times = iter((0.0, 0.075))
    state_policy, _, _ = _policy(
        use_state=True,
        use_vision=False,
        clock=lambda: next(state_times),
    )
    state_policy.act(
        _observation(frame=0, history=[], include_state=True, include_image=False)
    )
    assert state_policy.last_diagnostics["latency_ms"] == pytest.approx(75.0)
    assert state_policy.last_diagnostics["deadline_exceeded"] is True

    vision_times = iter((0.0, 0.075))
    vision_policy, _, _ = _policy(
        use_state=False,
        use_vision=True,
        clock=lambda: next(vision_times),
    )
    vision_policy.act(_observation(frame=0, history=[]))
    assert vision_policy.last_diagnostics["latency_ms"] == pytest.approx(75.0)
    assert vision_policy.last_diagnostics["visual_staleness_ms"] == 0.0
    assert vision_policy.last_diagnostics["action_age_ms"] == pytest.approx(75.0)
    assert vision_policy.last_diagnostics["deadline_exceeded"] is False


def test_visual_deadline_includes_frame_staleness_and_current_action_latency() -> None:
    times = iter((0.0, 0.010, 0.050, 0.051, 0.100, 0.180))
    policy, _, _ = _policy(
        use_state=False,
        use_vision=True,
        clock=lambda: next(times),
    )
    history: list[np.ndarray] = []
    for _ in range(3):
        action = policy.act(_observation(frame=0, history=history))
        history.append(action)

    diagnostics = policy.last_diagnostics
    assert diagnostics["visual_staleness_ms"] == pytest.approx(100.0)
    assert diagnostics["latency_ms"] == pytest.approx(80.0)
    assert diagnostics["action_age_ms"] == pytest.approx(180.0)
    assert diagnostics["visual_age_ms"] == diagnostics["action_age_ms"]
    assert diagnostics["deadline_exceeded"] is True


def test_cached_action_inherits_previous_planning_wall_time() -> None:
    times = iter((0.0, 0.080, 0.081, 0.082))
    policy, _, _ = _policy(
        use_state=False,
        use_vision=True,
        clock=lambda: next(times),
    )
    first = policy.act(_observation(frame=0, history=[]))
    policy.act(_observation(frame=0, history=[first]))

    diagnostics = policy.last_diagnostics
    assert diagnostics["nominal_visual_staleness_ms"] == pytest.approx(50.0)
    assert diagnostics["wall_visual_staleness_ms"] == pytest.approx(81.0)
    assert diagnostics["wall_action_age_ms"] == pytest.approx(82.0)
    assert diagnostics["action_age_ms"] == pytest.approx(82.0)
    assert diagnostics["deadline_exceeded"] is False


def test_late_visual_action_is_executed_and_reported_without_fallback() -> None:
    times = iter((0.0, 0.120))
    policy, _, _ = _policy(
        use_state=False,
        use_vision=True,
        clock=lambda: next(times),
    )

    action = policy.act(_observation(frame=0, history=[]))

    assert action.shape == (8,)
    assert policy.last_diagnostics["action_age_ms"] == pytest.approx(120.0)
    assert policy.last_diagnostics["deadline_exceeded"] is True
    assert policy.last_diagnostics["fallback_used"] is False


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
