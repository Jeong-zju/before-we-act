from __future__ import annotations

import pytest
import torch

from before_we_act.care_belief import CAREBeliefOutput
from before_we_act.care_belief_v2 import (
    CAREBeliefV2Config,
    CAREBeliefV2Head,
    CARELossV2Config,
    PrefixCandidateActionEncoder,
    canonical_target_scale,
    care_v2_training_loss,
    reference_ranking_loss,
    robust_task_component_scales,
)


def small_config(**kwargs: object) -> CAREBeliefV2Config:
    values: dict[str, object] = {
        "d_model": 32,
        "action_horizon": 20,
        "action_tokens": 4,
        "action_width": 16,
        "heads": 4,
        "layers": 1,
        "dropout": 0.0,
        "action_prefix_steps": 1,
    }
    values.update(kwargs)
    return CAREBeliefV2Config(**values)


def test_prefix_encoder_ignores_unexecuted_tail() -> None:
    torch.manual_seed(3)
    config = small_config(action_prefix_steps=1)
    encoder = PrefixCandidateActionEncoder(config).eval()
    chunks = torch.randn(2, 6, config.action_horizon, config.action_dim)
    changed = chunks.clone()
    changed[:, :, 1:] += 100.0
    first = encoder(chunks)
    second = encoder(changed)
    assert torch.allclose(first, second)
    assert encoder.sample_indices.tolist() == [0]


def test_prefix_encoder_supports_a_matching_multi_step_window() -> None:
    config = small_config(action_prefix_steps=4)
    encoder = PrefixCandidateActionEncoder(config).eval()
    assert encoder.sample_indices.tolist() == [0, 1, 2, 3]


def test_v2_head_keeps_legacy_output_contract() -> None:
    torch.manual_seed(5)
    config = small_config()
    model = CAREBeliefV2Head(config).eval()
    memory = torch.randn(3, 9, config.d_model)
    mask = torch.ones(3, 9, dtype=torch.bool)
    chunks = torch.randn(3, 6, config.action_horizon, config.action_dim)
    horizon = torch.tensor([0, 1, 3])
    output = model(memory, mask, chunks, horizon)
    assert output.quantiles.shape == (3, 6, 3, 5)
    assert output.hard_safety_logit.shape == (3, 6)
    assert torch.equal(output.quantiles[:, 0], torch.zeros_like(output.quantiles[:, 0]))


def test_v2_head_optional_utility_scale_exposes_physical_units() -> None:
    torch.manual_seed(6)
    config = small_config()
    model = CAREBeliefV2Head(config).eval()
    memory = torch.randn(2, 7, config.d_model)
    mask = torch.ones(2, 7, dtype=torch.bool)
    chunks = torch.randn(2, 6, config.action_horizon, config.action_dim)
    horizon = torch.tensor([0, 1])
    raw = model(memory, mask, chunks, horizon)
    scale = torch.tensor([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]])
    physical = model(
        memory, mask, chunks, horizon, utility_scale=scale
    )
    torch.testing.assert_close(
        physical.quantiles[:, 1:], raw.quantiles[:, 1:] * scale[:, None, :, None]
    )
    assert torch.equal(physical.quantiles[:, 0], torch.zeros_like(physical.quantiles[:, 0]))
    torch.testing.assert_close(physical.hard_safety_logit, raw.hard_safety_logit)
    torch.testing.assert_close(physical.candidate_state, raw.candidate_state)


def test_v2_head_rejects_invalid_utility_scale() -> None:
    config = small_config()
    model = CAREBeliefV2Head(config).eval()
    memory = torch.randn(1, 4, config.d_model)
    mask = torch.ones(1, 4, dtype=torch.bool)
    chunks = torch.randn(1, 6, config.action_horizon, config.action_dim)
    horizon = torch.tensor([0])
    with pytest.raises(ValueError):
        model(memory, mask, chunks, horizon, utility_scale=torch.ones(1, 2))
    with pytest.raises(ValueError):
        model(
            memory,
            mask,
            chunks,
            horizon,
            utility_scale=torch.tensor([[1.0, float("nan"), 1.0]]),
        )


def test_utility_scale_cancels_inverse_scale_pinball_gradient() -> None:
    torch.manual_seed(8)
    config = small_config()
    normalized_model = CAREBeliefV2Head(config).eval()
    physical_model = CAREBeliefV2Head(config).eval()
    physical_model.load_state_dict(normalized_model.state_dict())
    memory = torch.randn(2, 7, config.d_model)
    mask = torch.ones(2, 7, dtype=torch.bool)
    chunks = torch.randn(2, 6, config.action_horizon, config.action_dim)
    horizon = torch.tensor([0, 1])
    normalized_target = torch.randn(2, 6, 3)
    normalized_target[:, 0] = 0.0
    scale = torch.tensor([[5e-4, 1e-3, 2e-3], [7e-4, 2e-3, 3e-3]])
    physical_target = normalized_target * scale[:, None, :]
    safety = torch.zeros(2, 6)
    pinball_only = CARELossV2Config(
        consistency_weight=0.0,
        candidate_ranking_weight=0.0,
        reference_ranking_weight=0.0,
        safety_weight=0.0,
    )
    normalized_output = normalized_model(memory, mask, chunks, horizon)
    normalized_loss, _ = care_v2_training_loss(
        normalized_output,
        normalized_target,
        safety,
        "care",
        target_scale=torch.ones_like(scale),
        loss_config=pinball_only,
    )
    physical_output = physical_model(
        memory, mask, chunks, horizon, utility_scale=scale
    )
    physical_loss, _ = care_v2_training_loss(
        physical_output,
        physical_target,
        safety,
        "care",
        target_scale=scale,
        loss_config=pinball_only,
    )
    normalized_loss.backward()
    physical_loss.backward()
    torch.testing.assert_close(physical_loss, normalized_loss)
    torch.testing.assert_close(
        physical_model.advantage.weight.grad,
        normalized_model.advantage.weight.grad,
        rtol=1e-5,
        atol=1e-6,
    )


def test_zero_effective_safety_weight_removes_one_class_bce_gradient() -> None:
    torch.manual_seed(9)
    quantiles = torch.randn(2, 6, 3, 5).sort(-1).values
    quantiles[:, 0] = 0.0
    safety_logit = torch.randn(2, 6, requires_grad=True)
    output = CAREBeliefOutput(
        quantiles=quantiles.requires_grad_(),
        hard_safety_logit=safety_logit,
        candidate_state=torch.empty(2, 6, 0),
    )
    target = torch.randn(2, 6, 3)
    target[:, 0] = 0.0
    loss, pieces = care_v2_training_loss(
        output,
        target,
        torch.zeros(2, 6),
        "care",
        target_scale=torch.ones(2, 3),
        loss_config=CARELossV2Config(safety_weight=0.0),
    )
    loss.backward()
    assert pieces["hard_safety"] == 0.0
    assert safety_logit.grad is not None
    assert torch.count_nonzero(safety_logit.grad) == 0


def test_reference_ranking_includes_structural_zero_candidate() -> None:
    scores = torch.tensor([[0.0, 0.2, -0.1]], requires_grad=True)
    targets = torch.tensor([[0.0, 1.0, -1.0]])
    scale = torch.ones(1)
    loss = reference_ranking_loss(scores, targets, scale, minimum_gap=1e-3)
    loss.backward()
    # The reference-vs-positive and reference-vs-negative comparisons both
    # contribute gradients to the non-reference candidates.
    assert torch.isfinite(loss)
    assert scores.grad is not None
    assert scores.grad[0, 1] < 0
    assert scores.grad[0, 2] > 0


def test_v2_loss_scales_components_but_preserves_output_units() -> None:
    torch.manual_seed(7)
    batch, candidates, components, quantiles = 4, 6, 3, 5
    output = CAREBeliefOutput(
        quantiles=torch.randn(batch, candidates, components, quantiles),
        hard_safety_logit=torch.randn(batch, candidates),
        candidate_state=torch.randn(batch, candidates, 8),
    )
    output.quantiles = output.quantiles.sort(-1).values
    output.quantiles[:, 0] = 0.0
    target = torch.randn(batch, candidates, components) * 1e-3
    target[:, 0] = 0.0
    safety = torch.zeros(batch, candidates)
    loss, pieces = care_v2_training_loss(
        output,
        target,
        safety,
        "care",
        target_scale=torch.tensor([1e-3, 2e-3, 3e-3]),
    )
    assert torch.isfinite(loss)
    assert set(pieces) == {
        "pinball_scaled",
        "did_consistency_scaled",
        "candidate_ranking_scaled",
        "reference_ranking_scaled",
        "hard_safety",
    }
    # The selector-facing output is not divided in-place by the training unit.
    assert output.quantiles.shape == (batch, candidates, components, quantiles)


def test_target_scale_and_robust_scale_contracts_fail_closed() -> None:
    target = torch.zeros(2, 6, 3)
    with pytest.raises(ValueError):
        canonical_target_scale(target, [1.0, 0.0, 1.0])
    prepared = torch.zeros(3, 2, 6, 2, 3)
    usable = torch.ones(3, 2, dtype=torch.bool)
    task_id = torch.tensor([0, 0, 1])
    scales = robust_task_component_scales(prepared, usable, task_id)
    assert scales.shape == (2, 3)
    assert torch.all(scales > 0)


def test_loss_config_rejects_negative_weights() -> None:
    with pytest.raises(ValueError):
        CARELossV2Config(reference_ranking_weight=-1.0)
    with pytest.raises(ValueError):
        CARELossV2Config(ranking_min_gap=float("nan"))
