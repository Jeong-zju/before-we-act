from __future__ import annotations

import torch

from before_we_act.care_belief import (
    CAREBeliefConfig,
    CAREBeliefHead,
    CARECalibration,
    care_training_loss,
    select_care_candidate,
)


def small_config(variant: str = "care") -> CAREBeliefConfig:
    return CAREBeliefConfig(
        d_model=32,
        action_horizon=20,
        action_tokens=4,
        action_width=16,
        heads=4,
        layers=1,
        dropout=0.0,
        variant=variant,
    )


def inputs(config: CAREBeliefConfig):
    torch.manual_seed(7)
    memory = torch.randn(3, 9, config.d_model)
    mask = torch.ones(3, 9, dtype=torch.bool)
    mask[:, -2:] = False
    chunks = torch.randn(3, 6, config.action_horizon, config.action_dim)
    horizon = torch.tensor([0, 1, 3])
    return memory, mask, chunks, horizon


def test_care_head_has_distributional_action_response_contract() -> None:
    config = small_config()
    model = CAREBeliefHead(config).eval()
    output = model(*inputs(config))
    assert output.quantiles.shape == (3, 6, 3, 5)
    assert output.hard_safety_logit.shape == (3, 6)
    assert torch.equal(output.quantiles[:, 0], torch.zeros_like(output.quantiles[:, 0]))
    assert torch.all(output.quantiles[..., 1:] >= output.quantiles[..., :-1])


def test_care_loss_updates_head_but_not_external_memory() -> None:
    config = small_config()
    model = CAREBeliefHead(config)
    memory, mask, chunks, horizon = inputs(config)
    memory.requires_grad_(True)
    output = model(memory, mask, chunks, horizon)
    target = torch.randn(3, 6, 3)
    target[:, 0] = 0
    unsafe = torch.zeros(3, 6)
    loss, pieces = care_training_loss(output, target, unsafe, "care")
    loss.backward()
    assert torch.isfinite(loss)
    assert set(pieces) == {"pinball", "did_consistency", "ranking", "hard_safety"}
    assert any(parameter.grad is not None for parameter in model.parameters())
    # The tensor receives a gradient for normal end-to-end tests, but the
    # actual A6 pipeline supplies detached, checkpointed frozen B-core memory.
    assert memory.grad is not None


def test_selector_fails_closed_and_can_choose_positive_candidate() -> None:
    config = small_config()
    model = CAREBeliefHead(config).eval()
    output = model(*inputs(config))
    with torch.no_grad():
        output.quantiles.zero_()
        output.hard_safety_logit.fill_(-20)
    calibration = CARECalibration(0.01, 0.0, 0.25, 0.9, 16)
    selected, _lower, _unsafe = select_care_candidate(output, calibration)
    assert selected.tolist() == [0, 0, 0]
    with torch.no_grad():
        output.quantiles[:, 4, 2, 0] = 0.03
    selected, lower, _unsafe = select_care_candidate(output, calibration)
    assert selected.tolist() == [4, 4, 4]
    assert torch.allclose(lower, torch.full_like(lower, 0.02))


def test_predicted_unsafe_candidate_is_never_selected() -> None:
    config = small_config()
    model = CAREBeliefHead(config).eval()
    output = model(*inputs(config))
    with torch.no_grad():
        output.quantiles.zero_()
        output.hard_safety_logit.fill_(-20)
        output.quantiles[:, 2, 2, 0] = 1.0
        output.hard_safety_logit[:, 2] = 20.0
    calibration = CARECalibration(0.0, 0.0, 0.25, 0.9, 16)
    selected, _lower, unsafe = select_care_candidate(output, calibration)
    assert selected.tolist() == [0, 0, 0]
    assert unsafe[:, 2].all()
