from __future__ import annotations

import torch

from before_we_act.action_generator.r4_base import load_r12_r4_config
from before_we_act.action_generator.act_chunk import build_core


def test_act_core_train_and_inference_contract():
    config = load_r12_r4_config("configs/before_we_act/r12_action/p2.yaml")
    core = build_core(
        {
            **config.component,
            "horizon": 100,
            "joint_action_dim": 32,
            "belief_dim": 96,
            "condition_tokens": 37,
        }
    )
    tokens = torch.randn(2, 37, 96)
    token_mask = torch.ones(2, 37, dtype=torch.bool)
    actions = torch.randn(2, 100, 32)
    mask = torch.ones_like(actions, dtype=torch.bool)
    losses = core.training_loss(tokens, token_mask, actions, mask)
    assert losses.keys() == {
        "loss",
        "l1",
        "plan_kl",
        "plan_prior_kl",
        "plan_recognition_kl",
        "plan_proposal_std",
        "plan_posterior_std",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss"].backward()
    proposal_gradient = core.plan_proposal[-1].weight.grad
    assert proposal_gradient is not None
    assert float(proposal_gradient.abs().sum()) > 0
    core.eval()
    with torch.no_grad():
        first = core.sample(tokens, token_mask)
        second = core.sample(tokens, token_mask, noise=torch.randn_like(actions))
    assert first.shape == (2, 100, 32)
    torch.testing.assert_close(first, second)


def test_act_plan_proposal_is_current_condition_dependent_and_mask_aware():
    config = load_r12_r4_config("configs/before_we_act/r12_action/p2.yaml")
    core = build_core(
        {
            **config.component,
            "horizon": 100,
            "joint_action_dim": 32,
            "belief_dim": 96,
            "condition_tokens": 37,
        }
    ).eval()
    tokens = torch.randn(1, 37, 96)
    mask = torch.ones(1, 37, dtype=torch.bool)
    with torch.no_grad():
        core.plan_proposal[-1].weight.normal_(std=0.01)
    with torch.no_grad():
        base = core.sample(tokens, mask)
        changed = core.sample(tokens + 0.5, mask)
        hidden = tokens.clone()
        hidden[:, -1] = 10_000
        hidden_mask = mask.clone()
        hidden_mask[:, -1] = False
        reference_mask = mask.clone()
        reference_mask[:, -1] = False
        masked = core.sample(hidden, hidden_mask)
        reference = core.sample(tokens, reference_mask)
    assert not torch.equal(base, changed)
    torch.testing.assert_close(masked, reference)
