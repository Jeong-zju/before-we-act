from __future__ import annotations

import torch

from before_we_act.base_relative_belief import BaseConditionedBeliefPrior
from before_we_act.base_relative_belief_losses import (
    _conditional_kl,
    bradley_terry_preference_loss,
)
from before_we_act.team_belief.predictive_core import TeamBeliefConfig


def small_config() -> TeamBeliefConfig:
    return TeamBeliefConfig(
        n_belief_tokens=4,
        n_evidence_queries=2,
        event_capacity=2,
        temporal_layers=1,
        d_model=32,
        vision_dim=24,
        state_dim=9,
        action_dim=8,
        heads=8,
        dropout=0.0,
        belief_factors=3,
        belief_classes=5,
    )


def test_base_prior_has_factorized_categorical_contract() -> None:
    config = small_config()
    prior = BaseConditionedBeliefPrior(config)
    output = prior(torch.randn(3, 7, config.d_model))
    assert output.probs.shape == (3, 4, 3, 5)
    assert output.log_probs.shape == output.probs.shape
    assert output.entropy.shape == (3, 4, 3)
    torch.testing.assert_close(output.probs.sum(-1), torch.ones(3, 4, 3))
    assert float(output.probs.min()) >= config.belief_unimix / config.belief_classes


def test_conditional_kl_split_gives_each_direction_the_intended_gradient() -> None:
    runtime_logits = torch.randn(2, 4, 3, 5, requires_grad=True)
    prior_logits = torch.randn(2, 4, 3, 5, requires_grad=True)
    runtime_log = runtime_logits.log_softmax(-1)
    runtime = runtime_log.exp()
    prior_log = prior_logits.log_softmax(-1)
    prior = prior_log.exp()
    prior_fit, bottleneck, diagnostic = _conditional_kl(
        runtime_log, runtime, prior_log, prior
    )

    prior_fit.backward(retain_graph=True)
    assert runtime_logits.grad is None
    assert prior_logits.grad is not None
    assert float(prior_logits.grad.abs().sum()) > 0.0

    runtime_logits.grad = None
    prior_logits.grad = None
    bottleneck.backward()
    assert runtime_logits.grad is not None
    assert float(runtime_logits.grad.abs().sum()) > 0.0
    assert prior_logits.grad is None
    assert not diagnostic.requires_grad


def test_bradley_terry_rewards_lower_matched_error() -> None:
    active = torch.tensor([True, True, False])
    good, good_accuracy, good_margin = bradley_terry_preference_loss(
        torch.tensor([0.1, 0.2, 8.0]),
        torch.tensor([0.4, 0.3, 0.0]),
        active,
        temperature=0.1,
        margin=torch.tensor([0.2, 0.05, 0.0]),
    )
    bad, bad_accuracy, bad_margin = bradley_terry_preference_loss(
        torch.tensor([0.4, 0.3, 8.0]),
        torch.tensor([0.1, 0.2, 0.0]),
        active,
        temperature=0.1,
        margin=torch.tensor([0.2, 0.05, 0.0]),
    )
    assert good < bad
    assert good_accuracy == 1.0
    assert bad_accuracy == 0.0
    assert good_margin == 1.0
    assert bad_margin == 0.0


def test_bradley_terry_has_exactly_zero_gradient_after_finite_margin() -> None:
    positive = torch.tensor([0.2], requires_grad=True)
    negative = torch.tensor([0.5], requires_grad=True)
    loss, accuracy, satisfaction = bradley_terry_preference_loss(
        positive,
        negative,
        torch.tensor([True]),
        temperature=0.1,
        margin=torch.tensor([0.1]),
    )
    loss.backward()

    assert loss == 0.0
    assert accuracy == 1.0
    assert satisfaction == 1.0
    assert positive.grad is not None and positive.grad.item() == 0.0
    assert negative.grad is not None and negative.grad.item() == 0.0


def test_bradley_terry_pushes_both_errors_only_until_margin() -> None:
    positive = torch.tensor([0.2], requires_grad=True)
    negative = torch.tensor([0.21], requires_grad=True)
    loss, _accuracy, satisfaction = bradley_terry_preference_loss(
        positive,
        negative,
        torch.tensor([True]),
        temperature=0.1,
        margin=torch.tensor([0.1]),
    )
    loss.backward()

    assert loss > 0.0
    assert satisfaction == 0.0
    assert positive.grad is not None and positive.grad.item() > 0.0
    assert negative.grad is not None and negative.grad.item() < 0.0
