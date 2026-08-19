"""Training-only base-relative belief components for roadmap Step 4.

The deployed policy is still the Step-3 predictive belief policy.  This module
adds only a restricted prior ``p(B | C)`` that can read the frozen base action
context ``C``.  The prior is used to estimate and penalize information in the
runtime belief which the base already carries; it is never exported.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from before_we_act.predictive_team_belief_training import (
    TeamBeliefExperiment,
    TeamBeliefExperimentOutput,
)
from before_we_act.team_belief.predictive_core import TeamBeliefConfig


@dataclass
class BaseConditionedPriorOutput:
    log_probs: torch.Tensor
    probs: torch.Tensor
    entropy: torch.Tensor


@dataclass
class BaseRelativeBeliefOutput:
    candidate: object
    direct_prediction: torch.Tensor
    direct_residual: torch.Tensor
    direct_gate: torch.Tensor
    base_prior: BaseConditionedPriorOutput


class BaseConditionedBeliefPrior(nn.Module):
    """Predict the categorical runtime belief from frozen base context only.

    ``decoded_action_hidden`` is the cached output of the frozen B0-H action
    decoder.  No legal-history tensor, teacher tensor, future target, episode
    identity or simulator value is accepted by this interface.
    """

    def __init__(self, config: TeamBeliefConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.slot_queries = nn.Parameter(
            torch.randn(1, config.n_belief_tokens, d) * 0.02
        )
        self.context_norm = nn.LayerNorm(d)
        self.query_norm = nn.LayerNorm(d)
        self.cross_attention = nn.MultiheadAttention(
            d,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
            bias=False,
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.categorical_head = nn.Linear(
            d, config.belief_factors * config.belief_classes
        )

    def forward(
        self, decoded_action_hidden: torch.Tensor
    ) -> BaseConditionedPriorOutput:
        if (
            decoded_action_hidden.ndim != 3
            or decoded_action_hidden.shape[-1] != self.config.d_model
        ):
            raise ValueError(
                "base-conditioned prior expects [batch,horizon,d_model] C"
            )
        batch = decoded_action_hidden.shape[0]
        context = self.context_norm(decoded_action_hidden)
        query = self.query_norm(self.slot_queries).expand(batch, -1, -1)
        attended = self.cross_attention(
            query, context, context, need_weights=False
        )[0]
        hidden = self.fusion(torch.cat((query, attended), dim=-1))
        logits = self.categorical_head(hidden).view(
            batch,
            self.config.n_belief_tokens,
            self.config.belief_factors,
            self.config.belief_classes,
        )
        base_probs = logits.float().softmax(-1)
        uniform = 1.0 / self.config.belief_classes
        probs = (
            (1.0 - self.config.belief_unimix) * base_probs
            + self.config.belief_unimix * uniform
        )
        log_probs = probs.log()
        entropy = -(probs * log_probs).sum(-1)
        return BaseConditionedPriorOutput(
            log_probs=log_probs,
            probs=probs,
            entropy=entropy,
        )


class BaseRelativeBeliefExperiment(TeamBeliefExperiment):
    """Step-4 candidate with a removable training-only ``p(B | C)``."""

    def __init__(self, config: TeamBeliefConfig) -> None:
        super().__init__(config)
        self.base_conditioned_prior = BaseConditionedBeliefPrior(config)

    def forward(self, batch: dict[str, torch.Tensor]) -> BaseRelativeBeliefOutput:
        base: TeamBeliefExperimentOutput = super().forward(batch)
        prior = self.base_conditioned_prior(batch["decoded_action_hidden"])
        return BaseRelativeBeliefOutput(
            candidate=base.candidate,
            direct_prediction=base.direct_prediction,
            direct_residual=base.direct_residual,
            direct_gate=base.direct_gate,
            base_prior=prior,
        )


__all__ = [
    "BaseConditionedBeliefPrior",
    "BaseConditionedPriorOutput",
    "BaseRelativeBeliefExperiment",
    "BaseRelativeBeliefOutput",
]
