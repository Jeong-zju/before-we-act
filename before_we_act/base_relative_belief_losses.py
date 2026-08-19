"""Variational base-relative control losses for roadmap Step 4."""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from before_we_act.base_relative_belief import BaseRelativeBeliefOutput
from before_we_act.team_belief.losses import (
    TeamBeliefLossWeights,
    compute_team_belief_losses,
)


@dataclass(frozen=True)
class BaseRelativeLossWeights:
    """Frozen Step-4 coefficients.

    The split conditional KL trains ``p(B|C)`` in both arms.  Only
    ``conditional_bottleneck`` sends the reverse pressure into runtime belief,
    which makes the beta=0 control meaningful rather than comparing against an
    intentionally untrained prior.
    """

    base: TeamBeliefLossWeights
    conditional_prior_fit: float
    conditional_bottleneck: float
    bradley_terry: float
    bradley_terry_temperature: float
    bradley_terry_margin_fraction: float
    bradley_terry_margin_cap: float

    def __post_init__(self) -> None:
        for name in (
            "conditional_prior_fit",
            "conditional_bottleneck",
            "bradley_terry",
            "bradley_terry_margin_fraction",
            "bradley_terry_margin_cap",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.bradley_terry_temperature <= 0.0:
            raise ValueError("Bradley-Terry temperature must be positive")


def _row_action_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != target.shape or mask.shape != prediction.shape[:2]:
        raise ValueError("row action-MSE contract differs")
    squared = (prediction - target).float().square().mean(-1)
    return (squared * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def _conditional_kl(
    runtime_log_probs: torch.Tensor,
    runtime_probs: torch.Tensor,
    prior_log_probs: torch.Tensor,
    prior_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shapes = {
        tuple(runtime_log_probs.shape),
        tuple(runtime_probs.shape),
        tuple(prior_log_probs.shape),
        tuple(prior_probs.shape),
    }
    if len(shapes) != 1 or runtime_probs.ndim != 4:
        raise ValueError(
            "runtime/prior beliefs must share [batch,token,factor,class]"
        )
    # Train the restricted prior to fit q(B|H), whether beta_b is on or off.
    prior_fit = (
        runtime_probs.detach()
        * (runtime_log_probs.detach() - prior_log_probs)
    ).sum(-1).mean()
    # Compress q only toward the current restricted prior.  Detaching p keeps
    # this term from merely teaching p twice and implements the VIB direction.
    bottleneck = (
        runtime_probs
        * (runtime_log_probs - prior_log_probs.detach())
    ).sum(-1).mean()
    diagnostic = (
        runtime_probs.detach()
        * (runtime_log_probs.detach() - prior_log_probs.detach())
    ).sum(-1).mean()
    return prior_fit, bottleneck, diagnostic


def bradley_terry_preference_loss(
    positive_error: torch.Tensor,
    negative_error: torch.Tensor,
    active: torch.Tensor,
    *,
    temperature: float,
    margin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a finite-margin Bradley--Terry loss and two diagnostics.

    A plain Bradley--Terry NLL has no finite optimum when the negative score is
    a trainable action error: it can always reduce the loss by making the
    shuffled action arbitrarily bad.  We keep its smooth likelihood geometry
    while the requested margin is unmet, then subtract the loss at the margin
    and clamp at zero.  Consequently the counterfactual branch receives
    exactly zero gradient once ``negative_error-positive_error >= margin``.
    """

    if (
        positive_error.shape != negative_error.shape
        or active.shape != positive_error.shape
        or margin.shape != positive_error.shape
        or active.dtype != torch.bool
    ):
        raise ValueError("Bradley-Terry row contract differs")
    if temperature <= 0.0:
        raise ValueError("Bradley-Terry temperature must be positive")
    if torch.any(margin < 0):
        raise ValueError("Bradley-Terry margin must be non-negative")
    if not active.any():
        zero = positive_error.new_zeros(())
        return zero, zero, zero
    gap = negative_error - positive_error
    logits = gap / temperature
    margin_logits = margin.detach() / temperature
    row_loss = (
        F.softplus(-logits) - F.softplus(-margin_logits)
    ).clamp_min(0.0)
    loss = row_loss[active].mean()
    accuracy = (gap[active] > 0).float().mean().to(loss.dtype)
    margin_satisfaction = (gap[active] >= margin[active]).float().mean()
    return loss, accuracy, margin_satisfaction.to(loss.dtype)


def compute_base_relative_losses(
    output: BaseRelativeBeliefOutput,
    action_target: torch.Tensor,
    action_mask: torch.Tensor,
    teammate_delta_target: torch.Tensor,
    teammate_delta_mask: torch.Tensor,
    teammate_action_target: torch.Tensor,
    teammate_action_mask: torch.Tensor,
    weights: BaseRelativeLossWeights,
    *,
    swapped_output=None,
    counterfactual_prediction: torch.Tensor | None = None,
    counterfactual_residual_target: torch.Tensor | None = None,
    counterfactual_action_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute VBC with Bradley-Terry action pairing.

    The old hinge coefficient is forcibly zeroed so the action comparison is
    represented exactly once.
    """

    base_weights = replace(weights.base, action_pairing=0.0)
    result = compute_team_belief_losses(
        output.candidate,
        action_target,
        action_mask,
        teammate_delta_target,
        teammate_delta_mask,
        teammate_action_target,
        teammate_action_mask,
        base_weights,
        swapped_output=swapped_output,
    )
    prior_fit, bottleneck, conditional_kl = _conditional_kl(
        output.candidate.belief.categorical_log_probs,
        output.candidate.belief.categorical_probs,
        output.base_prior.log_probs,
        output.base_prior.probs,
    )

    reference = result["total"]
    bradley_terry = reference.new_zeros(())
    active_fraction = reference.new_zeros(())
    preference_accuracy = reference.new_zeros(())
    margin_satisfaction = reference.new_zeros(())
    pairing_margin = reference.new_zeros(())
    pairing_gap = reference.new_zeros(())
    matched_error = reference.new_zeros(())
    counterfactual_error = reference.new_zeros(())
    if counterfactual_prediction is not None:
        if (
            counterfactual_residual_target is None
            or counterfactual_action_mask is None
        ):
            raise ValueError("Bradley-Terry pairing requires target and mask")
        positive = _row_action_mse(
            output.candidate.prediction, action_target, action_mask
        )
        negative = _row_action_mse(
            counterfactual_prediction, action_target, action_mask
        )
        residual_target = action_target - output.candidate.base_prediction
        common_mask = action_mask & counterfactual_action_mask
        target_separation = _row_action_mse(
            residual_target, counterfactual_residual_target, common_mask
        )
        active = common_mask.any(-1) & (target_separation > 1e-6)
        if active.any():
            margin_rows = (
                weights.bradley_terry_margin_fraction
                * target_separation.detach()
            ).clamp(max=weights.bradley_terry_margin_cap)
            # P(matched preferred) = sigmoid((E_negative-E_positive)/tau).
            bradley_terry, preference_accuracy, margin_satisfaction = (
                bradley_terry_preference_loss(
                    positive,
                    negative,
                    active,
                    temperature=weights.bradley_terry_temperature,
                    margin=margin_rows,
                )
            )
            bradley_terry = bradley_terry.to(reference.dtype)
            preference_accuracy = preference_accuracy.to(reference.dtype)
            margin_satisfaction = margin_satisfaction.to(reference.dtype)
            matched_error = positive[active].mean().to(reference.dtype)
            counterfactual_error = negative[active].mean().to(reference.dtype)
            pairing_margin = margin_rows[active].mean().to(reference.dtype)
            pairing_gap = (negative - positive)[active].mean().to(
                reference.dtype
            )
        active_fraction = active.float().mean().to(reference.dtype)

    total = (
        result["total"]
        + weights.conditional_prior_fit * prior_fit
        + weights.conditional_bottleneck * bottleneck
        + weights.bradley_terry * bradley_terry
    )
    result.update(
        {
            "total": total,
            "conditional_prior_fit": prior_fit,
            "conditional_bottleneck": bottleneck,
            "conditional_kl": conditional_kl,
            "bradley_terry": bradley_terry,
            "bradley_terry_active_fraction": active_fraction,
            "bradley_terry_preference_accuracy": preference_accuracy,
            "bradley_terry_margin_satisfaction": margin_satisfaction,
            "bradley_terry_margin": pairing_margin,
            "bradley_terry_gap": pairing_gap,
            "bradley_terry_matched_error": matched_error,
            "bradley_terry_counterfactual_error": counterfactual_error,
        }
    )
    return result


__all__ = [
    "BaseRelativeLossWeights",
    "bradley_terry_preference_loss",
    "compute_base_relative_losses",
]
