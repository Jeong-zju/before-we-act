"""Masked, separately reported training objectives for the 3-N2 model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from before_we_act.team_belief.n2_core import FUTURE_OFFSETS_SECONDS

if TYPE_CHECKING:
    from before_we_act.b3_n2_model import B3N2PolicyOutput


@dataclass(frozen=True)
class B3N2LossWeights:
    """No defaults: the activation contract must freeze every coefficient."""

    action: float
    action_posterior_kl: float
    teacher_alignment: float
    future_latent: float
    teacher_reconstruction: float
    teammate_delta: float
    teammate_action: float
    exchange_consistency: float
    anti_collapse: float


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    expanded = weight.expand_as(value)
    return (value * expanded).sum() / expanded.sum().clamp_min(1)


def _gaussian_kl(
    student_mu: torch.Tensor,
    student_sigma: torch.Tensor,
    teacher_mu: torch.Tensor,
    teacher_sigma: torch.Tensor,
) -> torch.Tensor:
    teacher_mu = teacher_mu.detach()
    teacher_sigma = teacher_sigma.detach().clamp_min(1e-4)
    student_sigma = student_sigma.clamp_min(1e-4)
    return (
        torch.log(teacher_sigma / student_sigma)
        + (
            student_sigma.square() + (student_mu - teacher_mu).square()
        )
        / (2 * teacher_sigma.square())
        - 0.5
    ).mean()


def _future_anchor_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    anchor_mask: torch.Tensor,
    view_mask: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    per_anchor: list[torch.Tensor] = []
    for index in range(len(FUTURE_OFFSETS_SECONDS)):
        active = view_mask[:, index] & anchor_mask[:, index : index + 1]
        if active.any():
            active_target = target[:, index][active]
            scale = active_target.float().var(unbiased=False).clamp_min(1e-4)
            value = (
                prediction[:, index][active] - active_target
            ).float().square().mean() / scale
            per_anchor.append(value.to(reference.dtype))
        else:
            per_anchor.append(reference.new_zeros(()))
    valid_anchor = (
        view_mask & anchor_mask.unsqueeze(-1)
    ).any((0, 2))
    if valid_anchor.any():
        aggregate = torch.stack(per_anchor)[valid_anchor].mean()
    else:
        aggregate = reference.new_zeros(())
    return aggregate, per_anchor


def compute_b3_n2_losses(
    output: B3N2PolicyOutput,
    action_target: torch.Tensor,
    action_mask: torch.Tensor,
    teammate_delta_target: torch.Tensor,
    teammate_delta_mask: torch.Tensor,
    teammate_action_target: torch.Tensor,
    teammate_action_mask: torch.Tensor,
    weights: B3N2LossWeights,
    *,
    swapped_output: B3N2PolicyOutput | None = None,
) -> dict[str, torch.Tensor]:
    """Compute N2 losses without merging the four future-anchor reports."""

    if output.teacher is None:
        raise ValueError("3-N2 training losses require the privileged teacher output")
    if action_target.shape != output.prediction.shape:
        raise ValueError("action target/prediction shape differs")
    if action_mask.shape != output.prediction.shape[:2]:
        raise ValueError("action mask shape differs")
    action = _masked_mean(
        (output.prediction - action_target).square().mean(-1), action_mask
    )
    if output.action_posterior_mu is None or output.action_posterior_logvar is None:
        posterior_kl = action.new_zeros(())
    else:
        posterior_kl = -0.5 * (
            1
            + output.action_posterior_logvar
            - output.action_posterior_mu.square()
            - output.action_posterior_logvar.exp()
        ).mean()

    teacher_alignment = _gaussian_kl(
        output.belief.mu,
        output.belief.sigma,
        output.teacher.mu,
        output.teacher.sigma,
    )
    prediction = output.belief.future_latent_prediction
    target = output.teacher.future_latent_target
    anchor_mask = output.teacher.future_anchor_mask
    view_mask = output.teacher.future_view_mask
    if (
        prediction.shape != target.shape
        or output.teacher.future_latent_reconstruction.shape != target.shape
        or anchor_mask.shape != prediction.shape[:2]
        or view_mask.shape != prediction.shape[:3]
    ):
        raise ValueError("future prediction/teacher anchor contract differs")
    future_latent, per_anchor = _future_anchor_losses(
        prediction, target, anchor_mask, view_mask, action
    )
    teacher_reconstruction, teacher_per_anchor = _future_anchor_losses(
        output.teacher.future_latent_reconstruction,
        target,
        anchor_mask,
        view_mask,
        action,
    )

    if teammate_delta_target.shape != output.belief.teammate_state_delta_prediction.shape:
        raise ValueError("teammate delta target shape differs")
    if teammate_delta_mask.shape != teammate_delta_target.shape[:2]:
        raise ValueError("teammate delta mask shape differs")
    teammate_delta = _masked_mean(
        (
            output.belief.teammate_state_delta_prediction
            - teammate_delta_target
        ).square(),
        teammate_delta_mask,
    )
    if teammate_action_target.shape != output.belief.teammate_action_mean.shape:
        raise ValueError("teammate action target shape differs")
    if teammate_action_mask.shape != teammate_action_target.shape[:2]:
        raise ValueError("teammate action mask shape differs")
    teammate_action_logvar = output.belief.teammate_action_logvar
    teammate_action = _masked_mean(
        0.5
        * (
            teammate_action_logvar
            + (teammate_action_target - output.belief.teammate_action_mean).square()
            * torch.exp(-teammate_action_logvar)
        ),
        teammate_action_mask,
    )

    exchange = action.new_zeros(())
    if swapped_output is not None:
        if swapped_output.belief.mu.shape != output.belief.mu.shape:
            raise ValueError("swapped belief shape differs")
        token_permutation = torch.arange(
            output.belief.mu.shape[1], device=output.belief.mu.device
        )
        token_permutation[:2] = torch.tensor(
            (1, 0), device=token_permutation.device
        )
        exchanged = swapped_output.belief.mu.index_select(1, token_permutation)
        exchange = F.mse_loss(output.belief.mu, exchanged)

    # A variance floor is only an anti-collapse guard, not an ARB semantic
    # target.  It is computed across examples and slots, never episode IDs.
    feature_std = output.belief.mu.float().flatten(0, 1).std(0, unbiased=False)
    anti_collapse = F.relu(0.1 - feature_std).mean().to(action.dtype)
    total = (
        weights.action * action
        + weights.action_posterior_kl * posterior_kl
        + weights.teacher_alignment * teacher_alignment
        + weights.future_latent * future_latent
        + weights.teacher_reconstruction * teacher_reconstruction
        + weights.teammate_delta * teammate_delta
        + weights.teammate_action * teammate_action
        + weights.exchange_consistency * exchange
        + weights.anti_collapse * anti_collapse
    )
    result = {
        "total": total,
        "action": action,
        "action_posterior_kl": posterior_kl,
        "teacher_alignment": teacher_alignment,
        "future_latent": future_latent,
        "teacher_reconstruction": teacher_reconstruction,
        "teammate_delta": teammate_delta,
        "teammate_action": teammate_action,
        "exchange_consistency": exchange,
        "anti_collapse": anti_collapse,
    }
    for seconds, value in zip(FUTURE_OFFSETS_SECONDS, per_anchor, strict=True):
        result[f"future_{seconds:.1f}s"] = value
    for seconds, value in zip(
        FUTURE_OFFSETS_SECONDS, teacher_per_anchor, strict=True
    ):
        result[f"teacher_future_{seconds:.1f}s"] = value
    return result


__all__ = ["B3N2LossWeights", "compute_b3_n2_losses"]
