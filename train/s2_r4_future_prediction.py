"""Masked peer/shared losses used by the S2-R4 team capability gate."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def masked_peer_future_prediction_losses(
    predicted_state: Tensor,
    target_state: Tensor,
    state_valid: Tensor,
    predicted_visual: Tensor,
    target_visual: Tensor,
    visual_valid: Tensor,
    valid_agents: Tensor,
) -> dict[str, Tensor]:
    """Compare every focal agent's peer predictions against target-agent futures."""

    if predicted_state.ndim != 5:
        raise ValueError("peer state predictions must be [B,focal,target,F,S]")
    batch_size, agents, targets, futures, _ = predicted_state.shape
    if agents != targets or target_state.shape != (
        batch_size,
        targets,
        futures,
        predicted_state.shape[-1],
    ):
        raise ValueError("peer state targets have an invalid shape")
    if predicted_visual.ndim != 6 or target_visual.shape != (
        batch_size,
        targets,
        futures,
        predicted_visual.shape[-2],
        predicted_visual.shape[-1],
    ):
        raise ValueError("peer visual targets have an invalid shape")
    if state_valid.shape != target_state.shape[:-1]:
        raise ValueError("peer state validity has an invalid shape")
    if visual_valid.shape != target_visual.shape[:3]:
        raise ValueError("peer visual validity has an invalid shape")
    if valid_agents.shape != (batch_size, agents):
        raise ValueError("valid_agents must be [B,A]")

    off_diagonal = ~torch.eye(
        agents,
        dtype=valid_agents.dtype,
        device=valid_agents.device,
    )
    pair_valid = (
        valid_agents[:, :, None]
        & valid_agents[:, None, :]
        & off_diagonal[None]
    )
    state_mask = (
        pair_valid[:, :, :, None]
        & state_valid[:, None]
    )
    visual_mask = (
        pair_valid[:, :, :, None]
        & visual_valid[:, None]
    )
    expanded_state = target_state[:, None].expand_as(predicted_state)
    expanded_visual = target_visual[:, None].expand_as(predicted_visual)
    state_values = F.smooth_l1_loss(
        predicted_state.float(),
        expanded_state.float(),
        reduction="none",
    ).mean(dim=-1)
    visual_values = 1.0 - F.cosine_similarity(
        predicted_visual.float(),
        expanded_visual.float(),
        dim=-1,
        eps=1e-6,
    )
    visual_values = visual_values.mean(dim=-1)
    state_per = _masked_per_sample(state_values, state_mask, allow_empty=False)
    visual_per = _masked_per_sample(
        visual_values,
        visual_mask,
        allow_empty=True,
    )
    return {
        "loss": (state_per + visual_per).mean(),
        "state": state_per.mean(),
        "visual": visual_per.mean(),
        "per_trajectory": state_per + visual_per,
    }


def masked_shared_future_prediction_losses(
    predicted_visual: Tensor,
    target_visual: Tensor,
    future_valid: Tensor,
    valid_agents: Tensor,
) -> dict[str, Tensor]:
    """Compare each focal agent's global-slot prediction to shared futures."""

    if predicted_visual.ndim != 5:
        raise ValueError("shared predictions must be [B,focal,F,G,D]")
    batch_size, agents, futures, grid, width = predicted_visual.shape
    if target_visual.shape != (batch_size, futures, grid, width):
        raise ValueError("shared visual target has an invalid shape")
    if future_valid.shape != (batch_size, futures):
        raise ValueError("shared future validity must be [B,F]")
    if valid_agents.shape != (batch_size, agents):
        raise ValueError("valid_agents must be [B,A]")
    expanded = target_visual[:, None].expand_as(predicted_visual)
    values = 1.0 - F.cosine_similarity(
        predicted_visual.float(),
        expanded.float(),
        dim=-1,
        eps=1e-6,
    )
    values = values.mean(dim=-1)
    valid = valid_agents[:, :, None] & future_valid[:, None]
    per = _masked_per_sample(values, valid, allow_empty=False)
    return {
        "loss": per.mean(),
        "visual": per.mean(),
        "per_trajectory": per,
    }


def peer_actions_shuffled_by_focal(
    candidate_actions: Tensor,
    valid_agents: Tensor,
) -> Tensor:
    """Roll only peer actions across the batch; preserve each focal own action."""

    if candidate_actions.ndim != 4:
        raise ValueError("candidate_actions must be [B,A,H,D]")
    batch_size, agents = candidate_actions.shape[:2]
    if batch_size < 2:
        raise ValueError("peer-action shuffle requires at least two samples")
    if valid_agents.shape != (batch_size, agents):
        raise ValueError("valid_agents must be [B,A]")
    rolled = candidate_actions.roll(1, dims=0)
    by_focal = rolled[:, None].expand(
        -1,
        agents,
        -1,
        -1,
        -1,
    ).clone()
    diagonal = torch.arange(
        agents,
        device=candidate_actions.device,
    )
    by_focal[:, diagonal, diagonal] = candidate_actions[:, diagonal]
    invalid_target = ~valid_agents[:, None, :, None, None]
    return by_focal.masked_fill(invalid_target, 0.0)


def _masked_per_sample(
    values: Tensor,
    valid: Tensor,
    *,
    allow_empty: bool,
) -> Tensor:
    if values.shape != valid.shape or values.ndim < 2:
        raise ValueError("masked values and validity must share shape")
    weights = valid.to(values)
    axes = tuple(range(1, values.ndim))
    denominator = weights.sum(dim=axes)
    if not allow_empty and not bool(denominator.gt(0).all()):
        raise ValueError("every sample needs at least one valid target")
    return (values * weights).sum(dim=axes) / denominator.clamp_min(1.0)


__all__ = [
    "masked_peer_future_prediction_losses",
    "masked_shared_future_prediction_losses",
    "peer_actions_shuffled_by_focal",
]
