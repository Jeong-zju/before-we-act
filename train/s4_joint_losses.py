"""S4 joint objectives with equal agents inside equal-weight teams.

Every component follows the same reduction order:

1. mean over valid target axes for each focal agent;
2. mean over all valid focal agents in a team;
3. mean over the fixed batch of team windows.

Optional peer/shared targets contribute a differentiable zero when absent.  An
empty team is never removed from the batch denominator, so target availability
cannot silently reweight the task/episode-balanced sampler.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import math
from numbers import Real
from typing import Callable

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class S4JointLoss(Mapping[str, Tensor]):
    """Scalar joint loss and its fixed-contract component breakdown."""

    total: Tensor
    flow: Tensor
    state: Tensor
    visual: Tensor
    own_state: Tensor
    peer_state: Tensor
    own_visual: Tensor
    peer_visual: Tensor
    shared_visual: Tensor

    _KEYS = (
        "loss",
        "total",
        "flow",
        "state",
        "visual",
        "own_state",
        "peer_state",
        "own_visual",
        "peer_visual",
        "shared_visual",
    )

    @property
    def loss(self) -> Tensor:
        return self.total

    def __getitem__(self, key: str) -> Tensor:
        if key == "loss":
            return self.total
        if key not in self._KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._KEYS)

    def __len__(self) -> int:
        return len(self._KEYS)

    def to_dict(self) -> dict[str, Tensor]:
        return {key: self[key] for key in self}


def _validate_valid_agents(valid_agent_mask: Tensor, *, batch: int, agents: int) -> None:
    if valid_agent_mask.dtype != torch.bool:
        raise TypeError("valid_agent_mask must have dtype bool")
    if valid_agent_mask.shape != (batch, agents):
        raise ValueError("valid_agent_mask must be [B,A]")
    if not bool(valid_agent_mask.any(dim=1).all()):
        raise ValueError("every team must contain at least one valid agent")


def _broadcast_mask(mask: Tensor, values: Tensor, *, name: str) -> Tensor:
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype bool")
    if mask.device != values.device:
        raise TypeError(f"{name} and values must share a device")
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    try:
        return torch.broadcast_to(expanded, values.shape)
    except RuntimeError as exc:
        raise ValueError(f"{name} cannot broadcast to {tuple(values.shape)}") from exc


def hierarchical_agent_team_mean(
    values: Tensor,
    valid: Tensor,
    valid_agent_mask: Tensor,
    *,
    allow_empty_agents: bool,
) -> Tensor:
    """Reduce ``[B,A,...]`` values without agent- or team-count bias.

    When ``allow_empty_agents`` is true, a valid focal agent with no optional
    target contributes zero but remains in its team's agent denominator.
    """

    if values.ndim < 3:
        raise ValueError("hierarchical values must be [B,A,...]")
    batch, agents = values.shape[:2]
    _validate_valid_agents(valid_agent_mask, batch=batch, agents=agents)
    if valid_agent_mask.device != values.device:
        raise TypeError("valid_agent_mask and values must share a device")
    expanded = _broadcast_mask(valid, values, name="valid target mask")
    agent_shape = (batch, agents) + (1,) * (values.ndim - 2)
    expanded = expanded & valid_agent_mask.reshape(agent_shape)
    axes = tuple(range(2, values.ndim))
    denominator = expanded.sum(dim=axes)
    missing = valid_agent_mask & denominator.eq(0)
    if not allow_empty_agents and bool(missing.any()):
        raise ValueError("every valid agent needs at least one target")

    # Multiplication cannot mask NaN (NaN*0 is NaN), whereas where() makes
    # optional invalid payloads inert and preserves zero gradients for them.
    numerator = torch.where(expanded, values, torch.zeros_like(values)).sum(
        dim=axes
    )
    per_agent = numerator / denominator.clamp_min(1).to(values)
    team_denominator = valid_agent_mask.sum(dim=1)
    per_team = (
        per_agent * valid_agent_mask.to(per_agent)
    ).sum(dim=1) / team_denominator.to(per_agent)
    return per_team.mean()


def s4_flow_loss(
    prediction: Tensor,
    target: Tensor,
    valid_agent_mask: Tensor,
    valid_horizon_mask: Tensor,
) -> Tensor:
    """MSE over ``[B,A,H,D]`` with equal valid agents and equal teams."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Flow prediction and target must share [B,A,H,D]")
    batch, agents, horizon, action_dim = prediction.shape
    _validate_valid_agents(valid_agent_mask, batch=batch, agents=agents)
    if valid_horizon_mask.shape == (batch, horizon):
        valid = valid_horizon_mask[:, None, :, None]
    elif valid_horizon_mask.shape == (batch, agents, horizon):
        valid = valid_horizon_mask[:, :, :, None]
    elif valid_horizon_mask.shape == prediction.shape:
        valid = valid_horizon_mask
    else:
        raise ValueError(
            "valid_horizon_mask must be [B,H], [B,A,H], or [B,A,H,D]"
        )
    if action_dim == 0:
        raise ValueError("Flow action dimension must be non-empty")
    values = (prediction.float() - target.float()).square()
    return hierarchical_agent_team_mean(
        values,
        valid,
        valid_agent_mask,
        allow_empty_agents=False,
    )


def s4_own_state_loss(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    valid_agent_mask: Tensor,
) -> Tensor:
    """Masked Smooth-L1 for own state ``[B,A,F,S]``."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("own state prediction and target must share [B,A,F,S]")
    values = F.smooth_l1_loss(
        prediction.float(), target.float(), reduction="none"
    )
    return hierarchical_agent_team_mean(
        values,
        valid,
        valid_agent_mask,
        allow_empty_agents=False,
    )


def _visual_distance(prediction: Tensor, target: Tensor) -> Tensor:
    return 1.0 - F.cosine_similarity(
        prediction.float(), target.float(), dim=-1, eps=1.0e-6
    )


def s4_own_visual_loss(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    valid_agent_mask: Tensor,
) -> Tensor:
    """Masked cosine distance for own visual ``[B,A,F,G,D]``."""

    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError(
            "own visual prediction and target must share [B,A,F,G,D]"
        )
    return hierarchical_agent_team_mean(
        _visual_distance(prediction, target),
        valid,
        valid_agent_mask,
        allow_empty_agents=True,
    )


def _expanded_peer_target(
    prediction: Tensor, target: Tensor, *, name: str
) -> Tensor:
    batch, focal_agents, target_agents = prediction.shape[:3]
    if focal_agents != target_agents:
        raise ValueError(f"{name} focal and target agent axes must match")
    if target.shape == prediction.shape:
        return target
    expected = (batch, target_agents, *prediction.shape[3:])
    if target.shape != expected:
        raise ValueError(
            f"{name} target must be {expected} or {tuple(prediction.shape)}"
        )
    return target[:, None].expand_as(prediction)


def _peer_mask(
    valid: Tensor,
    values: Tensor,
    valid_agent_mask: Tensor,
    *,
    name: str,
) -> Tensor:
    batch, focal_agents, target_agents = values.shape[:3]
    if valid.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype bool")
    if valid.device != values.device or valid_agent_mask.device != values.device:
        raise TypeError(f"{name}, values, and valid_agent_mask must share a device")
    if valid.shape == (batch, target_agents, values.shape[3]):
        target_valid = valid[:, None, :, :, None]
    elif valid.shape == (
        batch,
        focal_agents,
        target_agents,
        values.shape[3],
    ):
        target_valid = valid[:, :, :, :, None]
    elif valid.shape == values.shape:
        target_valid = valid
    else:
        raise ValueError(
            f"{name} must be [B,target,F], [B,focal,target,F], or full shape"
        )
    off_diagonal = ~torch.eye(
        focal_agents,
        dtype=torch.bool,
        device=valid_agent_mask.device,
    )
    pair_valid = (
        valid_agent_mask[:, :, None]
        & valid_agent_mask[:, None, :]
        & off_diagonal[None]
    )
    return target_valid & pair_valid[:, :, :, None, None]


def s4_peer_state_loss(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    valid_agent_mask: Tensor,
) -> Tensor:
    """Masked Smooth-L1 for peer state ``[B,focal,target,F,S]``."""

    if prediction.ndim != 5:
        raise ValueError("peer state prediction must be [B,focal,target,F,S]")
    _validate_valid_agents(
        valid_agent_mask, batch=prediction.shape[0], agents=prediction.shape[1]
    )
    expanded_target = _expanded_peer_target(
        prediction, target, name="peer state"
    )
    values = F.smooth_l1_loss(
        prediction.float(), expanded_target.float(), reduction="none"
    )
    mask = _peer_mask(
        valid, values, valid_agent_mask, name="peer state validity"
    )
    return hierarchical_agent_team_mean(
        values,
        mask,
        valid_agent_mask,
        allow_empty_agents=True,
    )


def s4_peer_visual_loss(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    valid_agent_mask: Tensor,
) -> Tensor:
    """Masked cosine distance for peer visual ``[B,focal,target,F,G,D]``."""

    if prediction.ndim != 6:
        raise ValueError(
            "peer visual prediction must be [B,focal,target,F,G,D]"
        )
    _validate_valid_agents(
        valid_agent_mask, batch=prediction.shape[0], agents=prediction.shape[1]
    )
    expanded_target = _expanded_peer_target(
        prediction, target, name="peer visual"
    )
    values = _visual_distance(prediction, expanded_target)
    mask = _peer_mask(
        valid, values, valid_agent_mask, name="peer visual validity"
    )
    return hierarchical_agent_team_mean(
        values,
        mask,
        valid_agent_mask,
        allow_empty_agents=True,
    )


def _expanded_shared_target(
    prediction: Tensor, target: Tensor
) -> Tensor:
    batch, agents = prediction.shape[:2]
    if target.shape == prediction.shape:
        return target
    expected = (batch, *prediction.shape[2:])
    if target.shape != expected:
        raise ValueError(
            f"shared visual target must be {expected} or {tuple(prediction.shape)}"
        )
    return target[:, None].expand_as(prediction)


def s4_shared_visual_loss(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    valid_agent_mask: Tensor,
) -> Tensor:
    """Masked cosine distance for shared visual ``[B,focal,F,G,D]``."""

    if prediction.ndim != 5:
        raise ValueError("shared visual prediction must be [B,focal,F,G,D]")
    batch, agents, futures, grid, _ = prediction.shape
    _validate_valid_agents(valid_agent_mask, batch=batch, agents=agents)
    expanded_target = _expanded_shared_target(prediction, target)
    values = _visual_distance(prediction, expanded_target)
    if valid.shape == (batch, futures):
        mask = valid[:, None, :, None]
    elif valid.shape == (batch, agents, futures):
        mask = valid[:, :, :, None]
    elif valid.shape == (batch, agents, futures, grid):
        mask = valid
    else:
        raise ValueError(
            "shared visual validity must be [B,F], [B,A,F], or [B,A,F,G]"
        )
    return hierarchical_agent_team_mean(
        values,
        mask,
        valid_agent_mask,
        allow_empty_agents=True,
    )


def _optional_loss(
    prediction: Tensor | None,
    target: Tensor | None,
    valid: Tensor | None,
    *,
    reference: Tensor,
    name: str,
    compute: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    valid_agent_mask: Tensor,
) -> Tensor:
    supplied = (prediction is not None, target is not None, valid is not None)
    if not any(supplied):
        return reference.float().reshape(-1)[:0].sum()
    if not all(supplied):
        raise ValueError(f"{name} prediction, target, and validity are all-or-none")
    assert prediction is not None and target is not None and valid is not None
    return compute(prediction, target, valid, valid_agent_mask)


def _loss_weight(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return normalized


def s4_joint_losses(
    *,
    flow_prediction: Tensor,
    flow_target: Tensor,
    flow_valid_mask: Tensor,
    valid_agent_mask: Tensor,
    own_state_prediction: Tensor,
    own_state_target: Tensor,
    own_state_valid_mask: Tensor,
    own_visual_prediction: Tensor,
    own_visual_target: Tensor,
    own_visual_valid_mask: Tensor,
    peer_state_prediction: Tensor | None = None,
    peer_state_target: Tensor | None = None,
    peer_state_valid_mask: Tensor | None = None,
    peer_visual_prediction: Tensor | None = None,
    peer_visual_target: Tensor | None = None,
    peer_visual_valid_mask: Tensor | None = None,
    shared_visual_prediction: Tensor | None = None,
    shared_visual_target: Tensor | None = None,
    shared_visual_valid_mask: Tensor | None = None,
    flow_loss_weight: float = 1.0,
    state_loss_weight: float = 0.25,
    visual_loss_weight: float = 0.25,
) -> S4JointLoss:
    """Build the fixed R7/R8 Flow + own/peer/shared future objective."""

    flow = s4_flow_loss(
        flow_prediction,
        flow_target,
        valid_agent_mask,
        flow_valid_mask,
    )
    own_state = s4_own_state_loss(
        own_state_prediction,
        own_state_target,
        own_state_valid_mask,
        valid_agent_mask,
    )
    own_visual = s4_own_visual_loss(
        own_visual_prediction,
        own_visual_target,
        own_visual_valid_mask,
        valid_agent_mask,
    )
    peer_state = _optional_loss(
        peer_state_prediction,
        peer_state_target,
        peer_state_valid_mask,
        reference=own_state_prediction,
        name="peer state",
        compute=s4_peer_state_loss,
        valid_agent_mask=valid_agent_mask,
    )
    peer_visual = _optional_loss(
        peer_visual_prediction,
        peer_visual_target,
        peer_visual_valid_mask,
        reference=own_visual_prediction,
        name="peer visual",
        compute=s4_peer_visual_loss,
        valid_agent_mask=valid_agent_mask,
    )
    shared_visual = _optional_loss(
        shared_visual_prediction,
        shared_visual_target,
        shared_visual_valid_mask,
        reference=own_visual_prediction,
        name="shared visual",
        compute=s4_shared_visual_loss,
        valid_agent_mask=valid_agent_mask,
    )
    state = (own_state + peer_state) / 2.0
    visual = (own_visual + peer_visual + shared_visual) / 3.0
    total = (
        _loss_weight(flow_loss_weight, name="flow_loss_weight") * flow
        + _loss_weight(state_loss_weight, name="state_loss_weight") * state
        + _loss_weight(visual_loss_weight, name="visual_loss_weight") * visual
    )
    return S4JointLoss(
        total=total,
        flow=flow,
        state=state,
        visual=visual,
        own_state=own_state,
        peer_state=peer_state,
        own_visual=own_visual,
        peer_visual=peer_visual,
        shared_visual=shared_visual,
    )


S4JointLosses = S4JointLoss
joint_s4_losses = s4_joint_losses
s4_joint_loss = s4_joint_losses


__all__ = [
    "S4JointLoss",
    "S4JointLosses",
    "hierarchical_agent_team_mean",
    "joint_s4_losses",
    "s4_flow_loss",
    "s4_joint_losses",
    "s4_joint_loss",
    "s4_own_state_loss",
    "s4_own_visual_loss",
    "s4_peer_state_loss",
    "s4_peer_visual_loss",
    "s4_shared_visual_loss",
]
