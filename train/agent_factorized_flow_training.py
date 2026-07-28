"""Rectified-Flow sampling and uniform masked loss for S1-R1 F1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FlowMatchingBatch:
    action_inputs: Tensor
    target_velocity: Tensor
    initial_actions: Tensor
    flow_time: Tensor


def make_flow_matching_batch(
    target_actions: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> FlowMatchingBatch:
    if target_actions.ndim != 3:
        raise ValueError("target_actions must be [B,H,A]")
    initial = torch.randn(
        target_actions.shape,
        device=target_actions.device,
        dtype=target_actions.dtype,
        generator=generator,
    )
    flow_time = torch.rand(
        target_actions.shape[0],
        device=target_actions.device,
        dtype=target_actions.dtype,
        generator=generator,
    )
    interpolation = flow_time[:, None, None]
    return FlowMatchingBatch(
        action_inputs=(1.0 - interpolation) * initial
        + interpolation * target_actions,
        target_velocity=target_actions - initial,
        initial_actions=initial,
        flow_time=flow_time,
    )


def uniform_masked_flow_mse(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
) -> Tensor:
    """Average valid dimensions per agent sample, then average agents equally."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share [B,H,A]")
    if valid.shape == prediction.shape[:2]:
        mask = valid[:, :, None].expand_as(prediction)
    elif valid.shape == prediction.shape:
        mask = valid
    else:
        raise ValueError("valid must be [B,H] or [B,H,A]")
    weights = mask.to(prediction.dtype)
    denominator = weights.sum(dim=(1, 2))
    if not bool(denominator.gt(0).all()):
        raise ValueError("every per-agent sample needs at least one valid action")
    per_agent = (
        (prediction - target).square() * weights
    ).sum(dim=(1, 2)) / denominator
    return per_agent.mean()


__all__ = [
    "FlowMatchingBatch",
    "make_flow_matching_batch",
    "uniform_masked_flow_mse",
]
