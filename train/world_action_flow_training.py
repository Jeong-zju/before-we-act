"""S3-R6 grouped Rectified-Flow objectives."""

from __future__ import annotations

import torch
from torch import Tensor


def grouped_flow_matching_batch(
    target_actions: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    if target_actions.ndim != 4:
        raise ValueError("target_actions must be [B,A,H,D]")
    source = torch.randn(
        target_actions.shape,
        device=target_actions.device,
        dtype=target_actions.dtype,
        generator=generator,
    )
    tau = torch.rand(
        target_actions.shape[0],
        device=target_actions.device,
        dtype=target_actions.dtype,
        generator=generator,
    )
    interpolation = tau[:, None, None, None]
    return (
        (1.0 - interpolation) * source + interpolation * target_actions,
        target_actions - source,
        tau,
    )


def grouped_masked_flow_mse(
    prediction: Tensor,
    target: Tensor,
    valid_agent_mask: Tensor,
    valid_horizon_mask: Tensor,
) -> Tensor:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must share [B,A,H,D]")
    if valid_agent_mask.shape != prediction.shape[:2]:
        raise ValueError("valid_agent_mask must be [B,A]")
    if valid_horizon_mask.shape != (prediction.shape[0], prediction.shape[2]):
        raise ValueError("valid_horizon_mask must be [B,H]")
    mask = (
        valid_agent_mask[:, :, None, None]
        & valid_horizon_mask[:, None, :, None]
    ).expand_as(prediction)
    weights = mask.to(prediction)
    denominator = weights.sum(dim=(1, 2, 3))
    if not bool(denominator.gt(0).all()):
        raise ValueError("every grouped sample needs a valid action")
    per_sample = ((prediction - target).square() * weights).sum(
        dim=(1, 2, 3)
    ) / denominator
    return per_sample.mean()


__all__ = ["grouped_flow_matching_batch", "grouped_masked_flow_mse"]
