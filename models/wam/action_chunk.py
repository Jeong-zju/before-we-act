"""Stateful action-chunk primitives used by Joint WAM."""

from __future__ import annotations

import torch
from torch import Tensor

from models.wam.config import ActionChunkConfig


def shift_action_chunk_warm_start(
    previous_chunk: Tensor,
    config: ActionChunkConfig,
    *,
    executed_steps: int | None = None,
) -> Tensor:
    """Shift an existing chunk by actual executed steps and repeat its tail.

    The returned tensor has the same shape as the input.  This is the only
    warm-start meaning: observations condition the later Flow correction, while
    this function deterministically preserves the unexecuted temporal plan.
    """

    if previous_chunk.ndim < 2:
        raise ValueError("previous_chunk must end in [horizon,action_dim]")
    if previous_chunk.shape[-2:] != (config.horizon, config.action_dim):
        raise ValueError(
            "previous_chunk must end in "
            f"[{config.horizon},{config.action_dim}]"
        )
    if not torch.is_floating_point(previous_chunk):
        raise TypeError("previous_chunk must be floating point")
    if not bool(torch.isfinite(previous_chunk).all()):
        raise ValueError("previous_chunk must be finite")
    steps = config.execution_steps if executed_steps is None else int(executed_steps)
    if steps <= 0 or steps >= config.horizon:
        raise ValueError("executed_steps must be in [1,horizon)")
    remaining = previous_chunk[..., steps:, :]
    tail = previous_chunk[..., -1:, :].expand(
        *previous_chunk.shape[:-2], steps, config.action_dim
    )
    return torch.cat((remaining, tail), dim=-2)


__all__ = ["shift_action_chunk_warm_start"]
