"""Pure CARE-v2 causal target and near-term loss helpers."""
from __future__ import annotations

import torch


def legal_decision_count(length: int, action_lag_rows: int = 1) -> int:
    count = int(length) - int(action_lag_rows)
    if count < 1:
        raise ValueError("episode has no legal causal decision")
    return count


def target_bounds(
    episode_start: int,
    episode_end: int,
    local_t: int,
    horizon: int,
    action_lag_rows: int = 1,
) -> tuple[int, int]:
    count = legal_decision_count(episode_end - episode_start, action_lag_rows)
    if not 0 <= int(local_t) < count:
        raise IndexError(local_t)
    first = int(episode_start) + int(local_t) + int(action_lag_rows)
    return first, min(int(episode_end), first + int(horizon))


def executed_history_bounds(local_t: int, history_steps: int) -> tuple[int, int]:
    """Return [first,end) released action rows visible at post-action row t.

    Row zero is intentionally excluded: reset-time deployment has observed no
    policy-issued action yet.  At t>0, row t is the latest executed command.
    """

    local_t = int(local_t)
    return max(1, local_t - int(history_steps) + 1), local_t + 1


def horizon_weights(
    horizon: int,
    decay_steps: float,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if horizon < 1 or decay_steps < 0:
        raise ValueError("invalid horizon weighting contract")
    offsets = torch.arange(horizon, device=device, dtype=dtype)
    if decay_steps == 0:
        return torch.ones_like(offsets)
    return torch.exp(-offsets / float(decay_steps))
