"""Deployable proprioception features for the Phase 1 RWM-AR."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from models.wam.normalizer import NormalizationStats


class StateFeatureEncoder(nn.Module):
    """Normalize continuous state/action values and expand yaw as sin/cos.

    The module accepts only deployable state and commanded action tensors.  No
    privileged event fields exist in its signature or parameters.
    """

    def __init__(
        self,
        stats: NormalizationStats,
        *,
        yaw_indices: Sequence[int] = (2, 13),
    ) -> None:
        super().__init__()
        state_dim = int(stats.state_mean.shape[0])
        action_dim = int(stats.action_mean.shape[0])
        yaw = tuple(int(index) for index in yaw_indices)
        if len(set(yaw)) != len(yaw):
            raise ValueError("yaw_indices must be unique")
        if any(index < 0 or index >= state_dim for index in yaw):
            raise ValueError("yaw_indices contain an out-of-range index")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.yaw_indices = yaw
        self.feature_dim = state_dim + len(yaw)
        self.register_buffer(
            "state_mean", torch.as_tensor(stats.state_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "state_std", torch.as_tensor(stats.state_std, dtype=torch.float32)
        )
        self.register_buffer(
            "action_mean", torch.as_tensor(stats.action_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "action_std", torch.as_tensor(stats.action_std, dtype=torch.float32)
        )

    def encode_state(self, state: Tensor) -> Tensor:
        self._validate_last_dim(state, self.state_dim, "state")
        normalized = (state - self.state_mean) / self.state_std
        parts: list[Tensor] = []
        yaw_set = set(self.yaw_indices)
        for index in range(self.state_dim):
            if index in yaw_set:
                parts.extend(
                    (
                        torch.sin(state[..., index : index + 1]),
                        torch.cos(state[..., index : index + 1]),
                    )
                )
            else:
                parts.append(normalized[..., index : index + 1])
        return torch.cat(parts, dim=-1)

    def normalize_action(self, action: Tensor) -> Tensor:
        self._validate_last_dim(action, self.action_dim, "action")
        return (action - self.action_mean) / self.action_std

    @staticmethod
    def _validate_last_dim(value: Tensor, expected: int, name: str) -> None:
        if not torch.is_floating_point(value):
            raise TypeError(f"{name} must be floating point")
        if value.ndim == 0 or value.shape[-1] != expected:
            raise ValueError(
                f"{name} must end in dimension {expected}, got {tuple(value.shape)}"
            )


__all__ = ["StateFeatureEncoder"]
