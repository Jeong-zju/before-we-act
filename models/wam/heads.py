"""Probability and auxiliary prediction heads for a single RWM member."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass(frozen=True)
class RWMHeadOutput:
    normalized_delta_mean: Tensor
    normalized_delta_log_std: Tensor
    gripper_closed_logit: Tensor
    reward_symlog: Tensor
    done_logit: Tensor
    success_logit: Tensor
    failure_logit: Tensor
    response_progress: Tensor
    coordination_error: Tensor
    executed_action: Tensor


class RWMHeads(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        action_dim: int,
        *,
        closed_count: int,
        min_log_std: float,
        max_log_std: float,
    ) -> None:
        super().__init__()
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.delta_mean = nn.Linear(hidden_dim, state_dim)
        self.delta_log_std = nn.Linear(hidden_dim, state_dim)
        self.gripper_closed = nn.Linear(hidden_dim, closed_count)
        self.reward = nn.Linear(hidden_dim, 1)
        self.done = nn.Linear(hidden_dim, 1)
        self.success = nn.Linear(hidden_dim, 1)
        self.failure = nn.Linear(hidden_dim, 1)
        self.response_progress_head = nn.Linear(hidden_dim, 1)
        self.coordination_error_head = nn.Linear(hidden_dim, 1)
        self.executed_action_head = nn.Linear(hidden_dim, action_dim)
        nn.init.constant_(self.delta_log_std.bias, -2.0)

    def forward(self, hidden: Tensor) -> RWMHeadOutput:
        return RWMHeadOutput(
            normalized_delta_mean=self.delta_mean(hidden),
            normalized_delta_log_std=self.delta_log_std(hidden).clamp(
                self.min_log_std, self.max_log_std
            ),
            gripper_closed_logit=self.gripper_closed(hidden),
            reward_symlog=self.reward(hidden),
            done_logit=self.done(hidden),
            success_logit=self.success(hidden),
            failure_logit=self.failure(hidden),
            response_progress=self.response_progress_head(hidden),
            coordination_error=self.coordination_error_head(hidden),
            executed_action=self.executed_action_head(hidden).tanh(),
        )


__all__ = ["RWMHeadOutput", "RWMHeads"]
