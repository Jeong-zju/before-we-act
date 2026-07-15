"""Tensor-only models used exclusively by the accepted Phase 0 baseline suite."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.api import PolicyInputs, PolicyOutput, WorldModelInputs, WorldModelOutput


@dataclass(frozen=True)
class LinearWorldModelConfig:
    state_dim: int
    action_dim: int

    def __post_init__(self) -> None:
        if self.state_dim <= 0 or self.action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")


class LinearWorldModel(nn.Module):
    """Linear dynamics, reward, done, success, and failure baseline."""

    def __init__(self, config: LinearWorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.state_dim + config.action_dim, config.state_dim + 4)

    def forward(self, inputs: WorldModelInputs) -> WorldModelOutput:
        _validate_world_inputs(inputs, self.config.state_dim, self.config.action_dim)
        raw = self.linear(torch.cat((inputs.state, inputs.action), dim=-1))
        state_end = self.config.state_dim
        return WorldModelOutput(
            next_state=raw[..., :state_end],
            reward=raw[..., state_end : state_end + 1],
            done_logit=raw[..., state_end + 1 : state_end + 2],
            success_logit=raw[..., state_end + 2 : state_end + 3],
            failure_logit=raw[..., state_end + 3 : state_end + 4],
            diagnostics={},
        )


@dataclass(frozen=True)
class OneStepMLPWorldModelConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    hidden_layers: int = 3
    predict_delta: bool = True

    def __post_init__(self) -> None:
        for name in ("state_dim", "action_dim", "hidden_dim", "hidden_layers"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")


class OneStepMLPWorldModel(nn.Module):
    """Phase 0 one-step MLP, intentionally distinct from the Phase 1 RWM."""

    def __init__(self, config: OneStepMLPWorldModelConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        input_dim = config.state_dim + config.action_dim
        for index in range(config.hidden_layers):
            layers.extend(
                (
                    nn.Linear(
                        input_dim if index == 0 else config.hidden_dim,
                        config.hidden_dim,
                    ),
                    nn.SiLU(),
                )
            )
        layers.append(nn.Linear(config.hidden_dim, config.state_dim + 4))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: WorldModelInputs) -> WorldModelOutput:
        _validate_world_inputs(inputs, self.config.state_dim, self.config.action_dim)
        raw = self.network(torch.cat((inputs.state, inputs.action), dim=-1))
        state_end = self.config.state_dim
        state_prediction = raw[..., :state_end]
        next_state = (
            inputs.state + state_prediction
            if self.config.predict_delta
            else state_prediction
        )
        return WorldModelOutput(
            next_state=next_state,
            reward=raw[..., state_end : state_end + 1],
            done_logit=raw[..., state_end + 1 : state_end + 2],
            success_logit=raw[..., state_end + 2 : state_end + 3],
            failure_logit=raw[..., state_end + 3 : state_end + 4],
            diagnostics={"state_prediction": state_prediction},
        )


@dataclass(frozen=True)
class ActionPriorConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 128
    hidden_layers: int = 2

    def __post_init__(self) -> None:
        for name in ("state_dim", "action_dim", "hidden_dim", "hidden_layers"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")


class ActionPrior(nn.Module):
    """Behavior-cloning prior that consumes only the deployable proprioception."""

    def __init__(self, config: ActionPriorConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        for index in range(config.hidden_layers):
            layers.extend(
                (
                    nn.Linear(
                        config.state_dim if index == 0 else config.hidden_dim,
                        config.hidden_dim,
                    ),
                    nn.SiLU(),
                )
            )
        layers.append(nn.Linear(config.hidden_dim, config.action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: PolicyInputs) -> PolicyOutput:
        state = inputs.state
        if state.ndim == 0 or state.shape[-1] != self.config.state_dim:
            raise ValueError(
                f"state must end in dimension {self.config.state_dim}, "
                f"got {tuple(state.shape)}"
            )
        if not torch.is_floating_point(state):
            raise TypeError("state must be a floating-point tensor")
        return PolicyOutput(action=torch.tanh(self.network(state)), diagnostics={})


def _validate_world_inputs(
    inputs: WorldModelInputs,
    state_dim: int,
    action_dim: int,
) -> None:
    if inputs.state.ndim == 0 or inputs.state.shape[-1] != state_dim:
        raise ValueError(
            f"state must end in dimension {state_dim}, got {tuple(inputs.state.shape)}"
        )
    if inputs.action.ndim == 0 or inputs.action.shape[-1] != action_dim:
        raise ValueError(
            f"action must end in dimension {action_dim}, got {tuple(inputs.action.shape)}"
        )
    if inputs.state.shape[:-1] != inputs.action.shape[:-1]:
        raise ValueError("state and action must have identical leading dimensions")
    if not torch.is_floating_point(inputs.state) or not torch.is_floating_point(
        inputs.action
    ):
        raise TypeError("state and action must be floating-point tensors")
    if (
        inputs.state.dtype != inputs.action.dtype
        or inputs.state.device != inputs.action.device
    ):
        raise TypeError("state and action must share dtype and device")


__all__ = [
    "ActionPrior",
    "ActionPriorConfig",
    "LinearWorldModel",
    "LinearWorldModelConfig",
    "OneStepMLPWorldModel",
    "OneStepMLPWorldModelConfig",
]
