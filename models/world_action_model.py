"""A small environment-agnostic one-step world-action model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.api import WorldModelInputs, WorldModelOutput


@dataclass(frozen=True)
class WorldActionModelConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    hidden_layers: int = 3
    predict_delta: bool = True

    def __post_init__(self) -> None:
        for name in ("state_dim", "action_dim", "hidden_dim", "hidden_layers"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


class WorldActionModel(nn.Module):
    """Predict one transition using only the supplied state and action tensors."""

    def __init__(self, config: WorldActionModelConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        input_dim = config.state_dim + config.action_dim
        for index in range(config.hidden_layers):
            layers.append(
                nn.Linear(
                    input_dim if index == 0 else config.hidden_dim, config.hidden_dim
                )
            )
            layers.append(nn.SiLU())
        layers.append(nn.Linear(config.hidden_dim, config.state_dim + 2))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: WorldModelInputs) -> WorldModelOutput:
        self._validate(inputs)
        raw = self.network(torch.cat((inputs.state, inputs.action), dim=-1))
        state_prediction = raw[..., : self.config.state_dim]
        reward = raw[..., self.config.state_dim : self.config.state_dim + 1]
        done_logit = raw[..., self.config.state_dim + 1 :]
        next_state = (
            inputs.state + state_prediction
            if self.config.predict_delta
            else state_prediction
        )
        return WorldModelOutput(
            next_state=next_state,
            reward=reward,
            done_logit=done_logit,
            diagnostics={"state_prediction": state_prediction},
        )

    def _validate(self, inputs: WorldModelInputs) -> None:
        state = inputs.state
        action = inputs.action
        if state.ndim == 0 or state.shape[-1] != self.config.state_dim:
            raise ValueError(
                f"state must end in dimension {self.config.state_dim}, got {tuple(state.shape)}"
            )
        if action.ndim == 0 or action.shape[-1] != self.config.action_dim:
            raise ValueError(
                f"action must end in dimension {self.config.action_dim}, got {tuple(action.shape)}"
            )
        if state.shape[:-1] != action.shape[:-1]:
            raise ValueError("state and action must have identical leading dimensions")
        if not torch.is_floating_point(state) or not torch.is_floating_point(action):
            raise TypeError("state and action must be floating-point tensors")
        if state.dtype != action.dtype or state.device != action.device:
            raise TypeError("state and action must share dtype and device")


__all__ = ["WorldActionModel", "WorldActionModelConfig"]
