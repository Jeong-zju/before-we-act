"""Validated configuration for the recurrent world model and action policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RWMARConfig:
    """Architecture defaults fixed by the technical plan for recurrent world model."""

    state_dim: int = 22
    action_dim: int = 8
    history_horizon: int = 32
    train_forecast_horizon: int = 16
    planning_horizon: int = 20
    encoder_hidden_dim: int = 256
    gru_hidden_dim: int = 256
    gru_layers: int = 2
    dropout: float = 0.0
    predict_delta: bool = True
    min_log_std: float = -8.0
    max_log_std: float = 2.0
    yaw_indices: tuple[int, ...] = (2, 13)
    gripper_closed_indices: tuple[int, ...] = (7, 18)

    def __post_init__(self) -> None:
        integer_fields = (
            "state_dim",
            "action_dim",
            "history_horizon",
            "train_forecast_horizon",
            "planning_horizon",
            "encoder_hidden_dim",
            "gru_hidden_dim",
            "gru_layers",
        )
        for name in integer_fields:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if self.min_log_std >= self.max_log_std:
            raise ValueError("min_log_std must be smaller than max_log_std")
        for name in ("yaw_indices", "gripper_closed_indices"):
            indices = tuple(int(index) for index in getattr(self, name))
            if len(indices) != len(set(indices)):
                raise ValueError(f"{name} must be unique")
            if any(index < 0 or index >= self.state_dim for index in indices):
                raise ValueError(f"{name} contains an out-of-range index")
            object.__setattr__(self, name, indices)


@dataclass(frozen=True)
class ActionPriorConfig:
    """Belief-conditioned tanh-Gaussian behavior prior."""

    feature_dim: int
    action_dim: int = 8
    hidden_dim: int = 256
    hidden_layers: int = 2
    min_log_std: float = -5.0
    max_log_std: float = 1.0

    def __post_init__(self) -> None:
        for name in ("feature_dim", "action_dim", "hidden_dim", "hidden_layers"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_log_std >= self.max_log_std:
            raise ValueError("min_log_std must be smaller than max_log_std")


@dataclass(frozen=True)
class ActionChunkConfig:
    """Locked Joint WAM action-chunk and warm-start contract."""

    action_dim: int = 8
    horizon: int = 8
    execution_steps: int = 2
    solver_steps: int = 4
    warm_start_mode: str = "shift_repeat_last"

    def __post_init__(self) -> None:
        for name in ("action_dim", "horizon", "execution_steps", "solver_steps"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.execution_steps >= self.horizon:
            raise ValueError("execution_steps must be smaller than horizon")
        if self.warm_start_mode != "shift_repeat_last":
            raise ValueError("Joint WAM requires shift_repeat_last warm start")


@dataclass(frozen=True)
class StatefulActionFlowConfig:
    """Single-expert rectified flow used by Joint WAM."""

    feature_dim: int
    action_dim: int = 8
    horizon: int = 8
    hidden_dim: int = 512
    hidden_layers: int = 4
    time_embedding_dim: int = 32
    anchor_hidden_dim: int = 256
    anchor_hidden_layers: int = 2
    anchor_min_log_std: float = -5.0
    anchor_max_log_std: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "feature_dim",
            "action_dim",
            "horizon",
            "hidden_dim",
            "hidden_layers",
            "time_embedding_dim",
            "anchor_hidden_dim",
            "anchor_hidden_layers",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.time_embedding_dim % 2:
            raise ValueError("time_embedding_dim must be even")
        if self.anchor_min_log_std >= self.anchor_max_log_std:
            raise ValueError("invalid anchor log-std bounds")


__all__ = [
    "ActionChunkConfig",
    "ActionPriorConfig",
    "RWMARConfig",
    "StatefulActionFlowConfig",
]
