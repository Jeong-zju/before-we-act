"""Validated configuration for recurrent WAM members and ensembles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RWMARConfig:
    """Architecture defaults fixed by the technical plan for Phase 1."""

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
class RWMUEnsembleConfig:
    """Independent-member configuration for the Phase 2 RWM-U ensemble."""

    ensemble_size: int = 5
    bootstrap: bool = True

    def __post_init__(self) -> None:
        if self.ensemble_size < 2:
            raise ValueError("RWM-U ensemble_size must be at least 2")
        if not self.bootstrap:
            raise ValueError("Phase 2 RWM-U requires episode bootstrap sampling")


@dataclass(frozen=True)
class RWMURiskConfig:
    """Weights for planner-facing risk scores; rewards remain outside this API."""

    epistemic_weight: float = 1.0
    aleatoric_weight: float = 0.1
    failure_weight: float = 1.0
    action_ood_weight: float = 0.5
    action_ood_threshold: float = 3.0

    def __post_init__(self) -> None:
        for name in (
            "epistemic_weight",
            "aleatoric_weight",
            "failure_weight",
            "action_ood_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.action_ood_threshold <= 0.0:
            raise ValueError("action_ood_threshold must be positive")


@dataclass(frozen=True)
class WAMPlanningHeadConfig:
    """Belief-conditioned behavior prior and Monte-Carlo value head."""

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


__all__ = [
    "RWMARConfig",
    "RWMUEnsembleConfig",
    "RWMURiskConfig",
    "WAMPlanningHeadConfig",
]
