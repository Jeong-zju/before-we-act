"""Validated tensor-shape configuration for the Phase 1 RWM-AR model."""

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


__all__ = ["RWMARConfig"]
