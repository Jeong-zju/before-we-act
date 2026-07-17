"""Sequence and imagined-rollout tensor contracts for recurrent world model and later."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from torch import Tensor


@dataclass(frozen=True)
class WorldModelSequenceInputs:
    """Episode-safe history used to infer the current recurrent belief."""

    states: Tensor
    past_actions: Tensor
    valid_mask: Tensor


@dataclass(frozen=True)
class WorldModelRolloutInputs:
    """A history plus candidate future actions for imagined rollout."""

    history: WorldModelSequenceInputs
    candidate_actions: Tensor
    num_particles: int = 1

    def __post_init__(self) -> None:
        if self.num_particles <= 0:
            raise ValueError("num_particles must be positive")


@dataclass(frozen=True)
class WorldModelRolloutOutput:
    """Phase-stable rollout surface; implementation details stay in diagnostics."""

    state_distribution: Mapping[str, Tensor]
    rewards: Tensor
    termination: Mapping[str, Tensor]
    uncertainty: Mapping[str, Tensor]
    diagnostics: Mapping[str, Tensor] = field(default_factory=dict)


__all__ = [
    "WorldModelRolloutInputs",
    "WorldModelRolloutOutput",
    "WorldModelSequenceInputs",
]
