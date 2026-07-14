"""Tensor-only contracts for policy and world-action models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable

from torch import Tensor


@dataclass(frozen=True)
class PolicyInputs:
    """All information that a VLA-style policy may consume for one batch."""

    state: Tensor
    images: Mapping[str, Tensor] = field(default_factory=dict)
    language_tokens: Tensor | None = None
    attention_mask: Tensor | None = None
    context: Mapping[str, Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyOutput:
    """Policy prediction with optional tensor diagnostics."""

    action: Tensor
    diagnostics: Mapping[str, Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class WorldModelInputs:
    """Explicit inputs for one-step world-action prediction."""

    state: Tensor
    action: Tensor
    context: Mapping[str, Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class WorldModelOutput:
    """One-step state, reward, and termination predictions."""

    next_state: Tensor
    reward: Tensor
    done_logit: Tensor
    diagnostics: Mapping[str, Tensor] = field(default_factory=dict)


@runtime_checkable
class PolicyModel(Protocol):
    """Structural contract implemented by VLA-style policy models."""

    def forward(self, inputs: PolicyInputs) -> PolicyOutput: ...


@runtime_checkable
class WorldModel(Protocol):
    """Structural contract implemented by world-action models."""

    def forward(self, inputs: WorldModelInputs) -> WorldModelOutput: ...


__all__ = [
    "PolicyInputs",
    "PolicyModel",
    "PolicyOutput",
    "WorldModel",
    "WorldModelInputs",
    "WorldModelOutput",
]
