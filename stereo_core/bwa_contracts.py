"""Frozen Before-We-Act contracts for the CoRE-native model path.

The types in this module deliberately contain no trainable state.  R9 freezes
their deployment boundary; later rounds may register extensions, but may not
add privileged observations or change the native action/temporal semantics.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class CoreViewTokens:
    """Frozen local/global tokens and the parent's original fusion output."""

    local_tokens: torch.Tensor
    global_tokens: torch.Tensor
    parent_fused: torch.Tensor


@dataclass(frozen=True)
class CoreDeploymentContext:
    """Deployment-legal temporal context; future/label fields are forbidden."""

    view_token_history: torch.Tensor | None = None
    qpos_history: torch.Tensor | None = None
    executed_action_history: torch.Tensor | None = None
    history_mask: torch.Tensor | None = None
    episode_reset: torch.Tensor | None = None
    fixed_camera_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        metadata = self.fixed_camera_metadata or {}
        forbidden = {
            "future",
            "future_rgb",
            "future_tokens",
            "target",
            "task_id",
            "agent_id",
            "simulator_state",
            "privileged_state",
        }
        present = forbidden.intersection(metadata)
        if present:
            raise ValueError(f"privileged deployment metadata is forbidden: {sorted(present)}")
        allowed = {
            "global_view_mask",
            "local_view_mask",
            "diagnostic_intervention",
            "calibration_sha256",
        }
        unknown = set(metadata) - allowed
        if unknown:
            raise ValueError(f"unknown deployment metadata is forbidden: {sorted(unknown)}")


@dataclass(frozen=True)
class PerceptionOutput:
    """A perception extension's residual, auxiliary losses and diagnostics."""

    tokens: torch.Tensor
    auxiliary: Mapping[str, torch.Tensor] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class CorePerceptionExtension(nn.Module, ABC):
    """Registered R10 extension interface; R9's default remains identity."""

    kind: str = "abstract"

    @abstractmethod
    def forward(
        self,
        views: CoreViewTokens,
        state_vec: torch.Tensor,
        deployment_context: CoreDeploymentContext | None,
    ) -> PerceptionOutput:
        """Return a residual shaped exactly like ``views.parent_fused``."""

    @property
    @abstractmethod
    def perception_gate(self) -> torch.Tensor:
        """Return the scalar/vector gate applied only by the frozen parent."""


@dataclass(frozen=True)
class CoreContext:
    """One encoded CoRE decision shared by native and forced-role decoding."""

    views: CoreViewTokens
    observation: torch.Tensor
    state_vec: torch.Tensor
    latent: torch.Tensor
    memory: torch.Tensor
    query: torch.Tensor
    dense_routes: torch.Tensor
    sparse_routes: torch.Tensor
    provenance: Mapping[str, Any]
    auxiliary: Mapping[str, torch.Tensor] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CoreCandidateBank:
    """Normalized CoRE chunks: candidate 0 plus one chunk per forced role."""

    chunks: torch.Tensor
    source: Sequence[str]
    routes: torch.Tensor
    valid_mask: torch.Tensor
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.chunks.ndim != 4:
            raise ValueError(f"chunks must be [B,K,H,D], got {tuple(self.chunks.shape)}")
        batch, candidates = self.chunks.shape[:2]
        if candidates != len(self.source):
            raise ValueError("candidate source count does not match chunks")
        if self.valid_mask.shape != (batch, candidates):
            raise ValueError("valid_mask must be [B,K]")
        if self.routes.shape[:2] != (batch, candidates):
            raise ValueError("routes must begin with [B,K]")


def apply_perception_residual(
    views: CoreViewTokens,
    state_vec: torch.Tensor,
    deployment_context: CoreDeploymentContext | None,
    extension: CorePerceptionExtension | None,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor], Mapping[str, Any]]:
    """The only legal R10 injection boundary: ``x0 + tanh(g) * delta``."""

    if extension is None:
        return views.parent_fused, {}, {"perception_extension": "identity"}
    output = extension(views, state_vec, deployment_context)
    if output.tokens.shape != views.parent_fused.shape:
        raise ValueError(
            "perception residual shape mismatch: "
            f"{tuple(output.tokens.shape)} != {tuple(views.parent_fused.shape)}"
        )
    gate = extension.perception_gate
    observation = views.parent_fused + torch.tanh(gate).to(
        dtype=output.tokens.dtype, device=output.tokens.device
    ) * output.tokens
    diagnostics = dict(output.diagnostics)
    diagnostics["perception_extension"] = extension.kind
    diagnostics["perception_gate"] = gate.detach()
    return observation, output.auxiliary, diagnostics
