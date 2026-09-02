"""Protocol-isolated learned proposal prototype for MARS CARE.

This module is deliberately not imported by the frozen formal CARE path.  It
only defines a candidate generator and its initialization loss; runtime
execution, branch collection, scoring, calibration, and selection remain
unchanged until a separate protocol opts into a learned-proposal checkpoint.

Candidate zero is an external B-core/TUNE reference chunk and is copied into
the output without arithmetic, so it remains element-for-element identical.
The other slots share one strictly local memory reader and one residual output
projection.  Slot embeddings are their only slot-specific parameters.

The existing prepared fixed candidate library may initialize these slots, but
it is a *bootstrap-only* behavioral prior.  Its old branch outcomes are not
valid supervision for learned candidates.  After initialization, every
learned candidate family needs fresh paired reactive/replay branch collection.
The normalized residual bound is not a physical legality certificate: a later
runtime adapter must de-normalize, enforce action/rate/gripper bounds, and fail
closed to candidate zero before any learned alternative is executed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F


FIXED_LIBRARY_LABEL_USAGE = "bootstrap_only"


@dataclass(frozen=True)
class MARSCARELearnedProposalConfig:
    """Frozen tensor contract for the MARS learned-proposal prototype."""

    d_model: int = 384
    action_horizon: int = 100
    action_dim: int = 8
    candidates: int = 6
    heads: int = 8
    hidden_width: int = 768
    dropout: float = 0.1
    normalized_residual_limit: tuple[float, ...] = (0.25,) * 8

    def __post_init__(self) -> None:
        if self.action_horizon != 100 or self.action_dim != 8:
            raise ValueError("MARS learned proposals require H=100 and A=8")
        if self.candidates < 2:
            raise ValueError(
                "MARS learned proposals require a reference and an alternative"
            )
        if self.d_model <= 0 or self.d_model % self.heads:
            raise ValueError("proposal width must be positive and divisible by heads")
        if self.hidden_width <= 0:
            raise ValueError("proposal hidden width must be positive")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("proposal dropout must lie in [0,1)")
        if len(self.normalized_residual_limit) != self.action_dim:
            raise ValueError("normalized residual limit must be action-dimensional")
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in self.normalized_residual_limit
        ):
            raise ValueError("normalized residual limits must be finite and positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "MARSCARELearnedProposalConfig":
        row = dict(value)
        if "normalized_residual_limit" in row:
            row["normalized_residual_limit"] = tuple(
                row["normalized_residual_limit"]
            )
        return cls(**row)


@dataclass
class MARSCARELearnedProposalOutput:
    """Learned candidates and auditable normalized residuals."""

    candidates_normalized: torch.Tensor
    residuals_normalized: torch.Tensor
    slot_state: torch.Tensor
    reference_normalized: torch.Tensor


class MARSCARELearnedProposalHead(nn.Module):
    """Generate bounded alternatives around an exact external reference.

    ``local_memory`` is the legal per-arm belief/action context.  No arm ID,
    peer state, peer action, simulator state, or privileged task signal is
    accepted by this interface.  Every alternative attends to the same memory;
    learned slot embeddings provide stable candidate identities.
    """

    def __init__(self, config: MARSCARELearnedProposalConfig) -> None:
        super().__init__()
        self.config = config
        alternative_count = config.candidates - 1
        self.slot_embedding = nn.Parameter(
            torch.empty(1, alternative_count, config.d_model)
        )
        nn.init.trunc_normal_(self.slot_embedding, std=0.02)
        self.query_norm = nn.LayerNorm(config.d_model)
        self.memory_norm = nn.LayerNorm(config.d_model)
        self.memory_reader = nn.MultiheadAttention(
            config.d_model,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
            bias=False,
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(2 * config.d_model),
            nn.Linear(2 * config.d_model, config.hidden_width),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_width, config.d_model),
            nn.GELU(),
        )
        # This projection is shared across slots.  It is zero initialized so a
        # newly attached, untrained head cannot silently perturb the reference;
        # fixed-library coverage supervision breaks the initial slot collapse.
        self.residual_output = nn.Linear(
            config.d_model, config.action_horizon * config.action_dim
        )
        nn.init.zeros_(self.residual_output.weight)
        nn.init.zeros_(self.residual_output.bias)
        self.register_buffer(
            "normalized_residual_limit",
            torch.tensor(config.normalized_residual_limit).view(1, 1, 1, -1),
        )

    def forward(
        self,
        reference_normalized: torch.Tensor,
        local_memory: torch.Tensor,
        local_memory_mask: torch.Tensor,
    ) -> MARSCARELearnedProposalOutput:
        config = self.config
        if reference_normalized.ndim != 3 or tuple(reference_normalized.shape[1:]) != (
            config.action_horizon,
            config.action_dim,
        ):
            raise ValueError("reference chunk must be [batch,100,8]")
        if local_memory.ndim != 3 or local_memory.shape[-1] != config.d_model:
            raise ValueError("local proposal memory must be [batch,tokens,d_model]")
        if local_memory.shape[0] != reference_normalized.shape[0]:
            raise ValueError("reference and local-memory batch sizes differ")
        if (
            local_memory_mask.shape != local_memory.shape[:2]
            or local_memory_mask.dtype != torch.bool
        ):
            raise ValueError("local proposal memory mask must be boolean [batch,tokens]")
        if not bool(local_memory_mask.any(dim=1).all()):
            raise ValueError("every proposal row requires at least one legal memory token")
        if reference_normalized.device != local_memory.device:
            raise ValueError("reference and local proposal memory must share a device")
        if local_memory_mask.device != local_memory.device:
            raise ValueError("local proposal memory and mask must share a device")
        if (
            not reference_normalized.is_floating_point()
            or not local_memory.is_floating_point()
        ):
            raise ValueError("proposal reference and memory must be floating point")
        if not bool(torch.isfinite(reference_normalized).all()):
            raise ValueError("proposal reference contains a non-finite value")
        if not bool(torch.isfinite(local_memory).all()):
            raise ValueError("local proposal memory contains a non-finite value")

        slots = self.slot_embedding.to(local_memory).expand(
            local_memory.shape[0], -1, -1
        )
        normalized_memory = self.memory_norm(local_memory)
        attended = self.memory_reader(
            self.query_norm(slots),
            normalized_memory,
            normalized_memory,
            key_padding_mask=~local_memory_mask,
            need_weights=False,
        )[0]
        slot_state = self.fusion(torch.cat((slots, attended), dim=-1))
        raw_residual = self.residual_output(slot_state).view(
            local_memory.shape[0],
            config.candidates - 1,
            config.action_horizon,
            config.action_dim,
        )
        alternative_residuals = torch.tanh(raw_residual) * (
            self.normalized_residual_limit.to(raw_residual)
        )
        # Keep the external reference's dtype, and put it into slot zero by a
        # direct concat rather than ``reference + 0``.  This is the identity
        # anchor required for selector-off/fail-closed comparisons.
        alternative_residuals = alternative_residuals.to(reference_normalized)
        zero_residual = torch.zeros_like(reference_normalized).unsqueeze(1)
        residuals = torch.cat((zero_residual, alternative_residuals), dim=1)
        alternatives = reference_normalized.unsqueeze(1) + alternative_residuals
        candidates = torch.cat(
            (reference_normalized.unsqueeze(1), alternatives), dim=1
        )
        return MARSCARELearnedProposalOutput(
            candidates_normalized=candidates,
            residuals_normalized=residuals,
            slot_state=slot_state,
            reference_normalized=reference_normalized,
        )


@dataclass(frozen=True)
class MARSCAREProposalBootstrapLossConfig:
    """Weights for fixed-library initialization, never final CARE training."""

    coverage_weight: float = 1.0
    diversity_weight: float = 0.05
    coverage_beta: float = 0.02
    diversity_margin: float = 0.025

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in (self.coverage_weight, self.diversity_weight)
        ):
            raise ValueError("bootstrap loss weights must be finite and non-negative")
        if not math.isfinite(float(self.coverage_beta)) or self.coverage_beta <= 0.0:
            raise ValueError("bootstrap coverage beta must be finite and positive")
        if (
            not math.isfinite(float(self.diversity_margin))
            or self.diversity_margin < 0.0
        ):
            raise ValueError("bootstrap diversity margin must be finite and non-negative")

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def normalize_fixed_library_candidates(
    candidate_chunks: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> torch.Tensor:
    """Normalize existing physical fixed-library chunks with reference stats.

    This adapter is intentionally small: it makes the already prepared MARS
    ``candidate_chunks`` usable for proposal bootstrap without assigning their
    stale branch labels to the new learned candidates.
    """

    if candidate_chunks.ndim != 4 or tuple(candidate_chunks.shape[2:]) != (100, 8):
        raise ValueError("fixed-library chunks must be [batch,candidate,100,8]")
    mean = torch.as_tensor(
        action_mean, device=candidate_chunks.device, dtype=candidate_chunks.dtype
    ).reshape(-1)
    std = torch.as_tensor(
        action_std, device=candidate_chunks.device, dtype=candidate_chunks.dtype
    ).reshape(-1)
    if mean.shape != (8,) or std.shape != (8,):
        raise ValueError("fixed-library action statistics must each have width 8")
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
        raise ValueError("fixed-library action statistics must be finite")
    if bool((std <= 0).any()):
        raise ValueError("fixed-library action standard deviation must be positive")
    return (candidate_chunks - mean.view(1, 1, 1, 8)) / std.view(1, 1, 1, 8)


def mars_care_proposal_bootstrap_loss(
    output: MARSCARELearnedProposalOutput,
    fixed_library_candidates_normalized: torch.Tensor,
    *,
    config: MARSCAREProposalBootstrapLossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Initialize stable learned slots from the prepared fixed library.

    Coverage is candidate-ID aligned because the prepared library has stable
    slot semantics.  The diversity margin includes candidate zero, preventing
    all learned alternatives from merely reproducing the reference.  Neither
    term uses old reactive/replay outcomes; those outcomes become stale as soon
    as the candidate generator changes and must be recollected.
    """

    loss_config = config or MARSCAREProposalBootstrapLossConfig()
    candidates = output.candidates_normalized
    residuals = output.residuals_normalized
    fixed = fixed_library_candidates_normalized
    if candidates.ndim != 4 or tuple(candidates.shape[2:]) != (100, 8):
        raise ValueError("learned proposal output must be [batch,candidate,100,8]")
    if fixed.shape != candidates.shape:
        raise ValueError("fixed-library and learned candidate tensors must match")
    if residuals.shape != candidates.shape:
        raise ValueError("learned proposal residual tensor shape differs")
    if output.reference_normalized.shape != candidates[:, 0].shape:
        raise ValueError("learned proposal reference tensor shape differs")
    if not torch.equal(candidates[:, 0], output.reference_normalized):
        raise ValueError("learned proposal candidate zero lost its identity anchor")
    if not torch.equal(fixed[:, 0], output.reference_normalized):
        raise ValueError(
            "fixed-library candidate zero must equal the normalized reference"
        )
    if not bool(torch.isfinite(fixed).all()):
        raise ValueError("fixed-library bootstrap target contains a non-finite value")

    target_residuals = (fixed[:, 1:] - fixed[:, :1]).detach()
    coverage = F.smooth_l1_loss(
        residuals[:, 1:], target_residuals, beta=loss_config.coverage_beta
    )

    flat = residuals.flatten(2).float()
    pairwise_rms = (
        (flat[:, :, None] - flat[:, None, :]).square().mean(-1) + 1e-12
    ).sqrt()
    candidate_count = residuals.shape[1]
    upper = torch.triu(
        torch.ones(
            candidate_count,
            candidate_count,
            dtype=torch.bool,
            device=residuals.device,
        ),
        diagonal=1,
    )
    diversity = F.relu(
        float(loss_config.diversity_margin) - pairwise_rms[:, upper]
    ).square().mean()
    total = (
        float(loss_config.coverage_weight) * coverage
        + float(loss_config.diversity_weight) * diversity
    )
    return total, {
        "bootstrap_coverage": coverage.detach(),
        "diversity": diversity.detach(),
        "total": total.detach(),
    }


__all__ = [
    "FIXED_LIBRARY_LABEL_USAGE",
    "MARSCARELearnedProposalConfig",
    "MARSCARELearnedProposalHead",
    "MARSCARELearnedProposalOutput",
    "MARSCAREProposalBootstrapLossConfig",
    "mars_care_proposal_bootstrap_loss",
    "normalize_fixed_library_candidates",
]
