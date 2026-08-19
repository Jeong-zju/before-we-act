"""Distributional CARE action--response belief head and conservative selector.

The frozen CARE reference policy remains the only action generator. This module scores the
six fixed CARE candidates from legal reference belief/event tokens and candidate
chunks; it never emits a continuous action residual.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F


CARE_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
CARE_HORIZONS = (8, 16, 32, 64)


@dataclass(frozen=True)
class CAREBeliefConfig:
    d_model: int = 384
    action_dim: int = 8
    action_horizon: int = 100
    action_tokens: int = 16
    action_width: int = 128
    heads: int = 8
    layers: int = 2
    dropout: float = 0.1
    candidates: int = 6
    outcome_components: int = 3
    quantiles: tuple[float, ...] = CARE_QUANTILES
    horizons: tuple[int, ...] = CARE_HORIZONS
    variant: str = "care"
    action_std: tuple[float, ...] = (1.0,) * 8

    def __post_init__(self) -> None:
        if self.variant not in {"care", "reactive_only", "replay_only", "capacity"}:
            raise ValueError(f"unsupported CARE scorer variant: {self.variant}")
        if self.d_model % self.heads:
            raise ValueError("CARE width must be divisible by attention heads")
        if len(self.action_std) != self.action_dim or min(self.action_std) <= 0:
            raise ValueError("CARE action scale must be positive and action-dimensional")
        if tuple(sorted(self.quantiles)) != tuple(self.quantiles):
            raise ValueError("CARE quantiles must be sorted")
        if 0.5 not in self.quantiles:
            raise ValueError("CARE quantiles must include the median")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CAREBeliefConfig":
        row = dict(value)
        for key in ("quantiles", "horizons", "action_std"):
            if key in row:
                row[key] = tuple(row[key])
        return cls(**row)


@dataclass
class CAREBeliefOutput:
    quantiles: torch.Tensor
    hard_safety_logit: torch.Tensor
    candidate_state: torch.Tensor

    @property
    def direct(self) -> torch.Tensor:
        return self.quantiles[:, :, 0]

    @property
    def response(self) -> torch.Tensor:
        return self.quantiles[:, :, 1]

    @property
    def total(self) -> torch.Tensor:
        return self.quantiles[:, :, 2]


class CandidateActionEncoder(nn.Module):
    """Encode a candidate relative to the frozen reference action chunk."""

    def __init__(self, config: CAREBeliefConfig) -> None:
        super().__init__()
        self.config = config
        indices = torch.linspace(
            0, config.action_horizon - 1, config.action_tokens
        ).round().to(torch.long)
        self.register_buffer("sample_indices", indices)
        self.register_buffer(
            "action_std", torch.tensor(config.action_std).view(1, 1, 1, -1)
        )
        self.input = nn.Linear(config.action_dim, config.action_width)
        self.position = nn.Parameter(
            torch.randn(1, 1, config.action_tokens, config.action_width) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            config.action_width,
            4,
            config.action_width * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.summary = nn.Sequential(
            nn.LayerNorm(config.action_width),
            nn.Linear(config.action_width, config.d_model),
        )

    def forward(self, candidate_chunks: torch.Tensor) -> torch.Tensor:
        expected = (
            candidate_chunks.shape[0],
            self.config.candidates,
            self.config.action_horizon,
            self.config.action_dim,
        )
        if tuple(candidate_chunks.shape) != expected:
            raise ValueError(f"CARE candidate tensor differs: {candidate_chunks.shape}")
        delta = candidate_chunks - candidate_chunks[:, :1]
        sampled = delta.index_select(2, self.sample_indices)
        sampled = sampled / self.action_std.to(sampled)
        token = self.input(sampled) + self.position.to(sampled)
        batch, candidates, steps, width = token.shape
        token = self.temporal(token.reshape(batch * candidates, steps, width))
        token = token.mean(1).reshape(batch, candidates, width)
        return self.summary(token)


class CAREBeliefHead(nn.Module):
    """Cross-attent candidate queries into frozen belief/event memory."""

    def __init__(self, config: CAREBeliefConfig) -> None:
        super().__init__()
        self.config = config
        self.action_encoder = CandidateActionEncoder(config)
        self.horizon_embedding = nn.Embedding(len(config.horizons), config.d_model)
        self.query_norm = nn.LayerNorm(config.d_model)
        self.memory_norm = nn.LayerNorm(config.d_model)
        self.cross_attention = nn.MultiheadAttention(
            config.d_model,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.capacity_memory = nn.Sequential(
            nn.LayerNorm(config.d_model), nn.Linear(config.d_model, config.d_model)
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(2 * config.d_model),
            nn.Linear(2 * config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
        )
        self.advantage = nn.Linear(
            config.d_model,
            config.outcome_components * len(config.quantiles),
        )
        self.hard_safety = nn.Linear(config.d_model, 1)

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        candidate_chunks: torch.Tensor,
        horizon_index: torch.Tensor,
    ) -> CAREBeliefOutput:
        if memory.ndim != 3 or memory.shape[-1] != self.config.d_model:
            raise ValueError("CARE memory must be [batch,tokens,d_model]")
        if memory_mask.shape != memory.shape[:2] or memory_mask.dtype != torch.bool:
            raise ValueError("CARE memory mask differs")
        if horizon_index.shape != (memory.shape[0],):
            raise ValueError("CARE horizon index must be [batch]")
        query = self.action_encoder(candidate_chunks)
        query = query + self.horizon_embedding(horizon_index).unsqueeze(1)
        if self.config.variant == "capacity":
            weights = memory_mask.unsqueeze(-1).to(memory.dtype)
            pooled = (memory * weights).sum(1) / weights.sum(1).clamp_min(1)
            attended = self.capacity_memory(pooled).unsqueeze(1).expand_as(query)
        else:
            attended = self.cross_attention(
                self.query_norm(query),
                self.memory_norm(memory),
                self.memory_norm(memory),
                key_padding_mask=~memory_mask,
                need_weights=False,
            )[0]
        state = self.fusion(torch.cat((query, attended), dim=-1))
        raw = self.advantage(state).view(
            memory.shape[0],
            self.config.candidates,
            self.config.outcome_components,
            len(self.config.quantiles),
        )
        # Quantile crossing is prevented structurally at inference.  The
        # training loss also contains a crossing penalty so sorting is rarely
        # active after convergence.
        quantiles = raw.sort(-1).values
        safety = self.hard_safety(state).squeeze(-1)
        zero = torch.zeros_like(quantiles[:, :1])
        quantiles = torch.cat((zero, quantiles[:, 1:]), dim=1)
        safe_reference = torch.full_like(safety[:, :1], -20.0)
        safety = torch.cat((safe_reference, safety[:, 1:]), dim=1)
        return CAREBeliefOutput(quantiles, safety, state)


def pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantiles: Sequence[float] = CARE_QUANTILES,
) -> torch.Tensor:
    if prediction.shape[:-1] != target.shape:
        raise ValueError("pinball prediction/target shape differs")
    q = prediction.new_tensor(tuple(float(value) for value in quantiles))
    error = target.unsqueeze(-1) - prediction
    return torch.maximum(q * error, (q - 1.0) * error).mean()


def pairwise_ranking_loss(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Bounded pairwise logistic loss over the five non-reference candidates."""

    if scores.shape != targets.shape or scores.ndim != 2:
        raise ValueError("CARE ranking tensors must be matching [batch,candidate]")
    score_delta = scores[:, :, None] - scores[:, None, :]
    target_delta = targets[:, :, None] - targets[:, None, :]
    mask = target_delta.abs() > 1e-6
    if not mask.any():
        return scores.sum() * 0.0
    sign = target_delta.sign()
    return F.softplus(-sign[mask] * score_delta[mask]).mean()


def care_training_loss(
    output: CAREBeliefOutput,
    target: torch.Tensor,
    hard_safety: torch.Tensor,
    variant: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the registered loss and its auditable components.

    ``target[...,0:3]`` stores replay/direct advantage, response interaction,
    and reactive/total advantage respectively.  Candidate zero is omitted
    because its relative advantage is exactly zero by definition.
    """

    prediction = output.quantiles[:, 1:]
    target = target[:, 1:]
    safety_target = hard_safety[:, 1:].float()
    median = output.quantiles.shape[-1] // 2
    if variant == "care":
        primary = pinball_loss(prediction, target)
        consistency = (prediction[:, :, 2] - prediction[:, :, 0] - prediction[:, :, 1]).abs().mean()
        rank_scores = prediction[:, :, 2, median]
        rank_targets = target[:, :, 2]
    elif variant == "replay_only":
        primary = pinball_loss(prediction[:, :, 0], target[:, :, 0])
        consistency = primary * 0.0
        rank_scores = prediction[:, :, 0, median]
        rank_targets = target[:, :, 0]
    else:
        primary = pinball_loss(prediction[:, :, 2], target[:, :, 2])
        consistency = primary * 0.0
        rank_scores = prediction[:, :, 2, median]
        rank_targets = target[:, :, 2]
    ranking = pairwise_ranking_loss(rank_scores, rank_targets)
    safety = F.binary_cross_entropy_with_logits(
        output.hard_safety_logit[:, 1:], safety_target
    )
    total = primary + 0.20 * consistency + 0.10 * ranking + 0.10 * safety
    return total, {
        "pinball": primary.detach(),
        "did_consistency": consistency.detach(),
        "ranking": ranking.detach(),
        "hard_safety": safety.detach(),
    }


@dataclass(frozen=True)
class CARECalibration:
    lower_correction: float
    selector_delta: float
    hard_safety_probability_max: float
    nominal_simultaneous_coverage: float
    primary_horizon: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CARECalibration":
        return cls(**value)


def select_care_candidate(
    output: CAREBeliefOutput,
    calibration: CARECalibration,
    *,
    variant: str = "care",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select one candidate or fail closed to reference candidate zero."""

    component = 0 if variant == "replay_only" else 2
    lower = output.quantiles[:, :, component, 0] - calibration.lower_correction
    unsafe = output.hard_safety_logit.sigmoid() > calibration.hard_safety_probability_max
    lower = lower.masked_fill(unsafe, -torch.inf)
    lower[:, 0] = 0.0
    best_lower, best = lower.max(1)
    best = torch.where(
        best_lower > calibration.selector_delta,
        best,
        torch.zeros_like(best),
    )
    return best, best_lower, unsafe


__all__ = [
    "CAREBeliefConfig",
    "CAREBeliefHead",
    "CAREBeliefOutput",
    "CARECalibration",
    "CARE_HORIZONS",
    "CARE_QUANTILES",
    "care_training_loss",
    "pairwise_ranking_loss",
    "pinball_loss",
    "select_care_candidate",
]
