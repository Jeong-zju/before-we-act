"""Protocol-isolated CARE scorer-v2 components.

The v2 scorer preserves CARE's distributional candidate-belief framework and
the frozen reference candidate.  It fixes three training/runtime alignment
issues without changing the legacy scorer or its checkpoint contract:

* candidate features cover only the intervention prefix that is executed;
* ranking explicitly compares every non-reference candidate with candidate 0;
* utility targets are optimized in fixed robust units while predictions remain
  in the original physical utility units used by calibration and selection.

Nothing in this module is imported by the frozen formal MARS pipeline.  A new
run must opt into the v2 checkpoint format explicitly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from before_we_act.care_belief import (
    CAREBeliefConfig,
    CAREBeliefHead,
    CAREBeliefOutput,
    CARE_QUANTILES,
    pinball_loss,
)


@dataclass(frozen=True)
class CAREBeliefV2Config(CAREBeliefConfig):
    """CARE scorer configuration with an explicit executed-action prefix."""

    action_prefix_steps: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 1 <= int(self.action_prefix_steps) <= int(self.action_horizon):
            raise ValueError("CARE v2 action prefix must lie within the action horizon")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CAREBeliefV2Config":
        row = dict(value)
        for key in ("quantiles", "horizons", "action_std"):
            if key in row:
                row[key] = tuple(row[key])
        return cls(**row)


class PrefixCandidateActionEncoder(nn.Module):
    """Encode only candidate actions covered by the intervention contract.

    For the current MARS one-step branch corpus, ``action_prefix_steps=1``
    makes the representation invariant to all 99 unexecuted tail actions.  A
    future 4/8/16-step ablation can use the matching prefix without changing
    the surrounding CARE belief or selector interfaces.
    """

    def __init__(self, config: CAREBeliefV2Config) -> None:
        super().__init__()
        self.config = config
        token_count = min(int(config.action_tokens), int(config.action_prefix_steps))
        indices = torch.linspace(
            0, int(config.action_prefix_steps) - 1, token_count
        ).round().to(torch.long)
        self.register_buffer("sample_indices", indices)
        self.register_buffer(
            "action_std", torch.tensor(config.action_std).view(1, 1, 1, -1)
        )
        self.input = nn.Linear(config.action_dim, config.action_width)
        self.position = nn.Parameter(
            torch.randn(1, 1, token_count, config.action_width) * 0.02
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
            raise ValueError(
                f"CARE v2 candidate tensor differs: {candidate_chunks.shape}"
            )
        delta = candidate_chunks - candidate_chunks[:, :1]
        sampled = delta.index_select(2, self.sample_indices)
        sampled = sampled / self.action_std.to(sampled)
        token = self.input(sampled) + self.position.to(sampled)
        batch, candidates, steps, width = token.shape
        token = self.temporal(token.reshape(batch * candidates, steps, width))
        token = token.mean(1).reshape(batch, candidates, width)
        return self.summary(token)


class CAREBeliefV2Head(CAREBeliefHead):
    """Legacy-compatible CARE belief head with prefix-matched action queries.

    ``utility_scale`` is an optional training/runtime unit contract.  The
    inherited head predicts dimensionless normalized quantiles; multiplying
    non-reference candidates by a positive per-row component scale exposes
    physical utility units to the selector.  When the loss divides by the same
    scale, gradients with respect to the inherited raw head stay normalized
    instead of growing like ``1 / utility_scale``.

    Omitting ``utility_scale`` preserves the original v2 API exactly.
    """

    config: CAREBeliefV2Config

    def __init__(self, config: CAREBeliefV2Config) -> None:
        super().__init__(config)
        # All downstream layers and the output/selector contract are unchanged.
        # Only the candidate query is rebuilt with the executed prefix.
        self.action_encoder = PrefixCandidateActionEncoder(config)

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        candidate_chunks: torch.Tensor,
        horizon_index: torch.Tensor,
        *,
        utility_scale: torch.Tensor | None = None,
    ) -> CAREBeliefOutput:
        output = super().forward(
            memory, memory_mask, candidate_chunks, horizon_index
        )
        if utility_scale is None:
            return output
        expected = (memory.shape[0], self.config.outcome_components)
        if tuple(utility_scale.shape) != expected:
            raise ValueError(
                "CARE v2 utility scale must be [batch,outcome_component]"
            )
        scale = utility_scale.to(
            device=output.quantiles.device, dtype=output.quantiles.dtype
        )
        if not torch.isfinite(scale).all() or bool((scale <= 0).any()):
            raise ValueError("CARE v2 utility scale must be finite and positive")
        physical = output.quantiles[:, 1:] * scale[:, None, :, None]
        quantiles = torch.cat((torch.zeros_like(output.quantiles[:, :1]), physical), dim=1)
        return CAREBeliefOutput(
            quantiles=quantiles,
            hard_safety_logit=output.hard_safety_logit,
            candidate_state=output.candidate_state,
        )


@dataclass(frozen=True)
class CARELossV2Config:
    """Auditable weights for the protocol-isolated scorer-v2 objective."""

    consistency_weight: float = 0.20
    candidate_ranking_weight: float = 0.10
    reference_ranking_weight: float = 0.10
    safety_weight: float = 0.10
    ranking_min_gap: float = 1e-3

    def __post_init__(self) -> None:
        values = (
            self.consistency_weight,
            self.candidate_ranking_weight,
            self.reference_ranking_weight,
            self.safety_weight,
            self.ranking_min_gap,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
            raise ValueError("CARE v2 loss weights and ranking gap must be non-negative")

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def canonical_target_scale(
    target: torch.Tensor, target_scale: float | Sequence[float] | torch.Tensor
) -> torch.Tensor:
    """Return positive component scales as ``[batch, component]``.

    Scaling is applied inside the loss, so the model output remains in the
    original utility units and needs no inference-time decode.
    """

    if target.ndim != 3:
        raise ValueError("CARE v2 target must be [batch,candidate,component]")
    batch, _candidates, components = target.shape
    scale = torch.as_tensor(target_scale, device=target.device, dtype=target.dtype)
    if scale.ndim == 0:
        scale = scale.expand(batch, components)
    elif tuple(scale.shape) == (components,):
        scale = scale.unsqueeze(0).expand(batch, components)
    elif tuple(scale.shape) != (batch, components):
        raise ValueError(
            "CARE v2 target scale must be scalar, [component], or [batch,component]"
        )
    if not torch.isfinite(scale).all() or bool((scale <= 0).any()):
        raise ValueError("CARE v2 target scale must be finite and positive")
    return scale


def scaled_pairwise_ranking_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    scale: torch.Tensor,
    *,
    minimum_gap: float,
) -> torch.Tensor:
    """Pairwise logistic ranking in dimensionless robust utility units."""

    if scores.shape != targets.shape or scores.ndim != 2:
        raise ValueError("CARE v2 ranking tensors must be matching [batch,candidate]")
    if scale.shape != (scores.shape[0],):
        raise ValueError("CARE v2 ranking scale must be [batch]")
    normalized_scores = scores / scale[:, None]
    normalized_targets = targets / scale[:, None]
    score_delta = normalized_scores[:, :, None] - normalized_scores[:, None, :]
    target_delta = normalized_targets[:, :, None] - normalized_targets[:, None, :]
    upper = torch.triu(
        torch.ones(
            scores.shape[1], scores.shape[1], dtype=torch.bool, device=scores.device
        ),
        diagonal=1,
    )
    mask = (target_delta.abs() > float(minimum_gap)) & upper.unsqueeze(0)
    if not bool(mask.any()):
        return scores.sum() * 0.0
    sign = target_delta.sign()
    return F.softplus(-sign[mask] * score_delta[mask]).mean()


def reference_ranking_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    scale: torch.Tensor,
    *,
    minimum_gap: float,
) -> torch.Tensor:
    """Train the exact decision CARE makes: candidate versus reference zero."""

    if scores.shape != targets.shape or scores.ndim != 2:
        raise ValueError(
            "CARE v2 reference ranking tensors must be matching [batch,candidate]"
        )
    if scale.shape != (scores.shape[0],):
        raise ValueError("CARE v2 reference ranking scale must be [batch]")
    score_delta = (scores[:, 1:] - scores[:, :1]) / scale[:, None]
    target_delta = (targets[:, 1:] - targets[:, :1]) / scale[:, None]
    mask = target_delta.abs() > float(minimum_gap)
    if not bool(mask.any()):
        return scores.sum() * 0.0
    return F.softplus(-target_delta.sign()[mask] * score_delta[mask]).mean()


def care_v2_training_loss(
    output: CAREBeliefOutput,
    target: torch.Tensor,
    hard_safety: torch.Tensor,
    variant: str,
    *,
    target_scale: float | Sequence[float] | torch.Tensor,
    loss_config: CARELossV2Config | None = None,
    quantiles: Sequence[float] = CARE_QUANTILES,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """CARE loss with prefix-independent robust units and reference ranking.

    Candidate zero remains fixed to exactly zero and is still omitted from the
    pinball regression.  It is deliberately included in the new reference
    ranking term because the deployed selector must decide whether a nonzero
    candidate is better than that reference.
    """

    config = loss_config or CARELossV2Config()
    if output.quantiles.ndim != 4 or target.ndim != 3:
        raise ValueError("CARE v2 output/target rank differs")
    if output.quantiles.shape[:3] != target.shape:
        raise ValueError("CARE v2 output/target shape differs")
    if hard_safety.shape != target.shape[:2]:
        raise ValueError("CARE v2 safety target shape differs")
    if variant not in {"care", "reactive_only", "replay_only", "capacity"}:
        raise ValueError(f"unsupported CARE v2 scorer variant: {variant}")

    scale = canonical_target_scale(target, target_scale)
    prediction = output.quantiles[:, 1:]
    nonreference_target = target[:, 1:]
    normalized_prediction = prediction / scale[:, None, :, None]
    normalized_target = nonreference_target / scale[:, None, :]
    median = tuple(float(value) for value in quantiles).index(0.5)

    if variant == "care":
        primary = pinball_loss(normalized_prediction, normalized_target, quantiles)
        consistency = (
            prediction[:, :, 2]
            - prediction[:, :, 0]
            - prediction[:, :, 1]
        ).abs().div(scale[:, None, 2, None]).mean()
        component = 2
    elif variant == "replay_only":
        primary = pinball_loss(
            normalized_prediction[:, :, 0], normalized_target[:, :, 0], quantiles
        )
        consistency = primary * 0.0
        component = 0
    else:
        primary = pinball_loss(
            normalized_prediction[:, :, 2], normalized_target[:, :, 2], quantiles
        )
        consistency = primary * 0.0
        component = 2

    rank_scores = output.quantiles[:, :, component, median]
    rank_targets = target[:, :, component]
    rank_scale = scale[:, component]
    candidate_ranking = scaled_pairwise_ranking_loss(
        rank_scores[:, 1:],
        rank_targets[:, 1:],
        rank_scale,
        minimum_gap=config.ranking_min_gap,
    )
    reference_ranking = reference_ranking_loss(
        rank_scores,
        rank_targets,
        rank_scale,
        minimum_gap=config.ranking_min_gap,
    )
    if config.safety_weight > 0.0:
        safety = F.binary_cross_entropy_with_logits(
            output.hard_safety_logit[:, 1:], hard_safety[:, 1:].float()
        )
    else:
        # A one-class all-safe corpus provides no supervision for detecting
        # violations.  Its BCE would still backpropagate through the shared
        # candidate state and can overwhelm the small utility signal, so a
        # protocol that records effective safety_weight=0 bypasses it exactly.
        safety = output.hard_safety_logit[:, 1:].sum() * 0.0
    total = (
        primary
        + config.consistency_weight * consistency
        + config.candidate_ranking_weight * candidate_ranking
        + config.reference_ranking_weight * reference_ranking
        + config.safety_weight * safety
    )
    return total, {
        "pinball_scaled": primary.detach(),
        "did_consistency_scaled": consistency.detach(),
        "candidate_ranking_scaled": candidate_ranking.detach(),
        "reference_ranking_scaled": reference_ranking.detach(),
        "hard_safety": safety.detach(),
    }


def robust_task_component_scales(
    targets: torch.Tensor,
    usable: torch.Tensor,
    task_id: torch.Tensor,
    *,
    quantile: float = 0.90,
    floor: float = 1e-4,
) -> torch.Tensor:
    """Estimate fixed per-task/component units from all usable branch labels."""

    if targets.ndim != 5:
        raise ValueError(
            "CARE v2 prepared targets must be [family,horizon,candidate,repeat,component]"
        )
    if usable.shape != targets.shape[:2] or task_id.shape != (targets.shape[0],):
        raise ValueError("CARE v2 prepared target metadata shape differs")
    if not 0.0 < float(quantile) <= 1.0 or not float(floor) > 0.0:
        raise ValueError("CARE v2 robust scale quantile/floor is invalid")
    if task_id.numel() == 0 or bool((task_id < 0).any()):
        raise ValueError("CARE v2 task ids must be non-empty and non-negative")

    components = targets.shape[-1]
    rows: list[torch.Tensor] = []
    for current_task in range(int(task_id.max()) + 1):
        family_mask = task_id == current_task
        values: list[torch.Tensor] = []
        for horizon in range(targets.shape[1]):
            selected = family_mask & usable[:, horizon]
            if bool(selected.any()):
                # Candidate zero is a structural zero and must not define the
                # numerical unit for non-reference utility regression.
                values.append(
                    targets[selected, horizon, 1:].reshape(-1, components).abs()
                )
        if not values:
            raise ValueError(f"CARE v2 task {current_task} has no usable targets")
        value = torch.cat(values, dim=0).float()
        scale = torch.quantile(value, float(quantile), dim=0).clamp_min(float(floor))
        rows.append(scale)
    return torch.stack(rows)


__all__ = [
    "CAREBeliefV2Config",
    "CAREBeliefV2Head",
    "CARELossV2Config",
    "PrefixCandidateActionEncoder",
    "canonical_target_scale",
    "care_v2_training_loss",
    "reference_ranking_loss",
    "robust_task_component_scales",
    "scaled_pairwise_ranking_loss",
]
