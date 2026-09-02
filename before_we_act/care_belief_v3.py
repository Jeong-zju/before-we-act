"""Protocol-isolated CARE scorer-v3 conditioning ablation.

This module preserves CARE's finite candidate interface, paired
direct/response/total targets, distributional quantiles, structural candidate
zero, and calibrated selector.  It only makes three pieces of already-declared
metadata explicit to the scorer:

* a stable embedding for each of the six candidate slots;
* the benchmark task identity that is already part of every policy input; and
* a robust target unit for each task *and outcome horizon*.

The v1/v2 modules and checkpoints are intentionally untouched.  A v3 model is
diagnostic-only until it passes the same family-disjoint OOF and closed-loop
admission gates as v2.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn

from before_we_act.care_belief import CAREBeliefOutput
from before_we_act.care_belief_v2 import (
    CAREBeliefV2Config,
    CAREBeliefV2Head,
)


@dataclass(frozen=True)
class CAREBeliefV3Config(CAREBeliefV2Config):
    """CARE-v2 plus explicit, ablatable candidate/task conditioning."""

    task_count: int = 4
    use_candidate_slot_embedding: bool = True
    use_task_embedding: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if int(self.task_count) < 1:
            raise ValueError("CARE v3 task count must be positive")
        if not isinstance(self.use_candidate_slot_embedding, bool):
            raise ValueError("CARE v3 candidate-slot flag must be boolean")
        if not isinstance(self.use_task_embedding, bool):
            raise ValueError("CARE v3 task-embedding flag must be boolean")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CAREBeliefV3Config":
        row = dict(value)
        for key in ("quantiles", "horizons", "action_std"):
            if key in row:
                row[key] = tuple(row[key])
        return cls(**row)


class CAREBeliefV3Head(CAREBeliefV2Head):
    """CARE scorer with stable candidate identity and task conditioning.

    The two embeddings modify only the candidate query.  The inherited action
    encoder, memory cross-attention, fusion block, D/R/A quantile heads, and
    hard-safety head retain their original roles and dimensions.
    """

    config: CAREBeliefV3Config

    def __init__(self, config: CAREBeliefV3Config) -> None:
        super().__init__(config)
        self.candidate_slot_embedding = (
            nn.Embedding(config.candidates, config.d_model)
            if config.use_candidate_slot_embedding
            else None
        )
        self.task_embedding = (
            nn.Embedding(config.task_count, config.d_model)
            if config.use_task_embedding
            else None
        )

    def _task_id(self, task_id: torch.Tensor, batch: int) -> torch.Tensor:
        if task_id.shape != (batch,) or task_id.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("CARE v3 task id must be an integer [batch] tensor")
        canonical = task_id.to(dtype=torch.long)
        if bool((canonical < 0).any()) or bool(
            (canonical >= int(self.config.task_count)).any()
        ):
            raise ValueError("CARE v3 task id is out of range")
        return canonical

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        candidate_chunks: torch.Tensor,
        horizon_index: torch.Tensor,
        task_id: torch.Tensor,
        *,
        utility_scale: torch.Tensor | None = None,
    ) -> CAREBeliefOutput:
        if memory.ndim != 3 or memory.shape[-1] != self.config.d_model:
            raise ValueError("CARE memory must be [batch,tokens,d_model]")
        if memory_mask.shape != memory.shape[:2] or memory_mask.dtype != torch.bool:
            raise ValueError("CARE memory mask differs")
        if horizon_index.shape != (memory.shape[0],):
            raise ValueError("CARE horizon index must be [batch]")
        task_id = self._task_id(task_id, memory.shape[0]).to(memory.device)

        query = self.action_encoder(candidate_chunks)
        query = query + self.horizon_embedding(horizon_index).unsqueeze(1)
        if self.candidate_slot_embedding is not None:
            slots = torch.arange(
                self.config.candidates, dtype=torch.long, device=query.device
            )
            query = query + self.candidate_slot_embedding(slots).unsqueeze(0)
        if self.task_embedding is not None:
            query = query + self.task_embedding(task_id).unsqueeze(1)

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
        quantiles = raw.sort(-1).values
        safety = self.hard_safety(state).squeeze(-1)

        # Candidate zero remains the bit-exact B-core/TUNE reference regardless
        # of its query embedding.  Only non-reference candidates are learned.
        quantiles = torch.cat(
            (torch.zeros_like(quantiles[:, :1]), quantiles[:, 1:]), dim=1
        )
        safety = torch.cat(
            (torch.full_like(safety[:, :1], -20.0), safety[:, 1:]), dim=1
        )
        if utility_scale is not None:
            expected = (memory.shape[0], self.config.outcome_components)
            if tuple(utility_scale.shape) != expected:
                raise ValueError(
                    "CARE v3 utility scale must be [batch,outcome_component]"
                )
            scale = utility_scale.to(device=quantiles.device, dtype=quantiles.dtype)
            if not torch.isfinite(scale).all() or bool((scale <= 0).any()):
                raise ValueError("CARE v3 utility scale must be finite and positive")
            physical = quantiles[:, 1:] * scale[:, None, :, None]
            quantiles = torch.cat((torch.zeros_like(quantiles[:, :1]), physical), dim=1)
        return CAREBeliefOutput(
            quantiles=quantiles,
            hard_safety_logit=safety,
            candidate_state=state,
        )


def robust_task_horizon_component_scales(
    targets: torch.Tensor,
    usable: torch.Tensor,
    task_id: torch.Tensor,
    *,
    quantile: float = 0.90,
    floor: float = 1e-4,
) -> torch.Tensor:
    """Return fixed robust units as ``[task,horizon,component]``.

    Candidate zero is excluded because it is a structural zero.  Unlike v2's
    task/component scale, no long-horizon outlier can define the numerical unit
    used by a short-horizon training row (or vice versa).
    """

    if targets.ndim != 5:
        raise ValueError(
            "CARE v3 prepared targets must be [family,horizon,candidate,repeat,component]"
        )
    if usable.shape != targets.shape[:2] or task_id.shape != (targets.shape[0],):
        raise ValueError("CARE v3 prepared target metadata shape differs")
    if not 0.0 < float(quantile) <= 1.0 or not float(floor) > 0.0:
        raise ValueError("CARE v3 robust scale quantile/floor is invalid")
    if task_id.numel() == 0 or bool((task_id < 0).any()):
        raise ValueError("CARE v3 task ids must be non-empty and non-negative")

    components = targets.shape[-1]
    tasks: list[torch.Tensor] = []
    for current_task in range(int(task_id.max()) + 1):
        family_mask = task_id == current_task
        horizons: list[torch.Tensor] = []
        for horizon in range(targets.shape[1]):
            selected = family_mask & usable[:, horizon]
            if not bool(selected.any()):
                raise ValueError(
                    f"CARE v3 task {current_task}/horizon {horizon} has no usable targets"
                )
            value = targets[selected, horizon, 1:].reshape(-1, components).abs()
            scale = torch.quantile(
                value.float(), float(quantile), dim=0
            ).clamp_min(float(floor))
            horizons.append(scale)
        tasks.append(torch.stack(horizons))
    return torch.stack(tasks)


__all__ = [
    "CAREBeliefV3Config",
    "CAREBeliefV3Head",
    "robust_task_horizon_component_scales",
]
