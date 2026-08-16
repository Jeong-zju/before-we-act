"""Compact raw team-signal representation and matched action probes."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F

from before_we_act.raw_team_signal_data import (
    ACTION_PROBE_HORIZON,
    CAPACITY_CANDIDATES,
    FUTURE_OFFSETS,
)


@dataclass
class SignalCapacityOutput:
    tokens: torch.Tensor
    future_visual: torch.Tensor
    teammate_qpos: torch.Tensor
    teammate_delta: torch.Tensor


@dataclass
class TeamSignalOutput:
    history: torch.Tensor
    history_summary: torch.Tensor
    task_embedding: torch.Tensor
    capacities: Mapping[int, SignalCapacityOutput]


class RawTeamSignalEncoder(nn.Module):
    """Causal legal-history encoder with nested pre-registered token capacities."""

    def __init__(
        self,
        *,
        d_model: int = 384,
        temporal_layers: int = 2,
        heads: int = 8,
        capacities: tuple[int, ...] = CAPACITY_CANDIDATES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.temporal_layers = int(temporal_layers)
        self.heads = int(heads)
        self.capacities_n = tuple(int(value) for value in capacities)
        if self.capacities_n != CAPACITY_CANDIDATES:
            raise ValueError("3-N1 capacities are frozen to [4, 8, 16]")
        self.visual_projection = nn.Linear(768, d_model)
        self.visual_pair = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )
        self.qpos_projection = nn.Linear(9, d_model, bias=False)
        self.action_projection = nn.Linear(8, d_model, bias=False)
        self.task_embedding = nn.Embedding(6, d_model)
        self.position = nn.Parameter(torch.randn(1, 16, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model,
            heads,
            4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=temporal_layers)
        self.history_norm = nn.LayerNorm(d_model)
        self.queries = nn.ParameterDict()
        self.readers = nn.ModuleDict()
        self.token_norms = nn.ModuleDict()
        self.visual_heads = nn.ModuleDict()
        self.qpos_heads = nn.ModuleDict()
        self.delta_heads = nn.ModuleDict()
        for capacity in self.capacities_n:
            key = str(capacity)
            self.queries[key] = nn.Parameter(torch.randn(capacity, d_model) * 0.02)
            self.readers[key] = nn.MultiheadAttention(
                d_model, heads, dropout=dropout, batch_first=True
            )
            self.token_norms[key] = nn.LayerNorm(d_model)
            self.visual_heads[key] = nn.Linear(
                d_model, len(FUTURE_OFFSETS) * 2 * 768
            )
            self.qpos_heads[key] = nn.Linear(d_model, 9)
            self.delta_heads[key] = nn.Linear(
                d_model, len(FUTURE_OFFSETS) * 9
            )

    def encode_history(
        self,
        history_visual: torch.Tensor,
        history_qpos: torch.Tensor,
        history_action: torch.Tensor,
        history_mask: torch.Tensor,
        action_history_mask: torch.Tensor,
        task_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if history_visual.shape[1:] != (16, 2, 768):
            raise ValueError("3-N1 visual history contract differs")
        if history_qpos.shape[1:] != (16, 9) or history_action.shape[1:] != (16, 8):
            raise ValueError("3-N1 proprioceptive history contract differs")
        if history_mask.shape != history_qpos.shape[:2]:
            raise ValueError("3-N1 history mask differs")
        if not torch.all(history_mask[:, -1]):
            raise ValueError("current 3-N1 observation must be valid")
        visual = self.visual_projection(history_visual)
        visual = self.visual_pair(torch.cat((visual[:, :, 0], visual[:, :, 1]), dim=-1))
        observation_weight = history_mask.unsqueeze(-1).to(visual.dtype)
        action_weight = action_history_mask.unsqueeze(-1).to(visual.dtype)
        task = self.task_embedding(task_index)
        token = (
            visual * observation_weight
            + self.qpos_projection(history_qpos) * observation_weight
            + self.action_projection(history_action) * action_weight
            + self.position.to(visual.dtype)
            + task.unsqueeze(1)
        )
        valid = history_mask | action_history_mask
        encoded = self.temporal(token, src_key_padding_mask=~valid)
        encoded = self.history_norm(encoded) * valid.unsqueeze(-1).to(encoded.dtype)
        summary = encoded.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return encoded, summary, task

    def forward(
        self,
        history_visual: torch.Tensor,
        history_qpos: torch.Tensor,
        history_action: torch.Tensor,
        history_mask: torch.Tensor,
        action_history_mask: torch.Tensor,
        task_index: torch.Tensor,
    ) -> TeamSignalOutput:
        history, history_summary, task = self.encode_history(
            history_visual,
            history_qpos,
            history_action,
            history_mask,
            action_history_mask,
            task_index,
        )
        valid = history_mask | action_history_mask
        outputs: dict[int, SignalCapacityOutput] = {}
        for capacity in self.capacities_n:
            key = str(capacity)
            query = self.queries[key].unsqueeze(0).expand(history.shape[0], -1, -1)
            attended = self.readers[key](
                query,
                history,
                history,
                key_padding_mask=~valid,
                need_weights=False,
            )[0]
            tokens = self.token_norms[key](query + attended)
            pooled = tokens.mean(1)
            outputs[capacity] = SignalCapacityOutput(
                tokens=tokens,
                future_visual=self.visual_heads[key](pooled).view(
                    history.shape[0], len(FUTURE_OFFSETS), 2, 768
                ),
                teammate_qpos=self.qpos_heads[key](pooled),
                teammate_delta=self.delta_heads[key](pooled).view(
                    history.shape[0], len(FUTURE_OFFSETS), 9
                ),
            )
        return TeamSignalOutput(history, history_summary, task, outputs)

    def config_dict(self) -> dict:
        return {
            "d_model": self.d_model,
            "temporal_layers": self.temporal_layers,
            "heads": self.heads,
            "capacities": list(self.capacities_n),
        }


def masked_mse(value: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared = (value - target).square()
    weight = mask.to(squared.dtype)
    while weight.ndim < squared.ndim:
        weight = weight.unsqueeze(-1)
    expanded = weight.expand_as(squared)
    return (squared * expanded).sum() / expanded.sum().clamp_min(1)


def representation_losses(
    output: TeamSignalOutput,
    batch: Mapping[str, torch.Tensor],
    *,
    anti_collapse_weight: float = 0.01,
) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    totals = []
    for capacity, item in output.capacities.items():
        future_visual = masked_mse(
            item.future_visual, batch["future_visual"], batch["future_mask"]
        )
        teammate_qpos = F.mse_loss(item.teammate_qpos, batch["teammate_qpos"])
        teammate_delta = masked_mse(
            item.teammate_delta, batch["teammate_delta"], batch["future_mask"]
        )
        flat = item.tokens.reshape(-1, item.tokens.shape[-1]).float()
        feature_std = flat.std(0, unbiased=False)
        anti_collapse = F.relu(0.1 - feature_std).mean()
        total = (
            future_visual + teammate_qpos + teammate_delta
        ) / 3 + anti_collapse_weight * anti_collapse
        losses[f"capacity_{capacity}"] = total
        losses[f"future_visual_{capacity}"] = future_visual
        losses[f"teammate_qpos_{capacity}"] = teammate_qpos
        losses[f"teammate_delta_{capacity}"] = teammate_delta
        losses[f"anti_collapse_{capacity}"] = anti_collapse
        totals.append(total)
    losses["total"] = torch.stack(totals).mean()
    return losses


class MatchedActionProbe(nn.Module):
    """Identical MLP head used for every N1 representation/control cell."""

    def __init__(self, d_model: int = 384) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, ACTION_PROBE_HORIZON * 8),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).view(value.shape[0], ACTION_PROBE_HORIZON, 8)


class TeamActionProbeSet(nn.Module):
    CONDITIONS = ("belief", "hidden", "time", "row_shuffle")

    def __init__(self, d_model: int = 384) -> None:
        super().__init__()
        self.d_model = d_model
        self.time_projection = nn.Sequential(
            nn.Linear(6 + 5, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.probes = nn.ModuleDict(
            {
                f"{condition}_{capacity}": MatchedActionProbe(d_model)
                for condition in self.CONDITIONS
                for capacity in CAPACITY_CANDIDATES
            }
        )

    def time_feature(self, task_index: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(task_index, num_classes=6).to(phase.dtype)
        phase_features = torch.stack(
            (
                phase,
                phase.square(),
                torch.sin(math.pi * phase),
                torch.cos(math.pi * phase),
                torch.ones_like(phase),
            ),
            dim=-1,
        )
        return self.time_projection(torch.cat((one_hot, phase_features), dim=-1))

    def forward_cell(self, condition: str, capacity: int, value: torch.Tensor) -> torch.Tensor:
        return self.probes[f"{condition}_{capacity}"](value)


__all__ = [
    "MatchedActionProbe",
    "TeamActionProbeSet",
    "SignalCapacityOutput",
    "RawTeamSignalEncoder",
    "TeamSignalOutput",
    "masked_mse",
    "representation_losses",
]
