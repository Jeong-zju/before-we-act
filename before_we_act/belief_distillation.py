"""Action-grounded privileged teacher and deployment-legal belief student."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from before_we_act.raw_team_signal_data import ACTION_PROBE_HORIZON, FUTURE_OFFSETS
from before_we_act.raw_team_signal_model import MatchedActionProbe
from before_we_act.action_grounded_belief import BELIEF_TOKEN_CAPACITY, ActionGroundedDataset


@dataclass(frozen=True)
class TeacherOutput:
    tokens: torch.Tensor
    action: torch.Tensor
    teammate_action_mean: torch.Tensor
    teammate_action_logvar: torch.Tensor
    teammate_delta: torch.Tensor


@dataclass(frozen=True)
class StudentOutput:
    tokens: torch.Tensor
    token_logvar: torch.Tensor
    action: torch.Tensor
    teammate_action_mean: torch.Tensor
    teammate_action_logvar: torch.Tensor
    teammate_delta: torch.Tensor
    future_visual: torch.Tensor


class PrivilegedTokenEncoder(nn.Module):
    TOKEN_COUNT = 2 + len(FUTURE_OFFSETS) + ACTION_PROBE_HORIZON

    def __init__(self, d_model: int = 384) -> None:
        super().__init__()
        self.qpos = nn.Linear(9, d_model)
        self.delta = nn.Linear(9, d_model)
        self.action = nn.Linear(8, d_model)
        self.type_embedding = nn.Parameter(
            torch.randn(1, self.TOKEN_COUNT, d_model) * 0.02
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, oracle: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current = oracle["teammate_qpos"]
        previous = oracle["previous_teammate_qpos"]
        tokens = torch.cat(
            (
                self.qpos(current).unsqueeze(1),
                self.qpos(previous).unsqueeze(1),
                self.delta(oracle["teammate_delta"]),
                self.action(oracle["oracle_teammate_action"]),
            ),
            dim=1,
        )
        valid = torch.cat(
            (
                torch.ones(current.shape[0], 2, dtype=torch.bool, device=current.device),
                oracle["future_mask"].bool(),
                oracle["oracle_teammate_action_mask"].bool(),
            ),
            dim=1,
        )
        return self.norm(tokens + self.type_embedding.to(tokens.dtype)), valid


class ResidualActionHead(nn.Module):
    """Zero-init action correction from H querying all belief tokens."""

    def __init__(self, d_model: int = 384, tokens: int = BELIEF_TOKEN_CAPACITY) -> None:
        super().__init__()
        self.tokens = int(tokens)
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(d_model, 8, dropout=0.1, batch_first=True)
        self.output = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, ACTION_PROBE_HORIZON * 8),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def residual(self, h: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1:] != (self.tokens, h.shape[-1]):
            raise ValueError("R1 residual full-token contract differs")
        query = self.query_norm(h).unsqueeze(1)
        memory = self.memory_norm(tokens)
        attended = self.cross(query, memory, memory, need_weights=False)[0].squeeze(1)
        value = self.output(torch.cat((h, attended), dim=-1))
        return value.view(h.shape[0], ACTION_PROBE_HORIZON, 8)

    def forward(
        self, base_action: torch.Tensor, h: torch.Tensor, tokens: torch.Tensor
    ) -> torch.Tensor:
        return base_action + self.residual(h, tokens)


class PrivilegedBeliefTeacher(nn.Module):
    def __init__(self, d_model: int = 384, tokens: int = BELIEF_TOKEN_CAPACITY) -> None:
        super().__init__()
        self.tokens_n = int(tokens)
        self.privileged = PrivilegedTokenEncoder(d_model)
        self.h_projection = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.queries = nn.Parameter(torch.randn(1, tokens, d_model) * 0.02)
        self.reader = nn.MultiheadAttention(d_model, 8, dropout=0.1, batch_first=True)
        self.token_norm = nn.LayerNorm(d_model)
        self.action_residual = ResidualActionHead(d_model, tokens)
        self.teammate_action = nn.Linear(d_model, ACTION_PROBE_HORIZON * 8 * 2)
        self.teammate_delta = nn.Linear(d_model, len(FUTURE_OFFSETS) * 9)

    def belief_tokens(
        self, h: torch.Tensor, oracle: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        privileged, valid = self.privileged(oracle)
        memory = torch.cat((self.h_projection(h).unsqueeze(1), privileged), dim=1)
        mask = torch.cat(
            (torch.ones(h.shape[0], 1, dtype=torch.bool, device=h.device), valid), dim=1
        )
        query = self.queries.expand(h.shape[0], -1, -1)
        attended = self.reader(
            query, memory, memory, key_padding_mask=~mask, need_weights=False
        )[0]
        return self.token_norm(query + attended)

    def forward(
        self,
        base_action: torch.Tensor,
        h: torch.Tensor,
        oracle: Mapping[str, torch.Tensor],
    ) -> TeacherOutput:
        tokens = self.belief_tokens(h, oracle)
        pooled = tokens.mean(1)
        action_parameters = self.teammate_action(pooled).view(
            h.shape[0], ACTION_PROBE_HORIZON, 8, 2
        )
        return TeacherOutput(
            tokens=tokens,
            action=self.action_residual(base_action, h, tokens),
            teammate_action_mean=action_parameters[..., 0],
            teammate_action_logvar=action_parameters[..., 1].clamp(-8.0, 5.0),
            teammate_delta=self.teammate_delta(pooled).view(
                h.shape[0], len(FUTURE_OFFSETS), 9
            ),
        )


class LegalBeliefStudent(nn.Module):
    """Only frozen B0-H legal history tokens are visible to this student."""

    def __init__(self, d_model: int = 384, tokens: int = BELIEF_TOKEN_CAPACITY) -> None:
        super().__init__()
        self.tokens_n = int(tokens)
        self.queries = nn.Parameter(torch.randn(1, tokens, d_model) * 0.02)
        self.reader = nn.MultiheadAttention(d_model, 8, dropout=0.1, batch_first=True)
        self.token_norm = nn.LayerNorm(d_model)
        self.logvar = nn.Linear(d_model, d_model)
        self.action_residual = ResidualActionHead(d_model, tokens)
        self.teammate_action = nn.Linear(d_model, ACTION_PROBE_HORIZON * 8 * 2)
        self.teammate_delta = nn.Linear(d_model, len(FUTURE_OFFSETS) * 9)
        self.future_visual = nn.Linear(d_model, len(FUTURE_OFFSETS) * 2 * 768)

    def belief_tokens(self, history: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        query = self.queries.expand(history.shape[0], -1, -1)
        attended = self.reader(
            query,
            history,
            history,
            key_padding_mask=~history_mask.bool(),
            need_weights=False,
        )[0]
        return self.token_norm(query + attended)

    def forward(
        self,
        base_action: torch.Tensor,
        h: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> StudentOutput:
        tokens = self.belief_tokens(history, history_mask)
        pooled = tokens.mean(1)
        action_parameters = self.teammate_action(pooled).view(
            h.shape[0], ACTION_PROBE_HORIZON, 8, 2
        )
        return StudentOutput(
            tokens=tokens,
            token_logvar=self.logvar(tokens).clamp(-8.0, 5.0),
            action=self.action_residual(base_action, h, tokens),
            teammate_action_mean=action_parameters[..., 0],
            teammate_action_logvar=action_parameters[..., 1].clamp(-8.0, 5.0),
            teammate_delta=self.teammate_delta(pooled).view(
                h.shape[0], len(FUTURE_OFFSETS), 9
            ),
            future_visual=self.future_visual(pooled).view(
                h.shape[0], len(FUTURE_OFFSETS), 2, 768
            ),
        )

    def belief_off(self, base_action: torch.Tensor) -> torch.Tensor:
        return base_action


class DirectReactiveControl(nn.Module):
    """Action-only capacity control with the same 16-query history bottleneck."""

    def __init__(self, d_model: int = 384, tokens: int = BELIEF_TOKEN_CAPACITY) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, tokens, d_model) * 0.02)
        self.reader = nn.MultiheadAttention(d_model, 8, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.action_residual = ResidualActionHead(d_model, tokens)

    def forward(
        self,
        base_action: torch.Tensor,
        h: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.queries.expand(history.shape[0], -1, -1)
        tokens = self.norm(
            query
            + self.reader(
                query,
                history,
                history,
                key_padding_mask=~history_mask.bool(),
                need_weights=False,
            )[0]
        )
        return self.action_residual(base_action, h, tokens)


def gaussian_nll(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    loss = 0.5 * (logvar + (target - mean).square() * torch.exp(-logvar))
    weight = mask.to(loss.dtype)
    while weight.ndim < loss.ndim:
        weight = weight.unsqueeze(-1)
    return (loss * weight).sum() / weight.expand_as(loss).sum().clamp_min(1)


def oracle_fields(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: batch[key] for key in ActionGroundedDataset.ORACLE_ONLY_FIELDS}


__all__ = [
    "DirectReactiveControl",
    "LegalBeliefStudent",
    "PrivilegedBeliefTeacher",
    "ResidualActionHead",
    "StudentOutput",
    "TeacherOutput",
    "gaussian_nll",
    "oracle_fields",
]
