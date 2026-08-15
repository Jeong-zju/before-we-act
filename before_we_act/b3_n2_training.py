"""Train-time wrappers and the pre-registered reactive control for Step 3-N2."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from before_we_act.b3_n2_model import (
    B3N2PolicyOutput,
    DirectBeliefResidual,
)
from before_we_act.team_belief.n2_core import (
    B3N2Config,
    PredictiveTeamBeliefCore,
    TeacherBeliefInputs,
)


class ReactiveHistoryControl(nn.Module):
    """Direct legal-history residual without predictive state or event memory."""

    def __init__(self, config: B3N2Config) -> None:
        super().__init__()
        d = config.d_model
        self.visual = nn.Linear(config.vision_dim, d, bias=False)
        self.qpos = nn.Linear(config.state_dim, d, bias=False)
        self.action = nn.Linear(config.action_dim, d, bias=False)
        self.position = nn.Parameter(torch.randn(1, 16, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d,
            config.heads,
            4 * d,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.temporal_layers)
        self.norm = nn.LayerNorm(d)
        self.residual = DirectBeliefResidual(d, config.action_dim)

    def forward(
        self,
        decoded: torch.Tensor,
        visual: torch.Tensor,
        qpos: torch.Tensor,
        action: torch.Tensor,
        history_mask: torch.Tensor,
        action_mask: torch.Tensor,
        task_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if visual.ndim != 5 or visual.shape[-2] != 1:
            raise ValueError("reactive control expects pooled legal DINO history")
        pooled_visual = visual.squeeze(-2).mean(2)
        observed = history_mask.unsqueeze(-1).to(decoded.dtype)
        executed = action_mask.unsqueeze(-1).to(decoded.dtype)
        token = (
            self.visual(pooled_visual) * observed
            + self.qpos(qpos) * observed
            + self.action(action) * executed
            + task_token.unsqueeze(1)
            + self.position.to(decoded.dtype)
        )
        valid = history_mask | action_mask
        token = self.encoder(token, src_key_padding_mask=~valid)
        token = self.norm(token) * valid.unsqueeze(-1).to(token.dtype)
        sigma = torch.ones_like(token)
        reliability = torch.ones(
            decoded.shape[0], 1, 1, device=decoded.device, dtype=decoded.dtype
        )
        return self.residual(decoded, token, sigma, reliability)


@dataclass
class B3N2ExperimentOutput:
    candidate: B3N2PolicyOutput
    direct_prediction: torch.Tensor
    direct_residual: torch.Tensor
    direct_gate: torch.Tensor


class B3N2Experiment(nn.Module):
    """The complete N2 candidate and its concurrently trained direct control."""

    def __init__(self, config: B3N2Config) -> None:
        super().__init__()
        self.config = config
        self.belief_core = PredictiveTeamBeliefCore(config, include_teacher=True)
        self.belief_residual = DirectBeliefResidual(config.d_model, config.action_dim)
        self.direct_control = ReactiveHistoryControl(config)

    def forward(self, batch: dict[str, torch.Tensor]) -> B3N2ExperimentOutput:
        belief = self.belief_core(
            batch["runtime_visual_tokens"],
            batch["runtime_visual_mask"],
            batch["history_qpos"],
            batch["history_action"],
            batch["history_mask"],
            batch["action_history_mask"],
            batch["task_token"],
            batch["episode_reset_mask"],
            future_action=batch["action"],
            future_action_mask=batch["action_mask"],
        )
        teacher = self.belief_core.forward_teacher(
            TeacherBeliefInputs(
                current_visual_tokens=batch["teacher_current_visual_tokens"],
                current_visual_mask=batch["teacher_current_visual_mask"],
                future_visual_tokens=batch["teacher_future_visual_tokens"],
                future_visual_mask=batch["teacher_future_visual_mask"],
                future_anchor_mask=batch["teacher_future_anchor_mask"],
                agent_state=batch["teacher_agent_state"],
                agent_mask=batch["teacher_agent_mask"],
                relative_agent_role=batch["teacher_relative_agent_role"],
            ),
            batch["task_token"],
            future_action=batch["action"],
            future_action_mask=batch["action_mask"],
        )
        residual, gate = self.belief_residual(
            batch["decoded_action_hidden"],
            belief.mu,
            belief.sigma,
            belief.reliability,
        )
        prediction = batch["base_action"] + residual
        candidate = B3N2PolicyOutput(
            prediction=prediction,
            base_prediction=batch["base_action"],
            belief_residual=residual,
            residual_gate=gate,
            action_posterior_mu=None,
            action_posterior_logvar=None,
            belief=belief,
            teacher=teacher,
            current_visual_raw=batch["runtime_visual_tokens"][:, -1, :, 0],
            dense_routes=prediction.new_empty(prediction.shape[0], 0),
        )
        direct_residual, direct_gate = self.direct_control(
            batch["decoded_action_hidden"],
            batch["runtime_visual_tokens"],
            batch["history_qpos"],
            batch["history_action"],
            batch["history_mask"],
            batch["action_history_mask"],
            batch["task_token"],
        )
        return B3N2ExperimentOutput(
            candidate=candidate,
            direct_prediction=batch["base_action"] + direct_residual,
            direct_residual=direct_residual,
            direct_gate=direct_gate,
        )


def paired_permutation(pair_id: torch.Tensor) -> torch.Tensor:
    """Return the opposite-arm row for every pre-registered paired example."""

    if pair_id.ndim != 1:
        raise ValueError("pair ids must be rank one")
    rows: dict[int, list[int]] = {}
    for index, value in enumerate(pair_id.detach().cpu().tolist()):
        rows.setdefault(int(value), []).append(index)
    if any(len(indices) != 2 for indices in rows.values()):
        raise ValueError("every N2 pair id must occur exactly twice")
    permutation = torch.empty_like(pair_id)
    for first, second in rows.values():
        permutation[first] = second
        permutation[second] = first
    return permutation


__all__ = [
    "B3N2Experiment",
    "B3N2ExperimentOutput",
    "ReactiveHistoryControl",
    "paired_permutation",
]
