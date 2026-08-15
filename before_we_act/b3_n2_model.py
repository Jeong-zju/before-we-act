"""Integrated Step 3-N2 B-core action policy.

This module is intentionally dormant until a 3-N1 receipt supplies the four
required capacity choices in :class:`B3N2Config`.  It reuses the exact Step-2
history/action backbone, replaces the generic hidden residual with a
belief-conditioned direct residual, and keeps privileged teacher processing
behind an explicit removable branch.  The core runtime forward never accepts
teacher tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from before_we_act.b0h_model import B0HPolicy
from before_we_act.team_belief.n2_core import (
    B3N2Config,
    BeliefCoreOutput,
    BeliefRuntimeState,
    PredictiveTeamBeliefCore,
    TeacherBeliefInputs,
    TeacherBeliefOutput,
)


@dataclass
class B3N2PolicyOutput:
    prediction: torch.Tensor
    base_prediction: torch.Tensor
    belief_residual: torch.Tensor
    residual_gate: torch.Tensor
    action_posterior_mu: torch.Tensor | None
    action_posterior_logvar: torch.Tensor | None
    belief: BeliefCoreOutput
    teacher: TeacherBeliefOutput | None
    current_visual_raw: torch.Tensor
    dense_routes: torch.Tensor


class DirectBeliefResidual(nn.Module):
    """Zero-init direct residual whose action queries read every B token."""

    def __init__(self, d_model: int, action_dim: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model, 8, dropout=0.1, batch_first=True
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )
        self.gate = nn.Linear(d_model, 1)
        self.output = nn.Linear(d_model, action_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        action_hidden: torch.Tensor,
        belief_mu: torch.Tensor,
        belief_sigma: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if action_hidden.ndim != 3 or belief_mu.ndim != 3:
            raise ValueError("action hidden and belief must be rank-three tensors")
        if belief_mu.shape != belief_sigma.shape:
            raise ValueError("belief mu/sigma shape differs")
        if action_hidden.shape[0] != belief_mu.shape[0]:
            raise ValueError("action/belief batch differs")
        if reliability.shape != (action_hidden.shape[0], 1, 1):
            raise ValueError("belief reliability must be [batch,1,1]")
        # ``belief_sigma`` is now normalized categorical entropy.  It remains
        # in the interface for diagnostics, but must never rescale memory:
        # reciprocal uncertainty amplified the old Gaussian collapse.
        memory = self.memory_norm(belief_mu)
        attended = self.cross_attention(
            self.query_norm(action_hidden), memory, memory, need_weights=False
        )[0]
        state = self.fusion(torch.cat((action_hidden, attended), dim=-1))
        learned_gate = torch.sigmoid(self.gate(state))
        residual = reliability.to(state.dtype) * learned_gate * self.output(state)
        return residual, learned_gate


class B3N2Policy(B0HPolicy):
    """B0-H action backbone plus the full symmetric predictive B-core."""

    def __init__(
        self,
        n2_config: B3N2Config,
        state_dim: int = 9,
        action_dim: int = 8,
        *,
        horizon: int = 100,
        d_model: int = 384,
        enc_layers: int = 4,
        dec_layers: int = 7,
        roles: int = 4,
        role_rank: int = 32,
        history_layers: int = 2,
        dino_model: str,
        include_teacher: bool = True,
    ) -> None:
        if (
            n2_config.d_model != d_model
            or n2_config.state_dim != state_dim
            or n2_config.action_dim != action_dim
        ):
            raise ValueError("B-core and action-backbone tensor widths must match")
        if n2_config.vision_dim != 768:
            raise ValueError("the frozen DINOv3 ViT-B evidence width is 768")
        super().__init__(
            state_dim,
            action_dim,
            variant="hidden_residual",
            horizon=horizon,
            d_model=d_model,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            roles=roles,
            role_rank=role_rank,
            history_layers=history_layers,
            dino_model=dino_model,
        )
        self.n2_config = n2_config
        self.belief_core = PredictiveTeamBeliefCore(
            n2_config, include_teacher=include_teacher
        )
        self.direct_belief_residual = DirectBeliefResidual(d_model, action_dim)

    @staticmethod
    def _window_reset_mask(history_mask: torch.Tensor) -> torch.Tensor:
        if history_mask.ndim != 2 or history_mask.dtype != torch.bool:
            raise ValueError("history mask must be boolean [batch,time]")
        # A self-contained TeamTemporalSample never crosses an episode.  Its
        # first legal observation is therefore the clean recurrent boundary.
        return history_mask & (history_mask.to(torch.int64).cumsum(1) == 1)

    def forward(
        self,
        global_rgb: torch.Tensor,
        local_rgb: torch.Tensor,
        history_visual_raw: torch.Tensor,
        history_qpos: torch.Tensor,
        history_action: torch.Tensor,
        history_mask: torch.Tensor,
        action_history_mask: torch.Tensor,
        task_bytes: torch.Tensor,
        task_text_mask: torch.Tensor,
        episode_reset: torch.Tensor,
        actions: torch.Tensor | None = None,
        *,
        episode_reset_mask: torch.Tensor | None = None,
        initial_belief_state: BeliefRuntimeState | None = None,
        teacher_inputs: TeacherBeliefInputs | None = None,
        belief_enabled: bool = True,
    ) -> B3N2PolicyOutput:
        context = self._decode_action_context(
            global_rgb,
            local_rgb,
            history_visual_raw,
            history_qpos,
            history_action,
            history_mask,
            action_history_mask,
            task_bytes,
            task_text_mask,
            episode_reset,
            actions,
        )
        if episode_reset_mask is None:
            if initial_belief_state is not None:
                raise ValueError(
                    "incremental belief state requires an explicit episode reset mask"
                )
            reset_mask = self._window_reset_mask(history_mask)
        else:
            reset_mask = episode_reset_mask
        if reset_mask.shape != history_mask.shape or reset_mask.dtype != torch.bool:
            raise ValueError("belief episode reset mask must be boolean [batch,time]")
        if episode_reset.shape != (history_mask.shape[0],):
            raise ValueError("action-backbone episode reset shape differs")
        if torch.any(episode_reset & ~reset_mask[:, -1]):
            raise ValueError("episode start must reset B at the current timestep")

        # The B runtime path is closed over the same legal frozen-DINO history
        # consumed by B0-H.  The exact current pooled features replace the
        # cached last slot, matching B0-H and preventing a second caller-owned
        # visual channel from smuggling privileged/future information into B.
        runtime_visual_tokens = torch.cat(
            (
                history_visual_raw[:, :-1].to(context.current_visual_raw.dtype),
                context.current_visual_raw.unsqueeze(1),
            ),
            dim=1,
        ).unsqueeze(-2)
        runtime_visual_mask = history_mask[:, :, None, None].expand(
            -1, -1, runtime_visual_tokens.shape[2], 1
        )
        belief = self.belief_core(
            runtime_visual_tokens,
            runtime_visual_mask,
            history_qpos,
            history_action,
            history_mask,
            action_history_mask,
            context.task_token,
            reset_mask,
            initial_state=initial_belief_state,
        )
        base_prediction = self.out(context.decoded)
        if self.hidden_residual is None:
            raise RuntimeError("3-N2 requires the formal hidden-residual B0-H base")
        history_context = context.history_summary.unsqueeze(1).expand(
            -1, self.horizon, -1
        )
        base_prediction = base_prediction + self.hidden_residual(
            torch.cat((context.decoded, history_context), dim=-1)
        )
        residual, residual_gate = self.direct_belief_residual(
            context.decoded, belief.mu, belief.sigma, belief.reliability
        )
        if belief_enabled:
            prediction = base_prediction + residual
        else:
            # This branch is intentionally structural.  A diagnostic B-off
            # cannot be contaminated by NaN values or finite-precision 0*x.
            residual = torch.zeros_like(base_prediction)
            residual_gate = torch.zeros_like(residual_gate)
            prediction = base_prediction

        teacher = None
        if teacher_inputs is not None:
            teacher = self.belief_core.forward_teacher(
                teacher_inputs, context.task_token
            )
        if self.last_dense_routes is None:
            raise RuntimeError("action routing diagnostics were not produced")
        return B3N2PolicyOutput(
            prediction=prediction,
            base_prediction=base_prediction,
            belief_residual=residual,
            residual_gate=residual_gate,
            action_posterior_mu=context.mu,
            action_posterior_logvar=context.logvar,
            belief=belief,
            teacher=teacher,
            current_visual_raw=context.current_visual_raw,
            dense_routes=self.last_dense_routes,
        )

    def strip_teacher_(self) -> "B3N2Policy":
        self.belief_core.strip_teacher_()
        return self

    def deployment_state_dict(self) -> Mapping[str, torch.Tensor]:
        """Return a state dict that contains no privileged teacher weights."""

        prefix = "belief_core.teacher_branch."
        return {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith(prefix)
        }


__all__ = [
    "B3N2Policy",
    "B3N2PolicyOutput",
    "DirectBeliefResidual",
]
