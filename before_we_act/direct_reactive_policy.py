"""Deploy the frozen, matched direct-reactive control from B-core training.

The formal B-core training checkpoint contains a separately optimized
``direct_control`` branch.  The ordinary deployment export intentionally omits
that branch, so closed-loop attribution must reconstruct it from the immutable
training checkpoint and the same frozen B0-H checkpoint used by B-core.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from before_we_act.predictive_team_belief_training import ReactiveHistoryControl
from before_we_act.team_belief.predictive_core import TeamBeliefConfig
from before_we_act.temporal_action_backbone import TemporalActionBackboneOps


@dataclass
class DirectReactivePolicyOutput:
    prediction: torch.Tensor
    base_prediction: torch.Tensor
    direct_residual: torch.Tensor
    direct_gate: torch.Tensor
    current_visual_raw: torch.Tensor


def prefixed_state(
    state: Mapping[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    """Extract a non-empty submodule state while removing its prefix."""

    selected = {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not selected:
        raise ValueError(f"checkpoint has no state under {prefix!r}")
    return selected


class DirectReactiveDeploymentPolicy(nn.Module):
    """B0-H plus the pre-registered legal-history direct residual.

    This is not ``belief-off``.  Belief-off returns B0-H exactly, whereas this
    policy adds the independently trained ``ReactiveHistoryControl`` residual.
    It consumes the same current RGB, own state, executed-action history and
    task text as the full B-core deployment path, but no belief state, teacher
    tensor, future value, episode identifier or simulator-private state.
    """

    VARIANTS = TemporalActionBackboneOps.VARIANTS

    def __init__(
        self,
        team_config: TeamBeliefConfig,
        *,
        state_dim: int = 9,
        action_dim: int = 8,
        horizon: int = 100,
        d_model: int = 384,
        enc_layers: int = 4,
        dec_layers: int = 7,
        roles: int = 4,
        role_rank: int = 32,
        history_layers: int = 2,
        dino_model: str,
    ) -> None:
        if (
            team_config.d_model != d_model
            or team_config.state_dim != state_dim
            or team_config.action_dim != action_dim
        ):
            raise ValueError("direct-control and B0-H tensor widths differ")
        nn.Module.__init__(self)
        TemporalActionBackboneOps._initialize_temporal_action_backbone(
            self,
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
        self.direct_control = ReactiveHistoryControl(team_config)

    train = TemporalActionBackboneOps.train
    _raw_vision_tokens = TemporalActionBackboneOps._raw_vision_tokens
    _paired_tokens_and_raw_pool = TemporalActionBackboneOps._paired_tokens_and_raw_pool
    _task_token = TemporalActionBackboneOps._task_token
    _encode_history = TemporalActionBackboneOps._encode_history
    _route_action_queries = TemporalActionBackboneOps._route_action_queries
    _decode_action_context = TemporalActionBackboneOps._decode_action_context

    def load_frozen_sources(
        self,
        b0h_state: Mapping[str, torch.Tensor],
        training_state: Mapping[str, torch.Tensor],
    ) -> None:
        incompatible = self.load_state_dict(b0h_state, strict=False)
        allowed_missing = {
            key for key in self.state_dict() if key.startswith("direct_control.")
        }
        if set(incompatible.missing_keys) != allowed_missing:
            raise RuntimeError(
                "B0-H load did not differ only by the direct control: "
                f"{incompatible.missing_keys}"
            )
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"B0-H load has unexpected keys: {incompatible.unexpected_keys}"
            )
        self.direct_control.load_state_dict(
            prefixed_state(training_state, "direct_control."), strict=True
        )

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
    ) -> DirectReactivePolicyOutput:
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
            None,
        )
        base_prediction = self.out(context.decoded)
        if self.hidden_residual is None:
            raise RuntimeError("direct control requires the B0-H hidden residual")
        history_context = context.history_summary.unsqueeze(1).expand(
            -1, self.horizon, -1
        )
        base_prediction = base_prediction + self.hidden_residual(
            torch.cat((context.decoded, history_context), dim=-1)
        )
        runtime_visual_tokens = torch.cat(
            (
                history_visual_raw[:, :-1].to(context.current_visual_raw.dtype),
                context.current_visual_raw.unsqueeze(1),
            ),
            dim=1,
        ).unsqueeze(-2)
        direct_residual, direct_gate = self.direct_control(
            context.decoded,
            runtime_visual_tokens,
            history_qpos,
            history_action,
            history_mask,
            action_history_mask,
            context.task_token,
        )
        return DirectReactivePolicyOutput(
            prediction=base_prediction + direct_residual,
            base_prediction=base_prediction,
            direct_residual=direct_residual,
            direct_gate=direct_gate,
            current_visual_raw=context.current_visual_raw,
        )


__all__ = [
    "DirectReactiveDeploymentPolicy",
    "DirectReactivePolicyOutput",
    "prefixed_state",
]
