"""Standalone temporal-history policies with legal, non-social inputs."""
from __future__ import annotations

import torch
from torch import nn

from before_we_act.temporal_action_backbone import (
    TemporalActionBackboneOps,
    TemporalActionContext,
)


class TemporalHistoryPolicy(nn.Module):
    """Project-owned B0-H action model with a shared 16-step encoder.

    ``history_only`` exposes legal history to the ordinary action path.
    ``hidden_residual`` adds the frozen zero-initialized residual that reads
    decoded action and history hidden states. Neither variant accepts B/P/T.
    """

    VARIANTS = TemporalActionBackboneOps.VARIANTS

    def __init__(
        self,
        state_dim: int = 9,
        action_dim: int = 8,
        *,
        variant: str,
        horizon: int = 100,
        d_model: int = 384,
        enc_layers: int = 4,
        dec_layers: int = 7,
        roles: int = 4,
        role_rank: int = 32,
        history_layers: int = 2,
        dino_model: str,
        image_height: int = 480,
        image_width: int = 640,
        strict_dino_contract: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        TemporalActionBackboneOps._initialize_temporal_action_backbone(
            self,
            state_dim,
            action_dim,
            variant=variant,
            horizon=horizon,
            d_model=d_model,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            roles=roles,
            role_rank=role_rank,
            history_layers=history_layers,
            dino_model=dino_model,
            image_height=image_height,
            image_width=image_width,
            strict_dino_contract=strict_dino_contract,
        )

    train = TemporalActionBackboneOps.train
    _raw_vision_tokens = TemporalActionBackboneOps._raw_vision_tokens
    _paired_tokens_and_raw_pool = TemporalActionBackboneOps._paired_tokens_and_raw_pool
    _task_token = TemporalActionBackboneOps._task_token
    _encode_history = TemporalActionBackboneOps._encode_history
    _route_action_queries = TemporalActionBackboneOps._route_action_queries
    _decode_action_context = TemporalActionBackboneOps._decode_action_context

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
        return_routing: bool = False,
        counterfactual: bool = False,
        return_current_visual: bool = False,
    ):
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
        base_prediction = self.out(context.decoded)
        residual = torch.zeros_like(base_prediction)
        if self.hidden_residual is not None:
            history_context = context.history_summary.unsqueeze(1).expand(
                -1, self.horizon, -1
            )
            residual = self.hidden_residual(
                torch.cat((context.decoded, history_context), dim=-1)
            )
        prediction = base_prediction + residual

        if not return_routing:
            if return_current_visual:
                return (
                    prediction,
                    context.mu,
                    context.logvar,
                    context.current_visual_raw,
                )
            return prediction, context.mu, context.logvar

        cf_predictions = prediction.new_empty(
            (0, self.horizon, self.roles_n, prediction.shape[-1])
        )
        cf_targets = prediction.new_empty((0, self.horizon, prediction.shape[-1]))
        if counterfactual and actions is not None:
            role_predictions = []
            for role in range(self.roles_n):
                forced = prediction.new_zeros((1, self.horizon, self.roles_n))
                forced[..., role] = 1
                role_decoded = self.decoder(
                    context.query[:1],
                    context.memory[:1],
                    context.observation[:1],
                    forced,
                )
                role_base = self.out(role_decoded)
                role_residual = torch.zeros_like(role_base)
                if self.hidden_residual is not None:
                    role_context = context.history_summary[:1].unsqueeze(1).expand(
                        -1, self.horizon, -1
                    )
                    role_residual = self.hidden_residual(
                        torch.cat((role_decoded, role_context), dim=-1)
                    )
                role_predictions.append(role_base + role_residual)
            cf_predictions = torch.stack(role_predictions, dim=2)
            cf_targets = actions[:1]
        return (
            prediction,
            context.mu,
            context.logvar,
            self.last_dense_routes,
            cf_predictions,
            cf_targets,
            base_prediction,
            residual,
            context.current_visual_raw,
        )


__all__ = ["TemporalActionContext", "TemporalHistoryPolicy"]
