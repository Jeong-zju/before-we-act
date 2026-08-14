"""Fair Step-2 B0-H policies with legal history and no social-state input."""
from __future__ import annotations

import torch
from torch import nn

from no_wrist_pair_model import NoWristPAIRRoute

from before_we_act.step2_temporal_data import HISTORY_STEPS, PAD_BYTE


class B0HPolicy(NoWristPAIRRoute):
    """W10 action backbone plus a shared 16-step temporal encoder.

    ``history_only`` exposes history to the ordinary action backbone.
    ``hidden_residual`` adds a zero-initialized direct residual that reads only
    ordinary decoded/history hidden states.  Neither variant accepts B/P/T.
    """

    VARIANTS = ("history_only", "hidden_residual")

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
    ) -> None:
        if variant not in self.VARIANTS:
            raise ValueError(f"unsupported B0-H variant: {variant}")
        super().__init__(
            state_dim,
            action_dim,
            horizon=horizon,
            d_model=d_model,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            roles=roles,
            role_rank=role_rank,
            dino_model=dino_model,
        )
        self.variant = variant
        self.history_layers_n = int(history_layers)
        self.history_pair = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )
        self.history_action = nn.Linear(action_dim, d_model, bias=False)
        self.task_byte_embedding = nn.Embedding(
            PAD_BYTE + 1, d_model, padding_idx=PAD_BYTE
        )
        self.history_position = nn.Parameter(
            torch.randn(1, HISTORY_STEPS, d_model) * 0.02
        )
        self.history_reset = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        history_layer = nn.TransformerEncoderLayer(
            d_model,
            8,
            d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.history_encoder = nn.TransformerEncoder(
            history_layer, num_layers=history_layers
        )
        self.history_norm = nn.LayerNorm(d_model)
        self.task_token_norm = nn.LayerNorm(d_model)
        if variant == "hidden_residual":
            self.hidden_residual = nn.Sequential(
                nn.LayerNorm(2 * d_model),
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, action_dim),
            )
            nn.init.zeros_(self.hidden_residual[-1].weight)
            nn.init.zeros_(self.hidden_residual[-1].bias)
        else:
            self.hidden_residual = None

    def _raw_vision_tokens(self, image: torch.Tensor) -> torch.Tensor:
        if tuple(image.shape[-2:]) != (480, 640):
            raise ValueError(
                f"B0-H requires original 640x480 RGB, got {tuple(image.shape[-2:])}"
            )
        normalized = (image - self.dino_mean) / self.dino_std
        self.vision.eval()
        with torch.no_grad():
            all_tokens = self.vision(pixel_values=normalized).last_hidden_state
            first_patch = 1 + int(
                getattr(self.vision.config, "num_register_tokens", 0)
            )
            tokens = all_tokens[:, first_patch:]
        if tokens.shape[1:] != (30 * 40, 768):
            raise ValueError(f"unexpected frozen DINO token grid: {tokens.shape}")
        return tokens

    def _paired_tokens_and_raw_pool(
        self, global_rgb: torch.Tensor, local_rgb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_raw = self._raw_vision_tokens(local_rgb)
        global_raw = self._raw_vision_tokens(global_rgb)
        local = self.vision_proj(local_raw) + self.local_view
        global_context = self.vision_proj(global_raw) + self.global_view
        observation = self.fusion(
            local,
            global_context,
            self.fusion_pos.to(dtype=local.dtype),
        )
        raw_pool = torch.stack((global_raw.mean(1), local_raw.mean(1)), dim=1)
        return observation, raw_pool

    def _task_token(
        self, task_bytes: torch.Tensor, task_text_mask: torch.Tensor
    ) -> torch.Tensor:
        if task_bytes.shape != task_text_mask.shape:
            raise ValueError("task byte/mask shape mismatch")
        embedded = self.task_byte_embedding(task_bytes)
        weights = task_text_mask.unsqueeze(-1).to(embedded.dtype)
        pooled = (embedded * weights).sum(1) / weights.sum(1).clamp_min(1)
        return self.task_token_norm(pooled)

    def _encode_history(
        self,
        history_visual_raw: torch.Tensor,
        current_visual_raw: torch.Tensor,
        history_qpos: torch.Tensor,
        history_action: torch.Tensor,
        history_mask: torch.Tensor,
        action_history_mask: torch.Tensor,
        task_bytes: torch.Tensor,
        task_text_mask: torch.Tensor,
        episode_reset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected_visual = (history_qpos.shape[0], HISTORY_STEPS, 2, 768)
        if tuple(history_visual_raw.shape) != expected_visual:
            raise ValueError(
                f"history visual contract differs: {history_visual_raw.shape}"
            )
        if tuple(history_qpos.shape[1:]) != (HISTORY_STEPS, 9):
            raise ValueError("history qpos contract differs")
        if tuple(history_action.shape[1:]) != (HISTORY_STEPS, 8):
            raise ValueError("history action contract differs")
        # The last cached feature is replaced by the exact raw feature computed
        # during this forward.  This prevents cache precision from changing the
        # current observation and makes deployment use the identical path.
        visual = torch.cat(
            (history_visual_raw[:, :-1].to(current_visual_raw.dtype),
             current_visual_raw.unsqueeze(1)),
            dim=1,
        )
        projected = self.vision_proj(visual)
        visual_token = self.history_pair(
            torch.cat((projected[:, :, 0], projected[:, :, 1]), dim=-1)
        )
        observation_weight = history_mask.unsqueeze(-1).to(visual_token.dtype)
        action_weight = action_history_mask.unsqueeze(-1).to(visual_token.dtype)
        task_token = self._task_token(task_bytes, task_text_mask)
        token = (
            visual_token * observation_weight
            + self.state(history_qpos) * observation_weight
            + self.history_action(history_action) * action_weight
            + self.history_position.to(dtype=visual_token.dtype)
            + task_token.unsqueeze(1)
        )
        token[:, -1:] = token[:, -1:] + (
            episode_reset[:, None, None].to(token.dtype) * self.history_reset
        )
        valid = history_mask | action_history_mask
        if not torch.all(valid[:, -1]):
            raise ValueError("current history slot must always be valid")
        encoded = self.history_encoder(token, src_key_padding_mask=~valid)
        encoded = self.history_norm(encoded)
        encoded = encoded * valid.unsqueeze(-1).to(encoded.dtype)
        summary = encoded.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return encoded, summary, task_token

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
        observation, current_visual_raw = self._paired_tokens_and_raw_pool(
            global_rgb, local_rgb
        )
        state_vec = self.state(history_qpos[:, -1])
        history, history_summary, task_token = self._encode_history(
            history_visual_raw,
            current_visual_raw,
            history_qpos,
            history_action,
            history_mask,
            action_history_mask,
            task_bytes,
            task_text_mask,
            episode_reset,
        )
        gates = self._pair_route(state_vec, observation)
        if actions is not None:
            encoded = self.posterior(self.action(actions) + self.pos)
            mu, logvar = self.latent(encoded.mean(1)).chunk(2, -1)
            logvar = logvar.clamp(-10.0, 5.0)
            latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            mu = logvar = None
            latent = torch.zeros(
                (global_rgb.shape[0], self.z_proj.in_features),
                device=global_rgb.device,
                dtype=state_vec.dtype,
            )
        memory = torch.cat(
            (
                state_vec.unsqueeze(1),
                self.z_proj(latent).unsqueeze(1),
                task_token.unsqueeze(1),
                history,
                observation,
            ),
            dim=1,
        )
        query = self.query.expand(global_rgb.shape[0], -1, -1)
        decoded = self.decoder(query, memory, observation, gates)
        base_prediction = self.out(decoded)
        residual = torch.zeros_like(base_prediction)
        if self.hidden_residual is not None:
            context = history_summary.unsqueeze(1).expand(-1, self.horizon, -1)
            residual = self.hidden_residual(torch.cat((decoded, context), dim=-1))
        prediction = base_prediction + residual

        if not return_routing:
            if return_current_visual:
                return prediction, mu, logvar, current_visual_raw
            return prediction, mu, logvar

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
                    query[:1], memory[:1], observation[:1], forced
                )
                role_base = self.out(role_decoded)
                role_residual = torch.zeros_like(role_base)
                if self.hidden_residual is not None:
                    role_context = history_summary[:1].unsqueeze(1).expand(
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
            mu,
            logvar,
            self.last_dense_routes,
            cf_predictions,
            cf_targets,
            base_prediction,
            residual,
            current_visual_raw,
        )


__all__ = ["B0HPolicy"]
