"""Stereo-CoRE adaptation for fixed RGB cameras without wrist RGB-D.

The original policy fuses a wrist RGB grid with an aligned depth grid.  This
variant keeps the same frozen DINOv3, cross-relative-bias fusion, ACT decoder,
and PAIR capability router, but replaces depth with the matching fixed agent
view and uses the global fixed view as cross-attention context.
"""
from __future__ import annotations

import math

import torch

from rgbd_patch_fusion import RGBDPatchFusion
from stereo_decoder_variants import ARCADecoder
from train_act import ACT


class NoWristPAIRRoute(ACT):
    """Global+matching-agent RGB CoRE policy for the no-wrist environment."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        horizon: int = 100,
        d_model: int = 384,
        enc_layers: int = 4,
        dec_layers: int = 7,
        roles: int = 4,
        role_rank: int = 32,
        dino_model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
    ) -> None:
        super().__init__(
            state_dim,
            action_dim,
            horizon,
            d_model,
            enc_layers,
            dec_layers,
            vision_backbone="dinov3_vitb16_frozen",
            dino_model=dino_model,
        )
        self.roles_n = roles
        self.role_rank = role_rank
        self.decoder = ARCADecoder(
            d_model,
            layers=dec_layers,
            roles=roles,
            rank=role_rank,
        )
        self.fusion = RGBDPatchFusion(
            d_model=d_model,
            heads=8,
            grid_h=30,
            grid_w=40,
            layers=2,
            ffn_dim=d_model * 4,
        )
        self.fusion_pos = torch.nn.Parameter(torch.randn(1, 30 * 40, d_model) * 0.02)
        self.local_view = torch.nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.global_view = torch.nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.route_state = torch.nn.Linear(d_model, d_model, bias=False)
        self.route_observation = torch.nn.Linear(d_model, d_model, bias=False)
        self.route_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, d_model),
            torch.nn.GELU(),
            torch.nn.Linear(d_model, d_model, bias=False),
        )
        self.role_prototypes = torch.nn.Parameter(torch.randn(roles, d_model) * 0.02)
        self.compatibility = torch.nn.Parameter(torch.empty(roles, roles))
        torch.nn.init.xavier_uniform_(self.compatibility)
        self.last_dense_routes = None
        self.last_sparse_routes = None

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision.eval()
        return self

    def _paired_tokens(self, global_rgb: torch.Tensor, local_rgb: torch.Tensor) -> torch.Tensor:
        local = self._vision_tokens(local_rgb) + self.local_view
        global_context = self._vision_tokens(global_rgb) + self.global_view
        if local.shape[1] != 30 * 40 or global_context.shape[1] != 30 * 40:
            raise ValueError(
                f"strict aligned 30x40 tokens required, got local={local.shape[1]} "
                f"and global={global_context.shape[1]}"
            )
        return self.fusion(
            local,
            global_context,
            self.fusion_pos.to(dtype=local.dtype),
        )

    def _pair_route(self, state: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        query = self.query.expand(state.shape[0], -1, -1)
        context = self.route_state(state) + self.route_observation(observation.mean(1))
        features = self.route_mlp(query + context.unsqueeze(1))
        logits = torch.matmul(features, self.role_prototypes.t()) / math.sqrt(features.shape[-1])
        dense = logits.softmax(-1)
        values, ids = logits.topk(2, dim=-1)
        sparse = torch.zeros_like(logits).scatter_(
            -1, ids, values.softmax(-1).to(logits.dtype)
        )
        self.last_dense_routes = dense
        self.last_sparse_routes = sparse
        return sparse

    def forward(
        self,
        global_rgb: torch.Tensor,
        local_rgb: torch.Tensor,
        qpos: torch.Tensor,
        actions: torch.Tensor | None = None,
        return_routing: bool = False,
        counterfactual: bool = False,
    ):
        observation = self._paired_tokens(global_rgb, local_rgb)
        state_vec = self.state(qpos)
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
            )
        memory = torch.cat(
            (state_vec.unsqueeze(1), self.z_proj(latent).unsqueeze(1), observation),
            dim=1,
        )
        query = self.query.expand(global_rgb.shape[0], -1, -1)
        prediction = self.out(self.decoder(query, memory, observation, gates))
        if not return_routing:
            return prediction, mu, logvar, observation.new_zeros(())

        cf_predictions = prediction.new_empty(
            (0, self.horizon, self.roles_n, prediction.shape[-1])
        )
        cf_targets = prediction.new_empty((0, self.horizon, prediction.shape[-1]))
        if counterfactual and actions is not None:
            role_predictions = []
            for role in range(self.roles_n):
                forced = prediction.new_zeros((1, self.horizon, self.roles_n))
                forced[..., role] = 1
                role_predictions.append(
                    self.out(self.decoder(query[:1], memory[:1], observation[:1], forced))
                )
            cf_predictions = torch.stack(role_predictions, dim=2)
            cf_targets = actions[:1]
        return (
            prediction,
            mu,
            logvar,
            observation.new_zeros(()),
            self.last_dense_routes,
            cf_predictions,
            cf_targets,
        )

