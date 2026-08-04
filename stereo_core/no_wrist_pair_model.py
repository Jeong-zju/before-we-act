"""Stereo-CoRE adaptation for fixed RGB cameras without wrist RGB-D.

The original policy fuses a wrist RGB grid with an aligned depth grid.  This
variant keeps the same frozen DINOv3, cross-relative-bias fusion, ACT decoder,
and PAIR capability router, but replaces depth with the matching fixed agent
view and uses the global fixed view as cross-attention context.
"""
from __future__ import annotations

import math

import torch

try:  # package import used by tests and new BWA entry points
    from .bwa_contracts import (
        CoreCandidateBank,
        CoreContext,
        CoreDeploymentContext,
        CorePerceptionExtension,
        CoreViewTokens,
        apply_perception_residual,
    )
    from .rgbd_patch_fusion import RGBDPatchFusion
    from .stereo_decoder_variants import ARCADecoder
    from .train_act import ACT
except ImportError:  # original direct-script deployment remains supported
    from bwa_contracts import (
        CoreCandidateBank,
        CoreContext,
        CoreDeploymentContext,
        CorePerceptionExtension,
        CoreViewTokens,
        apply_perception_residual,
    )
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
        # ``None`` is intentional: R9 adds no module, parameter or buffer.
        self.perception_extension: CorePerceptionExtension | None = None

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision.eval()
        return self

    def register_perception_extension(self, extension: CorePerceptionExtension) -> None:
        """Register the sole R10 extension without changing the native API."""
        if self.perception_extension is not None:
            raise RuntimeError("a perception extension is already registered")
        self.perception_extension = extension

    def encode_view_tokens(
        self, global_rgb: torch.Tensor, local_rgb: torch.Tensor
    ) -> CoreViewTokens:
        """Run each frozen DINO view once, then retain the parent fusion."""
        local = self._vision_tokens(local_rgb) + self.local_view
        global_context = self._vision_tokens(global_rgb) + self.global_view
        if local.shape[1] != 30 * 40 or global_context.shape[1] != 30 * 40:
            raise ValueError(
                f"strict aligned 30x40 tokens required, got local={local.shape[1]} "
                f"and global={global_context.shape[1]}"
            )
        parent_fused = self.fusion(
            local,
            global_context,
            self.fusion_pos.to(dtype=local.dtype),
        )
        return CoreViewTokens(local, global_context, parent_fused)

    def _paired_tokens(self, global_rgb: torch.Tensor, local_rgb: torch.Tensor) -> torch.Tensor:
        return self.encode_view_tokens(global_rgb, local_rgb).parent_fused

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

    def _sample_training_latent(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preserve the parent's posterior, clamp and RNG consumption order."""
        encoded = self.posterior(self.action(actions) + self.pos)
        mu, logvar = self.latent(encoded.mean(1)).chunk(2, -1)
        logvar = logvar.clamp(-10.0, 5.0)
        latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return latent, mu, logvar

    def encode_context(
        self,
        global_rgb: torch.Tensor,
        local_rgb: torch.Tensor,
        qpos: torch.Tensor,
        *,
        latent: torch.Tensor | None = None,
        deployment_context: CoreDeploymentContext | None = None,
        _views: CoreViewTokens | None = None,
        _state_vec: torch.Tensor | None = None,
        _sparse_routes: torch.Tensor | None = None,
    ) -> CoreContext:
        """Encode one legal deployment decision for any gate selection.

        Private precomputed arguments let ``forward`` preserve the original
        train-mode RNG order: fusion dropout, route, posterior sampling, decode.
        External callers use only the documented public arguments.
        """
        views = _views if _views is not None else self.encode_view_tokens(global_rgb, local_rgb)
        state_vec = _state_vec if _state_vec is not None else self.state(qpos)
        observation, auxiliary, diagnostics = apply_perception_residual(
            views, state_vec, deployment_context, self.perception_extension
        )
        sparse_routes = (
            _sparse_routes
            if _sparse_routes is not None and observation is views.parent_fused
            else self._pair_route(state_vec, observation)
        )
        if latent is None:
            latent = torch.zeros(
                (global_rgb.shape[0], self.z_proj.in_features),
                device=global_rgb.device,
            )
        memory = torch.cat(
            (state_vec.unsqueeze(1), self.z_proj(latent).unsqueeze(1), observation),
            dim=1,
        )
        query = self.query.expand(global_rgb.shape[0], -1, -1)
        return CoreContext(
            views=views,
            observation=observation,
            state_vec=state_vec,
            latent=latent,
            memory=memory,
            query=query,
            dense_routes=self.last_dense_routes,
            sparse_routes=sparse_routes,
            provenance={
                "policy": "NoWristPAIRRoute",
                "inputs": "current_global_rgb+matching_local_rgb+own_qpos",
                "deployment_context": deployment_context is not None,
            },
            auxiliary=auxiliary,
            diagnostics=diagnostics,
        )

    def decode_with_gates(
        self, context: CoreContext, gates: torch.Tensor
    ) -> torch.Tensor:
        """Decode a normalized chunk through the unchanged ARCA/out path."""
        return self.out(
            self.decoder(
                context.query,
                context.memory,
                context.observation,
                gates,
            )
        )

    @torch.no_grad()
    def propose_core_bank(
        self,
        global_rgb: torch.Tensor,
        local_rgb: torch.Tensor,
        qpos: torch.Tensor,
        *,
        deployment_context: CoreDeploymentContext | None = None,
    ) -> CoreCandidateBank:
        """Return base plus all four deployment-legal forced-role chunks."""
        context = self.encode_context(
            global_rgb,
            local_rgb,
            qpos,
            deployment_context=deployment_context,
        )
        base = self.decode_with_gates(context, context.sparse_routes)
        chunks = [base]
        routes = [context.sparse_routes]
        sources = ["core_base"]
        for role in range(self.roles_n):
            forced = base.new_zeros((base.shape[0], self.horizon, self.roles_n))
            forced[..., role] = 1
            chunks.append(self.decode_with_gates(context, forced))
            routes.append(forced)
            sources.append(f"core_forced_role_{role}")
        stacked = torch.stack(chunks, dim=1)
        route_bank = torch.stack(routes, dim=1)
        valid = torch.ones(stacked.shape[:2], dtype=torch.bool, device=stacked.device)
        return CoreCandidateBank(
            chunks=stacked,
            source=tuple(sources),
            routes=route_bank,
            valid_mask=valid,
            provenance={
                **context.provenance,
                "normalized": True,
                "candidate_zero": "bit_exact_native_sparse_route",
                "privileged_inputs": False,
            },
        )

    def forward(
        self,
        global_rgb: torch.Tensor,
        local_rgb: torch.Tensor,
        qpos: torch.Tensor,
        actions: torch.Tensor | None = None,
        return_routing: bool = False,
        counterfactual: bool = False,
        deployment_context: CoreDeploymentContext | None = None,
    ):
        views = self.encode_view_tokens(global_rgb, local_rgb)
        observation = views.parent_fused
        state_vec = self.state(qpos)
        gates = self._pair_route(state_vec, observation)
        if actions is not None:
            latent, mu, logvar = self._sample_training_latent(actions)
        else:
            mu = logvar = None
            latent = None
        context = self.encode_context(
            global_rgb,
            local_rgb,
            qpos,
            latent=latent,
            deployment_context=deployment_context,
            _views=views,
            _state_vec=state_vec,
            _sparse_routes=gates,
        )
        prediction = self.decode_with_gates(context, context.sparse_routes)
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
                    self.decode_with_gates(
                        CoreContext(
                            views=context.views,
                            observation=context.observation[:1],
                            state_vec=context.state_vec[:1],
                            latent=context.latent[:1],
                            memory=context.memory[:1],
                            query=context.query[:1],
                            dense_routes=context.dense_routes[:1],
                            sparse_routes=context.sparse_routes[:1],
                            provenance=context.provenance,
                            auxiliary=context.auxiliary,
                            diagnostics=context.diagnostics,
                        ),
                        forced,
                    )
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
