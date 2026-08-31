"""Project-local temporal action backbone with the frozen B0-H computation graph.

This module deliberately has no runtime dependency on Stereo-CoRE.  Its tensor
operations and public parameter names preserve the already evaluated B0-H/N2
signal flow so historical checkpoints remain loadable without changing the
belief residual or the action distribution represented by the model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os

import torch
from torch import nn
import torch.nn.functional as F

from before_we_act.temporal_history_data import HISTORY_STEPS, PAD_BYTE


@dataclass(frozen=True)
class TemporalActionContext:
    """Action-side tensors shared by the temporal and team-belief policies."""

    observation: torch.Tensor
    current_visual_raw: torch.Tensor
    history: torch.Tensor
    history_summary: torch.Tensor
    task_token: torch.Tensor
    query: torch.Tensor
    memory: torch.Tensor
    decoded: torch.Tensor
    mu: torch.Tensor | None
    logvar: torch.Tensor | None


class _GridRelativePositionBias(nn.Module):
    """Per-head decomposed 2-D relative attention bias."""

    def __init__(self, heads: int, grid_h: int, grid_w: int) -> None:
        super().__init__()
        self.heads = heads
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.row = nn.Parameter(torch.zeros(2 * grid_h - 1, heads))
        self.col = nn.Parameter(torch.zeros(2 * grid_w - 1, heads))
        nn.init.trunc_normal_(self.row, std=0.02)
        nn.init.trunc_normal_(self.col, std=0.02)

    def forward(self) -> torch.Tensor:
        rows = torch.arange(self.grid_h, device=self.row.device)
        cols = torch.arange(self.grid_w, device=self.col.device)
        row_bias = self.row[
            rows[:, None] - rows[None, :] + self.grid_h - 1
        ]
        col_bias = self.col[
            cols[:, None] - cols[None, :] + self.grid_w - 1
        ]
        return (
            row_bias[:, None, :, None, :] + col_bias[None, :, None, :, :]
        ).permute(4, 0, 1, 2, 3).reshape(
            self.heads,
            self.grid_h * self.grid_w,
            self.grid_h * self.grid_w,
        )


class _AlignedCrossAttention(nn.Module):
    """Cross-attention over two spatially aligned full-resolution token grids."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        grid_h: int,
        grid_w: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.bias = _GridRelativePositionBias(heads, grid_h, grid_w)
        self.dropout = dropout

    def forward(
        self, query_tokens: torch.Tensor, context_tokens: torch.Tensor
    ) -> torch.Tensor:
        batch, tokens, width = query_tokens.shape
        if context_tokens.shape != (batch, tokens, width):
            raise ValueError("aligned cross-attention requires identical token grids")
        query = self.q(query_tokens).view(
            batch, tokens, self.heads, self.head_dim
        ).transpose(1, 2)
        key = self.k(context_tokens).view(
            batch, tokens, self.heads, self.head_dim
        ).transpose(1, 2)
        value = self.v(context_tokens).view(
            batch, tokens, self.heads, self.head_dim
        ).transpose(1, 2)
        mask = self.bias().to(dtype=query.dtype, device=query.device).unsqueeze(0)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(attended.transpose(1, 2).reshape(batch, tokens, width))


class _MultiViewFusionBlock(nn.Module):
    """Self-encode both views and fuse the global context into local tokens."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        grid_h: int,
        grid_w: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.rgb_norm = nn.LayerNorm(d_model)
        self.depth_norm = nn.LayerNorm(d_model)
        self.cross_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.rgb_self = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.depth_self = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.cross = _AlignedCrossAttention(
            d_model, heads, grid_h, grid_w, dropout
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        local_tokens: torch.Tensor,
        global_tokens: torch.Tensor,
        position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_query = self.rgb_norm(local_tokens + position)
        global_query = self.depth_norm(global_tokens + position)
        local_tokens = local_tokens + self.dropout(
            self.rgb_self(
                local_query,
                local_query,
                local_tokens,
                need_weights=False,
            )[0]
        )
        global_tokens = global_tokens + self.dropout(
            self.depth_self(
                global_query,
                global_query,
                global_tokens,
                need_weights=False,
            )[0]
        )
        local_tokens = local_tokens + self.dropout(
            self.cross(self.cross_norm(local_tokens + position), global_tokens + position)
        )
        local_tokens = local_tokens + self.dropout(
            self.ffn(self.ffn_norm(local_tokens))
        )
        return local_tokens, global_tokens


class _AlignedMultiViewFusion(nn.Module):
    """Two-stage fusion of matching 30x40 local/global DINO patch grids."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        grid_h: int,
        grid_w: int,
        layers: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.blocks = nn.ModuleList(
            _MultiViewFusionBlock(
                d_model, heads, grid_h, grid_w, ffn_dim, dropout
            )
            for _ in range(layers)
        )

    def forward(
        self,
        local_tokens: torch.Tensor,
        global_tokens: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        expected = self.grid_h * self.grid_w
        if local_tokens.shape[1] != expected or global_tokens.shape[1] != expected:
            raise ValueError(
                f"expected aligned {self.grid_h}x{self.grid_w} tokens, got "
                f"{local_tokens.shape[1]} and {global_tokens.shape[1]}"
            )
        for block in self.blocks:
            local_tokens, global_tokens = block(
                local_tokens, global_tokens, position
            )
        return local_tokens


class _ResidualMLP(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class _RoleObservationAdapter(nn.Module):
    """Low-rank observation readout for one latent action role."""

    def __init__(self, d_model: int, rank: int) -> None:
        super().__init__()
        self.q = nn.Linear(d_model, rank, bias=False)
        self.k = nn.Linear(d_model, rank, bias=False)
        self.v = nn.Linear(d_model, rank, bias=False)
        self.out = nn.Linear(rank, d_model, bias=False)
        self.rank = rank

    def forward(
        self, action_query: torch.Tensor, observation: torch.Tensor
    ) -> torch.Tensor:
        score = torch.matmul(
            self.q(action_query), self.k(observation).transpose(-1, -2)
        ) / math.sqrt(self.rank)
        return self.out(torch.matmul(score.softmax(-1), self.v(observation)))


class _RoleConditionedActionDecoderLayer(nn.Module):
    """Action decoder layer with routed low-rank reads from current vision."""

    def __init__(
        self,
        d_model: int,
        roles: int,
        rank: int,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.ff = _ResidualMLP(d_model, 4 * d_model, dropout)
        self.adapters = nn.ModuleList(
            _RoleObservationAdapter(d_model, rank) for _ in range(roles)
        )

    def forward(
        self,
        action_tokens: torch.Tensor,
        memory: torch.Tensor,
        observation: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        action_tokens = action_tokens + self.drop1(
            self.self_attn(
                self.norm1(action_tokens),
                self.norm1(action_tokens),
                self.norm1(action_tokens),
                need_weights=False,
            )[0]
        )
        normalized = self.norm2(action_tokens)
        base = self.cross_attn(
            normalized, memory, memory, need_weights=False
        )[0]
        role_context = torch.zeros_like(base)
        for role_index, adapter in enumerate(self.adapters):
            role_context = role_context + gates[..., role_index : role_index + 1] * adapter(
                normalized, observation
            )
        action_tokens = action_tokens + self.drop2(base + role_context)
        return action_tokens + self.ff(self.norm3(action_tokens))


class _RoleConditionedActionDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        layers: int,
        roles: int,
        rank: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            _RoleConditionedActionDecoderLayer(
                d_model, roles, rank, dropout=dropout
            )
            for _ in range(layers)
        )

    def forward(
        self,
        action_tokens: torch.Tensor,
        memory: torch.Tensor,
        observation: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            action_tokens = layer(action_tokens, memory, observation, gates)
        return action_tokens


class TemporalActionBackboneOps:
    """Shared operations bound by project-owned ``nn.Module`` policy classes."""

    VARIANTS = ("history_only", "hidden_residual")

    def _initialize_temporal_action_backbone(
        self,
        state_dim: int,
        action_dim: int,
        *,
        variant: str,
        horizon: int,
        d_model: int,
        enc_layers: int,
        dec_layers: int,
        roles: int,
        role_rank: int,
        history_layers: int,
        dino_model: str,
        image_height: int = 480,
        image_width: int = 640,
        strict_dino_contract: bool = False,
    ) -> None:
        if not isinstance(self, nn.Module):
            raise TypeError("temporal action backbone requires nn.Module ownership")
        if variant not in self.VARIANTS:
            raise ValueError(f"unsupported B0-H variant: {variant}")

        from transformers import AutoImageProcessor, AutoModel

        token = os.environ.get("HF_TOKEN")
        processor = AutoImageProcessor.from_pretrained(dino_model, token=token)
        self.vision = AutoModel.from_pretrained(dino_model, token=token)
        if strict_dino_contract:
            observed = {
                "model_type": getattr(self.vision.config, "model_type", None),
                "hidden_size": getattr(self.vision.config, "hidden_size", None),
                "patch_size": getattr(self.vision.config, "patch_size", None),
                "num_register_tokens": getattr(
                    self.vision.config, "num_register_tokens", None
                ),
                "image_size": getattr(self.vision.config, "image_size", None),
            }
            expected = {
                "model_type": "dinov3_vit",
                "hidden_size": 768,
                "patch_size": 16,
                "num_register_tokens": 4,
                "image_size": 224,
            }
            if observed != expected:
                raise ValueError(
                    f"strict DINOv3 ViT-B/16 model contract differs: {observed}"
                )
            mean = tuple(float(value) for value in processor.image_mean)
            std = tuple(float(value) for value in processor.image_std)
            if mean != (0.485, 0.456, 0.406) or std != (0.229, 0.224, 0.225):
                raise ValueError("strict DINOv3 normalization contract differs")
        self.vision.requires_grad_(False)
        self.vision.eval()
        self.vision_backbone = "dinov3_vitb16_frozen"
        self.dino_model = dino_model
        self.register_buffer(
            "dino_mean", torch.tensor(processor.image_mean).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "dino_std", torch.tensor(processor.image_std).view(1, -1, 1, 1)
        )
        self.strict_dino_contract = bool(strict_dino_contract)
        self.vision_proj = nn.Linear(self.vision.config.hidden_size, d_model)
        # These are benchmark-native input/output widths, not hidden model
        # capacity.  The constructor has always exposed them; retaining them
        # here removes the historical RoboFactory-only validator below.
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.state = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.action = nn.Linear(action_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        self.query = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        posterior_layer = nn.TransformerEncoderLayer(
            d_model,
            8,
            d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        # Preserve the frozen recipe's random-initialization stream.  The old
        # implementation constructed a generic decoder before replacing it by
        # the role-conditioned decoder. Registering it briefly preserves both
        # RNG consumption and state-dict insertion order; it is replaced before
        # construction returns and never enters a forward computation.
        discarded_decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            8,
            d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.posterior = nn.TransformerEncoder(
            posterior_layer, num_layers=enc_layers
        )
        self.decoder = nn.TransformerDecoder(
            discarded_decoder_layer, num_layers=dec_layers
        )
        latent_dim = 32
        self.latent = nn.Linear(d_model, latent_dim * 2)
        self.z_proj = nn.Linear(latent_dim, d_model)
        self.out = nn.Linear(d_model, action_dim)
        self.horizon = horizon

        self.roles_n = roles
        self.role_rank = role_rank
        self.decoder = _RoleConditionedActionDecoder(
            d_model, layers=dec_layers, roles=roles, rank=role_rank
        )
        if image_height <= 0 or image_width <= 0:
            raise ValueError("DINO input dimensions must be positive")
        if image_height % 16 or image_width % 16:
            raise ValueError("DINOv3 ViT-B/16 input dimensions must be divisible by 16")
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.grid_h = self.image_height // 16
        self.grid_w = self.image_width // 16
        self.fusion = _AlignedMultiViewFusion(
            d_model=d_model,
            heads=8,
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            layers=2,
            ffn_dim=d_model * 4,
        )
        self.fusion_pos = nn.Parameter(
            torch.randn(1, self.grid_h * self.grid_w, d_model) * 0.02
        )
        self.local_view = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.global_view = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.route_state = nn.Linear(d_model, d_model, bias=False)
        self.route_observation = nn.Linear(d_model, d_model, bias=False)
        self.route_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.role_prototypes = nn.Parameter(torch.randn(roles, d_model) * 0.02)
        # Checkpoint-only tensor retained for strict loading of completed B0-H
        # and N2 runs.  It is frozen and never enters the computation graph.
        self.compatibility = nn.Parameter(torch.empty(roles, roles))
        nn.init.xavier_uniform_(self.compatibility)
        self.compatibility.requires_grad_(False)
        self.last_dense_routes: torch.Tensor | None = None
        self.last_sparse_routes: torch.Tensor | None = None

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

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        self.vision.eval()
        return self

    def _raw_vision_tokens(self, image: torch.Tensor) -> torch.Tensor:
        if tuple(image.shape[-2:]) != (self.image_height, self.image_width):
            raise ValueError(
                "B0-H image contract differs: expected "
                f"{self.image_width}x{self.image_height} RGB, got {tuple(image.shape[-2:])}"
            )
        normalized = (image - self.dino_mean) / self.dino_std
        self.vision.eval()
        with torch.no_grad():
            all_tokens = self.vision(pixel_values=normalized).last_hidden_state
            first_patch = 1 + int(
                getattr(self.vision.config, "num_register_tokens", 0)
            )
            tokens = all_tokens[:, first_patch:]
        if tokens.shape[1:] != (self.grid_h * self.grid_w, 768):
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
        if tuple(history_qpos.shape[1:]) != (HISTORY_STEPS, self.state_dim):
            raise ValueError(
                "history qpos contract differs: expected "
                f"{(HISTORY_STEPS, self.state_dim)}, got "
                f"{tuple(history_qpos.shape[1:])}"
            )
        if tuple(history_action.shape[1:]) != (HISTORY_STEPS, self.action_dim):
            raise ValueError(
                "history action contract differs: expected "
                f"{(HISTORY_STEPS, self.action_dim)}, got "
                f"{tuple(history_action.shape[1:])}"
            )
        visual = torch.cat(
            (
                history_visual_raw[:, :-1].to(current_visual_raw.dtype),
                current_visual_raw.unsqueeze(1),
            ),
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

    def _route_action_queries(
        self, state: torch.Tensor, observation: torch.Tensor
    ) -> torch.Tensor:
        query = self.query.expand(state.shape[0], -1, -1)
        context = self.route_state(state) + self.route_observation(
            observation.mean(1)
        )
        features = self.route_mlp(query + context.unsqueeze(1))
        logits = torch.matmul(
            features, self.role_prototypes.t()
        ) / math.sqrt(features.shape[-1])
        dense = logits.softmax(-1)
        values, indices = logits.topk(2, dim=-1)
        sparse = torch.zeros_like(logits).scatter_(
            -1, indices, values.softmax(-1).to(logits.dtype)
        )
        self.last_dense_routes = dense
        self.last_sparse_routes = sparse
        return sparse

    def _decode_action_context(
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
        actions: torch.Tensor | None,
    ) -> TemporalActionContext:
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
        gates = self._route_action_queries(state_vec, observation)
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
        return TemporalActionContext(
            observation=observation,
            current_visual_raw=current_visual_raw,
            history=history,
            history_summary=history_summary,
            task_token=task_token,
            query=query,
            memory=memory,
            decoded=decoded,
            mu=mu,
            logvar=logvar,
        )


__all__ = ["TemporalActionBackboneOps", "TemporalActionContext"]
