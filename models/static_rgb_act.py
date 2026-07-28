"""Shared-agent ACT decoders and chunk aggregation for existing static RGB data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class StaticRGBMoEACTConfig:
    state_dim: int = 18
    action_dim: int = 8
    horizon: int = 100
    vision_dim: int = 1024
    d_model: int = 384
    encoder_layers: int = 4
    decoder_layers: int = 7
    heads: int = 8
    ffn_dim: int = 1536
    latent_dim: int = 32
    experts: int = 4
    dropout: float = 0.1
    decoder_kind: str = "sparse_moe"
    dense_ffn_dim: int = 3072

    def __post_init__(self) -> None:
        integer_fields = (
            "state_dim",
            "action_dim",
            "horizon",
            "vision_dim",
            "d_model",
            "encoder_layers",
            "decoder_layers",
            "heads",
            "ffn_dim",
            "latent_dim",
            "experts",
            "dense_ffn_dim",
        )
        if any(int(getattr(self, name)) <= 0 for name in integer_fields):
            raise ValueError("Static RGB ACT dimensions must be positive")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if self.decoder_kind not in {"sparse_moe", "dense"}:
            raise ValueError("decoder_kind must be sparse_moe or dense")
        if self.decoder_kind == "sparse_moe" and self.experts < 2:
            raise ValueError("top-2 MoE requires at least two experts")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StaticRGBMoEACTConfig":
        return cls(**dict(payload))


class _Expert(nn.Module):
    def __init__(self, config: StaticRGBMoEACTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)


class Top2SparseMoE(nn.Module):
    """Top-2 token routing with a differentiable Switch-style balance loss."""

    def __init__(self, config: StaticRGBMoEACTConfig) -> None:
        super().__init__()
        self.expert_count = config.experts
        self.router = nn.Linear(config.d_model, config.experts, bias=False)
        self.experts = nn.ModuleList(_Expert(config) for _ in range(config.experts))

    def forward(self, value: Tensor) -> tuple[Tensor, Tensor]:
        shape = value.shape
        flat = value.reshape(-1, shape[-1])
        logits = self.router(flat)
        top_logits, top_ids = logits.topk(2, dim=-1)
        gates = top_logits.softmax(dim=-1)
        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            chosen = (top_ids == expert_id).nonzero(as_tuple=False)
            if not chosen.numel():
                continue
            token_ids, slots = chosen[:, 0], chosen[:, 1]
            routed = expert(flat.index_select(0, token_ids))
            output.index_add_(
                0,
                token_ids,
                routed * gates[token_ids, slots].unsqueeze(-1),
            )
        importance = logits.softmax(dim=-1).mean(dim=0)
        load = torch.bincount(
            top_ids.reshape(-1), minlength=self.expert_count
        ).to(flat.dtype)
        load = load / (2.0 * flat.shape[0])
        balance = self.expert_count * (importance * load).sum()
        return output.reshape(shape), balance


class DenseFeedForward(nn.Module):
    """Dense decoder FFN with the same auxiliary-loss interface as sparse MoE."""

    def __init__(self, config: StaticRGBMoEACTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.dense_ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dense_ffn_dim, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, value: Tensor) -> tuple[Tensor, Tensor]:
        # One is the neutral value of the existing ``router_aux - 1`` loss.
        return self.net(value), value.new_ones(())


class _MoEDecoderLayer(nn.Module):
    def __init__(self, config: StaticRGBMoEACTConfig) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            config.d_model,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_attention = nn.MultiheadAttention(
            config.d_model,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.moe: Top2SparseMoE | DenseFeedForward
        if config.decoder_kind == "sparse_moe":
            self.moe = Top2SparseMoE(config)
        else:
            self.moe = DenseFeedForward(config)

    def forward(self, query: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        normalized = self.norm1(query)
        query = query + self.dropout(
            self.self_attention(
                normalized,
                normalized,
                normalized,
                need_weights=False,
            )[0]
        )
        query = query + self.dropout(
            self.cross_attention(
                self.norm2(query),
                memory,
                memory,
                need_weights=False,
            )[0]
        )
        routed, balance = self.moe(self.norm3(query))
        return query + routed, balance


class _MoEDecoder(nn.Module):
    def __init__(self, config: StaticRGBMoEACTConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            _MoEDecoderLayer(config) for _ in range(config.decoder_layers)
        )

    def forward(self, query: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        balance = query.new_zeros(())
        for layer in self.layers:
            query, layer_balance = layer(query, memory)
            balance = balance + layer_balance
        return query, balance / len(self.layers)


class StaticRGBMoEACT(nn.Module):
    """ACT CVAE over frozen DINO patch tokens and one normalized Panda state."""

    def __init__(self, config: StaticRGBMoEACTConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_projection = nn.Sequential(
            nn.LayerNorm(config.vision_dim),
            nn.Linear(config.vision_dim, config.d_model),
        )
        self.state_projection = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.action_projection = nn.Linear(config.action_dim, config.d_model)
        self.posterior_position = nn.Parameter(
            torch.randn(1, config.horizon, config.d_model) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            config.d_model,
            config.heads,
            config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.posterior = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=False,
        )
        self.latent = nn.Linear(config.d_model, config.latent_dim * 2)
        self.latent_projection = nn.Linear(config.latent_dim, config.d_model)
        self.query = nn.Parameter(
            torch.randn(1, config.horizon, config.d_model) * 0.02
        )
        self.decoder = _MoEDecoder(config)
        self.action_head = nn.Linear(config.d_model, config.action_dim)

    def forward(
        self,
        vision_tokens: Tensor,
        state: Tensor,
        actions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor]:
        if vision_tokens.ndim != 3 or vision_tokens.shape[-1] != self.config.vision_dim:
            raise ValueError("vision_tokens must be [B,N,vision_dim]")
        if state.shape != (vision_tokens.shape[0], self.config.state_dim):
            raise ValueError("state must be [B,state_dim]")
        memory = [
            self.state_projection(state).unsqueeze(1),
            self.vision_projection(vision_tokens),
        ]
        if actions is None:
            mu = logvar = None
            latent = torch.zeros(
                vision_tokens.shape[0],
                self.config.latent_dim,
                device=vision_tokens.device,
                dtype=vision_tokens.dtype,
            )
        else:
            if actions.shape != (
                vision_tokens.shape[0],
                self.config.horizon,
                self.config.action_dim,
            ):
                raise ValueError("actions violate the ACT horizon contract")
            encoded = self.posterior(
                self.action_projection(actions) + self.posterior_position
            )
            mu, logvar = self.latent(encoded.mean(dim=1)).chunk(2, dim=-1)
            latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        memory.insert(1, self.latent_projection(latent).unsqueeze(1))
        decoded, balance = self.decoder(
            self.query.expand(vision_tokens.shape[0], -1, -1),
            torch.cat(memory, dim=1),
        )
        return self.action_head(decoded), mu, logvar, balance


class TemporalChunkEnsembler:
    """ACT-style exponential ensemble over chunks predicting the current step."""

    def __init__(self, *, horizon: int, decay: float = 0.01) -> None:
        if horizon <= 0 or not math.isfinite(decay) or decay <= 0.0:
            raise ValueError("horizon/decay must be positive")
        self.horizon = int(horizon)
        self.decay = float(decay)
        self.mode = "temporal_ensemble"
        self.step = 0
        self._chunks: list[tuple[int, Tensor]] = []

    def reset(self) -> None:
        self.step = 0
        self._chunks.clear()

    def push(self, chunk: Tensor) -> None:
        if chunk.ndim != 3 or chunk.shape[1] != self.horizon:
            raise ValueError("chunk must be [agents,horizon,action_dim]")
        self._chunks.append((self.step, chunk.detach().clone()))
        self._chunks = [
            (start, value)
            for start, value in self._chunks
            if self.step - start < self.horizon
        ]

    def current(self) -> Tensor:
        candidates = [
            chunk[:, self.step - start]
            for start, chunk in self._chunks
            if 0 <= self.step - start < self.horizon
        ]
        if not candidates:
            raise RuntimeError("temporal ensemble has no current prediction")
        ages = torch.arange(
            len(candidates) - 1,
            -1,
            -1,
            device=candidates[0].device,
            dtype=candidates[0].dtype,
        )
        weights = torch.exp(-self.decay * ages)
        weights = weights / weights.sum()
        return (torch.stack(candidates) * weights[:, None, None]).sum(dim=0)

    def advance(self) -> None:
        self.step += 1


class LatestChunkSelector:
    """Use action zero from the newest chunk when the policy replans every step."""

    def __init__(self, *, horizon: int) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.horizon = int(horizon)
        self.decay = None
        self.mode = "latest_chunk"
        self._latest: Tensor | None = None

    def reset(self) -> None:
        self._latest = None

    def push(self, chunk: Tensor) -> None:
        if chunk.ndim != 3 or chunk.shape[1] != self.horizon:
            raise ValueError("chunk must be [agents,horizon,action_dim]")
        self._latest = chunk.detach().clone()

    def current(self) -> Tensor:
        if self._latest is None:
            raise RuntimeError("latest chunk selector has no prediction")
        return self._latest[:, 0]

    def advance(self) -> None:
        # Inference pushes a freshly replanned chunk before every ``current`` call.
        return None


def build_chunk_aggregator(
    *,
    mode: str,
    horizon: int,
    decay: float = 0.01,
) -> TemporalChunkEnsembler | LatestChunkSelector:
    """Build the explicit inference ablation selected by the YAML config."""

    if mode == "temporal_ensemble":
        return TemporalChunkEnsembler(horizon=horizon, decay=decay)
    if mode == "latest_chunk":
        return LatestChunkSelector(horizon=horizon)
    raise ValueError("chunk aggregation must be temporal_ensemble or latest_chunk")


__all__ = [
    "DenseFeedForward",
    "LatestChunkSelector",
    "StaticRGBMoEACT",
    "StaticRGBMoEACTConfig",
    "TemporalChunkEnsembler",
    "Top2SparseMoE",
    "build_chunk_aggregator",
]
