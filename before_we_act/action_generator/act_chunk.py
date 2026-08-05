from __future__ import annotations

import math
import sys
import types
from typing import Mapping

import torch
from torch import nn


# The pinned ACT file imports IPython only to bind an unused debugging alias.
# Keep the copied algorithm byte-exact while satisfying that optional import.
if "IPython" not in sys.modules:
    try:
        __import__("IPython")
    except ModuleNotFoundError:
        stub = types.ModuleType("IPython")
        stub.embed = lambda *args, **kwargs: None
        sys.modules["IPython"] = stub

from before_we_act.upstream_components.act.detr.models.transformer import (  # noqa: E402
    Transformer,
    TransformerEncoder,
    TransformerEncoderLayer,
)


def sinusoid_table(length: int, width: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32)[:, None]
    divisor = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10_000.0) / width)
    )
    table = torch.zeros(length, width, dtype=torch.float32)
    table[:, 0::2] = torch.sin(position * divisor)
    table[:, 1::2] = torch.cos(position * divisor)
    return table[:, None]


class ACTActionChunkCore(nn.Module):
    """Official ACT CVAE/DETR action-chunk core behind the R12 codec."""

    def __init__(self, config: Mapping[str, object]) -> None:
        super().__init__()
        self.horizon = int(config["horizon"])
        self.action_dim = int(config["joint_action_dim"])
        self.hidden_dim = int(config["hidden_dim"])
        self.latent_dim = int(config["latent_dim"])
        self.kl_weight = float(config["kl_weight"])
        condition_tokens = int(config["condition_tokens"])
        self.condition_projection = nn.Linear(int(config["belief_dim"]), self.hidden_dim)
        self.condition_position = nn.Embedding(condition_tokens + 1, self.hidden_dim)
        self.query_embedding = nn.Embedding(self.horizon, self.hidden_dim)
        self.transformer = Transformer(
            d_model=self.hidden_dim,
            nhead=int(config["num_heads"]),
            num_encoder_layers=int(config["encoder_layers"]),
            num_decoder_layers=int(config["decoder_layers"]),
            dim_feedforward=int(config["dim_feedforward"]),
            dropout=float(config["dropout"]),
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=True,
        )
        posterior_layer = TransformerEncoderLayer(
            self.hidden_dim,
            int(config["num_heads"]),
            int(config["dim_feedforward"]),
            float(config["dropout"]),
            "relu",
            False,
        )
        self.posterior_encoder = TransformerEncoder(
            posterior_layer, int(config["encoder_layers"]), None
        )
        self.posterior_cls = nn.Embedding(1, self.hidden_dim)
        self.posterior_condition = nn.Linear(int(config["belief_dim"]), self.hidden_dim)
        self.posterior_action = nn.Linear(self.action_dim, self.hidden_dim)
        self.latent_stats = nn.Linear(self.hidden_dim, 2 * self.latent_dim)
        self.latent_projection = nn.Linear(self.latent_dim, self.hidden_dim)
        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.register_buffer(
            "posterior_position",
            sinusoid_table(self.horizon + 2, self.hidden_dim),
            persistent=True,
        )

    def _posterior(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = len(tokens)
        pooled = tokens.mean(dim=1)
        encoded = torch.cat(
            [
                self.posterior_cls.weight[None].expand(batch, -1, -1),
                self.posterior_condition(pooled)[:, None],
                self.posterior_action(actions),
            ],
            dim=1,
        ).transpose(0, 1)
        padded = ~mask.any(dim=-1)
        padding = torch.cat(
            [torch.zeros((batch, 2), dtype=torch.bool, device=tokens.device), padded], dim=1
        )
        output = self.posterior_encoder(
            encoded,
            src_key_padding_mask=padding,
            pos=self.posterior_position.to(encoded.dtype),
        )[0]
        mu, logvar = self.latent_stats(output).chunk(2, dim=-1)
        latent = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return latent, mu, logvar

    def _decode(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        source = torch.cat(
            [self.latent_projection(latent)[:, None], self.condition_projection(tokens)], dim=1
        )
        source_mask = torch.cat(
            [torch.ones((len(tokens), 1), dtype=torch.bool, device=tokens.device), token_mask], dim=1
        )
        hidden = self.transformer(
            source,
            ~source_mask,
            self.query_embedding.weight,
            self.condition_position.weight,
        )[0]
        return self.action_head(hidden)

    def training_loss(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        latent, mu, logvar = self._posterior(tokens, actions, mask)
        prediction = self._decode(tokens, token_mask, latent)
        l1 = ((prediction - actions).abs() * mask.to(prediction.dtype)).mean()
        kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())).sum(dim=-1).mean()
        return {"loss": l1 + self.kl_weight * kl, "l1": l1, "kl": kl}

    def sample(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del noise
        latent = torch.zeros((len(tokens), self.latent_dim), device=tokens.device, dtype=tokens.dtype)
        return self._decode(tokens, token_mask, latent)


def build_core(config: Mapping[str, object]) -> ACTActionChunkCore:
    expected = {
        "kind": "act_action_chunk_transformer",
        "hidden_dim": 256,
        "num_heads": 8,
        "encoder_layers": 4,
        "decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "latent_dim": 32,
        "kl_weight": 10.0,
        "condition_tokens": 21,
        "chunk_size": 100,
        "temporal_ensemble": True,
        "horizon": 100,
        "joint_action_dim": 32,
        "belief_dim": 96,
    }
    if dict(config) != expected:
        raise ValueError("R12-P2 ACT component config differs from the frozen official recipe")
    return ACTActionChunkCore(config)
