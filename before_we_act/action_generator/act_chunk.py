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
    """ACT CVAE/DETR chunk core with an R12-R4 current-condition plan prior."""

    def __init__(self, config: Mapping[str, object]) -> None:
        super().__init__()
        self.horizon = int(config["horizon"])
        self.action_dim = int(config["joint_action_dim"])
        self.hidden_dim = int(config["hidden_dim"])
        self.latent_dim = int(config["latent_dim"])
        self.plan_kl_weight = float(config["plan_kl_weight"])
        self.kl_balance_alpha = float(config["kl_balance_alpha"])
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
        proposal_hidden = int(config["plan_prior_hidden_dim"])
        self.plan_proposal = nn.Sequential(
            nn.LayerNorm(int(config["belief_dim"])),
            nn.Linear(int(config["belief_dim"]), proposal_hidden),
            nn.GELU(),
            nn.Linear(proposal_hidden, proposal_hidden),
            nn.GELU(),
            nn.Linear(proposal_hidden, 2 * self.latent_dim),
        )
        # R3 used z=0 at inference.  Preserve that behavior before Stage A
        # while retaining a trainable current-observation plan proposal.
        nn.init.zeros_(self.plan_proposal[-1].weight)
        nn.init.zeros_(self.plan_proposal[-1].bias)
        self.latent_projection = nn.Linear(self.latent_dim, self.hidden_dim)
        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.register_buffer(
            "posterior_position",
            sinusoid_table(self.horizon + 2, self.hidden_dim),
            persistent=True,
        )

    @staticmethod
    def _masked_pool(
        tokens: torch.Tensor, token_mask: torch.Tensor
    ) -> torch.Tensor:
        weights = token_mask[:, :, None].to(tokens.dtype)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

    def _proposal(
        self, tokens: torch.Tensor, token_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        statistics = self.plan_proposal(self._masked_pool(tokens, token_mask))
        mean, logvar = statistics.chunk(2, dim=-1)
        return mean, logvar.clamp(-10.0, 5.0)

    def _posterior(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = len(tokens)
        pooled = self._masked_pool(tokens, token_mask)
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
        logvar = logvar.clamp(-10.0, 5.0)
        latent = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return latent, mu, logvar

    @staticmethod
    def _gaussian_kl(
        posterior_mu: torch.Tensor,
        posterior_logvar: torch.Tensor,
        proposal_mu: torch.Tensor,
        proposal_logvar: torch.Tensor,
    ) -> torch.Tensor:
        return 0.5 * (
            proposal_logvar
            - posterior_logvar
            + (
                posterior_logvar.exp()
                + (posterior_mu - proposal_mu).square()
            )
            / proposal_logvar.exp()
            - 1.0
        ).sum(dim=-1).mean()

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
        latent, posterior_mu, posterior_logvar = self._posterior(
            tokens, token_mask, actions, mask
        )
        proposal_mu, proposal_logvar = self._proposal(tokens, token_mask)
        prediction = self._decode(tokens, token_mask, latent)
        l1 = (prediction - actions).abs().masked_select(mask).mean()
        # HULC/Dreamer-style KL balancing: the larger alpha puts most KL
        # gradient on the current-condition proposal while the recognition
        # distribution remains a stable action-aware target.
        prior_kl = self._gaussian_kl(
            posterior_mu.detach(),
            posterior_logvar.detach(),
            proposal_mu,
            proposal_logvar,
        )
        recognition_kl = self._gaussian_kl(
            posterior_mu,
            posterior_logvar,
            proposal_mu.detach(),
            proposal_logvar.detach(),
        )
        balanced_kl = (
            self.kl_balance_alpha * prior_kl
            + (1.0 - self.kl_balance_alpha) * recognition_kl
        )
        return {
            "loss": l1 + self.plan_kl_weight * balanced_kl,
            "l1": l1,
            "plan_kl": balanced_kl,
            "plan_prior_kl": prior_kl,
            "plan_recognition_kl": recognition_kl,
            "plan_proposal_std": torch.exp(0.5 * proposal_logvar).mean(),
            "plan_posterior_std": torch.exp(0.5 * posterior_logvar).mean(),
        }

    def sample(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del noise
        proposal_mean, _proposal_logvar = self._proposal(tokens, token_mask)
        return self._decode(tokens, token_mask, proposal_mean.to(tokens.dtype))


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
        "condition_tokens": 37,
        "chunk_size": 100,
        "temporal_ensemble": True,
        "plan_prior": "learned_current_condition",
        "plan_prior_hidden_dim": 256,
        "plan_kl_weight": 0.01,
        "kl_balance_alpha": 0.8,
        "horizon": 100,
        "joint_action_dim": 32,
        "belief_dim": 96,
    }
    if dict(config) != expected:
        raise ValueError("R12-R4 P2 ACT plan-prior component config differs")
    return ACTActionChunkCore(config)
