from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        width = in_dim
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = hidden_dim
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class EgoLocalWAMConfig:
    horizon: int = 16
    slots_per_agent: int = 4
    slot_dim: int = 128
    plan_codebook_size: int = 64
    plan_latent_dim: int = 64
    action_dim_per_agent: int = 4
    model_dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1
    hypothesis_weight_epsilon: float = 1e-8
    return_quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    failure_classes: int = 10


class EgoLocalWAM(nn.Module):
    """Ego-canonical dynamics conditioned on uncertain teammate plans.

    The state input contains only the ego robot's four belief slots.  There is
    intentionally no argument or projection for teammate-private slots.  Plan
    tensors are ego-first: index 0 is the ego plan and index 1 is one teammate
    plan hypothesis.  The returned joint action follows the same convention.
    Multiple teammate hypotheses are evaluated by repeating the ego state and
    passing one hypothesis per row.  ``teammate_hypothesis_weight`` is accepted
    only for shape/range validation and caller compatibility; it is not a model
    token.  Posterior aggregation belongs in the free-energy utilities rather
    than inside the conditional dynamics model.
    """

    def __init__(self, cfg: EgoLocalWAMConfig):
        super().__init__()
        if cfg.slots_per_agent <= 0:
            raise ValueError("slots_per_agent must be positive")
        if cfg.plan_codebook_size < 2:
            raise ValueError("plan_codebook_size must be at least 2")
        if cfg.model_dim % cfg.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.cfg = cfg

        self.slot_proj = nn.Linear(cfg.slot_dim, cfg.model_dim)
        self.slot_id_embed = nn.Embedding(cfg.slots_per_agent, cfg.model_dim)
        self.plan_code_embed = nn.Embedding(cfg.plan_codebook_size, cfg.model_dim)
        self.plan_residual_proj = nn.Linear(cfg.plan_latent_dim, cfg.model_dim)
        self.plan_role_embed = nn.Embedding(2, cfg.model_dim)  # ego, teammate hypothesis
        # Posterior weights belong to the outer E_q[G] aggregation.  They are
        # deliberately not encoded into p(y | belief, ego_plan, teammate_plan),
        # otherwise identical physical hypotheses would change merely because
        # the observer assigned them a different probability.
        self.token_type_embed = nn.Embedding(3, cfg.model_dim)  # slot, plan, future query
        self.time_embed = nn.Embedding(cfg.horizon, cfg.model_dim)
        self.future_queries = nn.Parameter(torch.randn(1, cfg.horizon, cfg.model_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=cfg.num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(cfg.model_dim)

        self.ego_slot_head = MLP(
            cfg.model_dim,
            cfg.ffn_dim,
            cfg.slots_per_agent * cfg.slot_dim,
            depth=3,
            dropout=cfg.dropout,
        )
        self.joint_action_head = MLP(
            cfg.model_dim,
            cfg.ffn_dim,
            2 * cfg.action_dim_per_agent,
            depth=3,
            dropout=cfg.dropout,
        )
        # Per-step physical outcome: contact logit, force proxy, task progress.
        self.physical_outcome_head = MLP(
            cfg.model_dim,
            cfg.ffn_dim,
            3,
            depth=3,
            dropout=cfg.dropout,
        )
        self.step_reward_head = MLP(
            cfg.model_dim, cfg.ffn_dim, 1, depth=3, dropout=cfg.dropout
        )
        self.return_quantile_head = MLP(
            cfg.model_dim,
            cfg.ffn_dim,
            len(cfg.return_quantiles),
            depth=3,
            dropout=cfg.dropout,
        )
        self.terminal_outcome_head = MLP(
            cfg.model_dim,
            cfg.ffn_dim,
            1 + cfg.failure_classes + 3,  # success, failure class, collision, force, time
            depth=3,
            dropout=cfg.dropout,
        )

    def _validate_inputs(
        self,
        ego_slots: torch.Tensor,
        plan_codes: torch.Tensor,
        plan_residuals: torch.Tensor,
        teammate_hypothesis_weight: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg
        if ego_slots.ndim != 3 or tuple(ego_slots.shape[1:]) != (cfg.slots_per_agent, cfg.slot_dim):
            raise ValueError(
                "ego_slots must have shape "
                f"[B, {cfg.slots_per_agent}, {cfg.slot_dim}], got {tuple(ego_slots.shape)}"
            )
        B = ego_slots.shape[0]
        if plan_codes.shape != (B, 2):
            raise ValueError(f"plan_codes must be ego-first [B, 2], got {tuple(plan_codes.shape)}")
        if plan_residuals.shape != (B, 2, cfg.plan_latent_dim):
            raise ValueError(
                f"plan_residuals must have shape [B, 2, {cfg.plan_latent_dim}], "
                f"got {tuple(plan_residuals.shape)}"
            )
        if (plan_codes < 0).any() or (plan_codes >= cfg.plan_codebook_size).any():
            raise ValueError(f"plan_codes must be in [0, {cfg.plan_codebook_size - 1}]")

        weights = teammate_hypothesis_weight.to(device=ego_slots.device, dtype=ego_slots.dtype)
        if weights.numel() != B:
            raise ValueError("teammate_hypothesis_weight must contain one scalar per batch item")
        weights = weights.reshape(B)
        if not torch.isfinite(weights).all() or (weights < 0).any() or (weights > 1.0 + 1e-6).any():
            raise ValueError("teammate_hypothesis_weight must be finite and in [0, 1]")
        return weights

    def build_tokens(
        self,
        ego_slots: torch.Tensor,
        plan_codes: torch.Tensor,
        plan_residuals: torch.Tensor,
        teammate_hypothesis_weight: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg
        self._validate_inputs(
            ego_slots,
            plan_codes,
            plan_residuals,
            teammate_hypothesis_weight,
        )
        B = ego_slots.shape[0]

        slot_ids = torch.arange(cfg.slots_per_agent, device=ego_slots.device)
        slot_tokens = (
            self.slot_proj(ego_slots)
            + self.slot_id_embed(slot_ids).unsqueeze(0)
            + self.token_type_embed.weight[0]
        )

        role_ids = torch.arange(2, device=ego_slots.device)
        plan_tokens = (
            self.plan_code_embed(plan_codes.long())
            + self.plan_residual_proj(plan_residuals)
            + self.plan_role_embed(role_ids).unsqueeze(0)
            + self.token_type_embed.weight[1]
        )

        time_ids = torch.arange(cfg.horizon, device=ego_slots.device)
        future_tokens = (
            self.future_queries.expand(B, -1, -1)
            + self.time_embed(time_ids).unsqueeze(0)
            + self.token_type_embed.weight[2]
        )
        return torch.cat([slot_tokens, plan_tokens, future_tokens], dim=1)

    def forward(
        self,
        ego_slots: torch.Tensor,
        plan_codes: torch.Tensor,
        plan_residuals: torch.Tensor,
        teammate_hypothesis_weight: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        tokens = self.build_tokens(
            ego_slots,
            plan_codes,
            plan_residuals,
            teammate_hypothesis_weight,
        )
        future = self.norm(self.encoder(tokens))[:, -cfg.horizon:]

        pred_ego_slots = self.ego_slot_head(future).reshape(
            -1,
            cfg.horizon,
            cfg.slots_per_agent,
            cfg.slot_dim,
        )
        pred_actions = self.joint_action_head(future)
        physical = self.physical_outcome_head(future)
        pred_contact_logits = physical[..., 0]
        pred_force = physical[..., 1]
        pred_progress = physical[..., 2]
        pred_step_reward = self.step_reward_head(future).squeeze(-1)
        pred_return_quantiles = self.return_quantile_head(future[:, -1])
        terminal = self.terminal_outcome_head(future[:, -1])

        return {
            "pred_ego_slots": pred_ego_slots,
            "pred_slots": pred_ego_slots,
            "pred_actions": pred_actions,
            "pred_joint_actions": pred_actions,
            "pred_physical_outcome": physical,
            "pred_contact_logits": pred_contact_logits,
            "pred_force": pred_force,
            "pred_progress": pred_progress,
            "pred_step_reward": pred_step_reward,
            "pred_return_quantiles": pred_return_quantiles,
            "pred_success_logits": terminal[:, 0],
            "pred_failure_logits": terminal[:, 1 : 1 + cfg.failure_classes],
            "pred_collision_logits": terminal[:, 1 + cfg.failure_classes],
            "pred_force_violation_logits": terminal[:, 2 + cfg.failure_classes],
            "pred_completion_time": F.softplus(
                terminal[:, 3 + cfg.failure_classes]
            ),
        }

    @torch.no_grad()
    def rollout(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        self.eval()
        return self.forward(*args, **kwargs)


@dataclass
class LocalIntentionConfig:
    slots_per_agent: int = 4
    slot_dim: int = 128
    plan_codebook_size: int = 64
    plan_latent_dim: int = 64
    message_metadata_dim: int = 4
    model_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1
    min_logvar: float = -6.0
    max_logvar: float = 3.0
    entropy_uncertainty_weight: float = 1.0
    variance_uncertainty_weight: float = 1.0


class LocalIntentionPosterior(nn.Module):
    """Infer teammate-plan hypotheses from ego-local belief only.

    ``received_message_metadata`` represents channel/envelope facts such as
    availability, age, delay, or reliability.  The API intentionally has no
    teammate-private slots and no true/reply plan-code argument.  Uncertainty
    is a deterministic posterior statistic (categorical entropy plus mixture
    residual variance), not an untrained scalar head.
    """

    def __init__(self, cfg: LocalIntentionConfig):
        super().__init__()
        if cfg.plan_codebook_size < 2:
            raise ValueError("plan_codebook_size must be at least 2")
        if cfg.model_dim % cfg.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.cfg = cfg

        self.slot_proj = nn.Linear(cfg.slot_dim, cfg.model_dim)
        self.slot_id_embed = nn.Embedding(cfg.slots_per_agent, cfg.model_dim)
        self.ego_plan_code_embed = nn.Embedding(cfg.plan_codebook_size, cfg.model_dim)
        self.ego_plan_residual_proj = nn.Linear(cfg.plan_latent_dim, cfg.model_dim)
        self.agent_id_embed = nn.Embedding(2, cfg.model_dim)
        self.message_metadata_proj = MLP(
            cfg.message_metadata_dim,
            cfg.ffn_dim,
            cfg.model_dim,
            depth=2,
            dropout=cfg.dropout,
        )
        # query, slots, ego plan, agent id, message metadata
        self.type_embed = nn.Embedding(5, cfg.model_dim)
        self.query = nn.Parameter(torch.randn(1, 1, cfg.model_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=cfg.num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(cfg.model_dim)

        self.code_head = MLP(
            cfg.model_dim,
            cfg.ffn_dim,
            cfg.plan_codebook_size,
            depth=3,
            dropout=cfg.dropout,
        )
        conditional_dim = cfg.plan_codebook_size * cfg.plan_latent_dim
        self.residual_mu_head = MLP(
            cfg.model_dim,
            cfg.ffn_dim,
            conditional_dim,
            depth=3,
            dropout=cfg.dropout,
        )
        self.residual_logvar_head = MLP(
            cfg.model_dim,
            cfg.ffn_dim,
            conditional_dim,
            depth=3,
            dropout=cfg.dropout,
        )

    def _validate_inputs(
        self,
        ego_slots: torch.Tensor,
        ego_plan_code: torch.Tensor,
        ego_plan_residual: torch.Tensor,
        agent_id: torch.Tensor,
        received_message_metadata: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        if ego_slots.ndim != 3 or tuple(ego_slots.shape[1:]) != (cfg.slots_per_agent, cfg.slot_dim):
            raise ValueError(
                f"ego_slots must have shape [B, {cfg.slots_per_agent}, {cfg.slot_dim}]"
            )
        B = ego_slots.shape[0]
        code = ego_plan_code.to(device=ego_slots.device, dtype=torch.long).reshape(-1)
        ids = agent_id.to(device=ego_slots.device, dtype=torch.long).reshape(-1)
        if code.shape != (B,) or ids.shape != (B,):
            raise ValueError("ego_plan_code and agent_id must each have shape [B]")
        if (code < 0).any() or (code >= cfg.plan_codebook_size).any():
            raise ValueError(f"ego_plan_code must be in [0, {cfg.plan_codebook_size - 1}]")
        if (ids < 0).any() or (ids > 1).any():
            raise ValueError("agent_id must be 0 or 1")
        if ego_plan_residual.shape != (B, cfg.plan_latent_dim):
            raise ValueError(f"ego_plan_residual must have shape [B, {cfg.plan_latent_dim}]")
        if received_message_metadata.shape != (B, cfg.message_metadata_dim):
            raise ValueError(
                "received_message_metadata must have shape "
                f"[B, {cfg.message_metadata_dim}]"
            )
        if not torch.isfinite(received_message_metadata).all():
            raise ValueError("received_message_metadata must be finite")
        return code, ids

    def forward(
        self,
        ego_slots: torch.Tensor,
        ego_plan_code: torch.Tensor,
        ego_plan_residual: torch.Tensor,
        agent_id: torch.Tensor,
        received_message_metadata: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        code, ids = self._validate_inputs(
            ego_slots,
            ego_plan_code,
            ego_plan_residual,
            agent_id,
            received_message_metadata,
        )
        B = ego_slots.shape[0]
        slot_ids = torch.arange(cfg.slots_per_agent, device=ego_slots.device)

        query = self.query.expand(B, -1, -1) + self.type_embed.weight[0]
        slot_tokens = (
            self.slot_proj(ego_slots)
            + self.slot_id_embed(slot_ids).unsqueeze(0)
            + self.type_embed.weight[1]
        )
        plan_token = (
            self.ego_plan_code_embed(code)
            + self.ego_plan_residual_proj(ego_plan_residual)
            + self.type_embed.weight[2]
        ).unsqueeze(1)
        id_token = (self.agent_id_embed(ids) + self.type_embed.weight[3]).unsqueeze(1)
        metadata_token = (
            self.message_metadata_proj(received_message_metadata) + self.type_embed.weight[4]
        ).unsqueeze(1)

        encoded = self.norm(
            self.encoder(torch.cat([query, slot_tokens, plan_token, id_token, metadata_token], dim=1))
        )
        posterior_token = encoded[:, 0]

        code_logits = self.code_head(posterior_token)
        code_probs = code_logits.softmax(dim=-1)
        residual_mu_by_code = self.residual_mu_head(posterior_token).reshape(
            B,
            cfg.plan_codebook_size,
            cfg.plan_latent_dim,
        )
        residual_logvar_by_code = self.residual_logvar_head(posterior_token).reshape(
            B,
            cfg.plan_codebook_size,
            cfg.plan_latent_dim,
        ).clamp(min=cfg.min_logvar, max=cfg.max_logvar)

        weights = code_probs.unsqueeze(-1)
        residual_var_by_code = residual_logvar_by_code.exp()
        mixture_mu = (weights * residual_mu_by_code).sum(dim=1)
        mixture_second_moment = (
            weights * (residual_var_by_code + residual_mu_by_code.pow(2))
        ).sum(dim=1)
        mixture_variance = (mixture_second_moment - mixture_mu.pow(2)).clamp_min(1e-8)
        mixture_logvar = mixture_variance.log()

        code_entropy = -(code_probs * code_probs.clamp_min(1e-8).log()).sum(dim=-1)
        normalized_entropy = code_entropy / math.log(cfg.plan_codebook_size)
        residual_variance = mixture_variance.mean(dim=-1)
        uncertainty = (
            cfg.entropy_uncertainty_weight * normalized_entropy
            + cfg.variance_uncertainty_weight * residual_variance
        )

        return {
            "target_code_logits": code_logits,
            "code_logits": code_logits,
            "code_probabilities": code_probs,
            "residual_mu_by_code": residual_mu_by_code,
            "residual_logvar_by_code": residual_logvar_by_code,
            "target_residual_mu": mixture_mu,
            "target_residual_logvar": mixture_logvar,
            "residual_variance": residual_variance,
            "code_entropy": code_entropy,
            "normalized_code_entropy": normalized_entropy,
            "uncertainty": uncertainty,
        }

    @torch.no_grad()
    def topk_hypotheses(
        self,
        ego_slots: torch.Tensor,
        ego_plan_code: torch.Tensor,
        ego_plan_residual: torch.Tensor,
        agent_id: torch.Tensor,
        received_message_metadata: torch.Tensor,
        k: int,
    ) -> Dict[str, torch.Tensor]:
        out = self.forward(
            ego_slots,
            ego_plan_code,
            ego_plan_residual,
            agent_id,
            received_message_metadata,
        )
        k = min(max(1, int(k)), self.cfg.plan_codebook_size)
        weights, codes = out["code_probabilities"].topk(k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        gather_index = codes.unsqueeze(-1).expand(-1, -1, self.cfg.plan_latent_dim)
        residual_mu = out["residual_mu_by_code"].gather(1, gather_index)
        residual_logvar = out["residual_logvar_by_code"].gather(1, gather_index)
        return {
            "plan_codes": codes,
            "plan_residual_mu": residual_mu,
            "plan_residual_logvar": residual_logvar,
            "hypothesis_weights": weights,
            "uncertainty": out["uncertainty"],
        }
