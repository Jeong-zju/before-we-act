from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class WAMConfig:
    horizon: int = 16
    num_agents: int = 2
    slots_per_agent: int = 4
    slot_dim: int = 128
    plan_codebook_size: int = 32
    plan_latent_dim: int = 64
    action_dim: int = 8
    model_dim: int = 1024
    num_layers: int = 16
    num_heads: int = 16
    ffn_dim: int = 4096
    dropout: float = 0.1
    use_checkpoint: bool = True
    slot_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    contact_loss_weight: float = 0.2
    force_loss_weight: float = 0.2
    progress_loss_weight: float = 0.5
    smooth_action_weight: float = 0.02


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LatentWorldActionModel(nn.Module):
    def __init__(self, cfg: WAMConfig):
        super().__init__()
        self.cfg = cfg

        self.slot_proj = nn.Linear(cfg.slot_dim, cfg.model_dim)
        self.plan_code_embed = nn.Embedding(cfg.plan_codebook_size, cfg.model_dim)
        self.plan_residual_proj = nn.Linear(cfg.plan_latent_dim, cfg.model_dim)
        self.plan_type_embed = nn.Parameter(torch.randn(1, cfg.num_agents, cfg.model_dim) * 0.02)

        self.agent_embed = nn.Embedding(cfg.num_agents, cfg.model_dim)
        self.slot_embed = nn.Embedding(cfg.slots_per_agent, cfg.model_dim)
        self.token_type_embed = nn.Embedding(3, cfg.model_dim)  # 0 slot, 1 plan, 2 future query
        self.time_embed = nn.Embedding(cfg.horizon, cfg.model_dim)
        self.future_queries = nn.Parameter(torch.randn(1, cfg.horizon, cfg.model_dim) * 0.02)

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=cfg.model_dim,
                    nhead=cfg.num_heads,
                    dim_feedforward=cfg.ffn_dim,
                    dropout=cfg.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(cfg.model_dim)

        self.slot_head = MLP(cfg.model_dim, cfg.ffn_dim, cfg.num_agents * cfg.slots_per_agent * cfg.slot_dim, depth=3, dropout=cfg.dropout)
        self.action_head = MLP(cfg.model_dim, cfg.ffn_dim, cfg.action_dim, depth=3, dropout=cfg.dropout)
        self.contact_head = MLP(cfg.model_dim, cfg.ffn_dim, 1, depth=3, dropout=cfg.dropout)
        self.force_head = MLP(cfg.model_dim, cfg.ffn_dim, 1, depth=3, dropout=cfg.dropout)
        self.progress_head = MLP(cfg.model_dim, cfg.ffn_dim, 1, depth=3, dropout=cfg.dropout)

    def build_tokens(self, current_slots: torch.Tensor, plan_codes: torch.Tensor, plan_residuals: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B = current_slots.shape[0]

        # current_slots: [B, A, S, slot_dim]
        A = cfg.num_agents
        S = cfg.slots_per_agent

        slot_tokens = self.slot_proj(current_slots)
        agent_ids = torch.arange(A, device=current_slots.device).view(1, A, 1).expand(B, A, S)
        slot_ids = torch.arange(S, device=current_slots.device).view(1, 1, S).expand(B, A, S)
        slot_tokens = slot_tokens + self.agent_embed(agent_ids) + self.slot_embed(slot_ids) + self.token_type_embed.weight[0]
        slot_tokens = slot_tokens.reshape(B, A * S, cfg.model_dim)

        # plan_codes: [B, A], plan_residuals: [B, A, D]
        plan_tokens = self.plan_code_embed(plan_codes.clamp_min(0).clamp_max(cfg.plan_codebook_size - 1))
        plan_tokens = plan_tokens + self.plan_residual_proj(plan_residuals) + self.plan_type_embed[:, :A, :] + self.token_type_embed.weight[1]

        time_ids = torch.arange(cfg.horizon, device=current_slots.device)
        query_tokens = self.future_queries.expand(B, -1, -1) + self.time_embed(time_ids).unsqueeze(0) + self.token_type_embed.weight[2]

        return torch.cat([slot_tokens, plan_tokens, query_tokens], dim=1)

    def forward(self, current_slots: torch.Tensor, plan_codes: torch.Tensor, plan_residuals: torch.Tensor) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        x = self.build_tokens(current_slots, plan_codes, plan_residuals)

        for layer in self.layers:
            if self.training and cfg.use_checkpoint:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        x = self.norm(x)
        future_tokens = x[:, -cfg.horizon:, :]

        pred_slots = self.slot_head(future_tokens).reshape(
            -1,
            cfg.horizon,
            cfg.num_agents,
            cfg.slots_per_agent,
            cfg.slot_dim,
        )
        pred_actions = self.action_head(future_tokens)
        pred_contact_logits = self.contact_head(future_tokens).squeeze(-1)
        pred_force = self.force_head(future_tokens).squeeze(-1)
        pred_progress = self.progress_head(future_tokens).squeeze(-1)

        return {
            "pred_slots": pred_slots,
            "pred_actions": pred_actions,
            "pred_contact_logits": pred_contact_logits,
            "pred_force": pred_force,
            "pred_progress": pred_progress,
        }

    @torch.no_grad()
    def rollout(self, current_slots: torch.Tensor, plan_codes: torch.Tensor, plan_residuals: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        return self.forward(current_slots, plan_codes, plan_residuals)


def compute_wam_losses(model: LatentWorldActionModel, batch: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cfg = model.cfg

    out = model(targets["current_slots"], targets["plan_codes"], targets["plan_residuals"])

    loss_slots = F.mse_loss(out["pred_slots"], targets["future_slots"])
    loss_actions = F.mse_loss(out["pred_actions"], batch["future_actions"])
    loss_contact = F.binary_cross_entropy_with_logits(out["pred_contact_logits"], batch["target_contact"])
    loss_force = F.smooth_l1_loss(out["pred_force"], batch["target_force"])
    loss_progress = F.mse_loss(out["pred_progress"], batch["target_progress"])

    if out["pred_actions"].shape[1] > 1:
        diff = out["pred_actions"][:, 1:] - out["pred_actions"][:, :-1]
        loss_smooth = diff.pow(2).mean()
    else:
        loss_smooth = out["pred_actions"].sum() * 0.0

    loss = (
        cfg.slot_loss_weight * loss_slots
        + cfg.action_loss_weight * loss_actions
        + cfg.contact_loss_weight * loss_contact
        + cfg.force_loss_weight * loss_force
        + cfg.progress_loss_weight * loss_progress
        + cfg.smooth_action_weight * loss_smooth
    )

    return {
        "loss": loss,
        "loss_slots": loss_slots.detach(),
        "loss_actions": loss_actions.detach(),
        "loss_contact": loss_contact.detach(),
        "loss_force": loss_force.detach(),
        "loss_progress": loss_progress.detach(),
        "loss_smooth": loss_smooth.detach(),
        "pred_slots": out["pred_slots"].detach(),
        "pred_actions": out["pred_actions"].detach(),
        "pred_contact_prob": torch.sigmoid(out["pred_contact_logits"]).detach(),
        "pred_force": out["pred_force"].detach(),
        "pred_progress": out["pred_progress"].detach(),
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def make_config_from_args(args) -> WAMConfig:
    return WAMConfig(
        horizon=args.horizon,
        num_agents=2,
        slots_per_agent=args.slots_per_agent,
        slot_dim=args.slot_dim,
        plan_codebook_size=args.plan_codebook_size,
        plan_latent_dim=args.plan_latent_dim,
        action_dim=8,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        use_checkpoint=bool(args.use_checkpoint),
        slot_loss_weight=args.slot_loss_weight,
        action_loss_weight=args.action_loss_weight,
        contact_loss_weight=args.contact_loss_weight,
        force_loss_weight=args.force_loss_weight,
        progress_loss_weight=args.progress_loss_weight,
        smooth_action_weight=args.smooth_action_weight,
    )
