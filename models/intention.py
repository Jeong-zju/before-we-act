from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class IntentionConfig:
    slot_dim: int = 128
    slots_per_agent: int = 4
    plan_codebook_size: int = 32
    plan_latent_dim: int = 64
    model_dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1
    num_phases: int = 9
    ce_weight: float = 1.0
    residual_weight: float = 1.0
    kl_weight: float = 0.01
    consistency_weight: float = 0.2
    entropy_weight: float = 0.001


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class IntentionInferenceModel(nn.Module):
    def __init__(self, cfg: IntentionConfig):
        super().__init__()
        self.cfg = cfg

        self.slot_proj = nn.Linear(cfg.slot_dim, cfg.model_dim)
        self.ego_plan_code_embed = nn.Embedding(cfg.plan_codebook_size, cfg.model_dim)
        self.ego_plan_residual_proj = nn.Linear(cfg.plan_latent_dim, cfg.model_dim)
        self.ego_id_embed = nn.Embedding(2, cfg.model_dim)
        self.phase_embed = nn.Embedding(cfg.num_phases, cfg.model_dim)
        self.pose_proj = MLP(6, cfg.ffn_dim, cfg.model_dim, depth=3, dropout=cfg.dropout)

        self.type_embed = nn.Embedding(5, cfg.model_dim)  # slots, plan, id, phase, pose
        self.query = nn.Parameter(torch.randn(1, 1, cfg.model_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=cfg.num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(cfg.model_dim)

        self.code_head = MLP(cfg.model_dim, cfg.ffn_dim, cfg.plan_codebook_size, depth=3, dropout=cfg.dropout)
        self.mu_head = MLP(cfg.model_dim, cfg.ffn_dim, cfg.plan_latent_dim, depth=3, dropout=cfg.dropout)
        self.logvar_head = MLP(cfg.model_dim, cfg.ffn_dim, cfg.plan_latent_dim, depth=3, dropout=cfg.dropout)
        self.uncertainty_head = MLP(cfg.model_dim, cfg.ffn_dim, 1, depth=3, dropout=cfg.dropout)

    def forward(
        self,
        ego_slots: torch.Tensor,
        ego_plan_codes: torch.Tensor,
        ego_plan_residuals: torch.Tensor,
        ego_id: torch.Tensor,
        phase_history: torch.Tensor,
        rel_target_pose: torch.Tensor,
        object_rel_pose: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B = ego_slots.shape[0]

        slot_tokens = self.slot_proj(ego_slots) + self.type_embed.weight[0]
        plan_token = (
            self.ego_plan_code_embed(ego_plan_codes)
            + self.ego_plan_residual_proj(ego_plan_residuals)
            + self.type_embed.weight[1]
        ).unsqueeze(1)

        id_token = self.ego_id_embed(ego_id).unsqueeze(1) + self.type_embed.weight[2]

        phase_token = self.phase_embed(phase_history.clamp_min(0).clamp_max(self.cfg.num_phases - 1)).mean(dim=1, keepdim=True)
        phase_token = phase_token + self.type_embed.weight[3]

        pose = torch.cat([rel_target_pose, object_rel_pose], dim=-1)
        pose_token = self.pose_proj(pose).unsqueeze(1) + self.type_embed.weight[4]

        query = self.query.expand(B, -1, -1)

        x = torch.cat([query, slot_tokens, plan_token, id_token, phase_token, pose_token], dim=1)
        h = self.norm(self.encoder(x))
        q = h[:, 0]

        logvar = self.logvar_head(q).clamp(min=-3.0, max=3.0)

        return {
            "target_code_logits": self.code_head(q),
            "target_residual_mu": self.mu_head(q),
            "target_residual_logvar": logvar,
            "uncertainty": F.softplus(self.uncertainty_head(q).squeeze(-1)),
        }

    @torch.no_grad()
    def infer_teammate_plan(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        self.eval()
        out = self.forward(*args, **kwargs)
        out["target_code"] = out["target_code_logits"].argmax(dim=-1)
        out["target_residual"] = out["target_residual_mu"]
        return out


def gaussian_kl_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1).mean()


def gaussian_nll(target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # Stable proxy for residual regression.
    # The previous pure Gaussian NLL can become strongly negative by driving logvar
    # to the lower clamp, which makes the total loss hard to interpret and yields
    # over-confident residual uncertainty.
    residual_mse = F.mse_loss(mu, target)
    variance_reg = 0.01 * logvar.pow(2).mean()
    return residual_mse + variance_reg


def categorical_entropy(logits: torch.Tensor) -> torch.Tensor:
    p = logits.softmax(dim=-1)
    logp = logits.log_softmax(dim=-1)
    return -(p * logp).sum(dim=-1).mean()


def compute_intention_losses(
    model: IntentionInferenceModel,
    batch: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    consistency: Dict[str, torch.Tensor] | None = None,
) -> Dict[str, torch.Tensor]:
    cfg = model.cfg

    out = model(
        ego_slots=targets["ego_slots"],
        ego_plan_codes=targets["ego_plan_codes"],
        ego_plan_residuals=targets["ego_plan_residuals"],
        ego_id=batch["ego_id"],
        phase_history=batch["phase_history"],
        rel_target_pose=batch["rel_target_pose"],
        object_rel_pose=batch["object_rel_pose"],
    )

    loss_ce = F.cross_entropy(out["target_code_logits"], targets["target_plan_codes"])
    loss_residual = gaussian_nll(
        targets["target_plan_residuals"],
        out["target_residual_mu"],
        out["target_residual_logvar"],
    )
    loss_kl = gaussian_kl_to_standard_normal(out["target_residual_mu"], out["target_residual_logvar"])
    entropy = categorical_entropy(out["target_code_logits"])

    if consistency is not None:
        loss_consistency = consistency["loss_consistency"]
    else:
        loss_consistency = out["target_residual_mu"].sum() * 0.0

    loss = (
        cfg.ce_weight * loss_ce
        + cfg.residual_weight * loss_residual
        + cfg.kl_weight * loss_kl
        + cfg.consistency_weight * loss_consistency
        - cfg.entropy_weight * entropy
    )

    pred_code = out["target_code_logits"].argmax(dim=-1)
    code_acc = (pred_code == targets["target_plan_codes"]).float().mean()

    residual_mse = F.mse_loss(out["target_residual_mu"], targets["target_plan_residuals"])

    return {
        "loss": loss,
        "loss_ce": loss_ce.detach(),
        "loss_residual": loss_residual.detach(),
        "loss_kl": loss_kl.detach(),
        "loss_consistency": loss_consistency.detach(),
        "entropy": entropy.detach(),
        "code_acc": code_acc.detach(),
        "residual_mse": residual_mse.detach(),
        "pred_code": pred_code.detach(),
        "pred_residual": out["target_residual_mu"].detach(),
        "uncertainty": out["uncertainty"].detach(),
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def make_config_from_args(args) -> IntentionConfig:
    return IntentionConfig(
        slot_dim=args.slot_dim,
        slots_per_agent=args.slots_per_agent,
        plan_codebook_size=args.plan_codebook_size,
        plan_latent_dim=args.plan_latent_dim,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        ce_weight=args.ce_weight,
        residual_weight=args.residual_weight,
        kl_weight=args.kl_weight,
        consistency_weight=args.consistency_weight,
        entropy_weight=args.entropy_weight,
    )
