from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SlotEncoderConfig:
    history: int = 8
    local_dim: int = 17
    slot_dim: int = 128
    hidden_dim: int = 256
    num_object_slots: int = 2
    num_layers: int = 4
    num_heads: int = 4
    num_phases: int = 9
    plan_codebook_size: int = 32
    dropout: float = 0.1
    pose_weight: float = 1.0
    object_weight: float = 1.0
    contact_weight: float = 0.2
    force_weight: float = 0.1
    phase_weight: float = 0.2
    plan_weight: float = 0.2


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


class AgentObjectSlotEncoder(nn.Module):
    def __init__(self, cfg: SlotEncoderConfig):
        super().__init__()
        self.cfg = cfg

        self.history_encoder = MLP(cfg.history * cfg.local_dim, cfg.hidden_dim, cfg.slot_dim, depth=3, dropout=cfg.dropout)
        self.agent_id_embed = nn.Embedding(2, cfg.slot_dim)
        self.phase_embed = nn.Embedding(cfg.num_phases, cfg.slot_dim)

        self.self_query = nn.Parameter(torch.randn(1, 1, cfg.slot_dim) * 0.02)
        self.other_query = nn.Parameter(torch.randn(1, 1, cfg.slot_dim) * 0.02)
        self.object_queries = nn.Parameter(torch.randn(1, cfg.num_object_slots, cfg.slot_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.slot_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.slot_transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=cfg.num_layers,
            enable_nested_tensor=False,
        )
        self.out_norm = nn.LayerNorm(cfg.slot_dim)

        self.self_pose_head = MLP(cfg.slot_dim, cfg.hidden_dim, 3, depth=3, dropout=cfg.dropout)
        self.other_pose_head = MLP(cfg.slot_dim, cfg.hidden_dim, 3, depth=3, dropout=cfg.dropout)
        self.object_pose_head = MLP(cfg.slot_dim, cfg.hidden_dim, 3, depth=3, dropout=cfg.dropout)
        self.contact_head = MLP(cfg.slot_dim * (2 + cfg.num_object_slots), cfg.hidden_dim, 1, depth=3, dropout=cfg.dropout)
        self.force_head = MLP(cfg.slot_dim * (2 + cfg.num_object_slots), cfg.hidden_dim, 1, depth=3, dropout=cfg.dropout)
        self.phase_head = MLP(cfg.slot_dim * (2 + cfg.num_object_slots), cfg.hidden_dim, cfg.num_phases, depth=3, dropout=cfg.dropout)
        self.plan_head = MLP(cfg.slot_dim, cfg.hidden_dim, cfg.plan_codebook_size, depth=3, dropout=cfg.dropout)

    def forward(self, local_history: torch.Tensor, agent_id: torch.Tensor, phase_history: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        B = local_history.shape[0]

        h = local_history.reshape(B, -1)
        context = self.history_encoder(h)
        context = context + self.agent_id_embed(agent_id)

        if phase_history is not None:
            phase_context = self.phase_embed(phase_history.clamp_min(0).clamp_max(cfg.num_phases - 1)).mean(dim=1)
            context = context + phase_context

        queries = torch.cat(
            [
                self.self_query.expand(B, -1, -1),
                self.other_query.expand(B, -1, -1),
                self.object_queries.expand(B, -1, -1),
            ],
            dim=1,
        )

        tokens = queries + context.unsqueeze(1)
        slots = self.out_norm(self.slot_transformer(tokens))

        self_slot = slots[:, 0]
        other_slot = slots[:, 1]
        object_slots = slots[:, 2:]
        object_slot = object_slots.mean(dim=1)

        flat = slots.reshape(B, -1)

        return {
            "slots": slots,
            "self_slot": self_slot,
            "other_slot": other_slot,
            "object_slots": object_slots,
            "pred_self_pose": self.self_pose_head(self_slot),
            "pred_other_rel_pose": self.other_pose_head(other_slot),
            "pred_object_rel_pose": self.object_pose_head(object_slot),
            "pred_contact_logit": self.contact_head(flat).squeeze(-1),
            "pred_force": self.force_head(flat).squeeze(-1),
            "pred_phase_logits": self.phase_head(flat),
            "pred_plan_logits": self.plan_head(self_slot),
        }

    @torch.no_grad()
    def encode_slots(self, local_history: torch.Tensor, agent_id: torch.Tensor, phase_history: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        self.eval()
        return self.forward(local_history, agent_id, phase_history)


def compute_slot_losses(
    model: AgentObjectSlotEncoder,
    batch: Dict[str, torch.Tensor],
    norm_stats: Dict[str, torch.Tensor] | None = None,
) -> Dict[str, torch.Tensor]:
    cfg = model.cfg

    local = batch["local_history"]
    self_pose = batch["self_pose"]
    other_rel = batch["other_rel_pose"]
    object_rel = batch["object_rel_pose"]

    if norm_stats is not None:
        local = (local - norm_stats["local_mean"].view(1, 1, -1)) / norm_stats["local_std"].view(1, 1, -1)
        self_pose_tgt = (self_pose - norm_stats["self_pose_mean"].view(1, -1)) / norm_stats["self_pose_std"].view(1, -1)
        other_rel_tgt = (other_rel - norm_stats["other_rel_mean"].view(1, -1)) / norm_stats["other_rel_std"].view(1, -1)
        object_rel_tgt = (object_rel - norm_stats["object_rel_mean"].view(1, -1)) / norm_stats["object_rel_std"].view(1, -1)
    else:
        self_pose_tgt = self_pose
        other_rel_tgt = other_rel
        object_rel_tgt = object_rel

    out = model(local, batch["agent_id"], batch.get("phase_history", None))

    loss_self_pose = F.mse_loss(out["pred_self_pose"], self_pose_tgt)
    loss_other_pose = F.mse_loss(out["pred_other_rel_pose"], other_rel_tgt)
    loss_object_pose = F.mse_loss(out["pred_object_rel_pose"], object_rel_tgt)
    loss_contact = F.binary_cross_entropy_with_logits(out["pred_contact_logit"], batch["contact"])
    loss_force = F.smooth_l1_loss(out["pred_force"], batch["force_proxy"])
    loss_phase = F.cross_entropy(out["pred_phase_logits"], batch["phase"])

    plan_token = batch["plan_token"]
    valid_plan = plan_token >= 0
    if valid_plan.any():
        loss_plan = F.cross_entropy(out["pred_plan_logits"][valid_plan], plan_token[valid_plan].clamp_max(cfg.plan_codebook_size - 1))
    else:
        loss_plan = out["pred_plan_logits"].sum() * 0.0

    loss = (
        cfg.pose_weight * (loss_self_pose + loss_other_pose)
        + cfg.object_weight * loss_object_pose
        + cfg.contact_weight * loss_contact
        + cfg.force_weight * loss_force
        + cfg.phase_weight * loss_phase
        + cfg.plan_weight * loss_plan
    )

    return {
        "loss": loss,
        "loss_self_pose": loss_self_pose.detach(),
        "loss_other_pose": loss_other_pose.detach(),
        "loss_object_pose": loss_object_pose.detach(),
        "loss_contact": loss_contact.detach(),
        "loss_force": loss_force.detach(),
        "loss_phase": loss_phase.detach(),
        "loss_plan": loss_plan.detach(),
        "pred_self_pose": out["pred_self_pose"].detach(),
        "pred_other_rel_pose": out["pred_other_rel_pose"].detach(),
        "pred_object_rel_pose": out["pred_object_rel_pose"].detach(),
        "pred_contact_prob": torch.sigmoid(out["pred_contact_logit"]).detach(),
        "pred_force": out["pred_force"].detach(),
        "pred_phase": out["pred_phase_logits"].argmax(dim=-1).detach(),
        "pred_plan": out["pred_plan_logits"].argmax(dim=-1).detach(),
        "slots": out["slots"].detach(),
    }


def make_config_from_args(args) -> SlotEncoderConfig:
    return SlotEncoderConfig(
        history=args.history,
        local_dim=17,
        slot_dim=args.slot_dim,
        hidden_dim=args.hidden_dim,
        num_object_slots=args.num_object_slots,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_phases=9,
        plan_codebook_size=args.plan_codebook_size,
        dropout=args.dropout,
        pose_weight=args.pose_weight,
        object_weight=args.object_weight,
        contact_weight=args.contact_weight,
        force_weight=args.force_weight,
        phase_weight=args.phase_weight,
        plan_weight=args.plan_weight,
    )
