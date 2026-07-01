from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PlanTokenizerConfig:
    horizon: int = 16
    action_dim: int = 4
    traj_dim: int = 5
    num_phases: int = 9
    latent_dim: int = 64
    hidden_dim: int = 256
    codebook_size: int = 64
    commitment_weight: float = 0.25
    phase_weight: float = 0.1
    residual_weight: float = 0.05
    residual_dropout: float = 0.0
    stop_residual_grad_to_encoder: bool = False


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, latent_dim: int, commitment_weight: float = 0.25):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.latent_dim = int(latent_dim)
        self.commitment_weight = float(commitment_weight)

        self.embedding = nn.Embedding(self.codebook_size, self.latent_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        # z_e: [B, D]
        distances = (
            z_e.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * z_e @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1).unsqueeze(0)
        )

        code_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(code_indices)

        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.commitment_weight * commitment_loss

        # Straight-through estimator.
        z_q_st = z_e + (z_q - z_e).detach()

        return {
            "z_q": z_q_st,
            "z_q_raw": z_q,
            "code_indices": code_indices,
            "vq_loss": vq_loss,
            "codebook_loss": codebook_loss.detach(),
            "commitment_loss": commitment_loss.detach(),
        }


class PlanTokenizer(nn.Module):
    def __init__(self, cfg: PlanTokenizerConfig):
        super().__init__()
        self.cfg = cfg

        in_dim = cfg.horizon * (cfg.action_dim + cfg.traj_dim)
        out_action_dim = cfg.horizon * cfg.action_dim
        out_traj_dim = cfg.horizon * cfg.traj_dim
        out_phase_dim = cfg.horizon * cfg.num_phases

        self.encoder = MLP(in_dim, cfg.hidden_dim, cfg.latent_dim, depth=4)
        self.residual_head = MLP(cfg.latent_dim, cfg.hidden_dim, cfg.latent_dim, depth=2)
        self.vq = VectorQuantizer(cfg.codebook_size, cfg.latent_dim, cfg.commitment_weight)

        self.decoder = MLP(cfg.latent_dim * 2, cfg.hidden_dim, out_action_dim + out_traj_dim + out_phase_dim, depth=4)

    def encode(self, actions: torch.Tensor, trajectory: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = actions.shape[0]
        x = torch.cat([actions, trajectory], dim=-1).reshape(B, -1)
        z_e = self.encoder(x)
        residual_input = z_e.detach() if self.cfg.stop_residual_grad_to_encoder else z_e
        residual = self.residual_head(residual_input)
        vq_out = self.vq(z_e)
        return {
            "z_e": z_e,
            "z_q": vq_out["z_q"],
            "z_q_raw": vq_out["z_q_raw"],
            "code_indices": vq_out["code_indices"],
            "residual": residual,
            "vq_loss": vq_out["vq_loss"],
            "codebook_loss": vq_out["codebook_loss"],
            "commitment_loss": vq_out["commitment_loss"],
        }

    def decode(self, z_q: torch.Tensor, residual: torch.Tensor) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        if self.training and cfg.residual_dropout > 0:
            keep = torch.rand(residual.shape[0], 1, device=residual.device, dtype=residual.dtype)
            keep = (keep >= cfg.residual_dropout).to(residual.dtype)
            residual = residual * keep
        z = torch.cat([z_q, residual], dim=-1)
        out = self.decoder(z)

        a_dim = cfg.horizon * cfg.action_dim
        x_dim = cfg.horizon * cfg.traj_dim
        p_dim = cfg.horizon * cfg.num_phases

        recon_actions = out[:, :a_dim].reshape(-1, cfg.horizon, cfg.action_dim)
        recon_traj = out[:, a_dim:a_dim + x_dim].reshape(-1, cfg.horizon, cfg.traj_dim)
        phase_logits = out[:, a_dim + x_dim:a_dim + x_dim + p_dim].reshape(-1, cfg.horizon, cfg.num_phases)

        return {
            "recon_actions": recon_actions,
            "recon_trajectory": recon_traj,
            "phase_logits": phase_logits,
        }

    def forward(self, actions: torch.Tensor, trajectory: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = self.encode(actions, trajectory)
        dec = self.decode(enc["z_q"], enc["residual"])
        return {**enc, **dec}

    @torch.no_grad()
    def encode_future_segment(self, actions: torch.Tensor, trajectory: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        return self.encode(actions, trajectory)

    @torch.no_grad()
    def decode_plan_latent(self, code_indices: torch.Tensor, residual: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        z_q = self.vq.embedding(code_indices)
        return self.decode(z_q, residual)


def compute_losses(
    model: PlanTokenizer,
    batch: Dict[str, torch.Tensor],
    action_mean: torch.Tensor | None = None,
    action_std: torch.Tensor | None = None,
    traj_mean: torch.Tensor | None = None,
    traj_std: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    cfg = model.cfg

    actions = batch["actions"]
    trajectory = batch["trajectory"]
    phase = batch["phase"]

    if action_mean is not None:
        actions_norm = (actions - action_mean.view(1, 1, -1)) / action_std.view(1, 1, -1)
    else:
        actions_norm = actions

    if traj_mean is not None:
        traj_norm = (trajectory - traj_mean.view(1, 1, -1)) / traj_std.view(1, 1, -1)
    else:
        traj_norm = trajectory

    out = model(actions_norm, traj_norm)

    loss_action = F.mse_loss(out["recon_actions"], actions_norm)
    loss_traj = F.mse_loss(out["recon_trajectory"], traj_norm)
    loss_phase = F.cross_entropy(out["phase_logits"].reshape(-1, cfg.num_phases), phase.reshape(-1))
    loss_residual = out["residual"].pow(2).mean()

    loss = (
        loss_action
        + loss_traj
        + out["vq_loss"]
        + cfg.phase_weight * loss_phase
        + cfg.residual_weight * loss_residual
    )

    return {
        "loss": loss,
        "loss_action": loss_action.detach(),
        "loss_traj": loss_traj.detach(),
        "loss_phase": loss_phase.detach(),
        "loss_vq": out["vq_loss"].detach(),
        "loss_residual": loss_residual.detach(),
        "code_indices": out["code_indices"].detach(),
        "recon_actions": out["recon_actions"].detach(),
        "recon_trajectory": out["recon_trajectory"].detach(),
        "phase_logits": out["phase_logits"].detach(),
    }


def codebook_usage(code_indices: torch.Tensor, codebook_size: int) -> Dict[str, float]:
    counts = torch.bincount(code_indices.reshape(-1).cpu(), minlength=codebook_size).float()
    probs = counts / counts.sum().clamp_min(1.0)
    nonzero = int((counts > 0).sum().item())
    entropy = float(-(probs[probs > 0] * probs[probs > 0].log()).sum().item())
    perplexity = float(torch.exp(torch.tensor(entropy)).item())
    return {
        "used_codes": nonzero,
        "usage_ratio": nonzero / codebook_size,
        "entropy": entropy,
        "perplexity": perplexity,
    }


def make_config_from_args(args) -> PlanTokenizerConfig:
    return PlanTokenizerConfig(
        horizon=args.horizon,
        action_dim=4,
        traj_dim=5,
        num_phases=9,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        codebook_size=args.codebook_size,
        commitment_weight=args.commitment_weight,
        phase_weight=args.phase_weight,
        residual_weight=args.residual_weight,
        residual_dropout=args.residual_dropout,
        stop_residual_grad_to_encoder=bool(args.stop_residual_grad_to_encoder),
    )
