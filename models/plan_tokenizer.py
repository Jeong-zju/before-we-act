from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


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


# ---------------------------------------------------------------------------
# Action-only tokenizer
# ---------------------------------------------------------------------------


@dataclass
class ActionOnlyPlanTokenizerConfig:
    """Configuration for :class:`ActionOnlyPlanTokenizer`.

    The  encoder deliberately has no trajectory or outcome input.  A future
    trajectory may still be reconstructed by an auxiliary decoder during
    training, but it can never affect the plan code or residual sent at
    inference time.
    """

    horizon: int = 16
    action_dim: int = 4
    latent_dim: int = 64
    hidden_dim: int = 256
    codebook_size: int = 64
    commitment_weight: float = 0.25
    soft_assignment_temperature: float = 0.5
    usage_balance_weight: float = 0.1
    residual_weight: float = 0.05
    residual_dropout: float = 0.2
    auxiliary_traj_dim: int = 0
    auxiliary_traj_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.horizon <= 0 or self.action_dim <= 0:
            raise ValueError("horizon and action_dim must be positive")
        if self.latent_dim <= 0 or self.hidden_dim <= 0 or self.codebook_size <= 1:
            raise ValueError("latent_dim/hidden_dim must be positive and codebook_size must exceed one")
        if self.soft_assignment_temperature <= 0:
            raise ValueError("soft_assignment_temperature must be positive")
        if self.commitment_weight < 0:
            raise ValueError("commitment_weight cannot be negative")
        if not 0.0 <= self.residual_dropout <= 1.0:
            raise ValueError("residual_dropout must be in [0, 1]")
        if self.auxiliary_traj_dim < 0:
            raise ValueError("auxiliary_traj_dim cannot be negative")
        if min(self.usage_balance_weight, self.residual_weight, self.auxiliary_traj_weight) < 0:
            raise ValueError("loss weights cannot be negative")


class SoftUsageVectorQuantizer(VectorQuantizer):
    """Hard VQ for execution plus a differentiable batch-usage objective.

    Nearest-neighbour codes remain the actual transmitted representation.  A
    temperature-controlled soft assignment is used only to expose gradients
    when the batch-average code distribution collapses.
    """

    def __init__(
        self,
        codebook_size: int,
        latent_dim: int,
        commitment_weight: float = 0.25,
        temperature: float = 0.5,
    ):
        super().__init__(codebook_size, latent_dim, commitment_weight)
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def forward(self, z_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        if z_e.ndim != 2 or z_e.shape[-1] != self.latent_dim:
            raise ValueError(f"z_e must have shape [B, {self.latent_dim}]")

        distances = (
            z_e.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * z_e @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1).unsqueeze(0)
        )
        soft_assignments = F.softmax(-distances / self.temperature, dim=-1)
        mean_usage = soft_assignments.mean(dim=0)

        # KL(mean_usage || Uniform).  It is zero only for balanced soft usage
        # and retains gradients to both the encoder and codebook embeddings.
        usage_balance_loss = (
            mean_usage * (mean_usage.clamp_min(1e-10).log() + torch.log(mean_usage.new_tensor(self.codebook_size)))
        ).sum()
        soft_entropy = -(mean_usage * mean_usage.clamp_min(1e-10).log()).sum()

        code_indices = torch.argmin(distances, dim=1)
        z_q_raw = self.embedding(code_indices)
        codebook_loss = F.mse_loss(z_q_raw, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q_raw.detach())
        vq_loss = codebook_loss + self.commitment_weight * commitment_loss
        z_q = z_e + (z_q_raw - z_e).detach()

        return {
            "z_q": z_q,
            "z_q_raw": z_q_raw,
            "code_indices": code_indices,
            "vq_loss": vq_loss,
            "codebook_loss": codebook_loss.detach(),
            "commitment_loss": commitment_loss.detach(),
            "soft_assignments": soft_assignments,
            "soft_code_usage": mean_usage,
            "usage_balance_loss": usage_balance_loss,
            "soft_perplexity": soft_entropy.exp(),
        }


class ActionOnlyPlanTokenizer(nn.Module):
    """ plan tokenizer whose encoder consumes future actions only."""

    def __init__(self, cfg: ActionOnlyPlanTokenizerConfig):
        super().__init__()
        self.cfg = cfg
        action_flat_dim = cfg.horizon * cfg.action_dim
        auxiliary_flat_dim = cfg.horizon * cfg.auxiliary_traj_dim

        self.encoder = MLP(action_flat_dim, cfg.hidden_dim, cfg.latent_dim, depth=4)
        self.residual_head = MLP(cfg.latent_dim, cfg.hidden_dim, cfg.latent_dim, depth=2)
        self.vq = SoftUsageVectorQuantizer(
            cfg.codebook_size,
            cfg.latent_dim,
            cfg.commitment_weight,
            cfg.soft_assignment_temperature,
        )
        self.decoder = MLP(
            cfg.latent_dim * 2,
            cfg.hidden_dim,
            action_flat_dim + auxiliary_flat_dim,
            depth=4,
        )

    def _validate_actions(self, actions: torch.Tensor) -> None:
        expected = (self.cfg.horizon, self.cfg.action_dim)
        if actions.ndim != 3 or tuple(actions.shape[1:]) != expected:
            raise ValueError(f"actions must have shape [B, {expected[0]}, {expected[1]}]")

    def encode(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode an executable action segment; no outcome argument exists."""

        self._validate_actions(actions)
        z_e = self.encoder(actions.reshape(actions.shape[0], -1))
        residual = self.residual_head(z_e)
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
            "soft_assignments": vq_out["soft_assignments"],
            "soft_code_usage": vq_out["soft_code_usage"],
            "usage_balance_loss": vq_out["usage_balance_loss"],
            "soft_perplexity": vq_out["soft_perplexity"],
        }

    def decode(self, z_q: torch.Tensor, residual: torch.Tensor) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        if z_q.ndim != 2 or residual.shape != z_q.shape or z_q.shape[-1] != cfg.latent_dim:
            raise ValueError(f"z_q and residual must both have shape [B, {cfg.latent_dim}]")

        residual_used = F.dropout(residual, p=cfg.residual_dropout, training=self.training)
        decoded = self.decoder(torch.cat([z_q, residual_used], dim=-1))
        action_size = cfg.horizon * cfg.action_dim
        out: Dict[str, torch.Tensor] = {
            "recon_actions": decoded[:, :action_size].reshape(-1, cfg.horizon, cfg.action_dim),
            "decoder_residual": residual_used,
        }
        if cfg.auxiliary_traj_dim > 0:
            out["recon_auxiliary_trajectory"] = decoded[:, action_size:].reshape(
                -1, cfg.horizon, cfg.auxiliary_traj_dim
            )
        return out

    def forward(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = self.encode(actions)
        return {**enc, **self.decode(enc["z_q"], enc["residual"])}

    @torch.no_grad()
    def encode_action_segment(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        return self.encode(actions)

    @torch.no_grad()
    def encode_future_segment(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Alias matching the legacy naming without accepting trajectory truth."""

        return self.encode_action_segment(actions)

    @torch.no_grad()
    def decode_plan_latent(self, code_indices: torch.Tensor, residual: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        if code_indices.ndim != 1 or residual.ndim != 2 or code_indices.shape[0] != residual.shape[0]:
            raise ValueError("code_indices must be [B] and residual must be [B, D]")
        return self.decode(self.vq.embedding(code_indices), residual)


def compute_action_only_plan_losses(
    model: ActionOnlyPlanTokenizer,
    batch: Mapping[str, torch.Tensor],
    action_mean: torch.Tensor | None = None,
    action_std: torch.Tensor | None = None,
    auxiliary_traj_mean: torch.Tensor | None = None,
    auxiliary_traj_std: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Compute  losses without ever passing trajectory truth to the encoder."""

    cfg = model.cfg
    actions = batch["actions"]
    if action_mean is not None:
        if action_std is None:
            raise ValueError("action_std is required when action_mean is supplied")
        actions_in = (actions - action_mean.view(1, 1, -1)) / action_std.view(1, 1, -1).clamp_min(1e-6)
    else:
        actions_in = actions

    out = model(actions_in)
    loss_action = F.mse_loss(out["recon_actions"], actions_in)
    loss_residual = out["residual"].pow(2).mean()

    loss_auxiliary = loss_action.new_zeros(())
    if cfg.auxiliary_traj_dim > 0 and cfg.auxiliary_traj_weight > 0:
        if "trajectory" not in batch:
            raise KeyError("batch['trajectory'] is required only for the configured auxiliary decoder loss")
        trajectory = batch["trajectory"]
        expected = (actions.shape[0], cfg.horizon, cfg.auxiliary_traj_dim)
        if tuple(trajectory.shape) != expected:
            raise ValueError(f"trajectory auxiliary target must have shape {expected}")
        if auxiliary_traj_mean is not None:
            if auxiliary_traj_std is None:
                raise ValueError("auxiliary_traj_std is required with auxiliary_traj_mean")
            trajectory = (trajectory - auxiliary_traj_mean.view(1, 1, -1)) / auxiliary_traj_std.view(
                1, 1, -1
            ).clamp_min(1e-6)
        loss_auxiliary = F.mse_loss(out["recon_auxiliary_trajectory"], trajectory)

    loss = (
        loss_action
        + out["vq_loss"]
        + cfg.usage_balance_weight * out["usage_balance_loss"]
        + cfg.residual_weight * loss_residual
        + cfg.auxiliary_traj_weight * loss_auxiliary
    )
    return {
        "loss": loss,
        "loss_action": loss_action.detach(),
        "loss_vq": out["vq_loss"].detach(),
        "loss_usage_balance": out["usage_balance_loss"].detach(),
        "loss_residual": loss_residual.detach(),
        "loss_auxiliary_trajectory": loss_auxiliary.detach(),
        "soft_perplexity": out["soft_perplexity"].detach(),
        "soft_code_usage": out["soft_code_usage"].detach(),
        "code_indices": out["code_indices"].detach(),
        "residual": out["residual"].detach(),
        "recon_actions": out["recon_actions"].detach(),
    }


@dataclass
class PlanCodeSupport:
    """Empirical plan-code and residual support measured from real encodings.

    All tensors are kept on CPU so the object is directly suitable for a
    ``torch.save`` checkpoint.  Candidate sampling first follows empirical code
    usage, then samples a per-code diagonal Gaussian residual.
    """

    codebook_size: int
    min_count: int
    counts: torch.Tensor
    probabilities: torch.Tensor
    residual_mean: torch.Tensor
    residual_std: torch.Tensor

    def __post_init__(self) -> None:
        self.codebook_size = int(self.codebook_size)
        self.min_count = int(self.min_count)
        if self.codebook_size <= 0 or self.min_count <= 0:
            raise ValueError("codebook_size and min_count must be positive")
        self.counts = self.counts.detach().to(device="cpu", dtype=torch.long).clone()
        self.probabilities = self.probabilities.detach().to(device="cpu", dtype=torch.float32).clone()
        self.residual_mean = self.residual_mean.detach().to(device="cpu", dtype=torch.float32).clone()
        self.residual_std = self.residual_std.detach().to(device="cpu", dtype=torch.float32).clone()
        if self.counts.shape != (self.codebook_size,) or self.probabilities.shape != (self.codebook_size,):
            raise ValueError("counts/probabilities must have shape [codebook_size]")
        if self.residual_mean.ndim != 2 or self.residual_mean.shape[0] != self.codebook_size:
            raise ValueError("residual_mean must have shape [codebook_size, residual_dim]")
        if self.residual_std.shape != self.residual_mean.shape:
            raise ValueError("residual_std must match residual_mean")
        if not torch.isfinite(self.probabilities).all() or (self.probabilities < 0).any():
            raise ValueError("probabilities must be finite and non-negative")
        if not torch.isfinite(self.residual_mean).all() or not torch.isfinite(self.residual_std).all():
            raise ValueError("residual statistics must be finite")
        if (self.residual_std <= 0).any():
            raise ValueError("residual_std must be strictly positive")
        if self.active_codes.numel() == 0:
            raise ValueError("no code reaches min_count; lower the threshold or collect more encodings")

    @property
    def residual_dim(self) -> int:
        return int(self.residual_mean.shape[1])

    @property
    def active_codes(self) -> torch.Tensor:
        return torch.nonzero(self.counts >= self.min_count, as_tuple=False).flatten()

    def to_dict(self) -> Dict[str, Any]:
        """Return a checkpoint-serializable representation."""

        return {
            "format_version": 1,
            "codebook_size": self.codebook_size,
            "min_count": self.min_count,
            "counts": self.counts.clone(),
            "probabilities": self.probabilities.clone(),
            "active_codes": self.active_codes.clone(),
            "residual_mean": self.residual_mean.clone(),
            "residual_std": self.residual_std.clone(),
        }

    state_dict = to_dict

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> "PlanCodeSupport":
        version = int(state.get("format_version", 1))
        if version != 1:
            raise ValueError(f"unsupported PlanCodeSupport format_version={version}")
        return cls(
            codebook_size=int(state["codebook_size"]),
            min_count=int(state["min_count"]),
            counts=torch.as_tensor(state["counts"]),
            probabilities=torch.as_tensor(state["probabilities"]),
            residual_mean=torch.as_tensor(state["residual_mean"]),
            residual_std=torch.as_tensor(state["residual_std"]),
        )

    load_state_dict = from_dict

    def sample(
        self,
        num_samples: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        residual_scale: float = 1.0,
        ensure_code_diversity: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")

        active = self.active_codes
        active_probabilities = self.probabilities[active]
        if active_probabilities.sum() <= 0:
            # Counts are authoritative if a hand-edited checkpoint omitted
            # probabilities.
            active_probabilities = self.counts[active].float()
        active_probabilities = active_probabilities / active_probabilities.sum()
        if ensure_code_diversity:
            coverage = min(num_samples, int(active.numel()))
            covered_positions = torch.multinomial(
                active_probabilities,
                num_samples=coverage,
                replacement=False,
                generator=generator,
            )
            remaining = num_samples - coverage
            if remaining > 0:
                repeated_positions = torch.multinomial(
                    active_probabilities,
                    num_samples=remaining,
                    replacement=True,
                    generator=generator,
                )
                sampled_positions = torch.cat([covered_positions, repeated_positions])
            else:
                sampled_positions = covered_positions
        else:
            sampled_positions = torch.multinomial(
                active_probabilities,
                num_samples=num_samples,
                replacement=True,
                generator=generator,
            )
        code_indices = active[sampled_positions]
        mean = self.residual_mean[code_indices]
        std = self.residual_std[code_indices]
        noise = torch.randn(mean.shape, dtype=mean.dtype, generator=generator)
        residual = mean + float(residual_scale) * std * noise
        if device is not None:
            code_indices = code_indices.to(device)
            residual = residual.to(device)
        return {"code_indices": code_indices, "residual": residual}


class PlanCodeSupportAccumulator:
    """Streaming accumulator for empirical code and residual statistics."""

    def __init__(self, codebook_size: int, residual_dim: int):
        if codebook_size <= 0 or residual_dim <= 0:
            raise ValueError("codebook_size and residual_dim must be positive")
        self.codebook_size = int(codebook_size)
        self.residual_dim = int(residual_dim)
        self.counts = torch.zeros(self.codebook_size, dtype=torch.long)
        self.residual_sum = torch.zeros(self.codebook_size, self.residual_dim, dtype=torch.float64)
        self.residual_sq_sum = torch.zeros_like(self.residual_sum)

    def update(self, code_indices: torch.Tensor, residual: torch.Tensor) -> None:
        if residual.ndim < 2 or residual.shape[-1] != self.residual_dim:
            raise ValueError(f"residual must have final dimension {self.residual_dim}")
        codes = code_indices.detach().reshape(-1).to(device="cpu", dtype=torch.long)
        values = residual.detach().reshape(-1, self.residual_dim).to(device="cpu", dtype=torch.float64)
        if codes.shape[0] != values.shape[0]:
            raise ValueError("code_indices and residual must contain the same number of samples")
        if codes.numel() == 0:
            return
        if (codes < 0).any() or (codes >= self.codebook_size).any():
            raise ValueError("code index lies outside the configured codebook")
        if not torch.isfinite(values).all():
            raise ValueError("residual values must be finite")

        self.counts += torch.bincount(codes, minlength=self.codebook_size)
        index = codes.unsqueeze(-1).expand(-1, self.residual_dim)
        self.residual_sum.scatter_add_(0, index, values)
        self.residual_sq_sum.scatter_add_(0, index, values.square())

    def build(self, min_count: int = 1, std_floor: float = 1e-3) -> PlanCodeSupport:
        if min_count <= 0 or std_floor <= 0:
            raise ValueError("min_count and std_floor must be positive")
        total = self.counts.sum()
        if total <= 0:
            raise ValueError("cannot build plan support before observing encodings")

        denom = self.counts.clamp_min(1).to(torch.float64).unsqueeze(-1)
        mean = self.residual_sum / denom
        variance = (self.residual_sq_sum / denom - mean.square()).clamp_min(0.0)
        std = variance.sqrt().clamp_min(float(std_floor))
        probabilities = self.counts.float() / total.float()
        return PlanCodeSupport(
            codebook_size=self.codebook_size,
            min_count=min_count,
            counts=self.counts,
            probabilities=probabilities,
            residual_mean=mean.float(),
            residual_std=std.float(),
        )


def build_plan_code_support(
    code_indices: torch.Tensor,
    residual: torch.Tensor,
    codebook_size: int,
    *,
    min_count: int = 1,
    std_floor: float = 1e-3,
) -> PlanCodeSupport:
    """Build sampling support from encoded dataset usage (never hard-coded IDs)."""

    if residual.ndim < 2:
        raise ValueError("residual must have a final residual feature dimension")
    accumulator = PlanCodeSupportAccumulator(codebook_size, int(residual.shape[-1]))
    accumulator.update(code_indices, residual)
    return accumulator.build(min_count=min_count, std_floor=std_floor)
