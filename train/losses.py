"""Losses for the deployable, ego-local FE-PC-WAM models.

The functions in this module make the information boundary visible in their
signatures.  Privileged teammate actions and physical outcomes are targets
only: they are never forwarded to the belief encoder or intention model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from models.decentralized import EgoLocalWAM, LocalIntentionPosterior


@dataclass(frozen=True)
class EgoWAMLossConfig:
    slot_weight: float = 1.0
    ego_action_weight: float = 1.0
    teammate_action_weight: float = 0.5
    contact_weight: float = 0.5
    force_weight: float = 0.5
    progress_weight: float = 1.0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class IntentionLossConfig:
    code_weight: float = 1.0
    residual_weight: float = 0.25
    variance_floor: float = 1e-5

    def __post_init__(self) -> None:
        if self.code_weight < 0.0 or self.residual_weight < 0.0:
            raise ValueError("intention loss weights must be non-negative")
        if self.variance_floor <= 0.0:
            raise ValueError("variance_floor must be positive")


def compute_ego_wam_losses(
    model: EgoLocalWAM,
    *,
    ego_slots: torch.Tensor,
    plan_codes: torch.Tensor,
    plan_residuals: torch.Tensor,
    teammate_hypothesis_weight: torch.Tensor,
    target_ego_slots: torch.Tensor,
    target_ego_actions: torch.Tensor,
    privileged_target_teammate_actions: torch.Tensor,
    target_contact: torch.Tensor,
    target_force: torch.Tensor,
    target_progress: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    config: EgoWAMLossConfig | None = None,
) -> Dict[str, torch.Tensor]:
    """Train an ego-local WAM with privileged values used only as targets.

    ``plan_codes`` and ``plan_residuals`` are ego-first.  During oracle
    conditioning their teammate row may come from the teammate's action-only
    plan tokenizer.  At deployment that row is supplied by the local intention
    posterior or by an actually received reply.
    """

    cfg = config or EgoWAMLossConfig()
    out = model(
        ego_slots=ego_slots,
        plan_codes=plan_codes,
        plan_residuals=plan_residuals,
        teammate_hypothesis_weight=teammate_hypothesis_weight,
    )
    pred_slots = out["pred_ego_slots"]
    pred_actions = out["pred_actions"]
    B, H = pred_actions.shape[:2]
    action_dim = model.cfg.action_dim_per_agent

    expected_slots = (B, H, model.cfg.slots_per_agent, model.cfg.slot_dim)
    if tuple(target_ego_slots.shape) != expected_slots:
        raise ValueError(f"target_ego_slots must have shape {expected_slots}")
    expected_action = (B, H, action_dim)
    if tuple(target_ego_actions.shape) != expected_action:
        raise ValueError(f"target_ego_actions must have shape {expected_action}")
    if tuple(privileged_target_teammate_actions.shape) != expected_action:
        raise ValueError(
            f"privileged_target_teammate_actions must have shape {expected_action}"
        )

    mask = _time_mask(valid_mask, B, H, pred_actions)
    slot_loss = _masked_mean(
        F.smooth_l1_loss(pred_slots, target_ego_slots, reduction="none").mean(dim=(-1, -2)),
        mask,
    )
    ego_action_loss = _masked_mean(
        F.mse_loss(pred_actions[..., :action_dim], target_ego_actions, reduction="none").mean(-1),
        mask,
    )
    teammate_action_loss = _masked_mean(
        F.mse_loss(
            pred_actions[..., action_dim : 2 * action_dim],
            privileged_target_teammate_actions,
            reduction="none",
        ).mean(-1),
        mask,
    )

    contact = _as_time_target(target_contact, B, H, pred_actions, "target_contact")
    force = _as_time_target(target_force, B, H, pred_actions, "target_force")
    progress = _as_time_target(target_progress, B, H, pred_actions, "target_progress")
    contact_loss = _masked_mean(
        F.binary_cross_entropy_with_logits(
            out["pred_contact_logits"], contact.clamp(0.0, 1.0), reduction="none"
        ),
        mask,
    )
    force_loss = _masked_mean(
        F.smooth_l1_loss(out["pred_force"], force, reduction="none"), mask
    )
    progress_loss = _masked_mean(
        F.smooth_l1_loss(out["pred_progress"], progress, reduction="none"), mask
    )

    total = (
        cfg.slot_weight * slot_loss
        + cfg.ego_action_weight * ego_action_loss
        + cfg.teammate_action_weight * teammate_action_loss
        + cfg.contact_weight * contact_loss
        + cfg.force_weight * force_loss
        + cfg.progress_weight * progress_loss
    )
    return {
        "loss": total,
        "loss_slots": slot_loss.detach(),
        "loss_ego_actions": ego_action_loss.detach(),
        "loss_privileged_teammate_actions": teammate_action_loss.detach(),
        "loss_contact": contact_loss.detach(),
        "loss_force": force_loss.detach(),
        "loss_progress": progress_loss.detach(),
        "predictions": out,
    }


def compute_local_intention_losses(
    model: LocalIntentionPosterior,
    *,
    ego_slots: torch.Tensor,
    ego_plan_code: torch.Tensor,
    ego_plan_residual: torch.Tensor,
    agent_id: torch.Tensor,
    received_message_metadata: torch.Tensor,
    target_teammate_code: torch.Tensor,
    target_teammate_residual: torch.Tensor,
    active_code_mask: torch.Tensor | None = None,
    config: IntentionLossConfig | None = None,
) -> Dict[str, torch.Tensor]:
    """Supervise the local posterior with action-only teammate plan targets.

    The target plan is a training label.  It is not an input to
    :class:`LocalIntentionPosterior`, which only sees the ego belief, ego plan,
    identity, and metadata for messages that have already arrived.
    """

    cfg = config or IntentionLossConfig()
    out = model(
        ego_slots=ego_slots,
        ego_plan_code=ego_plan_code,
        ego_plan_residual=ego_plan_residual,
        agent_id=agent_id,
        received_message_metadata=received_message_metadata,
    )
    logits = out["code_logits"]
    B, K = logits.shape
    target_code = target_teammate_code.to(device=logits.device, dtype=torch.long).reshape(-1)
    if target_code.shape != (B,):
        raise ValueError("target_teammate_code must have shape [B]")
    if (target_code < 0).any() or (target_code >= K).any():
        raise ValueError("target_teammate_code lies outside the codebook")

    masked_logits, support_mask = _mask_code_logits(logits, active_code_mask)
    if support_mask is not None and not support_mask[target_code].all():
        raise ValueError("target_teammate_code is outside the empirical active-code support")
    code_loss = F.cross_entropy(masked_logits, target_code)

    residual = target_teammate_residual.to(
        device=logits.device, dtype=out["residual_mu_by_code"].dtype
    )
    D = out["residual_mu_by_code"].shape[-1]
    if residual.shape != (B, D):
        raise ValueError(f"target_teammate_residual must have shape [{B}, {D}]")
    gather = target_code[:, None, None].expand(B, 1, D)
    mu = out["residual_mu_by_code"].gather(1, gather).squeeze(1)
    logvar = out["residual_logvar_by_code"].gather(1, gather).squeeze(1)
    variance = logvar.exp().clamp_min(cfg.variance_floor)
    residual_nll = 0.5 * (variance.log() + (residual - mu).square() / variance).mean()

    total = cfg.code_weight * code_loss + cfg.residual_weight * residual_nll
    probabilities = masked_logits.softmax(dim=-1)
    confidence, predicted_code = probabilities.max(dim=-1)
    correct = predicted_code.eq(target_code)
    brier = (
        probabilities
        - F.one_hot(target_code, num_classes=K).to(dtype=probabilities.dtype)
    ).square().sum(dim=-1).mean()
    return {
        "loss": total,
        "loss_code": code_loss.detach(),
        "loss_residual_nll": residual_nll.detach(),
        "code_accuracy": correct.float().mean().detach(),
        "mean_confidence": confidence.mean().detach(),
        "brier_score": brier.detach(),
        "posterior": out,
        "supported_code_probabilities": probabilities.detach(),
    }


def _mask_code_logits(
    logits: torch.Tensor, active_code_mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if active_code_mask is None:
        return logits, None
    mask = active_code_mask.to(device=logits.device, dtype=torch.bool).reshape(-1)
    if mask.shape != (logits.shape[-1],):
        raise ValueError("active_code_mask must have shape [codebook_size]")
    if not mask.any():
        raise ValueError("active_code_mask must contain at least one active code")
    return logits.masked_fill(~mask.unsqueeze(0), torch.finfo(logits.dtype).min), mask


def _time_mask(
    value: torch.Tensor | None,
    batch: int,
    horizon: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    if value is None:
        return torch.ones(batch, horizon, device=reference.device, dtype=reference.dtype)
    if value.shape != (batch, horizon):
        raise ValueError(f"valid_mask must have shape [{batch}, {horizon}]")
    return value.to(device=reference.device, dtype=reference.dtype)


def _as_time_target(
    value: torch.Tensor,
    batch: int,
    horizon: int,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    target = value.to(device=reference.device, dtype=reference.dtype)
    if target.shape == (batch, horizon, 1):
        target = target.squeeze(-1)
    if target.shape != (batch, horizon):
        raise ValueError(f"{name} must have shape [{batch}, {horizon}] or [{batch}, {horizon}, 1]")
    return target


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.shape != mask.shape:
        raise ValueError(f"masked value/mask shapes differ: {value.shape} vs {mask.shape}")
    denominator = mask.sum().clamp_min(1.0)
    return (value * mask).sum() / denominator
