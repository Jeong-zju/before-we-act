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
    step_reward_weight: float = 1.0
    return_quantile_weight: float = 1.0
    quantile_crossing_weight: float = 0.1
    terminal_success_weight: float = 0.5
    terminal_failure_weight: float = 0.25
    constraint_weight: float = 0.5
    completion_time_weight: float = 0.1
    branch_ranking_weight: float = 0.5

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
    target_reward: torch.Tensor | None = None,
    target_success: torch.Tensor | None = None,
    target_failure_reason: torch.Tensor | None = None,
    target_collision: torch.Tensor | None = None,
    target_force_violation: torch.Tensor | None = None,
    branch_plan_codes: torch.Tensor | None = None,
    branch_plan_residuals: torch.Tensor | None = None,
    branch_returns: torch.Tensor | None = None,
    branch_valid: torch.Tensor | None = None,
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
    zero = progress_loss.new_zeros(())
    step_reward_loss = zero
    return_quantile_loss = zero
    quantile_crossing_loss = zero
    success_loss = zero
    failure_loss = zero
    collision_loss = zero
    force_violation_loss = zero
    completion_time_loss = zero
    branch_ranking_loss = zero
    if target_reward is not None:
        reward = _as_time_target(target_reward, B, H, pred_actions, "target_reward")
        step_reward_loss = _masked_mean(
            F.smooth_l1_loss(out["pred_step_reward"], reward, reduction="none"), mask
        )
        realized_return = (reward * mask.to(reward.dtype)).sum(dim=1)
        quantiles = out["pred_return_quantiles"]
        levels = quantiles.new_tensor(model.cfg.return_quantiles).view(1, -1)
        error = realized_return[:, None] - quantiles
        return_quantile_loss = torch.maximum(
            levels * error, (levels - 1.0) * error
        ).mean()
        if quantiles.shape[1] > 1:
            quantile_crossing_loss = F.relu(
                quantiles[:, :-1] - quantiles[:, 1:]
            ).mean()
    terminal_success = None
    if target_success is not None:
        success_target = _as_time_target(
            target_success, B, H, pred_actions, "target_success"
        )
        terminal_success = success_target.max(dim=1).values
        success_loss = F.binary_cross_entropy_with_logits(
            out["pred_success_logits"], terminal_success
        )
    if target_failure_reason is not None:
        reason = target_failure_reason.reshape(B, H).long()[:, -1]
        reason = reason.clamp(0, model.cfg.failure_classes - 1)
        failure_loss = F.cross_entropy(out["pred_failure_logits"], reason)
    if target_collision is not None:
        collision = _as_time_target(
            target_collision, B, H, pred_actions, "target_collision"
        ).max(dim=1).values
        collision_loss = F.binary_cross_entropy_with_logits(
            out["pred_collision_logits"], collision.clamp(0.0, 1.0)
        )
    if target_force_violation is not None:
        violation = _as_time_target(
            target_force_violation, B, H, pred_actions, "target_force_violation"
        ).max(dim=1).values
        force_violation_loss = F.binary_cross_entropy_with_logits(
            out["pred_force_violation_logits"], violation.clamp(0.0, 1.0)
        )
    if target_success is not None:
        done_step = success_target.argmax(dim=1).to(pred_actions.dtype)
        no_terminal = success_target.max(dim=1).values <= 0
        done_step = torch.where(no_terminal, done_step.new_full(done_step.shape, H - 1), done_step)
        completion_target = (done_step + 1.0) / float(H)
        completion_time_loss = F.smooth_l1_loss(
            out["pred_completion_time"], completion_target
        )
    branch_args = (
        branch_plan_codes,
        branch_plan_residuals,
        branch_returns,
        branch_valid,
    )
    if any(value is not None for value in branch_args):
        if not all(value is not None for value in branch_args):
            raise ValueError("all branch-ranking tensors must be supplied together")
        assert branch_plan_codes is not None
        assert branch_plan_residuals is not None
        assert branch_returns is not None
        assert branch_valid is not None
        N = int(branch_plan_codes.shape[1])
        if branch_plan_codes.shape != (B, N, 2):
            raise ValueError("branch_plan_codes must have shape [B,N,2]")
        valid = branch_valid.to(device=ego_slots.device, dtype=torch.bool)
        if valid.any():
            active_rows = valid.any(dim=1)
            active_count = int(active_rows.sum().item())
            branch_out = model(
                ego_slots=ego_slots[active_rows, None]
                .expand(-1, N, -1, -1)
                .reshape(active_count * N, model.cfg.slots_per_agent, model.cfg.slot_dim),
                plan_codes=branch_plan_codes[active_rows].reshape(active_count * N, 2),
                plan_residuals=branch_plan_residuals[active_rows].reshape(
                    active_count * N, 2, model.cfg.plan_latent_dim
                ),
                teammate_hypothesis_weight=torch.ones(
                    active_count * N, device=ego_slots.device, dtype=ego_slots.dtype
                ),
            )
            predicted_return = branch_out["pred_return_quantiles"].mean(dim=-1).reshape(
                active_count, N
            )
            target_branch_return = branch_returns[active_rows].to(
                device=predicted_return.device, dtype=predicted_return.dtype
            )
            active_valid = valid[active_rows]
            target_difference = (
                target_branch_return[:, :, None] - target_branch_return[:, None, :]
            )
            predicted_difference = (
                predicted_return[:, :, None] - predicted_return[:, None, :]
            )
            pair_valid = (
                active_valid[:, :, None]
                & active_valid[:, None, :]
                & (target_difference.abs() > 1e-4)
            )
            if pair_valid.any():
                branch_ranking_loss = F.softplus(
                    -target_difference[pair_valid].sign()
                    * predicted_difference[pair_valid]
                ).mean()

    total = (
        cfg.slot_weight * slot_loss
        + cfg.ego_action_weight * ego_action_loss
        + cfg.teammate_action_weight * teammate_action_loss
        + cfg.contact_weight * contact_loss
        + cfg.force_weight * force_loss
        + cfg.progress_weight * progress_loss
        + cfg.step_reward_weight * step_reward_loss
        + cfg.return_quantile_weight * return_quantile_loss
        + cfg.quantile_crossing_weight * quantile_crossing_loss
        + cfg.terminal_success_weight * success_loss
        + cfg.terminal_failure_weight * failure_loss
        + cfg.constraint_weight * (collision_loss + force_violation_loss)
        + cfg.completion_time_weight * completion_time_loss
        + cfg.branch_ranking_weight * branch_ranking_loss
    )
    return {
        "loss": total,
        "loss_slots": slot_loss.detach(),
        "loss_ego_actions": ego_action_loss.detach(),
        "loss_privileged_teammate_actions": teammate_action_loss.detach(),
        "loss_contact": contact_loss.detach(),
        "loss_force": force_loss.detach(),
        "loss_progress": progress_loss.detach(),
        "loss_step_reward": step_reward_loss.detach(),
        "loss_return_quantiles": return_quantile_loss.detach(),
        "loss_quantile_crossing": quantile_crossing_loss.detach(),
        "loss_terminal_success": success_loss.detach(),
        "loss_terminal_failure": failure_loss.detach(),
        "loss_collision_risk": collision_loss.detach(),
        "loss_force_violation_risk": force_violation_loss.detach(),
        "loss_completion_time": completion_time_loss.detach(),
        "loss_branch_ranking": branch_ranking_loss.detach(),
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
