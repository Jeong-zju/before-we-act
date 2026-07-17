"""Masked multi-step objectives for recurrent world model RWM-AR training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from models.wam.recurrent_dynamics import RWMARRolloutPredictions
from models.wam.rollout import symlog, wrap_to_pi


@dataclass(frozen=True)
class RWMLossWeights:
    state_mean: float = 1.0
    state_nll: float = 0.1
    gripper_closed: float = 1.0
    reward: float = 0.1
    done: float = 0.1
    terminal: float = 0.1
    auxiliary: float = 0.05

    def __post_init__(self) -> None:
        if any(value < 0.0 for value in vars(self).values()):
            raise ValueError("loss weights must be non-negative")


def compute_rwm_loss(
    predictions: RWMARRolloutPredictions,
    batch: Mapping[str, Tensor],
    *,
    delta_std: Tensor,
    yaw_indices: tuple[int, ...],
    closed_indices: tuple[int, ...],
    positive_weights: Mapping[str, Tensor],
    horizon_decay: float,
    weights: RWMLossWeights,
) -> tuple[Tensor, dict[str, Tensor]]:
    horizon = predictions.next_state_mean.shape[1]
    target_state = batch["target_states"][:, :horizon]
    mask = batch["forecast_mask"][:, :horizon]
    horizon_weights = horizon_decay ** torch.arange(
        horizon, dtype=target_state.dtype, device=target_state.device
    )
    step_weight = mask.to(target_state.dtype) * horizon_weights.unsqueeze(0)

    error = predictions.next_state_mean - target_state
    for yaw_index in yaw_indices:
        error = _replace_column(error, yaw_index, wrap_to_pi(error[..., yaw_index]))
    normalized_error = error / delta_std
    continuous = torch.ones(
        target_state.shape[-1], dtype=torch.bool, device=target_state.device
    )
    continuous[list(closed_indices)] = False
    log_std = predictions.normalized_delta_log_std[..., continuous].float()
    continuous_error = normalized_error[..., continuous].float()
    gaussian = 0.5 * (
        continuous_error.square() * torch.exp(-2.0 * log_std) + 2.0 * log_std
    )
    state_nll = _masked_step_mean(gaussian.mean(dim=-1), step_weight)
    # A learned Gaussian can lower NLL by inflating variance on hard dimensions,
    # which weakens the mean gradient and can make a tiny-dataset overfit test
    # appear successful.  Keep an explicit autoregressive mean objective so the
    # imagined trajectory itself must match the target, not just cover it.
    state_mean_mse = _masked_step_mean(
        continuous_error.square().mean(dim=-1), step_weight
    )

    closed_target = target_state[..., list(closed_indices)]
    closed_bce = F.binary_cross_entropy_with_logits(
        predictions.gripper_closed_logit,
        closed_target,
        reduction="none",
    ).mean(dim=-1)
    gripper_closed = _masked_step_mean(closed_bce, step_weight)

    reward = _masked_step_mean(
        F.smooth_l1_loss(
            predictions.reward_symlog,
            symlog(batch["rewards"][:, :horizon]),
            reduction="none",
        ).squeeze(-1),
        step_weight,
    )
    outcome_losses: dict[str, Tensor] = {}
    for label, field in (
        ("done", "dones"),
        ("success", "successes"),
        ("failure", "failures"),
    ):
        logits = getattr(predictions, f"{label}_logit")
        bce = F.binary_cross_entropy_with_logits(
            logits,
            batch[field][:, :horizon],
            pos_weight=positive_weights[label],
            reduction="none",
        ).squeeze(-1)
        outcome_losses[label] = _masked_step_mean(bce, step_weight)

    progress = _masked_step_mean(
        F.smooth_l1_loss(
            predictions.response_progress,
            batch["response_progress"][:, :horizon],
            reduction="none",
        ).squeeze(-1),
        step_weight,
    )
    coordination = _masked_step_mean(
        F.smooth_l1_loss(
            predictions.coordination_error,
            batch["coordination_error"][:, :horizon],
            reduction="none",
        ).squeeze(-1),
        step_weight,
    )
    executed_action = _masked_step_mean(
        F.mse_loss(
            predictions.executed_action,
            batch["executed_actions"][:, :horizon],
            reduction="none",
        ).mean(dim=-1),
        step_weight,
    )
    auxiliary = progress + coordination + executed_action
    total = (
        weights.state_mean * state_mean_mse
        + weights.state_nll * state_nll
        + weights.gripper_closed * gripper_closed
        + weights.reward * reward
        + weights.done * outcome_losses["done"]
        + weights.terminal * (outcome_losses["success"] + outcome_losses["failure"])
        + weights.auxiliary * auxiliary
    )
    components = {
        "total": total.detach(),
        "state_mean_mse": state_mean_mse.detach(),
        "state_nll": state_nll.detach(),
        "gripper_closed_bce": gripper_closed.detach(),
        "reward": reward.detach(),
        "done": outcome_losses["done"].detach(),
        "success": outcome_losses["success"].detach(),
        "failure": outcome_losses["failure"].detach(),
        "response_progress": progress.detach(),
        "coordination_error": coordination.detach(),
        "executed_action": executed_action.detach(),
    }
    return total, components


def _masked_step_mean(value: Tensor, step_weight: Tensor) -> Tensor:
    if value.shape != step_weight.shape:
        raise ValueError(
            f"masked value shape {value.shape} != weight shape {step_weight.shape}"
        )
    denominator = step_weight.sum().clamp_min(1.0)
    return (value * step_weight).sum() / denominator


def _replace_column(value: Tensor, index: int, column: Tensor) -> Tensor:
    parts = list(value.split(1, dim=-1))
    parts[index] = column.unsqueeze(-1)
    return torch.cat(parts, dim=-1)


__all__ = ["RWMLossWeights", "compute_rwm_loss"]
