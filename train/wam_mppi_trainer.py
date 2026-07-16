"""Train and audit Phase 3 behavior-prior and Monte-Carlo value heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from models.wam import (
    RWMARWorldModel,
    WAMPlanningHeads,
    WorldModelSequenceInputs,
)
from models.wam.rollout import symlog

ProgressCallback = Callable[[Mapping[str, float | int]], None]


@dataclass(frozen=True)
class PlanningHeadsTrainConfig:
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 10.0
    action_prior_weight: float = 1.0
    value_weight: float = 1.0
    max_steps: int = -1

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.learning_rate <= 0.0:
            raise ValueError("epochs and learning_rate must be positive")
        if self.weight_decay < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("invalid weight decay or gradient clip")
        if self.action_prior_weight < 0.0 or self.value_weight < 0.0:
            raise ValueError("planning-head loss weights must be non-negative")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("max_steps must be -1 or positive")


def freeze_world_model(model: RWMARWorldModel) -> RWMARWorldModel:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def train_planning_heads(
    heads: WAMPlanningHeads,
    world_model: RWMARWorldModel,
    loader: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
    config: PlanningHeadsTrainConfig,
    progress: ProgressCallback | None = None,
) -> tuple[list[float], int]:
    freeze_world_model(world_model.to(device))
    heads.to(device).train()
    optimizer = torch.optim.AdamW(
        heads.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[float] = []
    completed_steps = 0
    for epoch in range(config.epochs):
        for raw_batch in loader:
            batch = _prepare_batch(raw_batch, device)
            with torch.no_grad():
                _, _, features = world_model.encode_planning_history(
                    _history(batch)
                )
            output = heads(features)
            action_losses = heads.action_nll(features, batch["actions"])
            prior_weights = batch["action_prior_weights"].reshape(-1)
            eligible = prior_weights.sum()
            if float(eligible.detach()) > 0.0:
                action_loss = (action_losses * prior_weights).sum() / eligible
            else:
                action_loss = output.action_mean.sum() * 0.0
            value_loss = F.smooth_l1_loss(
                output.value_symlog,
                symlog(batch["returns_to_go"]),
            )
            loss = (
                config.action_prior_weight * action_loss
                + config.value_weight * value_loss
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite Phase 3 planning-head loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                heads.parameters(), config.gradient_clip_norm
            )
            optimizer.step()
            completed_steps += 1
            loss_value = float(loss.detach().cpu())
            history.append(loss_value)
            if progress is not None:
                progress(
                    {
                        "epoch": epoch + 1,
                        "epochs": config.epochs,
                        "step": completed_steps,
                        "loss": loss_value,
                        "action_nll": float(action_loss.detach().cpu()),
                        "value_huber": float(value_loss.detach().cpu()),
                    }
                )
            if config.max_steps > 0 and completed_steps >= config.max_steps:
                return history, completed_steps
    return history, completed_steps


@torch.inference_mode()
def evaluate_planning_heads(
    heads: WAMPlanningHeads,
    world_model: RWMARWorldModel,
    loader: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
    max_batches: int = -1,
    progress: ProgressCallback | None = None,
) -> dict[str, float | int | None]:
    freeze_world_model(world_model.to(device))
    heads.to(device).eval()
    samples = 0
    eligible_samples = 0
    action_squared_error = 0.0
    selected_action_squared_error = 0.0
    value_absolute_error = 0.0
    value_squared_error = 0.0
    predicted_values: list[np.ndarray] = []
    target_values: list[np.ndarray] = []
    for batch_index, raw_batch in enumerate(loader):
        batch = _prepare_batch(raw_batch, device)
        _, _, features = world_model.encode_planning_history(_history(batch))
        output = heads(features)
        actions = torch.tanh(output.action_mean)
        action_error = (actions - batch["actions"]).square().mean(dim=-1)
        weights = batch["action_prior_weights"].reshape(-1)
        selected = weights > 0.0
        values = output.value.reshape(-1)
        targets = batch["returns_to_go"].reshape(-1)
        count = int(actions.shape[0])
        samples += count
        eligible_samples += int(selected.sum().cpu())
        action_squared_error += float(action_error.sum().cpu())
        if bool(selected.any()):
            selected_action_squared_error += float(action_error[selected].sum().cpu())
        value_absolute_error += float((values - targets).abs().sum().cpu())
        value_squared_error += float((values - targets).square().sum().cpu())
        predicted_values.append(values.float().cpu().numpy())
        target_values.append(targets.float().cpu().numpy())
        if progress is not None:
            progress({"batch": batch_index + 1, "samples": samples})
        if max_batches > 0 and batch_index + 1 >= max_batches:
            break
    if samples == 0:
        raise RuntimeError("cannot evaluate planning heads on an empty loader")
    predicted = np.concatenate(predicted_values)
    targets = np.concatenate(target_values)
    correlation = None
    if predicted.size > 1 and np.std(predicted) > 0.0 and np.std(targets) > 0.0:
        correlation = float(np.corrcoef(predicted, targets)[0, 1])
    return {
        "samples": samples,
        "eligible_action_prior_samples": eligible_samples,
        "action_rmse": float(np.sqrt(action_squared_error / samples)),
        "selected_action_rmse": (
            float(np.sqrt(selected_action_squared_error / eligible_samples))
            if eligible_samples
            else None
        ),
        "value_mae": value_absolute_error / samples,
        "value_rmse": float(np.sqrt(value_squared_error / samples)),
        "value_pearson": correlation,
    }


def _prepare_batch(
    batch: Mapping[str, Tensor], device: torch.device
) -> dict[str, Tensor]:
    required = (
        "states",
        "past_actions",
        "valid_mask",
        "candidate_actions",
        "returns_to_go",
        "action_prior_weights",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise KeyError(f"planning batch is missing {missing}")
    return {
        "states": batch["states"].to(device, non_blocking=True),
        "past_actions": batch["past_actions"].to(device, non_blocking=True),
        "valid_mask": batch["valid_mask"].to(device, non_blocking=True),
        "actions": batch["candidate_actions"][:, 0].to(
            device, non_blocking=True
        ),
        "returns_to_go": batch["returns_to_go"].to(device, non_blocking=True),
        "action_prior_weights": batch["action_prior_weights"].to(
            device, non_blocking=True
        ),
    }


def _history(batch: Mapping[str, Tensor]) -> WorldModelSequenceInputs:
    return WorldModelSequenceInputs(
        states=batch["states"],
        past_actions=batch["past_actions"],
        valid_mask=batch["valid_mask"],
    )


__all__ = [
    "PlanningHeadsTrainConfig",
    "evaluate_planning_heads",
    "freeze_world_model",
    "train_planning_heads",
]
