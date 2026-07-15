"""Statistics, curriculum optimization, and validation for Phase 1 RWM-AR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor

from models.wam import NormalizationStats, RWMARWorldModel, WorldModelSequenceInputs
from models.wam.rollout import wrap_to_pi
from train.rwm_ar_losses import RWMLossWeights, compute_rwm_loss

ProgressCallback = Callable[[Mapping[str, float | int]], None]


@dataclass(frozen=True)
class RWMTrainConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 10.0
    horizon_decay: float = 0.95
    use_amp: bool = True
    max_steps: int = -1
    loss_weights: RWMLossWeights = RWMLossWeights()

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError(
                "learning_rate must be positive and weight_decay non-negative"
            )
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if not 0.0 < self.horizon_decay <= 1.0:
            raise ValueError("horizon_decay must be in (0,1]")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("max_steps must be -1 or positive")


class _Moments:
    def __init__(self, width: int) -> None:
        self.count = 0
        self.sum = np.zeros(width, dtype=np.float64)
        self.square_sum = np.zeros(width, dtype=np.float64)

    def update(self, value: Tensor) -> None:
        array = value.detach().cpu().numpy().reshape(-1, value.shape[-1])
        self.count += array.shape[0]
        self.sum += array.sum(axis=0, dtype=np.float64)
        self.square_sum += np.square(array, dtype=np.float64).sum(axis=0)

    def finish(self, floor: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise ValueError("cannot fit statistics from an empty loader")
        mean = self.sum / self.count
        variance = np.maximum(self.square_sum / self.count - np.square(mean), 0.0)
        return (
            mean.astype(np.float32),
            np.maximum(np.sqrt(variance), floor).astype(np.float32),
        )


def fit_wam_normalization(
    loader: Iterable[Mapping[str, Tensor]],
    *,
    state_dim: int,
    action_dim: int,
    yaw_indices: tuple[int, ...],
    std_floor: float = 1e-3,
    max_batches: int = -1,
    progress: ProgressCallback | None = None,
) -> NormalizationStats:
    if std_floor <= 0.0:
        raise ValueError("std_floor must be positive")
    state = _Moments(state_dim)
    action = _Moments(action_dim)
    delta = _Moments(state_dim)
    reward = _Moments(1)
    for batch_index, batch in enumerate(loader, start=1):
        current = batch["states"][:, -1]
        target = batch["target_states"][:, 0]
        raw_delta = target - current
        for yaw_index in yaw_indices:
            raw_delta[..., yaw_index] = wrap_to_pi(raw_delta[..., yaw_index])
        state.update(current)
        action.update(batch["candidate_actions"][:, 0])
        delta.update(raw_delta)
        reward.update(batch["rewards"][:, 0])
        if progress is not None:
            progress({"batch": batch_index})
        if max_batches > 0 and batch_index >= max_batches:
            break
    state_mean, state_std = state.finish(std_floor)
    action_mean, action_std = action.finish(std_floor)
    delta_mean, delta_std = delta.finish(std_floor)
    reward_mean, reward_std = reward.finish(std_floor)
    return NormalizationStats(
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        delta_mean=delta_mean,
        delta_std=delta_std,
        reward_mean=reward_mean,
        reward_std=reward_std,
    )


def fit_wam_label_stats(
    loader: Iterable[Mapping[str, Tensor]],
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, dict[str, float | int | bool]]:
    fields = {"done": "dones", "success": "successes", "failure": "failures"}
    positives = dict.fromkeys(fields, 0)
    samples = 0
    for batch_index, batch in enumerate(loader, start=1):
        count = int(batch["states"].shape[0])
        samples += count
        for label, field in fields.items():
            positives[label] += int((batch[field][:, 0] >= 0.5).sum())
        if progress is not None:
            progress({"batch": batch_index})
    if samples == 0:
        raise ValueError("cannot fit labels from an empty loader")
    return {
        label: {
            "positive_count": count,
            "negative_count": samples - count,
            "prevalence": count / samples,
            "positive_weight": (samples - count) / count if count else 1.0,
            "has_both_classes": 0 < count < samples,
        }
        for label, count in positives.items()
    }


def make_positive_weights(
    label_stats: Mapping[str, Mapping[str, float | int | bool]],
    device: torch.device,
) -> dict[str, Tensor]:
    return {
        label: torch.tensor(float(label_stats[label]["positive_weight"]), device=device)
        for label in ("done", "success", "failure")
    }


def build_optimizer(
    model: RWMARWorldModel, config: RWMTrainConfig
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def train_curriculum_stage(
    model: RWMARWorldModel,
    loader: Iterable[Mapping[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    horizon: int,
    epochs: int,
    config: RWMTrainConfig,
    positive_weights: Mapping[str, Tensor],
    device: torch.device,
    completed_steps: int = 0,
    progress: ProgressCallback | None = None,
    teacher_forcing: bool = False,
) -> tuple[list[float], int]:
    if horizon <= 0 or epochs <= 0:
        raise ValueError("horizon and epochs must be positive")
    model.to(device)
    model.train()
    losses: list[float] = []
    amp_enabled = config.use_amp and device.type == "cuda"
    amp_dtype = _preferred_amp_dtype(device)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )
    for epoch in range(epochs):
        for raw_batch in loader:
            if config.max_steps > 0 and completed_steps >= config.max_steps:
                return losses, completed_steps
            batch = _batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                history = _history_from_batch(batch)
                actions = batch["candidate_actions"][:, :horizon]
                predictions = (
                    model.predict_teacher_forced(
                        history, actions, batch["target_states"][:, :horizon]
                    )
                    if teacher_forcing
                    else model.predict(history, actions)
                )
                loss, components = compute_rwm_loss(
                    predictions,
                    batch,
                    delta_std=model.delta_std,
                    yaw_indices=model.config.yaw_indices,
                    closed_indices=model.config.gripper_closed_indices,
                    positive_weights=positive_weights,
                    horizon_decay=config.horizon_decay,
                    weights=config.loss_weights,
                )
            if not bool(torch.isfinite(loss)):
                bad_components = [
                    name
                    for name, value in components.items()
                    if not bool(torch.isfinite(value).all())
                ]
                bad_predictions = [
                    name
                    for name in predictions.__dataclass_fields__
                    if not bool(torch.isfinite(getattr(predictions, name)).all())
                ]
                raise FloatingPointError(
                    "non-finite RWM loss before backward; "
                    f"components={bad_components}, predictions={bad_predictions}, "
                    f"amp={amp_enabled}, amp_dtype={amp_dtype}, delta_std_min="
                    f"{float(model.delta_std.min().detach().cpu()):.3g}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            scaler.step(optimizer)
            scaler.update()
            completed_steps += 1
            value = float(loss.detach().cpu())
            losses.append(value)
            if progress is not None:
                progress(
                    {
                        "epoch": epoch + 1,
                        "epochs": epochs,
                        "step": completed_steps,
                        "loss": value,
                        "state_mean_mse": float(
                            components["state_mean_mse"].cpu()
                        ),
                        "state_nll": float(components["state_nll"].cpu()),
                        "gradient_norm": float(gradient_norm.detach().cpu()),
                    }
                )
    return losses, completed_steps


@torch.no_grad()
def evaluate_wam_loss(
    model: RWMARWorldModel,
    loader: Iterable[Mapping[str, Tensor]],
    *,
    horizon: int,
    config: RWMTrainConfig,
    positive_weights: Mapping[str, Tensor],
    device: torch.device,
    max_batches: int = -1,
    progress: ProgressCallback | None = None,
    teacher_forcing: bool = False,
) -> dict[str, float | int]:
    model.to(device)
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    for batch_index, raw_batch in enumerate(loader, start=1):
        batch = _batch_to_device(raw_batch, device)
        history = _history_from_batch(batch)
        actions = batch["candidate_actions"][:, :horizon]
        predictions = (
            model.predict_teacher_forced(
                history, actions, batch["target_states"][:, :horizon]
            )
            if teacher_forcing
            else model.predict(history, actions)
        )
        _, components = compute_rwm_loss(
            predictions,
            batch,
            delta_std=model.delta_std,
            yaw_indices=model.config.yaw_indices,
            closed_indices=model.config.gripper_closed_indices,
            positive_weights=positive_weights,
            horizon_decay=config.horizon_decay,
            weights=config.loss_weights,
        )
        for name, value in components.items():
            totals[name] = totals.get(name, 0.0) + float(value.cpu())
        batches += 1
        if progress is not None:
            progress({"batch": batch_index})
        if max_batches > 0 and batch_index >= max_batches:
            break
    if batches == 0:
        return {"batches": 0}
    return {
        "batches": batches,
        **{name: value / batches for name, value in totals.items()},
    }


def _history_from_batch(batch: Mapping[str, Tensor]) -> WorldModelSequenceInputs:
    return WorldModelSequenceInputs(
        states=batch["states"],
        past_actions=batch["past_actions"],
        valid_mask=batch["valid_mask"],
    )


def _batch_to_device(
    batch: Mapping[str, Tensor], device: torch.device
) -> dict[str, Tensor]:
    return {
        name: value.to(device, non_blocking=True)
        for name, value in batch.items()
        if isinstance(value, Tensor)
    }


def _preferred_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


__all__ = [
    "ProgressCallback",
    "RWMTrainConfig",
    "build_optimizer",
    "evaluate_wam_loss",
    "fit_wam_label_stats",
    "fit_wam_normalization",
    "make_positive_weights",
    "train_curriculum_stage",
]
