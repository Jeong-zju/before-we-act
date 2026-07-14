"""Optimization and metrics for the Phase 0 world/action baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models import ActionPrior, PolicyInputs, WorldModelInputs

ProgressCallback = Callable[[Mapping[str, float | int]], None]


@dataclass(frozen=True)
class NormalizationStats:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    delta_mean: np.ndarray
    delta_std: np.ndarray
    reward_mean: np.ndarray
    reward_std: np.ndarray

    def save(self, path: str) -> None:
        np.savez(path, **vars(self))

    def tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            name: torch.as_tensor(value, dtype=torch.float32, device=device)
            for name, value in vars(self).items()
        }


@dataclass(frozen=True)
class BaselineTrainConfig:
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_steps: int = -1
    reward_weight: float = 0.1
    done_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("epochs/lr must be positive and weight decay non-negative")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("max_steps must be -1 or positive")
        if self.reward_weight < 0.0 or self.done_weight < 0.0:
            raise ValueError("loss weights must be non-negative")


class _Moments:
    def __init__(self, width: int) -> None:
        self.count = 0
        self.sum = np.zeros(width, dtype=np.float64)
        self.square_sum = np.zeros(width, dtype=np.float64)

    def update(self, value: torch.Tensor) -> None:
        array = value.detach().cpu().numpy().reshape(-1, value.shape[-1])
        self.count += array.shape[0]
        self.sum += array.sum(axis=0, dtype=np.float64)
        self.square_sum += np.square(array, dtype=np.float64).sum(
            axis=0, dtype=np.float64
        )

    def finish(self, *, std_floor: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise ValueError("cannot normalize an empty dataset")
        mean = self.sum / self.count
        variance = np.maximum(self.square_sum / self.count - np.square(mean), 0.0)
        std = np.maximum(np.sqrt(variance), std_floor)
        return mean.astype(np.float32), std.astype(np.float32)


def fit_normalization(
    loader: Iterable[Mapping[str, torch.Tensor]],
    *,
    state_dim: int,
    action_dim: int,
    max_batches: int = -1,
    progress: ProgressCallback | None = None,
) -> NormalizationStats:
    state = _Moments(state_dim)
    action = _Moments(action_dim)
    delta = _Moments(state_dim)
    reward = _Moments(1)
    for batch_index, batch in enumerate(loader):
        current = batch["states"][:, -1]
        target = batch["target_states"][:, 0]
        state.update(current)
        action.update(batch["candidate_actions"][:, 0])
        delta.update(target - current)
        reward.update(batch["rewards"][:, 0])
        if progress is not None:
            progress({"batch": batch_index + 1})
        if max_batches > 0 and batch_index + 1 >= max_batches:
            break
    state_mean, state_std = state.finish()
    action_mean, action_std = action.finish()
    delta_mean, delta_std = delta.finish()
    reward_mean, reward_std = reward.finish()
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


def train_baseline(
    model: nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    stats: NormalizationStats,
    config: BaselineTrainConfig,
    device: torch.device,
    progress: ProgressCallback | None = None,
) -> list[float]:
    model.to(device)
    model.train()
    normalization = stats.tensors(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    losses: list[float] = []
    completed_steps = 0
    for epoch in range(config.epochs):
        epoch_total = 0.0
        epoch_steps = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prepared = _prepare_batch(batch, normalization, device)
            if isinstance(model, ActionPrior):
                output = model(PolicyInputs(state=prepared["normalized_state"]))
                loss = F.mse_loss(output.action, prepared["action"])
            else:
                output = model(
                    WorldModelInputs(
                        state=prepared["normalized_state"],
                        action=prepared["normalized_action"],
                    )
                )
                state_loss = F.mse_loss(
                    output.next_state, prepared["normalized_delta"]
                )
                reward_loss = F.mse_loss(
                    output.reward, prepared["normalized_reward"]
                )
                done_loss = F.binary_cross_entropy_with_logits(
                    output.done_logit, prepared["done"]
                )
                loss = (
                    state_loss
                    + config.reward_weight * reward_loss
                    + config.done_weight * done_loss
                )
            loss.backward()
            optimizer.step()
            epoch_total += float(loss.detach().cpu())
            epoch_steps += 1
            completed_steps += 1
            if progress is not None:
                progress(
                    {
                        "epoch": epoch + 1,
                        "epochs": config.epochs,
                        "step": completed_steps,
                        "loss": float(loss.detach().cpu()),
                    }
                )
            if config.max_steps > 0 and completed_steps >= config.max_steps:
                break
        losses.append(epoch_total / max(epoch_steps, 1))
        if config.max_steps > 0 and completed_steps >= config.max_steps:
            break
    return losses


@torch.no_grad()
def evaluate_baseline(
    model: nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    stats: NormalizationStats,
    device: torch.device,
    progress: ProgressCallback | None = None,
) -> dict[str, float | int]:
    model.to(device)
    model.eval()
    normalization = stats.tensors(device)
    sample_count = 0
    squared_error = 0.0
    normalized_squared_error = 0.0
    reward_absolute_error = 0.0
    done_correct = 0
    action_squared_error = 0.0
    for batch in loader:
        prepared = _prepare_batch(batch, normalization, device)
        batch_size = prepared["state"].shape[0]
        sample_count += batch_size
        if progress is not None:
            progress({"samples": sample_count})
        if isinstance(model, ActionPrior):
            output = model(PolicyInputs(state=prepared["normalized_state"]))
            action_squared_error += float(
                torch.square(output.action - prepared["action"]).sum().cpu()
            )
            continue
        output = model(
            WorldModelInputs(
                state=prepared["normalized_state"],
                action=prepared["normalized_action"],
            )
        )
        predicted_delta = (
            output.next_state * normalization["delta_std"]
            + normalization["delta_mean"]
        )
        predicted_state = prepared["state"] + predicted_delta
        error = predicted_state - prepared["target_state"]
        squared_error += float(torch.square(error).sum().cpu())
        normalized_squared_error += float(
            torch.square(error / normalization["state_std"]).sum().cpu()
        )
        predicted_reward = (
            output.reward * normalization["reward_std"]
            + normalization["reward_mean"]
        )
        reward_absolute_error += float(
            torch.abs(predicted_reward - prepared["reward"]).sum().cpu()
        )
        done_correct += int(
            ((output.done_logit >= 0.0) == (prepared["done"] >= 0.5)).sum().cpu()
        )
    if sample_count == 0:
        return {"samples": 0}
    if isinstance(model, ActionPrior):
        return {
            "samples": sample_count,
            "action_rmse": float(
                np.sqrt(action_squared_error / (sample_count * model.config.action_dim))
            ),
        }
    state_dim = int(normalization["state_mean"].numel())
    return {
        "samples": sample_count,
        "state_rmse": float(np.sqrt(squared_error / (sample_count * state_dim))),
        "state_nrmse": float(
            np.sqrt(normalized_squared_error / (sample_count * state_dim))
        ),
        "reward_mae": reward_absolute_error / sample_count,
        "done_accuracy": done_correct / sample_count,
    }


def _prepare_batch(
    batch: Mapping[str, torch.Tensor],
    normalization: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    state = batch["states"][:, -1].to(device)
    action = batch["candidate_actions"][:, 0].to(device)
    target_state = batch["target_states"][:, 0].to(device)
    reward = batch["rewards"][:, 0].to(device)
    done = batch["dones"][:, 0].to(device)
    delta = target_state - state
    return {
        "state": state,
        "action": action,
        "target_state": target_state,
        "reward": reward,
        "done": done,
        "normalized_state": (state - normalization["state_mean"])
        / normalization["state_std"],
        "normalized_action": (action - normalization["action_mean"])
        / normalization["action_std"],
        "normalized_delta": (delta - normalization["delta_mean"])
        / normalization["delta_std"],
        "normalized_reward": (reward - normalization["reward_mean"])
        / normalization["reward_std"],
    }


__all__ = [
    "BaselineTrainConfig",
    "NormalizationStats",
    "ProgressCallback",
    "evaluate_baseline",
    "fit_normalization",
    "train_baseline",
]
