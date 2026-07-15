"""Optimization and metrics exclusively for the accepted Phase 0 baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models import ActionPrior, PolicyInputs, WorldModelInputs
from models.wam.normalizer import NormalizationStats

ProgressCallback = Callable[[Mapping[str, float | int]], None]

BINARY_LABEL_FIELDS = {
    "done": "dones",
    "success": "successes",
    "failure": "failures",
}


@dataclass(frozen=True)
class BaselineTrainConfig:
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_steps: int = -1
    reward_weight: float = 0.1
    done_weight: float = 0.1
    outcome_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("epochs/lr must be positive and weight decay non-negative")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("max_steps must be -1 or positive")
        if (
            self.reward_weight < 0.0
            or self.done_weight < 0.0
            or self.outcome_weight < 0.0
        ):
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


def fit_binary_label_stats(
    loader: Iterable[Mapping[str, torch.Tensor]],
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, dict[str, float | int | bool]]:
    """Count rare terminal labels and derive positive BCE weights from train only."""

    positives = dict.fromkeys(BINARY_LABEL_FIELDS, 0)
    sample_count = 0
    for batch_index, batch in enumerate(loader):
        batch_size = int(batch["states"].shape[0])
        sample_count += batch_size
        for label, field in BINARY_LABEL_FIELDS.items():
            values = batch[field][:, 0].reshape(-1)
            positives[label] += int((values >= 0.5).sum().item())
        if progress is not None:
            progress({"batch": batch_index + 1})
    if sample_count == 0:
        raise ValueError("cannot fit binary label statistics on an empty dataset")

    result: dict[str, dict[str, float | int | bool]] = {}
    for label, positive_count in positives.items():
        negative_count = sample_count - positive_count
        has_both_classes = positive_count > 0 and negative_count > 0
        result[label] = {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "prevalence": positive_count / sample_count,
            "positive_weight": (
                negative_count / positive_count if positive_count > 0 else 1.0
            ),
            "has_both_classes": has_both_classes,
        }
    return result


def train_baseline(
    model: nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    stats: NormalizationStats,
    config: BaselineTrainConfig,
    device: torch.device,
    binary_label_stats: Mapping[
        str, Mapping[str, float | int | bool]
    ] | None = None,
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
    positive_weights = {
        label: torch.tensor(
            float((binary_label_stats or {}).get(label, {}).get("positive_weight", 1.0)),
            dtype=torch.float32,
            device=device,
        )
        for label in BINARY_LABEL_FIELDS
    }
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
                state_loss = F.mse_loss(output.next_state, prepared["normalized_delta"])
                reward_loss = F.mse_loss(output.reward, prepared["normalized_reward"])
                done_loss = F.binary_cross_entropy_with_logits(
                    output.done_logit,
                    prepared["done"],
                    pos_weight=positive_weights["done"],
                )
                success_loss = F.binary_cross_entropy_with_logits(
                    output.success_logit,
                    prepared["success"],
                    pos_weight=positive_weights["success"],
                )
                failure_loss = F.binary_cross_entropy_with_logits(
                    output.failure_logit,
                    prepared["failure"],
                    pos_weight=positive_weights["failure"],
                )
                loss = (
                    state_loss
                    + config.reward_weight * reward_loss
                    + config.done_weight * done_loss
                    + config.outcome_weight * (success_loss + failure_loss)
                )
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            epoch_total += loss_value
            epoch_steps += 1
            completed_steps += 1
            if progress is not None:
                progress(
                    {
                        "epoch": epoch + 1,
                        "epochs": config.epochs,
                        "step": completed_steps,
                        "loss": loss_value,
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
    classification_thresholds: Mapping[str, float] | None = None,
    calibrate_thresholds: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    model.to(device)
    model.eval()
    normalization = stats.tensors(device)
    sample_count = 0
    squared_error = 0.0
    normalized_squared_error = 0.0
    reward_absolute_error = 0.0
    action_squared_error = 0.0
    classification_scores: dict[str, list[torch.Tensor]] = {
        name: [] for name in BINARY_LABEL_FIELDS
    }
    classification_labels: dict[str, list[torch.Tensor]] = {
        name: [] for name in BINARY_LABEL_FIELDS
    }
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
            output.next_state * normalization["delta_std"] + normalization["delta_mean"]
        )
        predicted_state = prepared["state"] + predicted_delta
        error = predicted_state - prepared["target_state"]
        squared_error += float(torch.square(error).sum().cpu())
        normalized_squared_error += float(
            torch.square(error / normalization["state_std"]).sum().cpu()
        )
        predicted_reward = (
            output.reward * normalization["reward_std"] + normalization["reward_mean"]
        )
        reward_absolute_error += float(
            torch.abs(predicted_reward - prepared["reward"]).sum().cpu()
        )
        for label, score in (
            ("done", output.done_logit),
            ("success", output.success_logit),
            ("failure", output.failure_logit),
        ):
            classification_scores[label].append(score.detach().reshape(-1).cpu())
            classification_labels[label].append(
                prepared[label].detach().reshape(-1).cpu()
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
    scores = {
        label: torch.cat(parts).numpy()
        for label, parts in classification_scores.items()
    }
    labels = {
        label: torch.cat(parts).numpy()
        for label, parts in classification_labels.items()
    }
    calibration: dict[str, dict[str, float | str | None]] = {}
    if calibrate_thresholds:
        calibration = {
            label: calibrate_binary_threshold(labels[label], scores[label])
            for label in BINARY_LABEL_FIELDS
        }
        selected_thresholds = {
            label: float(result["threshold"])
            for label, result in calibration.items()
        }
    else:
        selected_thresholds = {
            label: float((classification_thresholds or {}).get(label, 0.5))
            for label in BINARY_LABEL_FIELDS
        }
    classification = {
        label: binary_classification_metrics(
            labels[label],
            scores[label],
            threshold=selected_thresholds[label],
        )
        for label in BINARY_LABEL_FIELDS
    }
    result: dict[str, Any] = {
        "samples": sample_count,
        "state_rmse": float(np.sqrt(squared_error / (sample_count * state_dim))),
        "state_nrmse": float(
            np.sqrt(normalized_squared_error / (sample_count * state_dim))
        ),
        "reward_mae": reward_absolute_error / sample_count,
        "done_accuracy": classification["done"]["threshold_metrics"]["accuracy"],
        "classification": classification,
    }
    if calibration:
        result["calibration"] = calibration
    return result


def _prepare_batch(
    batch: Mapping[str, torch.Tensor],
    normalization: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    state = batch["states"][:, -1].to(device, non_blocking=True)
    action = batch["candidate_actions"][:, 0].to(device, non_blocking=True)
    target_state = batch["target_states"][:, 0].to(device, non_blocking=True)
    reward = batch["rewards"][:, 0].to(device, non_blocking=True)
    done = batch["dones"][:, 0].to(device, non_blocking=True)
    success = batch["successes"][:, 0].to(device, non_blocking=True)
    failure = batch["failures"][:, 0].to(device, non_blocking=True)
    delta = target_state - state
    return {
        "state": state,
        "action": action,
        "target_state": target_state,
        "reward": reward,
        "done": done,
        "success": success,
        "failure": failure,
        "normalized_state": (state - normalization["state_mean"])
        / normalization["state_std"],
        "normalized_action": (action - normalization["action_mean"])
        / normalization["action_std"],
        "normalized_delta": (delta - normalization["delta_mean"])
        / normalization["delta_std"],
        "normalized_reward": (reward - normalization["reward_mean"])
        / normalization["reward_std"],
    }


def calibrate_binary_threshold(
    labels: np.ndarray,
    logits: np.ndarray,
) -> dict[str, float | str | None]:
    """Select a probability threshold by validation F1, preferring recall on ties."""

    y_true, y_score = _binary_arrays(labels, logits)
    positive_count = int(y_true.sum())
    negative_count = int(y_true.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return {
            "threshold": 0.5,
            "objective": "max_f1",
            "objective_value": None,
            "precision": None,
            "recall": None,
            "status": "unavailable_single_class",
        }

    probabilities = _sigmoid(y_score)
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_probabilities = probabilities[order]
    sorted_labels = y_true[order]
    group_ends = np.flatnonzero(
        np.r_[sorted_probabilities[1:] != sorted_probabilities[:-1], True]
    )
    true_positives = np.cumsum(sorted_labels)[group_ends]
    predicted_positives = group_ends + 1
    precision = true_positives / predicted_positives
    recall = true_positives / positive_count
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=np.float64),
        where=(precision + recall) > 0.0,
    )
    best_value = float(f1.max())
    tied = np.flatnonzero(np.isclose(f1, best_value, rtol=1e-12, atol=1e-12))
    best = int(tied[np.argmax(recall[tied])])
    return {
        "threshold": float(sorted_probabilities[group_ends[best]]),
        "objective": "max_f1",
        "objective_value": best_value,
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "status": "calibrated",
    }


def binary_classification_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Report imbalance-aware ranking, baseline, calibration, and threshold metrics."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("classification threshold must be in [0,1]")
    y_true, y_score = _binary_arrays(labels, logits)
    probabilities = _sigmoid(y_score)
    predictions = probabilities >= threshold
    positives = y_true == 1
    negatives = ~positives
    true_positive = int(np.logical_and(predictions, positives).sum())
    false_positive = int(np.logical_and(predictions, negatives).sum())
    true_negative = int(np.logical_and(~predictions, negatives).sum())
    false_negative = int(np.logical_and(~predictions, positives).sum())
    positive_count = true_positive + false_negative
    negative_count = true_negative + false_positive
    sample_count = positive_count + negative_count
    prevalence = positive_count / sample_count
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / positive_count if positive_count else None
    specificity = true_negative / negative_count if negative_count else None
    f1 = None
    if recall is not None:
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
    balanced_accuracy = (
        (recall + specificity) / 2.0
        if recall is not None and specificity is not None
        else None
    )
    majority_class = int(positive_count > negative_count)
    majority_accuracy = max(positive_count, negative_count) / sample_count
    return {
        "positive_count": positive_count,
        "negative_count": negative_count,
        "prevalence": prevalence,
        "roc_auc": _binary_roc_auc(y_true, y_score),
        "average_precision": _binary_average_precision(y_true, y_score),
        "brier_score": float(np.mean(np.square(probabilities - y_true))),
        "ece_10_bin": _expected_calibration_error(y_true, probabilities, bins=10),
        "baselines": {
            "majority_class": majority_class,
            "majority_accuracy": majority_accuracy,
            "random_roc_auc": 0.5,
            "prevalence_average_precision": prevalence,
        },
        "threshold": threshold,
        "threshold_metrics": {
            "accuracy": (true_positive + true_negative) / sample_count,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        },
    }


def _binary_arrays(
    labels: np.ndarray, logits: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(labels).reshape(-1).astype(np.int64, copy=False)
    y_score = np.asarray(logits, dtype=np.float64).reshape(-1)
    if y_true.size == 0 or y_true.size != y_score.size:
        raise ValueError("binary labels and logits must be non-empty and aligned")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("binary labels must contain only 0 and 1")
    if not np.isfinite(y_score).all():
        raise ValueError("binary logits contain NaN/Inf")
    return y_true, y_score


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    result = np.empty_like(logits, dtype=np.float64)
    nonnegative = logits >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
    exponent = np.exp(logits[~nonnegative])
    result[~nonnegative] = exponent / (1.0 + exponent)
    return result


def _binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def _binary_average_precision(
    labels: np.ndarray, scores: np.ndarray
) -> float | None:
    positive_count = int(labels.sum())
    if positive_count == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    group_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    true_positives = np.cumsum(sorted_labels)[group_ends]
    precision = true_positives / (group_ends + 1)
    recall = true_positives / positive_count
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, *, bins: int
) -> float:
    bin_indices = np.minimum((probabilities * bins).astype(np.int64), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = bin_indices == index
        if mask.any():
            error += float(mask.mean()) * abs(
                float(probabilities[mask].mean()) - float(labels[mask].mean())
            )
    return error


__all__ = [
    "BaselineTrainConfig",
    "BINARY_LABEL_FIELDS",
    "NormalizationStats",
    "ProgressCallback",
    "binary_classification_metrics",
    "calibrate_binary_threshold",
    "evaluate_baseline",
    "fit_binary_label_stats",
    "fit_normalization",
    "train_baseline",
]
