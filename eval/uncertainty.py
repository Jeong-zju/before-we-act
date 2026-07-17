"""Calibration, OOD, event-aligned, and ensemble uncertainty acceptance metrics for world-model ensemble RWM-U."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from models.wam import (
    NormalizationStats,
    RWMARWorldModel,
    RWMUEnsemble,
    WorldModelSequenceInputs,
)
from models.wam.rollout import wrap_to_pi

ProgressCallback = Callable[[Mapping[str, int]], None]

_COVERAGE_Z = {"50": 0.6744897501960817, "90": 1.6448536269514722, "95": 1.959963984540054}


@dataclass(frozen=True)
class OODActionPerturbation:
    """Deterministic, bounded action shift used as the world-model ensemble OOD suite."""

    scale: float = 2.5
    offset_std: float = 4.0
    action_low: float = -1.0
    action_high: float = 1.0

    def __post_init__(self) -> None:
        if self.scale <= 1.0 or self.offset_std <= 0.0:
            raise ValueError("OOD scale must exceed 1 and offset_std must be positive")
        if self.action_low >= self.action_high:
            raise ValueError("action_low must be smaller than action_high")

    def apply(
        self,
        actions: Tensor,
        action_mean: Tensor,
        action_std: Tensor,
    ) -> Tensor:
        mean = action_mean.to(device=actions.device, dtype=actions.dtype)
        std = action_std.to(device=actions.device, dtype=actions.dtype)
        normalized = (actions - mean) / std
        coordinates = torch.arange(
            actions.numel(), device=actions.device, dtype=torch.int64
        ).reshape(actions.shape)
        sign = torch.where(
            coordinates.remainder(2) == 0,
            torch.ones((), device=actions.device, dtype=actions.dtype),
            -torch.ones((), device=actions.device, dtype=actions.dtype),
        )
        shifted = mean + std * (self.scale * normalized + self.offset_std * sign)
        return shifted.clamp(self.action_low, self.action_high)


@torch.no_grad()
def fit_variance_calibration(
    ensemble: RWMUEnsemble,
    loader: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
    horizon: int,
    max_batches: int = -1,
    scale_min: float = 0.05,
    scale_max: float = 20.0,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Fit a per-dimension variance scale on validation data only."""

    if horizon <= 0 or not 0.0 < scale_min <= scale_max:
        raise ValueError("invalid calibration horizon or scale bounds")
    ensemble.to(device).eval()
    state_dim = ensemble.member_config.state_dim
    squared_error = torch.zeros(state_dim, dtype=torch.float64)
    predicted_variance = torch.zeros(state_dim, dtype=torch.float64)
    values = torch.zeros(state_dim, dtype=torch.int64)
    batches = 0
    for batch_index, raw_batch in enumerate(loader, start=1):
        batch = _batch_to_device(raw_batch, device)
        if batch["candidate_actions"].shape[1] < horizon:
            raise ValueError("calibration dataset forecast horizon is too short")
        predictions = ensemble.predict(
            _history(batch), batch["candidate_actions"][:, :horizon]
        )
        moments = ensemble.state_moments(predictions)
        target = batch["target_states"][:, :horizon]
        valid = batch["forecast_mask"][:, :horizon]
        error = _state_error(
            moments["mean"], target, ensemble.member_config.yaw_indices
        )
        mask = valid.unsqueeze(-1).expand_as(error)
        squared_error += (error.square() * mask).sum(dim=(0, 1)).double().cpu()
        predicted_variance += (
            moments["total_variance"] * mask
        ).sum(dim=(0, 1)).double().cpu()
        values += mask.sum(dim=(0, 1)).cpu()
        batches += 1
        if progress is not None:
            progress({"batch": batch_index})
        if max_batches > 0 and batch_index >= max_batches:
            break
    if batches == 0 or not bool((values > 0).all()):
        raise ValueError("variance calibration received no complete state values")
    scale = (squared_error / predicted_variance.clamp_min(1e-12)).clamp(
        scale_min, scale_max
    )
    for index in ensemble.member_config.gripper_closed_indices:
        scale[index] = 1.0
    return {
        "format_version": "wam.rwm_u.calibration/1",
        "source_split": "validation",
        "batches": batches,
        "forecast_values": int(values.sum()),
        "variance_scale": [float(value) for value in scale],
        "scale_min": scale_min,
        "scale_max": scale_max,
    }


@torch.no_grad()
def evaluate_rwm_u(
    ensemble: RWMUEnsemble,
    loader: Iterable[Mapping[str, Tensor]],
    stats: NormalizationStats,
    *,
    device: torch.device,
    calibration: Mapping[str, Any],
    horizons: Sequence[int] = (1, 5, 10, 20, 40),
    teacher_forcing_model: RWMARWorldModel | None = None,
    ood: OODActionPerturbation = OODActionPerturbation(),
    event_horizon: int = 5,
    event_progress_min: float = 0.01,
    event_slowdown_min: float = 1e-3,
    event_asymmetry_min: float = 1e-3,
    event_ambiguity_fraction: float = 0.25,
    max_batches: int = -1,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    horizons = tuple(sorted(set(int(value) for value in horizons)))
    if not horizons or horizons[0] <= 0 or event_horizon not in horizons:
        raise ValueError("horizons must be positive and include event_horizon")
    max_horizon = max(horizons)
    scale = torch.as_tensor(
        calibration["variance_scale"], dtype=torch.float32, device=device
    )
    if scale.shape != (ensemble.member_config.state_dim,):
        raise ValueError("variance calibration dimension does not match model")
    ensemble.to(device).eval()
    if teacher_forcing_model is not None:
        teacher_forcing_model.to(device).eval()
    accumulator = _EnsembleAccumulator(
        horizons,
        stats,
        variance_scale=scale,
        yaw_indices=ensemble.member_config.yaw_indices,
        closed_indices=ensemble.member_config.gripper_closed_indices,
        event_horizon=event_horizon,
        event_progress_min=event_progress_min,
        event_slowdown_min=event_slowdown_min,
        event_asymmetry_min=event_asymmetry_min,
        event_ambiguity_fraction=event_ambiguity_fraction,
    )
    batches = 0
    for batch_index, raw_batch in enumerate(loader, start=1):
        batch = _batch_to_device(raw_batch, device)
        if batch["candidate_actions"].shape[1] < max_horizon:
            raise ValueError("evaluation dataset forecast horizon is too short")
        history = _history(batch)
        actions = batch["candidate_actions"][:, :max_horizon]
        predictions = ensemble.predict(history, actions)
        moments = ensemble.state_moments(predictions)
        risk = ensemble.risk_scores(predictions, actions)
        ood_actions = ood.apply(actions, ensemble.action_mean, ensemble.action_std)
        ood_predictions = ensemble.predict(history, ood_actions)
        ood_moments = ensemble.state_moments(ood_predictions)
        ood_risk = ensemble.risk_scores(ood_predictions, ood_actions)
        teacher_predictions = (
            teacher_forcing_model.predict(history, actions)
            if teacher_forcing_model is not None
            else None
        )
        accumulator.update(
            predictions,
            moments,
            risk,
            ood_moments,
            ood_risk,
            teacher_predictions,
            batch,
        )
        batches += 1
        if progress is not None:
            progress({"batch": batch_index})
        if max_batches > 0 and batch_index >= max_batches:
            break
    if batches == 0:
        raise ValueError("RWM-U evaluation received no batches")
    result = accumulator.finish()
    result.update(
        {
            "batches": batches,
            "horizons": list(horizons),
            "ood_perturbation": {
                "scale": ood.scale,
                "offset_std": ood.offset_std,
                "action_low": ood.action_low,
                "action_high": ood.action_high,
            },
        }
    )
    return result


class _EnsembleAccumulator:
    def __init__(
        self,
        horizons: Sequence[int],
        stats: NormalizationStats,
        *,
        variance_scale: Tensor,
        yaw_indices: tuple[int, ...],
        closed_indices: tuple[int, ...],
        event_horizon: int,
        event_progress_min: float,
        event_slowdown_min: float,
        event_asymmetry_min: float,
        event_ambiguity_fraction: float,
    ) -> None:
        self.horizons = tuple(horizons)
        self.state_mean = torch.as_tensor(stats.state_mean, dtype=torch.float32)
        self.state_std = torch.as_tensor(stats.state_std, dtype=torch.float32)
        self.variance_scale = variance_scale
        self.yaw_indices = yaw_indices
        self.closed_indices = closed_indices
        self.continuous_indices = tuple(
            index
            for index in range(int(self.state_std.numel()))
            if index not in closed_indices
        )
        self.sums = {
            horizon: {
                "samples": 0,
                "ensemble_normalized_squared_error": 0.0,
                "member0_normalized_squared_error": 0.0,
                "teacher_normalized_squared_error": 0.0,
                "teacher_values": 0,
                "continuous_values": 0,
                "finite": 0,
                "violations": 0,
                "nll_sum": 0.0,
                "coverage": {name: 0 for name in _COVERAGE_Z},
                "actual_error": [],
                "epistemic": [],
                "ood_epistemic": [],
            }
            for horizon in horizons
        }
        self.event_horizon = event_horizon
        self.event_progress_min = float(event_progress_min)
        self.event_slowdown_min = float(event_slowdown_min)
        self.event_asymmetry_min = float(event_asymmetry_min)
        self.event_ambiguity_fraction = float(event_ambiguity_fraction)
        self.event = {
            "samples": 0,
            "member_predictions": 0,
            "member_ambiguous": 0,
            "member_correct": 0,
            "ensemble_mean_ambiguous": 0,
            "ensemble_mean_correct": 0,
        }
        self.risk_id: list[np.ndarray] = []
        self.risk_ood: list[np.ndarray] = []

    def update(
        self,
        predictions: Any,
        moments: Mapping[str, Tensor],
        risk: Mapping[str, Tensor],
        ood_moments: Mapping[str, Tensor],
        ood_risk: Mapping[str, Tensor],
        teacher_predictions: Any | None,
        batch: Mapping[str, Tensor],
    ) -> None:
        std = self.state_std.to(moments["mean"].device)
        continuous = list(self.continuous_indices)
        for horizon in self.horizons:
            index = horizon - 1
            valid = batch["forecast_mask"][:, index]
            if not bool(valid.any()):
                continue
            mean = moments["mean"][valid, index]
            target = batch["target_states"][valid, index]
            error = _state_error(mean, target, self.yaw_indices)
            normalized = error[:, continuous] / std[continuous]
            member0_error = _state_error(
                predictions.next_state_mean[0, valid, index],
                target,
                self.yaw_indices,
            )
            item = self.sums[horizon]
            count = int(valid.sum())
            continuous_values = int(normalized.numel())
            item["samples"] += count
            item["continuous_values"] += continuous_values
            item["ensemble_normalized_squared_error"] += float(
                normalized.square().sum().cpu()
            )
            item["member0_normalized_squared_error"] += float(
                (member0_error[:, continuous] / std[continuous]).square().sum().cpu()
            )
            if teacher_predictions is not None:
                teacher_error = _state_error(
                    teacher_predictions.next_state_mean[valid, index],
                    target,
                    self.yaw_indices,
                )
                item["teacher_normalized_squared_error"] += float(
                    (teacher_error[:, continuous] / std[continuous])
                    .square()
                    .sum()
                    .cpu()
                )
                item["teacher_values"] += continuous_values
            finite = torch.isfinite(mean).all(dim=-1)
            item["finite"] += int(finite.sum())
            standardized_state = (mean - self.state_mean.to(mean.device)) / std
            violation = (~finite) | (standardized_state.abs() > 20.0).any(dim=-1)
            closed = mean[:, list(self.closed_indices)]
            violation |= ((closed < -1e-5) | (closed > 1.0 + 1e-5)).any(dim=-1)
            item["violations"] += int(violation.sum())

            variance = (
                moments["total_variance"][valid, index]
                * self.variance_scale.to(mean.device)
            )[:, continuous].clamp_min(1e-12)
            continuous_error = error[:, continuous]
            item["nll_sum"] += float(
                (
                    0.5
                    * (
                        np.log(2.0 * np.pi)
                        + variance.log()
                        + continuous_error.square() / variance
                    )
                )
                .sum()
                .cpu()
            )
            for name, z_value in _COVERAGE_Z.items():
                item["coverage"][name] += int(
                    (continuous_error.abs() <= z_value * variance.sqrt()).sum()
                )
            actual_error = torch.sqrt(normalized.square().mean(dim=-1))
            epistemic = _normalized_epistemic(
                moments["epistemic_variance"][valid, index], std, continuous
            )
            ood_epistemic = _normalized_epistemic(
                ood_moments["epistemic_variance"][valid, index], std, continuous
            )
            item["actual_error"].append(actual_error.float().cpu().numpy())
            item["epistemic"].append(epistemic.float().cpu().numpy())
            item["ood_epistemic"].append(ood_epistemic.float().cpu().numpy())

            if horizon == self.event_horizon:
                self._update_event(predictions, moments, batch, valid, index)

        self.risk_id.append(risk["total"].float().cpu().numpy().reshape(-1))
        self.risk_ood.append(ood_risk["total"].float().cpu().numpy().reshape(-1))

    def _update_event(
        self,
        predictions: Any,
        moments: Mapping[str, Tensor],
        batch: Mapping[str, Tensor],
        valid: Tensor,
        index: int,
    ) -> None:
        current = batch["states"][valid, -1]
        target = batch["target_states"][valid, index]
        progress = batch["response_progress"][valid, index, 0]
        current_speed = _agent_linear_speed(current)
        target_slowdown = (current_speed - _agent_linear_speed(target)).clamp_min(0.0)
        asymmetry = (target_slowdown[:, 0] - target_slowdown[:, 1]).abs()
        selected = (
            (progress >= self.event_progress_min)
            & (target_slowdown.max(dim=-1).values >= self.event_slowdown_min)
            & (asymmetry >= self.event_asymmetry_min)
        )
        if not bool(selected.any()):
            return
        actual_agent = target_slowdown[selected].argmax(dim=-1)
        selected_asymmetry = asymmetry[selected]
        member_states = predictions.next_state_mean[:, valid, index][:, selected]
        member_slowdown = (
            current_speed[selected].unsqueeze(0) - _agent_linear_speed(member_states)
        ).clamp_min(0.0)
        member_difference = (member_slowdown[..., 0] - member_slowdown[..., 1]).abs()
        ambiguity_threshold = (
            self.event_ambiguity_fraction * selected_asymmetry
        ).unsqueeze(0)
        member_ambiguous = member_difference < ambiguity_threshold
        member_correct = member_slowdown.argmax(dim=-1) == actual_agent.unsqueeze(0)

        mean_state = moments["mean"][valid, index][selected]
        mean_slowdown = (
            current_speed[selected] - _agent_linear_speed(mean_state)
        ).clamp_min(0.0)
        mean_ambiguous = (
            (mean_slowdown[:, 0] - mean_slowdown[:, 1]).abs()
            < self.event_ambiguity_fraction * selected_asymmetry
        )
        mean_correct = mean_slowdown.argmax(dim=-1) == actual_agent
        samples = int(selected.sum())
        self.event["samples"] += samples
        self.event["member_predictions"] += int(member_ambiguous.numel())
        self.event["member_ambiguous"] += int(member_ambiguous.sum())
        self.event["member_correct"] += int(member_correct.sum())
        self.event["ensemble_mean_ambiguous"] += int(mean_ambiguous.sum())
        self.event["ensemble_mean_correct"] += int(mean_correct.sum())

    def finish(self) -> dict[str, Any]:
        exact: dict[str, Any] = {}
        ood_exact: dict[str, Any] = {}
        for horizon, item in self.sums.items():
            values = int(item["continuous_values"])
            samples = int(item["samples"])
            if not samples or not values:
                exact[str(horizon)] = {"samples": 0}
                ood_exact[str(horizon)] = {"samples": 0}
                continue
            actual_error = np.concatenate(item["actual_error"])
            epistemic = np.concatenate(item["epistemic"])
            ood_epistemic = np.concatenate(item["ood_epistemic"])
            exact[str(horizon)] = {
                "samples": samples,
                "ensemble_mean_continuous_nrmse": float(
                    np.sqrt(item["ensemble_normalized_squared_error"] / values)
                ),
                "member0_continuous_nrmse": float(
                    np.sqrt(item["member0_normalized_squared_error"] / values)
                ),
                "teacher_forcing_continuous_nrmse": (
                    float(
                        np.sqrt(
                            item["teacher_normalized_squared_error"]
                            / item["teacher_values"]
                        )
                    )
                    if item["teacher_values"] > 0
                    else None
                ),
                "finite_rollout_rate": item["finite"] / samples,
                "state_constraint_violation_rate": item["violations"] / samples,
                "gaussian_nll": item["nll_sum"] / values,
                "interval_coverage": {
                    name: count / values for name, count in item["coverage"].items()
                },
                "uncertainty_error_spearman": _spearman(epistemic, actual_error),
                "mean_epistemic_score": float(epistemic.mean()),
                "mean_actual_error": float(actual_error.mean()),
            }
            id_mean = float(epistemic.mean())
            ood_mean = float(ood_epistemic.mean())
            labels = np.concatenate(
                (np.zeros_like(epistemic), np.ones_like(ood_epistemic))
            )
            scores = np.concatenate((epistemic, ood_epistemic))
            ood_exact[str(horizon)] = {
                "samples_per_class": int(epistemic.size),
                "auroc": _binary_auroc(labels, scores),
                "mean_id_epistemic": id_mean,
                "mean_ood_epistemic": ood_mean,
                "epistemic_ratio": ood_mean / max(id_mean, 1e-12),
            }
        event_samples = int(self.event["samples"])
        member_predictions = int(self.event["member_predictions"])
        event = {
            "horizon": self.event_horizon,
            "samples": event_samples,
            "available": event_samples > 0,
            "member_dominant_agent_accuracy": (
                self.event["member_correct"] / member_predictions
                if member_predictions
                else None
            ),
            "member_ambiguous_braking_rate": (
                self.event["member_ambiguous"] / member_predictions
                if member_predictions
                else None
            ),
            "ensemble_mean_dominant_agent_accuracy": (
                self.event["ensemble_mean_correct"] / event_samples
                if event_samples
                else None
            ),
            "ensemble_mean_ambiguous_braking_rate": (
                self.event["ensemble_mean_ambiguous"] / event_samples
                if event_samples
                else None
            ),
        }
        risk_id = np.concatenate(self.risk_id)
        risk_ood = np.concatenate(self.risk_ood)
        return {
            "exact_horizon": exact,
            "ood": {"exact_horizon": ood_exact},
            "event_aligned": event,
            "risk": {
                "id_mean": float(risk_id.mean()),
                "id_p95": float(np.quantile(risk_id, 0.95)),
                "ood_mean": float(risk_ood.mean()),
                "ood_p95": float(np.quantile(risk_ood, 0.95)),
                "ood_to_id_mean_ratio": float(
                    risk_ood.mean() / max(risk_id.mean(), 1e-12)
                ),
            },
        }


def _batch_to_device(
    batch: Mapping[str, Tensor], device: torch.device
) -> dict[str, Tensor]:
    return {
        name: value.to(device, non_blocking=True)
        for name, value in batch.items()
        if isinstance(value, Tensor)
    }


def _history(batch: Mapping[str, Tensor]) -> WorldModelSequenceInputs:
    return WorldModelSequenceInputs(
        batch["states"], batch["past_actions"], batch["valid_mask"]
    )


def _state_error(
    predicted: Tensor, target: Tensor, yaw_indices: tuple[int, ...]
) -> Tensor:
    error = predicted - target
    for yaw_index in yaw_indices:
        error = _replace_column(error, yaw_index, wrap_to_pi(error[..., yaw_index]))
    return error


def _normalized_epistemic(
    variance: Tensor, state_std: Tensor, continuous_indices: Sequence[int]
) -> Tensor:
    indices = list(continuous_indices)
    return torch.sqrt((variance[:, indices] / state_std[indices].square()).mean(dim=-1))


def _agent_linear_speed(state: Tensor) -> Tensor:
    first = torch.linalg.vector_norm(state[..., [3, 4]], dim=-1)
    second = torch.linalg.vector_norm(state[..., [14, 15]], dim=-1)
    return torch.stack((first, second), dim=-1)


def _binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = labels == 1
    positive_count = int(positive.sum())
    negative_count = int((~positive).sum())
    if not positive_count or not negative_count:
        return None
    ranks = _average_ranks(scores)
    return float(
        (ranks[positive].sum() - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def _spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if first.size < 2 or first.size != second.size:
        return None
    first_rank = _average_ranks(first)
    second_rank = _average_ranks(second)
    first_rank -= first_rank.mean()
    second_rank -= second_rank.mean()
    denominator = np.sqrt(
        np.square(first_rank).sum() * np.square(second_rank).sum()
    )
    if denominator <= 0.0:
        return None
    return float(np.dot(first_rank, second_rank) / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    cursor = 0
    while cursor < values.size:
        stop = cursor + 1
        while stop < values.size and values[order[stop]] == values[order[cursor]]:
            stop += 1
        ranks[order[cursor:stop]] = 0.5 * (cursor + stop - 1) + 1.0
        cursor = stop
    return ranks


def _replace_column(value: Tensor, index: int, column: Tensor) -> Tensor:
    parts = list(value.split(1, dim=-1))
    parts[index] = column.unsqueeze(-1)
    return torch.cat(parts, dim=-1)


__all__ = [
    "OODActionPerturbation",
    "evaluate_rwm_u",
    "fit_variance_calibration",
]
