"""Phase 1 RWM-AR open-loop metrics and Phase 0 recursive baselines."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from models import OneStepMLPWorldModel, WorldModelInputs
from models.wam import NormalizationStats, RWMARWorldModel, WorldModelSequenceInputs
from models.wam.rollout import wrap_to_pi
from train.phase0_baselines import (
    binary_classification_metrics,
    calibrate_binary_threshold,
)

ProgressCallback = Callable[[Mapping[str, int]], None]

STATE_GROUPS = {
    "base_pose": (0, 1, 2, 11, 12, 13),
    "base_velocity": (3, 4, 5, 14, 15, 16),
    "gripper": (6, 7, 17, 18),
    "base_effort": (8, 9, 10, 19, 20, 21),
}


@dataclass
class Phase0RecursiveBaseline:
    model: OneStepMLPWorldModel
    stats: NormalizationStats
    yaw_indices: tuple[int, ...] = (2, 13)

    @torch.no_grad()
    def rollout(self, current: Tensor, actions: Tensor) -> Tensor:
        normalization = self.stats.tensors(current.device)
        predictions: list[Tensor] = []
        state = current
        for step in range(actions.shape[1]):
            normalized_state = (state - normalization["state_mean"]) / normalization[
                "state_std"
            ]
            normalized_action = (
                actions[:, step] - normalization["action_mean"]
            ) / normalization["action_std"]
            output = self.model(
                WorldModelInputs(state=normalized_state, action=normalized_action)
            )
            delta = (
                output.next_state * normalization["delta_std"]
                + normalization["delta_mean"]
            )
            state = state + delta
            for yaw_index in self.yaw_indices:
                state = _replace_column(
                    state, yaw_index, wrap_to_pi(state[..., yaw_index])
                )
            predictions.append(state)
        return torch.stack(predictions, dim=1)


@torch.no_grad()
def evaluate_open_loop(
    model: RWMARWorldModel,
    loader: Iterable[Mapping[str, Tensor]],
    stats: NormalizationStats,
    *,
    device: torch.device,
    horizons: Sequence[int] = (1, 5, 10, 20, 40),
    phase0_mlp: Phase0RecursiveBaseline | None = None,
    classification_thresholds: Mapping[str, float] | None = None,
    calibrate_thresholds: bool = False,
    max_batches: int = -1,
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    horizons = tuple(sorted(set(int(item) for item in horizons)))
    if not horizons or horizons[0] <= 0:
        raise ValueError("horizons must be positive")
    max_horizon = max(horizons)
    model.to(device).eval()
    if phase0_mlp is not None:
        phase0_mlp.model.to(device).eval()
    accumulators = {
        "rwm_ar": _OpenLoopAccumulator(horizons, stats),
        "constant_velocity": _OpenLoopAccumulator(horizons, stats),
    }
    if phase0_mlp is not None:
        accumulators["phase0_mlp_recursive"] = _OpenLoopAccumulator(horizons, stats)
    example: dict[str, np.ndarray] = {}
    batches = 0
    for batch_index, raw_batch in enumerate(loader, start=1):
        batch = {
            name: value.to(device, non_blocking=True)
            for name, value in raw_batch.items()
            if isinstance(value, Tensor)
        }
        if batch["candidate_actions"].shape[1] < max_horizon:
            raise ValueError("evaluation dataset forecast horizon is too short")
        history = WorldModelSequenceInputs(
            batch["states"], batch["past_actions"], batch["valid_mask"]
        )
        actions = batch["candidate_actions"][:, :max_horizon]
        rwm = model.predict(history, actions)
        current = batch["states"][:, -1]
        predictions = {
            "rwm_ar": rwm.next_state_mean,
            "constant_velocity": constant_velocity_rollout(current, max_horizon),
        }
        if phase0_mlp is not None:
            predictions["phase0_mlp_recursive"] = phase0_mlp.rollout(current, actions)
        for name, state_predictions in predictions.items():
            accumulators[name].update(
                state_predictions,
                batch,
                outcome_predictions=(rwm if name == "rwm_ar" else None),
            )
        if not example:
            valid_steps = int(batch["forecast_mask"][0, :max_horizon].sum())
            example = {
                "steps": np.arange(valid_steps + 1),
                "actual": torch.cat(
                    (
                        current[0:1, None],
                        batch["target_states"][0:1, :valid_steps],
                    ),
                    dim=1,
                )[0]
                .cpu()
                .numpy(),
            }
            for name, value in predictions.items():
                example[name] = (
                    torch.cat((current[0:1, None], value[0:1, :valid_steps]), dim=1)[0]
                    .cpu()
                    .numpy()
                )
        batches += 1
        if progress is not None:
            progress({"batch": batch_index})
        if max_batches > 0 and batch_index >= max_batches:
            break
    if batches == 0:
        raise ValueError("open-loop evaluation received no batches")

    rwm_accumulator = accumulators["rwm_ar"]
    calibration: dict[str, Any] = {}
    if calibrate_thresholds:
        calibration = {
            label: calibrate_binary_threshold(
                *rwm_accumulator.classification_arrays(label, horizon=1)
            )
            for label in ("done", "success", "failure")
        }
        thresholds = {
            label: float(values["threshold"]) for label, values in calibration.items()
        }
    else:
        thresholds = {
            label: float((classification_thresholds or {}).get(label, 0.5))
            for label in ("done", "success", "failure")
        }
    result = {
        "batches": batches,
        "horizons": list(horizons),
        "models": {
            name: accumulator.finish(
                thresholds=thresholds if name == "rwm_ar" else None
            )
            for name, accumulator in accumulators.items()
        },
    }
    if calibration:
        result["outcome_calibration"] = calibration
    return result, example


def constant_velocity_rollout(current: Tensor, horizon: int) -> Tensor:
    state = current
    predictions: list[Tensor] = []
    for _ in range(horizon):
        state = state.clone()
        for offset in (0, 11):
            state[..., offset] += 0.05 * state[..., offset + 3]
            state[..., offset + 1] += 0.05 * state[..., offset + 4]
            state[..., offset + 2] = wrap_to_pi(
                state[..., offset + 2] + 0.05 * state[..., offset + 5]
            )
        predictions.append(state)
    return torch.stack(predictions, dim=1)


class _OpenLoopAccumulator:
    def __init__(self, horizons: Sequence[int], stats: NormalizationStats) -> None:
        self.horizons = tuple(horizons)
        self.state_mean = torch.as_tensor(stats.state_mean, dtype=torch.float32)
        self.state_std = torch.as_tensor(stats.state_std, dtype=torch.float32)
        self.gripper_closed_indices = (7, 18)
        self.continuous_indices = tuple(
            index
            for index in range(int(self.state_std.numel()))
            if index not in self.gripper_closed_indices
        )
        self.sums = {
            horizon: {
                "samples": 0,
                "squared_error": 0.0,
                "normalized_squared_error": 0.0,
                "continuous_normalized_squared_error": 0.0,
                "continuous_values": 0,
                "gripper_closed_squared_error": 0.0,
                "gripper_closed_values": 0,
                "reward_absolute_error": 0.0,
                "constraint_violations": 0,
                "finite_rollouts": 0,
                "group_squared_error": {name: 0.0 for name in STATE_GROUPS},
                "event_velocity_squared_error": 0.0,
                "event_velocity_values": 0,
            }
            for horizon in self.horizons
        }
        self.classification: dict[int, dict[str, dict[str, list[Tensor]]]] = {
            horizon: {
                label: {"labels": [], "logits": []}
                for label in ("done", "success", "failure")
            }
            for horizon in self.horizons
        }

    def update(
        self,
        predictions: Tensor,
        batch: Mapping[str, Tensor],
        *,
        outcome_predictions: Any | None,
    ) -> None:
        std = self.state_std.to(predictions.device)
        initial_velocity = batch["states"][:, -1, list(STATE_GROUPS["base_velocity"])]
        for horizon in self.horizons:
            index = horizon - 1
            valid = batch["forecast_mask"][:, index]
            if not bool(valid.any()):
                continue
            predicted = predictions[valid, index]
            target = batch["target_states"][valid, index]
            error = predicted - target
            for yaw_index in (2, 13):
                error = _replace_column(
                    error, yaw_index, wrap_to_pi(error[..., yaw_index])
                )
            item = self.sums[horizon]
            count = int(valid.sum())
            item["samples"] += count
            item["squared_error"] += float(error.square().sum().cpu())
            item["normalized_squared_error"] += float(
                (error / std).square().sum().cpu()
            )
            continuous_error = error[:, list(self.continuous_indices)]
            continuous_std = std[list(self.continuous_indices)]
            item["continuous_normalized_squared_error"] += float(
                (continuous_error / continuous_std).square().sum().cpu()
            )
            item["continuous_values"] += int(continuous_error.numel())
            closed_error = error[:, list(self.gripper_closed_indices)]
            item["gripper_closed_squared_error"] += float(
                closed_error.square().sum().cpu()
            )
            item["gripper_closed_values"] += int(closed_error.numel())
            finite = torch.isfinite(predicted).all(dim=-1)
            item["finite_rollouts"] += int(finite.sum())
            mean = self.state_mean.to(predictions.device)
            standardized = (predicted - mean) / std
            # A permissive 20-sigma bound catches explosions without treating
            # ordinary task distribution shift as a hard physical constraint.
            violation = (~finite) | (standardized.abs() > 20.0).any(dim=-1)
            closed = predicted[:, [7, 18]]
            violation |= ((closed < -1e-5) | (closed > 1.0 + 1e-5)).any(dim=-1)
            item["constraint_violations"] += int(violation.sum())
            for group, indices in STATE_GROUPS.items():
                item["group_squared_error"][group] += float(
                    error[:, list(indices)].square().sum().cpu()
                )
            target_velocity = target[:, list(STATE_GROUPS["base_velocity"])]
            decelerating = (
                target_velocity.norm(dim=-1)
                < initial_velocity[valid].norm(dim=-1) - 1e-4
            )
            event = decelerating | (batch["response_progress"][valid, index, 0] > 0.0)
            if bool(event.any()):
                velocity_error = error[:, list(STATE_GROUPS["base_velocity"])][event]
                item["event_velocity_squared_error"] += float(
                    velocity_error.square().sum().cpu()
                )
                item["event_velocity_values"] += int(velocity_error.numel())
            if outcome_predictions is not None:
                item["reward_absolute_error"] += float(
                    (
                        outcome_predictions.reward[valid, index]
                        - batch["rewards"][valid, index]
                    )
                    .abs()
                    .sum()
                    .cpu()
                )
                for label, field in (
                    ("done", "dones"),
                    ("success", "successes"),
                    ("failure", "failures"),
                ):
                    self.classification[horizon][label]["labels"].append(
                        batch[field][valid, index].reshape(-1).cpu()
                    )
                    self.classification[horizon][label]["logits"].append(
                        getattr(outcome_predictions, f"{label}_logit")[valid, index]
                        .reshape(-1)
                        .cpu()
                    )

    def classification_arrays(
        self, label: str, *, horizon: int
    ) -> tuple[np.ndarray, np.ndarray]:
        values = self.classification[horizon][label]
        if not values["labels"]:
            return np.empty(0), np.empty(0)
        return (
            torch.cat(values["labels"]).numpy(),
            torch.cat(values["logits"]).numpy(),
        )

    def finish(self, *, thresholds: Mapping[str, float] | None) -> dict[str, Any]:
        metrics: dict[str, Any] = {"exact_horizon": {}}
        state_dim = int(self.state_std.numel())
        for horizon, item in self.sums.items():
            samples = int(item["samples"])
            if samples == 0:
                metrics["exact_horizon"][str(horizon)] = {"samples": 0}
                continue
            values = {
                "samples": samples,
                "state_rmse": float(
                    np.sqrt(item["squared_error"] / (samples * state_dim))
                ),
                "state_nrmse": float(
                    np.sqrt(item["normalized_squared_error"] / (samples * state_dim))
                ),
                "continuous_state_nrmse": float(
                    np.sqrt(
                        item["continuous_normalized_squared_error"]
                        / item["continuous_values"]
                    )
                ),
                "gripper_closed_rmse": float(
                    np.sqrt(
                        item["gripper_closed_squared_error"]
                        / item["gripper_closed_values"]
                    )
                ),
                "finite_rollout_rate": item["finite_rollouts"] / samples,
                "state_constraint_violation_rate": item["constraint_violations"]
                / samples,
                "state_group_rmse": {
                    group: float(np.sqrt(value / (samples * len(STATE_GROUPS[group]))))
                    for group, value in item["group_squared_error"].items()
                },
            }
            if self.classification[horizon]["done"]["labels"]:
                values["reward_mae"] = item["reward_absolute_error"] / samples
            if item["event_velocity_values"]:
                values["event_velocity_rmse"] = float(
                    np.sqrt(
                        item["event_velocity_squared_error"]
                        / item["event_velocity_values"]
                    )
                )
            if thresholds is not None:
                values["classification"] = {
                    label: binary_classification_metrics(
                        *self.classification_arrays(label, horizon=horizon),
                        threshold=float(thresholds[label]),
                    )
                    for label in ("done", "success", "failure")
                }
            metrics["exact_horizon"][str(horizon)] = values
        return metrics


def _replace_column(value: Tensor, index: int, column: Tensor) -> Tensor:
    parts = list(value.split(1, dim=-1))
    parts[index] = column.unsqueeze(-1)
    return torch.cat(parts, dim=-1)


__all__ = [
    "Phase0RecursiveBaseline",
    "constant_velocity_rollout",
    "evaluate_open_loop",
]
