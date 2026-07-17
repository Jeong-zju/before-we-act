"""Offline and closed-loop checks for the action-flow warm-up."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor

from models.wam import (
    ActionPrior,
    RWMARWorldModel,
    StatefulActionFlow,
    WorldModelSequenceInputs,
)
from models.wam.rollout import wrap_to_pi
from train.action_flow import action_prior_teacher_chunk, synthetic_shifted_warm_start

ProgressCallback = Callable[[Mapping[str, int]], None]


@torch.inference_mode()
def evaluate_action_flow_offline(
    flow: StatefulActionFlow,
    world_model: RWMARWorldModel,
    loader: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
    execution_steps: int = 2,
    solver_steps: int = 4,
    solver: str = "euler",
    max_batches: int = -1,
    progress: ProgressCallback | None = None,
    action_prior: ActionPrior | None = None,
) -> dict[str, Any]:
    """Evaluate full action chunks and valid expert-conditioned world targets."""

    flow.to(device).eval()
    world_model.to(device).eval()
    action_squared = 0.0
    warm_squared = 0.0
    first_squared = 0.0
    action_values = 0
    first_values = 0
    prior_teacher_squared = 0.0
    prior_teacher_values = 0
    prior_teacher_chunk_squared = 0.0
    prior_teacher_warm_squared = 0.0
    prior_teacher_chunk_values = 0
    prior_teacher_horizon_squared = np.zeros(flow.config.horizon, dtype=np.float64)
    prior_teacher_horizon_values = np.zeros(flow.config.horizon, dtype=np.int64)
    selected_samples = 0
    samples = 0
    generated_values = 0
    non_finite_actions = 0
    out_of_bounds_actions = 0
    horizon_squared = np.zeros(flow.config.horizon, dtype=np.float64)
    horizon_values = np.zeros(flow.config.horizon, dtype=np.int64)
    continuous = world_model.continuous_state_mask
    world_squared = 0.0
    world_values = 0
    world_horizon_squared = np.zeros(flow.config.horizon, dtype=np.float64)
    world_horizon_values = np.zeros(flow.config.horizon, dtype=np.int64)
    generated_world_values = 0
    generated_world_non_finite = 0
    generated_failure_sum = 0.0
    generated_failure_max = 0.0
    generated_robot_distance_max = 0.0
    for batch_index, raw_batch in enumerate(loader, start=1):
        batch = {
            name: value.to(device, non_blocking=True)
            for name, value in raw_batch.items()
        }
        history = WorldModelSequenceInputs(
            states=batch["states"],
            past_actions=batch["past_actions"],
            valid_mask=batch["valid_mask"],
        )
        hidden, current_state, features = world_model.encode_planning_history(history)
        target = batch["candidate_actions"][:, : flow.config.horizon]
        cold = flow.generate(
            features,
            solver_steps=solver_steps,
            solver=solver,
        )
        warm_seed = synthetic_shifted_warm_start(target, execution_steps)
        warm = flow.generate(
            features,
            initial_actions=warm_seed,
            solver_steps=solver_steps,
            solver=solver,
        )
        quality = batch.get("action_quality_weights")
        selected = (
            torch.ones(target.shape[0], device=device, dtype=torch.bool)
            if quality is None
            else quality.reshape(-1) > 0.0
        )
        samples += int(target.shape[0])
        selected_count = int(selected.sum().cpu())
        selected_samples += selected_count
        if selected_count:
            error = (cold[selected] - target[selected]).square()
            warm_error = (warm[selected] - target[selected]).square()
            action_squared += float(error.sum().cpu())
            warm_squared += float(warm_error.sum().cpu())
            first_squared += float(error[:, 0].sum().cpu())
            action_values += int(error.numel())
            first_values += int(error[:, 0].numel())
            horizon_squared += error.sum(dim=(0, 2)).cpu().numpy()
            horizon_values += selected_count * flow.config.action_dim
            if action_prior is not None:
                teacher = action_prior_teacher_chunk(
                    world_model,
                    action_prior,
                    hidden,
                    current_state,
                    steps=flow.config.horizon,
                )
                prior_action = teacher[:, 0]
                prior_teacher_squared += float(
                    (cold[selected, 0] - prior_action[selected]).square().sum().cpu()
                )
                prior_teacher_values += int(prior_action[selected].numel())
                teacher_error = (cold[selected] - teacher[selected]).square()
                prior_teacher_chunk_squared += float(teacher_error.sum().cpu())
                prior_teacher_chunk_values += int(teacher_error.numel())
                prior_teacher_horizon_squared += teacher_error.sum(
                    dim=(0, 2)
                ).cpu().numpy()
                prior_teacher_horizon_values += selected_count * flow.config.action_dim
                teacher_warm_seed = synthetic_shifted_warm_start(
                    teacher, execution_steps
                )
                teacher_warm = flow.generate(
                    features,
                    initial_actions=teacher_warm_seed,
                    solver_steps=solver_steps,
                    solver=solver,
                )
                prior_teacher_warm_squared += float(
                    (teacher_warm[selected] - teacher[selected]).square().sum().cpu()
                )
        generated_values += int(cold.numel())
        non_finite_actions += int((~torch.isfinite(cold)).sum().cpu())
        out_of_bounds_actions += int(((cold < -1.0) | (cold > 1.0)).sum().cpu())

        expert_world = world_model.predict_from_encoded_history(
            hidden, current_state, target, sample_state=False
        )
        generated_world = world_model.predict_from_encoded_history(
            hidden, current_state, cold, sample_state=False
        )
        difference = expert_world.next_state_mean - batch["target_states"][
            :, : flow.config.horizon
        ]
        for yaw_index in world_model.config.yaw_indices:
            difference[..., yaw_index] = wrap_to_pi(difference[..., yaw_index])
        normalized = difference[..., continuous] / world_model.features.state_std[
            continuous
        ]
        squared = normalized.square()
        world_squared += float(squared.sum().cpu())
        world_values += int(squared.numel())
        world_horizon_squared += squared.sum(dim=(0, 2)).cpu().numpy()
        world_horizon_values += target.shape[0] * int(continuous.sum().cpu())

        generated_state = generated_world.next_state_mean
        generated_world_values += int(generated_state.numel())
        generated_world_non_finite += int(
            (~torch.isfinite(generated_state)).sum().cpu()
        )
        failure = generated_world.failure_logit.sigmoid()
        generated_failure_sum += float(failure.sum().cpu())
        generated_failure_max = max(
            generated_failure_max, float(failure.max().cpu())
        )
        distance = torch.linalg.vector_norm(
            generated_state[..., 0:2] - generated_state[..., 11:13], dim=-1
        )
        generated_robot_distance_max = max(
            generated_robot_distance_max, float(distance.max().cpu())
        )
        if progress is not None:
            progress({"batch": batch_index, "samples": samples})
        if max_batches > 0 and batch_index >= max_batches:
            break
    if not samples or not selected_samples or not world_values:
        raise RuntimeError("offline action-flow evaluation has no usable samples")
    return {
        "samples": samples,
        "selected_action_samples": selected_samples,
        "cold_action_chunk_rmse": float(np.sqrt(action_squared / action_values)),
        "warm_action_chunk_rmse": float(np.sqrt(warm_squared / action_values)),
        "cold_first_action_rmse": float(np.sqrt(first_squared / first_values)),
        "action_prior_teacher_first_action_rmse": (
            float(np.sqrt(prior_teacher_squared / prior_teacher_values))
            if prior_teacher_values
            else None
        ),
        "action_prior_teacher_chunk_rmse": (
            float(np.sqrt(prior_teacher_chunk_squared / prior_teacher_chunk_values))
            if prior_teacher_chunk_values
            else None
        ),
        "action_prior_teacher_warm_chunk_rmse": (
            float(np.sqrt(prior_teacher_warm_squared / prior_teacher_chunk_values))
            if prior_teacher_chunk_values
            else None
        ),
        "action_prior_teacher_horizon_rmse": (
            [
                float(np.sqrt(value / count))
                for value, count in zip(
                    prior_teacher_horizon_squared,
                    prior_teacher_horizon_values,
                    strict=True,
                )
            ]
            if prior_teacher_chunk_values
            else None
        ),
        "cold_action_horizon_rmse": [
            float(np.sqrt(value / count))
            for value, count in zip(
                horizon_squared, horizon_values, strict=True
            )
        ],
        "generated_action_non_finite": non_finite_actions,
        "generated_action_out_of_bounds": out_of_bounds_actions,
        "generated_action_values": generated_values,
        "expert_action_world_state_nrmse": float(
            np.sqrt(world_squared / world_values)
        ),
        "expert_action_world_horizon_nrmse": [
            float(np.sqrt(value / count))
            for value, count in zip(
                world_horizon_squared, world_horizon_values, strict=True
            )
        ],
        "expert_action_world_target_source": "dataset_next_state",
        "generated_action_world_target_source": None,
        "generated_action_demo_state_is_ground_truth": False,
        "generated_action_world_non_finite": generated_world_non_finite,
        "generated_action_world_values": generated_world_values,
        "generated_action_world_failure_probability_mean": (
            generated_failure_sum
            / (samples * flow.config.horizon)
        ),
        "generated_action_world_failure_probability_max": generated_failure_max,
        "generated_action_world_robot_distance_max": generated_robot_distance_max,
    }


def action_flow_acceptance_report(
    metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    held_out_seed_overlap: int,
    minimum_episodes: int,
    minimum_success_rate: float,
    maximum_prior_regression: float,
    smoke: bool,
) -> dict[str, Any]:
    """Apply action-flow direct-execution checks without allowing fallback masking."""

    suites: dict[str, Any] = {}
    all_passed = held_out_seed_overlap == 0
    for suite_name in ("standard", "challenge"):
        policies = metrics.get(suite_name, {})
        flow = policies.get("joint_wam_flow")
        prior = policies.get("action_prior")
        if not isinstance(flow, Mapping) or not isinstance(prior, Mapping):
            suites[suite_name] = {
                "passed": False,
                "checks": {"required_policies_present": False},
            }
            all_passed = False
            continue
        flow_rate = float(flow["success_rate"])
        prior_rate = float(prior["success_rate"])
        mode_counts = dict(flow.get("planner_modes", {}))
        mode_total = sum(int(value) for value in mode_counts.values())
        direct_rate = (
            int(mode_counts.get("joint_wam_flow", 0)) / mode_total
            if mode_total
            else 0.0
        )
        residual = dict(flow.get("applied_flow_residual", {}))
        if smoke:
            success_check = flow_rate >= 2.0 / 3.0
            regression_check = True
        else:
            success_check = flow_rate + 1e-12 >= minimum_success_rate
            regression_check = (
                prior_rate - flow_rate <= maximum_prior_regression + 1e-12
            )
        checks = {
            "minimum_episodes": int(flow["episodes"]) >= minimum_episodes
            and int(prior["episodes"]) >= minimum_episodes,
            "minimum_direct_success_rate": success_check,
            "prior_success_regression": regression_check,
            "all_actions_finite_and_bounded": bool(
                flow["all_actions_finite_and_bounded"]
            ),
            "no_privileged_state_leakage": not bool(
                flow["privileged_state_leakage"]
            ),
            "direct_flow_execution_rate": direct_rate >= 1.0 - 1e-12,
            "flow_residual_applied": int(residual.get("samples", 0)) > 0
            and float(residual.get("max") or 0.0) > 1e-8,
            "fallback_disabled": float(flow["fallback_trigger_rate"]) == 0.0,
            "no_premature_stationary_success": int(
                flow["premature_stationary_successes"]
            )
            == 0,
        }
        passed = all(checks.values())
        suites[suite_name] = {
            "passed": passed,
            "checks": checks,
            "flow_success_rate": flow_rate,
            "prior_success_rate": prior_rate,
            "prior_regression": prior_rate - flow_rate,
            "direct_flow_execution_rate": direct_rate,
        }
        all_passed = all_passed and passed
    return {
        "stage": "action_flow_warmup",
        "protocol": "smoke" if smoke else "20_seed_gate",
        "passed": all_passed,
        "held_out_seed_overlap": int(held_out_seed_overlap),
        "thresholds": {
            "minimum_episodes": minimum_episodes,
            "minimum_success_rate": minimum_success_rate,
            "maximum_prior_regression": maximum_prior_regression,
            "direct_flow_execution_rate": 1.0,
            "fallback_trigger_rate": 0.0,
        },
        "suites": suites,
    }


__all__ = [
    "evaluate_action_flow_offline",
    "action_flow_acceptance_report",
]
