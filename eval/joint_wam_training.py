"""Offline training checks for the Joint World-Action Model.

Generated-action consistency deliberately uses the frozen initialization world-model
prediction under the *same* action chunk as its target.  Dataset future states
are valid targets only for the recorded expert action and are never relabelled
as ground truth for a generated or deployed action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor

from eval.action_flow import action_flow_acceptance_report
from models.wam import RWMARWorldModel, StatefulActionFlow, WorldModelSequenceInputs
from models.wam.recurrent_dynamics import RWMARRolloutPredictions
from models.wam.rollout import wrap_to_pi
from train.action_flow import synthetic_shifted_warm_start
from train.joint_wam_checkpointing import GENERATED_ACTION_WORLD_TARGET_SOURCE

ProgressCallback = Callable[[Mapping[str, int]], None]

_EXPERT_TARGET_SOURCE = "dataset_next_state_under_same_expert_action"
_GENERATED_TARGET_SOURCE = GENERATED_ACTION_WORLD_TARGET_SOURCE
_DEPLOYED_TARGET_SOURCE = "frozen_world_model_same_deployed_actions"
_CONSISTENCY_MODES = (
    "cold_generated",
    "warm_generated",
    "cold_deployed",
    "warm_deployed",
)


@dataclass
class _ActionStats:
    bound_atol: float
    values: int = 0
    non_finite: int = 0
    out_of_bounds: int = 0

    def add(self, actions: Tensor) -> None:
        finite = torch.isfinite(actions)
        self.values += int(actions.numel())
        self.non_finite += int((~finite).sum().cpu())
        out = finite & (
            (actions < -1.0 - self.bound_atol) | (actions > 1.0 + self.bound_atol)
        )
        self.out_of_bounds += int(out.sum().cpu())

    def report(self) -> dict[str, Any]:
        finite = self.values > 0 and self.non_finite == 0
        bounded = self.values > 0 and self.out_of_bounds == 0
        return {
            "values": self.values,
            "non_finite": self.non_finite,
            "out_of_bounds": self.out_of_bounds,
            "all_finite": finite,
            "all_bounded": bounded,
            "finite_and_bounded": finite and bounded,
        }


@dataclass
class _StateNRMSE:
    horizon: int
    squared: float = 0.0
    values: int = 0
    non_finite: int = 0
    horizon_squared: np.ndarray = field(init=False)
    horizon_values: np.ndarray = field(init=False)
    horizon_non_finite: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.horizon_squared = np.zeros(self.horizon, dtype=np.float64)
        self.horizon_values = np.zeros(self.horizon, dtype=np.int64)
        self.horizon_non_finite = np.zeros(self.horizon, dtype=np.int64)

    def add(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        mask: Tensor,
        state_std: Tensor,
        continuous: Tensor,
        yaw_indices: tuple[int, ...],
    ) -> None:
        if prediction.shape != target.shape:
            raise ValueError("state prediction and target shapes differ")
        if prediction.shape[1] != self.horizon:
            raise ValueError("state prediction horizon differs from evaluator horizon")
        if mask.shape != prediction.shape[:2]:
            raise ValueError("state metric mask must have shape [B,H]")
        difference = prediction - target
        for yaw_index in yaw_indices:
            difference[..., yaw_index] = wrap_to_pi(difference[..., yaw_index])
        normalized = difference[..., continuous] / state_std[continuous].to(difference)
        selected = mask.to(dtype=torch.bool).unsqueeze(-1).expand_as(normalized)
        for step in range(self.horizon):
            step_values = normalized[:, step][selected[:, step]]
            expected = int(step_values.numel())
            if expected == 0:
                continue
            finite = torch.isfinite(step_values)
            non_finite = int((~finite).sum().cpu())
            self.horizon_values[step] += expected
            self.horizon_non_finite[step] += non_finite
            if bool(finite.any()):
                square = step_values[finite].double().square().sum()
                self.horizon_squared[step] += float(square.cpu())
            self.values += expected
            self.non_finite += non_finite
        self.squared = float(self.horizon_squared.sum())

    def report(self) -> dict[str, Any]:
        rmse = (
            math.sqrt(self.squared / self.values)
            if self.values > 0 and self.non_finite == 0
            else None
        )
        horizon_rmse: list[float | None] = []
        for square, values, non_finite in zip(
            self.horizon_squared,
            self.horizon_values,
            self.horizon_non_finite,
            strict=True,
        ):
            horizon_rmse.append(
                math.sqrt(float(square) / int(values))
                if values > 0 and non_finite == 0
                else None
            )
        return {
            "state_nrmse": rmse,
            "state_horizon_nrmse": horizon_rmse,
            "state_values": self.values,
            "state_non_finite": self.non_finite,
        }


@dataclass
class _ConsistencyStats:
    horizon: int
    state: _StateNRMSE = field(init=False)
    risk_squared: float = 0.0
    risk_values: int = 0
    risk_non_finite: int = 0
    risk_max_abs: float = 0.0
    progress_squared: float = 0.0
    progress_values: int = 0
    progress_non_finite: int = 0
    progress_max_abs: float = 0.0
    joint_prediction_values: int = 0
    joint_prediction_non_finite: int = 0
    teacher_prediction_values: int = 0
    teacher_prediction_non_finite: int = 0

    def __post_init__(self) -> None:
        self.state = _StateNRMSE(self.horizon)

    def add(
        self,
        joint: RWMARRolloutPredictions,
        teacher: RWMARRolloutPredictions,
        *,
        state_std: Tensor,
        continuous: Tensor,
        yaw_indices: tuple[int, ...],
    ) -> None:
        mask = torch.ones(
            joint.next_state_mean.shape[:2],
            device=joint.next_state_mean.device,
            dtype=torch.bool,
        )
        self.state.add(
            joint.next_state_mean,
            teacher.next_state_mean,
            mask=mask,
            state_std=state_std,
            continuous=continuous,
            yaw_indices=yaw_indices,
        )
        self._add_scalar(
            joint.failure_logit.sigmoid(),
            teacher.failure_logit.sigmoid(),
            kind="risk",
        )
        self._add_scalar(
            joint.response_progress,
            teacher.response_progress,
            kind="progress",
        )
        joint_values, joint_non_finite = _prediction_finiteness(joint)
        teacher_values, teacher_non_finite = _prediction_finiteness(teacher)
        self.joint_prediction_values += joint_values
        self.joint_prediction_non_finite += joint_non_finite
        self.teacher_prediction_values += teacher_values
        self.teacher_prediction_non_finite += teacher_non_finite

    def _add_scalar(self, prediction: Tensor, target: Tensor, *, kind: str) -> None:
        if prediction.shape != target.shape:
            raise ValueError(f"{kind} prediction and teacher target shapes differ")
        difference = prediction - target
        values = int(difference.numel())
        finite = torch.isfinite(difference)
        non_finite = int((~finite).sum().cpu())
        square = 0.0
        maximum = 0.0
        if bool(finite.any()):
            selected = difference[finite].double()
            square = float(selected.square().sum().cpu())
            maximum = float(selected.abs().max().cpu())
        if kind == "risk":
            self.risk_values += values
            self.risk_non_finite += non_finite
            self.risk_squared += square
            self.risk_max_abs = max(self.risk_max_abs, maximum)
        else:
            self.progress_values += values
            self.progress_non_finite += non_finite
            self.progress_squared += square
            self.progress_max_abs = max(self.progress_max_abs, maximum)

    def report(self, *, action_source: str, target_source: str) -> dict[str, Any]:
        state = self.state.report()
        risk_rmse = (
            math.sqrt(self.risk_squared / self.risk_values)
            if self.risk_values > 0 and self.risk_non_finite == 0
            else None
        )
        progress_rmse = (
            math.sqrt(self.progress_squared / self.progress_values)
            if self.progress_values > 0 and self.progress_non_finite == 0
            else None
        )
        all_finite = bool(
            state["state_values"]
            and state["state_non_finite"] == 0
            and self.risk_values
            and self.risk_non_finite == 0
            and self.progress_values
            and self.progress_non_finite == 0
            and self.joint_prediction_values
            and self.joint_prediction_non_finite == 0
            and self.teacher_prediction_values
            and self.teacher_prediction_non_finite == 0
        )
        return {
            "action_source": action_source,
            "target_source": target_source,
            "demo_future_is_ground_truth": False,
            **state,
            "risk_probability_rmse": risk_rmse,
            "risk_probability_max_abs_error": (
                self.risk_max_abs if self.risk_non_finite == 0 else None
            ),
            "risk_values": self.risk_values,
            "risk_non_finite": self.risk_non_finite,
            "response_progress_rmse": progress_rmse,
            "response_progress_max_abs_error": (
                self.progress_max_abs if self.progress_non_finite == 0 else None
            ),
            "progress_values": self.progress_values,
            "progress_non_finite": self.progress_non_finite,
            "joint_prediction_values": self.joint_prediction_values,
            "joint_prediction_non_finite": self.joint_prediction_non_finite,
            "teacher_prediction_values": self.teacher_prediction_values,
            "teacher_prediction_non_finite": self.teacher_prediction_non_finite,
            "all_values_finite": all_finite,
        }


@torch.inference_mode()
def evaluate_joint_wam_offline(
    joint_world: RWMARWorldModel,
    frozen_teacher: RWMARWorldModel,
    flow: StatefulActionFlow,
    loader: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
    execution_steps: int = 2,
    solver_steps: int = 4,
    solver: str = "euler",
    anchor_residual_scale: float = 0.1,
    normalized_action_clip: float = 10.0,
    fixed_actions: Mapping[int, float] | None = None,
    action_bound_atol: float = 1e-6,
    max_batches: int = -1,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Evaluate a Joint WAM without counterfactual demo-state relabelling.

    ``cold_generated`` and ``warm_generated`` are raw flow chunks.  The two
    ``*_deployed`` chunks apply the runtime frozen-anchor residual blend.  Every
    consistency comparison runs the Joint WAM and frozen teacher on the exact
    same chunk and uses the frozen prediction as the counterfactual target.
    """

    fixed = _validate_offline_contract(
        joint_world,
        frozen_teacher,
        flow,
        execution_steps=execution_steps,
        solver_steps=solver_steps,
        solver=solver,
        anchor_residual_scale=anchor_residual_scale,
        normalized_action_clip=normalized_action_clip,
        fixed_actions=fixed_actions,
        action_bound_atol=action_bound_atol,
        max_batches=max_batches,
    )
    joint_world.to(device).eval()
    frozen_teacher.to(device).eval()
    flow.to(device).eval()
    horizon = flow.config.horizon
    action_stats = {
        mode: _ActionStats(action_bound_atol) for mode in _CONSISTENCY_MODES
    }
    consistency = {mode: _ConsistencyStats(horizon) for mode in _CONSISTENCY_MODES}
    joint_expert = _StateNRMSE(horizon)
    teacher_expert = _StateNRMSE(horizon)
    expert_joint_prediction_values = 0
    expert_joint_prediction_non_finite = 0
    expert_teacher_prediction_values = 0
    expert_teacher_prediction_non_finite = 0
    samples = 0
    batches = 0
    continuous = frozen_teacher.continuous_state_mask
    state_std = frozen_teacher.features.state_std
    yaw_indices = tuple(frozen_teacher.config.yaw_indices)

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
        joint_hidden, joint_state, joint_features = joint_world.encode_planning_history(
            history
        )
        teacher_hidden, teacher_state, _ = frozen_teacher.encode_planning_history(
            history
        )
        target_actions = batch["candidate_actions"][:, :horizon]
        if target_actions.shape[1:] != (horizon, flow.config.action_dim):
            raise ValueError(
                "candidate action chunk does not satisfy the Joint WAM contract"
            )
        warm_seed = synthetic_shifted_warm_start(target_actions, execution_steps)
        cold_generated = flow.generate(
            joint_features,
            solver_steps=solver_steps,
            solver=solver,
            normalized_clip=normalized_action_clip,
        )
        warm_generated = flow.generate(
            joint_features,
            initial_actions=warm_seed,
            solver_steps=solver_steps,
            solver=solver,
            normalized_clip=normalized_action_clip,
        )
        anchor = _anchor_chunk(
            joint_world,
            flow,
            joint_hidden,
            joint_state,
            horizon=horizon,
            fixed_actions=fixed,
        )
        cold_deployed = _deployed_chunk(
            anchor,
            cold_generated,
            residual_scale=anchor_residual_scale,
            fixed_actions=fixed,
        )
        warm_deployed = _deployed_chunk(
            anchor,
            warm_generated,
            residual_scale=anchor_residual_scale,
            fixed_actions=fixed,
        )
        action_chunks = {
            "cold_generated": cold_generated,
            "warm_generated": warm_generated,
            "cold_deployed": cold_deployed,
            "warm_deployed": warm_deployed,
        }
        for mode, actions in action_chunks.items():
            action_stats[mode].add(actions)
            joint_prediction = joint_world.predict_from_encoded_history(
                joint_hidden,
                joint_state,
                actions,
                sample_state=False,
            )
            teacher_prediction = frozen_teacher.predict_from_encoded_history(
                teacher_hidden,
                teacher_state,
                actions,
                sample_state=False,
            )
            consistency[mode].add(
                joint_prediction,
                teacher_prediction,
                state_std=state_std,
                continuous=continuous,
                yaw_indices=yaw_indices,
            )

        joint_expert_prediction = joint_world.predict_from_encoded_history(
            joint_hidden,
            joint_state,
            target_actions,
            sample_state=False,
        )
        teacher_expert_prediction = frozen_teacher.predict_from_encoded_history(
            teacher_hidden,
            teacher_state,
            target_actions,
            sample_state=False,
        )
        forecast_mask = batch.get("forecast_mask")
        if forecast_mask is None:
            forecast_mask = torch.ones(
                target_actions.shape[:2], device=device, dtype=torch.bool
            )
        else:
            forecast_mask = forecast_mask[:, :horizon].to(dtype=torch.bool)
        target_states = batch["target_states"][:, :horizon]
        joint_expert.add(
            joint_expert_prediction.next_state_mean,
            target_states,
            mask=forecast_mask,
            state_std=state_std,
            continuous=continuous,
            yaw_indices=yaw_indices,
        )
        teacher_expert.add(
            teacher_expert_prediction.next_state_mean,
            target_states,
            mask=forecast_mask,
            state_std=state_std,
            continuous=continuous,
            yaw_indices=yaw_indices,
        )
        values, non_finite = _prediction_finiteness(joint_expert_prediction)
        expert_joint_prediction_values += values
        expert_joint_prediction_non_finite += non_finite
        values, non_finite = _prediction_finiteness(teacher_expert_prediction)
        expert_teacher_prediction_values += values
        expert_teacher_prediction_non_finite += non_finite

        batches = batch_index
        samples += int(target_actions.shape[0])
        if progress is not None:
            progress({"batch": batch_index, "samples": samples})
        if max_batches > 0 and batch_index >= max_batches:
            break

    if samples == 0 or joint_expert.values == 0:
        raise RuntimeError("offline Joint WAM evaluation has no usable samples")

    action_reports = {name: item.report() for name, item in action_stats.items()}
    action_sources = {
        "cold_generated": "flow_cold_generation",
        "warm_generated": "flow_warm_generation_from_shifted_expert_chunk",
        "cold_deployed": "frozen_anchor_plus_scaled_cold_flow_residual",
        "warm_deployed": "frozen_anchor_plus_scaled_warm_flow_residual",
    }
    consistency_reports = {
        mode: item.report(
            action_source=action_sources[mode],
            target_source=(
                _GENERATED_TARGET_SOURCE
                if mode.endswith("generated")
                else _DEPLOYED_TARGET_SOURCE
            ),
        )
        for mode, item in consistency.items()
    }
    joint_expert_report = joint_expert.report()
    teacher_expert_report = teacher_expert.report()
    expert_all_finite = bool(
        joint_expert_report["state_non_finite"] == 0
        and teacher_expert_report["state_non_finite"] == 0
        and expert_joint_prediction_values
        and expert_joint_prediction_non_finite == 0
        and expert_teacher_prediction_values
        and expert_teacher_prediction_non_finite == 0
    )
    all_actions_finite = all(
        bool(report["all_finite"]) for report in action_reports.values()
    )
    all_actions_bounded = all(
        bool(report["all_bounded"]) for report in action_reports.values()
    )
    all_consistency_finite = all(
        bool(report["all_values_finite"]) for report in consistency_reports.values()
    )
    return {
        "format_version": "wam.joint_wam.offline/1",
        "model": "joint_wam",
        "batches": batches,
        "samples": samples,
        "horizon": horizon,
        "execution_steps": execution_steps,
        "solver_steps": solver_steps,
        "solver": solver,
        "anchor_residual_scale": anchor_residual_scale,
        "warm_start_source": "shifted_expert_action_chunk_for_offline_stress_only",
        "frozen_teacher_parameters_frozen": all(
            not parameter.requires_grad for parameter in frozen_teacher.parameters()
        ),
        "frozen_anchor_parameters_frozen": all(
            not parameter.requires_grad for parameter in flow.anchor_prior.parameters()
        ),
        "target_contract": {
            "expert_action_dataset_world": {
                "target_source": _EXPERT_TARGET_SOURCE,
                "action_source": "recorded_expert_action",
            },
            "generated_action_teacher_consistency": {
                "target_source": _GENERATED_TARGET_SOURCE,
                "demo_future_is_ground_truth": False,
            },
            "deployed_action_teacher_consistency": {
                "target_source": _DEPLOYED_TARGET_SOURCE,
                "demo_future_is_ground_truth": False,
            },
        },
        "expert_action_dataset_world": {
            "target_source": _EXPERT_TARGET_SOURCE,
            "joint": {
                **joint_expert_report,
                "prediction_values": expert_joint_prediction_values,
                "prediction_non_finite": expert_joint_prediction_non_finite,
            },
            "frozen_teacher": {
                **teacher_expert_report,
                "prediction_values": expert_teacher_prediction_values,
                "prediction_non_finite": expert_teacher_prediction_non_finite,
            },
            "all_values_finite": expert_all_finite,
        },
        "action_generation": action_reports,
        "teacher_consistency": consistency_reports,
        "expert_action_world_state_nrmse": joint_expert_report["state_nrmse"],
        "expert_action_world_horizon_nrmse": joint_expert_report["state_horizon_nrmse"],
        "expert_action_world_target_source": _EXPERT_TARGET_SOURCE,
        "frozen_teacher_expert_action_world_state_nrmse": teacher_expert_report[
            "state_nrmse"
        ],
        "generated_action_world_target_source": _GENERATED_TARGET_SOURCE,
        "generated_action_demo_state_is_ground_truth": False,
        "deployed_action_world_target_source": _DEPLOYED_TARGET_SOURCE,
        "deployed_action_demo_state_is_ground_truth": False,
        "all_generated_actions_finite": all_actions_finite,
        "all_generated_actions_bounded": all_actions_bounded,
        "all_teacher_consistency_values_finite": all_consistency_finite,
        "all_generated_values_finite": (all_actions_finite and all_consistency_finite),
        "all_offline_values_finite": (
            expert_all_finite and all_actions_finite and all_consistency_finite
        ),
    }


def joint_wam_offline_acceptance_report(
    metrics: Mapping[str, Any],
    *,
    world_model_parameter_delta: float | None = None,
    shared_history_parameter_delta: float | None = None,
    world_parameter_delta: float | None = None,
    world_head_parameter_delta: float | None = None,
    action_flow_parameter_delta: float | None = None,
    flow_parameter_delta: float | None = None,
    anchor_prior_parameter_delta: float | None = None,
    frozen_teacher_parameter_delta: float | None = None,
    source_checkpoints_immutable: bool | None = None,
    checkpoint_reload_max_abs_diff: float | None = None,
    strict_reload_max_abs_diff: float | None = None,
    branch_gradient_maxima: Mapping[str, Any] | None = None,
    maximum_expert_world_nrmse: float | None = None,
    maximum_generated_teacher_state_nrmse: float | None = None,
) -> dict[str, Any]:
    """Apply the evidence requirements for calling a Joint WAM artifact a Joint WAM."""

    world_delta = _coalesce_alias(
        world_parameter_delta,
        world_head_parameter_delta,
        "world parameter delta",
    )
    flow_delta = _coalesce_alias(
        action_flow_parameter_delta,
        flow_parameter_delta,
        "action-flow parameter delta",
    )
    reload_difference = _coalesce_alias(
        checkpoint_reload_max_abs_diff,
        strict_reload_max_abs_diff,
        "strict reload maximum difference",
    )
    gradient_evidence = branch_gradient_maxima or {}
    required_gradient_paths = (
        "action_to_flow_gradient_norm",
        "action_to_backbone_gradient_norm",
        "world_to_backbone_gradient_norm",
        "consistency_to_flow_gradient_norm",
        "consistency_to_backbone_gradient_norm",
    )
    gradient_maxima = {
        name: gradient_evidence.get(name) for name in required_gradient_paths
    }
    consistency = metrics.get("teacher_consistency")
    required_consistency_present = isinstance(consistency, Mapping) and all(
        isinstance(consistency.get(mode), Mapping) for mode in _CONSISTENCY_MODES
    )
    consistency_finite = bool(
        required_consistency_present
        and all(
            consistency[mode].get("all_values_finite") is True
            and _finite_number(consistency[mode].get("state_nrmse"))
            and _finite_number(consistency[mode].get("risk_probability_rmse"))
            and _finite_number(consistency[mode].get("response_progress_rmse"))
            for mode in _CONSISTENCY_MODES
        )
    )
    consistency_target_contract = bool(
        required_consistency_present
        and all(
            consistency[mode].get("target_source")
            == (
                _GENERATED_TARGET_SOURCE
                if mode.endswith("generated")
                else _DEPLOYED_TARGET_SOURCE
            )
            and consistency[mode].get("demo_future_is_ground_truth") is False
            for mode in _CONSISTENCY_MODES
        )
    )
    expert_nrmse = metrics.get("expert_action_world_state_nrmse")
    generated_state_values = (
        [float(consistency[mode]["state_nrmse"]) for mode in _CONSISTENCY_MODES]
        if consistency_finite
        else []
    )
    maximum_generated_state_nrmse = (
        max(generated_state_values) if generated_state_values else None
    )
    checks = {
        "model_is_joint_wam": metrics.get("model") == "joint_wam",
        "expert_action_dataset_world_is_finite": bool(
            _finite_number(metrics.get("expert_action_world_state_nrmse"))
            and metrics.get("expert_action_world_target_source")
            == _EXPERT_TARGET_SOURCE
        ),
        "generated_action_target_is_frozen_teacher": bool(
            metrics.get("generated_action_world_target_source")
            == _GENERATED_TARGET_SOURCE
            and metrics.get("generated_action_demo_state_is_ground_truth") is False
        ),
        "deployed_action_target_is_frozen_teacher": bool(
            metrics.get("deployed_action_world_target_source")
            == _DEPLOYED_TARGET_SOURCE
            and metrics.get("deployed_action_demo_state_is_ground_truth") is False
        ),
        "cold_and_warm_actions_finite": metrics.get("all_generated_actions_finite")
        is True,
        "cold_and_warm_actions_bounded": metrics.get("all_generated_actions_bounded")
        is True,
        "teacher_consistency_present_and_finite": consistency_finite,
        "teacher_consistency_target_contract": consistency_target_contract,
        "all_generated_values_finite": metrics.get("all_generated_values_finite")
        is True,
        "all_offline_values_finite": metrics.get("all_offline_values_finite") is True,
        "world_model_parameter_delta_nonzero": _positive_finite(
            world_model_parameter_delta
        ),
        "shared_history_parameter_delta_nonzero": _positive_finite(
            shared_history_parameter_delta
        ),
        "world_parameter_delta_nonzero": _positive_finite(world_delta),
        "action_flow_parameter_delta_nonzero": _positive_finite(flow_delta),
        "anchor_prior_immutable": _exact_zero(anchor_prior_parameter_delta),
        "anchor_prior_frozen": metrics.get("frozen_anchor_parameters_frozen") is True,
        "frozen_teacher_parameters_frozen": metrics.get(
            "frozen_teacher_parameters_frozen"
        )
        is True,
        "frozen_teacher_immutable": _exact_zero(frozen_teacher_parameter_delta),
        "source_checkpoints_immutable": source_checkpoints_immutable is True,
        "strict_checkpoint_reload_exact": _exact_zero(reload_difference),
        "action_to_flow_gradient_nonzero": _positive_finite(
            gradient_maxima["action_to_flow_gradient_norm"]
        ),
        "action_to_backbone_gradient_nonzero": _positive_finite(
            gradient_maxima["action_to_backbone_gradient_norm"]
        ),
        "world_to_backbone_gradient_nonzero": _positive_finite(
            gradient_maxima["world_to_backbone_gradient_norm"]
        ),
        "consistency_to_flow_gradient_nonzero": _positive_finite(
            gradient_maxima["consistency_to_flow_gradient_norm"]
        ),
        "consistency_to_backbone_gradient_nonzero": _positive_finite(
            gradient_maxima["consistency_to_backbone_gradient_norm"]
        ),
        "expert_world_nrmse_within_guard": bool(
            maximum_expert_world_nrmse is None
            or (
                _finite_number(expert_nrmse)
                and float(expert_nrmse) <= maximum_expert_world_nrmse
            )
        ),
        "generated_teacher_state_nrmse_within_guard": bool(
            maximum_generated_teacher_state_nrmse is None
            or (
                _finite_number(maximum_generated_state_nrmse)
                and float(maximum_generated_state_nrmse)
                <= maximum_generated_teacher_state_nrmse
            )
        ),
    }
    return {
        "format_version": "wam.joint_wam.acceptance/1",
        "model": "joint_wam",
        "passed": all(checks.values()),
        "checks": checks,
        "parameter_deltas": {
            "world_model": world_model_parameter_delta,
            "shared_history": shared_history_parameter_delta,
            "world": world_delta,
            "action_flow": flow_delta,
            "anchor_prior": anchor_prior_parameter_delta,
            "frozen_teacher": frozen_teacher_parameter_delta,
        },
        "branch_gradient_maxima": gradient_maxima,
        "source_checkpoints_immutable": source_checkpoints_immutable,
        "checkpoint_reload_max_abs_diff": reload_difference,
        "offline_summary": {
            "samples": metrics.get("samples"),
            "expert_action_world_state_nrmse": metrics.get(
                "expert_action_world_state_nrmse"
            ),
            "maximum_generated_teacher_state_nrmse": (maximum_generated_state_nrmse),
            "all_generated_values_finite": metrics.get("all_generated_values_finite"),
            "all_generated_actions_bounded": metrics.get(
                "all_generated_actions_bounded"
            ),
        },
        "offline_guards": {
            "maximum_expert_world_nrmse": maximum_expert_world_nrmse,
            "maximum_generated_teacher_state_nrmse": (
                maximum_generated_teacher_state_nrmse
            ),
        },
    }


def joint_wam_closed_loop_acceptance_report(
    metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    held_out_seed_overlap: int,
    minimum_episodes: int,
    minimum_success_rate: float,
    maximum_prior_regression: float,
    smoke: bool,
) -> dict[str, Any]:
    """Reuse the action-flow warm-up direct-flow gate while identifying the artifact as Joint WAM."""

    report = action_flow_acceptance_report(
        metrics,
        held_out_seed_overlap=held_out_seed_overlap,
        minimum_episodes=minimum_episodes,
        minimum_success_rate=minimum_success_rate,
        maximum_prior_regression=maximum_prior_regression,
        smoke=smoke,
    )
    return {**report, "model": "joint_wam"}


def joint_wam_training_acceptance_report(
    metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    held_out_seed_overlap: int,
    minimum_episodes: int,
    minimum_success_rate: float,
    maximum_prior_regression: float,
    smoke: bool,
) -> dict[str, Any]:
    """Backward-friendly name for the Joint WAM closed-loop smoke/20-seed gate."""

    return joint_wam_closed_loop_acceptance_report(
        metrics,
        held_out_seed_overlap=held_out_seed_overlap,
        minimum_episodes=minimum_episodes,
        minimum_success_rate=minimum_success_rate,
        maximum_prior_regression=maximum_prior_regression,
        smoke=smoke,
    )


def _validate_offline_contract(
    joint_world: RWMARWorldModel,
    frozen_teacher: RWMARWorldModel,
    flow: StatefulActionFlow,
    *,
    execution_steps: int,
    solver_steps: int,
    solver: str,
    anchor_residual_scale: float,
    normalized_action_clip: float,
    fixed_actions: Mapping[int, float] | None,
    action_bound_atol: float,
    max_batches: int,
) -> dict[int, float]:
    if joint_world.config.state_dim != frozen_teacher.config.state_dim:
        raise ValueError("Joint and frozen teacher state dimensions differ")
    if joint_world.config.action_dim != frozen_teacher.config.action_dim:
        raise ValueError("Joint and frozen teacher action dimensions differ")
    if tuple(joint_world.config.yaw_indices) != tuple(
        frozen_teacher.config.yaw_indices
    ):
        raise ValueError("Joint and frozen teacher yaw contracts differ")
    if tuple(joint_world.config.gripper_closed_indices) != tuple(
        frozen_teacher.config.gripper_closed_indices
    ):
        raise ValueError("Joint and frozen teacher gripper contracts differ")
    if flow.config.feature_dim != joint_world.planning_feature_dim:
        raise ValueError("flow feature dimension does not match Joint WAM")
    if flow.config.action_dim != joint_world.config.action_dim:
        raise ValueError("flow action dimension does not match Joint WAM")
    if flow.config.horizon > joint_world.config.train_forecast_horizon:
        raise ValueError("flow horizon exceeds Joint WAM training horizon")
    if flow.config.horizon > frozen_teacher.config.train_forecast_horizon:
        raise ValueError("flow horizon exceeds frozen teacher training horizon")
    if not torch.equal(
        joint_world.features.state_std.detach().cpu(),
        frozen_teacher.features.state_std.detach().cpu(),
    ):
        raise ValueError("Joint and frozen teacher state normalization differs")
    if execution_steps <= 0 or execution_steps >= flow.config.horizon:
        raise ValueError("execution_steps must be in [1,H)")
    if solver_steps <= 0:
        raise ValueError("solver_steps must be positive")
    if solver not in {"euler", "heun"}:
        raise ValueError("solver must be euler or heun")
    if not 0.0 < anchor_residual_scale <= 1.0:
        raise ValueError("anchor_residual_scale must be in (0,1]")
    if normalized_action_clip <= 0.0:
        raise ValueError("normalized_action_clip must be positive")
    if action_bound_atol < 0.0:
        raise ValueError("action_bound_atol cannot be negative")
    if max_batches == 0 or max_batches < -1:
        raise ValueError("max_batches must be -1 or positive")
    configured = {3: 1.0, 7: 1.0} if fixed_actions is None else dict(fixed_actions)
    fixed: dict[int, float] = {}
    for raw_index, raw_value in configured.items():
        index = int(raw_index)
        value = float(raw_value)
        if index < 0 or index >= flow.config.action_dim:
            raise ValueError("fixed action index is out of range")
        if not math.isfinite(value) or value < -1.0 or value > 1.0:
            raise ValueError("fixed action value must be finite and bounded")
        fixed[index] = value
    return fixed


def _anchor_chunk(
    world: RWMARWorldModel,
    flow: StatefulActionFlow,
    hidden: Tensor,
    current_state: Tensor,
    *,
    horizon: int,
    fixed_actions: Mapping[int, float],
) -> Tensor:
    actions: list[Tensor] = []
    recurrent = hidden
    state = current_state
    for _ in range(horizon):
        features = world.planning_features(recurrent, state)
        action = _fix_actions(flow.anchor_action(features), fixed_actions)
        actions.append(action)
        recurrent, state, _ = world.imagine_step(
            recurrent,
            state,
            action,
            sample_state=False,
        )
    return torch.stack(actions, dim=1)


def _deployed_chunk(
    anchor: Tensor,
    generated: Tensor,
    *,
    residual_scale: float,
    fixed_actions: Mapping[int, float],
) -> Tensor:
    if anchor.shape != generated.shape:
        raise ValueError("anchor and generated chunks must have identical shapes")
    deployed = anchor + residual_scale * (generated - anchor)
    return _fix_actions(deployed, fixed_actions)


def _fix_actions(actions: Tensor, fixed_actions: Mapping[int, float]) -> Tensor:
    result = actions.clamp(-1.0, 1.0).clone()
    for index, value in fixed_actions.items():
        result[..., index] = value
    return result


def _prediction_finiteness(
    predictions: RWMARRolloutPredictions,
) -> tuple[int, int]:
    values = 0
    non_finite = 0
    for name in predictions.__dataclass_fields__:
        tensor = getattr(predictions, name)
        values += int(tensor.numel())
        non_finite += int((~torch.isfinite(tensor)).sum().cpu())
    return values, non_finite


def _coalesce_alias(
    primary: float | None,
    alias: float | None,
    name: str,
) -> float | None:
    if primary is None:
        return alias
    if alias is None:
        return primary
    if float(primary) != float(alias):
        raise ValueError(f"conflicting values for {name}")
    return primary


def _finite_number(value: Any) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _positive_finite(value: Any) -> bool:
    return _finite_number(value) and float(value) > 0.0


def _exact_zero(value: Any) -> bool:
    return _finite_number(value) and float(value) == 0.0


__all__ = [
    "evaluate_joint_wam_offline",
    "joint_wam_training_acceptance_report",
    "joint_wam_closed_loop_acceptance_report",
    "joint_wam_offline_acceptance_report",
]
