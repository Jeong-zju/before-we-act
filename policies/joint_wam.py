"""Stateful direct-execution policy for Joint WAM."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch

from models.wam import (
    ActionChunkConfig,
    ActionPrior,
    RWMARWorldModel,
    StatefulActionFlow,
    WorldModelSequenceInputs,
    shift_action_chunk_warm_start,
)


@dataclass(frozen=True)
class JointWAMPolicyConfig:
    action_chunk: ActionChunkConfig = ActionChunkConfig()
    solver: str = "euler"
    anchor_residual_scale: float = 0.1
    normalized_action_clip: float = 10.0
    observation_residual_nrmse_max: float = 3.0
    risk_veto: bool = True
    max_failure_probability: float = 0.9
    max_predicted_robot_distance: float = 1.15
    max_action_ood: float = 1.0
    action_ood_threshold: float = 3.0
    latency_budget_ms: float = 50.0
    fallback_enabled: bool = False

    def __post_init__(self) -> None:
        if self.solver not in {"euler", "heun"}:
            raise ValueError("solver must be euler or heun")
        if not 0.0 < self.anchor_residual_scale <= 1.0:
            raise ValueError("anchor_residual_scale must be in (0,1]")
        for name in (
            "normalized_action_clip",
            "observation_residual_nrmse_max",
            "max_failure_probability",
            "max_predicted_robot_distance",
            "max_action_ood",
            "action_ood_threshold",
            "latency_budget_ms",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")


class JointWAMPolicy:
    """Reuse two actions, then correct the shifted eight-step flow chunk."""

    def __init__(
        self,
        world_model: RWMARWorldModel,
        flow: StatefulActionFlow,
        *,
        config: JointWAMPolicyConfig | None = None,
        fallback_world_model: RWMARWorldModel | None = None,
        fallback_prior: ActionPrior | None = None,
        fixed_actions: Mapping[int, float] | None = None,
        distillation_callback: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None], None
        ]
        | None = None,
        clock: Any = time.perf_counter,
    ) -> None:
        self.config = config or JointWAMPolicyConfig()
        if flow.config.feature_dim != world_model.planning_feature_dim:
            raise ValueError("flow feature dimension does not match world model")
        if flow.config.action_dim != world_model.config.action_dim:
            raise ValueError("flow action dimension does not match world model")
        if flow.config.horizon != self.config.action_chunk.horizon:
            raise ValueError("flow horizon does not match the action-chunk contract")
        if flow.config.action_dim != self.config.action_chunk.action_dim:
            raise ValueError("flow action dimension does not match chunk contract")
        if (fallback_world_model is None) != (fallback_prior is None):
            raise ValueError("fallback world model and prior must be provided together")
        if self.config.fallback_enabled and fallback_prior is None:
            raise ValueError("enabled fallback requires the frozen action prior")
        self.world_model = world_model.eval()
        self.flow = flow.eval()
        self.fallback_world_model = (
            None if fallback_world_model is None else fallback_world_model.eval()
        )
        self.fallback_prior = None if fallback_prior is None else fallback_prior.eval()
        for module in (
            self.world_model,
            self.flow,
            self.fallback_world_model,
            self.fallback_prior,
        ):
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        parameter = next(world_model.parameters())
        self.device = parameter.device
        self.dtype = parameter.dtype
        self.fixed_actions = {
            int(index): float(value) for index, value in (fixed_actions or {}).items()
        }
        self.distillation_callback = distillation_callback
        for index, value in self.fixed_actions.items():
            if not 0 <= index < flow.config.action_dim:
                raise ValueError("fixed action index is out of range")
            if not -1.0 <= value <= 1.0:
                raise ValueError("fixed action values must be in [-1,1]")
        horizon = world_model.config.history_horizon
        self._states: deque[np.ndarray] = deque(maxlen=horizon)
        self._actions: deque[np.ndarray] = deque(maxlen=max(horizon - 1, 1))
        self._chunk: torch.Tensor | None = None
        self._chunk_anchor: torch.Tensor | None = None
        self._chunk_world: torch.Tensor | None = None
        self._chunk_cursor = 0
        self._executed_since_generation = 0
        self._predicted_next_state: torch.Tensor | None = None
        self.clock = clock
        self.last_diagnostics: dict[str, Any] = {}

    def reset(self) -> None:
        self._states.clear()
        self._actions.clear()
        self._chunk = None
        self._chunk_anchor = None
        self._chunk_world = None
        self._chunk_cursor = 0
        self._executed_since_generation = 0
        self._predicted_next_state = None
        self.last_diagnostics = {}

    @torch.inference_mode()
    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        keys = {str(key) for key in observation}
        if "privileged_state" in observation:
            raise RuntimeError("privileged_state leakage into Joint WAM policy")
        if "proprioception" not in observation:
            raise KeyError("Joint WAM requires observation['proprioception']")
        state = np.asarray(observation["proprioception"], dtype=np.float32)
        expected = self.world_model.config.state_dim
        if state.shape != (expected,) or not np.isfinite(state).all():
            raise ValueError(f"proprioception must be finite with shape {(expected,)}")
        state_tensor = torch.as_tensor(
            state, device=self.device, dtype=self.dtype
        ).unsqueeze(0)
        residual = self._observation_residual(state_tensor)
        self._states.append(state.copy())
        start = self.clock()
        replan_reason = self._replan_reason(residual)
        planned_mode = "none"
        fallback_reason = "none"
        warm_start_used = False
        risk_veto_regenerated = False
        diagnostics: dict[str, Any] = {
            "observation_residual_nrmse": residual,
        }
        executed_mode = "joint_wam_flow"
        try:
            if replan_reason != "reuse":
                planned_mode = "joint_wam_flow"
                history = self._history(self.world_model)
                hidden, current_state, features = (
                    self.world_model.encode_planning_history(history)
                )
                anchor_chunk = self._anchor_chunk(hidden, current_state)
                warm_actions = self._warm_start_actions()
                generation_initial = warm_actions
                warm_start_used = warm_actions is not None
                generated_chunk = self.flow.generate(
                    features,
                    initial_actions=warm_actions,
                    solver_steps=self.config.action_chunk.solver_steps,
                    solver=self.config.solver,
                    normalized_clip=self.config.normalized_action_clip,
                )
                chunk = self._anchored_chunk(anchor_chunk, generated_chunk)
                world = self.world_model.predict_from_encoded_history(
                    hidden, current_state, chunk, sample_state=False
                )
                diagnostics = {
                    **self._world_diagnostics(chunk, world),
                    "observation_residual_nrmse": residual,
                }
                if (
                    self.config.risk_veto
                    and self._unsafe(diagnostics)
                    and warm_actions is not None
                ):
                    risk_veto_regenerated = True
                    generation_initial = None
                    generated_chunk = self.flow.generate(
                        features,
                        solver_steps=self.config.action_chunk.solver_steps,
                        solver=self.config.solver,
                        normalized_clip=self.config.normalized_action_clip,
                    )
                    chunk = self._anchored_chunk(anchor_chunk, generated_chunk)
                    world = self.world_model.predict_from_encoded_history(
                        hidden, current_state, chunk, sample_state=False
                    )
                    diagnostics = {
                        **self._world_diagnostics(chunk, world),
                        "observation_residual_nrmse": residual,
                    }
                if self.config.risk_veto and self._unsafe(diagnostics):
                    if self.config.fallback_enabled:
                        action = self._fallback_action()
                        executed_mode = "action_prior_risk_fallback"
                        fallback_reason = "world_risk_veto"
                        self._chunk = None
                        self._chunk_anchor = None
                        self._chunk_world = None
                        self._chunk_cursor = 0
                        self._executed_since_generation = 0
                        self._predicted_next_state = None
                        return self._finish_action(
                            action,
                            keys=keys,
                            start=start,
                            diagnostics=diagnostics,
                            planned_mode=planned_mode,
                            executed_mode=executed_mode,
                            fallback_reason=fallback_reason,
                            replan_reason=replan_reason,
                            warm_start_used=warm_start_used,
                            risk_veto_regenerated=risk_veto_regenerated,
                        )
                    diagnostics["risk_veto_unresolved"] = True
                if self.distillation_callback is not None:
                    self.distillation_callback(
                        features,
                        hidden,
                        current_state,
                        generation_initial,
                    )
                self._chunk = chunk[0]
                self._chunk_anchor = anchor_chunk[0]
                self._chunk_world = world.next_state_mean[0]
                self._chunk_cursor = 0
                self._executed_since_generation = 0
            if (
                self._chunk is None
                or self._chunk_anchor is None
                or self._chunk_world is None
            ):
                raise RuntimeError("flow generation produced no action chunk")
            action_index = self._chunk_cursor
            action = self._chunk[action_index].detach().cpu().numpy()
            diagnostics["applied_flow_residual_max"] = float(
                (self._chunk[action_index] - self._chunk_anchor[action_index])
                .abs()
                .max()
                .cpu()
            )
            self._predicted_next_state = (
                self._chunk_world[action_index].detach().clone()
            )
            self._chunk_cursor += 1
            self._executed_since_generation += 1
        except (RuntimeError, FloatingPointError, ValueError) as error:
            if self.config.fallback_enabled:
                action = self._fallback_action()
                executed_mode = "action_prior_flow_error_fallback"
            else:
                action = self._safe_stop()
                executed_mode = "safe_stop_flow_error"
            fallback_reason = f"flow_error:{type(error).__name__}"
            self._chunk = None
            self._chunk_anchor = None
            self._chunk_world = None
            self._chunk_cursor = 0
            self._executed_since_generation = 0
            self._predicted_next_state = None
        return self._finish_action(
            action,
            keys=keys,
            start=start,
            diagnostics=diagnostics,
            planned_mode=planned_mode,
            executed_mode=executed_mode,
            fallback_reason=fallback_reason,
            replan_reason=replan_reason,
            warm_start_used=warm_start_used,
            risk_veto_regenerated=risk_veto_regenerated,
        )

    def _finish_action(
        self,
        raw_action: np.ndarray,
        *,
        keys: set[str],
        start: float,
        diagnostics: Mapping[str, Any],
        planned_mode: str,
        executed_mode: str,
        fallback_reason: str,
        replan_reason: str,
        warm_start_used: bool,
        risk_veto_regenerated: bool,
    ) -> np.ndarray:
        action = np.clip(np.asarray(raw_action, dtype=np.float32), -1.0, 1.0)
        if (
            action.shape != (self.flow.config.action_dim,)
            or not np.isfinite(action).all()
        ):
            action = self._safe_stop()
            executed_mode = "safe_stop_invalid_action"
            fallback_reason = "invalid_flow_action"
        for index, value in self.fixed_actions.items():
            action[index] = value
        self._actions.append(action.copy())
        latency_ms = float((self.clock() - start) * 1000.0)
        direct = executed_mode == "joint_wam_flow"
        self.last_diagnostics = {
            **diagnostics,
            "latency_ms": latency_ms,
            "planning_latency_ms": latency_ms,
            "planned_mode": planned_mode,
            "executed_mode": executed_mode,
            "plan_executed": direct,
            "direct_flow_generated": planned_mode == "joint_wam_flow",
            "direct_flow_executed": direct,
            "deadline_exceeded": latency_ms > self.config.latency_budget_ms,
            "fallback_reason": fallback_reason,
            "fallback_enabled": self.config.fallback_enabled,
            "replan_reason": replan_reason,
            "warm_start_used": warm_start_used,
            "risk_veto_regenerated": risk_veto_regenerated,
            "executed_since_generation": self._executed_since_generation,
            "observation_keys": sorted(keys),
            "privileged_state_seen": False,
        }
        return action.astype(np.float32, copy=False)

    def _replan_reason(self, residual: float | None) -> str:
        if self._chunk is None:
            return "cold_start"
        if (
            residual is not None
            and residual > self.config.observation_residual_nrmse_max
        ):
            return "observation_prediction_residual"
        if self._chunk_cursor >= self.config.action_chunk.horizon:
            return "chunk_exhausted"
        if self._executed_since_generation >= self.config.action_chunk.execution_steps:
            return "scheduled_execute_steps"
        return "reuse"

    def _warm_start_actions(self) -> torch.Tensor | None:
        if self._chunk is None or self._executed_since_generation <= 0:
            return None
        if self._executed_since_generation >= self.config.action_chunk.horizon:
            return None
        shifted = shift_action_chunk_warm_start(
            self._chunk,
            self.config.action_chunk,
            executed_steps=self._executed_since_generation,
        )
        return shifted.unsqueeze(0)

    def _observation_residual(self, state: torch.Tensor) -> float | None:
        if self._predicted_next_state is None:
            return None
        difference = state[0] - self._predicted_next_state
        for yaw_index in self.world_model.config.yaw_indices:
            difference[yaw_index] = (
                torch.remainder(difference[yaw_index] + torch.pi, 2.0 * torch.pi)
                - torch.pi
            )
        continuous = self.world_model.continuous_state_mask
        normalized = (
            difference[continuous] / self.world_model.features.state_std[continuous]
        )
        return float(normalized.square().mean().sqrt().cpu())

    def _world_diagnostics(self, chunk: torch.Tensor, world: Any) -> dict[str, float]:
        failure = world.failure_logit.sigmoid()
        distance = torch.linalg.vector_norm(
            world.next_state_mean[..., 0:2] - world.next_state_mean[..., 11:13],
            dim=-1,
        )
        normalized_action = self.flow.normalize_actions(chunk)
        action_ood = torch.relu(
            normalized_action.abs() - self.config.action_ood_threshold
        ).mean(dim=-1)
        return {
            "failure_probability": float(failure.max().cpu()),
            "predicted_robot_distance": float(distance.max().cpu()),
            "action_ood": float(action_ood.max().cpu()),
            "expected_return": float(world.reward.sum().cpu()),
        }

    def _anchor_chunk(
        self, hidden: torch.Tensor, current_state: torch.Tensor
    ) -> torch.Tensor:
        actions: list[torch.Tensor] = []
        recurrent = hidden
        state = current_state
        for _ in range(self.config.action_chunk.horizon):
            features = self.world_model.planning_features(recurrent, state)
            action = self.flow.anchor_action(features)
            action = self._fix_chunk(action)
            actions.append(action)
            recurrent, state, _ = self.world_model.imagine_step(
                recurrent, state, action, sample_state=False
            )
        return torch.stack(actions, dim=1)

    def _anchored_chunk(
        self, anchor: torch.Tensor, generated: torch.Tensor
    ) -> torch.Tensor:
        residual = generated - anchor
        result = anchor + self.config.anchor_residual_scale * residual
        return self._fix_chunk(result)

    def _unsafe(self, diagnostics: Mapping[str, Any]) -> bool:
        return bool(
            float(diagnostics.get("failure_probability", float("inf")))
            > self.config.max_failure_probability
            or float(diagnostics.get("predicted_robot_distance", float("inf")))
            > self.config.max_predicted_robot_distance
            or float(diagnostics.get("action_ood", float("inf")))
            > self.config.max_action_ood
        )

    def _fix_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        result = chunk.clamp(-1.0, 1.0).clone()
        for index, value in self.fixed_actions.items():
            result[..., index] = value
        return result

    def _fallback_action(self) -> np.ndarray:
        if self.fallback_world_model is None or self.fallback_prior is None:
            raise RuntimeError("action prior fallback is unavailable")
        history = self._history(self.fallback_world_model)
        _, _, features = self.fallback_world_model.encode_planning_history(history)
        action = self.fallback_prior.deterministic_action(features)[0]
        return action.clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)

    def _safe_stop(self) -> np.ndarray:
        action = np.zeros(self.flow.config.action_dim, dtype=np.float32)
        for index, value in self.fixed_actions.items():
            action[index] = value
        return action

    def _history(self, model: RWMARWorldModel) -> WorldModelSequenceInputs:
        config = model.config
        count = len(self._states)
        offset = config.history_horizon - count
        states = torch.zeros(
            1,
            config.history_horizon,
            config.state_dim,
            device=self.device,
            dtype=self.dtype,
        )
        states[0, offset:] = torch.as_tensor(
            np.stack(self._states), device=self.device, dtype=self.dtype
        )
        past_actions = torch.zeros(
            1,
            config.history_horizon - 1,
            config.action_dim,
            device=self.device,
            dtype=self.dtype,
        )
        needed = max(count - 1, 0)
        if needed:
            past_actions[0, offset:] = torch.as_tensor(
                np.stack(list(self._actions)[-needed:]),
                device=self.device,
                dtype=self.dtype,
            )
        valid_mask = torch.zeros(
            1, config.history_horizon, device=self.device, dtype=torch.bool
        )
        valid_mask[0, offset:] = True
        return WorldModelSequenceInputs(states, past_actions, valid_mask)


__all__ = ["JointWAMPolicyConfig", "JointWAMPolicy"]
