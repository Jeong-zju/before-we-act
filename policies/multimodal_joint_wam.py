"""Direct stateful policy for the Phase M1 multimodal Joint WAM."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch

from models.wam import (
    ActionChunkConfig,
    RWMARWorldModel,
    StatefulActionFlow,
    shift_action_chunk_warm_start,
)
from models.wam_multimodal import LatentWAM, VisionEncoderOutput
from policies.joint_wam import JointWAMPolicy, JointWAMPolicyConfig


@dataclass(frozen=True)
class MultimodalJointWAMPolicyConfig:
    action_chunk: ActionChunkConfig = ActionChunkConfig()
    solver: str = "euler"
    normalized_action_clip: float = 10.0
    visual_residual_scale: float = 1.0
    replan_warm_start_enabled: bool = True
    cooperative_residual_scale: float = 0.1
    latency_budget_ms: float = 50.0
    maximum_visual_age_ms: float = 100.0
    control_period_ms: float = 50.0
    visual_history_frames: int = 2
    fixed_actions: tuple[tuple[int, float], ...] = ((3, 1.0), (7, 1.0))
    fallback_enabled: bool = False

    def __post_init__(self) -> None:
        if self.solver not in {"euler", "heun"}:
            raise ValueError("solver must be euler or heun")
        if self.action_chunk.horizon != 8 or self.action_chunk.execution_steps != 2:
            raise ValueError("Phase M1 requires an 8-step chunk and execute-2 control")
        if self.visual_residual_scale != 1.0:
            raise ValueError("formal M1 visual control requires residual scale 1.0")
        if not 0.0 < self.cooperative_residual_scale <= 1.0:
            raise ValueError("cooperative residual scale must be in (0,1]")
        if (
            self.latency_budget_ms <= 0.0
            or self.maximum_visual_age_ms <= 0.0
            or self.control_period_ms <= 0.0
        ):
            raise ValueError("latency/age budgets must be positive")
        if self.visual_history_frames <= 0:
            raise ValueError("visual history must be positive")
        if self.fallback_enabled:
            raise ValueError("formal M1 direct policy does not permit a fallback")


class MultimodalJointWAMPolicy:
    """Single-expert 8-step flow with RGB-refresh-aligned replanning.

    The retained cooperative-stop task uses a gate-zero multimodal residual and
    delegates the action path to the immutable legacy Joint WAM.  RGB is still
    encoded by the M1 backbone, so this is an explicit preservation bypass, not
    an unobserved policy substitution.
    """

    def __init__(
        self,
        model: LatentWAM,
        action_flow: StatefulActionFlow,
        legacy_world_model: RWMARWorldModel,
        legacy_action_flow: StatefulActionFlow,
        config: MultimodalJointWAMPolicyConfig | None = None,
        *,
        device: str | torch.device = "cpu",
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.model = model.to(device).eval()
        self.flow = action_flow.to(device).eval()
        self.legacy_world_model = legacy_world_model.to(device).eval()
        self.legacy_flow = legacy_action_flow.to(device).eval()
        self.config = config or MultimodalJointWAMPolicyConfig()
        self.device = torch.device(device)
        self.dtype = next(self.flow.parameters()).dtype
        self.clock = clock
        if self.model.planning_feature_dim != self.flow.config.feature_dim:
            raise ValueError("M1 policy feature dimensions differ")
        if self.flow.config.horizon != self.config.action_chunk.horizon:
            raise ValueError("M1 policy/flow chunk horizons differ")
        self.flow.freeze_anchor()
        self.legacy_flow.freeze_anchor()
        for module in (
            self.model,
            self.flow,
            self.legacy_world_model,
            self.legacy_flow,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.fixed_actions = {
            int(index): float(value) for index, value in self.config.fixed_actions
        }
        self._states: deque[np.ndarray] = deque(
            maxlen=self.model.world_model.config.history_horizon
        )
        self._actions: deque[np.ndarray] = deque(
            maxlen=max(self.model.world_model.config.history_horizon - 1, 1)
        )
        self._images: deque[np.ndarray] = deque(
            maxlen=self.config.visual_history_frames
        )
        self._image_indices: deque[int] = deque(
            maxlen=self.config.visual_history_frames
        )
        self._vision_features: deque[VisionEncoderOutput] = deque(
            maxlen=self.config.visual_history_frames
        )
        self._chunk: torch.Tensor | None = None
        self._chunk_cursor = 0
        self._executed_since_generation = 0
        self._control_step = 0
        self._last_new_image_step = 0
        self._last_new_image_received_at: float | None = None
        self.last_diagnostics: dict[str, Any] = {}
        self._legacy_policy = JointWAMPolicy(
            self.legacy_world_model,
            self.legacy_flow,
            config=JointWAMPolicyConfig(
                action_chunk=self.config.action_chunk,
                solver=self.config.solver,
                anchor_residual_scale=self.config.cooperative_residual_scale,
                normalized_action_clip=self.config.normalized_action_clip,
                latency_budget_ms=self.config.latency_budget_ms,
                fallback_enabled=False,
                risk_veto=True,
            ),
            fixed_actions=self.fixed_actions,
            clock=self.clock,
        )

    @property
    def canonical_variant(self) -> str:
        if self.model.config.use_state and not self.model.config.use_vision:
            return "state_only"
        if self.model.config.use_vision and not self.model.config.use_state:
            return "vision_only"
        if self.model.config.capacity_control == "future_head":
            return "state_vision_future"
        if self.model.config.capacity_control == "action_mlp":
            return "parameter_matched_mlp"
        return "state_vision_no_future"

    @property
    def action_source(self) -> str:
        return f"m1_{self.canonical_variant}"

    def reset(self) -> None:
        self._states.clear()
        self._actions.clear()
        self._images.clear()
        self._image_indices.clear()
        self._vision_features.clear()
        self._chunk = None
        self._chunk_cursor = 0
        self._executed_since_generation = 0
        self._control_step = 0
        self._last_new_image_step = 0
        self._last_new_image_received_at = None
        self.last_diagnostics = {}
        self._legacy_policy.reset()

    @torch.inference_mode()
    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        start = self.clock()
        presented = _leaf_paths(observation)
        forbidden = _forbidden_runtime_paths(presented)
        if forbidden:
            raise RuntimeError(
                f"forbidden observation leaked into M1 policy: {list(forbidden)}"
            )
        task_id = self._task_id(observation)
        self._consume_presented_action_history(observation)
        image: np.ndarray | None = None
        frame_index = -1
        is_new = False
        if self.model.config.use_vision:
            image, frame_index, is_new = self._fixed_image(observation)
            if is_new:
                self._images.append(image)
                self._image_indices.append(frame_index)
                self._vision_features.append(self._encode_visual_frame(image))
                self._last_new_image_step = self._control_step
                # The runner exposes a simulation timestamp, which cannot be
                # subtracted from a monotonic wall clock during accelerated
                # evaluation.  Record when the raw frame is received instead;
                # subsequent cached actions then inherit any earlier planning
                # overrun rather than resetting their age to a nominal 50 ms.
                self._last_new_image_received_at = start
        state = self._state(observation) if self.model.config.use_state else None
        if state is not None:
            self._states.append(state)
        consumes_vision = self.model.config.use_vision
        nominal_visual_staleness_ms = (
            self.config.control_period_ms
            * (self._control_step - self._last_new_image_step)
            if consumes_vision
            else 0.0
        )
        wall_visual_staleness_ms = (
            max(0.0, (start - self._last_new_image_received_at) * 1000.0)
            if consumes_vision and self._last_new_image_received_at is not None
            else 0.0
        )
        visual_staleness_ms = max(nominal_visual_staleness_ms, wall_visual_staleness_ms)
        # A late frame/action remains part of the closed-loop evidence.  The
        # policy has no fallback, so aborting here would censor exactly the
        # deadline misses the runtime report is required to measure.  Execute
        # the direct model action and mark ``deadline_exceeded`` below.

        # The exact cooperative preservation bypass is state-based.  A
        # vision-only ablation therefore stays on its own direct visual flow
        # and never requests proprioception.  State-only uses the bypass but
        # does not request or encode RGB.
        if task_id == "cooperative_stop" and self.model.config.use_state:
            action = self._act_cooperative_bypass(
                observation,
                image=image,
                frame_index=frame_index,
                is_new=is_new,
                start=start,
                presented=presented,
                visual_staleness_ms=visual_staleness_ms,
            )
            self._control_step += 1
            self._actions.append(action.copy())
            return action

        if task_id not in self.model.config.task_vocabulary:
            raise ValueError(f"M1 policy received unknown task {task_id!r}")
        if self.model.config.use_state and not self._states:
            raise RuntimeError("M1 state history was not initialized")
        replan = (
            self._chunk is None
            or self._executed_since_generation
            >= self.config.action_chunk.execution_steps
            or self._chunk_cursor >= self.config.action_chunk.horizon
            or (is_new and self._chunk is not None)
        )
        planned_mode = "none"
        warm_start_used = False
        if replan:
            planned_mode = "m1_latent_flow"
            states, past_actions, valid_mask = self._history()
            images = self._image_history() if self.model.config.use_vision else None
            task_index = self.model.task_indices(task_id, device=self.device)
            if isinstance(self.model, LatentWAM) and self.model.config.use_vision:
                encoding = self.model.encode(
                    states,
                    past_actions,
                    valid_mask,
                    images,
                    task_index,
                    vision_features=self._vision_feature_history(),
                )
            else:
                # Test doubles and legacy protocol shims retain the original
                # five-positional-argument surface.
                encoding = self.model.encode(
                    states,
                    past_actions,
                    valid_mask,
                    images,
                    task_index,
                )
            warm = self._warm_start()
            warm_start_used = warm is not None
            generated = self.flow.generate(
                encoding.planning_features,
                initial_actions=warm,
                solver_steps=self.config.action_chunk.solver_steps,
                solver=self.config.solver,
                normalized_clip=self.config.normalized_action_clip,
            )
            anchor_action = self.flow.anchor_action(encoding.planning_features)
            anchor = anchor_action[:, None].expand_as(generated)
            chunk = anchor + self.config.visual_residual_scale * (generated - anchor)
            chunk = self._fix_chunk(chunk)
            self._chunk = chunk[0]
            self._chunk_cursor = 0
            self._executed_since_generation = 0
        if self._chunk is None:
            raise RuntimeError("M1 direct flow produced no chunk")
        action = self._chunk[self._chunk_cursor].detach().cpu().numpy()
        action = self._finish(action)
        self._chunk_cursor += 1
        self._executed_since_generation += 1
        self._control_step += 1
        self._actions.append(action.copy())
        finished = self.clock()
        latency_ms = float((finished - start) * 1000.0)
        # ``visual_staleness_ms`` is capture-to-decision-start age induced by
        # the declared 20 Hz control / 10 Hz RGB decimation.  Sensor-to-action
        # age must also include the wall-clock work needed to produce this
        # action; otherwise a 50 ms-old frame plus an 80 ms plan would be
        # incorrectly reported as a safe 50 ms action.
        wall_action_age_ms = (
            max(0.0, (finished - self._last_new_image_received_at) * 1000.0)
            if consumes_vision and self._last_new_image_received_at is not None
            else 0.0
        )
        action_age_ms = (
            max(
                nominal_visual_staleness_ms + latency_ms,
                wall_action_age_ms,
            )
            if consumes_vision
            else latency_ms
        )
        deadline_budget_ms = (
            self.config.maximum_visual_age_ms
            if consumes_vision
            else self.config.latency_budget_ms
        )
        consumed = ["task.id", "task.text", "past_executed_actions"]
        if self.model.config.use_vision:
            consumed.extend(("images.fixed", "image_frame_indices.fixed"))
        if self.model.config.use_state:
            consumed.append("proprioception")
        self.last_diagnostics = {
            "action_source": self.action_source,
            "executed_mode": "m1_latent_flow",
            "planned_mode": planned_mode,
            "plan_executed": True,
            "direct_flow_generated": planned_mode == "m1_latent_flow",
            "direct_flow_executed": True,
            "fallback_used": False,
            "fallback_reason": "none",
            "fallback_enabled": False,
            "single_active_expert": True,
            "warm_start_used": warm_start_used,
            "latency_ms": latency_ms,
            "planning_latency_ms": latency_ms if planned_mode != "none" else 0.0,
            # A vision-conditioned plan can use the declared 10 Hz / 100 ms
            # decimated path.  State-only has no such decimation contract and
            # therefore remains subject to the direct 50 ms budget.
            "deadline_exceeded": (
                action_age_ms > deadline_budget_ms
                if consumes_vision
                else planned_mode != "none" and latency_ms > deadline_budget_ms
            ),
            "deadline_budget_ms": deadline_budget_ms,
            "deadline_mode": (
                "decimated_visual" if consumes_vision else "direct_state"
            ),
            "visual_staleness_ms": visual_staleness_ms,
            "nominal_visual_staleness_ms": nominal_visual_staleness_ms,
            "wall_visual_staleness_ms": wall_visual_staleness_ms,
            "wall_action_age_ms": wall_action_age_ms,
            "action_age_ms": action_age_ms,
            # Compatibility alias; unlike the old implementation this is the
            # full capture-to-action age, not frame staleness alone.
            "visual_age_ms": action_age_ms,
            "visual_frame_index": frame_index,
            "new_visual_frame": is_new,
            "presented_observation_paths": presented,
            "consumed_observation_paths": sorted(consumed),
            "privileged_state_seen": False,
            "model_variant": self.canonical_variant,
            "visual_residual_scale": self.config.visual_residual_scale,
            "replan_warm_start_enabled": self.config.replan_warm_start_enabled,
        }
        return action

    def _act_cooperative_bypass(
        self,
        observation: Mapping[str, Any],
        *,
        image: np.ndarray | None,
        frame_index: int,
        is_new: bool,
        start: float,
        presented: tuple[str, ...],
        visual_staleness_ms: float,
    ) -> np.ndarray:
        if "proprioception" not in observation:
            raise KeyError("cooperative preservation path requires proprioception")
        # The main state+vision model explicitly consumes RGB before applying
        # a zero multimodal residual gate.  The state-only contrast never sees
        # RGB, including on this retained task.
        if self.model.config.use_vision:
            if (
                image is None
                or self.model.vision_encoder is None
                or not self._vision_features
            ):
                raise RuntimeError("cooperative multimodal bypass requires fixed RGB")
        legacy_observation = {"proprioception": observation["proprioception"]}
        action = self._legacy_policy.act(legacy_observation)
        legacy = dict(self._legacy_policy.last_diagnostics)
        if legacy.get("direct_flow_executed") is not True:
            raise RuntimeError(
                "cooperative preservation flow failed instead of executing directly"
            )
        finished = self.clock()
        latency_ms = float((finished - start) * 1000.0)
        nominal_visual_staleness_ms = (
            self.config.control_period_ms
            * (self._control_step - self._last_new_image_step)
            if self.model.config.use_vision
            else 0.0
        )
        wall_action_age_ms = (
            max(0.0, (finished - self._last_new_image_received_at) * 1000.0)
            if self.model.config.use_vision
            and self._last_new_image_received_at is not None
            else 0.0
        )
        action_age_ms = (
            max(
                nominal_visual_staleness_ms + latency_ms,
                wall_action_age_ms,
            )
            if self.model.config.use_vision
            else latency_ms
        )
        deadline_budget_ms = (
            self.config.maximum_visual_age_ms
            if self.model.config.use_vision
            else self.config.latency_budget_ms
        )
        consumed = ["proprioception", "task.id"]
        if self.model.config.use_vision:
            consumed.extend(("images.fixed", "image_frame_indices.fixed"))
        diagnostics = {
            **legacy,
            "action_source": self.action_source,
            "executed_mode": "m1_cooperative_gate_zero_legacy_flow",
            "fallback_used": False,
            "fallback_enabled": False,
            "fallback_reason": "none",
            "single_active_expert": True,
            "latency_ms": latency_ms,
            "planning_latency_ms": latency_ms,
            "deadline_exceeded": action_age_ms > deadline_budget_ms,
            "deadline_budget_ms": deadline_budget_ms,
            "deadline_mode": (
                "decimated_visual" if self.model.config.use_vision else "direct_state"
            ),
            "visual_staleness_ms": visual_staleness_ms,
            "nominal_visual_staleness_ms": nominal_visual_staleness_ms,
            "wall_action_age_ms": wall_action_age_ms,
            "action_age_ms": action_age_ms,
            "visual_age_ms": action_age_ms,
            "multimodal_residual_gate": 0.0,
            "legacy_action_path": "immutable_joint_wam_direct",
            "presented_observation_paths": presented,
            "consumed_observation_paths": sorted(consumed),
            "privileged_state_seen": False,
        }
        if self.model.config.use_vision:
            diagnostics.update(
                {
                    "visual_frame_index": frame_index,
                    "new_visual_frame": is_new,
                }
            )
        self.last_diagnostics = diagnostics
        return self._finish(action)

    def _task_id(self, observation: Mapping[str, Any]) -> str:
        task = observation.get("task")
        if not isinstance(task, Mapping) or not str(task.get("id", "")):
            raise KeyError("M1 policy requires observation['task']['id']")
        if not str(task.get("text", "")):
            raise KeyError("M1 policy requires observation['task']['text']")
        return str(task["id"])

    def _consume_presented_action_history(self, observation: Mapping[str, Any]) -> None:
        if "past_executed_actions" not in observation:
            raise KeyError("M1 policy requires past_executed_actions")
        history = np.asarray(observation["past_executed_actions"], dtype=np.float32)
        if history.size == 0:
            history = np.zeros((0, self.flow.config.action_dim), dtype=np.float32)
        if history.ndim != 2 or history.shape[1] != self.flow.config.action_dim:
            raise ValueError("past_executed_actions must have shape [T,A]")
        if not np.isfinite(history).all():
            raise ValueError("past_executed_actions contains NaN or Inf")
        # The runner may expose a bounded suffix shorter than the 32-state
        # recurrent window.  Reconcile that authoritative executed suffix with
        # our longer local history instead of silently using commanded actions
        # or discarding earlier aligned transitions.
        if history.shape[0]:
            retained = list(self._actions)
            suffix = [action.copy() for action in history[-self._actions.maxlen :]]
            if len(suffix) < len(retained):
                retained[-len(suffix) :] = suffix
                merged = retained
            else:
                merged = suffix
            self._actions.clear()
            self._actions.extend(merged[-self._actions.maxlen :])

    def _state(self, observation: Mapping[str, Any]) -> np.ndarray:
        if "proprioception" not in observation:
            raise KeyError("state-enabled M1 policy requires proprioception")
        state = np.asarray(observation["proprioception"], dtype=np.float32)
        expected = self.model.world_model.config.state_dim
        if state.shape != (expected,) or not np.isfinite(state).all():
            raise ValueError(f"proprioception must be finite with shape {(expected,)}")
        return state.copy()

    def _fixed_image(
        self, observation: Mapping[str, Any]
    ) -> tuple[np.ndarray, int, bool]:
        images = observation.get("images")
        indices = observation.get("image_frame_indices")
        if not isinstance(images, Mapping) or "fixed" not in images:
            raise KeyError("M1 policy requires raw images.fixed")
        if not isinstance(indices, Mapping) or "fixed" not in indices:
            raise KeyError("M1 policy requires image_frame_indices.fixed")
        image = np.asarray(images["fixed"])
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("images.fixed must be uint8 HWC RGB")
        frame_index = int(indices["fixed"])
        if frame_index < 0:
            raise ValueError("fixed RGB frame index cannot be negative")
        is_new = not self._image_indices or frame_index != self._image_indices[-1]
        if self._image_indices and frame_index < self._image_indices[-1]:
            raise ValueError("fixed RGB frame index moved backwards")
        if (
            self._image_indices
            and frame_index == self._image_indices[-1]
            and not np.array_equal(image, self._images[-1])
        ):
            raise ValueError("fixed RGB changed without a new frame index")
        return image.copy(), frame_index, is_new

    def _encode_visual_frame(self, image: np.ndarray) -> VisionEncoderOutput:
        if self.model.vision_encoder is None:
            raise RuntimeError("M1 visual feature cache requires an encoder")
        rgb = torch.as_tensor(image, device=self.device).permute(2, 0, 1).unsqueeze(0)
        output = self.model.vision_encoder(rgb)
        return VisionEncoderOutput(
            spatial_tokens=output.spatial_tokens.detach(),
            pooled_latent=output.pooled_latent.detach(),
        )

    def _vision_feature_history(self) -> VisionEncoderOutput:
        if not self._vision_features or len(self._vision_features) != len(self._images):
            raise RuntimeError("M1 visual feature cache is not aligned with raw RGB")
        spatial = (
            torch.stack(
                [value.spatial_tokens[0] for value in self._vision_features], dim=0
            )
            .unsqueeze(0)
            .unsqueeze(2)
        )
        pooled = (
            torch.stack(
                [value.pooled_latent[0] for value in self._vision_features], dim=0
            )
            .unsqueeze(0)
            .unsqueeze(2)
        )
        return VisionEncoderOutput(
            spatial_tokens=spatial.detach(),
            pooled_latent=pooled.detach(),
        )

    def _history(
        self,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not self.model.config.use_state:
            return None, None, None
        horizon = self.model.world_model.config.history_horizon
        count = len(self._states)
        offset = horizon - count
        states = torch.zeros(
            1,
            horizon,
            self.model.world_model.config.state_dim,
            device=self.device,
            dtype=self.dtype,
        )
        states[0, offset:] = torch.as_tensor(
            np.stack(self._states), device=self.device, dtype=self.dtype
        )
        actions = torch.zeros(
            1,
            horizon - 1,
            self.flow.config.action_dim,
            device=self.device,
            dtype=self.dtype,
        )
        needed = max(count - 1, 0)
        if needed:
            if len(self._actions) < needed:
                raise RuntimeError(
                    "presented executed-action history is shorter than state history"
                )
            actions[0, offset:] = torch.as_tensor(
                np.stack(list(self._actions)[-needed:]),
                device=self.device,
                dtype=self.dtype,
            )
        valid = torch.zeros(1, horizon, device=self.device, dtype=torch.bool)
        valid[0, offset:] = True
        return states, actions, valid

    def _image_history(self) -> torch.Tensor:
        if not self._images:
            raise RuntimeError("M1 visual history is empty")
        images = torch.as_tensor(np.stack(self._images), device=self.device)
        return images.permute(0, 3, 1, 2).unsqueeze(0).unsqueeze(2)

    def _warm_start(self) -> torch.Tensor | None:
        if not self.config.replan_warm_start_enabled:
            return None
        if self._chunk is None or self._executed_since_generation <= 0:
            return None
        if self._executed_since_generation >= self.config.action_chunk.horizon:
            return None
        return shift_action_chunk_warm_start(
            self._chunk,
            self.config.action_chunk,
            executed_steps=self._executed_since_generation,
        ).unsqueeze(0)

    def _fix_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        result = chunk.clamp(-1.0, 1.0).clone()
        for index, value in self.fixed_actions.items():
            result[..., index] = value
        return result

    def _finish(self, raw: np.ndarray) -> np.ndarray:
        action = np.asarray(raw, dtype=np.float32).reshape(-1)
        if (
            action.shape != (self.flow.config.action_dim,)
            or not np.isfinite(action).all()
        ):
            raise RuntimeError("M1 direct policy produced an invalid action")
        action = np.clip(action, -1.0, 1.0)
        for index, value in self.fixed_actions.items():
            action[index] = value
        return action.astype(np.float32, copy=False)


def _leaf_paths(value: Any, prefix: str = "") -> tuple[str, ...]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        if not value and prefix:
            paths.append(prefix)
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_leaf_paths(value[key], child))
    elif prefix:
        paths.append(prefix)
    return tuple(paths)


def _forbidden_runtime_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    exact = {
        "privileged_state",
        "cue_id",
        "cue_variant",
        "rendered_cue_variant",
        "event_truth",
        "event_active",
        "visual_signal_active",
    }
    rejected: list[str] = []
    for path in paths:
        segments = tuple(segment.lower() for segment in path.split("."))
        if any(
            segment in exact
            or segment.startswith("future_")
            or segment.startswith("next_observation")
            for segment in segments
        ):
            rejected.append(path)
    return tuple(sorted(rejected))


__all__ = ["MultimodalJointWAMPolicy", "MultimodalJointWAMPolicyConfig"]
