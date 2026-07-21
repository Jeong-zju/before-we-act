"""Auditable policies for the Phase M0 visual-required benchmark."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from envs.visual_required_env import (
    VISUAL_REQUIRED_TASKS,
    VISUAL_REQUIRED_TASK_TEXTS,
    VisualRequiredEnv,
    control_action,
)


_COMMON_CONSUMED_PATHS = (
    "past_executed_actions",
    "proprioception",
    "task.id",
    "task.text",
)
_VISION_CONSUMED_PATHS = (
    "images.fixed",
    *_COMMON_CONSUMED_PATHS,
)


class PrivilegedScriptedOraclePolicy:
    """Truth oracle used only to prove that each environment task is solvable."""

    def __init__(self, env: VisualRequiredEnv) -> None:
        self.env = env
        self.last_diagnostics: dict[str, Any] = {}

    def reset(self) -> None:
        self.last_diagnostics = {}

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        action = np.asarray(self.env.scripted_action(), dtype=np.float32)
        if action.shape != (self.env.action_dim,) or not np.isfinite(action).all():
            raise RuntimeError("privileged scripted oracle produced an invalid action")
        self.last_diagnostics = {
            "action_source": "privileged_scripted_oracle",
            "executed_mode": "privileged_scripted_oracle",
            "consumed_observation_paths": (),
            "observation_keys": sorted(str(key) for key in observation),
            # It reads environment truth, but no privileged *observation* was
            # presented.  The evaluator audits the action source separately.
            "privileged_state_seen": False,
        }
        return action


class StateOnlyPolicy:
    """Strong blind controller which assumes one balanced cue branch."""

    def __init__(self, *, blind_cue_variant: int = 0) -> None:
        if int(blind_cue_variant) not in {0, 1}:
            raise ValueError("blind_cue_variant must be 0 or 1")
        self.blind_cue_variant = int(blind_cue_variant)
        self._step_count = 0
        self.last_diagnostics: dict[str, Any] = {}

    def reset(self) -> None:
        self._step_count = 0
        self.last_diagnostics = {}

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        state, task_id, history = _common_inputs(observation)
        action = control_action(
            task_id,
            self.blind_cue_variant,
            state,
            step_count=self._step_count,
        )
        action = _consume_action_history(action, history)
        self._step_count += 1
        self.last_diagnostics = {
            "action_source": "state_only",
            "executed_mode": "state_only",
            "consumed_observation_paths": _COMMON_CONSUMED_PATHS,
            "observation_keys": sorted(str(key) for key in observation),
            "privileged_state_seen": False,
            "blind_cue_variant": self.blind_cue_variant,
            "action_history_length": int(history.shape[0]),
        }
        return action


class VisionOraclePolicy:
    """Pixel oracle which has no environment reference and fails closed on RGB."""

    def __init__(self) -> None:
        self._step_count = 0
        self._decoded_cue: int | None = None
        self.last_diagnostics: dict[str, Any] = {}

    def reset(self) -> None:
        self._step_count = 0
        self._decoded_cue = None
        self.last_diagnostics = {}

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        state, task_id, history = _common_inputs(observation)
        image = _fixed_rgb(observation)
        decoded, confidence = _decode_cue(task_id, image)
        if decoded is None and self._decoded_cue is not None:
            control_cue = self._decoded_cue
        elif decoded is None and task_id == "visual_event_stop":
            # Before onset the physical signal is deliberately neutral.  Cruise
            # is cue-independent and consumes no simulator event truth.
            control_cue = 1
        elif decoded in {0, 1}:
            self._decoded_cue = int(decoded)
            control_cue = self._decoded_cue
        else:
            raise RuntimeError(f"could not decode a valid visual cue for {task_id}")
        action = control_action(
            task_id,
            control_cue,
            state,
            step_count=self._step_count,
        )
        action = _consume_action_history(action, history)
        self._step_count += 1
        self.last_diagnostics = {
            "action_source": "vision_oracle",
            "executed_mode": "vision_oracle",
            "consumed_observation_paths": _VISION_CONSUMED_PATHS,
            "observation_keys": sorted(str(key) for key in observation),
            "privileged_state_seen": False,
            "decoded_cue_variant": self._decoded_cue,
            "cue_decode_confidence": float(confidence),
            "visual_signal_decoded": decoded is not None,
            "action_history_length": int(history.shape[0]),
        }
        return action


def _common_inputs(
    observation: Mapping[str, Any],
) -> tuple[np.ndarray, str, np.ndarray]:
    if not isinstance(observation, Mapping):
        raise TypeError("policy observation must be a mapping")
    if "proprioception" not in observation:
        raise KeyError("policy requires observation['proprioception']")
    state = np.asarray(observation["proprioception"], dtype=np.float32)
    if state.shape != (22,) or not np.isfinite(state).all():
        raise ValueError("proprioception must be finite with shape (22,)")

    task = observation.get("task")
    if not isinstance(task, Mapping):
        raise KeyError("policy requires observation['task'] with id/text")
    task_id = str(task.get("id", ""))
    task_text = str(task.get("text", ""))
    if task_id not in VISUAL_REQUIRED_TASKS:
        raise ValueError(f"unknown visual-required task id {task_id!r}")
    if task_text != VISUAL_REQUIRED_TASK_TEXTS[task_id]:
        raise ValueError("task text does not match the canonical cue-independent text")

    if "past_executed_actions" not in observation:
        raise KeyError("policy requires observation['past_executed_actions']")
    raw_history = np.asarray(observation["past_executed_actions"], dtype=np.float32)
    if raw_history.size == 0:
        history = np.zeros((0, 8), dtype=np.float32)
    elif raw_history.ndim == 1 and raw_history.shape == (8,):
        history = raw_history.reshape(1, 8)
    elif raw_history.ndim == 2 and raw_history.shape[1] == 8:
        history = raw_history
    else:
        raise ValueError("past_executed_actions must have shape [T,8]")
    if not np.isfinite(history).all():
        raise ValueError("past_executed_actions must be finite")
    return state.copy(), task_id, history.copy()


def _fixed_rgb(observation: Mapping[str, Any]) -> np.ndarray:
    images = observation.get("images")
    if not isinstance(images, Mapping) or "fixed" not in images:
        raise KeyError("vision oracle requires observation['images']['fixed']")
    raw = np.asarray(images["fixed"])
    if raw.ndim != 3 or raw.shape[2] != 3 or min(raw.shape[:2]) < 32:
        raise ValueError("images.fixed must have uint8-compatible shape [H,W,3]")
    if not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw).all():
        raise ValueError("images.fixed must contain finite numeric RGB values")
    if np.any(raw < 0) or np.any(raw > 255):
        raise ValueError("images.fixed values must be in [0,255]")
    image = raw.astype(np.uint8, copy=True)
    if float(image.std()) < 1.0:
        raise ValueError("images.fixed is empty or effectively constant")
    return image


def _decode_cue(task_id: str, image: np.ndarray) -> tuple[int | None, float]:
    rgb = image.astype(np.int16, copy=False)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    width = image.shape[1]
    if task_id == "visual_event_stop":
        height = image.shape[0]
        top, bottom = int(0.36 * height), int(np.ceil(0.51 * height))
        left, right = int(0.08 * width), int(np.ceil(0.92 * width))
        red = red[top:bottom, left:right]
        green = green[top:bottom, left:right]
        blue = blue[top:bottom, left:right]
        stop = (red >= 120) & (green <= 40) & (blue <= 40)
        go = (green >= 120) & (red <= 40) & (blue <= 50)
        stop_count, go_count = int(stop.sum()), int(go.sum())
        minimum = max(8, image.shape[0] * image.shape[1] // 500)
        if max(stop_count, go_count) < minimum:
            return None, 0.0
        if stop_count == go_count:
            raise RuntimeError("event signal pixels are missing or ambiguous")
        total = stop_count + go_count
        return (0 if stop_count > go_count else 1), abs(stop_count - go_count) / total
    if task_id == "visual_target_select":
        mask = (
            (red >= 100)
            & (green >= 75)
            & (blue <= 40)
            & (red - blue >= 75)
            & (green - blue >= 55)
        )
    elif task_id == "visual_obstacle_avoid":
        mask = (
            (red >= 100)
            & (green >= 28)
            & (green <= 80)
            & (blue <= 30)
            & (red - green >= 55)
        )
    else:  # guarded by _common_inputs, retained as a fail-closed boundary.
        raise ValueError(f"unknown visual-required task {task_id!r}")
    ys, xs = np.nonzero(mask)
    minimum = max(8, image.shape[0] * image.shape[1] // 700)
    if xs.size < minimum:
        return None, 0.0
    center_x = float(xs.mean())
    deadband = max(1.0, width * 0.04)
    if abs(center_x - 0.5 * (width - 1)) <= deadband:
        raise RuntimeError("visual marker is centered and therefore ambiguous")
    cue = 0 if center_x < 0.5 * (width - 1) else 1
    confidence = min(1.0, abs(center_x - 0.5 * (width - 1)) / (0.35 * width))
    return cue, float(confidence)


def _consume_action_history(action: np.ndarray, history: np.ndarray) -> np.ndarray:
    """Use the last executed command as a small deterministic smoothing prior."""

    result = np.asarray(action, dtype=np.float32).copy()
    if history.shape[0]:
        motion = np.asarray([0, 1, 2, 4, 5, 6], dtype=np.int64)
        result[motion] = 0.95 * result[motion] + 0.05 * history[-1, motion]
    result[[3, 7]] = 1.0
    return np.clip(result, -1.0, 1.0).astype(np.float32)


__all__ = [
    "PrivilegedScriptedOraclePolicy",
    "StateOnlyPolicy",
    "VisionOraclePolicy",
]
