"""Runtime for the strictly decentralized DuoBench DINO B0-H policy.

``DuoB0HRuntime`` owns one history and one temporal-ensemble queue per arm,
but a single shared policy module.  The only image supplied to an arm is the
shared head image and that arm's own wrist image; all proprioception and past
actions are local.  Actions are absolute joint targets (binary gripper).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import torch

from before_we_act.temporal_history_data import task_text_tensor
from before_we_act.temporal_history_policy import TemporalHistoryPolicy
from deployment.duo_act.action_target import (
    ACTION_TARGET_CONTRACT_ID,
    ACTION_TARGET_CONTRACT_SHA256,
    canonicalize_controller_action,
)

from .data import (
    ACTION_DIM,
    ACTION_HORIZON,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    HISTORY_STEPS,
    STATE_DIM,
    TASKS,
    TASK_TEXT,
    resize_rgb_batch,
)
from .preprocessing import (
    DINO_NORMALIZATION_ID,
    IMAGE_PREPROCESS_ID,
    validate_dino_normalization_buffers,
)


def _as_frame(value: Any) -> np.ndarray:
    """Convert common RCS/LeRobot RGB containers to ``H,W,3`` uint8."""

    if isinstance(value, Mapping):
        if "rgb" in value:
            value = value["rgb"]
        if isinstance(value, Mapping):
            value = value.get("data", value.get("image", value))
    array = np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected RGB frame HxWx3, got {array.shape}")
    if array.dtype != np.uint8:
        maximum = float(array.max()) if array.size else 0.0
        if np.issubdtype(array.dtype, np.floating) and maximum <= 1.0:
            array = np.rint(array * 255.0)
        array = np.asarray(array, dtype=np.uint8)
    return np.ascontiguousarray(array)


def _arm_entry(observation: Mapping[str, Any], arm: int) -> Mapping[str, Any]:
    key = "left" if arm == 0 else "right"
    if key in observation and isinstance(observation[key], Mapping):
        return observation[key]
    agents = observation.get("agent") or observation.get("agents")
    if isinstance(agents, Mapping):
        for candidate in (f"panda-{arm}", f"panda_{arm}", str(arm)):
            if candidate in agents:
                return agents[candidate]
    raise KeyError(f"missing DuoBench arm {arm} observation")


def _arm_qpos(observation: Mapping[str, Any], arm: int) -> np.ndarray:
    entry = _arm_entry(observation, arm)
    if "joints" in entry:
        joints = np.asarray(entry["joints"], dtype=np.float32).reshape(-1)
        gripper = np.asarray(entry.get("gripper", [0.0]), dtype=np.float32).reshape(-1)
        if len(joints) < 7 or len(gripper) < 1:
            raise ValueError(f"invalid arm {arm} qpos shape")
        return np.concatenate((joints[:7], gripper[:1])).astype(np.float32)
    for key in ("qpos", "joint_positions", "position"):
        if key in entry:
            value = np.asarray(entry[key], dtype=np.float32).reshape(-1)
            if value.size >= 8:
                return value[:8].astype(np.float32)
    raise KeyError(f"missing qpos/joints for DuoBench arm {arm}")


def _frames(observation: Mapping[str, Any], arm: int) -> tuple[np.ndarray, np.ndarray]:
    frames = observation.get("frames")
    if isinstance(frames, Mapping):
        head_value = frames.get("head")
        wrist_value = frames.get("left_wrist" if arm == 0 else "right_wrist")
        if head_value is not None and wrist_value is not None:
            return _as_frame(head_value), _as_frame(wrist_value)
    # A few exported replay records use flat camera names.
    head = observation.get("head")
    if head is None:
        head = observation.get("head_camera")
    wrist = observation.get("left_wrist" if arm == 0 else "right_wrist")
    if head is None or wrist is None:
        sensor = observation.get("sensor_data")
        if isinstance(sensor, Mapping):
            head = sensor.get("head_camera_global")
            if head is None:
                head = sensor.get("head")
            wrist = sensor.get(f"head_camera_agent{arm}")
            if wrist is None:
                wrist = sensor.get("left" if arm == 0 else "right")
    if head is None or wrist is None:
        raise KeyError(f"missing head/own-wrist camera for arm {arm}")
    return _as_frame(head), _as_frame(wrist)


@dataclass
class _ArmHistory:
    visual: deque
    qpos: deque
    action: deque

    @classmethod
    def create(cls) -> "_ArmHistory":
        return cls(
            visual=deque(maxlen=HISTORY_STEPS - 1),
            qpos=deque(maxlen=HISTORY_STEPS - 1),
            action=deque(maxlen=HISTORY_STEPS),
        )


class _AbsoluteEnsemble:
    def __init__(self, arms: Sequence[int], decay: float = 0.01):
        if decay < 0:
            raise ValueError("ensemble decay must be non-negative")
        self.arms = tuple(int(arm) for arm in arms)
        self.decay = float(decay)
        self.chunks: dict[int, list[tuple[int, np.ndarray]]] = {
            arm: [] for arm in self.arms
        }

    def reset(self) -> None:
        for values in self.chunks.values():
            values.clear()

    def add_and_plan(self, step: int, predictions: np.ndarray) -> dict[int, np.ndarray]:
        """Append a proposal and return the full 100-step consolidated plan.

        CARE constructs its counterfactual candidates from the *already
        temporally ensembled* reference/base plan.  The old public helper only
        returned the first command, which made it tempting for callers to feed
        a ``[arms,8]`` array back into the ensemble (and, more subtly, changed
        the registered candidate semantics).  Keep the queue in absolute joint
        coordinates and expose the complete plan explicitly; ``add_and_select``
        remains a backwards-compatible first-command view below.
        """
        if predictions.shape != (len(self.arms), ACTION_HORIZON, ACTION_DIM):
            raise ValueError(
                "B0-H prediction shape differs: "
                f"expected {(len(self.arms), ACTION_HORIZON, ACTION_DIM)}, got {predictions.shape}"
            )
        result: dict[int, np.ndarray] = {}
        for row, arm in enumerate(self.arms):
            values = self.chunks[arm]
            values.append((step, np.asarray(predictions[row], dtype=np.float32).copy()))
            values[:] = [
                (proposal, chunk)
                for proposal, chunk in values
                if 0 <= step - proposal < ACTION_HORIZON
            ]
            plan: list[np.ndarray] = []
            for offset in range(ACTION_HORIZON):
                absolute_step = int(step) + offset
                available = [
                    (proposal, chunk[absolute_step - proposal])
                    for proposal, chunk in values
                    if 0 <= absolute_step - proposal < ACTION_HORIZON
                ]
                if not available:
                    # The newly appended proposal must cover its complete
                    # horizon.  Failing here is safer than silently shortening
                    # a candidate plan at a branch/validation boundary.
                    raise RuntimeError("newest proposal does not cover its horizon")
                # Newer proposals receive larger weights while preserving the
                # exact absolute action frame (no q-ref rebasing is needed).
                age = np.asarray(
                    [absolute_step - proposal for proposal, _ in available],
                    dtype=np.float64,
                )
                weights = np.exp(-self.decay * age)
                weights /= weights.sum()
                plan.append(
                    np.sum(
                        np.stack([value for _proposal, value in available])
                        * weights[:, None],
                        axis=0,
                    ).astype(np.float32)
                )
            result[arm] = np.asarray(plan, dtype=np.float32)
        return result

    def add_and_select(self, step: int, predictions: np.ndarray) -> dict[int, np.ndarray]:
        """Append a proposal and return its current (first) command."""

        plans = self.add_and_plan(step, predictions)
        return {arm: plan[0].copy() for arm, plan in plans.items()}


def _controller_equivalent_payload(
    selected: Mapping[int, np.ndarray],
    arms: Sequence[int],
    *,
    action_spaces: Mapping[str, Any] | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    """Build commands using the pinned MuJoCo controller contract.

    ``action_spaces`` remains in the signature for compatibility with older
    callers, but its conservative RCS/Gym bounds are intentionally ignored.
    The canonicalizer reproduces the position actuator ``ctrlrange`` and
    binary-gripper operation used for every prepared training target.
    """

    del action_spaces
    arm_ids = tuple(int(arm) for arm in arms)
    raw = np.stack(
        [np.asarray(selected[arm], dtype=np.float32) for arm in arm_ids], axis=0
    )
    expected = (len(arm_ids), ACTION_DIM)
    if raw.shape != expected:
        raise ValueError(
            f"Duo selected action shape differs: expected {expected}, got {raw.shape}"
        )
    canonical = canonicalize_controller_action(raw)
    payload: dict[str, dict[str, np.ndarray]] = {}
    for row, arm in enumerate(arm_ids):
        key = "left" if arm == 0 else "right"
        payload[key] = {
            "joints": canonical[row, :7].copy(),
            "gripper": canonical[row, 7:8].copy(),
        }
    return payload, canonical


class DuoB0HRuntime:
    """Shared-weight, local-observation runtime for two DuoBench arms."""

    def __init__(
        self,
        model: TemporalHistoryPolicy,
        stats: Mapping[str, torch.Tensor | np.ndarray],
        *,
        device: torch.device,
        image_height: int = DEFAULT_IMAGE_HEIGHT,
        image_width: int = DEFAULT_IMAGE_WIDTH,
        ensemble_decay: float = 0.01,
        arms: Sequence[int] = (0, 1),
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.arms = tuple(int(arm) for arm in arms)
        if self.arms != (0, 1):
            raise ValueError("Duo B0-H currently requires the two shared-policy arms (0,1)")
        self.q_mean = torch.as_tensor(stats["q_mean"], device=device, dtype=torch.float32)
        self.q_std = torch.as_tensor(stats["q_std"], device=device, dtype=torch.float32)
        self.a_mean = torch.as_tensor(stats["a_mean"], device=device, dtype=torch.float32)
        self.a_std = torch.as_tensor(stats["a_std"], device=device, dtype=torch.float32)
        for name, value in (("q_mean", self.q_mean), ("q_std", self.q_std), ("a_mean", self.a_mean), ("a_std", self.a_std)):
            if tuple(value.shape) != (8,) or not torch.isfinite(value).all() or (name.endswith("std") and torch.any(value <= 0)):
                raise ValueError(f"invalid B0-H normalization {name}")
        if getattr(self.model, "strict_dino_contract", False) is not True:
            raise ValueError(
                "formal Duo B0-H runtime requires strict_dino_contract=True"
            )
        validate_dino_normalization_buffers(self.model.dino_mean, self.model.dino_std)
        self.histories = {arm: _ArmHistory.create() for arm in self.arms}
        self.ensemble = _AbsoluteEnsemble(self.arms, ensemble_decay)
        self.step_index = 0
        self.task: str | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cuda:0",
        dino_model: str | None = None,
        ensemble_decay: float = 0.01,
    ) -> "DuoB0HRuntime":
        path = Path(checkpoint)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(saved, Mapping):
            raise ValueError("checkpoint payload is not a mapping")
        config = saved.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("checkpoint has no mapping config")
        if saved.get("format") != "before-we-act.duobench.dino-b0h/1":
            raise ValueError("checkpoint is not the DuoBench DINO B0-H format")
        if (
            config.get("policy_family") != "TemporalHistoryPolicy"
            # ``CARE`` identifies the method family, not the concrete policy.
            # It is intentionally required (rather than defaulted) so a
            # legacy/hand-written checkpoint cannot pass by omitting the
            # method field and then be loaded as the formal reference.
            or config.get("method_family") != "CARE"
            or config.get("architecture") != "TemporalHistoryPolicy_hidden_residual"
            or config.get("image_preprocess_id") != IMAGE_PREPROCESS_ID
            or config.get("dino_normalization_id") != DINO_NORMALIZATION_ID
            or config.get("strict_dino_contract") is not True
        ):
            raise ValueError(
                "checkpoint is not the project-owned TemporalHistoryPolicy "
                "hidden-residual B0-H architecture"
            )
        if config.get("vision_backbone") != "dinov3_vitb16_frozen":
            raise ValueError("checkpoint does not use frozen DINOv3 ViT-B/16")
        if config.get("action_encoding") != "absolute_joint7_binary_gripper1":
            raise ValueError("checkpoint does not contain absolute action8 encoding")
        model_name = dino_model or config.get("dino_model")
        if not model_name:
            raise ValueError("DINO model path is missing from checkpoint")
        dev = torch.device(device)
        model = TemporalHistoryPolicy(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            variant="hidden_residual",
            horizon=ACTION_HORIZON,
            d_model=int(config.get("d_model", 384)),
            enc_layers=int(config.get("enc_layers", 4)),
            dec_layers=int(config.get("dec_layers", 7)),
            roles=int(config.get("roles", 4)),
            role_rank=int(config.get("role_rank", 32)),
            history_layers=int(config.get("history_layers", 2)),
            dino_model=str(model_name),
            image_height=int(config.get("image_height", DEFAULT_IMAGE_HEIGHT)),
            image_width=int(config.get("image_width", DEFAULT_IMAGE_WIDTH)),
            strict_dino_contract=True,
        )
        model.load_state_dict(saved["model"], strict=True)
        return cls(
            model,
            saved["stats"],
            device=dev,
            image_height=int(config.get("image_height", DEFAULT_IMAGE_HEIGHT)),
            image_width=int(config.get("image_width", DEFAULT_IMAGE_WIDTH)),
            ensemble_decay=ensemble_decay,
        )

    def reset(self, task: str) -> None:
        if task not in TASKS:
            raise ValueError(f"unknown DuoBench task: {task}")
        self.task = task
        self.step_index = 0
        self.ensemble.reset()
        self.histories = {arm: _ArmHistory.create() for arm in self.arms}

    def _history_batch(self, qnorm: torch.Tensor, task: str) -> dict[str, torch.Tensor]:
        visual = torch.zeros(
            len(self.arms), HISTORY_STEPS, 2, 768, dtype=torch.float16, device=self.device
        )
        qpos = torch.zeros(
            len(self.arms), HISTORY_STEPS, STATE_DIM, dtype=torch.float32, device=self.device
        )
        action = torch.zeros(
            len(self.arms), HISTORY_STEPS, ACTION_DIM, dtype=torch.float32, device=self.device
        )
        hmask = torch.zeros(len(self.arms), HISTORY_STEPS, dtype=torch.bool, device=self.device)
        amask = torch.zeros(len(self.arms), HISTORY_STEPS, dtype=torch.bool, device=self.device)
        reset = []
        for row, arm in enumerate(self.arms):
            history = self.histories[arm]
            if len(history.visual) != len(history.qpos):
                raise RuntimeError("Duo B0-H visual/qpos history drift")
            if history.visual:
                first = HISTORY_STEPS - 1 - len(history.visual)
                visual[row, first:-1] = torch.stack(tuple(history.visual)).to(self.device)
                qpos[row, first:-1] = torch.stack(tuple(history.qpos)).to(self.device)
                hmask[row, first:-1] = True
            qpos[row, -1] = qnorm[row]
            hmask[row, -1] = True
            if history.action:
                first = HISTORY_STEPS - len(history.action)
                action[row, first:] = torch.stack(tuple(history.action)).to(self.device)
                amask[row, first:] = True
            reset.append(not history.visual and not history.action)
        text, text_mask = task_text_tensor(TASK_TEXT[task])
        return {
            "history_visual_raw": visual,
            "history_qpos": qpos,
            "history_action": action,
            "history_mask": hmask,
            "action_history_mask": amask,
            "task_bytes": text.unsqueeze(0).expand(len(self.arms), -1).to(self.device),
            "task_text_mask": text_mask.unsqueeze(0).expand(len(self.arms), -1).to(self.device),
            "episode_reset": torch.tensor(reset, dtype=torch.bool, device=self.device),
        }

    def _append_observation(self, visual: torch.Tensor, qnorm: torch.Tensor) -> None:
        for row, arm in enumerate(self.arms):
            self.histories[arm].visual.append(visual[row].detach().float().cpu())
            self.histories[arm].qpos.append(qnorm[row].detach().float().cpu())

    def _append_action(self, normalized: Mapping[int, torch.Tensor]) -> None:
        for arm in self.arms:
            self.histories[arm].action.append(normalized[arm].detach().float().cpu())

    @torch.inference_mode()
    def act(
        self,
        observation: Mapping[str, Any],
        task: str | None = None,
        *,
        action_spaces: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
        """Predict and execute one synchronized Duo step.

        ``action_spaces`` is accepted only for API compatibility and is never
        used for clipping.  The emitted action is canonicalized with the same
        pinned MuJoCo controller contract as the full training dataset.
        """

        task = task or self.task
        if task is None:
            raise ValueError("runtime.reset(task) or act(..., task=...) is required")
        if task not in TASKS:
            raise ValueError(task)
        if self.task != task:
            self.reset(task)
        heads, wrists, qposes = [], [], []
        for arm in self.arms:
            head, wrist = _frames(observation, arm)
            heads.append(resize_rgb_batch(head, self.image_height, self.image_width))
            wrists.append(resize_rgb_batch(wrist, self.image_height, self.image_width))
            qposes.append(_arm_qpos(observation, arm))
        head_tensor = torch.stack(heads).to(self.device).float().div_(255)
        wrist_tensor = torch.stack(wrists).to(self.device).float().div_(255)
        qraw = torch.as_tensor(np.stack(qposes), device=self.device, dtype=torch.float32)
        qnorm = (qraw - self.q_mean) / self.q_std
        temporal = self._history_batch(qnorm, task)
        if self.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction, _mu, _logvar, current_visual = self.model(
                    head_tensor,
                    wrist_tensor,
                    **temporal,
                    return_current_visual=True,
                )
        else:
            prediction, _mu, _logvar, current_visual = self.model(
                head_tensor, wrist_tensor, **temporal, return_current_visual=True
            )
        self._append_observation(current_visual, qnorm)
        decoded = (prediction.float() * self.a_std + self.a_mean).cpu().numpy()
        selected = self.ensemble.add_and_select(self.step_index, decoded)
        output, absolute = _controller_equivalent_payload(
            selected, self.arms, action_spaces=action_spaces
        )
        normalized: dict[int, torch.Tensor] = {}
        for row, arm in enumerate(self.arms):
            command = torch.as_tensor(
                absolute[row], device=self.device, dtype=torch.float32
            )
            normalized[arm] = (command - self.a_mean) / self.a_std
        self._append_action(normalized)
        diagnostics = {
            "task": task,
            "step": self.step_index,
            "action_encoding": "absolute_joint7_binary_gripper1",
            "strictly_decentralized": True,
            "image_shape": [self.image_height, self.image_width],
            "temporal_ensemble_decay": self.ensemble.decay,
            "image_preprocess_id": IMAGE_PREPROCESS_ID,
            "dino_normalization_id": DINO_NORMALIZATION_ID,
            "policy_family": "TemporalHistoryPolicy",
            "method_family": "CARE",
            "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
            "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
            "rcs_api_limits_used_for_canonicalization": False,
        }
        self.step_index += 1
        return output, diagnostics


__all__ = [
    "DuoB0HRuntime",
    "_AbsoluteEnsemble",
    "_controller_equivalent_payload",
]
