"""Direct inference policy for a scratch-trained multimodal M1 checkpoint."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
import time
from typing import Any, Callable, TYPE_CHECKING

import numpy as np
import torch

from models.wam import (
    ActionChunkConfig,
    AffineActionCodec,
    StatefulActionFlow,
    shift_action_chunk_warm_start,
)
from models.wam_multimodal import LatentWAM

if TYPE_CHECKING:
    from train.m1_scratch_builder import ScratchM1Bundle


@dataclass(frozen=True)
class ScratchM1PolicyConfig:
    """Runtime contract for one 16D scratch policy with no legacy bypass."""

    action_chunk: ActionChunkConfig = field(
        default_factory=lambda: ActionChunkConfig(
            action_dim=16,
            horizon=8,
            execution_steps=2,
            solver_steps=4,
        )
    )
    camera_order: tuple[str, ...] = ("global",)
    visual_history_frames: int = 2
    solver: str = "euler"
    normalized_action_clip: float = 10.0
    replan_on_new_image: bool = True
    replan_warm_start_enabled: bool = True
    action_history_key: str = "past_executed_actions"

    def __post_init__(self) -> None:
        cameras = tuple(str(value) for value in self.camera_order)
        if not cameras or any(not value for value in cameras):
            raise ValueError("scratch policy camera_order cannot be empty")
        if len(set(cameras)) != len(cameras):
            raise ValueError("scratch policy camera_order must be unique")
        if self.action_chunk.horizon != 8 or self.action_chunk.execution_steps != 2:
            raise ValueError("scratch M1 requires an 8-step chunk and execute-2")
        if self.solver not in {"euler", "heun"}:
            raise ValueError("scratch policy solver must be euler or heun")
        if self.visual_history_frames <= 0 or self.normalized_action_clip <= 0.0:
            raise ValueError("invalid scratch policy history/clip controls")
        if not self.action_history_key:
            raise ValueError("scratch policy action_history_key cannot be empty")
        object.__setattr__(self, "camera_order", cameras)


class ScratchM1Policy:
    """Run the new M1 expert and decode canonical actions to controller units."""

    def __init__(
        self,
        model: LatentWAM,
        action_flow: StatefulActionFlow,
        action_codec: AffineActionCodec,
        config: ScratchM1PolicyConfig | None = None,
        *,
        device: str | torch.device = "cpu",
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config or ScratchM1PolicyConfig()
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.flow = action_flow.to(self.device).eval()
        self.codec = action_codec.to(self.device).eval()
        self.clock = clock
        self.dtype = next(self.flow.parameters()).dtype
        action_dim = self.flow.config.action_dim
        if self.flow.has_anchor or self.flow.config.anchor_mode != "none":
            raise ValueError("scratch policy rejects a legacy action anchor")
        if self.model.planning_feature_dim != self.flow.config.feature_dim:
            raise ValueError("scratch policy model/flow feature dimensions differ")
        if self.model.world_model.config.action_dim != action_dim:
            raise ValueError("scratch policy world/flow action dimensions differ")
        if self.codec.action_dim != action_dim:
            raise ValueError("scratch policy codec/flow action dimensions differ")
        if self.config.action_chunk.action_dim != action_dim:
            raise ValueError("scratch policy config/flow action dimensions differ")
        if self.config.action_chunk.horizon != self.flow.config.horizon:
            raise ValueError("scratch policy config/flow horizons differ")
        for module in (self.model, self.flow, self.codec):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        if self.model.vision_encoder is not None:
            self.model.vision_encoder.eval()

        history_horizon = self.model.world_model.config.history_horizon
        self._states: deque[np.ndarray] = deque(maxlen=history_horizon)
        self._canonical_actions: deque[np.ndarray] = deque(
            maxlen=max(history_horizon - 1, 1)
        )
        self._images: deque[np.ndarray] = deque(
            maxlen=self.config.visual_history_frames
        )
        self._image_indices: deque[tuple[int, ...]] = deque(
            maxlen=self.config.visual_history_frames
        )
        self._chunk: torch.Tensor | None = None
        self._chunk_cursor = 0
        self._executed_since_generation = 0
        self.last_diagnostics: dict[str, Any] = {}

    @classmethod
    def from_bundle(
        cls,
        bundle: "ScratchM1Bundle",
        config: ScratchM1PolicyConfig | None = None,
        *,
        device: str | torch.device = "cpu",
        clock: Callable[[], float] = time.perf_counter,
    ) -> "ScratchM1Policy":
        return cls(
            bundle.model,
            bundle.action_flow,
            bundle.action_codec,
            config,
            device=device,
            clock=clock,
        )

    def reset(self) -> None:
        self._states.clear()
        self._canonical_actions.clear()
        self._images.clear()
        self._image_indices.clear()
        self._chunk = None
        self._chunk_cursor = 0
        self._executed_since_generation = 0
        self.last_diagnostics = {}

    @torch.inference_mode()
    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        started_at = self.clock()
        presented = _leaf_paths(observation)
        forbidden = _forbidden_runtime_paths(presented)
        if forbidden:
            raise RuntimeError(
                f"forbidden observation leaked into scratch policy: {list(forbidden)}"
            )
        task_id = self._task_id(observation)
        self._consume_raw_action_history(observation)
        is_new_image = False
        frame_indices: tuple[int, ...] = ()
        if self.model.config.use_vision:
            images, frame_indices, is_new_image = self._camera_frame(observation)
            if is_new_image:
                self._images.append(images)
                self._image_indices.append(frame_indices)
        if self.model.config.use_state:
            self._states.append(self._state(observation))

        replan = (
            self._chunk is None
            or self._executed_since_generation
            >= self.config.action_chunk.execution_steps
            or self._chunk_cursor >= self.config.action_chunk.horizon
            or (
                self.config.replan_on_new_image
                and is_new_image
                and self._chunk is not None
            )
        )
        warm_start_used = False
        if replan:
            states, past_actions, valid = self._history()
            images = self._image_history() if self.model.config.use_vision else None
            task_index = self.model.task_indices(task_id, device=self.device)
            encoding = self.model.encode(
                states,
                past_actions,
                valid,
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
            self._chunk = generated[0].clamp(-1.0, 1.0)
            self._chunk_cursor = 0
            self._executed_since_generation = 0
        if self._chunk is None:
            raise RuntimeError("scratch policy failed to produce an action chunk")

        canonical = self._chunk[self._chunk_cursor].detach()
        raw = self.codec.decode(canonical, clip=True)
        if not isinstance(raw, torch.Tensor):
            raise TypeError("scratch codec returned an unexpected action type")
        canonical_np = canonical.cpu().numpy().astype(np.float32, copy=True)
        raw_np = raw.cpu().numpy().astype(np.float32, copy=False)
        self._validate_raw_output(raw_np)
        self._canonical_actions.append(canonical_np)
        self._chunk_cursor += 1
        self._executed_since_generation += 1
        latency_ms = float((self.clock() - started_at) * 1000.0)
        consumed = ["task.id", "task.text"]
        if self.model.config.use_state:
            consumed.append("proprioception")
        if self.model.config.use_vision:
            consumed.extend(
                f"{root}.{camera}"
                for root in ("images", "image_frame_indices")
                for camera in self.config.camera_order
            )
        if self.config.action_history_key in observation:
            consumed.append(self.config.action_history_key)
        self.last_diagnostics = {
            "action_source": "m1_scratch_latent_flow",
            "initialization_mode": "scratch",
            "action_anchor_mode": "none",
            "legacy_bypass_used": False,
            "fallback_used": False,
            "direct_flow_generated": replan,
            "warm_start_used": warm_start_used,
            "model_action_domain": self.codec.config.encoded_domain,
            "controller_action_domain": self.codec.config.raw_domain,
            "action_codec_sha256": self.codec.semantic_sha256,
            "action_dim": self.codec.action_dim,
            "latency_ms": latency_ms,
            "frame_indices": list(frame_indices),
            "new_visual_frame": is_new_image,
            "presented_observation_paths": presented,
            "consumed_observation_paths": sorted(consumed),
            "privileged_state_seen": False,
        }
        return raw_np

    def _task_id(self, observation: Mapping[str, Any]) -> str:
        task = observation.get("task")
        if not isinstance(task, Mapping) or not str(task.get("id", "")):
            raise KeyError("scratch M1 policy requires task.id")
        if not str(task.get("text", "")):
            raise KeyError("scratch M1 policy requires task.text")
        task_id = str(task["id"])
        if task_id not in self.model.config.task_vocabulary:
            raise ValueError(f"scratch M1 policy received unknown task {task_id!r}")
        return task_id

    def _state(self, observation: Mapping[str, Any]) -> np.ndarray:
        state = np.asarray(observation.get("proprioception"), dtype=np.float32)
        expected = self.model.world_model.config.state_dim
        if state.shape != (expected,) or not np.isfinite(state).all():
            raise ValueError(f"proprioception must be finite with shape {(expected,)}")
        return state.copy()

    def _consume_raw_action_history(self, observation: Mapping[str, Any]) -> None:
        key = self.config.action_history_key
        if key not in observation:
            return
        raw = np.asarray(observation[key], dtype=np.float32)
        if raw.size == 0:
            raw = np.zeros((0, self.codec.action_dim), dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != self.codec.action_dim:
            raise ValueError(f"{key} must have shape [T,{self.codec.action_dim}]")
        canonical = self.codec.encode(raw, validate=True)
        if not isinstance(canonical, np.ndarray):
            raise TypeError("scratch codec returned an unexpected history type")
        suffix = [value.copy() for value in canonical[-self._canonical_actions.maxlen :]]
        if suffix:
            retained = list(self._canonical_actions)
            if len(suffix) < len(retained):
                retained[-len(suffix) :] = suffix
                suffix = retained
            self._canonical_actions.clear()
            self._canonical_actions.extend(suffix[-self._canonical_actions.maxlen :])

    def _camera_frame(
        self, observation: Mapping[str, Any]
    ) -> tuple[np.ndarray, tuple[int, ...], bool]:
        images = observation.get("images")
        indices = observation.get("image_frame_indices")
        if not isinstance(images, Mapping) or not isinstance(indices, Mapping):
            raise KeyError("scratch M1 policy requires images/image_frame_indices")
        frames: list[np.ndarray] = []
        frame_indices: list[int] = []
        for camera in self.config.camera_order:
            if camera not in images or camera not in indices:
                raise KeyError(f"scratch M1 policy requires camera {camera!r}")
            frame = np.asarray(images[camera])
            if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
                raise ValueError(f"images.{camera} must be uint8 HWC RGB")
            index = int(indices[camera])
            if index < 0:
                raise ValueError("scratch M1 image frame index cannot be negative")
            frames.append(frame.copy())
            frame_indices.append(index)
        shapes = {frame.shape for frame in frames}
        if len(shapes) != 1:
            raise ValueError("scratch M1 camera images must have equal shapes")
        index_tuple = tuple(frame_indices)
        if self._image_indices:
            previous = self._image_indices[-1]
            if any(now < old for now, old in zip(index_tuple, previous, strict=True)):
                raise ValueError("scratch M1 image frame index moved backwards")
        is_new = not self._image_indices or index_tuple != self._image_indices[-1]
        stacked = np.stack(frames, axis=0)
        if not is_new and not np.array_equal(stacked, self._images[-1]):
            raise ValueError("scratch M1 RGB changed without a new frame index")
        return stacked, index_tuple, is_new

    def _history(
        self,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not self.model.config.use_state:
            return None, None, None
        horizon = self.model.world_model.config.history_horizon
        count = len(self._states)
        if count == 0:
            raise RuntimeError("scratch M1 state history is empty")
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
            self.codec.action_dim,
            device=self.device,
            dtype=self.dtype,
        )
        needed = max(count - 1, 0)
        if needed:
            if len(self._canonical_actions) < needed:
                raise RuntimeError("scratch M1 action/state histories are not aligned")
            actions[0, offset:] = torch.as_tensor(
                np.stack(list(self._canonical_actions)[-needed:]),
                device=self.device,
                dtype=self.dtype,
            )
        valid = torch.zeros(1, horizon, device=self.device, dtype=torch.bool)
        valid[0, offset:] = True
        return states, actions, valid

    def _image_history(self) -> torch.Tensor:
        if not self._images:
            raise RuntimeError("scratch M1 visual history is empty")
        # Stored as [T,Cam,H,W,C]; model input is [B,T,Cam,C,H,W].
        values = torch.as_tensor(np.stack(self._images), device=self.device)
        return values.permute(0, 1, 4, 2, 3).unsqueeze(0)

    def _warm_start(self) -> torch.Tensor | None:
        if not self.config.replan_warm_start_enabled:
            return None
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

    def _validate_raw_output(self, action: np.ndarray) -> None:
        if action.shape != (self.codec.action_dim,) or not np.isfinite(action).all():
            raise RuntimeError("scratch M1 policy produced an invalid raw action")
        low = np.asarray(self.codec.config.low, dtype=np.float32)
        high = np.asarray(self.codec.config.high, dtype=np.float32)
        if np.any(action < low - 1e-6) or np.any(action > high + 1e-6):
            raise RuntimeError("decoded scratch M1 action exceeded controller bounds")


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


__all__ = ["ScratchM1Policy", "ScratchM1PolicyConfig"]
