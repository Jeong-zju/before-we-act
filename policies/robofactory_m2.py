"""Stateful direct policy for RoboFactory Phase M2 checkpoints."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn

from models.wam import AffineActionCodec, AffineActionCodecConfig
from models.wam_multimodal import BlockCausalWAM
from train.m2_training import encode_pooled_vision, m2_model_context
from train.robofactory_multitask_dataset import encode_task_text


@dataclass(frozen=True)
class RoboFactoryM2PolicyConfig:
    camera_order: tuple[str, ...] = ("global",)
    execution_steps: int = 2
    solver_steps: int = 4
    solver: str = "heun"
    normalized_action_clip: float = 10.0
    warm_start: bool = True
    future_path: bool = False

    def __post_init__(self) -> None:
        cameras = tuple(str(value) for value in self.camera_order)
        if not cameras or len(cameras) != len(set(cameras)):
            raise ValueError("M2 policy camera order must be non-empty and unique")
        if (
            self.execution_steps <= 0
            or self.solver_steps <= 0
            or self.normalized_action_clip <= 0.0
        ):
            raise ValueError("M2 execution/solver steps must be positive")
        if self.solver not in {"euler", "heun"}:
            raise ValueError("M2 solver must be euler or heun")
        object.__setattr__(self, "camera_order", cameras)


class RoboFactoryM2Policy:
    def __init__(
        self,
        model: BlockCausalWAM,
        vision_encoder: nn.Module,
        task_runtime: Sequence[Mapping[str, Any]],
        config: RoboFactoryM2PolicyConfig | None = None,
        *,
        device: str | torch.device = "cpu",
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.model = model.to(device).eval()
        self.vision_encoder = vision_encoder.to(device).eval()
        self.config = config or RoboFactoryM2PolicyConfig()
        self.device = torch.device(device)
        self.clock = clock
        self.dtype = next(self.model.parameters()).dtype
        if len(self.config.camera_order) > self.model.config.max_cameras:
            raise ValueError("policy cameras exceed the model camera slots")
        if self.config.execution_steps >= self.model.config.action_horizon:
            raise ValueError("execution_steps must be smaller than the action horizon")
        runtime = [dict(value) for value in task_runtime]
        if len(runtime) != self.model.config.num_tasks:
            raise ValueError("M2 runtime task count differs from the model")
        self.runtime = {str(value["task_id"]): value for value in runtime}
        if len(self.runtime) != len(runtime):
            raise ValueError("M2 runtime task ids must be unique")
        self.codecs = {
            task_id: AffineActionCodec(
                AffineActionCodecConfig.from_dict(value["action_codec"])
            ).to(self.device)
            for task_id, value in self.runtime.items()
        }
        self.action_normalization: dict[str, tuple[Tensor, Tensor]] = {}
        for task_id, value in self.runtime.items():
            action_dim = int(value["action_dim"])
            action_horizon = int(
                value.get("action_horizon", self.model.config.action_horizon)
            )
            if not self.config.execution_steps < action_horizon <= (
                self.model.config.action_horizon
            ):
                raise ValueError(
                    f"M2 task {task_id!r} action horizon is incompatible with "
                    "execution_steps/model maximum"
                )
            value["action_horizon"] = action_horizon
            mean = torch.as_tensor(
                value["action_mean"], device=self.device, dtype=self.dtype
            )
            std = torch.as_tensor(
                value["action_std"], device=self.device, dtype=self.dtype
            )
            if (
                tuple(mean.shape) != (action_dim,)
                or tuple(std.shape) != (action_dim,)
                or not bool(torch.isfinite(mean).all())
                or not bool(torch.isfinite(std).all())
                or not bool(std.gt(0.0).all())
            ):
                raise ValueError(
                    f"M2 task {task_id!r} action normalization is invalid"
                )
            self.action_normalization[task_id] = (mean, std)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.vision_encoder.parameters():
            parameter.requires_grad_(False)
        output_dim = int(getattr(self.vision_encoder, "output_dim", -1))
        if output_dim != self.model.config.visual_feature_dim:
            raise ValueError("M2 vision encoder width differs from the checkpoint")
        self._states: deque[np.ndarray] = deque(maxlen=self.model.config.history_steps)
        self._actions: deque[np.ndarray] = deque(maxlen=self.model.config.history_steps - 1)
        self._images: deque[np.ndarray] = deque(maxlen=self.model.config.visual_history_steps)
        self._image_frame_indices: deque[tuple[int, ...]] = deque(
            maxlen=self.model.config.visual_history_steps
        )
        self._chunk: Tensor | None = None
        self._chunk_cursor = 0
        self._executed_since_generation = 0
        self.last_diagnostics: dict[str, Any] = {}
        self.last_future: dict[str, np.ndarray] | None = None

    def reset(self) -> None:
        self._states.clear()
        self._actions.clear()
        self._images.clear()
        self._image_frame_indices.clear()
        self._chunk = None
        self._chunk_cursor = 0
        self._executed_since_generation = 0
        self.last_diagnostics = {}
        self.last_future = None

    @torch.inference_mode()
    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        started = self.clock()
        task = observation.get("task")
        if not isinstance(task, Mapping):
            raise KeyError("M2 policy requires task.id and task.text")
        task_id = str(task.get("id", ""))
        task_text = str(task.get("text", ""))
        if task_id not in self.runtime or not task_text.strip():
            raise ValueError(f"unknown/empty M2 task condition {task_id!r}")
        runtime = self.runtime[task_id]
        if tuple(map(str, runtime["camera_order"])) != self.config.camera_order:
            raise ValueError(
                f"M2 task {task_id!r} camera order differs from the environment"
            )
        state = np.asarray(observation.get("proprioception"), dtype=np.float32)
        state_dim = int(runtime["state_dim"])
        if state.shape != (state_dim,) or not np.isfinite(state).all():
            raise ValueError(f"M2 task {task_id!r} state must be finite [{state_dim}]")
        self._states.append(state.copy())
        image, indices, is_new = self._read_images(observation)
        if is_new:
            self._images.append(image)
            self._image_frame_indices.append(indices)
        replan = (
            self._chunk is None
            or self._chunk_cursor >= int(runtime["action_horizon"])
            or self._executed_since_generation >= self.config.execution_steps
        )
        warm_start_used = False
        if replan:
            batch, visual = self._context(task_id, task_text, runtime)
            context = m2_model_context(batch, visual)
            initial = self._warm_start(
                action_horizon=int(runtime["action_horizon"])
            )
            warm_start_used = initial is not None
            generated = self.model.generate_actions(
                context,
                initial_actions=initial,
                solver_steps=self.config.solver_steps,
                solver=self.config.solver,
                normalized_clip=self.config.normalized_action_clip,
            )
            self._chunk = generated[0]
            self._chunk_cursor = 0
            self._executed_since_generation = 0
            if self.config.future_path:
                flow_time = torch.ones(1, device=self.device, dtype=self.dtype)
                future = self.model(
                    **context,
                    action_inputs=generated,
                    flow_time=flow_time,
                    initial_actions=initial,
                    future_action_condition=generated,
                    include_future=True,
                )
                assert future.future_states is not None
                assert future.future_visual_latents is not None
                self.last_future = {
                    "states_normalized": future.future_states[0].float().cpu().numpy(),
                    "visual_latents": future.future_visual_latents[0].float().cpu().numpy(),
                }
        if self._chunk is None:
            raise RuntimeError("M2 failed to create an action chunk")
        normalized = self._chunk[self._chunk_cursor]
        action_dim = int(runtime["action_dim"])
        action_mean, action_std = self.action_normalization[task_id]
        canonical_task = (
            normalized[:action_dim] * action_std + action_mean
        ).clamp(-1.0, 1.0)
        raw = self.codecs[task_id].decode(canonical_task, clip=True)
        if not isinstance(raw, Tensor):
            raise TypeError("M2 action codec returned a non-tensor")
        raw_np = raw.float().cpu().numpy().astype(np.float32, copy=False)
        if raw_np.shape != (action_dim,) or not np.isfinite(raw_np).all():
            raise RuntimeError("M2 decoded action violates the task contract")
        self._actions.append(normalized.float().cpu().numpy().copy())
        self._chunk_cursor += 1
        self._executed_since_generation += 1
        latency_ms = (self.clock() - started) * 1000.0
        self.last_diagnostics = {
            "action_source": (
                "m2_block_causal_future_path"
                if self.config.future_path
                else "m2_block_causal_fast_path"
            ),
            "fallback_used": False,
            "direct_model_action": True,
            "task_id": task_id,
            "task_text": task_text,
            "task_index": int(runtime["task_index"]),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "action_horizon": int(runtime["action_horizon"]),
            "action_space": "per_task_zscore_canonical_unit_action",
            "warm_start_used": warm_start_used,
            "future_path": self.config.future_path,
            "new_visual_frame": is_new,
            "frame_indices": list(indices),
            "latency_ms": latency_ms,
            "consumed_observation_paths": [
                "task.id",
                "task.text",
                "proprioception",
                *[f"images.{name}" for name in self.config.camera_order],
                *[
                    f"image_frame_indices.{name}"
                    for name in self.config.camera_order
                ],
            ],
        }
        return raw_np

    def _context(
        self,
        task_id: str,
        task_text: str,
        runtime: Mapping[str, Any],
    ) -> tuple[dict[str, Tensor], Tensor]:
        config = self.model.config
        states = torch.zeros(
            (1, config.history_steps, config.max_state_dim),
            device=self.device,
            dtype=self.dtype,
        )
        state_valid = torch.zeros(
            (1, config.history_steps), device=self.device, dtype=torch.bool
        )
        state_dim = int(runtime["state_dim"])
        mean = np.asarray(runtime["state_mean"], dtype=np.float32)
        std = np.asarray(runtime["state_std"], dtype=np.float32)
        normalized_states = [(value - mean) / std for value in self._states]
        start = config.history_steps - len(normalized_states)
        states[0, start:, :state_dim] = torch.as_tensor(
            np.stack(normalized_states), device=self.device, dtype=self.dtype
        )
        state_valid[0, start:] = True
        past_actions = torch.zeros(
            (1, config.history_steps - 1, config.max_action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        past_valid = torch.zeros(
            (1, config.history_steps - 1), device=self.device, dtype=torch.bool
        )
        if self._actions:
            values = list(self._actions)[-(config.history_steps - 1) :]
            action_start = config.history_steps - 1 - len(values)
            past_actions[0, action_start:] = torch.as_tensor(
                np.stack(values), device=self.device, dtype=self.dtype
            )
            past_valid[0, action_start:] = True
        images = torch.zeros(
            (
                1,
                config.visual_history_steps,
                len(self.config.camera_order),
                3,
                self._images[-1].shape[-2],
                self._images[-1].shape[-1],
            ),
            device=self.device,
            dtype=torch.uint8,
        )
        image_values = list(self._images)
        image_start = config.visual_history_steps - len(image_values)
        images[0, image_start:] = torch.as_tensor(
            np.stack(image_values), device=self.device, dtype=torch.uint8
        )
        grid_tokens = config.visual_grid_height * config.visual_grid_width
        if grid_tokens == 1:
            task_encoded = encode_pooled_vision(
                self.vision_encoder,
                images,
            )
            encoded_shape = (
                1,
                config.visual_history_steps,
                config.max_cameras,
                config.visual_feature_dim,
            )
        else:
            method = getattr(self.vision_encoder, "forward_spatial_grid", None)
            if not callable(method):
                raise TypeError(
                    "M2 spatial visual tokens require "
                    "vision_encoder.forward_spatial_grid"
                )
            visual_output = method(
                images,
                grid_height=config.visual_grid_height,
                grid_width=config.visual_grid_width,
            )
            task_encoded = visual_output.spatial_tokens
            if task_encoded.shape[-2:] != (
                grid_tokens,
                config.visual_feature_dim,
            ):
                raise ValueError("M2 vision encoder returned an invalid spatial grid")
            encoded_shape = (
                1,
                config.visual_history_steps,
                config.max_cameras,
                grid_tokens,
                config.visual_feature_dim,
            )
        task_encoded = task_encoded.to(dtype=self.dtype)
        encoded = torch.zeros(
            encoded_shape,
            device=self.device,
            dtype=self.dtype,
        )
        camera_slots = torch.as_tensor(
            runtime["camera_slot_indices"],
            device=self.device,
            dtype=torch.long,
        )
        if (
            camera_slots.shape != (len(self.config.camera_order),)
            or int(camera_slots.min()) < 0
            or int(camera_slots.max()) >= config.max_cameras
            or camera_slots.unique().numel() != camera_slots.numel()
        ):
            raise ValueError("M2 runtime camera slots are invalid")
        encoded[:, :, camera_slots] = task_encoded
        image_valid = torch.zeros(
            (
                1,
                config.visual_history_steps,
                config.max_cameras,
            ),
            device=self.device,
            dtype=torch.bool,
        )
        image_valid[:, image_start:, camera_slots] = True
        camera_agent_index = torch.full(
            (1, config.max_cameras),
            config.max_agents,
            device=self.device,
            dtype=torch.long,
        )
        camera_agents = torch.as_tensor(
            runtime["camera_agent_indices"],
            device=self.device,
            dtype=torch.long,
        )
        if (
            camera_agents.shape != camera_slots.shape
            or int(camera_agents.min()) < 0
            or int(camera_agents.max()) > config.max_agents
        ):
            raise ValueError("M2 runtime camera-agent identities are invalid")
        camera_agent_index[:, camera_slots] = camera_agents
        state_dimension_mask = torch.zeros(
            (1, config.max_state_dim), device=self.device, dtype=torch.bool
        )
        state_dimension_mask[:, :state_dim] = True
        action_dim = int(runtime["action_dim"])
        action_dimension_mask = torch.zeros(
            (1, config.max_action_dim), device=self.device, dtype=torch.bool
        )
        action_dimension_mask[:, :action_dim] = True
        action_horizon_mask = torch.arange(
            config.action_horizon, device=self.device
        )[None].lt(int(runtime["action_horizon"]))
        batch = {
            "states": states,
            "state_valid_mask": state_valid,
            "state_dimension_mask": state_dimension_mask,
            "past_actions": past_actions,
            "past_action_valid_mask": past_valid,
            "action_dimension_mask": action_dimension_mask,
            "image_valid_mask": image_valid,
            "camera_agent_index": camera_agent_index,
            "action_horizon_mask": action_horizon_mask,
            "task_text_tokens": encode_task_text(
                task_text, max_tokens=config.max_text_tokens
            ).to(self.device)[None],
            "task_index": torch.tensor(
                [int(runtime["task_index"])], device=self.device, dtype=torch.long
            ),
            "embodiment_index": torch.tensor(
                [int(runtime["agent_count"]) - 1], device=self.device, dtype=torch.long
            ),
        }
        return batch, encoded

    def _warm_start(self, *, action_horizon: int) -> Tensor | None:
        if not self.config.warm_start or self._chunk is None:
            return None
        shift = self.config.execution_steps
        if not shift < action_horizon <= self._chunk.shape[0]:
            raise ValueError("task action horizon is incompatible with warm-start")
        warm = torch.zeros_like(self._chunk)
        warm[: action_horizon - shift] = self._chunk[
            shift:action_horizon
        ]
        warm[action_horizon - shift : action_horizon] = self._chunk[
            action_horizon - 1
        ]
        return warm.unsqueeze(0)

    def _read_images(
        self, observation: Mapping[str, Any]
    ) -> tuple[np.ndarray, tuple[int, ...], bool]:
        images = observation.get("images")
        frame_indices = observation.get("image_frame_indices")
        if not isinstance(images, Mapping) or not isinstance(frame_indices, Mapping):
            raise KeyError("M2 policy requires images and image_frame_indices mappings")
        values: list[np.ndarray] = []
        indices: list[int] = []
        for camera in self.config.camera_order:
            image = np.asarray(images.get(camera))
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f"M2 camera {camera!r} must be lossless uint8 HWC")
            index = frame_indices.get(camera)
            if not isinstance(index, (int, np.integer)) or int(index) < 0:
                raise ValueError(f"M2 camera {camera!r} frame index is invalid")
            values.append(np.moveaxis(image, -1, 0).copy())
            indices.append(int(index))
        identity = tuple(indices)
        is_new = not self._image_frame_indices or identity != self._image_frame_indices[-1]
        return np.stack(values), identity, is_new


__all__ = ["RoboFactoryM2Policy", "RoboFactoryM2PolicyConfig"]
