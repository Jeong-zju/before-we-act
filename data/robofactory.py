"""Streaming adapter from RoboFactory/ManiSkill trajectories to WAM records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence

import h5py
import numpy as np

from data.exporters.base import ExportObserver, TrajectoryExporter
from data.trajectory import FieldSpec, TrajectorySchema
from envs.runtime import RolloutSummary, SimulationTransition

ROBOFACTORY_SCHEMA_VERSION = "robofactory.mani_skill/1.0"
ROBOFACTORY_M1_SCHEMA_VERSION = "wam.robofactory.multimodal/1.0"
ROBOFACTORY_M1_PROFILE = "robofactory_m1"
COMMAND_ECHO_ACTION_SOURCE = "command_echo"
_EPISODE_PATTERN = re.compile(r"^traj_(\d+)$")
_FLAT_SINGLE_AGENT_NAME = "panda-0"


@dataclass(frozen=True)
class RoboFactoryAgentLayout:
    """One source agent and the target-safe alias used for exported fields."""

    source_name: str
    target_name: str
    action_shape: tuple[int, ...]
    qpos_shape: tuple[int, ...]
    qvel_shape: tuple[int, ...]


@dataclass(frozen=True)
class RoboFactoryCameraLayout:
    """One RGB stream and its available calibration matrices."""

    source_name: str
    target_name: str
    image_shape: tuple[int, ...]
    calibration_fields: tuple[str, ...]


@dataclass(frozen=True)
class RoboFactoryLayout:
    """Dataset-wide feature contract inferred from the first trajectory."""

    agents: tuple[RoboFactoryAgentLayout, ...]
    cameras: tuple[RoboFactoryCameraLayout, ...]
    has_rewards: bool
    has_success: bool
    has_failure: bool

    @property
    def state_size(self) -> int:
        return sum(
            int(np.prod(agent.qpos_shape)) + int(np.prod(agent.qvel_shape))
            for agent in self.agents
        )

    @property
    def action_size(self) -> int:
        return sum(int(np.prod(agent.action_shape)) for agent in self.agents)


@dataclass(frozen=True)
class RoboFactoryEpisode:
    """Sidecar metadata joined to one ``traj_<id>`` HDF5 group."""

    source_id: int
    source_key: str
    seed: int | None
    success: bool | None
    metadata: Mapping[str, Any]


class RoboFactoryDataset:
    """Read RoboFactory HDF5 lazily and emit aligned format-neutral transitions.

    RoboFactory follows the ManiSkill convention: actions and terminal labels
    have length ``T`` while observations and images have length ``T + 1``.
    This adapter never loads a complete episode into memory.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        metadata_path: str | Path | None = None,
    ) -> None:
        self.path = Path(path).expanduser()
        if not self.path.is_file():
            raise FileNotFoundError(f"RoboFactory HDF5 file does not exist: {self.path}")
        self.metadata_path = (
            Path(metadata_path).expanduser()
            if metadata_path is not None
            else self.path.with_suffix(".json")
        )
        self.sidecar = self._load_sidecar()
        self.env_info = _mapping(self.sidecar.get("env_info", {}), "env_info")
        self.env_id = str(self.env_info.get("env_id", self.path.stem))
        self._file = h5py.File(self.path, "r")
        try:
            self.episodes = self._load_episodes()
            if not self.episodes:
                raise ValueError("RoboFactory dataset contains no traj_<id> groups")
            first_group = self._file[self.episodes[0].source_key]
            self._flat_single_agent_layout = self._uses_flat_single_agent_layout(
                first_group
            )
            self.layout = self._infer_layout(first_group)
            self._validate_episode(
                self._file[self.episodes[0].source_key], self.episodes[0]
            )
        except BaseException:
            self._file.close()
            raise

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "RoboFactoryDataset":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def build_schema(
        self,
        *,
        profile: str = "robofactory",
        cameras: Sequence[str] | None = None,
        include_images: bool = True,
        include_calibration: bool = True,
        include_agent_fields: bool = True,
    ) -> TrajectorySchema:
        """Build the legacy schema or the explicit M1 scratch data contract."""

        normalized_profile = profile.lower().replace("-", "_")
        selected_cameras = self._select_cameras(cameras)
        if normalized_profile in {"m1_scratch", ROBOFACTORY_M1_PROFILE}:
            return self._build_m1_schema(
                cameras=selected_cameras,
                include_images=include_images,
                include_calibration=include_calibration,
                include_agent_fields=include_agent_fields,
            )
        if normalized_profile != "robofactory":
            raise ValueError(
                f"unknown RoboFactory schema profile {profile!r}; expected "
                "'robofactory' or 'm1-scratch'"
            )

        fields: list[FieldSpec] = [
            FieldSpec("timestamp", "timestamp", "float64"),
            FieldSpec("frame_index", "frame_index", "int64"),
            FieldSpec("episode_index", "episode_index", "int64"),
            FieldSpec("task", "task"),
            FieldSpec("observation.state", "observation.proprioception", "float32"),
            FieldSpec("action", "action", "float32"),
            FieldSpec(
                "next_observation.state",
                "next_observation.proprioception",
                "float32",
            ),
        ]
        if self.layout.has_rewards:
            fields.append(FieldSpec("next.reward", "reward", "float32"))
        fields.extend(
            (
                FieldSpec("next.terminated", "terminated", "bool"),
                FieldSpec("next.truncated", "truncated", "bool"),
                FieldSpec("next.done", "done", "bool"),
            )
        )
        if self.layout.has_success:
            fields.append(FieldSpec("next.success", "info.success", "bool"))
        if self.layout.has_failure:
            fields.append(FieldSpec("next.failure", "info.failure", "bool"))

        if include_agent_fields:
            for agent in self.layout.agents:
                prefix = f"observation.agents.{agent.target_name}"
                next_prefix = f"next_observation.agents.{agent.target_name}"
                fields.extend(
                    (
                        FieldSpec(
                            f"{prefix}.qpos",
                            f"observation.agents.{agent.target_name}.qpos",
                            "float32",
                        ),
                        FieldSpec(
                            f"{prefix}.qvel",
                            f"observation.agents.{agent.target_name}.qvel",
                            "float32",
                        ),
                        FieldSpec(
                            f"agents.{agent.target_name}.action",
                            f"info.agent_actions.{agent.target_name}",
                            "float32",
                        ),
                        FieldSpec(
                            f"{next_prefix}.qpos",
                            f"next_observation.agents.{agent.target_name}.qpos",
                            "float32",
                        ),
                        FieldSpec(
                            f"{next_prefix}.qvel",
                            f"next_observation.agents.{agent.target_name}.qvel",
                            "float32",
                        ),
                    )
                )

        if include_images:
            fields.extend(
                FieldSpec(
                    f"observation.images.{camera.target_name}",
                    f"images.{camera.target_name}",
                    "uint8",
                )
                for camera in selected_cameras
            )
        if include_calibration:
            for camera in selected_cameras:
                for calibration in camera.calibration_fields:
                    fields.append(
                        FieldSpec(
                            "observation.camera_calibration."
                            f"{camera.target_name}.{calibration}",
                            "observation.camera_calibration."
                            f"{camera.target_name}.{calibration}",
                            "float32",
                        )
                    )
        return TrajectorySchema(
            profile="robofactory",
            fields=tuple(fields),
            version=ROBOFACTORY_SCHEMA_VERSION,
        )

    def _build_m1_schema(
        self,
        *,
        cameras: Sequence[RoboFactoryCameraLayout],
        include_images: bool,
        include_calibration: bool,
        include_agent_fields: bool,
    ) -> TrajectorySchema:
        fields: list[FieldSpec] = [
            FieldSpec("timestamp", "timestamp", "float64"),
            FieldSpec("frame_index", "frame_index", "int64"),
            FieldSpec("episode_index", "episode_index", "int64"),
            FieldSpec("seed", "metadata.seed", "int64"),
            FieldSpec("task.id", "metadata.task_id"),
            FieldSpec("task.text", "task"),
            FieldSpec("observation.state", "observation.proprioception", "float32"),
        ]
        if include_images:
            for camera in cameras:
                name = camera.target_name
                fields.extend(
                    (
                        FieldSpec(
                            f"observation.images.{name}", f"images.{name}", "uint8"
                        ),
                        FieldSpec(
                            f"observation.image_timestamp.{name}",
                            f"image_timestamps.{name}",
                            "float64",
                        ),
                        FieldSpec(
                            f"observation.image_state_timestamp.{name}",
                            f"image_state_timestamps.{name}",
                            "float64",
                        ),
                        FieldSpec(
                            f"observation.image_frame_index.{name}",
                            f"image_frame_indices.{name}",
                            "int64",
                        ),
                    )
                )
        fields.extend(
            (
                FieldSpec("action.commanded", "action", "float32"),
                FieldSpec("action.executed", "info.executed_action", "float32"),
                FieldSpec(
                    "next_observation.state",
                    "next_observation.proprioception",
                    "float32",
                ),
            )
        )
        if include_images:
            for camera in cameras:
                name = camera.target_name
                fields.extend(
                    (
                        FieldSpec(
                            f"next_observation.images.{name}",
                            f"next_images.{name}",
                            "uint8",
                        ),
                        FieldSpec(
                            f"next_observation.image_timestamp.{name}",
                            f"next_image_timestamps.{name}",
                            "float64",
                        ),
                        FieldSpec(
                            f"next_observation.image_state_timestamp.{name}",
                            f"next_image_state_timestamps.{name}",
                            "float64",
                        ),
                        FieldSpec(
                            f"next_observation.image_frame_index.{name}",
                            f"next_image_frame_indices.{name}",
                            "int64",
                        ),
                    )
                )
        if self.layout.has_rewards:
            fields.append(FieldSpec("reward", "reward", "float32"))
        fields.extend(
            (
                FieldSpec("terminated", "terminated", "bool"),
                FieldSpec("truncated", "truncated", "bool"),
                FieldSpec("done", "done", "bool"),
            )
        )
        if self.layout.has_success:
            fields.append(FieldSpec("success", "info.success", "bool"))
        if self.layout.has_failure:
            fields.append(FieldSpec("failure", "info.failure", "bool"))

        if include_agent_fields:
            for agent in self.layout.agents:
                name = agent.target_name
                fields.extend(
                    (
                        FieldSpec(
                            f"observation.agents.{name}.qpos",
                            f"observation.agents.{name}.qpos",
                            "float32",
                        ),
                        FieldSpec(
                            f"observation.agents.{name}.qvel",
                            f"observation.agents.{name}.qvel",
                            "float32",
                        ),
                        FieldSpec(
                            f"action.agents.{name}.commanded",
                            f"info.agent_actions.{name}",
                            "float32",
                        ),
                        FieldSpec(
                            f"action.agents.{name}.executed",
                            f"info.agent_executed_actions.{name}",
                            "float32",
                        ),
                        FieldSpec(
                            f"next_observation.agents.{name}.qpos",
                            f"next_observation.agents.{name}.qpos",
                            "float32",
                        ),
                        FieldSpec(
                            f"next_observation.agents.{name}.qvel",
                            f"next_observation.agents.{name}.qvel",
                            "float32",
                        ),
                    )
                )

        if include_calibration:
            for camera in cameras:
                name = camera.target_name
                for calibration in camera.calibration_fields:
                    fields.extend(
                        (
                            FieldSpec(
                                f"observation.camera_calibration.{name}.{calibration}",
                                f"observation.camera_calibration.{name}.{calibration}",
                                "float32",
                            ),
                            FieldSpec(
                                "next_observation.camera_calibration."
                                f"{name}.{calibration}",
                                "next_observation.camera_calibration."
                                f"{name}.{calibration}",
                                "float32",
                            ),
                        )
                    )
        return TrajectorySchema(
            profile=ROBOFACTORY_M1_PROFILE,
            fields=tuple(fields),
            version=ROBOFACTORY_M1_SCHEMA_VERSION,
        )

    def convert(
        self,
        exporters: Sequence[TrajectoryExporter],
        *,
        fps: float,
        schema: TrajectorySchema,
        task: str | None = None,
        task_id: str | None = None,
        executed_action_source: str | None = None,
        max_episodes: int | None = None,
        success_only: bool = False,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
        episode_indices: Sequence[int] | None = None,
        output_indices: Sequence[int] | None = None,
        compute_source_hashes: bool = True,
    ) -> dict[str, Any]:
        """Stream selected episodes through the configured output backends."""

        if fps <= 0 or not np.isfinite(fps):
            raise ValueError("fps must be finite and positive")
        if max_episodes is not None and max_episodes <= 0:
            raise ValueError("max_episodes must be positive when provided")
        selected = self._selected_episodes(
            max_episodes=max_episodes, success_only=success_only
        )
        if episode_indices is None:
            normalized_episode_indices = tuple(range(len(selected)))
        else:
            normalized_episode_indices = tuple(int(index) for index in episode_indices)
            if len(set(normalized_episode_indices)) != len(
                normalized_episode_indices
            ):
                raise ValueError("episode_indices cannot contain duplicates")
            if any(
                index < 0 or index >= len(selected)
                for index in normalized_episode_indices
            ):
                raise IndexError("episode_indices contains an out-of-range index")
            selected = tuple(selected[index] for index in normalized_episode_indices)
        if output_indices is None:
            normalized_output_indices = normalized_episode_indices
        else:
            normalized_output_indices = tuple(int(index) for index in output_indices)
        if len(normalized_output_indices) != len(selected):
            raise ValueError("output_indices must match the selected episode count")
        if len(set(normalized_output_indices)) != len(normalized_output_indices):
            raise ValueError("output_indices cannot contain duplicates")
        if any(index < 0 for index in normalized_output_indices):
            raise ValueError("output_indices cannot contain negative values")
        if not selected:
            raise ValueError("episode_indices selected no episodes")

        instruction = task.strip() if task is not None else _task_from_env_id(self.env_id)
        if not instruction:
            raise ValueError("task instruction cannot be empty")
        normalized_task_id = (
            task_id.strip() if task_id is not None else _task_id_from_env_id(self.env_id)
        )
        if any(field.source == "metadata.task_id" for field in schema.fields):
            if not normalized_task_id:
                raise ValueError("task_id cannot be empty for the M1 scratch schema")
        if executed_action_source is not None:
            executed_action_source = executed_action_source.lower().replace("-", "_")
        requires_executed_action = any(
            field.source == "info.executed_action" for field in schema.fields
        )
        if requires_executed_action and (
            executed_action_source != COMMAND_ECHO_ACTION_SOURCE
        ):
            raise ValueError(
                "the M1 scratch schema requires executed_action_source='command_echo'; "
                "this records an explicit command echo, not independent actuator feedback"
            )
        if any(field.source == "metadata.seed" for field in schema.fields):
            missing_seed_ids = [
                episode.source_id for episode in selected if episode.seed is None
            ]
            if missing_seed_ids:
                raise ValueError(
                    "the M1 scratch schema requires an episode seed; missing for source "
                    f"episodes {missing_seed_ids[:10]}"
                )

        observer = ExportObserver(exporters, fps=fps)
        summaries: list[dict[str, Any]] = []
        started = time.monotonic()
        exported_camera_names = _exported_camera_names(schema)
        exported_cameras = tuple(
            camera
            for name in exported_camera_names
            for camera in self.layout.cameras
            if camera.target_name == name
        )
        include_images = bool(exported_cameras)
        include_calibration = any(
            field.source.startswith(
                ("observation.camera_calibration.", "next_observation.camera_calibration.")
            )
            for field in schema.fields
        )
        try:
            for selection_index, (output_index, episode) in enumerate(
                zip(normalized_output_indices, selected, strict=True)
            ):
                group = self._file[episode.source_key]
                steps = self._validate_episode(group, episode)
                initial_observation = self._observation(
                    group, 0, include_calibration=include_calibration
                )
                episode_metadata = self._episode_metadata(
                    episode,
                    schema=schema,
                    fps=fps,
                    task_id=normalized_task_id,
                    exported_cameras=exported_cameras,
                    executed_action_source=executed_action_source,
                )
                observer.on_episode_start(
                    episode_index=output_index,
                    seed=episode.seed,
                    observation=initial_observation,
                    info={},
                    task=instruction,
                )
                total_reward = 0.0
                final_info: dict[str, Any] = {}
                episode_started = time.monotonic()
                observation = initial_observation
                for frame_index in range(steps):
                    reward = self._scalar(group, "rewards", frame_index, 0.0)
                    terminated = bool(group["terminated"][frame_index])
                    truncated = bool(group["truncated"][frame_index])
                    commanded_action = self._action(group, frame_index)
                    info = self._transition_info(
                        group,
                        frame_index,
                        commanded_action=commanded_action,
                        executed_action_source=executed_action_source,
                    )
                    next_observation = self._observation(
                        group,
                        frame_index + 1,
                        include_calibration=include_calibration,
                    )
                    transition = SimulationTransition(
                        episode_index=output_index,
                        frame_index=frame_index,
                        timestamp=frame_index / float(fps),
                        observation=observation,
                        action=commanded_action,
                        next_observation=next_observation,
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                        task=instruction,
                        images=(
                            self._images(group, frame_index, exported_cameras)
                            if include_images
                            else {}
                        ),
                        next_images=(
                            self._images(group, frame_index + 1, exported_cameras)
                            if include_images
                            else {}
                        ),
                        image_timestamps={
                            camera.target_name: frame_index / float(fps)
                            for camera in exported_cameras
                        },
                        next_image_timestamps={
                            camera.target_name: (frame_index + 1) / float(fps)
                            for camera in exported_cameras
                        },
                        image_state_timestamps={
                            camera.target_name: frame_index / float(fps)
                            for camera in exported_cameras
                        },
                        next_image_state_timestamps={
                            camera.target_name: (frame_index + 1) / float(fps)
                            for camera in exported_cameras
                        },
                        image_frame_indices={
                            camera.target_name: frame_index
                            for camera in exported_cameras
                        },
                        next_image_frame_indices={
                            camera.target_name: frame_index + 1
                            for camera in exported_cameras
                        },
                        metadata=episode_metadata,
                    )
                    observer.on_transition(transition)
                    if progress is not None:
                        progress(
                            {
                                "source_episode": episode.source_id,
                                "episode": selection_index + 1,
                                "episodes": len(selected),
                                "frame": frame_index + 1,
                                "frames": steps,
                            }
                        )
                    total_reward += reward
                    final_info = info
                    observation = next_observation
                summary = RolloutSummary(
                    episode_index=output_index,
                    seed=episode.seed,
                    steps=steps,
                    total_reward=(
                        total_reward if self.layout.has_rewards else float("nan")
                    ),
                    terminated=bool(group["terminated"][-1]),
                    truncated=bool(group["truncated"][-1]),
                    elapsed_wall_seconds=time.monotonic() - episode_started,
                    final_info=final_info,
                )
                observer.on_episode_end(summary)
                summaries.append(
                    {
                        "episode_index": output_index,
                        "source_episode_id": episode.source_id,
                        "seed": episode.seed,
                        "steps": steps,
                        "success": episode.success,
                        "terminated": summary.terminated,
                        "truncated": summary.truncated,
                    }
                )
        finally:
            observer.close()

        return {
            "format_version": (
                "robofactory.conversion_manifest/2.0"
                if schema.profile == ROBOFACTORY_M1_PROFILE
                else "robofactory.conversion_manifest/1.0"
            ),
            "schema_profile": schema.profile,
            "schema_version": schema.version,
            "source": {
                "hdf5": str(self.path.resolve()),
                "metadata_json": (
                    str(self.metadata_path.resolve())
                    if self.metadata_path.is_file()
                    else None
                ),
                "size_bytes": self.path.stat().st_size,
                "hdf5_sha256": (
                    _sha256_file(self.path) if compute_source_hashes else None
                ),
                "metadata_json_sha256": (
                    _sha256_file(self.metadata_path)
                    if compute_source_hashes and self.metadata_path.is_file()
                    else None
                ),
                "env_id": self.env_id,
                "metadata": {
                    str(key): value
                    for key, value in self.sidecar.items()
                    if key != "episodes"
                },
            },
            "task": instruction,
            "task_id": normalized_task_id,
            "fps": float(fps),
            "transition_semantics": "observation[t], action[t], observation[t+1]",
            "data_semantics": self._data_semantics(
                schema=schema,
                fps=fps,
                exported_cameras=exported_cameras,
                executed_action_source=executed_action_source,
            ),
            "field_mapping": self.field_mapping(schema=schema),
            "layout": {
                "state_size": self.layout.state_size,
                "action_size": self.layout.action_size,
                "agents": [asdict(agent) for agent in self.layout.agents],
                "cameras": [asdict(camera) for camera in self.layout.cameras],
                "exported_cameras": [
                    asdict(camera) for camera in exported_cameras
                ],
                "has_rewards": self.layout.has_rewards,
                "has_success": self.layout.has_success,
                "has_failure": self.layout.has_failure,
            },
            "fields": [asdict(field) for field in schema.fields],
            "filters": {
                "success_only": bool(success_only),
                "max_episodes": max_episodes,
            },
            "episodes": summaries,
            "elapsed_wall_seconds": time.monotonic() - started,
        }

    def conversion_totals(
        self,
        *,
        max_episodes: int | None = None,
        success_only: bool = False,
    ) -> tuple[int, int]:
        """Return selected episode and transition counts for progress displays."""

        if max_episodes is not None and max_episodes <= 0:
            raise ValueError("max_episodes must be positive when provided")
        selected = self._selected_episodes(
            max_episodes=max_episodes, success_only=success_only
        )
        transitions = sum(
            self._validate_episode(self._file[episode.source_key], episode)
            for episode in selected
        )
        return len(selected), transitions

    def field_mapping(
        self, *, schema: TrajectorySchema | None = None
    ) -> dict[str, Any]:
        """Describe every semantic rename and centralized concatenation order."""

        action_target = _field_name_for_source(schema, "action") or "action"
        exported_camera_names = (
            set(_exported_camera_names(schema))
            if schema is not None
            else {camera.target_name for camera in self.layout.cameras}
        )
        state_slices: list[dict[str, Any]] = []
        action_slices: list[dict[str, Any]] = []
        state_offset = 0
        action_offset = 0
        for agent in self.layout.agents:
            for component, shape in (
                ("qpos", agent.qpos_shape),
                ("qvel", agent.qvel_shape),
            ):
                size = int(np.prod(shape))
                source = (
                    f"obs/agent/{component}"
                    if self._flat_single_agent_layout
                    else f"obs/agent/{agent.source_name}/{component}"
                )
                state_slices.append(
                    {
                        "source": source,
                        "target": "observation.state",
                        "slice": [state_offset, state_offset + size],
                    }
                )
                state_offset += size
            size = int(np.prod(agent.action_shape))
            action_slices.append(
                {
                    "source": (
                        "actions"
                        if self._flat_single_agent_layout
                        else f"actions/{agent.source_name}"
                    ),
                    "target": action_target,
                    "slice": [action_offset, action_offset + size],
                }
            )
            action_offset += size
        return {
            "centralized_state": state_slices,
            "centralized_action": action_slices,
            "agent_names": {
                agent.source_name: agent.target_name for agent in self.layout.agents
            },
            "camera_names": {
                camera.source_name: f"observation.images.{camera.target_name}"
                for camera in self.layout.cameras
                if camera.target_name in exported_camera_names
            },
            "command_echo": (
                {
                    "source": action_target,
                    "target": "action.executed",
                    "equivalence": "exact_copy",
                    "independent_actuator_feedback": False,
                }
                if schema is not None
                and any(field.name == "action.executed" for field in schema.fields)
                else None
            ),
            "labels": {
                "rewards": (
                    _field_name_for_source(schema, "reward")
                    if self.layout.has_rewards
                    else None
                ),
                "terminated": _field_name_for_source(schema, "terminated"),
                "truncated": _field_name_for_source(schema, "truncated"),
                "terminated OR truncated": _field_name_for_source(schema, "done"),
                "success": (
                    _field_name_for_source(schema, "info.success")
                    if self.layout.has_success
                    else None
                ),
                "fail": (
                    _field_name_for_source(schema, "info.failure")
                    if self.layout.has_failure
                    else None
                ),
            },
        }

    def _data_semantics(
        self,
        *,
        schema: TrajectorySchema,
        fps: float,
        exported_cameras: Sequence[RoboFactoryCameraLayout],
        executed_action_source: str | None,
    ) -> dict[str, Any]:
        commanded_field = _field_name_for_source(schema, "action") or "action"
        executed_field = _field_name_for_source(schema, "info.executed_action")
        return {
            "state": {
                "field": "observation.state",
                "agent_order": [agent.source_name for agent in self.layout.agents],
                "per_agent_component_order": ["qpos", "qvel"],
                "ordering_rule": "natural_agent_order_then_qpos_then_qvel",
            },
            "action": {
                "commanded_field": commanded_field,
                "executed_field": executed_field,
                "history_field": commanded_field,
                "history_semantics": "past_controller_commands",
                "agent_order": [agent.source_name for agent in self.layout.agents],
                "control_mode": self._control_mode(),
                "executed_action_source": executed_action_source,
                "executed_action_equivalence": (
                    "exact_copy_of_commanded"
                    if executed_action_source == COMMAND_ECHO_ACTION_SOURCE
                    else None
                ),
                "independent_actuator_feedback_available": False,
            },
            "timing": {
                "control_hz": float(fps),
                "image_hz": float(fps),
                "sampling": "one_rgb_frame_per_control_step",
                "current": "state[t], commanded_action[t], rgb[t]",
                "next": "state[t+1], rgb[t+1]",
            },
            "vision": {
                "camera_order": [camera.target_name for camera in exported_cameras],
                "source_camera_order": [
                    camera.source_name for camera in exported_cameras
                ],
            },
        }

    def _select_cameras(
        self, camera_names: Sequence[str] | None
    ) -> tuple[RoboFactoryCameraLayout, ...]:
        if camera_names is None:
            return self.layout.cameras
        if not camera_names:
            raise ValueError("camera selection cannot be empty")
        selected: list[RoboFactoryCameraLayout] = []
        for raw_name in camera_names:
            name = str(raw_name).strip()
            matches = [
                camera
                for camera in self.layout.cameras
                if name in {camera.source_name, camera.target_name}
            ]
            if not matches:
                available = sorted(
                    {
                        item
                        for camera in self.layout.cameras
                        for item in (camera.source_name, camera.target_name)
                    }
                )
                raise ValueError(
                    f"unknown RoboFactory camera {name!r}; available: {available}"
                )
            camera = matches[0]
            if camera in selected:
                raise ValueError(f"duplicate RoboFactory camera selection {name!r}")
            selected.append(camera)
        return tuple(selected)

    def _control_mode(self) -> str | None:
        env_kwargs = self.env_info.get("env_kwargs", {})
        if not isinstance(env_kwargs, Mapping):
            return None
        value = env_kwargs.get("control_mode")
        return None if value is None else str(value)

    def _selected_episodes(
        self,
        *,
        max_episodes: int | None,
        success_only: bool,
    ) -> tuple[RoboFactoryEpisode, ...]:
        selected = tuple(
            episode
            for episode in self.episodes
            if not success_only or episode.success is True
        )
        if max_episodes is not None:
            selected = selected[:max_episodes]
        if not selected:
            raise ValueError("no RoboFactory episodes matched the requested filters")
        return selected

    def _load_sidecar(self) -> dict[str, Any]:
        if not self.metadata_path.is_file():
            return {}
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("RoboFactory metadata JSON must contain an object")
        return payload

    def _load_episodes(self) -> tuple[RoboFactoryEpisode, ...]:
        groups: dict[int, str] = {}
        for key in self._file.keys():
            match = _EPISODE_PATTERN.fullmatch(key)
            if match is not None and isinstance(self._file[key], h5py.Group):
                groups[int(match.group(1))] = key
        metadata_entries = self.sidecar.get("episodes", [])
        if metadata_entries:
            if not isinstance(metadata_entries, list):
                raise TypeError("RoboFactory metadata 'episodes' must be a list")
            episodes: list[RoboFactoryEpisode] = []
            seen: set[int] = set()
            for raw in metadata_entries:
                entry = _mapping(raw, "episode metadata")
                source_id = int(entry["episode_id"])
                if source_id in seen:
                    raise ValueError(f"duplicate episode_id {source_id} in metadata")
                if source_id not in groups:
                    raise KeyError(f"metadata references missing traj_{source_id} group")
                seen.add(source_id)
                episodes.append(
                    RoboFactoryEpisode(
                        source_id=source_id,
                        source_key=groups[source_id],
                        seed=_optional_int(
                            entry.get(
                                "episode_seed",
                                _mapping(entry.get("reset_kwargs", {}), "reset_kwargs").get(
                                    "seed"
                                ),
                            )
                        ),
                        success=(
                            _optional_bool(entry.get("success"))
                            if "success" in entry
                            else self._episode_success(groups[source_id])
                        ),
                        metadata=dict(entry),
                    )
                )
            missing = sorted(set(groups) - seen)
            if missing:
                raise ValueError(
                    "HDF5 trajectory groups missing from metadata: "
                    + ", ".join(str(item) for item in missing[:10])
                )
            return tuple(episodes)
        return tuple(
            RoboFactoryEpisode(
                source_id=source_id,
                source_key=groups[source_id],
                seed=None,
                success=self._episode_success(groups[source_id]),
                metadata={},
            )
            for source_id in sorted(groups)
        )

    def _infer_layout(self, group: h5py.Group) -> RoboFactoryLayout:
        actions = group.get("actions")
        observation_agents = _group(group, "obs/agent")
        if isinstance(actions, h5py.Dataset):
            source_agents = (_FLAT_SINGLE_AGENT_NAME,)
            if not all(
                isinstance(observation_agents.get(name), h5py.Dataset)
                for name in ("qpos", "qvel")
            ):
                raise ValueError(
                    "flat single-agent actions require obs/agent/qpos and "
                    "obs/agent/qvel datasets"
                )
        elif isinstance(actions, h5py.Group):
            source_agents = tuple(sorted(actions.keys(), key=_natural_key))
            if not source_agents:
                raise ValueError("RoboFactory trajectory has no agents under actions/")
            if set(source_agents) != set(observation_agents.keys()):
                raise ValueError(
                    "actions/ and obs/agent/ must contain the same agent names"
                )
        else:
            raise KeyError("required HDF5 dataset or group 'actions' is missing")
        agent_aliases = _unique_aliases(source_agents, _identifier)
        agents: list[RoboFactoryAgentLayout] = []
        for source_name in source_agents:
            if isinstance(actions, h5py.Dataset):
                action = actions
                observation_agent = observation_agents
            else:
                action = _dataset(actions, source_name)
                observation_agent = _group(observation_agents, source_name)
            qpos = _dataset(observation_agent, "qpos")
            qvel = _dataset(observation_agent, "qvel")
            agents.append(
                RoboFactoryAgentLayout(
                    source_name=source_name,
                    target_name=agent_aliases[source_name],
                    action_shape=tuple(action.shape[1:]),
                    qpos_shape=tuple(qpos.shape[1:]),
                    qvel_shape=tuple(qvel.shape[1:]),
                )
            )

        sensor_data = group.get("obs/sensor_data")
        sensor_params = group.get("obs/sensor_param")
        source_cameras: list[str] = []
        if isinstance(sensor_data, h5py.Group):
            for name in sorted(sensor_data.keys(), key=_natural_key):
                camera_group = sensor_data[name]
                if isinstance(camera_group, h5py.Group) and "rgb" in camera_group:
                    source_cameras.append(name)
        camera_aliases = _unique_aliases(source_cameras, _camera_alias)
        cameras: list[RoboFactoryCameraLayout] = []
        for source_name in source_cameras:
            image = _dataset(sensor_data[source_name], "rgb")
            calibration_fields: tuple[str, ...] = ()
            if isinstance(sensor_params, h5py.Group) and source_name in sensor_params:
                parameters = sensor_params[source_name]
                if isinstance(parameters, h5py.Group):
                    calibration_fields = tuple(
                        name
                        for name in ("intrinsic_cv", "extrinsic_cv", "cam2world_gl")
                        if name in parameters
                        and isinstance(parameters[name], h5py.Dataset)
                    )
            cameras.append(
                RoboFactoryCameraLayout(
                    source_name=source_name,
                    target_name=camera_aliases[source_name],
                    image_shape=tuple(image.shape[1:]),
                    calibration_fields=calibration_fields,
                )
            )
        return RoboFactoryLayout(
            agents=tuple(agents),
            cameras=tuple(cameras),
            has_rewards="rewards" in group,
            has_success="success" in group,
            has_failure="fail" in group,
        )

    def _validate_episode(
        self, group: h5py.Group, episode: RoboFactoryEpisode
    ) -> int:
        prefix = episode.source_key
        if (
            self._uses_flat_single_agent_layout(group)
            != self._flat_single_agent_layout
        ):
            raise ValueError(
                f"{prefix}: single-agent action representation changed within dataset"
            )
        steps: int | None = None
        for agent in self.layout.agents:
            action = self._agent_action_dataset(group, agent)
            if tuple(action.shape[1:]) != agent.action_shape:
                raise ValueError(f"{prefix}: action shape changed for {agent.source_name}")
            steps = int(action.shape[0]) if steps is None else steps
            if action.shape[0] != steps:
                raise ValueError(f"{prefix}: agent action lengths do not match")
        if steps is None or steps <= 0:
            raise ValueError(f"{prefix}: empty trajectories are not supported")

        observation_agents = _group(group, "obs/agent")
        if self._uses_flat_single_agent_layout(group):
            if len(self.layout.agents) != 1:
                raise ValueError(
                    f"{prefix}: flat actions are only valid for a single agent"
                )
        elif set(observation_agents.keys()) != {
            agent.source_name for agent in self.layout.agents
        }:
            raise ValueError(f"{prefix}: agent names differ from the dataset layout")
        for agent in self.layout.agents:
            observation_agent = self._agent_observation_group(group, agent)
            for name, shape in (("qpos", agent.qpos_shape), ("qvel", agent.qvel_shape)):
                dataset = _dataset(observation_agent, name)
                if dataset.shape[0] != steps + 1:
                    source_path = (
                        f"obs/agent/{name}"
                        if self._flat_single_agent_layout
                        else f"obs/agent/{agent.source_name}/{name}"
                    )
                    raise ValueError(
                        f"{prefix}: {source_path} must "
                        f"have T+1={steps + 1} rows, got {dataset.shape[0]}"
                    )
                if tuple(dataset.shape[1:]) != shape:
                    raise ValueError(f"{prefix}: {agent.source_name}/{name} shape changed")

        for label in ("terminated", "truncated"):
            dataset = _dataset(group, label)
            if dataset.shape != (steps,):
                raise ValueError(
                    f"{prefix}: {label} must have shape ({steps},), got {dataset.shape}"
                )
        for label, expected in (
            ("rewards", self.layout.has_rewards),
            ("success", self.layout.has_success),
            ("fail", self.layout.has_failure),
        ):
            if (label in group) != expected:
                raise ValueError(f"{prefix}: optional label {label!r} is inconsistent")
            if expected and _dataset(group, label).shape != (steps,):
                raise ValueError(f"{prefix}: {label} must have shape ({steps},)")

        sensor_data = group.get("obs/sensor_data")
        sensor_params = group.get("obs/sensor_param")
        for camera in self.layout.cameras:
            if not isinstance(sensor_data, h5py.Group):
                raise ValueError(f"{prefix}: obs/sensor_data is missing")
            image = _dataset(_group(sensor_data, camera.source_name), "rgb")
            if image.shape[0] != steps + 1 or tuple(image.shape[1:]) != camera.image_shape:
                raise ValueError(
                    f"{prefix}: RGB shape changed for camera {camera.source_name}"
                )
            if len(camera.image_shape) != 3 or camera.image_shape[-1] not in (1, 3, 4):
                raise ValueError(
                    f"{prefix}: camera {camera.source_name} RGB must be HWC"
                )
            for name in camera.calibration_fields:
                if not isinstance(sensor_params, h5py.Group):
                    raise ValueError(f"{prefix}: obs/sensor_param is missing")
                calibration = _dataset(
                    _group(sensor_params, camera.source_name), name
                )
                if calibration.shape[0] != steps + 1:
                    raise ValueError(
                        f"{prefix}: calibration {camera.source_name}/{name} "
                        f"must have T+1 rows"
                    )
        return steps

    def _observation(
        self,
        group: h5py.Group,
        index: int,
        *,
        include_calibration: bool,
    ) -> dict[str, Any]:
        agents: dict[str, dict[str, np.ndarray]] = {}
        state_parts: list[np.ndarray] = []
        for agent in self.layout.agents:
            source = self._agent_observation_group(group, agent)
            qpos = np.asarray(source["qpos"][index], dtype=np.float32)
            qvel = np.asarray(source["qvel"][index], dtype=np.float32)
            agents[agent.target_name] = {"qpos": qpos, "qvel": qvel}
            state_parts.extend((qpos.reshape(-1), qvel.reshape(-1)))
        calibration: dict[str, dict[str, np.ndarray]] = {}
        source_params = group.get("obs/sensor_param")
        if include_calibration and isinstance(source_params, h5py.Group):
            for camera in self.layout.cameras:
                if not camera.calibration_fields:
                    continue
                calibration[camera.target_name] = {
                    name: np.asarray(
                        source_params[camera.source_name][name][index],
                        dtype=np.float32,
                    )
                    for name in camera.calibration_fields
                }
        return {
            "proprioception": np.concatenate(state_parts).astype(
                np.float32, copy=False
            ),
            "agents": agents,
            "camera_calibration": calibration,
        }

    def _action(self, group: h5py.Group, index: int) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(
                    self._agent_action_dataset(group, agent)[index],
                    dtype=np.float32,
                ).reshape(-1)
                for agent in self.layout.agents
            ]
        ).astype(np.float32, copy=False)

    def _images(
        self,
        group: h5py.Group,
        index: int,
        cameras: Sequence[RoboFactoryCameraLayout],
    ) -> dict[str, np.ndarray]:
        sensor_data = group.get("obs/sensor_data")
        if not isinstance(sensor_data, h5py.Group):
            return {}
        return {
            camera.target_name: np.asarray(
                sensor_data[camera.source_name]["rgb"][index], dtype=np.uint8
            )
            for camera in cameras
        }

    def _transition_info(
        self,
        group: h5py.Group,
        index: int,
        *,
        commanded_action: np.ndarray,
        executed_action_source: str | None,
    ) -> dict[str, Any]:
        agent_actions = {
            agent.target_name: np.asarray(
                self._agent_action_dataset(group, agent)[index], dtype=np.float32
            )
            for agent in self.layout.agents
        }
        info: dict[str, Any] = {
            "agent_actions": agent_actions,
        }
        if executed_action_source == COMMAND_ECHO_ACTION_SOURCE:
            info["executed_action"] = commanded_action.copy()
            info["agent_executed_actions"] = {
                name: action.copy() for name, action in agent_actions.items()
            }
        if self.layout.has_success:
            info["success"] = bool(group["success"][index])
        if self.layout.has_failure:
            info["failure"] = bool(group["fail"][index])
        return info

    def _uses_flat_single_agent_layout(self, group: h5py.Group) -> bool:
        return isinstance(group.get("actions"), h5py.Dataset)

    def _agent_action_dataset(
        self,
        group: h5py.Group,
        agent: RoboFactoryAgentLayout,
    ) -> h5py.Dataset:
        actions = group.get("actions")
        if isinstance(actions, h5py.Dataset):
            if agent.source_name != _FLAT_SINGLE_AGENT_NAME:
                raise ValueError(
                    "flat single-agent actions can only map to "
                    f"{_FLAT_SINGLE_AGENT_NAME!r}"
                )
            return actions
        if isinstance(actions, h5py.Group):
            return _dataset(actions, agent.source_name)
        raise KeyError("required HDF5 dataset or group 'actions' is missing")

    def _agent_observation_group(
        self,
        group: h5py.Group,
        agent: RoboFactoryAgentLayout,
    ) -> h5py.Group:
        observations = _group(group, "obs/agent")
        if self._uses_flat_single_agent_layout(group):
            if agent.source_name != _FLAT_SINGLE_AGENT_NAME:
                raise ValueError(
                    "flat single-agent observations can only map to "
                    f"{_FLAT_SINGLE_AGENT_NAME!r}"
                )
            return observations
        return _group(observations, agent.source_name)

    def _episode_metadata(
        self,
        episode: RoboFactoryEpisode,
        *,
        schema: TrajectorySchema,
        fps: float,
        task_id: str,
        exported_cameras: Sequence[RoboFactoryCameraLayout],
        executed_action_source: str | None,
    ) -> dict[str, Any]:
        commanded_field = _field_name_for_source(schema, "action") or "action"
        executed_field = _field_name_for_source(schema, "info.executed_action")
        return {
            "schema_profile": schema.profile,
            "schema_version": schema.version,
            "source_format": "RoboFactory/ManiSkill HDF5",
            "source_episode_id": episode.source_id,
            "source_episode_key": episode.source_key,
            "source_env_id": self.env_id,
            "source_episode_metadata": dict(episode.metadata),
            "seed": episode.seed,
            "task_id": task_id,
            "control_mode": self._control_mode(),
            "control_frequency_hz": float(fps),
            "image_frequency_hz": float(fps),
            "agent_name_map": {
                agent.source_name: agent.target_name for agent in self.layout.agents
            },
            "agent_order": [agent.source_name for agent in self.layout.agents],
            "state_component_order": ["qpos", "qvel"],
            "state_layout": self.field_mapping(schema=schema)["centralized_state"],
            "action_layout": self.field_mapping(schema=schema)["centralized_action"],
            "camera_name_map": {
                camera.source_name: camera.target_name
                for camera in exported_cameras
            },
            "camera_order": [camera.target_name for camera in exported_cameras],
            "action_history_field": commanded_field,
            "action_history_semantics": "past_controller_commands",
            "executed_action_field": executed_field,
            "executed_action_source": executed_action_source,
            "executed_action_semantics": (
                "command_echo_assumed_actuator_executed"
                if executed_action_source == COMMAND_ECHO_ACTION_SOURCE
                else None
            ),
            "command_equals_executed_assumption": (
                executed_action_source == COMMAND_ECHO_ACTION_SOURCE
            ),
            "independent_actuator_feedback_available": False,
            "reward_available": self.layout.has_rewards,
        }

    def _episode_success(self, source_key: str) -> bool | None:
        group = self._file[source_key]
        success = group.get("success")
        if not isinstance(success, h5py.Dataset) or success.shape == (0,):
            return None
        return bool(success[-1])

    @staticmethod
    def _scalar(
        group: h5py.Group, name: str, index: int, default: float
    ) -> float:
        if name not in group:
            return float(default)
        return float(group[name][index])


def _group(root: h5py.Group, path: str) -> h5py.Group:
    value = root.get(path)
    if not isinstance(value, h5py.Group):
        raise KeyError(f"required HDF5 group {path!r} is missing")
    return value


def _dataset(root: h5py.Group, path: str) -> h5py.Dataset:
    value = root.get(path)
    if not isinstance(value, h5py.Dataset):
        raise KeyError(f"required HDF5 dataset {path!r} is missing")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def _identifier(value: str) -> str:
    result = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    if not result:
        result = "unnamed"
    if result[0].isdigit():
        result = f"item_{result}"
    return result


def _camera_alias(value: str) -> str:
    normalized = _identifier(value)
    match = re.fullmatch(r"(?:head_)?camera_agent_?(\d+)", normalized)
    if match is not None:
        return f"agent_{match.group(1)}"
    # Single-Panda upstream tasks expose their only workspace sensor as
    # ``head_camera``; multi-Panda tasks expose the equivalent shared view as
    # ``head_camera_global``.  Canonicalize both without changing RGB bytes.
    if normalized in {
        "head_camera",
        "head_camera_global",
        "camera_global",
        "global_camera",
    }:
        return "global"
    return normalized


def _unique_aliases(
    names: Sequence[str], transform: Callable[[str], str]
) -> dict[str, str]:
    result = {name: str(transform(name)) for name in names}
    reverse: dict[str, list[str]] = {}
    for source, target in result.items():
        reverse.setdefault(target, []).append(source)
    collisions = {
        target: sources for target, sources in reverse.items() if len(sources) > 1
    }
    if collisions:
        raise ValueError(f"normalized feature names collide: {collisions}")
    return result


def _task_from_env_id(env_id: str) -> str:
    value = re.sub(r"(?i)(?:[-_]rf)$", "", env_id).strip()
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    value = re.sub(r"[_-]+", " ", value)
    value = " ".join(value.split())
    return value[:1].upper() + value[1:] if value else "RoboFactory task"


def _task_id_from_env_id(env_id: str) -> str:
    value = re.sub(r"(?i)(?:[-_]rf)$", "", env_id).strip()
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return _identifier(value)


def _exported_camera_names(schema: TrajectorySchema) -> tuple[str, ...]:
    prefix = "observation.images."
    return tuple(
        field.name.removeprefix(prefix)
        for field in schema.fields
        if field.name.startswith(prefix)
    )


def _field_name_for_source(
    schema: TrajectorySchema | None, source: str
) -> str | None:
    if schema is None:
        return {
            "reward": "next.reward",
            "terminated": "next.terminated",
            "truncated": "next.truncated",
            "done": "next.done",
            "info.success": "next.success",
            "info.failure": "next.failure",
        }.get(source)
    return next(
        (field.name for field in schema.fields if field.source == source), None
    )


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "COMMAND_ECHO_ACTION_SOURCE",
    "ROBOFACTORY_M1_PROFILE",
    "ROBOFACTORY_M1_SCHEMA_VERSION",
    "ROBOFACTORY_SCHEMA_VERSION",
    "RoboFactoryAgentLayout",
    "RoboFactoryCameraLayout",
    "RoboFactoryDataset",
    "RoboFactoryEpisode",
    "RoboFactoryLayout",
]
