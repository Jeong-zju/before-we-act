"""Format-neutral trajectory fields and VLA/WAM schema profiles."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from envs.runtime import SimulationTransition

_MISSING = object()
PROPRIO_WAM_SCHEMA_VERSION = "wam.proprio/1.0"
MULTIMODAL_WAM_SCHEMA_VERSION = "wam.multimodal/1.1"


@dataclass(frozen=True)
class FieldSpec:
    """Map a format-neutral source path to an exported feature name."""

    name: str
    source: str
    dtype: str | None = None
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.source:
            raise ValueError("field name and source cannot be empty")
        if self.dtype is not None:
            np.dtype(self.dtype)

    @property
    def is_image(self) -> bool:
        return (
            self.name.startswith("observation.images.")
            or self.name.startswith("next_observation.images.")
            or self.source.startswith("images.")
            or self.source.startswith("next_images.")
        )


@dataclass(frozen=True)
class TrajectorySchema:
    """A named, immutable set of export fields."""

    profile: str
    fields: tuple[FieldSpec, ...]
    version: str = "trajectory.schema/1"

    def __post_init__(self) -> None:
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            raise ValueError(f"duplicate exported fields: {duplicates}")

    def resolve(self, transition: SimulationTransition) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for field in self.fields:
            try:
                value = extract_source(transition, field.source)
            except KeyError:
                if field.required:
                    raise KeyError(
                        f"required source {field.source!r} for {field.name!r} is missing"
                    ) from None
                continue
            result[field.name] = _coerce(value, field.dtype)
        return result

    def with_overrides(
        self,
        *,
        add: Sequence[FieldSpec] = (),
        drop: Iterable[str] = (),
    ) -> "TrajectorySchema":
        drop_names = set(drop)
        ordered = [field for field in self.fields if field.name not in drop_names]
        positions = {field.name: index for index, field in enumerate(ordered)}
        for field in add:
            if field.name in positions:
                ordered[positions[field.name]] = field
            else:
                positions[field.name] = len(ordered)
                ordered.append(field)
        return TrajectorySchema(
            profile=self.profile,
            fields=tuple(ordered),
            version=self.version,
        )


def schema_profile(
    profile: str,
    *,
    cameras: Sequence[str] = (),
) -> TrajectorySchema:
    """Build common VLA/WAM profiles, including optional live RGB streams."""

    profile = profile.lower().replace("-", "_")
    if profile in {"wam_multimodal", "multimodal_wam"} and not cameras:
        cameras = ("fixed",)
    if len(cameras) != len(set(cameras)):
        raise ValueError("camera names must be unique and ordered explicitly")
    if any(not str(camera).strip() for camera in cameras):
        raise ValueError("camera names cannot be empty")
    if profile in {"wam_multimodal", "multimodal_wam"} and any(
        "." in str(camera) or "/" in str(camera) for camera in cameras
    ):
        raise ValueError("multimodal camera names cannot contain '.' or '/'")
    common = (
        FieldSpec("timestamp", "timestamp", "float64"),
        FieldSpec("frame_index", "frame_index", "int64"),
        FieldSpec("episode_index", "episode_index", "int64"),
        FieldSpec("task", "task"),
    )
    external_images = tuple(
        FieldSpec(f"observation.images.{camera}", f"images.{camera}", "uint8")
        for camera in cameras
    )
    onboard_images = (
        FieldSpec(
            "observation.images.robot_0",
            "observation.robot_0.image",
            "uint8",
        ),
        FieldSpec(
            "observation.images.robot_1",
            "observation.robot_1.image",
            "uint8",
        ),
    )
    images = onboard_images + external_images
    if profile in {"vla", "lerobot", "robocasa"}:
        fields = (
            common
            + (
                FieldSpec("observation.state", "observation.proprioception", "float32"),
                FieldSpec("action", "action", "float32"),
                FieldSpec("next.reward", "reward", "float32"),
                FieldSpec("next.done", "done", "bool"),
                FieldSpec("next.success", "info.success", "bool", required=False),
            )
            + images
        )
        return TrajectorySchema(profile=profile, fields=fields)
    if profile in {"wam", "world_action_model"}:
        fields = (
            common
            + (
                FieldSpec(
                    "observation.agent_0", "observation.robot_0.state", "float32"
                ),
                FieldSpec(
                    "observation.agent_1", "observation.robot_1.state", "float32"
                ),
                FieldSpec(
                    "observation.object",
                    "observation.privileged_state.object_pose",
                    "float32",
                ),
                FieldSpec(
                    "observation.privileged_state",
                    "observation.privileged_state.state",
                    "float32",
                ),
                FieldSpec("action", "action", "float32"),
                FieldSpec(
                    "next_observation.agent_0",
                    "next_observation.robot_0.state",
                    "float32",
                ),
                FieldSpec(
                    "next_observation.agent_1",
                    "next_observation.robot_1.state",
                    "float32",
                ),
                FieldSpec(
                    "next_observation.object",
                    "next_observation.privileged_state.object_pose",
                    "float32",
                ),
                FieldSpec(
                    "next_observation.privileged_state",
                    "next_observation.privileged_state.state",
                    "float32",
                ),
                FieldSpec("reward", "reward", "float32"),
                FieldSpec("done", "done", "bool"),
                FieldSpec("success", "info.success", "bool", required=False),
                FieldSpec("failure", "info.failure", "bool", required=False),
                FieldSpec(
                    "response_progress",
                    "info.response_progress",
                    "float32",
                    required=False,
                ),
                FieldSpec(
                    "coordination_error",
                    "info.coordination_error",
                    "float32",
                    required=False,
                ),
                FieldSpec(
                    "braking_agent",
                    "info.braking_agent",
                    "int64",
                    required=False,
                ),
            )
            + images
        )
        return TrajectorySchema(profile=profile, fields=fields)
    if profile in {"wam_proprio", "proprio_wam", "proprioceptive_wam"}:
        fields = common + (
            FieldSpec("observation.state", "observation.proprioception", "float32"),
            FieldSpec("commanded_action", "action", "float32"),
            FieldSpec("executed_action", "info.executed_action", "float32"),
            FieldSpec(
                "next_observation.state",
                "next_observation.proprioception",
                "float32",
            ),
            FieldSpec("reward", "reward", "float32"),
            FieldSpec("terminated", "terminated", "bool"),
            FieldSpec("truncated", "truncated", "bool"),
            FieldSpec("done", "done", "bool"),
            FieldSpec("success", "info.success", "bool"),
            FieldSpec("failure", "info.failure", "bool"),
            FieldSpec("failure_reason", "info.failure_reason"),
            FieldSpec("response_progress", "info.response_progress", "float32"),
            FieldSpec("coordination_error", "info.coordination_error", "float32"),
            FieldSpec("schema_version", "metadata.schema_version"),
            FieldSpec("behavior_id", "metadata.behavior_id"),
            FieldSpec(
                "perturbation_config",
                "metadata.perturbation_config",
            ),
            FieldSpec(
                "environment_config",
                "metadata.environment_config",
            ),
            FieldSpec(
                "randomization_config",
                "metadata.randomization_config",
            ),
        )
        return TrajectorySchema(
            profile="wam_proprio",
            fields=fields,
            version=PROPRIO_WAM_SCHEMA_VERSION,
        )
    if profile in {"wam_multimodal", "multimodal_wam"}:
        current_image_fields: list[FieldSpec] = []
        next_image_fields: list[FieldSpec] = []
        current_camera_fields: list[FieldSpec] = []
        next_camera_fields: list[FieldSpec] = []
        for raw_camera in cameras:
            camera = str(raw_camera)
            current_image_fields.extend(
                (
                    FieldSpec(
                        f"observation.images.{camera}",
                        f"images.{camera}",
                        "uint8",
                    ),
                    FieldSpec(
                        f"observation.image_timestamp.{camera}",
                        f"image_timestamps.{camera}",
                        "float64",
                    ),
                    FieldSpec(
                        f"observation.image_state_timestamp.{camera}",
                        f"image_state_timestamps.{camera}",
                        "float64",
                    ),
                    FieldSpec(
                        f"observation.image_frame_index.{camera}",
                        f"image_frame_indices.{camera}",
                        "int64",
                    ),
                )
            )
            next_image_fields.extend(
                (
                    FieldSpec(
                        f"next_observation.images.{camera}",
                        f"next_images.{camera}",
                        "uint8",
                    ),
                    FieldSpec(
                        f"next_observation.image_timestamp.{camera}",
                        f"next_image_timestamps.{camera}",
                        "float64",
                    ),
                    FieldSpec(
                        f"next_observation.image_state_timestamp.{camera}",
                        f"next_image_state_timestamps.{camera}",
                        "float64",
                    ),
                    FieldSpec(
                        f"next_observation.image_frame_index.{camera}",
                        f"next_image_frame_indices.{camera}",
                        "int64",
                    ),
                )
            )
            current_camera_fields.extend(
                (
                    FieldSpec(
                        f"camera.intrinsics.{camera}",
                        f"camera_intrinsics.{camera}",
                        "float32",
                    ),
                    FieldSpec(
                        f"camera.extrinsics.{camera}",
                        f"camera_extrinsics.{camera}",
                        "float32",
                    ),
                    FieldSpec(
                        f"camera.resolution.{camera}",
                        f"camera_resolutions.{camera}",
                        "int64",
                    ),
                )
            )
            next_camera_fields.extend(
                (
                    FieldSpec(
                        f"next_camera.intrinsics.{camera}",
                        f"next_camera_intrinsics.{camera}",
                        "float32",
                    ),
                    FieldSpec(
                        f"next_camera.extrinsics.{camera}",
                        f"next_camera_extrinsics.{camera}",
                        "float32",
                    ),
                    FieldSpec(
                        f"next_camera.resolution.{camera}",
                        f"next_camera_resolutions.{camera}",
                        "int64",
                    ),
                )
            )
        fields = (
            (
                FieldSpec("timestamp", "timestamp", "float64"),
                FieldSpec("frame_index", "frame_index", "int64"),
                FieldSpec("episode_index", "episode_index", "int64"),
                FieldSpec("seed", "metadata.seed", "int64"),
                FieldSpec("task.text", "task"),
                FieldSpec("task.id", "metadata.task_id"),
                FieldSpec(
                    "observation.state", "observation.proprioception", "float32"
                ),
            )
            + tuple(current_image_fields)
            + (
                FieldSpec("action.commanded", "action", "float32"),
                FieldSpec("action.executed", "info.executed_action", "float32"),
                FieldSpec(
                    "next_observation.state",
                    "next_observation.proprioception",
                    "float32",
                ),
            )
            + tuple(next_image_fields)
            + tuple(current_camera_fields)
            + tuple(next_camera_fields)
            + (
                FieldSpec(
                    "event.visual_signal_active",
                    "info.visual_signal_active",
                    "bool",
                ),
                FieldSpec(
                    "event.visual_signal_onset_step",
                    "info.visual_signal_onset_step",
                    "int64",
                ),
                FieldSpec(
                    "event.visual_signal_kind",
                    "info.visual_signal_kind",
                ),
                FieldSpec(
                    "event.rendered_cue_variant",
                    "info.rendered_cue_variant",
                    "int64",
                ),
                FieldSpec("reward", "reward", "float32"),
                FieldSpec("terminated", "terminated", "bool"),
                FieldSpec("truncated", "truncated", "bool"),
                FieldSpec("done", "done", "bool"),
                FieldSpec("success", "info.success", "bool"),
                FieldSpec("failure", "info.failure", "bool"),
                FieldSpec("failure_reason", "info.failure_reason"),
                FieldSpec("schema_version", "metadata.schema_version"),
                FieldSpec("behavior_id", "metadata.behavior_id"),
                FieldSpec("environment_config", "metadata.environment_config"),
                FieldSpec(
                    "randomization_config", "metadata.randomization_config"
                ),
            )
        )
        return TrajectorySchema(
            profile="wam_multimodal",
            fields=fields,
            version=MULTIMODAL_WAM_SCHEMA_VERSION,
        )
    if profile in {"rmbench", "robotwin"}:
        fields = (
            common
            + (
                FieldSpec("observation.state", "observation.proprioception", "float32"),
                FieldSpec(
                    "observation.agent_0", "observation.robot_0.state", "float32"
                ),
                FieldSpec(
                    "observation.agent_1", "observation.robot_1.state", "float32"
                ),
                FieldSpec("action", "action", "float32"),
                FieldSpec(
                    "next_observation.state",
                    "next_observation.proprioception",
                    "float32",
                ),
                FieldSpec("reward", "reward", "float32"),
                FieldSpec("done", "done", "bool"),
                FieldSpec("success", "info.success", "bool", required=False),
                FieldSpec(
                    "response_progress",
                    "info.response_progress",
                    "float32",
                    required=False,
                ),
            )
            + images
        )
        return TrajectorySchema(profile=profile, fields=fields)
    raise ValueError(
        f"unknown schema profile {profile!r}; expected vla, wam, wam_proprio, "
        "wam_multimodal, robocasa, or rmbench"
    )


def extract_source(
    transition: SimulationTransition,
    source: str,
    *,
    default: Any = _MISSING,
) -> Any:
    """Resolve a dotted source path from one rollout transition."""

    roots: dict[str, Any] = {
        "observation": transition.observation,
        "action": transition.action,
        "next_observation": transition.next_observation,
        "reward": transition.reward,
        "terminated": transition.terminated,
        "truncated": transition.truncated,
        "done": transition.done,
        "info": transition.info,
        "task": transition.task,
        "timestamp": transition.timestamp,
        "frame_index": transition.frame_index,
        "episode_index": transition.episode_index,
        "images": transition.images,
        "next_images": transition.next_images,
        "image_timestamps": getattr(transition, "image_timestamps", {}),
        "next_image_timestamps": getattr(
            transition, "next_image_timestamps", {}
        ),
        "image_state_timestamps": getattr(
            transition, "image_state_timestamps", {}
        ),
        "next_image_state_timestamps": getattr(
            transition, "next_image_state_timestamps", {}
        ),
        "image_frame_indices": getattr(transition, "image_frame_indices", {}),
        "next_image_frame_indices": getattr(
            transition, "next_image_frame_indices", {}
        ),
        "camera_intrinsics": getattr(transition, "camera_intrinsics", {}),
        "next_camera_intrinsics": getattr(
            transition, "next_camera_intrinsics", {}
        ),
        "camera_extrinsics": getattr(transition, "camera_extrinsics", {}),
        "next_camera_extrinsics": getattr(
            transition, "next_camera_extrinsics", {}
        ),
        "camera_resolutions": getattr(transition, "camera_resolutions", {}),
        "next_camera_resolutions": getattr(
            transition, "next_camera_resolutions", {}
        ),
        "metadata": transition.metadata,
    }
    root_name, separator, remainder = source.partition(".")
    if root_name not in roots:
        if default is not _MISSING:
            return default
        raise KeyError(f"unknown trajectory source root {root_name!r}")
    value = roots[root_name]
    if not separator:
        return value
    for part in remainder.split("."):
        if isinstance(value, Mapping):
            if part not in value:
                if default is not _MISSING:
                    return default
                raise KeyError(f"source path {source!r} is missing component {part!r}")
            value = value[part]
        elif isinstance(value, (list, tuple, np.ndarray)) and part.isdigit():
            value = value[int(part)]
        else:
            if default is not _MISSING:
                return default
            raise KeyError(f"cannot descend into {part!r} in source path {source!r}")
    return value


def parse_field_assignment(value: str) -> FieldSpec:
    """Parse ``NAME=SOURCE`` or ``NAME=SOURCE::DTYPE`` CLI syntax."""

    if "=" not in value:
        raise ValueError("custom field must use NAME=SOURCE or NAME=SOURCE::DTYPE")
    name, source_and_dtype = value.split("=", 1)
    source, marker, dtype = source_and_dtype.rpartition("::")
    if not marker:
        source, dtype = source_and_dtype, None
    return FieldSpec(name=name.strip(), source=source.strip(), dtype=dtype or None)


def load_schema_json(path: str | Path) -> TrajectorySchema:
    """Load a custom schema from ``{"profile": ..., "fields": [...]}``."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = tuple(FieldSpec(**entry) for entry in payload["fields"])
    return TrajectorySchema(
        profile=str(payload.get("profile", "custom")),
        fields=fields,
        version=str(payload.get("version", "trajectory.schema/1")),
    )


def _coerce(value: Any, dtype: str | None) -> Any:
    if isinstance(value, str):
        if dtype is not None:
            raise TypeError("string fields cannot declare a NumPy dtype")
        return value
    array = np.asarray(value, dtype=np.dtype(dtype) if dtype else None)
    return array


__all__ = [
    "FieldSpec",
    "MULTIMODAL_WAM_SCHEMA_VERSION",
    "PROPRIO_WAM_SCHEMA_VERSION",
    "TrajectorySchema",
    "extract_source",
    "load_schema_json",
    "parse_field_assignment",
    "schema_profile",
]
