"""Format-neutral trajectory fields and VLA/WAM schema profiles."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from envs.runtime import SimulationTransition

_MISSING = object()


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
        return self.source.startswith("images.") or self.source.startswith(
            "next_images."
        )


@dataclass(frozen=True)
class TrajectorySchema:
    """A named, immutable set of export fields."""

    profile: str
    fields: tuple[FieldSpec, ...]

    def __post_init__(self) -> None:
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            raise ValueError(f"duplicate exported fields: {duplicates}")

    def resolve(self, transition: SimulationTransition) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for field in self.fields:
            value = extract_source(transition, field.source, default=_MISSING)
            if value is _MISSING:
                if field.required:
                    raise KeyError(
                        f"required source {field.source!r} for {field.name!r} is missing"
                    )
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
        return TrajectorySchema(profile=self.profile, fields=tuple(ordered))


def schema_profile(
    profile: str,
    *,
    cameras: Sequence[str] = (),
) -> TrajectorySchema:
    """Build common VLA/WAM profiles, including optional live RGB streams."""

    profile = profile.lower().replace("-", "_")
    common = (
        FieldSpec("timestamp", "timestamp", "float64"),
        FieldSpec("frame_index", "frame_index", "int64"),
        FieldSpec("episode_index", "episode_index", "int64"),
        FieldSpec("task", "task"),
    )
    images = tuple(
        FieldSpec(f"observation.images.{camera}", f"images.{camera}", "uint8")
        for camera in cameras
    )
    if profile in {"vla", "lerobot", "robocasa"}:
        fields = (
            common
            + (
                FieldSpec("observation.state", "observation.global_state", "float32"),
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
                FieldSpec("observation.agent_0", "observation.robot_0", "float32"),
                FieldSpec("observation.agent_1", "observation.robot_1", "float32"),
                FieldSpec("observation.object", "observation.object", "float32"),
                FieldSpec(
                    "observation.global_state", "observation.global_state", "float32"
                ),
                FieldSpec("action", "action", "float32"),
                FieldSpec(
                    "next_observation.agent_0", "next_observation.robot_0", "float32"
                ),
                FieldSpec(
                    "next_observation.agent_1", "next_observation.robot_1", "float32"
                ),
                FieldSpec(
                    "next_observation.object", "next_observation.object", "float32"
                ),
                FieldSpec(
                    "next_observation.global_state",
                    "next_observation.global_state",
                    "float32",
                ),
                FieldSpec("reward", "reward", "float32"),
                FieldSpec("done", "done", "bool"),
                FieldSpec("success", "info.success", "bool", required=False),
                FieldSpec("failure", "info.failure", "bool", required=False),
                FieldSpec("progress", "info.progress", "float32", required=False),
                FieldSpec("force", "info.force_proxy", "float32", required=False),
            )
            + images
        )
        return TrajectorySchema(profile=profile, fields=fields)
    if profile in {"rmbench", "robotwin"}:
        fields = (
            common
            + (
                FieldSpec("observation.state", "observation.global_state", "float32"),
                FieldSpec("observation.agent_0", "observation.robot_0", "float32"),
                FieldSpec("observation.agent_1", "observation.robot_1", "float32"),
                FieldSpec("action", "action", "float32"),
                FieldSpec(
                    "next_observation.state", "next_observation.global_state", "float32"
                ),
                FieldSpec("reward", "reward", "float32"),
                FieldSpec("done", "done", "bool"),
                FieldSpec("success", "info.success", "bool", required=False),
                FieldSpec("progress", "info.progress", "float32", required=False),
            )
            + images
        )
        return TrajectorySchema(profile=profile, fields=fields)
    raise ValueError(
        f"unknown schema profile {profile!r}; expected vla, wam, robocasa, or rmbench"
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
        profile=str(payload.get("profile", "custom")), fields=fields
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
    "TrajectorySchema",
    "extract_source",
    "load_schema_json",
    "parse_field_assignment",
    "schema_profile",
]
