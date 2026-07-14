"""LeRobotDataset v3 exporter using LeRobot's official writer API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from data.exporters.base import EpisodeMetadata
from data.trajectory import FieldSpec, TrajectorySchema
from envs.runtime import RolloutSummary, SimulationTransition


class LeRobotTrajectoryExporter:
    """Stream frames through ``LeRobotDataset.create/add_frame/save_episode``.

    The optional dependency is imported only when this backend is selected.
    That keeps HDF5 collection and the simulation environment usable without
    installing the full Hugging Face dataset/video stack.
    """

    _AUTO_FIELDS = {
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task",
        "task_index",
    }

    def __init__(
        self,
        root: str | Path,
        schema: TrajectorySchema,
        *,
        repo_id: str,
        fps: float,
        robot_type: str = "two_robot_carry",
        use_videos: bool = True,
        streaming_encoding: bool = True,
        dataset_factory: Callable[..., Any] | None = None,
    ) -> None:
        if fps <= 0.0 or not float(fps).is_integer():
            raise ValueError("LeRobot fps must be a positive integer")
        if not repo_id:
            raise ValueError("LeRobot repo_id cannot be empty")
        self.root = Path(root)
        self.schema = schema
        self.repo_id = repo_id
        self.fps = int(fps)
        self.robot_type = robot_type
        self.use_videos = bool(use_videos)
        self.streaming_encoding = bool(streaming_encoding)
        self._dataset_factory = dataset_factory
        self._dataset: Any | None = None
        self._feature_shapes: dict[str, tuple[int, ...]] = {}
        self._feature_dtypes: dict[str, str] = {}
        self._metadata: EpisodeMetadata | None = None
        self._steps = 0

    def start_episode(self, metadata: EpisodeMetadata) -> None:
        if self._metadata is not None:
            raise RuntimeError("previous LeRobot episode is still open")
        self._metadata = metadata
        self._steps = 0

    def write_transition(self, transition: SimulationTransition) -> None:
        if self._metadata is None:
            raise RuntimeError("start_episode must be called before write_transition")
        resolved = self.schema.resolve(transition)
        if self._dataset is None:
            self._create_dataset(resolved)
        frame = {
            name: self._frame_value(name, value)
            for name, value in resolved.items()
            if name not in self._AUTO_FIELDS
        }
        frame["task"] = transition.task
        self._dataset.add_frame(frame)
        self._steps += 1

    def end_episode(self, summary: RolloutSummary) -> None:
        if self._metadata is None:
            raise RuntimeError("no LeRobot episode is open")
        if summary.steps != self._steps:
            raise ValueError(
                f"rollout summary has {summary.steps} steps, exporter wrote {self._steps}"
            )
        if self._dataset is None:
            raise ValueError("LeRobot cannot save an empty episode")
        self._dataset.save_episode()
        self._metadata = None
        self._steps = 0

    def _create_dataset(self, resolved: dict[str, Any]) -> None:
        features: dict[str, dict[str, Any]] = {}
        field_by_name = {field.name: field for field in self.schema.fields}
        for name, value in resolved.items():
            if name in self._AUTO_FIELDS:
                continue
            field = field_by_name[name]
            feature, shape, dtype = self._feature(field, value)
            features[name] = feature
            self._feature_shapes[name] = shape
            self._feature_dtypes[name] = dtype
        factory = self._dataset_factory or self._official_factory()
        self.root.mkdir(parents=True, exist_ok=True)
        self._dataset = factory(
            repo_id=self.repo_id,
            root=self.root,
            fps=self.fps,
            robot_type=self.robot_type,
            features=features,
            use_videos=self.use_videos,
            streaming_encoding=self.streaming_encoding,
        )

    def _feature(
        self, field: FieldSpec, value: Any
    ) -> tuple[dict[str, Any], tuple[int, ...], str]:
        array = np.asarray(value)
        if field.is_image:
            if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
                raise ValueError(
                    f"LeRobot image {field.name!r} must be HWC, got {array.shape}"
                )
            dtype = "video" if self.use_videos else "image"
            names: Any = ["height", "width", "channel"]
            shape = tuple(array.shape)
        else:
            if array.dtype.kind in {"U", "S", "O"}:
                raise TypeError(
                    f"LeRobot custom feature {field.name!r} must be numeric; "
                    "use the canonical task field for language"
                )
            shape = tuple(array.shape) or (1,)
            dtype = str(array.dtype)
            names = [f"{field.name}_{index}" for index in range(int(np.prod(shape)))]
            if len(shape) > 1:
                names = None
        feature: dict[str, Any] = {"dtype": dtype, "shape": shape}
        if names is not None:
            feature["names"] = names
        return feature, shape, dtype

    def _frame_value(self, name: str, value: Any) -> np.ndarray:
        array = np.asarray(value)
        shape = self._feature_shapes[name]
        if array.ndim == 0:
            array = array.reshape(1)
        if tuple(array.shape) != shape:
            raise ValueError(
                f"LeRobot field {name!r} shape changed from {shape} to {array.shape}"
            )
        if self._feature_dtypes[name] in {"image", "video"}:
            return array.astype(np.uint8, copy=False)
        return array

    @staticmethod
    def _official_factory() -> Callable[..., Any]:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:  # pragma: no cover - depends on optional extra.
            raise RuntimeError(
                "LeRobot export requires the optional 'lerobot' package. "
                "Install a LeRobot release with Dataset v3 support, then rerun "
                "the same command; HDF5 export has no such dependency."
            ) from exc
        return LeRobotDataset.create

    def close(self) -> None:
        if self._dataset is not None:
            self._dataset.finalize()
            self._dataset = None
        self._metadata = None


__all__ = ["LeRobotTrajectoryExporter"]
