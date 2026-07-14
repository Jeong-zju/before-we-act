"""Incremental, one-episode-per-file HDF5 trajectory backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from data.exporters.base import EpisodeMetadata
from data.trajectory import TrajectorySchema, extract_source
from envs.runtime import RolloutSummary, SimulationTransition
from envs.video import StreamingVideoObserver


class HDF5TrajectoryExporter:
    """Write custom fields incrementally and optionally mirror RGB to MP4."""

    FORMAT_VERSION = "wam.trajectory.hdf5/1"

    def __init__(
        self,
        root: str | Path,
        schema: TrajectorySchema,
        *,
        compression: str | None = "gzip",
        stream_videos: bool = False,
        video_codec: str = "mp4v",
    ) -> None:
        self.root = Path(root)
        self.schema = schema
        self.compression = compression
        self.stream_videos = bool(stream_videos)
        self.video_codec = video_codec
        self._file: h5py.File | None = None
        self._partial_path: Path | None = None
        self._final_path: Path | None = None
        self._metadata: EpisodeMetadata | None = None
        self._videos: dict[str, StreamingVideoObserver] = {}
        self._steps = 0
        self._episode_metadata: dict[str, Any] | None = None

    def start_episode(self, metadata: EpisodeMetadata) -> None:
        if self._file is not None:
            raise RuntimeError("previous HDF5 episode is still open")
        self.root.mkdir(parents=True, exist_ok=True)
        self._metadata = metadata
        self._steps = 0
        self._final_path = self.root / f"episode_{metadata.episode_index:06d}.hdf5"
        self._partial_path = self._final_path.with_suffix(".partial.hdf5")
        if self._partial_path.exists():
            self._partial_path.unlink()
        self._file = h5py.File(self._partial_path, "w")
        self._file.attrs.update(
            {
                "format_version": self.FORMAT_VERSION,
                "schema_profile": self.schema.profile,
                "schema_version": self.schema.version,
                "fps": metadata.fps,
                "episode_index": metadata.episode_index,
                "seed": -1 if metadata.seed is None else metadata.seed,
                "task": metadata.task,
                "transition_semantics": "observation[t], action[t], observation[t+1]",
            }
        )
        schema_group = self._file.create_group("schema")
        for field in self.schema.fields:
            group = schema_group.create_group(_hdf5_path(field.name))
            group.attrs["source"] = field.source
            group.attrs["dtype"] = field.dtype or "inferred"
            group.attrs["required"] = field.required

        if self.stream_videos:
            for field in self.schema.fields:
                if not field.is_image:
                    continue
                stream = field.name.removeprefix("observation.images.").replace(
                    ".", "_"
                )
                path = (
                    self.root
                    / "videos"
                    / f"episode_{metadata.episode_index:06d}"
                    / f"{stream}.mp4"
                )
                video = StreamingVideoObserver(
                    path,
                    stream=stream,
                    fps=metadata.fps,
                    codec=self.video_codec,
                    frame_getter=lambda transition, source=field.source: extract_source(
                        transition, source
                    ),
                )
                video.on_episode_start(
                    episode_index=metadata.episode_index,
                    seed=metadata.seed,
                    observation=metadata.initial_observation,
                    info=metadata.initial_info,
                    task=metadata.task,
                )
                self._videos[stream] = video

    def write_transition(self, transition: SimulationTransition) -> None:
        if self._file is None:
            raise RuntimeError("start_episode must be called before write_transition")
        self._write_episode_metadata(transition.metadata)
        resolved = self.schema.resolve(transition)
        data = self._file.require_group("data")
        for name, value in resolved.items():
            self._append(data, _hdf5_path(name), value)
        for video in self._videos.values():
            video.on_transition(transition)
        self._steps += 1

    def end_episode(self, summary: RolloutSummary) -> None:
        if self._file is None or self._final_path is None or self._partial_path is None:
            raise RuntimeError("no HDF5 episode is open")
        if summary.steps != self._steps:
            raise ValueError(
                f"rollout summary has {summary.steps} steps, exporter wrote {self._steps}"
            )
        self._file.attrs["num_steps"] = self._steps
        self._file.attrs["total_reward"] = summary.total_reward
        self._file.attrs["terminated"] = summary.terminated
        self._file.attrs["truncated"] = summary.truncated
        for video in self._videos.values():
            video.on_episode_end(summary)
            video.close()
        self._videos.clear()
        self._file.flush()
        self._file.close()
        self._file = None
        self._partial_path.replace(self._final_path)
        self._partial_path = None
        self._final_path = None
        self._metadata = None
        self._episode_metadata = None

    def _write_episode_metadata(self, metadata: Mapping[str, Any]) -> None:
        if self._file is None:
            raise RuntimeError("no HDF5 episode is open")
        normalized = {
            str(key): _metadata_value(value) for key, value in metadata.items()
        }
        if self._episode_metadata is not None:
            if normalized != self._episode_metadata:
                raise ValueError("episode metadata changed within one trajectory")
            return
        self._episode_metadata = normalized
        self._file.attrs["episode_metadata_json"] = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
        )
        for key in ("behavior_id", "schema_version"):
            if key in normalized:
                self._file.attrs[key] = normalized[key]

    def _append(self, root: h5py.Group, path: str, value: Any) -> None:
        parts = path.split("/")
        group = root
        for part in parts[:-1]:
            group = group.require_group(part)
        name = parts[-1]
        if isinstance(value, str):
            array = np.asarray(value, dtype=object)
            dtype: Any = h5py.string_dtype(encoding="utf-8")
            sample_shape: tuple[int, ...] = ()
        else:
            array = np.asarray(value)
            dtype = array.dtype
            sample_shape = tuple(array.shape)
        if name not in group:
            kwargs: dict[str, Any] = {
                "shape": (0, *sample_shape),
                "maxshape": (None, *sample_shape),
                "chunks": (1, *sample_shape) if sample_shape else (1024,),
                "dtype": dtype,
            }
            if self.compression and sample_shape:
                kwargs["compression"] = self.compression
            group.create_dataset(name, **kwargs)
        dataset = group[name]
        if tuple(dataset.shape[1:]) != sample_shape:
            raise ValueError(
                f"field {path!r} shape changed from {dataset.shape[1:]} to {sample_shape}"
            )
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = value

    def close(self) -> None:
        for video in self._videos.values():
            video.close()
        self._videos.clear()
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
        self._episode_metadata = None


def _hdf5_path(name: str) -> str:
    return name.replace(".", "/").strip("/")


def _metadata_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _metadata_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"episode metadata value {value!r} is not JSON serializable")


__all__ = ["HDF5TrajectoryExporter"]
