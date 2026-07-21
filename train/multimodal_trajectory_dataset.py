"""Strict, episode-safe sequence windows for ``wam.multimodal/1.1``.

RGB tensors are returned as ``[time, camera, channel, height, width]`` uint8.
Camera order is explicit and never inferred from HDF5 or Python dictionary
iteration.  A camera resolution is encoded as ``[height, width]``.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterator, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from data.trajectory import MULTIMODAL_WAM_SCHEMA_VERSION
from train.trajectory_dataset import discover_episode_paths


@dataclass(frozen=True)
class MultimodalSequenceIndex:
    """One decision point inside one physical episode file."""

    file_index: int
    decision_t: int


@dataclass(frozen=True)
class MultimodalEpisodeRecord:
    """Validated shape and lineage metadata for one episode."""

    path: Path
    num_steps: int
    seed: int
    episode_index: int
    state_dim: int
    action_dim: int
    image_shape: tuple[int, int, int]
    fps: float
    task_text: str
    task_id: str


class MultimodalSequenceDataset(Dataset):
    """Load causal state/RGB/action histories without crossing episodes.

    ``past_actions`` contains actions that were actually executed between the
    history states.  Future ``candidate_actions`` remain commanded actions and
    ``executed_actions`` are returned separately.  Every current RGB timestamp
    must be no newer than its decision state timestamp; the same rule is
    applied to next-observation RGB against the corresponding next state.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        *,
        paths: Sequence[str | Path] | None = None,
        history_horizon: int = 32,
        forecast_horizon: int = 16,
        stride: int = 1,
        camera_order: Sequence[str] = ("fixed",),
        state_dim: int | None = None,
        action_dim: int | None = None,
        max_frame_age_seconds: float = 0.1,
        causal_tolerance_seconds: float = 1e-6,
        max_sensor_state_skew_seconds: float | None = None,
        hdf5_cache_size: int = 8,
    ) -> None:
        if paths is None:
            if data_dir is None:
                raise ValueError("data_dir or paths must be provided")
            resolved_paths = discover_episode_paths(data_dir)
        else:
            resolved_paths = [Path(path) for path in paths]
        if not resolved_paths:
            raise ValueError("paths cannot be empty")

        for name, value in (
            ("history_horizon", history_horizon),
            ("forecast_horizon", forecast_horizon),
            ("stride", stride),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if state_dim is not None and int(state_dim) <= 0:
            raise ValueError("state_dim must be positive when provided")
        if action_dim is not None and int(action_dim) <= 0:
            raise ValueError("action_dim must be positive when provided")
        if not np.isfinite(max_frame_age_seconds) or max_frame_age_seconds <= 0.0:
            raise ValueError("max_frame_age_seconds must be finite and positive")
        if (
            not np.isfinite(causal_tolerance_seconds)
            or causal_tolerance_seconds < 0.0
        ):
            raise ValueError("causal_tolerance_seconds must be finite and non-negative")
        if max_sensor_state_skew_seconds is not None and (
            not np.isfinite(max_sensor_state_skew_seconds)
            or max_sensor_state_skew_seconds < 0.0
        ):
            raise ValueError(
                "max_sensor_state_skew_seconds must be finite and non-negative"
            )
        if hdf5_cache_size < 0:
            raise ValueError("hdf5_cache_size cannot be negative")

        cameras = tuple(str(camera) for camera in camera_order)
        if not cameras or any(not camera for camera in cameras):
            raise ValueError("camera_order must contain at least one non-empty camera")
        if len(cameras) != len(set(cameras)):
            raise ValueError("camera_order must contain unique camera names")

        self.history_horizon = int(history_horizon)
        self.forecast_horizon = int(forecast_horizon)
        self.stride = int(stride)
        self.camera_order = cameras
        self.max_frame_age_seconds = float(max_frame_age_seconds)
        self.causal_tolerance_seconds = float(causal_tolerance_seconds)
        self.max_sensor_state_skew_seconds = max_sensor_state_skew_seconds
        self.hdf5_cache_size = int(hdf5_cache_size)
        self._cache_pid = os.getpid()
        self._cache: OrderedDict[str, h5py.File] = OrderedDict()

        records: list[MultimodalEpisodeRecord] = []
        expected_state_dim = None if state_dim is None else int(state_dim)
        expected_action_dim = None if action_dim is None else int(action_dim)
        expected_image_shape: tuple[int, int, int] | None = None
        for path in sorted(resolved_paths):
            record = self._inspect_episode(
                path,
                expected_state_dim=expected_state_dim,
                expected_action_dim=expected_action_dim,
            )
            if expected_state_dim is None:
                expected_state_dim = record.state_dim
            if expected_action_dim is None:
                expected_action_dim = record.action_dim
            if expected_image_shape is None:
                expected_image_shape = record.image_shape
            elif record.image_shape != expected_image_shape:
                raise ValueError(
                    f"{path} image shape {record.image_shape} does not match "
                    f"dataset image shape {expected_image_shape}"
                )
            records.append(record)

        assert expected_state_dim is not None
        assert expected_action_dim is not None
        assert expected_image_shape is not None
        self.state_dim = expected_state_dim
        self.action_dim = expected_action_dim
        self.image_shape = expected_image_shape
        self.records = records
        self.paths = [record.path for record in records]
        self.index = [
            MultimodalSequenceIndex(file_index=file_index, decision_t=decision_t)
            for file_index, record in enumerate(records)
            for decision_t in range(0, record.num_steps, self.stride)
        ]
        if not self.index:
            raise RuntimeError("no transitions are available in the selected episodes")

    def _inspect_episode(
        self,
        path: Path,
        *,
        expected_state_dim: int | None,
        expected_action_dim: int | None,
    ) -> MultimodalEpisodeRecord:
        with h5py.File(path, "r") as file:
            profile = str(file.attrs.get("schema_profile", ""))
            version = str(file.attrs.get("schema_version", ""))
            if profile != "wam_multimodal":
                raise ValueError(
                    f"{path} schema profile {profile!r} is not 'wam_multimodal'"
                )
            if version != MULTIMODAL_WAM_SCHEMA_VERSION:
                raise ValueError(
                    f"{path} schema version {version!r}; expected "
                    f"{MULTIMODAL_WAM_SCHEMA_VERSION!r}"
                )
            raw_camera_order = file.attrs.get("camera_order_json")
            if raw_camera_order is None:
                raise ValueError(f"{path} is missing camera_order_json")
            try:
                file_camera_order = tuple(json.loads(str(raw_camera_order)))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path} has invalid camera_order_json") from exc
            if file_camera_order != self.camera_order:
                raise ValueError(
                    f"{path} camera order {file_camera_order} does not match "
                    f"requested {self.camera_order}"
                )

            fps = float(file.attrs.get("fps", np.nan))
            if not np.isfinite(fps) or fps <= 0.0:
                raise ValueError(f"{path} has invalid control fps {fps!r}")
            num_steps = int(file.attrs.get("num_steps", -1))
            if num_steps <= 0:
                raise ValueError(f"{path} has no completed transitions")

            required = [
                "data/timestamp",
                "data/frame_index",
                "data/episode_index",
                "data/seed",
                "data/task/text",
                "data/task/id",
                "data/observation/state",
                "data/action/commanded",
                "data/action/executed",
                "data/next_observation/state",
                "data/event/visual_signal_active",
                "data/event/visual_signal_onset_step",
                "data/event/visual_signal_kind",
                "data/event/rendered_cue_variant",
                "data/reward",
                "data/terminated",
                "data/truncated",
                "data/done",
                "data/success",
                "data/failure",
                "data/failure_reason",
                "data/schema_version",
                "data/behavior_id",
                "data/environment_config",
                "data/randomization_config",
            ]
            for camera in self.camera_order:
                required.extend(
                    (
                        f"data/observation/images/{camera}",
                        f"data/observation/image_timestamp/{camera}",
                        f"data/observation/image_state_timestamp/{camera}",
                        f"data/observation/image_frame_index/{camera}",
                        f"data/next_observation/images/{camera}",
                        f"data/next_observation/image_timestamp/{camera}",
                        f"data/next_observation/image_state_timestamp/{camera}",
                        f"data/next_observation/image_frame_index/{camera}",
                        f"data/camera/intrinsics/{camera}",
                        f"data/camera/extrinsics/{camera}",
                        f"data/camera/resolution/{camera}",
                        f"data/next_camera/intrinsics/{camera}",
                        f"data/next_camera/extrinsics/{camera}",
                        f"data/next_camera/resolution/{camera}",
                    )
                )
            missing = [name for name in required if name not in file]
            if missing:
                raise KeyError(f"{path} is missing required datasets: {missing}")

            all_paths: list[str] = []
            file.visit(all_paths.append)
            privileged = [name for name in all_paths if "privileged" in name.lower()]
            if privileged:
                raise ValueError(
                    f"{path} contains forbidden privileged fields: {privileged[:5]}"
                )

            state = _dataset(file, "data/observation/state")
            next_state = _dataset(file, "data/next_observation/state")
            commanded = _dataset(file, "data/action/commanded")
            executed = _dataset(file, "data/action/executed")
            _require_dtype(state, np.float32, path)
            _require_dtype(next_state, np.float32, path)
            _require_dtype(commanded, np.float32, path)
            _require_dtype(executed, np.float32, path)
            _require_dtype(
                _dataset(file, "data/event/visual_signal_active"), np.bool_, path
            )
            _require_dtype(
                _dataset(file, "data/event/visual_signal_onset_step"), np.int64, path
            )
            _require_dtype(
                _dataset(file, "data/event/rendered_cue_variant"), np.int64, path
            )
            if state.ndim != 2 or next_state.shape != state.shape:
                raise ValueError(f"{path} state/next-state shapes are inconsistent")
            if commanded.ndim != 2 or executed.shape != commanded.shape:
                raise ValueError(f"{path} commanded/executed action shapes are inconsistent")
            if state.shape[0] != num_steps or commanded.shape[0] != num_steps:
                raise ValueError(f"{path} state/action length does not match num_steps")
            state_width = int(state.shape[1])
            action_width = int(commanded.shape[1])
            if expected_state_dim is not None and state_width != expected_state_dim:
                raise ValueError(
                    f"{path} state dimension {state_width} does not match "
                    f"{expected_state_dim}"
                )
            if expected_action_dim is not None and action_width != expected_action_dim:
                raise ValueError(
                    f"{path} action dimension {action_width} does not match "
                    f"{expected_action_dim}"
                )

            for name in required:
                dataset = _dataset(file, name)
                if dataset.shape[0] != num_steps:
                    raise ValueError(
                        f"{path} field {name!r} length {dataset.shape[0]} does not "
                        f"match num_steps {num_steps}"
                    )

            timestamps = np.asarray(file["data/timestamp"][:], dtype=np.float64)
            frame_indices = np.asarray(file["data/frame_index"][:], dtype=np.int64)
            if not np.isfinite(timestamps).all() or (
                num_steps > 1 and not np.all(np.diff(timestamps) > 0.0)
            ):
                raise ValueError(f"{path} transition timestamps are not finite/strict")
            if not np.array_equal(frame_indices, np.arange(num_steps, dtype=np.int64)):
                raise ValueError(f"{path} frame_index must be contiguous from zero")

            episode_index = _constant_integer(
                file["data/episode_index"], "episode_index", path
            )
            seed = _constant_integer(file["data/seed"], "seed", path)
            if episode_index != int(file.attrs.get("episode_index", -1)):
                raise ValueError(f"{path} episode_index field/attribute mismatch")
            if seed != int(file.attrs.get("seed", -1)):
                raise ValueError(f"{path} seed field/attribute mismatch")
            task_text = _constant_string(file["data/task/text"], "task.text", path)
            task_id = _constant_string(file["data/task/id"], "task.id", path)
            if not task_text or not task_id:
                raise ValueError(f"{path} task text/id cannot be empty")
            schema_versions = _string_values(file["data/schema_version"])
            if not np.all(schema_versions == MULTIMODAL_WAM_SCHEMA_VERSION):
                raise ValueError(f"{path} per-transition schema version drifted")
            _constant_string(file["data/behavior_id"], "behavior_id", path)
            for name in ("environment_config", "randomization_config"):
                value = _constant_string(file[f"data/{name}"], name, path)
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path} {name} is not valid JSON") from exc
                if not isinstance(parsed, dict):
                    raise ValueError(f"{path} {name} must encode a JSON object")

            image_shape: tuple[int, int, int] | None = None
            next_state_timestamps = _next_state_timestamps(timestamps, fps)
            for camera in self.camera_order:
                current_image = _dataset(
                    file, f"data/observation/images/{camera}"
                )
                next_image = _dataset(
                    file, f"data/next_observation/images/{camera}"
                )
                _require_dtype(current_image, np.uint8, path)
                _require_dtype(next_image, np.uint8, path)
                if (
                    current_image.ndim != 4
                    or current_image.shape[-1] != 3
                    or next_image.shape != current_image.shape
                ):
                    raise ValueError(
                        f"{path} camera {camera!r} must contain matching uint8 HWC RGB"
                    )
                camera_shape = tuple(int(value) for value in current_image.shape[1:])
                if image_shape is None:
                    image_shape = camera_shape
                elif camera_shape != image_shape:
                    raise ValueError(
                        f"{path} camera {camera!r} shape {camera_shape} does not match "
                        f"{image_shape}"
                    )

                intrinsics = _dataset(file, f"data/camera/intrinsics/{camera}")
                extrinsics = _dataset(file, f"data/camera/extrinsics/{camera}")
                resolution = _dataset(file, f"data/camera/resolution/{camera}")
                next_intrinsics = _dataset(
                    file, f"data/next_camera/intrinsics/{camera}"
                )
                next_extrinsics = _dataset(
                    file, f"data/next_camera/extrinsics/{camera}"
                )
                next_resolution = _dataset(
                    file, f"data/next_camera/resolution/{camera}"
                )
                for calibration in (intrinsics, next_intrinsics):
                    _require_dtype(calibration, np.float32, path)
                    if calibration.shape != (num_steps, 3, 3):
                        raise ValueError(
                            f"{path} camera {camera!r} intrinsics must be [T,3,3]"
                        )
                for calibration in (extrinsics, next_extrinsics):
                    _require_dtype(calibration, np.float32, path)
                    if calibration.shape != (num_steps, 4, 4):
                        raise ValueError(
                            f"{path} camera {camera!r} extrinsics must be [T,4,4]"
                        )
                for calibration in (resolution, next_resolution):
                    _require_dtype(calibration, np.int64, path)
                    if calibration.shape != (num_steps, 2):
                        raise ValueError(
                            f"{path} camera {camera!r} resolution must be [T,2]"
                        )
                current_intrinsics = np.asarray(intrinsics[:], dtype=np.float32)
                current_extrinsics = np.asarray(extrinsics[:], dtype=np.float32)
                current_resolution = np.asarray(resolution[:], dtype=np.int64)
                next_intrinsics_values = np.asarray(
                    next_intrinsics[:], dtype=np.float32
                )
                next_extrinsics_values = np.asarray(
                    next_extrinsics[:], dtype=np.float32
                )
                next_resolution_values = np.asarray(
                    next_resolution[:], dtype=np.int64
                )
                if not all(
                    np.isfinite(values).all()
                    for values in (
                        current_intrinsics,
                        current_extrinsics,
                        next_intrinsics_values,
                        next_extrinsics_values,
                    )
                ):
                    raise ValueError(
                        f"{path} camera {camera!r} calibration contains NaN/Inf"
                    )
                expected_resolution = np.asarray(camera_shape[:2], dtype=np.int64)
                if not np.all(current_resolution == expected_resolution) or not np.all(
                    next_resolution_values == expected_resolution
                ):
                    raise ValueError(
                        f"{path} camera {camera!r} resolution must be [height,width]"
                    )

                current_frame_indices = np.asarray(
                    file[f"data/observation/image_frame_index/{camera}"][:],
                    dtype=np.int64,
                )
                next_frame_indices = np.asarray(
                    file[f"data/next_observation/image_frame_index/{camera}"][:],
                    dtype=np.int64,
                )

                self._validate_camera_timing(
                    path,
                    camera,
                    timestamps,
                    next_state_timestamps,
                    current_image_timestamps=np.asarray(
                        file[f"data/observation/image_timestamp/{camera}"][:],
                        dtype=np.float64,
                    ),
                    current_image_state_timestamps=np.asarray(
                        file[f"data/observation/image_state_timestamp/{camera}"][:],
                        dtype=np.float64,
                    ),
                    current_image_frame_indices=current_frame_indices,
                    next_image_timestamps=np.asarray(
                        file[f"data/next_observation/image_timestamp/{camera}"][:],
                        dtype=np.float64,
                    ),
                    next_image_state_timestamps=np.asarray(
                        file[f"data/next_observation/image_state_timestamp/{camera}"][:],
                        dtype=np.float64,
                    ),
                    next_image_frame_indices=next_frame_indices,
                )
                self._validate_camera_calibration(
                    path,
                    camera,
                    current_frame_indices=current_frame_indices,
                    next_frame_indices=next_frame_indices,
                    current_intrinsics=current_intrinsics,
                    current_extrinsics=current_extrinsics,
                    current_resolution=current_resolution,
                    next_intrinsics=next_intrinsics_values,
                    next_extrinsics=next_extrinsics_values,
                    next_resolution=next_resolution_values,
                )

            assert image_shape is not None
            return MultimodalEpisodeRecord(
                path=path,
                num_steps=num_steps,
                seed=seed,
                episode_index=episode_index,
                state_dim=state_width,
                action_dim=action_width,
                image_shape=image_shape,
                fps=fps,
                task_text=task_text,
                task_id=task_id,
            )

    def _validate_camera_timing(
        self,
        path: Path,
        camera: str,
        state_timestamps: np.ndarray,
        next_state_timestamps: np.ndarray,
        *,
        current_image_timestamps: np.ndarray,
        current_image_state_timestamps: np.ndarray,
        current_image_frame_indices: np.ndarray,
        next_image_timestamps: np.ndarray,
        next_image_state_timestamps: np.ndarray,
        next_image_frame_indices: np.ndarray,
    ) -> None:
        timestamp_arrays = (
            current_image_timestamps,
            current_image_state_timestamps,
            next_image_timestamps,
            next_image_state_timestamps,
        )
        if any(not np.isfinite(values).all() for values in timestamp_arrays):
            raise ValueError(f"{path} camera {camera!r} timing contains NaN/Inf")
        if any(
            values.size > 1 and np.any(np.diff(values) < 0.0)
            for values in timestamp_arrays
        ):
            raise ValueError(f"{path} camera {camera!r} timing is not monotonic")
        for values in (current_image_frame_indices, next_image_frame_indices):
            if np.any(values < 0) or (values.size > 1 and np.any(np.diff(values) < 0)):
                raise ValueError(
                    f"{path} camera {camera!r} frame indices are invalid"
                )

        tolerance = self.causal_tolerance_seconds
        if np.any(current_image_timestamps > state_timestamps + tolerance) or np.any(
            current_image_state_timestamps > state_timestamps + tolerance
        ):
            raise ValueError(f"{path} camera {camera!r} current RGB leaks the future")
        if np.any(next_image_timestamps > next_state_timestamps + tolerance) or np.any(
            next_image_state_timestamps > next_state_timestamps + tolerance
        ):
            raise ValueError(f"{path} camera {camera!r} next RGB leaks the future")

        current_age = state_timestamps - current_image_timestamps
        next_age = next_state_timestamps - next_image_timestamps
        if np.any(current_age < -tolerance) or np.any(next_age < -tolerance):
            raise ValueError(f"{path} camera {camera!r} has negative frame age")
        maximum = self.max_frame_age_seconds + tolerance
        if np.any(current_age > maximum) or np.any(next_age > maximum):
            raise ValueError(
                f"{path} camera {camera!r} frame age exceeds "
                f"{self.max_frame_age_seconds:.6f}s"
            )

        if self.max_sensor_state_skew_seconds is not None:
            skew_limit = self.max_sensor_state_skew_seconds + tolerance
            current_skew = np.abs(
                current_image_timestamps - current_image_state_timestamps
            )
            next_skew = np.abs(next_image_timestamps - next_image_state_timestamps)
            if np.any(current_skew > skew_limit) or np.any(next_skew > skew_limit):
                raise ValueError(
                    f"{path} camera {camera!r} sensor/state skew exceeds "
                    f"{self.max_sensor_state_skew_seconds:.6f}s"
                )

        if state_timestamps.size > 1:
            if not np.array_equal(
                next_image_frame_indices[:-1], current_image_frame_indices[1:]
            ):
                raise ValueError(
                    f"{path} camera {camera!r} next/current frame references disagree"
                )
            if not np.allclose(
                next_image_timestamps[:-1],
                current_image_timestamps[1:],
                rtol=0.0,
                atol=tolerance,
            ) or not np.allclose(
                next_image_state_timestamps[:-1],
                current_image_state_timestamps[1:],
                rtol=0.0,
                atol=tolerance,
            ):
                raise ValueError(
                    f"{path} camera {camera!r} next/current timestamps disagree"
                )

    @staticmethod
    def _validate_camera_calibration(
        path: Path,
        camera: str,
        *,
        current_frame_indices: np.ndarray,
        next_frame_indices: np.ndarray,
        current_intrinsics: np.ndarray,
        current_extrinsics: np.ndarray,
        current_resolution: np.ndarray,
        next_intrinsics: np.ndarray,
        next_extrinsics: np.ndarray,
        next_resolution: np.ndarray,
    ) -> None:
        """Require one immutable calibration snapshot per captured RGB frame."""

        snapshots: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, str, int]] = {}
        streams = (
            (
                "current",
                current_frame_indices,
                current_intrinsics,
                current_extrinsics,
                current_resolution,
            ),
            (
                "next",
                next_frame_indices,
                next_intrinsics,
                next_extrinsics,
                next_resolution,
            ),
        )
        for phase, frame_indices, intrinsics, extrinsics, resolutions in streams:
            for row, frame_index in enumerate(frame_indices.tolist()):
                snapshot = snapshots.get(int(frame_index))
                if snapshot is None:
                    snapshots[int(frame_index)] = (
                        intrinsics[row].copy(),
                        extrinsics[row].copy(),
                        resolutions[row].copy(),
                        phase,
                        row,
                    )
                    continue
                prior_intrinsics, prior_extrinsics, prior_resolution, prior_phase, prior_row = snapshot
                if not (
                    np.array_equal(intrinsics[row], prior_intrinsics)
                    and np.array_equal(extrinsics[row], prior_extrinsics)
                    and np.array_equal(resolutions[row], prior_resolution)
                ):
                    raise ValueError(
                        f"{path} camera {camera!r} calibration sample-hold "
                        f"disagrees for frame {frame_index} between "
                        f"{prior_phase}[{prior_row}] and {phase}[{row}]"
                    )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str]:
        sample = self.index[item]
        record = self.records[sample.file_index]
        t = sample.decision_t
        history_start = max(0, t - self.history_horizon + 1)
        history_count = t - history_start + 1
        history_offset = self.history_horizon - history_count
        forecast_end = min(record.num_steps, t + self.forecast_horizon)
        forecast_count = forecast_end - t
        camera_count = len(self.camera_order)
        height, width, channels = record.image_shape
        assert channels == 3

        states = torch.zeros(self.history_horizon, self.state_dim)
        past_actions = torch.zeros(
            max(self.history_horizon - 1, 0), self.action_dim
        )
        past_commanded_actions = torch.zeros_like(past_actions)
        valid_mask = torch.zeros(self.history_horizon, dtype=torch.bool)
        past_action_mask = torch.zeros(
            max(self.history_horizon - 1, 0), dtype=torch.bool
        )
        candidate_actions = torch.zeros(self.forecast_horizon, self.action_dim)
        executed_actions = torch.zeros_like(candidate_actions)
        target_states = torch.zeros(self.forecast_horizon, self.state_dim)
        forecast_mask = torch.arange(self.forecast_horizon) < forecast_count

        images = torch.zeros(
            self.history_horizon,
            camera_count,
            channels,
            height,
            width,
            dtype=torch.uint8,
        )
        target_images = torch.zeros(
            self.forecast_horizon,
            camera_count,
            channels,
            height,
            width,
            dtype=torch.uint8,
        )
        image_valid_mask = torch.zeros(
            self.history_horizon, camera_count, dtype=torch.bool
        )
        target_image_valid_mask = torch.zeros(
            self.forecast_horizon, camera_count, dtype=torch.bool
        )
        state_timestamps = torch.zeros(self.history_horizon, dtype=torch.float64)
        target_state_timestamps = torch.zeros(
            self.forecast_horizon, dtype=torch.float64
        )
        state_frame_indices = torch.full(
            (self.history_horizon,), -1, dtype=torch.int64
        )
        image_timestamps = torch.zeros(
            self.history_horizon, camera_count, dtype=torch.float64
        )
        image_state_timestamps = torch.zeros_like(image_timestamps)
        image_frame_indices = torch.full(
            (self.history_horizon, camera_count), -1, dtype=torch.int64
        )
        image_age_seconds = torch.zeros_like(image_timestamps)
        target_image_timestamps = torch.zeros(
            self.forecast_horizon, camera_count, dtype=torch.float64
        )
        target_image_state_timestamps = torch.zeros_like(target_image_timestamps)
        target_image_frame_indices = torch.full(
            (self.forecast_horizon, camera_count), -1, dtype=torch.int64
        )
        target_image_age_seconds = torch.zeros_like(target_image_timestamps)
        camera_intrinsics = torch.zeros(
            self.history_horizon, camera_count, 3, 3, dtype=torch.float32
        )
        camera_extrinsics = torch.zeros(
            self.history_horizon, camera_count, 4, 4, dtype=torch.float32
        )
        camera_resolution = torch.zeros(
            self.history_horizon, camera_count, 2, dtype=torch.int64
        )
        target_camera_intrinsics = torch.zeros(
            self.forecast_horizon, camera_count, 3, 3, dtype=torch.float32
        )
        target_camera_extrinsics = torch.zeros(
            self.forecast_horizon, camera_count, 4, 4, dtype=torch.float32
        )
        target_camera_resolution = torch.zeros(
            self.forecast_horizon, camera_count, 2, dtype=torch.int64
        )

        with self._open_hdf5(record.path) as file:
            states[history_offset:].copy_(
                torch.from_numpy(
                    np.asarray(
                        file["data/observation/state"][history_start : t + 1],
                        dtype=np.float32,
                    )
                )
            )
            valid_mask[history_offset:] = True
            if history_count > 1:
                action_slice = slice(history_start, t)
                past_actions[history_offset:].copy_(
                    torch.from_numpy(
                        np.asarray(
                            file["data/action/executed"][action_slice],
                            dtype=np.float32,
                        )
                    )
                )
                past_commanded_actions[history_offset:].copy_(
                    torch.from_numpy(
                        np.asarray(
                            file["data/action/commanded"][action_slice],
                            dtype=np.float32,
                        )
                    )
                )
                past_action_mask[history_offset:] = True

            forecast_slice = slice(t, forecast_end)
            candidate_actions[:forecast_count].copy_(
                torch.from_numpy(
                    np.asarray(
                        file["data/action/commanded"][forecast_slice],
                        dtype=np.float32,
                    )
                )
            )
            executed_actions[:forecast_count].copy_(
                torch.from_numpy(
                    np.asarray(
                        file["data/action/executed"][forecast_slice],
                        dtype=np.float32,
                    )
                )
            )
            target_states[:forecast_count].copy_(
                torch.from_numpy(
                    np.asarray(
                        file["data/next_observation/state"][forecast_slice],
                        dtype=np.float32,
                    )
                )
            )

            history_timestamp_values = np.asarray(
                file["data/timestamp"][history_start : t + 1], dtype=np.float64
            )
            state_timestamps[history_offset:].copy_(
                torch.from_numpy(history_timestamp_values)
            )
            state_frame_indices[history_offset:].copy_(
                torch.from_numpy(
                    np.asarray(
                        file["data/frame_index"][history_start : t + 1],
                        dtype=np.int64,
                    )
                )
            )
            target_timestamp_values = _window_next_state_timestamps(
                file, t, forecast_end, record
            )
            target_state_timestamps[:forecast_count].copy_(
                torch.from_numpy(target_timestamp_values)
            )

            for camera_index, camera in enumerate(self.camera_order):
                current = np.asarray(
                    file[f"data/observation/images/{camera}"][history_start : t + 1],
                    dtype=np.uint8,
                ).transpose(0, 3, 1, 2)
                images[history_offset:, camera_index].copy_(torch.from_numpy(current))
                target = np.asarray(
                    file[f"data/next_observation/images/{camera}"][forecast_slice],
                    dtype=np.uint8,
                ).transpose(0, 3, 1, 2)
                target_images[:forecast_count, camera_index].copy_(
                    torch.from_numpy(target)
                )
                camera_intrinsics[history_offset:, camera_index].copy_(
                    torch.from_numpy(
                        np.asarray(
                            file[f"data/camera/intrinsics/{camera}"][
                                history_start : t + 1
                            ],
                            dtype=np.float32,
                        )
                    )
                )
                camera_extrinsics[history_offset:, camera_index].copy_(
                    torch.from_numpy(
                        np.asarray(
                            file[f"data/camera/extrinsics/{camera}"][
                                history_start : t + 1
                            ],
                            dtype=np.float32,
                        )
                    )
                )
                camera_resolution[history_offset:, camera_index].copy_(
                    torch.from_numpy(
                        np.asarray(
                            file[f"data/camera/resolution/{camera}"][
                                history_start : t + 1
                            ],
                            dtype=np.int64,
                        )
                    )
                )
                target_camera_intrinsics[:forecast_count, camera_index].copy_(
                    torch.from_numpy(
                        np.asarray(
                            file[f"data/next_camera/intrinsics/{camera}"][
                                forecast_slice
                            ],
                            dtype=np.float32,
                        )
                    )
                )
                target_camera_extrinsics[:forecast_count, camera_index].copy_(
                    torch.from_numpy(
                        np.asarray(
                            file[f"data/next_camera/extrinsics/{camera}"][
                                forecast_slice
                            ],
                            dtype=np.float32,
                        )
                    )
                )
                target_camera_resolution[:forecast_count, camera_index].copy_(
                    torch.from_numpy(
                        np.asarray(
                            file[f"data/next_camera/resolution/{camera}"][
                                forecast_slice
                            ],
                            dtype=np.int64,
                        )
                    )
                )
                image_valid_mask[history_offset:, camera_index] = True
                target_image_valid_mask[:forecast_count, camera_index] = True

                current_image_ts = np.asarray(
                    file[f"data/observation/image_timestamp/{camera}"][
                        history_start : t + 1
                    ],
                    dtype=np.float64,
                )
                current_image_state_ts = np.asarray(
                    file[f"data/observation/image_state_timestamp/{camera}"][
                        history_start : t + 1
                    ],
                    dtype=np.float64,
                )
                current_image_indices = np.asarray(
                    file[f"data/observation/image_frame_index/{camera}"][
                        history_start : t + 1
                    ],
                    dtype=np.int64,
                )
                image_timestamps[history_offset:, camera_index].copy_(
                    torch.from_numpy(current_image_ts)
                )
                image_state_timestamps[history_offset:, camera_index].copy_(
                    torch.from_numpy(current_image_state_ts)
                )
                image_frame_indices[history_offset:, camera_index].copy_(
                    torch.from_numpy(current_image_indices)
                )
                image_age_seconds[history_offset:, camera_index].copy_(
                    torch.from_numpy(history_timestamp_values - current_image_ts)
                )

                next_image_ts = np.asarray(
                    file[f"data/next_observation/image_timestamp/{camera}"][
                        forecast_slice
                    ],
                    dtype=np.float64,
                )
                next_image_state_ts = np.asarray(
                    file[f"data/next_observation/image_state_timestamp/{camera}"][
                        forecast_slice
                    ],
                    dtype=np.float64,
                )
                next_image_indices = np.asarray(
                    file[f"data/next_observation/image_frame_index/{camera}"][
                        forecast_slice
                    ],
                    dtype=np.int64,
                )
                target_image_timestamps[:forecast_count, camera_index].copy_(
                    torch.from_numpy(next_image_ts)
                )
                target_image_state_timestamps[:forecast_count, camera_index].copy_(
                    torch.from_numpy(next_image_state_ts)
                )
                target_image_frame_indices[:forecast_count, camera_index].copy_(
                    torch.from_numpy(next_image_indices)
                )
                target_image_age_seconds[:forecast_count, camera_index].copy_(
                    torch.from_numpy(target_timestamp_values - next_image_ts)
                )

            scalar_values = {
                name: np.asarray(file[f"data/{source}"][forecast_slice], dtype=np.float32)
                for name, source in (
                    ("rewards", "reward"),
                    ("terminated", "terminated"),
                    ("truncated", "truncated"),
                    ("dones", "done"),
                    ("successes", "success"),
                    ("failures", "failure"),
                )
            }

        result: dict[str, torch.Tensor | str] = {
            "states": states,
            "past_actions": past_actions,
            "past_commanded_actions": past_commanded_actions,
            "valid_mask": valid_mask,
            "past_action_mask": past_action_mask,
            "candidate_actions": candidate_actions,
            "executed_actions": executed_actions,
            "target_states": target_states,
            "forecast_mask": forecast_mask,
            "images": images,
            "target_images": target_images,
            "image_valid_mask": image_valid_mask,
            "target_image_valid_mask": target_image_valid_mask,
            "state_timestamps": state_timestamps,
            "target_state_timestamps": target_state_timestamps,
            "state_frame_indices": state_frame_indices,
            "image_timestamps": image_timestamps,
            "image_state_timestamps": image_state_timestamps,
            "image_frame_indices": image_frame_indices,
            "image_age_seconds": image_age_seconds,
            "target_image_timestamps": target_image_timestamps,
            "target_image_state_timestamps": target_image_state_timestamps,
            "target_image_frame_indices": target_image_frame_indices,
            "target_image_age_seconds": target_image_age_seconds,
            "camera_intrinsics": camera_intrinsics,
            "camera_extrinsics": camera_extrinsics,
            "camera_resolution": camera_resolution,
            "target_camera_intrinsics": target_camera_intrinsics,
            "target_camera_extrinsics": target_camera_extrinsics,
            "target_camera_resolution": target_camera_resolution,
            "episode_index": torch.tensor(record.episode_index, dtype=torch.int64),
            "episode_seed": torch.tensor(record.seed, dtype=torch.int64),
            "decision_t": torch.tensor(t, dtype=torch.int64),
            "task_text": record.task_text,
            "task_id": record.task_id,
        }
        for name, values in scalar_values.items():
            padded = torch.zeros(self.forecast_horizon, 1)
            padded[:forecast_count, 0].copy_(torch.from_numpy(values.reshape(-1)))
            result[name] = padded
        return result

    @contextmanager
    def _open_hdf5(self, path: Path) -> Iterator[h5py.File]:
        if self.hdf5_cache_size == 0:
            with h5py.File(path, "r") as file:
                yield file
            return
        if self._cache_pid != os.getpid():
            self.close()
            self._cache_pid = os.getpid()
        key = str(path.resolve())
        file = self._cache.pop(key, None)
        if file is None:
            file = h5py.File(path, "r")
        self._cache[key] = file
        while len(self._cache) > self.hdf5_cache_size:
            _, stale = self._cache.popitem(last=False)
            stale.close()
        yield file

    def close(self) -> None:
        for file in self._cache.values():
            file.close()
        self._cache.clear()

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        state["_cache_pid"] = os.getpid()
        return state

    def __del__(self) -> None:
        if hasattr(self, "_cache"):
            self.close()


def _dataset(file: h5py.File, path: str) -> h5py.Dataset:
    value = file.get(path)
    if not isinstance(value, h5py.Dataset):
        raise KeyError(f"required HDF5 dataset {path!r} is missing")
    return value


def _require_dtype(dataset: h5py.Dataset, dtype: np.dtype | type, path: Path) -> None:
    expected = np.dtype(dtype)
    if dataset.dtype != expected:
        raise TypeError(
            f"{path} field {dataset.name!r} has dtype {dataset.dtype}, expected {expected}"
        )


def _string_values(dataset: h5py.Dataset) -> np.ndarray:
    return np.asarray(dataset.asstr()[:], dtype=str).reshape(-1)


def _constant_string(dataset: h5py.Dataset, name: str, path: Path) -> str:
    values = _string_values(dataset)
    if values.size == 0 or np.any(values != values[0]):
        raise ValueError(f"{path} {name} changes inside the episode")
    return str(values[0])


def _constant_integer(dataset: h5py.Dataset, name: str, path: Path) -> int:
    values = np.asarray(dataset[:], dtype=np.int64).reshape(-1)
    if values.size == 0 or np.any(values != values[0]):
        raise ValueError(f"{path} {name} changes inside the episode")
    return int(values[0])


def _next_state_timestamps(timestamps: np.ndarray, fps: float) -> np.ndarray:
    result = np.empty_like(timestamps)
    if timestamps.size > 1:
        result[:-1] = timestamps[1:]
    result[-1] = timestamps[-1] + 1.0 / fps
    return result


def _window_next_state_timestamps(
    file: h5py.File,
    start: int,
    stop: int,
    record: MultimodalEpisodeRecord,
) -> np.ndarray:
    count = stop - start
    result = np.empty(count, dtype=np.float64)
    available_stop = min(stop + 1, record.num_steps)
    following = np.asarray(
        file["data/timestamp"][start + 1 : available_stop], dtype=np.float64
    )
    result[: following.size] = following
    if following.size < count:
        current = np.asarray(file["data/timestamp"][start:stop], dtype=np.float64)
        result[following.size :] = current[following.size :] + 1.0 / record.fps
    return result


__all__ = [
    "MultimodalEpisodeRecord",
    "MultimodalSequenceDataset",
    "MultimodalSequenceIndex",
]
