"""Manifest-bound, leakage-resistant windows for Phase M1.

This module deliberately does not discover or randomly split episode files.
Every sample is selected from the split recorded in the canonical M0 manifest,
after auditing cue pairs and template/scene/object isolation.  RGB tensors use
``[time, camera, channel, height, width]`` uint8 layout.

The normal ``__getitem__`` path contains deployable observations and supervised
trajectory targets only.  Event labels and sample lineage are available through
explicit offline-only methods so they cannot silently enter the policy input.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Callable, Collection, Iterator, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


M1_SCHEMA_PROFILE = "wam_multimodal"
M1_SCHEMA_VERSION = "wam.multimodal/1.1"
M1_MANIFEST_FORMAT = "wam.multimodal.m0.dataset/2"
CANONICAL_CAMERA_ORDER = ("fixed", "robot_0_camera", "robot_1_camera")
CANONICAL_SPLITS = ("train", "validation", "test")
CANONICAL_CUE_VARIANTS = (0, 1)
M1_PROBE_HORIZON = 8
M1_CAUSAL_PAIR_HORIZON = 8
M1_CONTROLLED_ACTION_DIMS = (0, 1, 2, 4, 5, 6)
M1_STATE_CAUSAL_HISTORY_WIDTH = 32
M1_STATE_CAUSAL_PAST_ACTIONS = 3
M1_STATE_CAUSAL_STATES = M1_STATE_CAUSAL_PAST_ACTIONS + 1
M1_STATE_CAUSAL_GAP = 1
M1_STATE_CAUSAL_MIN_ACTION_DELTA = 1e-3
M1_STATE_CAUSAL_LATERAL_ACTION_DIMS = (0, 4)
M1_STATE_CAUSAL_ZERO_DELTA_ACTION_DIMS = (1, 2, 5, 6)
M1_STATE_CAUSAL_ROBOT_X_STATE_DIMS = (0, 11)
M1_STATE_CAUSAL_FEEDBACK_EQUALITY_ATOL = 1e-7

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SPLIT_ALIASES = {"val": "validation", "valid": "validation"}


@dataclass(frozen=True)
class M1ManifestEpisode:
    """One immutable episode entry resolved from the canonical manifest."""

    path: Path
    relative_path: str
    hdf5_sha256: str
    episode_index: int
    seed: int
    physical_seed: int
    cue_id: int
    split: str
    task_id: str
    task_text: str
    template_id: str
    scene_id: str
    object_combination_id: str
    behavior_id: str
    num_steps: int


@dataclass(frozen=True)
class M1HDF5Metadata:
    """Shape metadata collected while validating one episode file."""

    state_dim: int
    action_dim: int
    image_shape_hwc: tuple[int, int, int]
    control_hz: float


@dataclass(frozen=True)
class M1SampleLineage:
    """Offline audit identity; intentionally never returned by ``__getitem__``."""

    path: Path
    decision_t: int


@dataclass(frozen=True)
class _M1WindowIndex:
    record_index: int
    decision_t: int
    visual_rows: tuple[int, ...]
    decision_window: bool


@dataclass(frozen=True)
class _M1CausalPairIndex:
    """Offline pair identity; never exposed as a deployable model input."""

    record_indices: tuple[int, int]
    decision_t: int
    visual_rows: tuple[tuple[int, ...], tuple[int, ...]]
    audit_sample_ids: tuple[str, str]


@dataclass(frozen=True)
class _M1StateCausalPairIndex:
    """Offline state-pair identity; only opaque hashes leave the dataset."""

    record_index: int
    decision_ts: tuple[int, int]
    visual_rows: tuple[tuple[int, ...], tuple[int, ...]]
    audit_sample_ids: tuple[str, str]


class M1ManifestIndex:
    """Strict parser and split auditor for the canonical multimodal manifest.

    Use :meth:`from_path` rather than constructing the class directly.  Strict
    HDF5 SHA and attribute verification is enabled by default; callers may reuse
    one instance across train/validation/test datasets to avoid hashing twice.
    """

    def __init__(
        self,
        *,
        manifest_path: Path,
        manifest_sha256: str,
        raw_manifest: Mapping[str, Any],
        episodes: Sequence[M1ManifestEpisode],
        hdf5_metadata: Mapping[int, M1HDF5Metadata],
        hdf5_sha256_verified: bool,
        hdf5_contract_verified: bool,
    ) -> None:
        self.manifest_path = manifest_path
        self.manifest_sha256 = manifest_sha256
        self.raw_manifest = MappingProxyType(dict(raw_manifest))
        self.episodes = tuple(episodes)
        self.camera_order = tuple(str(x) for x in raw_manifest["camera_order"])
        self.task_order = tuple(str(x) for x in raw_manifest["tasks"])
        self.task_to_index = MappingProxyType(
            {task_id: index for index, task_id in enumerate(self.task_order)}
        )
        self.control_hz = float(raw_manifest["control_hz"])
        self.image_hz = float(raw_manifest["image_hz"])
        self.hdf5_metadata = MappingProxyType(dict(hdf5_metadata))
        self.hdf5_sha256_verified = bool(hdf5_sha256_verified)
        self.hdf5_contract_verified = bool(hdf5_contract_verified)

        grouped: dict[str, list[M1ManifestEpisode]] = {
            split: [] for split in CANONICAL_SPLITS
        }
        for episode in self.episodes:
            grouped[episode.split].append(episode)
        self._by_split = {
            split: tuple(sorted(items, key=lambda item: item.episode_index))
            for split, items in grouped.items()
        }
        self._split_summaries = {
            split: _build_split_summary(
                split=split,
                records=records,
                schema_version=M1_SCHEMA_VERSION,
                camera_order=self.camera_order,
                task_order=self.task_order,
            )
            for split, records in self._by_split.items()
        }

    @classmethod
    def from_path(
        cls,
        manifest_path: str | Path,
        *,
        verify_hdf5_sha256: bool = True,
        verify_hdf5_contract: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> "M1ManifestIndex":
        """Load, audit, and optionally byte-verify a canonical manifest."""

        path = Path(manifest_path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"manifest path is not a file: {path}")
        manifest_bytes = path.read_bytes()
        try:
            raw = json.loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON manifest: {path}") from exc
        if not isinstance(raw, dict):
            raise ValueError("manifest root must be a JSON object")

        _validate_manifest_header(raw)
        records = _parse_manifest_episodes(raw, manifest_path=path)
        _audit_manifest_splits(raw, records)

        metadata: dict[int, M1HDF5Metadata] = {}
        if progress_callback is not None:
            progress_callback(0, len(records))
        for record_index, record in enumerate(records, start=1):
            if verify_hdf5_sha256:
                actual_sha256 = _sha256_file(record.path)
                if actual_sha256 != record.hdf5_sha256:
                    raise ValueError(
                        f"{record.path} SHA256 {actual_sha256} does not match "
                        f"manifest {record.hdf5_sha256}"
                    )
            if verify_hdf5_contract:
                metadata[record.episode_index] = _inspect_hdf5_episode(
                    record,
                    camera_order=CANONICAL_CAMERA_ORDER,
                    control_hz=float(raw["control_hz"]),
                    resolution=tuple(int(x) for x in raw["resolution"]),
                )
            if progress_callback is not None:
                progress_callback(record_index, len(records))

        return cls(
            manifest_path=path,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            raw_manifest=raw,
            episodes=records,
            hdf5_metadata=metadata,
            hdf5_sha256_verified=verify_hdf5_sha256,
            hdf5_contract_verified=verify_hdf5_contract,
        )

    def records_for_split(self, split: str) -> tuple[M1ManifestEpisode, ...]:
        """Return manifest records for one normalized, pre-audited split."""

        normalized = _normalize_split(split)
        return self._by_split[normalized]

    def split_summary(self, split: str) -> dict[str, Any]:
        """Return a JSON-safe copy of the checkpoint-grade split summary."""

        normalized = _normalize_split(split)
        return json.loads(_canonical_json(self._split_summaries[normalized]))

    def split_summary_sha256(self, split: str) -> str:
        """Hash the canonical split summary for checkpoint binding."""

        return _json_sha256(self.split_summary(split))

    def checkpoint_lineage(self, split: str) -> dict[str, Any]:
        """Return minimal immutable lineage to store inside a checkpoint."""

        normalized = _normalize_split(split)
        return {
            "manifest_format": M1_MANIFEST_FORMAT,
            "schema_profile": M1_SCHEMA_PROFILE,
            "schema_version": M1_SCHEMA_VERSION,
            "manifest_sha256": self.manifest_sha256,
            "split": normalized,
            "split_summary_sha256": self.split_summary_sha256(normalized),
            "camera_order": list(self.camera_order),
            "task_order": list(self.task_order),
            "task_to_index": dict(self.task_to_index),
            "hdf5_sha256_verified": self.hdf5_sha256_verified,
            "hdf5_contract_verified": self.hdf5_contract_verified,
        }


class M1WindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Lazy, episode-safe Phase M1 training windows.

    State history is left-padded at episode start, while action/future targets
    are admitted only when their complete eight-control-step horizon exists.
    The two visual history entries are distinct captured frames, not repeated
    20 Hz sample-hold rows.  By default only the fixed camera is read.

    Event arrays are read once while constructing the private decision-window
    index.  They are never exposed by :meth:`__getitem__`; use
    :meth:`probe_labels` explicitly for offline H=8 probing.
    """

    SAMPLE_KEYS = frozenset(
        {
            "states",
            "state_valid_mask",
            "past_actions",
            "past_action_valid_mask",
            "images",
            "task_index",
            "action_targets",
            "future_states",
            "future_images",
            "future_image_novelty_mask",
            "future_horizons",
        }
    )

    def __init__(
        self,
        manifest: M1ManifestIndex | str | Path,
        *,
        split: str,
        state_history: int = 32,
        action_chunk: int = 8,
        cameras: Sequence[str] = ("fixed",),
        visual_history: int = 2,
        future_horizons: Sequence[int] = (1, 2, 4, 8),
        stride: int = 1,
        decision_window_radius: int = 8,
        hdf5_cache_size: int = 8,
        verify_hdf5_sha256: bool = True,
        verify_hdf5_contract: bool = True,
    ) -> None:
        if isinstance(manifest, M1ManifestIndex):
            manifest_index = manifest
        else:
            manifest_index = M1ManifestIndex.from_path(
                manifest,
                verify_hdf5_sha256=verify_hdf5_sha256,
                verify_hdf5_contract=verify_hdf5_contract,
            )

        for name, value in (
            ("state_history", state_history),
            ("action_chunk", action_chunk),
            ("visual_history", visual_history),
            ("stride", stride),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(decision_window_radius) < 0:
            raise ValueError("decision_window_radius cannot be negative")
        if int(hdf5_cache_size) < 0:
            raise ValueError("hdf5_cache_size cannot be negative")

        requested_cameras = tuple(str(camera) for camera in cameras)
        if not requested_cameras:
            raise ValueError("cameras cannot be empty")
        if len(requested_cameras) != len(set(requested_cameras)):
            raise ValueError("cameras must be unique")
        unknown = set(requested_cameras).difference(manifest_index.camera_order)
        if unknown:
            raise ValueError(f"unknown cameras: {sorted(unknown)}")
        canonical_subset = tuple(
            camera
            for camera in manifest_index.camera_order
            if camera in requested_cameras
        )
        if requested_cameras != canonical_subset:
            raise ValueError(
                "cameras must preserve canonical manifest order "
                f"{manifest_index.camera_order}"
            )

        horizons = tuple(int(horizon) for horizon in future_horizons)
        if not horizons or any(horizon <= 0 for horizon in horizons):
            raise ValueError("future_horizons must contain positive offsets")
        if tuple(sorted(set(horizons))) != horizons:
            raise ValueError("future_horizons must be strictly increasing and unique")

        self.manifest = manifest_index
        self.split = _normalize_split(split)
        self.state_history = int(state_history)
        self.action_chunk = int(action_chunk)
        self.cameras = requested_cameras
        self.visual_history = int(visual_history)
        self.future_horizons = horizons
        self.stride = int(stride)
        self.decision_window_radius = int(decision_window_radius)
        self.hdf5_cache_size = int(hdf5_cache_size)
        self.task_order = manifest_index.task_order
        self.task_to_index = manifest_index.task_to_index
        self.records = manifest_index.records_for_split(self.split)
        if not self.records:
            raise ValueError(f"manifest split {self.split!r} contains no episodes")

        self._cache_pid = os.getpid()
        self._cache: OrderedDict[str, h5py.File] = OrderedDict()
        self._index: list[_M1WindowIndex] = []

        state_dim: int | None = None
        action_dim: int | None = None
        image_shape: tuple[int, int, int] | None = None
        complete_horizon = max(
            self.action_chunk, max(self.future_horizons), M1_PROBE_HORIZON
        )
        for record_index, record in enumerate(self.records):
            metadata = manifest_index.hdf5_metadata.get(record.episode_index)
            if metadata is None:
                metadata = _inspect_hdf5_episode(
                    record,
                    camera_order=manifest_index.camera_order,
                    control_hz=manifest_index.control_hz,
                    resolution=tuple(
                        int(x) for x in manifest_index.raw_manifest["resolution"]
                    ),
                )
            if state_dim is None:
                state_dim = metadata.state_dim
                action_dim = metadata.action_dim
                image_shape = metadata.image_shape_hwc
            elif (
                metadata.state_dim != state_dim
                or metadata.action_dim != action_dim
                or metadata.image_shape_hwc != image_shape
            ):
                raise ValueError(
                    f"{record.path} shape contract differs from earlier episodes"
                )

            with h5py.File(record.path, "r") as file:
                frame_indices = np.asarray(
                    file[f"data/observation/image_frame_index/{self.cameras[0]}"][:],
                    dtype=np.int64,
                )
                event_active = np.asarray(
                    file["data/event/visual_signal_active"][:], dtype=np.bool_
                )
                event_onset = np.asarray(
                    file["data/event/visual_signal_onset_step"][:], dtype=np.int64
                )
            anchors = _decision_anchors(event_active, event_onset)
            last_decision = record.num_steps - complete_horizon
            for decision_t in range(0, last_decision + 1, self.stride):
                visual_rows = _latest_unique_rows(
                    frame_indices, end_inclusive=decision_t, count=self.visual_history
                )
                if visual_rows is None:
                    continue
                is_decision_window = any(
                    abs(decision_t - anchor) <= self.decision_window_radius
                    for anchor in anchors
                )
                self._index.append(
                    _M1WindowIndex(
                        record_index=record_index,
                        decision_t=decision_t,
                        visual_rows=visual_rows,
                        decision_window=is_decision_window,
                    )
                )

        if not self._index:
            raise RuntimeError(
                f"split {self.split!r} has no windows with complete horizon "
                f"{complete_horizon} and {self.visual_history} distinct RGB frames"
            )
        assert state_dim is not None
        assert action_dim is not None
        assert image_shape is not None
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.image_shape_hwc = image_shape
        self._observationally_ambiguous_window_indices = (
            _find_observationally_ambiguous_windows(
                records=self.records,
                windows=self._index,
                state_history=self.state_history,
                action_chunk=self.action_chunk,
                cameras=self.cameras,
            )
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self._projected_item(index, self.SAMPLE_KEYS)

    def project(self, sample_keys: Collection[str]) -> Dataset[dict[str, torch.Tensor]]:
        """Return an index-compatible view that reads only requested tensors.

        The canonical dataset contract and lineage remain unchanged.  This
        view is exclusively an execution optimization: it prevents ablations
        and stages from decompressing RGB or loading targets they cannot use.
        """

        requested = frozenset(str(name) for name in sample_keys)
        if not requested:
            raise ValueError("M1 projected sample keys cannot be empty")
        unknown = requested.difference(self.SAMPLE_KEYS)
        if unknown:
            raise ValueError(f"unknown M1 projected sample keys: {sorted(unknown)}")
        return _M1WindowProjection(self, requested)

    def _projected_item(
        self,
        index: int,
        sample_keys: Collection[str],
    ) -> dict[str, torch.Tensor]:
        requested = frozenset(sample_keys)
        window = self._window(index)
        record = self.records[window.record_index]
        decision_t = window.decision_t
        history_start = max(0, decision_t - self.state_history + 1)
        valid_state_count = decision_t - history_start + 1
        history_offset = self.state_history - valid_state_count
        sample: dict[str, torch.Tensor] = {}

        if "task_index" in requested:
            sample["task_index"] = torch.tensor(
                self.task_to_index[record.task_id], dtype=torch.long
            )
        if "future_horizons" in requested:
            sample["future_horizons"] = torch.tensor(
                self.future_horizons, dtype=torch.long
            )

        with self._episode_file(record.path) as file:
            if {"states", "state_valid_mask"}.intersection(requested):
                states = np.zeros(
                    (self.state_history, self.state_dim), dtype=np.float32
                )
                states[history_offset:] = np.asarray(
                    file["data/observation/state"][history_start : decision_t + 1],
                    dtype=np.float32,
                )
                state_valid_mask = np.zeros(self.state_history, dtype=np.bool_)
                state_valid_mask[history_offset:] = True
                if "states" in requested:
                    sample["states"] = torch.from_numpy(states)
                if "state_valid_mask" in requested:
                    sample["state_valid_mask"] = torch.from_numpy(state_valid_mask)

            if {"past_actions", "past_action_valid_mask"}.intersection(requested):
                past_actions = np.zeros(
                    (max(0, self.state_history - 1), self.action_dim),
                    dtype=np.float32,
                )
                past_action_valid_mask = np.zeros(
                    max(0, self.state_history - 1), dtype=np.bool_
                )
                if self.state_history > 1 and decision_t > history_start:
                    executed = np.asarray(
                        file["data/action/executed"][history_start:decision_t],
                        dtype=np.float32,
                    )
                    past_actions[history_offset:] = executed
                    past_action_valid_mask[history_offset:] = True
                if "past_actions" in requested:
                    sample["past_actions"] = torch.from_numpy(past_actions)
                if "past_action_valid_mask" in requested:
                    sample["past_action_valid_mask"] = torch.from_numpy(
                        past_action_valid_mask
                    )

            if "images" in requested:
                sample["images"] = torch.from_numpy(
                    _read_rgb_rows(
                        file,
                        prefix="data/observation/images",
                        rows=window.visual_rows,
                        cameras=self.cameras,
                    )
                )
            if "action_targets" in requested:
                action_targets = np.asarray(
                    file["data/action/commanded"][
                        decision_t : decision_t + self.action_chunk
                    ],
                    dtype=np.float32,
                )
                sample["action_targets"] = torch.from_numpy(action_targets.copy())
            if "future_states" in requested:
                future_states = np.asarray(
                    file["data/next_observation/state"][
                        decision_t : decision_t + self.action_chunk
                    ],
                    dtype=np.float32,
                )
                sample["future_states"] = torch.from_numpy(future_states.copy())

            future_rows = tuple(
                decision_t + horizon - 1 for horizon in self.future_horizons
            )
            if "future_images" in requested:
                sample["future_images"] = torch.from_numpy(
                    _read_rgb_rows(
                        file,
                        prefix="data/next_observation/images",
                        rows=future_rows,
                        cameras=self.cameras,
                    )
                )
            if "future_image_novelty_mask" in requested:
                novelty = np.zeros(
                    (len(self.future_horizons), len(self.cameras)), dtype=np.bool_
                )
                for camera_index, camera in enumerate(self.cameras):
                    previous = int(
                        file[f"data/observation/image_frame_index/{camera}"][decision_t]
                    )
                    target_ids = np.asarray(
                        file[f"data/next_observation/image_frame_index/{camera}"][
                            list(future_rows)
                        ],
                        dtype=np.int64,
                    )
                    for horizon_index, target_id in enumerate(target_ids.tolist()):
                        novelty[horizon_index, camera_index] = target_id != previous
                        previous = int(target_id)
                sample["future_image_novelty_mask"] = torch.from_numpy(novelty)

        if frozenset(sample) != requested:  # pragma: no cover - invariant.
            raise RuntimeError("M1 projected sample key contract drifted")
        return sample

    def probe_labels(self, index: int) -> dict[str, torch.Tensor]:
        """Return explicit offline-only H=8 probe labels.

        This method must not be wired into policy forward.  The center position
        is derived from the two deployable robot poses in next-state; the event
        activity label is read separately and never appears in ``__getitem__``.
        """

        window = self._window(index)
        record = self.records[window.record_index]
        row = window.decision_t + M1_PROBE_HORIZON - 1
        with self._episode_file(record.path) as file:
            future_state = np.asarray(
                file["data/next_observation/state"][row], dtype=np.float32
            )
            if future_state.shape[0] < 13:
                raise ValueError(
                    f"{record.path} state width cannot encode two robot XY poses"
                )
            center_xy = 0.5 * (
                future_state[np.asarray((0, 1))] + future_state[np.asarray((11, 12))]
            )
            event_active = bool(file["data/event/visual_signal_active"][row])
        return {
            "h8_center_xy": torch.from_numpy(center_xy.astype(np.float32)),
            "h8_event_active": torch.tensor(event_active, dtype=torch.bool),
        }

    def sample_lineage(self, index: int) -> M1SampleLineage:
        """Return path/decision identity for an offline audit, never a sample."""

        window = self._window(index)
        return M1SampleLineage(
            path=self.records[window.record_index].path,
            decision_t=window.decision_t,
        )

    @property
    def decision_window_indices(self) -> tuple[int, ...]:
        """Indices eligible for decision-focused sampling (no event labels)."""

        return tuple(
            index for index, window in enumerate(self._index) if window.decision_window
        )

    @property
    def observationally_ambiguous_window_indices(self) -> tuple[int, ...]:
        """Windows whose identical deployable inputs have conflicting actions."""

        return tuple(sorted(self._observationally_ambiguous_window_indices))

    def sampling_weights(self, *, decision_window_boost: float = 2.0) -> torch.Tensor:
        """Return task-balanced weights with unobservable conflicts at zero."""

        boost = float(decision_window_boost)
        if not np.isfinite(boost) or boost <= 0.0:
            raise ValueError("decision_window_boost must be finite and positive")
        factors = np.asarray(
            [boost if window.decision_window else 1.0 for window in self._index],
            dtype=np.float64,
        )
        if self._observationally_ambiguous_window_indices:
            factors[list(self._observationally_ambiguous_window_indices)] = 0.0
        task_indices = np.asarray(
            [
                self.task_to_index[self.records[window.record_index].task_id]
                for window in self._index
            ],
            dtype=np.int64,
        )
        weights = np.empty_like(factors)
        present_tasks = np.unique(task_indices)
        for task_index in present_tasks.tolist():
            mask = task_indices == task_index
            task_total = float(factors[mask].sum())
            if task_total <= 0.0:
                task_id = self.task_order[int(task_index)]
                raise RuntimeError(
                    f"all {task_id!r} windows are observationally ambiguous"
                )
            weights[mask] = factors[mask] / task_total
        weights /= float(len(present_tasks))
        return torch.from_numpy(weights)

    def make_weighted_sampler(
        self,
        *,
        num_samples: int | None = None,
        decision_window_boost: float = 2.0,
        seed: int = 0,
    ) -> WeightedRandomSampler:
        """Build a reproducible replacement sampler from task-balanced weights."""

        count = len(self) if num_samples is None else int(num_samples)
        if count <= 0:
            raise ValueError("num_samples must be positive")
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        return WeightedRandomSampler(
            self.sampling_weights(decision_window_boost=decision_window_boost),
            num_samples=count,
            replacement=True,
            generator=generator,
        )

    def window_summary(self) -> dict[str, Any]:
        """Return the deterministic window/sample contract summary."""

        windows_by_task = Counter(
            self.records[window.record_index].task_id for window in self._index
        )
        decisions_by_task = Counter(
            self.records[window.record_index].task_id
            for window in self._index
            if window.decision_window
        )
        ambiguous_by_task = Counter(
            self.records[self._index[index].record_index].task_id
            for index in self._observationally_ambiguous_window_indices
        )
        ambiguous_ids = [
            _audit_sample_id(
                self.records[self._index[index].record_index],
                self._index[index].decision_t,
            )
            for index in sorted(self._observationally_ambiguous_window_indices)
        ]
        return {
            "split": self.split,
            "state_history": self.state_history,
            "action_chunk": self.action_chunk,
            "visual_history": self.visual_history,
            "cameras": list(self.cameras),
            "future_horizons": list(self.future_horizons),
            "probe_horizon": M1_PROBE_HORIZON,
            "stride": self.stride,
            "decision_window_radius": self.decision_window_radius,
            "windows": len(self),
            "decision_windows": len(self.decision_window_indices),
            "observationally_ambiguous_windows": len(
                self._observationally_ambiguous_window_indices
            ),
            "sampling_eligible_windows": len(self)
            - len(self._observationally_ambiguous_window_indices),
            "windows_by_task": {
                task: int(windows_by_task.get(task, 0)) for task in self.task_order
            },
            "decision_windows_by_task": {
                task: int(decisions_by_task.get(task, 0)) for task in self.task_order
            },
            "observationally_ambiguous_windows_by_task": {
                task: int(ambiguous_by_task.get(task, 0)) for task in self.task_order
            },
            "observationally_ambiguous_sample_ids_sha256": _json_sha256(ambiguous_ids),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "image_shape_hwc": list(self.image_shape_hwc),
            "sample_keys": sorted(self.SAMPLE_KEYS),
        }

    def window_summary_sha256(self) -> str:
        return _json_sha256(self.window_summary())

    def checkpoint_lineage(self) -> dict[str, Any]:
        """Return complete manifest + window hashes suitable for a checkpoint."""

        result = self.manifest.checkpoint_lineage(self.split)
        result.update(
            {
                "window_summary_sha256": self.window_summary_sha256(),
                "window_summary": self.window_summary(),
            }
        )
        return result

    def close(self) -> None:
        while self._cache:
            _, file = self._cache.popitem(last=False)
            file.close()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        state["_cache_pid"] = None
        return state

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown varies.
        try:
            self.close()
        except Exception:
            pass

    def _window(self, index: int) -> _M1WindowIndex:
        resolved = int(index)
        if resolved < 0:
            resolved += len(self._index)
        if resolved < 0 or resolved >= len(self._index):
            raise IndexError(index)
        return self._index[resolved]

    @contextmanager
    def _episode_file(self, path: Path) -> Iterator[h5py.File]:
        current_pid = os.getpid()
        if self._cache_pid != current_pid:
            self.close()
            self._cache_pid = current_pid
        if self.hdf5_cache_size == 0:
            with h5py.File(path, "r") as file:
                yield file
            return

        key = str(path)
        file = self._cache.pop(key, None)
        if file is None:
            file = h5py.File(path, "r")
        self._cache[key] = file
        while len(self._cache) > self.hdf5_cache_size:
            _, evicted = self._cache.popitem(last=False)
            evicted.close()
        yield file


class _M1WindowProjection(Dataset[dict[str, torch.Tensor]]):
    """Read-only field projection preserving the source dataset's index space."""

    def __init__(
        self,
        source: M1WindowDataset,
        sample_keys: frozenset[str],
    ) -> None:
        self.source = source
        self.sample_keys = sample_keys

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.source._projected_item(index, self.sample_keys)


class M1CausalPairDataset(Dataset[dict[str, Any]]):
    """Counterfactual RGB/action pairs built without cue or event labels.

    Pair identity is used only while constructing this offline dataset.  Each
    returned branch contains the same deployable state/action history and a
    different current RGB observation.  ``audit_sample_ids`` are opaque hashes
    for evidence accounting and must be removed before calling a model.
    """

    SAMPLE_KEYS = frozenset(
        {
            "states",
            "state_valid_mask",
            "past_actions",
            "past_action_valid_mask",
            "images",
            "image_valid_mask",
            "task_index",
            "action_targets",
            "audit_sample_ids",
        }
    )

    def __init__(
        self,
        manifest: M1ManifestIndex | str | Path,
        *,
        split: str,
        state_history: int = 32,
        action_chunk: int = M1_CAUSAL_PAIR_HORIZON,
        cameras: Sequence[str] = ("fixed",),
        visual_history: int = 2,
        hdf5_cache_size: int = 8,
        verify_hdf5_sha256: bool = True,
        verify_hdf5_contract: bool = True,
    ) -> None:
        if isinstance(manifest, M1ManifestIndex):
            manifest_index = manifest
        else:
            manifest_index = M1ManifestIndex.from_path(
                manifest,
                verify_hdf5_sha256=verify_hdf5_sha256,
                verify_hdf5_contract=verify_hdf5_contract,
            )
        if int(state_history) <= 0:
            raise ValueError("state_history must be positive")
        if int(action_chunk) != M1_CAUSAL_PAIR_HORIZON:
            raise ValueError(
                f"causal pairs require an H{M1_CAUSAL_PAIR_HORIZON} action chunk"
            )
        if int(visual_history) != 2:
            raise ValueError("causal pairs require a two-slot visual history")
        if int(hdf5_cache_size) < 0:
            raise ValueError("hdf5_cache_size cannot be negative")

        self.manifest = manifest_index
        self.split = _normalize_split(split)
        self.state_history = int(state_history)
        self.action_chunk = int(action_chunk)
        self.visual_history = int(visual_history)
        self.cameras = _canonical_camera_subset(manifest_index, cameras)
        self.hdf5_cache_size = int(hdf5_cache_size)
        self.task_order = manifest_index.task_order
        self.task_to_index = manifest_index.task_to_index
        self.records = manifest_index.records_for_split(self.split)
        if not self.records:
            raise ValueError(f"manifest split {self.split!r} contains no episodes")

        first = self.records[0]
        metadata = manifest_index.hdf5_metadata.get(first.episode_index)
        if metadata is None:
            metadata = _inspect_hdf5_episode(
                first,
                camera_order=manifest_index.camera_order,
                control_hz=manifest_index.control_hz,
                resolution=tuple(
                    int(value) for value in manifest_index.raw_manifest["resolution"]
                ),
            )
        self.state_dim = metadata.state_dim
        self.action_dim = metadata.action_dim
        self.image_shape_hwc = metadata.image_shape_hwc
        if max(M1_CONTROLLED_ACTION_DIMS) >= self.action_dim:
            raise ValueError("controlled action dimensions exceed the dataset width")

        self._cache_pid = os.getpid()
        self._cache: OrderedDict[str, h5py.File] = OrderedDict()
        self._index = _find_causal_pair_anchors(
            records=self.records,
            state_history=self.state_history,
            action_chunk=self.action_chunk,
            cameras=self.cameras,
            visual_history=self.visual_history,
        )
        if not self._index:
            raise RuntimeError(
                f"split {self.split!r} contains no observable causal RGB pairs"
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self._pair(index)
        records = tuple(self.records[value] for value in pair.record_indices)
        task_id = records[0].task_id
        decision_t = pair.decision_t
        history_start = max(0, decision_t - self.state_history + 1)
        valid_state_count = decision_t - history_start + 1
        history_offset = self.state_history - valid_state_count
        image_height, image_width, channels = self.image_shape_hwc
        if channels != 3:  # pragma: no cover - canonical HDF5 audit enforces this.
            raise RuntimeError("causal-pair images must be RGB")

        states = np.zeros((2, self.state_history, self.state_dim), dtype=np.float32)
        state_valid = np.zeros((2, self.state_history), dtype=np.bool_)
        past_actions = np.zeros(
            (2, max(0, self.state_history - 1), self.action_dim),
            dtype=np.float32,
        )
        past_valid = np.zeros((2, max(0, self.state_history - 1)), dtype=np.bool_)
        images = np.zeros(
            (
                2,
                self.visual_history,
                len(self.cameras),
                3,
                image_height,
                image_width,
            ),
            dtype=np.uint8,
        )
        image_valid = np.zeros(
            (2, self.visual_history, len(self.cameras)), dtype=np.bool_
        )
        action_targets = np.zeros(
            (2, self.action_chunk, self.action_dim), dtype=np.float32
        )

        for branch, (record, rows) in enumerate(
            zip(records, pair.visual_rows, strict=True)
        ):
            with self._episode_file(record.path) as file:
                states[branch, history_offset:] = np.asarray(
                    file["data/observation/state"][history_start : decision_t + 1],
                    dtype=np.float32,
                )
                state_valid[branch, history_offset:] = True
                if self.state_history > 1 and decision_t > history_start:
                    past_actions[branch, history_offset:] = np.asarray(
                        file["data/action/executed"][history_start:decision_t],
                        dtype=np.float32,
                    )
                    past_valid[branch, history_offset:] = True
                rgb = _read_rgb_rows(
                    file,
                    prefix="data/observation/images",
                    rows=rows,
                    cameras=self.cameras,
                )
                # Preserve the deployment-time position embedding.  At reset the
                # policy sees T=1 and therefore embeds the sole current frame at
                # temporal index 0.  A fixed-width T=2 causal-pair batch must
                # right-pad that history rather than shift the real frame to
                # temporal index 1.
                images[branch, : len(rows)] = rgb
                image_valid[branch, : len(rows)] = True
                action_targets[branch] = np.asarray(
                    file["data/action/commanded"][
                        decision_t : decision_t + self.action_chunk
                    ],
                    dtype=np.float32,
                )

        if not np.array_equal(states[0], states[1]) or not np.array_equal(
            past_actions[0], past_actions[1]
        ):
            raise RuntimeError(
                "causal pair deployable histories drifted after indexing"
            )
        controlled = np.asarray(M1_CONTROLLED_ACTION_DIMS, dtype=np.int64)
        if np.array_equal(
            action_targets[0, :, controlled], action_targets[1, :, controlled]
        ):
            raise RuntimeError("causal pair action contrast disappeared after indexing")

        task_value = self.task_to_index[task_id]
        sample: dict[str, Any] = {
            "states": torch.from_numpy(states),
            "state_valid_mask": torch.from_numpy(state_valid),
            "past_actions": torch.from_numpy(past_actions),
            "past_action_valid_mask": torch.from_numpy(past_valid),
            "images": torch.from_numpy(images),
            "image_valid_mask": torch.from_numpy(image_valid),
            "task_index": torch.full((2,), task_value, dtype=torch.long),
            "action_targets": torch.from_numpy(action_targets),
            "audit_sample_ids": pair.audit_sample_ids,
        }
        if frozenset(sample) != self.SAMPLE_KEYS:  # pragma: no cover - invariant.
            raise RuntimeError("M1 causal-pair sample key contract drifted")
        return sample

    def pair_summary(self) -> dict[str, Any]:
        pairs_by_task = Counter(
            self.records[pair.record_indices[0]].task_id for pair in self._index
        )
        anchors_by_task: dict[str, Counter[int]] = defaultdict(Counter)
        single_frame = 0
        audit_ids: list[str] = []
        for pair in self._index:
            task = self.records[pair.record_indices[0]].task_id
            anchors_by_task[task][pair.decision_t] += 1
            if all(len(rows) == 1 for rows in pair.visual_rows):
                single_frame += 1
            audit_ids.extend(pair.audit_sample_ids)
        return {
            "split": self.split,
            "state_history": self.state_history,
            "action_chunk": self.action_chunk,
            "visual_history_slots": self.visual_history,
            "cameras": list(self.cameras),
            "controlled_action_dims": list(M1_CONTROLLED_ACTION_DIMS),
            "pairs": len(self),
            "branch_samples": 2 * len(self),
            "single_effective_frame_pairs": single_frame,
            "two_effective_frame_pairs": len(self) - single_frame,
            "visual_history_alignment": "deployable_prefix_right_padding",
            "pairs_by_task": {
                task: int(pairs_by_task.get(task, 0)) for task in self.task_order
            },
            "anchor_t_by_task": {
                task: {
                    str(step): int(count)
                    for step, count in sorted(anchors_by_task[task].items())
                }
                for task in self.task_order
            },
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "image_shape_hwc": list(self.image_shape_hwc),
            "sample_keys": sorted(self.SAMPLE_KEYS),
            "audit_sample_ids_sha256": _json_sha256(sorted(audit_ids)),
        }

    def pair_summary_sha256(self) -> str:
        return _json_sha256(self.pair_summary())

    def checkpoint_lineage(self) -> dict[str, Any]:
        result = self.manifest.checkpoint_lineage(self.split)
        result.update(
            {
                "causal_pair_summary_sha256": self.pair_summary_sha256(),
                "causal_pair_summary": self.pair_summary(),
            }
        )
        return result

    def close(self) -> None:
        while self._cache:
            _, file = self._cache.popitem(last=False)
            file.close()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        state["_cache_pid"] = None
        return state

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown varies.
        try:
            self.close()
        except Exception:
            pass

    def _pair(self, index: int) -> _M1CausalPairIndex:
        resolved = int(index)
        if resolved < 0:
            resolved += len(self._index)
        if resolved < 0 or resolved >= len(self._index):
            raise IndexError(index)
        return self._index[resolved]

    @contextmanager
    def _episode_file(self, path: Path) -> Iterator[h5py.File]:
        current_pid = os.getpid()
        if self._cache_pid != current_pid:
            self.close()
            self._cache_pid = current_pid
        if self.hdf5_cache_size == 0:
            with h5py.File(path, "r") as file:
                yield file
            return
        key = str(path)
        file = self._cache.pop(key, None)
        if file is None:
            file = h5py.File(path, "r")
        self._cache[key] = file
        while len(self._cache) > self.hdf5_cache_size:
            _, evicted = self._cache.popitem(last=False)
            evicted.close()
        yield file


class M1StateCausalPairDataset(Dataset[dict[str, Any]]):
    """Proprioceptive causal pairs selected from deployable tensors only.

    Each pair is formed inside one episode at adjacent control decisions.  The
    two branches have identical task, fixed-camera RGB history, validity masks,
    and three-step executed-action history.  Their four-state histories and
    immediately executed controlled action targets must differ.  Cue, event,
    intervention, physical-randomization, and acceptance labels are never read
    by the selector.
    """

    SELECTOR_VERSION = "same_episode_gap1_s4_a3_lateral_feedback_v1"
    SAMPLE_KEYS = frozenset(
        {
            "states",
            "state_valid_mask",
            "past_actions",
            "past_action_valid_mask",
            "images",
            "image_valid_mask",
            "task_index",
            "action_targets",
            "audit_sample_ids",
        }
    )

    def __init__(
        self,
        manifest: M1ManifestIndex | str | Path,
        *,
        split: str,
        state_history: int = M1_STATE_CAUSAL_HISTORY_WIDTH,
        action_chunk: int = M1_CAUSAL_PAIR_HORIZON,
        cameras: Sequence[str] = ("fixed",),
        visual_history: int = 2,
        valid_state_steps: int = M1_STATE_CAUSAL_STATES,
        decision_gap: int = M1_STATE_CAUSAL_GAP,
        step0_min_action_delta: float = M1_STATE_CAUSAL_MIN_ACTION_DELTA,
        feedback_equality_atol: float = M1_STATE_CAUSAL_FEEDBACK_EQUALITY_ATOL,
        hdf5_cache_size: int = 8,
        verify_hdf5_sha256: bool = True,
        verify_hdf5_contract: bool = True,
    ) -> None:
        if isinstance(manifest, M1ManifestIndex):
            manifest_index = manifest
        else:
            manifest_index = M1ManifestIndex.from_path(
                manifest,
                verify_hdf5_sha256=verify_hdf5_sha256,
                verify_hdf5_contract=verify_hdf5_contract,
            )
        if int(action_chunk) != M1_CAUSAL_PAIR_HORIZON:
            raise ValueError(
                f"state causal pairs require an H{M1_CAUSAL_PAIR_HORIZON} action chunk"
            )
        if int(state_history) != M1_STATE_CAUSAL_HISTORY_WIDTH:
            raise ValueError(
                "state causal pairs require a 32-step padded state history"
            )
        if int(visual_history) != 2:
            raise ValueError("state causal pairs require two RGB history slots")
        if int(valid_state_steps) != M1_STATE_CAUSAL_STATES:
            raise ValueError("state causal pairs require four valid state steps")
        if int(decision_gap) != M1_STATE_CAUSAL_GAP:
            raise ValueError("state causal pairs require decision_gap=1")
        if (
            not np.isfinite(float(step0_min_action_delta))
            or float(step0_min_action_delta) <= 0.0
        ):
            raise ValueError("step0_min_action_delta must be finite and positive")
        if (
            not np.isfinite(float(feedback_equality_atol))
            or float(feedback_equality_atol) < 0.0
        ):
            raise ValueError("feedback_equality_atol must be finite and non-negative")
        if int(hdf5_cache_size) < 0:
            raise ValueError("hdf5_cache_size cannot be negative")

        selected_cameras = _canonical_camera_subset(manifest_index, cameras)
        if selected_cameras != ("fixed",):
            raise ValueError("state causal pairs require only the fixed camera")
        self.manifest = manifest_index
        self.split = _normalize_split(split)
        self.state_history = int(state_history)
        self.past_action_history = self.state_history - 1
        self.state_valid_steps = int(valid_state_steps)
        self.past_action_valid_steps = self.state_valid_steps - 1
        self.action_chunk = int(action_chunk)
        self.visual_history = int(visual_history)
        self.decision_gap = int(decision_gap)
        self.step0_min_action_delta = float(step0_min_action_delta)
        self.feedback_equality_atol = float(feedback_equality_atol)
        self.cameras = selected_cameras
        self.hdf5_cache_size = int(hdf5_cache_size)
        self.task_order = manifest_index.task_order
        self.task_to_index = manifest_index.task_to_index
        self.records = manifest_index.records_for_split(self.split)
        if not self.records:
            raise ValueError(f"manifest split {self.split!r} contains no episodes")

        first = self.records[0]
        metadata = manifest_index.hdf5_metadata.get(first.episode_index)
        if metadata is None:
            metadata = _inspect_hdf5_episode(
                first,
                camera_order=manifest_index.camera_order,
                control_hz=manifest_index.control_hz,
                resolution=tuple(
                    int(value) for value in manifest_index.raw_manifest["resolution"]
                ),
            )
        self.state_dim = metadata.state_dim
        self.action_dim = metadata.action_dim
        self.image_shape_hwc = metadata.image_shape_hwc
        if max(M1_CONTROLLED_ACTION_DIMS) >= self.action_dim:
            raise ValueError("controlled action dimensions exceed the dataset width")

        self._cache_pid = os.getpid()
        self._cache: OrderedDict[str, h5py.File] = OrderedDict()
        self._index, self._selector_counts = _find_state_causal_pair_anchors(
            records=self.records,
            action_chunk=self.action_chunk,
            cameras=self.cameras,
            valid_state_steps=self.state_valid_steps,
            decision_gap=self.decision_gap,
            step0_min_action_delta=self.step0_min_action_delta,
            feedback_equality_atol=self.feedback_equality_atol,
        )
        if not self._index:
            raise RuntimeError(
                f"split {self.split!r} contains no deployable state causal pairs"
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self._pair(index)
        record = self.records[pair.record_index]
        image_height, image_width, channels = self.image_shape_hwc
        if channels != 3:  # pragma: no cover - canonical HDF5 audit enforces this.
            raise RuntimeError("state causal-pair images must be RGB")

        states = np.zeros((2, self.state_history, self.state_dim), dtype=np.float32)
        state_valid = np.zeros((2, self.state_history), dtype=np.bool_)
        past_actions = np.zeros(
            (2, self.past_action_history, self.action_dim), dtype=np.float32
        )
        past_valid = np.zeros((2, self.past_action_history), dtype=np.bool_)
        images = np.empty(
            (
                2,
                self.visual_history,
                len(self.cameras),
                3,
                image_height,
                image_width,
            ),
            dtype=np.uint8,
        )
        image_valid = np.ones(
            (2, self.visual_history, len(self.cameras)), dtype=np.bool_
        )
        action_targets = np.empty(
            (2, self.action_chunk, self.action_dim), dtype=np.float32
        )

        with self._episode_file(record.path) as file:
            for branch, (decision_t, rows) in enumerate(
                zip(pair.decision_ts, pair.visual_rows, strict=True)
            ):
                history_start = decision_t - self.past_action_valid_steps
                states[branch, -self.state_valid_steps :] = np.asarray(
                    file["data/observation/state"][history_start : decision_t + 1],
                    dtype=np.float32,
                )
                state_valid[branch, -self.state_valid_steps :] = True
                past_actions[branch, -self.past_action_valid_steps :] = np.asarray(
                    file["data/action/executed"][history_start:decision_t],
                    dtype=np.float32,
                )
                past_valid[branch, -self.past_action_valid_steps :] = True
                images[branch] = _read_rgb_rows(
                    file,
                    prefix="data/observation/images",
                    rows=rows,
                    cameras=self.cameras,
                )
                action_targets[branch] = np.asarray(
                    file["data/action/commanded"][
                        decision_t : decision_t + self.action_chunk
                    ],
                    dtype=np.float32,
                )

        if np.array_equal(states[0], states[1]):
            raise RuntimeError("state causal-pair state contrast disappeared")
        if not np.array_equal(past_actions[0], past_actions[1]):
            raise RuntimeError("state causal-pair past-action histories drifted")
        if not np.array_equal(images[0], images[1]):
            raise RuntimeError("state causal-pair RGB histories drifted")
        if not np.array_equal(state_valid[0], state_valid[1]) or not np.array_equal(
            past_valid[0], past_valid[1]
        ):
            raise RuntimeError("state causal-pair validity masks drifted")
        execute_delta = action_targets[1, 0] - action_targets[0, 0]
        current_state_delta = states[1, -1] - states[0, -1]
        if not _is_state_feedback_action_delta(
            execute_delta,
            current_state_delta,
            minimum_action_delta=self.step0_min_action_delta,
            equality_atol=self.feedback_equality_atol,
        ):
            raise RuntimeError("state causal-pair execute-step contrast disappeared")

        task_value = self.task_to_index[record.task_id]
        sample: dict[str, Any] = {
            "states": torch.from_numpy(states),
            "state_valid_mask": torch.from_numpy(state_valid),
            "past_actions": torch.from_numpy(past_actions),
            "past_action_valid_mask": torch.from_numpy(past_valid),
            "images": torch.from_numpy(images),
            "image_valid_mask": torch.from_numpy(image_valid),
            "task_index": torch.full((2,), task_value, dtype=torch.long),
            "action_targets": torch.from_numpy(action_targets),
            "audit_sample_ids": pair.audit_sample_ids,
        }
        if frozenset(sample) != self.SAMPLE_KEYS:  # pragma: no cover - invariant.
            raise RuntimeError("M1 state causal-pair sample key contract drifted")
        return sample

    def pair_summary(self) -> dict[str, Any]:
        pairs_by_task = Counter(
            self.records[pair.record_index].task_id for pair in self._index
        )
        anchors_by_task: dict[str, Counter[int]] = defaultdict(Counter)
        audit_ids: list[str] = []
        for pair in self._index:
            task = self.records[pair.record_index].task_id
            anchors_by_task[task][pair.decision_ts[0]] += 1
            audit_ids.extend(pair.audit_sample_ids)
        return {
            "split": self.split,
            "selector_version": self.SELECTOR_VERSION,
            "same_episode": True,
            "decision_gap_control_steps": self.decision_gap,
            "state_history": self.state_history,
            "past_action_history": self.past_action_history,
            "state_valid_steps": self.state_valid_steps,
            "past_action_valid_steps": self.past_action_valid_steps,
            "action_chunk": self.action_chunk,
            "selector_required_action_delta_steps": 1,
            "minimum_execute_step0_max_abs_delta": (self.step0_min_action_delta),
            "non_lateral_zero_tolerance": self.feedback_equality_atol,
            "lateral_equality_atol": self.feedback_equality_atol,
            "state_feedback_sign_rule": (
                "mean_lateral_action_delta_times_mean_robot_x_delta_lt_zero"
            ),
            "visual_history_slots": self.visual_history,
            "cameras": list(self.cameras),
            "controlled_action_dims": list(M1_CONTROLLED_ACTION_DIMS),
            "lateral_action_dims": list(M1_STATE_CAUSAL_LATERAL_ACTION_DIMS),
            "zero_delta_action_dims": list(M1_STATE_CAUSAL_ZERO_DELTA_ACTION_DIMS),
            "robot_x_state_dims": list(M1_STATE_CAUSAL_ROBOT_X_STATE_DIMS),
            "pairs": len(self),
            "branch_samples": 2 * len(self),
            "pairs_by_task": {
                task: int(pairs_by_task.get(task, 0)) for task in self.task_order
            },
            "anchor_t_by_task": {
                task: {
                    str(step): int(count)
                    for step, count in sorted(anchors_by_task[task].items())
                }
                for task in self.task_order
            },
            "selector_counts": dict(self._selector_counts),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "image_shape_hwc": list(self.image_shape_hwc),
            "sample_keys": sorted(self.SAMPLE_KEYS),
            "audit_sample_ids_sha256": _json_sha256(sorted(audit_ids)),
        }

    def pair_summary_sha256(self) -> str:
        return _json_sha256(self.pair_summary())

    def checkpoint_lineage(self) -> dict[str, Any]:
        result = self.manifest.checkpoint_lineage(self.split)
        result.update(
            {
                "state_causal_pair_summary_sha256": self.pair_summary_sha256(),
                "state_causal_pair_summary": self.pair_summary(),
            }
        )
        return result

    def close(self) -> None:
        while self._cache:
            _, file = self._cache.popitem(last=False)
            file.close()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        state["_cache_pid"] = None
        return state

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown varies.
        try:
            self.close()
        except Exception:
            pass

    def _pair(self, index: int) -> _M1StateCausalPairIndex:
        resolved = int(index)
        if resolved < 0:
            resolved += len(self._index)
        if resolved < 0 or resolved >= len(self._index):
            raise IndexError(index)
        return self._index[resolved]

    @contextmanager
    def _episode_file(self, path: Path) -> Iterator[h5py.File]:
        current_pid = os.getpid()
        if self._cache_pid != current_pid:
            self.close()
            self._cache_pid = current_pid
        if self.hdf5_cache_size == 0:
            with h5py.File(path, "r") as file:
                yield file
            return
        key = str(path)
        file = self._cache.pop(key, None)
        if file is None:
            file = h5py.File(path, "r")
        self._cache[key] = file
        while len(self._cache) > self.hdf5_cache_size:
            _, evicted = self._cache.popitem(last=False)
            evicted.close()
        yield file


def load_m1_manifest(
    manifest_path: str | Path,
    *,
    verify_hdf5_sha256: bool = True,
    verify_hdf5_contract: bool = True,
) -> M1ManifestIndex:
    """Convenience wrapper around :meth:`M1ManifestIndex.from_path`."""

    return M1ManifestIndex.from_path(
        manifest_path,
        verify_hdf5_sha256=verify_hdf5_sha256,
        verify_hdf5_contract=verify_hdf5_contract,
    )


def _canonical_camera_subset(
    manifest: M1ManifestIndex, cameras: Sequence[str]
) -> tuple[str, ...]:
    requested = tuple(str(camera) for camera in cameras)
    if not requested:
        raise ValueError("cameras cannot be empty")
    if len(requested) != len(set(requested)):
        raise ValueError("cameras must be unique")
    unknown = set(requested).difference(manifest.camera_order)
    if unknown:
        raise ValueError(f"unknown cameras: {sorted(unknown)}")
    canonical = tuple(camera for camera in manifest.camera_order if camera in requested)
    if requested != canonical:
        raise ValueError(
            f"cameras must preserve canonical manifest order {manifest.camera_order}"
        )
    return requested


def _pair_record_groups(
    records: Sequence[M1ManifestEpisode],
) -> tuple[tuple[int, int], ...]:
    """Group records by physical randomization without consulting cue ids."""

    grouped: dict[tuple[str, int, str, str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        key = (
            record.task_id,
            record.physical_seed,
            record.scene_id,
            record.template_id,
            record.object_combination_id,
        )
        grouped[key].append(index)
    pairs: list[tuple[int, int]] = []
    for key, indices in grouped.items():
        if len(indices) != 2:
            raise ValueError(
                "causal pairing requires exactly two episodes for "
                f"task/physical/scene/template/object group {key!r}, got {len(indices)}"
            )
        ordered = tuple(sorted(indices, key=lambda value: records[value].episode_index))
        pairs.append((ordered[0], ordered[1]))
    return tuple(
        sorted(
            pairs,
            key=lambda values: (
                records[values[0]].task_id,
                records[values[0]].episode_index,
                records[values[1]].episode_index,
            ),
        )
    )


def _find_causal_pair_anchors(
    *,
    records: Sequence[M1ManifestEpisode],
    state_history: int,
    action_chunk: int,
    cameras: Sequence[str],
    visual_history: int,
) -> list[_M1CausalPairIndex]:
    controlled = np.asarray(M1_CONTROLLED_ACTION_DIMS, dtype=np.int64)
    result: list[_M1CausalPairIndex] = []
    for record_indices in _pair_record_groups(records):
        left = records[record_indices[0]]
        right = records[record_indices[1]]
        with (
            h5py.File(left.path, "r") as left_file,
            h5py.File(right.path, "r") as right_file,
        ):
            left_states = np.asarray(
                left_file["data/observation/state"], dtype=np.float32
            )
            right_states = np.asarray(
                right_file["data/observation/state"], dtype=np.float32
            )
            left_executed = np.asarray(
                left_file["data/action/executed"], dtype=np.float32
            )
            right_executed = np.asarray(
                right_file["data/action/executed"], dtype=np.float32
            )
            left_targets = np.asarray(
                left_file["data/action/commanded"], dtype=np.float32
            )
            right_targets = np.asarray(
                right_file["data/action/commanded"], dtype=np.float32
            )
            frame_indices = (
                tuple(
                    np.asarray(
                        left_file[f"data/observation/image_frame_index/{camera}"],
                        dtype=np.int64,
                    )
                    for camera in cameras
                ),
                tuple(
                    np.asarray(
                        right_file[f"data/observation/image_frame_index/{camera}"],
                        dtype=np.int64,
                    )
                    for camera in cameras
                ),
            )
            last_t = min(len(left_targets), len(right_targets)) - action_chunk
            selected: _M1CausalPairIndex | None = None
            for decision_t in range(last_t + 1):
                if not _deployable_history_equal(
                    left_states,
                    right_states,
                    left_executed,
                    right_executed,
                    decision_t=decision_t,
                    state_history=state_history,
                ):
                    continue
                left_action = left_targets[
                    decision_t : decision_t + action_chunk, controlled
                ]
                right_action = right_targets[
                    decision_t : decision_t + action_chunk, controlled
                ]
                if np.array_equal(left_action, right_action):
                    continue
                if _current_rgb_equal(
                    left_file,
                    right_file,
                    decision_t=decision_t,
                    cameras=cameras,
                ):
                    continue
                rows = tuple(
                    _latest_available_unique_rows(
                        indices[0],
                        end_inclusive=decision_t,
                        maximum_count=visual_history,
                    )
                    for indices in frame_indices
                )
                selected = _M1CausalPairIndex(
                    record_indices=record_indices,
                    decision_t=decision_t,
                    visual_rows=(rows[0], rows[1]),
                    audit_sample_ids=(
                        _audit_sample_id(left, decision_t),
                        _audit_sample_id(right, decision_t),
                    ),
                )
                break
        if selected is not None:
            result.append(selected)
    return result


def _find_state_causal_pair_anchors(
    *,
    records: Sequence[M1ManifestEpisode],
    action_chunk: int,
    cameras: Sequence[str],
    valid_state_steps: int,
    decision_gap: int,
    step0_min_action_delta: float,
    feedback_equality_atol: float,
) -> tuple[list[_M1StateCausalPairIndex], Mapping[str, int]]:
    """Select the earliest state-identifiable pair in each episode.

    The rejection cascade is intentionally expressed only in deployable tensor
    terms.  Its counters are returned so the complete selector behavior is
    bound into dataset lineage rather than leaving only the accepted examples
    auditable.
    """

    count_keys = (
        "episodes_considered",
        "candidate_steps_considered",
        "visual_history_incomplete",
        "visual_frame_id_history_mismatch",
        "past_action_history_mismatch",
        "current_state_equal",
        "execute_step0_delta_below_threshold",
        "execute_step0_non_lateral_delta",
        "execute_step0_lateral_mismatch",
        "state_feedback_sign_mismatch",
        "visual_tensor_history_mismatch",
        "eligible_pairs",
        "later_eligible_pairs_excluded",
        "episodes_without_pair",
        "selected_pairs",
    )
    counts: Counter[str] = Counter()
    controlled = np.asarray(M1_CONTROLLED_ACTION_DIMS, dtype=np.int64)
    lateral = np.asarray(M1_STATE_CAUSAL_LATERAL_ACTION_DIMS, dtype=np.int64)
    robot_x = np.asarray(M1_STATE_CAUSAL_ROBOT_X_STATE_DIMS, dtype=np.int64)
    result: list[_M1StateCausalPairIndex] = []
    for record_index, record in enumerate(records):
        counts["episodes_considered"] += 1
        selected: _M1StateCausalPairIndex | None = None
        with h5py.File(record.path, "r") as file:
            states = np.asarray(file["data/observation/state"], dtype=np.float32)
            executed = np.asarray(file["data/action/executed"], dtype=np.float32)
            commanded = np.asarray(file["data/action/commanded"], dtype=np.float32)
            non_lateral = np.asarray(
                M1_STATE_CAUSAL_ZERO_DELTA_ACTION_DIMS, dtype=np.int64
            )
            frame_indices = np.asarray(
                file["data/observation/image_frame_index/fixed"],
                dtype=np.int64,
            )
            last_left_t = len(commanded) - int(action_chunk) - int(decision_gap)
            for left_t in range(int(valid_state_steps) - 1, last_left_t + 1):
                counts["candidate_steps_considered"] += 1
                right_t = left_t + int(decision_gap)
                rows = (
                    _latest_available_unique_rows(
                        frame_indices,
                        end_inclusive=left_t,
                        maximum_count=2,
                    ),
                    _latest_available_unique_rows(
                        frame_indices,
                        end_inclusive=right_t,
                        maximum_count=2,
                    ),
                )
                if any(len(branch_rows) != 2 for branch_rows in rows):
                    counts["visual_history_incomplete"] += 1
                    continue
                frame_ids = tuple(
                    tuple(int(frame_indices[row]) for row in branch_rows)
                    for branch_rows in rows
                )
                if frame_ids[0] != frame_ids[1]:
                    counts["visual_frame_id_history_mismatch"] += 1
                    continue

                past_action_steps = int(valid_state_steps) - 1
                left_start = left_t - past_action_steps
                right_start = right_t - past_action_steps
                left_past = executed[left_start:left_t]
                right_past = executed[right_start:right_t]
                if not np.array_equal(left_past, right_past):
                    counts["past_action_history_mismatch"] += 1
                    continue
                left_states = states[left_start : left_t + 1]
                right_states = states[right_start : right_t + 1]
                current_state_delta = right_states[-1] - left_states[-1]
                if np.array_equal(left_states[-1], right_states[-1]):
                    counts["current_state_equal"] += 1
                    continue
                execute_delta = commanded[right_t] - commanded[left_t]
                if float(np.max(np.abs(execute_delta[controlled]))) < float(
                    step0_min_action_delta
                ):
                    counts["execute_step0_delta_below_threshold"] += 1
                    continue
                if bool(
                    np.any(
                        np.abs(execute_delta[non_lateral])
                        > float(feedback_equality_atol)
                    )
                ):
                    counts["execute_step0_non_lateral_delta"] += 1
                    continue
                lateral_delta = execute_delta[lateral]
                if bool(
                    np.any(np.abs(lateral_delta) < float(step0_min_action_delta))
                    or not np.isclose(
                        lateral_delta[0],
                        lateral_delta[1],
                        atol=float(feedback_equality_atol),
                        rtol=0.0,
                    )
                ):
                    counts["execute_step0_lateral_mismatch"] += 1
                    continue
                if (
                    float(lateral_delta.mean() * current_state_delta[robot_x].mean())
                    >= 0.0
                ):
                    counts["state_feedback_sign_mismatch"] += 1
                    continue
                rgb = tuple(
                    _read_rgb_rows(
                        file,
                        prefix="data/observation/images",
                        rows=branch_rows,
                        cameras=cameras,
                    )
                    for branch_rows in rows
                )
                if not np.array_equal(rgb[0], rgb[1]):
                    counts["visual_tensor_history_mismatch"] += 1
                    continue

                counts["eligible_pairs"] += 1
                candidate = _M1StateCausalPairIndex(
                    record_index=record_index,
                    decision_ts=(left_t, right_t),
                    visual_rows=(rows[0], rows[1]),
                    audit_sample_ids=(
                        _audit_sample_id(record, left_t),
                        _audit_sample_id(record, right_t),
                    ),
                )
                if selected is None:
                    selected = candidate
                else:
                    counts["later_eligible_pairs_excluded"] += 1
        if selected is None:
            counts["episodes_without_pair"] += 1
        else:
            counts["selected_pairs"] += 1
            result.append(selected)
    audited_counts = MappingProxyType(
        {key: int(counts.get(key, 0)) for key in count_keys}
    )
    return result, audited_counts


def _is_state_feedback_action_delta(
    execute_delta: np.ndarray,
    current_state_delta: np.ndarray,
    *,
    minimum_action_delta: float,
    equality_atol: float,
) -> bool:
    if execute_delta.ndim != 1 or current_state_delta.ndim != 1:
        return False
    if max(M1_CONTROLLED_ACTION_DIMS) >= len(execute_delta) or max(
        M1_STATE_CAUSAL_ROBOT_X_STATE_DIMS
    ) >= len(current_state_delta):
        return False
    controlled = np.asarray(M1_CONTROLLED_ACTION_DIMS, dtype=np.int64)
    lateral = np.asarray(M1_STATE_CAUSAL_LATERAL_ACTION_DIMS, dtype=np.int64)
    non_lateral = np.asarray(M1_STATE_CAUSAL_ZERO_DELTA_ACTION_DIMS, dtype=np.int64)
    robot_x = np.asarray(M1_STATE_CAUSAL_ROBOT_X_STATE_DIMS, dtype=np.int64)
    if float(np.max(np.abs(execute_delta[controlled]))) < float(
        minimum_action_delta
    ) or bool(np.any(np.abs(execute_delta[non_lateral]) > float(equality_atol))):
        return False
    lateral_delta = execute_delta[lateral]
    return bool(
        np.all(np.abs(lateral_delta) >= float(minimum_action_delta))
        and np.isclose(
            lateral_delta[0],
            lateral_delta[1],
            atol=float(equality_atol),
            rtol=0.0,
        )
        and float(lateral_delta.mean() * current_state_delta[robot_x].mean()) < 0.0
    )


def _find_observationally_ambiguous_windows(
    *,
    records: Sequence[M1ManifestEpisode],
    windows: Sequence[_M1WindowIndex],
    state_history: int,
    action_chunk: int,
    cameras: Sequence[str],
) -> frozenset[int]:
    """Find conflicting labels for bitwise-identical deployable observations."""

    lookup: dict[int, dict[int, int]] = defaultdict(dict)
    for index, window in enumerate(windows):
        lookup[window.record_index][window.decision_t] = index
    controlled = np.asarray(M1_CONTROLLED_ACTION_DIMS, dtype=np.int64)
    ambiguous: set[int] = set()
    for left_index, right_index in _pair_record_groups(records):
        common_steps = sorted(set(lookup[left_index]).intersection(lookup[right_index]))
        if not common_steps:
            continue
        left = records[left_index]
        right = records[right_index]
        with (
            h5py.File(left.path, "r") as left_file,
            h5py.File(right.path, "r") as right_file,
        ):
            left_states = np.asarray(
                left_file["data/observation/state"], dtype=np.float32
            )
            right_states = np.asarray(
                right_file["data/observation/state"], dtype=np.float32
            )
            left_executed = np.asarray(
                left_file["data/action/executed"], dtype=np.float32
            )
            right_executed = np.asarray(
                right_file["data/action/executed"], dtype=np.float32
            )
            left_targets = np.asarray(
                left_file["data/action/commanded"], dtype=np.float32
            )
            right_targets = np.asarray(
                right_file["data/action/commanded"], dtype=np.float32
            )
            for decision_t in common_steps:
                left_window_index = lookup[left_index][decision_t]
                right_window_index = lookup[right_index][decision_t]
                left_window = windows[left_window_index]
                right_window = windows[right_window_index]
                if not _deployable_history_equal(
                    left_states,
                    right_states,
                    left_executed,
                    right_executed,
                    decision_t=decision_t,
                    state_history=state_history,
                ):
                    continue
                if np.array_equal(
                    left_targets[decision_t : decision_t + action_chunk, controlled],
                    right_targets[decision_t : decision_t + action_chunk, controlled],
                ):
                    continue
                left_rgb = _read_rgb_rows(
                    left_file,
                    prefix="data/observation/images",
                    rows=left_window.visual_rows,
                    cameras=cameras,
                )
                right_rgb = _read_rgb_rows(
                    right_file,
                    prefix="data/observation/images",
                    rows=right_window.visual_rows,
                    cameras=cameras,
                )
                if np.array_equal(left_rgb, right_rgb):
                    ambiguous.update((left_window_index, right_window_index))
    return frozenset(ambiguous)


def _deployable_history_equal(
    left_states: np.ndarray,
    right_states: np.ndarray,
    left_executed: np.ndarray,
    right_executed: np.ndarray,
    *,
    decision_t: int,
    state_history: int,
) -> bool:
    start = max(0, int(decision_t) - int(state_history) + 1)
    return bool(
        np.array_equal(
            left_states[start : decision_t + 1],
            right_states[start : decision_t + 1],
        )
        and np.array_equal(
            left_executed[start:decision_t],
            right_executed[start:decision_t],
        )
    )


def _current_rgb_equal(
    left_file: h5py.File,
    right_file: h5py.File,
    *,
    decision_t: int,
    cameras: Sequence[str],
) -> bool:
    return all(
        np.array_equal(
            np.asarray(left_file[f"data/observation/images/{camera}"][decision_t]),
            np.asarray(right_file[f"data/observation/images/{camera}"][decision_t]),
        )
        for camera in cameras
    )


def _latest_available_unique_rows(
    frame_indices: np.ndarray, *, end_inclusive: int, maximum_count: int
) -> tuple[int, ...]:
    selected: list[int] = []
    seen: set[int] = set()
    for row in range(int(end_inclusive), -1, -1):
        frame_id = int(frame_indices[row])
        if frame_id in seen:
            continue
        selected.append(row)
        seen.add(frame_id)
        if len(selected) == int(maximum_count):
            break
    if not selected:
        raise RuntimeError("causal anchor has no deployable RGB frame")
    return tuple(reversed(selected))


def _audit_sample_id(record: M1ManifestEpisode, decision_t: int) -> str:
    return _json_sha256(
        {
            "episode_artifact_sha256": record.hdf5_sha256,
            "decision_t": int(decision_t),
        }
    )


def _validate_manifest_header(raw: Mapping[str, Any]) -> None:
    required = {
        "format_version",
        "phase",
        "schema_profile",
        "schema_version",
        "camera_order",
        "tasks",
        "cue_variants",
        "control_hz",
        "image_hz",
        "resolution",
        "split_counts",
        "formal_protocol",
        "raw_unannotated",
        "episodes",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise KeyError(f"manifest is missing required keys: {missing}")
    expected_values = {
        "format_version": M1_MANIFEST_FORMAT,
        "phase": "M0",
        "schema_profile": M1_SCHEMA_PROFILE,
        "schema_version": M1_SCHEMA_VERSION,
    }
    for key, expected in expected_values.items():
        if raw[key] != expected:
            raise ValueError(f"manifest {key} {raw[key]!r}; expected {expected!r}")
    if tuple(raw["camera_order"]) != CANONICAL_CAMERA_ORDER:
        raise ValueError(
            f"manifest camera_order must be {CANONICAL_CAMERA_ORDER}, "
            f"got {raw['camera_order']!r}"
        )
    if tuple(int(x) for x in raw["cue_variants"]) != CANONICAL_CUE_VARIANTS:
        raise ValueError(f"manifest cue_variants must be {CANONICAL_CUE_VARIANTS}")
    tasks = tuple(str(task) for task in raw["tasks"])
    if not tasks or len(tasks) != len(set(tasks)) or any(not task for task in tasks):
        raise ValueError("manifest tasks must be non-empty and unique")
    for name in ("control_hz", "image_hz"):
        value = float(raw[name])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"manifest {name} must be finite and positive")
    resolution = tuple(int(x) for x in raw["resolution"])
    if len(resolution) != 2 or any(x <= 0 for x in resolution):
        raise ValueError("manifest resolution must be [height, width]")
    if raw["formal_protocol"] is not True:
        raise ValueError("canonical M1 input requires formal_protocol=true")
    if raw["raw_unannotated"] is not True:
        raise ValueError("canonical M1 input requires raw_unannotated=true")
    if not isinstance(raw["episodes"], list) or not raw["episodes"]:
        raise ValueError("manifest episodes must be a non-empty list")


def _parse_manifest_episodes(
    raw: Mapping[str, Any], *, manifest_path: Path
) -> list[M1ManifestEpisode]:
    required = {
        "hdf5_path",
        "hdf5_sha256",
        "episode_index",
        "seed",
        "physical_seed",
        "cue_id",
        "split",
        "task_id",
        "task_text",
        "template_id",
        "scene_id",
        "object_combination_id",
        "behavior_id",
        "steps",
    }
    root = manifest_path.parent.resolve()
    tasks = set(str(task) for task in raw["tasks"])
    records: list[M1ManifestEpisode] = []
    for item_index, item in enumerate(raw["episodes"]):
        if not isinstance(item, dict):
            raise ValueError(f"manifest episode {item_index} must be an object")
        missing = sorted(required.difference(item))
        if missing:
            raise KeyError(f"manifest episode {item_index} missing keys: {missing}")
        relative_path = str(item["hdf5_path"])
        pure_path = PurePosixPath(relative_path)
        if (
            not relative_path
            or pure_path.is_absolute()
            or ".." in pure_path.parts
            or pure_path.suffix != ".hdf5"
        ):
            raise ValueError(
                f"manifest episode {item_index} has unsafe hdf5_path {relative_path!r}"
            )
        resolved = (root / Path(*pure_path.parts)).resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"episode path escapes manifest root: {relative_path}"
            ) from exc
        if not resolved.is_file():
            raise ValueError(f"episode path is not a file: {resolved}")
        sha256 = str(item["hdf5_sha256"])
        if _SHA256_RE.fullmatch(sha256) is None:
            raise ValueError(f"episode {item_index} has invalid hdf5_sha256")
        split = _normalize_split(str(item["split"]))
        task_id = str(item["task_id"])
        if task_id not in tasks:
            raise ValueError(f"episode {item_index} has unknown task_id {task_id!r}")
        cue_id = int(item["cue_id"])
        if cue_id not in CANONICAL_CUE_VARIANTS:
            raise ValueError(f"episode {item_index} has invalid cue_id {cue_id}")
        num_steps = int(item["steps"])
        if num_steps <= 0:
            raise ValueError(f"episode {item_index} must have positive steps")
        text_fields = (
            "task_text",
            "template_id",
            "scene_id",
            "object_combination_id",
            "behavior_id",
        )
        if any(not str(item[field]) for field in text_fields):
            raise ValueError(f"episode {item_index} contains an empty identity field")
        records.append(
            M1ManifestEpisode(
                path=resolved,
                relative_path=relative_path,
                hdf5_sha256=sha256,
                episode_index=int(item["episode_index"]),
                seed=int(item["seed"]),
                physical_seed=int(item["physical_seed"]),
                cue_id=cue_id,
                split=split,
                task_id=task_id,
                task_text=str(item["task_text"]),
                template_id=str(item["template_id"]),
                scene_id=str(item["scene_id"]),
                object_combination_id=str(item["object_combination_id"]),
                behavior_id=str(item["behavior_id"]),
                num_steps=num_steps,
            )
        )
    return records


def _audit_manifest_splits(
    raw: Mapping[str, Any], records: Sequence[M1ManifestEpisode]
) -> None:
    for field, values in (
        ("episode_index", [record.episode_index for record in records]),
        ("seed", [record.seed for record in records]),
        ("hdf5_path", [record.relative_path for record in records]),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"manifest {field} values must be globally unique")

    pairs: dict[tuple[str, int], list[M1ManifestEpisode]] = defaultdict(list)
    for record in records:
        pairs[(record.task_id, record.physical_seed)].append(record)
    for pair_key, pair_records in pairs.items():
        cue_ids = sorted(record.cue_id for record in pair_records)
        if cue_ids != list(CANONICAL_CUE_VARIANTS):
            raise ValueError(
                f"physical_seed cue pair {pair_key} must contain exactly "
                f"{CANONICAL_CUE_VARIANTS}, got {cue_ids}"
            )
        for field in (
            "split",
            "template_id",
            "scene_id",
            "object_combination_id",
            "task_text",
            "behavior_id",
        ):
            values = {getattr(record, field) for record in pair_records}
            if len(values) != 1:
                raise ValueError(
                    f"physical_seed cue pair {pair_key} is split across {field}: "
                    f"{sorted(values)}"
                )

    by_split = {
        split: [record for record in records if record.split == split]
        for split in CANONICAL_SPLITS
    }
    if any(not by_split[split] for split in CANONICAL_SPLITS):
        raise ValueError("manifest must contain train/validation/test episodes")
    for field in (
        "physical_seed",
        "template_id",
        "scene_id",
        "object_combination_id",
    ):
        sets = {
            split: {getattr(record, field) for record in split_records}
            for split, split_records in by_split.items()
        }
        for left_index, left in enumerate(CANONICAL_SPLITS):
            for right in CANONICAL_SPLITS[left_index + 1 :]:
                overlap = sets[left].intersection(sets[right])
                if overlap:
                    preview = sorted(overlap, key=str)[:3]
                    raise ValueError(
                        f"{field} overlaps between {left}/{right}: {preview}"
                    )

    split_counts = raw["split_counts"]
    if not isinstance(split_counts, dict) or set(split_counts) != set(CANONICAL_SPLITS):
        raise ValueError("manifest split_counts must contain train/validation/test")
    expected_tasks = set(str(task) for task in raw["tasks"])
    for split, split_records in by_split.items():
        declared = split_counts[split]
        if not isinstance(declared, dict):
            raise ValueError(f"split_counts.{split} must be an object")
        actual_cues = Counter(record.cue_id for record in split_records)
        expected = {
            "episodes": len(split_records),
            "physical_seeds": len({record.physical_seed for record in split_records}),
            "cue_counts": {
                str(cue): int(actual_cues.get(cue, 0)) for cue in CANONICAL_CUE_VARIANTS
            },
            "tasks": {record.task_id for record in split_records},
            "template_ids": {record.template_id for record in split_records},
        }
        if int(declared.get("episodes", -1)) != expected["episodes"]:
            raise ValueError(f"split_counts.{split}.episodes disagrees with episodes")
        if int(declared.get("physical_seeds", -1)) != expected["physical_seeds"]:
            raise ValueError(
                f"split_counts.{split}.physical_seeds disagrees with episodes"
            )
        declared_cues = {
            str(key): int(value)
            for key, value in dict(declared.get("cue_counts", {})).items()
        }
        if declared_cues != expected["cue_counts"]:
            raise ValueError(f"split_counts.{split}.cue_counts disagrees with episodes")
        if set(str(x) for x in declared.get("tasks", [])) != expected["tasks"]:
            raise ValueError(f"split_counts.{split}.tasks disagrees with episodes")
        if expected["tasks"] != expected_tasks:
            raise ValueError(f"split {split} does not cover every configured task")
        if (
            set(str(x) for x in declared.get("template_ids", []))
            != expected["template_ids"]
        ):
            raise ValueError(
                f"split_counts.{split}.template_ids disagrees with episodes"
            )


def _inspect_hdf5_episode(
    record: M1ManifestEpisode,
    *,
    camera_order: Sequence[str],
    control_hz: float,
    resolution: tuple[int, int],
) -> M1HDF5Metadata:
    with h5py.File(record.path, "r") as file:
        expected_attrs: dict[str, Any] = {
            "schema_profile": M1_SCHEMA_PROFILE,
            "schema_version": M1_SCHEMA_VERSION,
            "format_version": "wam.trajectory.hdf5/1",
            "episode_index": record.episode_index,
            "seed": record.seed,
            "num_steps": record.num_steps,
            "behavior_id": record.behavior_id,
            "task": record.task_text,
            "transition_semantics": "observation[t], action[t], observation[t+1]",
        }
        for key, expected in expected_attrs.items():
            if key not in file.attrs:
                raise ValueError(f"{record.path} is missing root attr {key!r}")
            actual = file.attrs[key]
            if isinstance(expected, str):
                actual = _text(actual)
            else:
                actual = int(actual)
            if actual != expected:
                raise ValueError(
                    f"{record.path} root attr {key}={actual!r}; expected {expected!r}"
                )
        fps = float(file.attrs.get("fps", np.nan))
        if not np.isfinite(fps) or not np.isclose(fps, control_hz, atol=1e-9):
            raise ValueError(
                f"{record.path} root attr fps={fps!r}; expected {control_hz}"
            )
        try:
            file_camera_order = tuple(
                str(x) for x in json.loads(_text(file.attrs["camera_order_json"]))
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{record.path} has invalid camera_order_json") from exc
        if file_camera_order != tuple(camera_order):
            raise ValueError(
                f"{record.path} camera order {file_camera_order}; expected "
                f"{tuple(camera_order)}"
            )
        _validate_episode_metadata_attr(
            file, record=record, camera_order=camera_order, resolution=resolution
        )

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
            "data/event/rendered_cue_variant",
        ]
        for camera in camera_order:
            required.extend(
                (
                    f"data/observation/images/{camera}",
                    f"data/observation/image_frame_index/{camera}",
                    f"data/next_observation/images/{camera}",
                    f"data/next_observation/image_frame_index/{camera}",
                )
            )
        missing = [name for name in required if name not in file]
        if missing:
            raise KeyError(f"{record.path} is missing required datasets: {missing}")
        for name in required:
            if file[name].shape[0] != record.num_steps:
                raise ValueError(
                    f"{record.path} dataset {name!r} length does not match num_steps"
                )

        state = file["data/observation/state"]
        next_state = file["data/next_observation/state"]
        commanded = file["data/action/commanded"]
        executed = file["data/action/executed"]
        if state.dtype != np.dtype(np.float32) or next_state.dtype != np.dtype(
            np.float32
        ):
            raise TypeError(f"{record.path} state datasets must be float32")
        if commanded.dtype != np.dtype(np.float32) or executed.dtype != np.dtype(
            np.float32
        ):
            raise TypeError(f"{record.path} action datasets must be float32")
        if state.ndim != 2 or next_state.shape != state.shape:
            raise ValueError(f"{record.path} state/next-state shapes disagree")
        if commanded.ndim != 2 or executed.shape != commanded.shape:
            raise ValueError(f"{record.path} commanded/executed shapes disagree")

        if not np.array_equal(
            np.asarray(file["data/frame_index"][:], dtype=np.int64),
            np.arange(record.num_steps, dtype=np.int64),
        ):
            raise ValueError(f"{record.path} frame_index must be contiguous from zero")
        timestamps = np.asarray(file["data/timestamp"][:], dtype=np.float64)
        if not np.isfinite(timestamps).all() or (
            len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0.0)
        ):
            raise ValueError(f"{record.path} timestamps must be finite and increasing")
        _require_constant_integer(
            file["data/episode_index"], record.episode_index, record.path
        )
        _require_constant_integer(file["data/seed"], record.seed, record.path)
        _require_constant_integer(
            file["data/event/rendered_cue_variant"], record.cue_id, record.path
        )
        _require_constant_text(file["data/task/id"], record.task_id, record.path)
        _require_constant_text(file["data/task/text"], record.task_text, record.path)
        if file["data/event/visual_signal_active"].dtype != np.dtype(np.bool_):
            raise TypeError(f"{record.path} visual_signal_active must be bool")
        if file["data/event/visual_signal_onset_step"].dtype != np.dtype(np.int64):
            raise TypeError(f"{record.path} visual_signal_onset_step must be int64")

        reference_obs: np.ndarray | None = None
        reference_next: np.ndarray | None = None
        image_shape_hwc: tuple[int, int, int] | None = None
        for camera in camera_order:
            observation_images = file[f"data/observation/images/{camera}"]
            next_images = file[f"data/next_observation/images/{camera}"]
            expected_shape = (record.num_steps, resolution[0], resolution[1], 3)
            if (
                observation_images.shape != expected_shape
                or next_images.shape != expected_shape
                or observation_images.dtype != np.dtype(np.uint8)
                or next_images.dtype != np.dtype(np.uint8)
            ):
                raise ValueError(
                    f"{record.path} camera {camera!r} must use matching uint8 HWC RGB "
                    f"shape {expected_shape}"
                )
            image_shape_hwc = tuple(int(x) for x in observation_images.shape[1:])
            obs_ids = np.asarray(
                file[f"data/observation/image_frame_index/{camera}"][:],
                dtype=np.int64,
            )
            next_ids = np.asarray(
                file[f"data/next_observation/image_frame_index/{camera}"][:],
                dtype=np.int64,
            )
            if np.any(obs_ids < 0) or np.any(next_ids < 0):
                raise ValueError(f"{record.path} contains negative image frame indices")
            if len(obs_ids) > 1 and (
                np.any(np.diff(obs_ids) < 0) or np.any(np.diff(next_ids) < 0)
            ):
                raise ValueError(f"{record.path} image frame indices must be monotonic")
            if len(obs_ids) > 1 and not np.array_equal(next_ids[:-1], obs_ids[1:]):
                raise ValueError(
                    f"{record.path} next/current image frame references disagree"
                )
            if reference_obs is None:
                reference_obs, reference_next = obs_ids, next_ids
            elif not np.array_equal(obs_ids, reference_obs) or not np.array_equal(
                next_ids, reference_next
            ):
                raise ValueError(
                    f"{record.path} camera frame indices are not synchronized"
                )
        assert image_shape_hwc is not None
        return M1HDF5Metadata(
            state_dim=int(state.shape[1]),
            action_dim=int(commanded.shape[1]),
            image_shape_hwc=image_shape_hwc,
            control_hz=fps,
        )


def _validate_episode_metadata_attr(
    file: h5py.File,
    *,
    record: M1ManifestEpisode,
    camera_order: Sequence[str],
    resolution: tuple[int, int],
) -> None:
    try:
        metadata = json.loads(_text(file.attrs["episode_metadata_json"]))
        randomization = json.loads(str(metadata["randomization_config"]))
        environment = json.loads(str(metadata["environment_config"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{record.path} has invalid episode_metadata_json") from exc
    expected_metadata = {
        "behavior_id": record.behavior_id,
        "schema_version": M1_SCHEMA_VERSION,
        "seed": record.seed,
        "task_id": record.task_id,
    }
    expected_randomization = {
        "cue_id": record.cue_id,
        "cue_variant": record.cue_id,
        "episode_seed": record.seed,
        "physical_seed": record.physical_seed,
        "split": record.split,
        "template_id": record.template_id,
        "scene_id": record.scene_id,
        "object_combination_id": record.object_combination_id,
    }
    expected_environment = {
        "camera_order": list(camera_order),
        "task_id": record.task_id,
        "randomization_template_id": record.template_id,
        "image_height": resolution[0],
        "image_width": resolution[1],
        "raw_unannotated": True,
    }
    for namespace, actual, expected in (
        ("episode_metadata", metadata, expected_metadata),
        ("randomization_config", randomization, expected_randomization),
        ("environment_config", environment, expected_environment),
    ):
        for key, expected_value in expected.items():
            if actual.get(key) != expected_value:
                raise ValueError(
                    f"{record.path} {namespace}.{key}={actual.get(key)!r}; "
                    f"expected {expected_value!r}"
                )


def _build_split_summary(
    *,
    split: str,
    records: Sequence[M1ManifestEpisode],
    schema_version: str,
    camera_order: Sequence[str],
    task_order: Sequence[str],
) -> dict[str, Any]:
    cue_counts = Counter(record.cue_id for record in records)
    task_counts: dict[str, dict[str, int]] = {}
    for task in task_order:
        selected = [record for record in records if record.task_id == task]
        task_counts[task] = {
            "episodes": len(selected),
            "transitions": sum(record.num_steps for record in selected),
            "physical_seed_pairs": len({record.physical_seed for record in selected}),
        }
    lineage = [
        {
            "episode_index": record.episode_index,
            "hdf5_path": record.relative_path,
            "hdf5_sha256": record.hdf5_sha256,
            "seed": record.seed,
            "physical_seed": record.physical_seed,
            "cue_id": record.cue_id,
            "task_id": record.task_id,
            "template_id": record.template_id,
            "scene_id": record.scene_id,
            "object_combination_id": record.object_combination_id,
            "steps": record.num_steps,
        }
        for record in sorted(records, key=lambda item: item.episode_index)
    ]
    return {
        "split": split,
        "schema_version": schema_version,
        "camera_order": list(camera_order),
        "task_order": list(task_order),
        "task_to_index": {task: index for index, task in enumerate(task_order)},
        "episodes": len(records),
        "transitions": sum(record.num_steps for record in records),
        "physical_seed_pairs": len({record.physical_seed for record in records}),
        "cue_counts": {
            str(cue): int(cue_counts.get(cue, 0)) for cue in CANONICAL_CUE_VARIANTS
        },
        "task_counts": task_counts,
        "template_ids": sorted({record.template_id for record in records}),
        "scene_count": len({record.scene_id for record in records}),
        "object_combination_count": len(
            {record.object_combination_id for record in records}
        ),
        "episode_lineage_sha256": _json_sha256(lineage),
    }


def _decision_anchors(
    event_active: np.ndarray, event_onset: np.ndarray
) -> tuple[int, ...]:
    if event_active.ndim != 1 or event_onset.shape != event_active.shape:
        raise ValueError("event arrays must be aligned 1D vectors")
    if event_active.size == 0:
        return (0,)
    unique_onsets = np.unique(event_onset)
    if unique_onsets.size != 1 or int(unique_onsets[0]) < 0:
        raise ValueError("visual_signal_onset_step must be one non-negative constant")
    # The exported transition event flag describes the corresponding next state,
    # so an onset at control step k first appears on transition row k-1.
    anchors = {max(0, int(unique_onsets[0]) - 1)}
    changes = np.flatnonzero(event_active[1:] != event_active[:-1]) + 1
    anchors.update(int(index) for index in changes.tolist())
    return tuple(sorted(anchors))


def _latest_unique_rows(
    frame_indices: np.ndarray, *, end_inclusive: int, count: int
) -> tuple[int, ...] | None:
    selected: list[int] = []
    seen: set[int] = set()
    for row in range(int(end_inclusive), -1, -1):
        frame_id = int(frame_indices[row])
        if frame_id in seen:
            continue
        selected.append(row)
        seen.add(frame_id)
        if len(selected) == count:
            return tuple(reversed(selected))
    return None


def _read_rgb_rows(
    file: h5py.File,
    *,
    prefix: str,
    rows: Sequence[int],
    cameras: Sequence[str],
) -> np.ndarray:
    per_camera: list[np.ndarray] = []
    row_list = list(int(row) for row in rows)
    for camera in cameras:
        hwc = np.asarray(file[f"{prefix}/{camera}"][row_list], dtype=np.uint8)
        per_camera.append(np.transpose(hwc, (0, 3, 1, 2)))
    # [V, C, 3, H, W], with an owned C-contiguous output for torch.
    return np.ascontiguousarray(np.stack(per_camera, axis=1))


def _normalize_split(split: str) -> str:
    normalized = _SPLIT_ALIASES.get(str(split), str(split))
    if normalized not in CANONICAL_SPLITS:
        raise ValueError(
            f"split must be one of {CANONICAL_SPLITS} (or 'val'), got {split!r}"
        )
    return normalized


def _require_constant_integer(dataset: h5py.Dataset, expected: int, path: Path) -> None:
    values = np.asarray(dataset[:], dtype=np.int64)
    if values.ndim != 1 or not np.all(values == int(expected)):
        raise ValueError(f"{path} dataset {dataset.name!r} is not constant {expected}")


def _require_constant_text(dataset: h5py.Dataset, expected: str, path: Path) -> None:
    try:
        values = dataset.asstr()[:]
    except TypeError:
        values = np.asarray([_text(value) for value in dataset[:]], dtype=object)
    if values.ndim != 1 or not np.all(values == expected):
        raise ValueError(
            f"{path} dataset {dataset.name!r} is not constant {expected!r}"
        )


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


__all__ = [
    "CANONICAL_CAMERA_ORDER",
    "M1_CAUSAL_PAIR_HORIZON",
    "M1_CONTROLLED_ACTION_DIMS",
    "M1_STATE_CAUSAL_GAP",
    "M1_STATE_CAUSAL_FEEDBACK_EQUALITY_ATOL",
    "M1_STATE_CAUSAL_HISTORY_WIDTH",
    "M1_STATE_CAUSAL_LATERAL_ACTION_DIMS",
    "M1_STATE_CAUSAL_MIN_ACTION_DELTA",
    "M1_STATE_CAUSAL_PAST_ACTIONS",
    "M1_STATE_CAUSAL_ROBOT_X_STATE_DIMS",
    "M1_STATE_CAUSAL_STATES",
    "M1_STATE_CAUSAL_ZERO_DELTA_ACTION_DIMS",
    "M1CausalPairDataset",
    "M1ManifestEpisode",
    "M1ManifestIndex",
    "M1SampleLineage",
    "M1StateCausalPairDataset",
    "M1WindowDataset",
    "load_m1_manifest",
]
