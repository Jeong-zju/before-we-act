"""Manifest-bound generic multimodal trajectory windows for scratch M1.

This protocol intentionally contains no visual-cue, event, paired-reset, or
privileged labels.  The training manifest owns episode splits and the number of
selected transitions in each HDF5 file.  The loader never discovers files or
re-splits episodes on its own.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
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

from models.wam.action_codec import (
    CANONICAL_ACTION_DOMAIN,
    AffineActionCodec,
    AffineActionCodecConfig,
)
from models.wam.normalizer import NormalizationStats


GENERIC_M1_MANIFEST_FORMAT = "wam.multimodal.trajectory.training_manifest/1"
GENERIC_M1_DATASET_PROTOCOL = "generic_multimodal_trajectory"
GENERIC_M1_SPLITS = ("train", "validation", "test")

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SPLIT_ALIASES = {"val": "validation", "valid": "validation"}


@dataclass(frozen=True)
class GenericM1Episode:
    """One immutable episode entry from the generic training manifest."""

    path: Path
    relative_path: str
    hdf5_sha256: str
    episode_index: int
    source_episode_id: int
    seed: int
    split: str
    task_id: str
    task_text: str
    num_steps: int
    recorded_steps: int
    success: bool


@dataclass(frozen=True)
class GenericM1HDF5Metadata:
    """Shape metadata collected while validating an episode."""

    state_dim: int
    action_dim: int
    image_shape_hwc: tuple[int, int, int]
    control_hz: float


@dataclass(frozen=True)
class GenericM1SampleLineage:
    """Offline sample identity; never returned by ``__getitem__``."""

    path: Path
    decision_t: int


@dataclass(frozen=True)
class _GenericM1Window:
    record_index: int
    decision_t: int
    visual_rows: tuple[int, ...]


class GenericM1ManifestIndex:
    """Strict parser for ``generic_multimodal_trajectory`` manifests."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        manifest_sha256: str,
        raw_manifest: Mapping[str, Any],
        episodes: Sequence[GenericM1Episode],
        hdf5_metadata: Mapping[int, GenericM1HDF5Metadata],
        hdf5_sha256_verified: bool,
        hdf5_contract_verified: bool,
        normalization_verified: bool,
    ) -> None:
        self.manifest_path = manifest_path
        self.manifest_sha256 = manifest_sha256
        self.raw_manifest = MappingProxyType(dict(raw_manifest))
        self.episodes = tuple(episodes)
        self.hdf5_metadata = MappingProxyType(dict(hdf5_metadata))
        self.hdf5_sha256_verified = bool(hdf5_sha256_verified)
        self.hdf5_contract_verified = bool(hdf5_contract_verified)
        self.normalization_verified = bool(normalization_verified)

        schema = _mapping(raw_manifest, "schema")
        state = _mapping(raw_manifest, "state")
        action = _mapping(raw_manifest, "action")
        vision = _mapping(raw_manifest, "vision")
        timing = _mapping(raw_manifest, "timing")
        normalization = _mapping(raw_manifest, "normalization")
        self.schema_profile = str(schema["profile"])
        self.schema_version = str(schema["version"])
        self.state_field = str(state["field"])
        self.next_state_field = str(state["next_field"])
        self.action_field = str(action["field"])
        self.history_action_field = str(action["history_field"])
        self.executed_action_field = str(action["executed_field"])
        self.action_storage_domain = str(
            action.get("storage_domain", action.get("domain", ""))
        )
        self.action_domain = str(action["domain"])
        codec = _mapping(action, "codec")
        self.action_codec: AffineActionCodec | None = None
        if codec.get("applied") is True:
            codec_config = AffineActionCodecConfig.from_dict(
                _mapping(codec, "config")
            )
            expected_codec_sha256 = str(codec.get("semantic_sha256", ""))
            if codec_config.sha256() != expected_codec_sha256:
                raise ValueError("action codec semantic SHA256 disagrees with config")
            self.action_codec = AffineActionCodec(codec_config)
        self.state_dim = int(state["dimension"])
        self.action_dim = int(action["dimension"])
        self.current_image_prefix = str(vision["current_prefix"])
        self.next_image_prefix = str(vision["next_prefix"])
        self.camera_order = tuple(str(value) for value in vision["camera_order"])
        self.control_hz = float(timing["control_hz"])
        self.image_hz = float(timing["image_hz"])
        self.normalization_path = _resolve_relative_file(
            manifest_path.parent,
            str(normalization["path"]),
            field="normalization.path",
        )
        self.normalization_sha256 = str(normalization["semantic_sha256"])
        self.transition_selection = str(
            _mapping(raw_manifest, "transition_selection")["mode"]
        )

        primary_task = str(_mapping(raw_manifest, "task")["id"])
        discovered_tasks = sorted({record.task_id for record in self.episodes})
        self.task_order = tuple(
            [primary_task]
            + [task_id for task_id in discovered_tasks if task_id != primary_task]
        )
        self.task_to_index = MappingProxyType(
            {task_id: index for index, task_id in enumerate(self.task_order)}
        )

        grouped: dict[str, list[GenericM1Episode]] = {
            split: [] for split in GENERIC_M1_SPLITS
        }
        for episode in self.episodes:
            grouped[episode.split].append(episode)
        self._by_split = {
            split: tuple(sorted(values, key=lambda item: item.episode_index))
            for split, values in grouped.items()
        }
        self._split_summaries = {
            split: _build_split_summary(self, split, records)
            for split, records in self._by_split.items()
        }

    @classmethod
    def from_path(
        cls,
        manifest_path: str | Path,
        *,
        verify_hdf5_sha256: bool = True,
        verify_hdf5_contract: bool = True,
        verify_normalization: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> "GenericM1ManifestIndex":
        """Load and fail-closed audit a generic M1 training manifest."""

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
        records = _parse_episodes(raw, manifest_path=path)
        _audit_manifest(raw, records)

        metadata: dict[int, GenericM1HDF5Metadata] = {}
        if progress_callback is not None:
            progress_callback(0, len(records))
        for position, record in enumerate(records, start=1):
            if verify_hdf5_sha256:
                actual = _sha256_file(record.path)
                if actual != record.hdf5_sha256:
                    raise ValueError(
                        f"{record.path} SHA256 {actual} does not match manifest "
                        f"{record.hdf5_sha256}"
                    )
            if verify_hdf5_contract:
                metadata[record.episode_index] = _inspect_hdf5_episode(
                    record,
                    raw=raw,
                )
            if progress_callback is not None:
                progress_callback(position, len(records))

        normalization_verified = False
        if verify_normalization:
            _verify_normalization(path.parent, raw)
            normalization_verified = True

        return cls(
            manifest_path=path,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            raw_manifest=raw,
            episodes=records,
            hdf5_metadata=metadata,
            hdf5_sha256_verified=verify_hdf5_sha256,
            hdf5_contract_verified=verify_hdf5_contract,
            normalization_verified=normalization_verified,
        )

    def records_for_split(self, split: str) -> tuple[GenericM1Episode, ...]:
        return self._by_split[_normalize_split(split)]

    def split_summary(self, split: str) -> dict[str, Any]:
        normalized = _normalize_split(split)
        return json.loads(_canonical_json(self._split_summaries[normalized]))

    def split_summary_sha256(self, split: str) -> str:
        return _json_sha256(self.split_summary(split))

    def checkpoint_lineage(self, split: str) -> dict[str, Any]:
        normalized = _normalize_split(split)
        return {
            "manifest_format": GENERIC_M1_MANIFEST_FORMAT,
            "dataset_protocol": GENERIC_M1_DATASET_PROTOCOL,
            "schema_profile": self.schema_profile,
            "schema_version": self.schema_version,
            "manifest_sha256": self.manifest_sha256,
            "split": normalized,
            "split_summary_sha256": self.split_summary_sha256(normalized),
            "camera_order": list(self.camera_order),
            "task_order": list(self.task_order),
            "task_to_index": dict(self.task_to_index),
            "transition_selection": self.transition_selection,
            "normalization_sha256": self.normalization_sha256,
            "action_domain": self.action_domain,
            "action_codec_sha256": (
                None
                if self.action_codec is None
                else self.action_codec.semantic_sha256
            ),
            "hdf5_sha256_verified": self.hdf5_sha256_verified,
            "hdf5_contract_verified": self.hdf5_contract_verified,
            "normalization_verified": self.normalization_verified,
        }

    def load_normalization(self) -> NormalizationStats:
        """Load the already-verified train-only normalization artifact."""

        stats = NormalizationStats.load(self.normalization_path)
        if stats.sha256() != self.normalization_sha256:
            raise ValueError("normalization semantic SHA256 drifted after indexing")
        return stats

    def __getstate__(self) -> dict[str, Any]:
        """Make the audited index safe for spawn/forkserver DataLoader workers."""

        state = dict(self.__dict__)
        state["raw_manifest"] = dict(self.raw_manifest)
        state["hdf5_metadata"] = dict(self.hdf5_metadata)
        state["task_to_index"] = dict(self.task_to_index)
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        values = dict(state)
        values["raw_manifest"] = MappingProxyType(dict(values["raw_manifest"]))
        values["hdf5_metadata"] = MappingProxyType(
            dict(values["hdf5_metadata"])
        )
        values["task_to_index"] = MappingProxyType(dict(values["task_to_index"]))
        self.__dict__.update(values)


class GenericM1WindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Episode-safe generic windows with the deployable M1 sample contract."""

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
            "future_states",
            "future_images",
            "future_image_novelty_mask",
            "future_horizons",
        }
    )
    HORIZON_MASK_KEYS = frozenset(
        {
            "action_target_valid_mask",
            "future_state_valid_mask",
            "future_visual_valid_mask",
        }
    )
    AVAILABLE_SAMPLE_KEYS = SAMPLE_KEYS | HORIZON_MASK_KEYS

    def __init__(
        self,
        manifest: GenericM1ManifestIndex | str | Path,
        *,
        split: str,
        state_history: int = 32,
        action_chunk: int = 8,
        cameras: Sequence[str] | None = None,
        visual_history: int = 2,
        future_horizons: Sequence[int] = (1, 2, 4, 8),
        stride: int = 1,
        hdf5_cache_size: int = 8,
        allow_incomplete_horizon: bool = False,
        allow_incomplete_visual_history: bool = False,
        verify_hdf5_sha256: bool = True,
        verify_hdf5_contract: bool = True,
        verify_normalization: bool = True,
    ) -> None:
        if isinstance(manifest, GenericM1ManifestIndex):
            manifest_index = manifest
        else:
            manifest_index = GenericM1ManifestIndex.from_path(
                manifest,
                verify_hdf5_sha256=verify_hdf5_sha256,
                verify_hdf5_contract=verify_hdf5_contract,
                verify_normalization=verify_normalization,
            )
        for name, value in (
            ("state_history", state_history),
            ("action_chunk", action_chunk),
            ("visual_history", visual_history),
            ("stride", stride),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(hdf5_cache_size) < 0:
            raise ValueError("hdf5_cache_size cannot be negative")

        requested_cameras = tuple(
            manifest_index.camera_order if cameras is None else map(str, cameras)
        )
        if not requested_cameras or len(requested_cameras) != len(
            set(requested_cameras)
        ):
            raise ValueError("cameras must be non-empty and unique")
        unknown = set(requested_cameras).difference(manifest_index.camera_order)
        if unknown:
            raise ValueError(f"unknown cameras: {sorted(unknown)}")
        ordered_subset = tuple(
            camera
            for camera in manifest_index.camera_order
            if camera in requested_cameras
        )
        if requested_cameras != ordered_subset:
            raise ValueError(
                "cameras must preserve manifest order "
                f"{manifest_index.camera_order}"
            )

        horizons = tuple(int(value) for value in future_horizons)
        if not horizons or any(value <= 0 for value in horizons):
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
        self.hdf5_cache_size = int(hdf5_cache_size)
        self.allow_incomplete_horizon = bool(allow_incomplete_horizon)
        self.allow_incomplete_visual_history = bool(
            allow_incomplete_visual_history
        )
        self.task_order = manifest_index.task_order
        self.task_to_index = manifest_index.task_to_index
        self.records = manifest_index.records_for_split(self.split)
        if not self.records:
            raise ValueError(f"manifest split {self.split!r} contains no episodes")

        self._cache_pid = os.getpid()
        self._cache: OrderedDict[str, h5py.File] = OrderedDict()
        self._ram_cache: dict[str, dict[str, torch.Tensor]] = {}
        self._ram_backing: dict[str, torch.Tensor] = {}
        self._ram_preload_paths: frozenset[str] = frozenset()
        self._ram_preload_shared = False
        self._index: list[_GenericM1Window] = []
        image_shape: tuple[int, int, int] | None = None
        complete_horizon = max(self.action_chunk, max(self.future_horizons))
        for record_index, record in enumerate(self.records):
            metadata = manifest_index.hdf5_metadata.get(record.episode_index)
            if metadata is None:
                metadata = _inspect_hdf5_episode(
                    record,
                    raw=manifest_index.raw_manifest,
                )
            if metadata.state_dim != manifest_index.state_dim:
                raise ValueError(f"{record.path} state dimension drifted")
            if metadata.action_dim != manifest_index.action_dim:
                raise ValueError(f"{record.path} action dimension drifted")
            if image_shape is None:
                image_shape = metadata.image_shape_hwc
            elif metadata.image_shape_hwc != image_shape:
                raise ValueError(f"{record.path} image shape differs across episodes")

            with h5py.File(record.path, "r") as file:
                frame_indices = np.asarray(
                    file[
                        f"data/observation/image_frame_index/"
                        f"{self.cameras[0]}"
                    ][: record.num_steps],
                    dtype=np.int64,
                )
            last_decision = (
                record.num_steps - 1
                if self.allow_incomplete_horizon
                else record.num_steps - complete_horizon
            )
            for decision_t in range(0, last_decision + 1, self.stride):
                rows = _latest_unique_rows(
                    frame_indices,
                    end_inclusive=decision_t,
                    count=self.visual_history,
                )
                if rows is None and self.allow_incomplete_visual_history:
                    rows = _latest_available_unique_rows(
                        frame_indices,
                        end_inclusive=decision_t,
                        maximum_count=self.visual_history,
                    )
                if rows is not None:
                    self._index.append(
                        _GenericM1Window(record_index, decision_t, rows)
                    )

        if not self._index:
            raise RuntimeError(
                f"split {self.split!r} has no windows with complete horizon "
                f"{complete_horizon} and {self.visual_history} distinct RGB frames"
            )
        assert image_shape is not None
        self.state_dim = manifest_index.state_dim
        self.action_dim = manifest_index.action_dim
        self.image_shape_hwc = image_shape

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        requested = (
            self.AVAILABLE_SAMPLE_KEYS
            if self.allow_incomplete_horizon
            else self.SAMPLE_KEYS
        )
        return self._projected_item(index, requested)

    def project(self, sample_keys: Collection[str]) -> Dataset[dict[str, torch.Tensor]]:
        requested = frozenset(str(name) for name in sample_keys)
        if not requested:
            raise ValueError("M1 projected sample keys cannot be empty")
        unknown = requested.difference(self.AVAILABLE_SAMPLE_KEYS)
        if unknown:
            raise ValueError(f"unknown M1 projected sample keys: {sorted(unknown)}")
        if requested.intersection(self.HORIZON_MASK_KEYS) and not (
            self.allow_incomplete_horizon
        ):
            raise ValueError(
                "horizon validity masks require allow_incomplete_horizon=True"
            )
        return _GenericM1Projection(self, requested)

    def estimate_ram_preload_bytes(
        self,
        sample_keys: Collection[str] | None = None,
    ) -> int:
        """Return the exact uncompressed byte count for a RAM-backed split."""

        paths = self._hdf5_paths_for_sample_keys(sample_keys)
        total = 0
        for record in self.records:
            with h5py.File(record.path, "r") as file:
                for path in paths:
                    dataset = file[path]
                    elements = int(np.prod((record.num_steps, *dataset.shape[1:])))
                    total += elements * int(dataset.dtype.itemsize)
        return total

    def preload_to_ram(
        self,
        sample_keys: Collection[str] | None = None,
        *,
        shared_memory: bool = True,
        progress_callback: Callable[[int, int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Decode selected HDF5 fields once into worker-shared CPU RAM.

        Shared tensors avoid one decompressed copy per DataLoader worker.  The
        manifest/HDF5 audit remains authoritative; this cache only replaces
        repeated runtime reads of already-verified numeric datasets.
        """

        paths = self._hdf5_paths_for_sample_keys(sample_keys)
        if self._ram_cache:
            missing = paths.difference(self._ram_preload_paths)
            if missing:
                raise RuntimeError(
                    "RAM preload is already materialized without required fields: "
                    f"{sorted(missing)}"
                )
            return self.ram_preload_summary()

        total_steps = sum(record.num_steps for record in self.records)
        first = self.records[0]
        with h5py.File(first.path, "r") as file:
            for path in sorted(paths):
                dataset = file[path]
                numpy_dtype = np.empty((), dtype=dataset.dtype)
                dtype = torch.from_numpy(numpy_dtype).dtype
                tensor = torch.empty(
                    (total_steps, *dataset.shape[1:]),
                    dtype=dtype,
                )
                if shared_memory:
                    tensor = tensor.share_memory_()
                self._ram_backing[path] = tensor

        total_bytes = 0
        offset = 0
        if progress_callback is not None:
            progress_callback(0, len(self.records), 0)
        for position, record in enumerate(self.records, start=1):
            episode: dict[str, torch.Tensor] = {}
            stop = offset + record.num_steps
            with h5py.File(record.path, "r") as file:
                for path in sorted(paths):
                    values = np.ascontiguousarray(file[path][: record.num_steps])
                    target = self._ram_backing[path][offset:stop]
                    target.copy_(torch.from_numpy(values))
                    episode[path] = target
                    total_bytes += target.numel() * target.element_size()
            self._ram_cache[str(record.path)] = episode
            offset = stop
            if progress_callback is not None:
                progress_callback(position, len(self.records), total_bytes)

        self._ram_preload_paths = paths
        self._ram_preload_shared = bool(shared_memory)
        return self.ram_preload_summary()

    def ram_preload_summary(self) -> dict[str, Any]:
        total_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in self._ram_backing.values()
        )
        return {
            "enabled": bool(self._ram_cache),
            "split": self.split,
            "episodes": len(self._ram_cache),
            "fields": sorted(self._ram_preload_paths),
            "bytes": total_bytes,
            "shared_memory": self._ram_preload_shared,
        }

    def clear_ram_preload(self) -> None:
        """Release the parent process' shared-RAM dataset references."""

        self._ram_cache.clear()
        self._ram_backing.clear()
        self._ram_preload_paths = frozenset()
        self._ram_preload_shared = False

    def _hdf5_paths_for_sample_keys(
        self,
        sample_keys: Collection[str] | None,
    ) -> frozenset[str]:
        requested = (
            self.SAMPLE_KEYS
            if sample_keys is None
            else frozenset(str(name) for name in sample_keys)
        )
        unknown = requested.difference(self.AVAILABLE_SAMPLE_KEYS)
        if unknown:
            raise ValueError(f"unknown M1 RAM preload sample keys: {sorted(unknown)}")
        paths: set[str] = set()
        if "states" in requested:
            paths.add(self.manifest.state_field)
        if "past_actions" in requested:
            paths.add(self.manifest.history_action_field)
        if "action_targets" in requested:
            paths.add(self.manifest.action_field)
        if "future_states" in requested:
            paths.add(self.manifest.next_state_field)
        if "images" in requested:
            paths.update(
                f"{self.manifest.current_image_prefix}/{camera}"
                for camera in self.cameras
            )
        if "future_images" in requested:
            paths.update(
                f"{self.manifest.next_image_prefix}/{camera}"
                for camera in self.cameras
            )
        if "future_image_novelty_mask" in requested:
            for camera in self.cameras:
                paths.add(f"data/observation/image_frame_index/{camera}")
                paths.add(f"data/next_observation/image_frame_index/{camera}")
        return frozenset(paths)

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
                    file[self.manifest.state_field][
                        history_start : decision_t + 1
                    ],
                    dtype=np.float32,
                )
                mask = np.zeros(self.state_history, dtype=np.bool_)
                mask[history_offset:] = True
                if "states" in requested:
                    sample["states"] = torch.from_numpy(states)
                if "state_valid_mask" in requested:
                    sample["state_valid_mask"] = torch.from_numpy(mask)

            if {"past_actions", "past_action_valid_mask"}.intersection(requested):
                past_actions = np.zeros(
                    (max(0, self.state_history - 1), self.action_dim),
                    dtype=np.float32,
                )
                mask = np.zeros(max(0, self.state_history - 1), dtype=np.bool_)
                if self.state_history > 1 and decision_t > history_start:
                    values = np.asarray(
                        file[self.manifest.history_action_field][
                            history_start:decision_t
                        ],
                        dtype=np.float32,
                    )
                    if self.manifest.action_codec is not None:
                        values = self.manifest.action_codec.encode(values)
                    past_actions[history_offset:] = values
                    mask[history_offset:] = True
                if "past_actions" in requested:
                    sample["past_actions"] = torch.from_numpy(past_actions)
                if "past_action_valid_mask" in requested:
                    sample["past_action_valid_mask"] = torch.from_numpy(mask)

            if {"images", "image_valid_mask"}.intersection(requested):
                image_valid = np.zeros(
                    (self.visual_history, len(self.cameras)), dtype=np.bool_
                )
                image_start = self.visual_history - len(window.visual_rows)
                image_valid[image_start:] = True
                if "images" in requested:
                    image_height, image_width, channels = self.image_shape_hwc
                    if channels != 3:  # pragma: no cover - manifest invariant.
                        raise RuntimeError("generic M1 images must be RGB")
                    images = np.zeros(
                        (
                            self.visual_history,
                            len(self.cameras),
                            3,
                            image_height,
                            image_width,
                        ),
                        dtype=np.uint8,
                    )
                    images[image_start:] = _read_rgb_rows(
                        file,
                        prefix=self.manifest.current_image_prefix,
                        rows=window.visual_rows,
                        cameras=self.cameras,
                    )
                    sample["images"] = torch.from_numpy(images)
                if "image_valid_mask" in requested:
                    sample["image_valid_mask"] = torch.from_numpy(image_valid)
            if "action_targets" in requested:
                available = min(self.action_chunk, record.num_steps - decision_t)
                values = np.asarray(
                    file[self.manifest.action_field][
                        decision_t : decision_t + available
                    ],
                    dtype=np.float32,
                )
                if self.manifest.action_codec is not None:
                    values = self.manifest.action_codec.encode(values)
                if self.allow_incomplete_horizon:
                    values = _repeat_last_to_horizon(values, self.action_chunk)
                sample["action_targets"] = torch.from_numpy(values.copy())
            if "future_states" in requested:
                available = min(self.action_chunk, record.num_steps - decision_t)
                values = np.asarray(
                    file[self.manifest.next_state_field][
                        decision_t : decision_t + available
                    ],
                    dtype=np.float32,
                )
                if self.allow_incomplete_horizon:
                    values = _repeat_last_to_horizon(values, self.action_chunk)
                sample["future_states"] = torch.from_numpy(values.copy())

            remaining = record.num_steps - decision_t
            future_rows = tuple(
                (
                    decision_t + min(horizon, remaining) - 1
                    if self.allow_incomplete_horizon
                    else decision_t + horizon - 1
                )
                for horizon in self.future_horizons
            )
            if "future_images" in requested:
                sample["future_images"] = torch.from_numpy(
                    _read_rgb_rows(
                        file,
                        prefix=self.manifest.next_image_prefix,
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
                        file[f"data/observation/image_frame_index/{camera}"][
                            decision_t
                        ]
                    )
                    unique_rows, inverse = np.unique(
                        np.asarray(future_rows, dtype=np.int64),
                        return_inverse=True,
                    )
                    target_ids = np.asarray(
                        file[f"data/next_observation/image_frame_index/{camera}"][
                            unique_rows.tolist()
                        ],
                        dtype=np.int64,
                    )[inverse]
                    for horizon_index, target_id in enumerate(target_ids.tolist()):
                        novelty[horizon_index, camera_index] = target_id != previous
                        previous = int(target_id)
                sample["future_image_novelty_mask"] = torch.from_numpy(novelty)
            if "action_target_valid_mask" in requested:
                sample["action_target_valid_mask"] = torch.arange(
                    self.action_chunk
                ).lt(remaining)
            if "future_state_valid_mask" in requested:
                sample["future_state_valid_mask"] = torch.arange(
                    self.action_chunk
                ).lt(remaining)
            if "future_visual_valid_mask" in requested:
                sample["future_visual_valid_mask"] = torch.tensor(
                    [horizon <= remaining for horizon in self.future_horizons],
                    dtype=torch.bool,
                )

        if frozenset(sample) != requested:  # pragma: no cover - invariant.
            raise RuntimeError("generic M1 projected sample key contract drifted")
        return sample

    def sample_lineage(self, index: int) -> GenericM1SampleLineage:
        window = self._window(index)
        return GenericM1SampleLineage(
            path=self.records[window.record_index].path,
            decision_t=window.decision_t,
        )

    @property
    def decision_window_indices(self) -> tuple[int, ...]:
        """Generic trajectories have no event-derived decision windows."""

        return ()

    @property
    def observationally_ambiguous_window_indices(self) -> tuple[int, ...]:
        """No cue-pair ambiguity audit is part of the generic protocol."""

        return ()

    def sampling_weights(self, *, decision_window_boost: float = 1.0) -> torch.Tensor:
        boost = float(decision_window_boost)
        if not np.isfinite(boost) or boost <= 0.0:
            raise ValueError("decision_window_boost must be finite and positive")
        if boost != 1.0:
            raise ValueError(
                "generic trajectory protocol has no decision windows; "
                "decision_window_boost must be 1.0"
            )
        task_indices = np.asarray(
            [
                self.task_to_index[self.records[window.record_index].task_id]
                for window in self._index
            ],
            dtype=np.int64,
        )
        weights = np.zeros(len(self._index), dtype=np.float64)
        present = np.unique(task_indices)
        for task_index in present.tolist():
            mask = task_indices == task_index
            weights[mask] = 1.0 / (float(mask.sum()) * len(present))
        return torch.from_numpy(weights)

    def make_weighted_sampler(
        self,
        *,
        num_samples: int | None = None,
        decision_window_boost: float = 1.0,
        seed: int = 0,
    ) -> WeightedRandomSampler:
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
        windows_by_task = Counter(
            self.records[window.record_index].task_id for window in self._index
        )
        return {
            "dataset_protocol": GENERIC_M1_DATASET_PROTOCOL,
            "split": self.split,
            "windows": len(self),
            "windows_by_task": dict(sorted(windows_by_task.items())),
            "state_history": self.state_history,
            "action_chunk": self.action_chunk,
            "visual_history": self.visual_history,
            "allow_incomplete_horizon": self.allow_incomplete_horizon,
            "allow_incomplete_visual_history": (
                self.allow_incomplete_visual_history
            ),
            "visual_history_alignment": "deployable_suffix_left_padding",
            "cameras": list(self.cameras),
            "future_horizons": list(self.future_horizons),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "image_shape_hwc": list(self.image_shape_hwc),
            "sample_keys": sorted(self.SAMPLE_KEYS),
            "decision_windows": 0,
            "causal_pairs": "not_supported",
            "transition_selection": self.manifest.transition_selection,
            "action_domain": self.manifest.action_domain,
            "action_codec_sha256": (
                None
                if self.manifest.action_codec is None
                else self.manifest.action_codec.semantic_sha256
            ),
        }

    def checkpoint_lineage(self) -> dict[str, Any]:
        summary = self.window_summary()
        return {
            **self.manifest.checkpoint_lineage(self.split),
            "window_summary": summary,
            "window_summary_sha256": _json_sha256(summary),
        }

    def close(self) -> None:
        for file in self._cache.values():
            file.close()
        self._cache.clear()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup.
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        state["_cache_pid"] = os.getpid()
        state["task_to_index"] = dict(self.task_to_index)
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        values = dict(state)
        values["task_to_index"] = MappingProxyType(dict(values["task_to_index"]))
        self.__dict__.update(values)

    def _window(self, index: int) -> _GenericM1Window:
        normalized = int(index)
        if normalized < 0:
            normalized += len(self._index)
        if normalized < 0 or normalized >= len(self._index):
            raise IndexError(index)
        return self._index[normalized]

    @contextmanager
    def _episode_file(
        self, path: Path
    ) -> Iterator[h5py.File | Mapping[str, torch.Tensor]]:
        ram_episode = self._ram_cache.get(str(path))
        if ram_episode is not None:
            yield ram_episode
            return
        if self.hdf5_cache_size == 0:
            with h5py.File(path, "r") as file:
                yield file
            return
        if os.getpid() != self._cache_pid:
            self.close()
            self._cache_pid = os.getpid()
        key = str(path)
        file = self._cache.pop(key, None)
        if file is None:
            file = h5py.File(path, "r")
        self._cache[key] = file
        while len(self._cache) > self.hdf5_cache_size:
            _, evicted = self._cache.popitem(last=False)
            evicted.close()
        yield file


class _GenericM1Projection(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        source: GenericM1WindowDataset,
        sample_keys: Collection[str],
    ) -> None:
        self.source = source
        self.sample_keys = frozenset(sample_keys)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.source._projected_item(index, self.sample_keys)


def _validate_manifest_header(raw: Mapping[str, Any]) -> None:
    if raw.get("format_version") != GENERIC_M1_MANIFEST_FORMAT:
        raise ValueError(
            f"generic manifest format must be {GENERIC_M1_MANIFEST_FORMAT!r}"
        )
    if raw.get("dataset_protocol") != GENERIC_M1_DATASET_PROTOCOL:
        raise ValueError(
            f"dataset_protocol must be {GENERIC_M1_DATASET_PROTOCOL!r}"
        )
    schema = _mapping(raw, "schema")
    for key in ("profile", "version", "hdf5_format_version"):
        if not str(schema.get(key, "")):
            raise ValueError(f"schema.{key} cannot be empty")
    state = _mapping(raw, "state")
    action = _mapping(raw, "action")
    for key in ("field", "next_field"):
        _validate_hdf5_path(str(state.get(key, "")), field=f"state.{key}")
    for key in ("field", "history_field", "executed_field"):
        _validate_hdf5_path(str(action.get(key, "")), field=f"action.{key}")
    for name, value in (
        ("state.dimension", state.get("dimension")),
        ("action.dimension", action.get("dimension")),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if state.get("dtype") != "float32" or action.get("dtype") != "float32":
        raise ValueError("generic M1 state/action dtype must be float32")
    action_domain = str(action.get("domain", ""))
    if not action_domain:
        raise ValueError("action.domain cannot be empty")
    codec = _mapping(action, "codec")
    applied = codec.get("applied")
    if not isinstance(applied, bool):
        raise ValueError("action.codec.applied must be boolean")
    if applied:
        codec_config = AffineActionCodecConfig.from_dict(
            _mapping(codec, "config")
        )
        if codec_config.action_dim != int(action["dimension"]):
            raise ValueError("action codec dimension disagrees with action.dimension")
        if action_domain != codec_config.encoded_domain:
            raise ValueError("action.domain disagrees with the applied codec")
        if str(action.get("storage_domain", "")) != codec_config.raw_domain:
            raise ValueError("action.storage_domain disagrees with the codec")
        if str(codec.get("semantic_sha256", "")) != codec_config.sha256():
            raise ValueError("action codec semantic SHA256 disagrees with config")
    elif action_domain == CANONICAL_ACTION_DOMAIN:
        raise ValueError("canonical action domain requires an applied codec")

    vision = _mapping(raw, "vision")
    cameras = vision.get("camera_order")
    if not isinstance(cameras, list) or not cameras or any(
        not isinstance(value, str) or not value for value in cameras
    ):
        raise ValueError("vision.camera_order must contain non-empty strings")
    if len(cameras) != len(set(cameras)):
        raise ValueError("vision.camera_order must contain unique cameras")
    for key in ("current_prefix", "next_prefix"):
        _validate_hdf5_path(str(vision.get(key, "")), field=f"vision.{key}")
    timing = _mapping(raw, "timing")
    for key in ("control_hz", "image_hz"):
        value = float(timing.get(key, np.nan))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"timing.{key} must be finite and positive")

    task = _mapping(raw, "task")
    if not str(task.get("id", "")) or not str(task.get("text", "")):
        raise ValueError("task.id and task.text cannot be empty")
    selection = _mapping(raw, "transition_selection")
    if selection.get("mode") not in {
        "all-recorded",
        "through-first-done-inclusive",
    }:
        raise ValueError("unsupported transition_selection.mode")
    _validate_hdf5_path(
        str(selection.get("terminal_field", "")),
        field="transition_selection.terminal_field",
    )
    split = _mapping(raw, "split_protocol")
    if split.get("unit") != "episode_seed":
        raise ValueError("generic M1 split unit must be episode_seed")
    if split.get("claim") != "seed_disjoint_only":
        raise ValueError("generic M1 split claim must be seed_disjoint_only")
    if not _SHA256_RE.fullmatch(str(split.get("assignment_sha256", ""))):
        raise ValueError("split_protocol.assignment_sha256 must be SHA256")
    counts = _mapping(raw, "split_counts")
    if set(counts) != set(GENERIC_M1_SPLITS):
        raise ValueError("split_counts must contain train/validation/test")
    normalization = _mapping(raw, "normalization")
    for key in ("file_sha256", "semantic_sha256"):
        if not _SHA256_RE.fullmatch(str(normalization.get(key, ""))):
            raise ValueError(f"normalization.{key} must be SHA256")
    if normalization.get("source_split") != "train":
        raise ValueError("normalization must be fit from train split")
    if str(normalization.get("action_domain", "")) != action_domain:
        raise ValueError("normalization.action_domain disagrees with action.domain")


def _parse_episodes(
    raw: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> tuple[GenericM1Episode, ...]:
    raw_episodes = raw.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError("manifest episodes must be a non-empty list")
    root = manifest_path.parent
    records: list[GenericM1Episode] = []
    for position, value in enumerate(raw_episodes):
        if not isinstance(value, dict):
            raise ValueError(f"episodes[{position}] must be an object")
        relative = str(value.get("hdf5_path", ""))
        path = _resolve_relative_file(
            root,
            relative,
            field=f"episodes[{position}].hdf5_path",
        )
        sha256 = str(value.get("hdf5_sha256", ""))
        if not _SHA256_RE.fullmatch(sha256):
            raise ValueError(f"episodes[{position}].hdf5_sha256 must be SHA256")
        split = _normalize_split(str(value.get("split", "")))
        episode_index = _positive_or_zero_integer(
            value.get("episode_index"), f"episodes[{position}].episode_index"
        )
        source_episode_id = _positive_or_zero_integer(
            value.get("source_episode_id"),
            f"episodes[{position}].source_episode_id",
        )
        seed = _positive_or_zero_integer(
            value.get("seed"), f"episodes[{position}].seed"
        )
        num_steps = _positive_integer(
            value.get("steps"), f"episodes[{position}].steps"
        )
        recorded_steps = _positive_integer(
            value.get("recorded_steps"),
            f"episodes[{position}].recorded_steps",
        )
        if num_steps > recorded_steps:
            raise ValueError(f"episodes[{position}] selected steps exceed recorded")
        task_id = str(value.get("task_id", ""))
        task_text = str(value.get("task_text", ""))
        if not task_id or not task_text:
            raise ValueError(f"episodes[{position}] task id/text cannot be empty")
        success = value.get("success")
        if not isinstance(success, bool):
            raise ValueError(f"episodes[{position}].success must be boolean")
        records.append(
            GenericM1Episode(
                path=path,
                relative_path=relative,
                hdf5_sha256=sha256,
                episode_index=episode_index,
                source_episode_id=source_episode_id,
                seed=seed,
                split=split,
                task_id=task_id,
                task_text=task_text,
                num_steps=num_steps,
                recorded_steps=recorded_steps,
                success=success,
            )
        )
    return tuple(records)


def _audit_manifest(
    raw: Mapping[str, Any],
    records: Sequence[GenericM1Episode],
) -> None:
    for label, values in (
        ("episode paths", [record.path for record in records]),
        ("episode indices", [record.episode_index for record in records]),
        ("source episode ids", [record.source_episode_id for record in records]),
        ("episode seeds", [record.seed for record in records]),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"generic manifest {label} must be globally unique")

    assignment = [
        {
            "episode_index": record.episode_index,
            "seed": record.seed,
            "split": record.split,
        }
        for record in sorted(records, key=lambda item: item.episode_index)
    ]
    declared_assignment = str(
        _mapping(raw, "split_protocol")["assignment_sha256"]
    )
    if _json_sha256(assignment) != declared_assignment:
        raise ValueError("split assignment hash disagrees with episode records")

    split_counts = _mapping(raw, "split_counts")
    for split in GENERIC_M1_SPLITS:
        selected = [record for record in records if record.split == split]
        if not selected:
            raise ValueError(f"manifest split {split!r} cannot be empty")
        declared = _mapping(split_counts, split)
        expected = {
            "episodes": len(selected),
            "selected_transitions": sum(record.num_steps for record in selected),
            "recorded_transitions": sum(
                record.recorded_steps for record in selected
            ),
            "unique_seeds": len({record.seed for record in selected}),
            "successes": sum(record.success for record in selected),
            "task_ids": sorted({record.task_id for record in selected}),
        }
        for key, expected_value in expected.items():
            if declared.get(key) != expected_value:
                raise ValueError(f"split_counts.{split}.{key} disagrees with episodes")

    totals = _mapping(raw, "totals")
    expected_totals = {
        "episodes": len(records),
        "selected_transitions": sum(record.num_steps for record in records),
        "recorded_transitions": sum(record.recorded_steps for record in records),
        "unique_seeds": len({record.seed for record in records}),
    }
    for key, value in expected_totals.items():
        if totals.get(key) != value:
            raise ValueError(f"totals.{key} disagrees with episodes")

    selected = _mapping(raw, "transition_selection")
    for key, value in (
        ("selected_transitions", expected_totals["selected_transitions"]),
        ("recorded_transitions", expected_totals["recorded_transitions"]),
    ):
        if selected.get(key) != value:
            raise ValueError(f"transition_selection.{key} disagrees with episodes")
    normalization = _mapping(raw, "normalization")
    train_steps = sum(
        record.num_steps for record in records if record.split == "train"
    )
    if normalization.get("transition_count") != train_steps:
        raise ValueError("normalization.transition_count is not train-only")


def _inspect_hdf5_episode(
    record: GenericM1Episode,
    *,
    raw: Mapping[str, Any],
) -> GenericM1HDF5Metadata:
    schema = _mapping(raw, "schema")
    state = _mapping(raw, "state")
    action = _mapping(raw, "action")
    vision = _mapping(raw, "vision")
    timing = _mapping(raw, "timing")
    selection = _mapping(raw, "transition_selection")
    cameras = tuple(str(value) for value in vision["camera_order"])
    state_field = str(state["field"])
    next_state_field = str(state["next_field"])
    action_field = str(action["field"])
    history_action_field = str(action["history_field"])
    executed_action_field = str(action["executed_field"])
    terminal_field = str(selection["terminal_field"])
    current_prefix = str(vision["current_prefix"])
    next_prefix = str(vision["next_prefix"])
    required = {
        "data/timestamp",
        "data/frame_index",
        "data/episode_index",
        "data/seed",
        "data/task/id",
        "data/task/text",
        state_field,
        next_state_field,
        action_field,
        history_action_field,
        executed_action_field,
        terminal_field,
    }
    for camera in cameras:
        required.update(
            {
                f"{current_prefix}/{camera}",
                f"{next_prefix}/{camera}",
                f"data/observation/image_frame_index/{camera}",
                f"data/next_observation/image_frame_index/{camera}",
                f"data/observation/image_timestamp/{camera}",
                f"data/next_observation/image_timestamp/{camera}",
            }
        )

    with h5py.File(record.path, "r") as file:
        if str(file.attrs.get("schema_profile", "")) != str(schema["profile"]):
            raise ValueError(f"{record.path} schema_profile disagrees with manifest")
        if str(file.attrs.get("schema_version", "")) != str(schema["version"]):
            raise ValueError(f"{record.path} schema_version disagrees with manifest")
        if str(file.attrs.get("format_version", "")) != str(
            schema["hdf5_format_version"]
        ):
            raise ValueError(f"{record.path} HDF5 format version drifted")
        if int(file.attrs.get("episode_index", -1)) != record.episode_index:
            raise ValueError(f"{record.path} episode_index drifted")
        if int(file.attrs.get("seed", -1)) != record.seed:
            raise ValueError(f"{record.path} seed drifted")
        if int(file.attrs.get("num_steps", -1)) != record.recorded_steps:
            raise ValueError(f"{record.path} recorded step count drifted")
        if str(file.attrs.get("task_id", "")) != record.task_id:
            raise ValueError(f"{record.path} task_id drifted")
        fps = float(file.attrs.get("fps", np.nan))
        if not np.isfinite(fps) or fps != float(timing["control_hz"]):
            raise ValueError(f"{record.path} control frequency drifted")
        raw_camera_order = file.attrs.get("camera_order_json")
        try:
            file_cameras = tuple(json.loads(str(raw_camera_order)))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{record.path} has invalid camera_order_json") from exc
        if file_cameras != cameras:
            raise ValueError(f"{record.path} camera order drifted")
        missing = sorted(name for name in required if name not in file)
        if missing:
            raise KeyError(f"{record.path} is missing required datasets: {missing}")

        current_state = file[state_field]
        next_state = file[next_state_field]
        commanded = file[action_field]
        history_action = file[history_action_field]
        executed = file[executed_action_field]
        for dataset in (current_state, next_state, commanded, history_action, executed):
            if dataset.dtype != np.dtype(np.float32):
                raise TypeError(f"{record.path} {dataset.name} must be float32")
            if int(dataset.shape[0]) != record.recorded_steps:
                raise ValueError(f"{record.path} {dataset.name} length drifted")
        state_dim = int(state["dimension"])
        action_dim = int(action["dimension"])
        if current_state.shape != (record.recorded_steps, state_dim):
            raise ValueError(f"{record.path} current state shape drifted")
        if next_state.shape != current_state.shape:
            raise ValueError(f"{record.path} next state shape drifted")
        if commanded.shape != (record.recorded_steps, action_dim):
            raise ValueError(f"{record.path} commanded action shape drifted")
        if history_action.shape != commanded.shape or executed.shape != commanded.shape:
            raise ValueError(f"{record.path} action history/executed shape drifted")
        arrays = (
            np.asarray(current_state[:], dtype=np.float32),
            np.asarray(next_state[:], dtype=np.float32),
            np.asarray(commanded[:], dtype=np.float32),
            np.asarray(history_action[:], dtype=np.float32),
            np.asarray(executed[:], dtype=np.float32),
        )
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError(f"{record.path} state/action contains NaN or Inf")
        if record.recorded_steps > 1 and not np.array_equal(
            arrays[1][:-1], arrays[0][1:]
        ):
            raise ValueError(f"{record.path} next/current state alignment drifted")
        if action.get("executed_semantics") == "command_echo_exact_copy" and not (
            np.array_equal(arrays[2], arrays[4])
        ):
            raise ValueError(f"{record.path} command echo is not an exact copy")
        codec_payload = _mapping(action, "codec")
        if codec_payload.get("applied") is True:
            codec = AffineActionCodec(
                AffineActionCodecConfig.from_dict(
                    _mapping(codec_payload, "config")
                )
            )
            codec.encode(arrays[2], validate=True)
            codec.encode(arrays[3], validate=True)

        timestamps = np.asarray(file["data/timestamp"][:], dtype=np.float64)
        frame_indices = np.asarray(file["data/frame_index"][:], dtype=np.int64)
        if timestamps.shape != (record.recorded_steps,) or not np.isfinite(
            timestamps
        ).all():
            raise ValueError(f"{record.path} timestamps are invalid")
        if record.recorded_steps > 1 and not np.all(np.diff(timestamps) > 0.0):
            raise ValueError(f"{record.path} timestamps are not strictly increasing")
        if not np.array_equal(
            frame_indices, np.arange(record.recorded_steps, dtype=np.int64)
        ):
            raise ValueError(f"{record.path} frame_index must be contiguous")
        if _constant_integer(file["data/episode_index"], record.path) != record.episode_index:
            raise ValueError(f"{record.path} per-row episode_index drifted")
        if _constant_integer(file["data/seed"], record.path) != record.seed:
            raise ValueError(f"{record.path} per-row seed drifted")
        if _constant_string(file["data/task/id"], record.path) != record.task_id:
            raise ValueError(f"{record.path} per-row task id drifted")
        if _constant_string(file["data/task/text"], record.path) != record.task_text:
            raise ValueError(f"{record.path} per-row task text drifted")

        done = np.asarray(file[terminal_field][:], dtype=np.bool_)
        if done.shape != (record.recorded_steps,):
            raise ValueError(f"{record.path} terminal field shape drifted")
        if selection["mode"] == "through-first-done-inclusive":
            true_rows = np.flatnonzero(done)
            if true_rows.size == 0 or int(true_rows[0]) + 1 != record.num_steps:
                raise ValueError(f"{record.path} selected boundary is not first done")
        elif record.num_steps != record.recorded_steps:
            raise ValueError(f"{record.path} all-recorded selection truncated rows")

        image_shape: tuple[int, int, int] | None = None
        for camera in cameras:
            current_image = file[f"{current_prefix}/{camera}"]
            next_image = file[f"{next_prefix}/{camera}"]
            if current_image.dtype != np.dtype(np.uint8) or next_image.dtype != np.dtype(
                np.uint8
            ):
                raise TypeError(f"{record.path} camera {camera!r} must be uint8")
            if (
                current_image.ndim != 4
                or current_image.shape[0] != record.recorded_steps
                or current_image.shape[-1] != 3
                or next_image.shape != current_image.shape
            ):
                raise ValueError(f"{record.path} camera {camera!r} RGB shape drifted")
            shape = tuple(int(value) for value in current_image.shape[1:])
            if image_shape is None:
                image_shape = shape
            elif shape != image_shape:
                raise ValueError(f"{record.path} camera image shapes differ")
            current_ids = np.asarray(
                file[f"data/observation/image_frame_index/{camera}"][:],
                dtype=np.int64,
            )
            next_ids = np.asarray(
                file[f"data/next_observation/image_frame_index/{camera}"][:],
                dtype=np.int64,
            )
            if not np.array_equal(current_ids, frame_indices) or not np.array_equal(
                next_ids, frame_indices + 1
            ):
                raise ValueError(f"{record.path} camera {camera!r} frame ids drifted")
            current_time = np.asarray(
                file[f"data/observation/image_timestamp/{camera}"][:],
                dtype=np.float64,
            )
            next_time = np.asarray(
                file[f"data/next_observation/image_timestamp/{camera}"][:],
                dtype=np.float64,
            )
            tolerance = 1e-9
            if not np.allclose(current_time, timestamps, rtol=0.0, atol=tolerance):
                raise ValueError(f"{record.path} camera {camera!r} time drifted")
            if record.recorded_steps > 1 and not np.allclose(
                next_time[:-1], current_time[1:], rtol=0.0, atol=tolerance
            ):
                raise ValueError(
                    f"{record.path} camera {camera!r} next/current time drifted"
                )
        assert image_shape is not None
        return GenericM1HDF5Metadata(
            state_dim=state_dim,
            action_dim=action_dim,
            image_shape_hwc=image_shape,
            control_hz=fps,
        )


def _verify_normalization(root: Path, raw: Mapping[str, Any]) -> None:
    normalization = _mapping(raw, "normalization")
    path = _resolve_relative_file(
        root,
        str(normalization["path"]),
        field="normalization.path",
    )
    actual_file_sha256 = _sha256_file(path)
    if actual_file_sha256 != str(normalization["file_sha256"]):
        raise ValueError("normalization file SHA256 disagrees with manifest")
    stats = NormalizationStats.load(path)
    if stats.sha256() != str(normalization["semantic_sha256"]):
        raise ValueError("normalization semantic SHA256 disagrees with manifest")
    state_dim = int(_mapping(raw, "state")["dimension"])
    action_dim = int(_mapping(raw, "action")["dimension"])
    if stats.state_mean.shape != (state_dim,) or stats.action_mean.shape != (
        action_dim,
    ):
        raise ValueError("normalization dimensions disagree with manifest")


def _build_split_summary(
    manifest: GenericM1ManifestIndex,
    split: str,
    records: Sequence[GenericM1Episode],
) -> dict[str, Any]:
    lineage = [
        {
            "relative_path": record.relative_path,
            "hdf5_sha256": record.hdf5_sha256,
            "episode_index": record.episode_index,
            "source_episode_id": record.source_episode_id,
            "seed": record.seed,
            "task_id": record.task_id,
            "steps": record.num_steps,
            "recorded_steps": record.recorded_steps,
        }
        for record in records
    ]
    return {
        "dataset_protocol": GENERIC_M1_DATASET_PROTOCOL,
        "schema_profile": manifest.schema_profile,
        "schema_version": manifest.schema_version,
        "split": split,
        "episodes": len(records),
        "transitions": sum(record.num_steps for record in records),
        "recorded_transitions": sum(record.recorded_steps for record in records),
        "unique_seeds": len({record.seed for record in records}),
        "task_counts": dict(sorted(Counter(record.task_id for record in records).items())),
        "task_order": list(manifest.task_order),
        "task_to_index": dict(manifest.task_to_index),
        "camera_order": list(manifest.camera_order),
        "state_dim": manifest.state_dim,
        "action_dim": manifest.action_dim,
        "transition_selection": manifest.transition_selection,
        "normalization_sha256": manifest.normalization_sha256,
        "episode_lineage_sha256": _json_sha256(lineage),
    }


def _read_rgb_rows(
    file: h5py.File | Mapping[str, torch.Tensor],
    *,
    prefix: str,
    rows: Sequence[int],
    cameras: Sequence[str],
) -> np.ndarray:
    unique_rows, inverse = np.unique(
        np.asarray(rows, dtype=np.int64), return_inverse=True
    )
    per_camera = []
    for camera in cameras:
        selected = np.asarray(
            file[f"{prefix}/{camera}"][unique_rows.tolist()], dtype=np.uint8
        )
        per_camera.append(selected[inverse])
    values = np.stack(per_camera, axis=1)
    return np.ascontiguousarray(values.transpose(0, 1, 4, 2, 3))


def _latest_unique_rows(
    frame_indices: np.ndarray,
    *,
    end_inclusive: int,
    count: int,
) -> tuple[int, ...] | None:
    rows: list[int] = []
    seen: set[int] = set()
    for row in range(int(end_inclusive), -1, -1):
        frame_id = int(frame_indices[row])
        if frame_id in seen:
            continue
        seen.add(frame_id)
        rows.append(row)
        if len(rows) == int(count):
            return tuple(reversed(rows))
    return None


def _latest_available_unique_rows(
    frame_indices: np.ndarray,
    *,
    end_inclusive: int,
    maximum_count: int,
) -> tuple[int, ...]:
    rows: list[int] = []
    seen: set[int] = set()
    for row in range(int(end_inclusive), -1, -1):
        frame_id = int(frame_indices[row])
        if frame_id in seen:
            continue
        seen.add(frame_id)
        rows.append(row)
        if len(rows) == int(maximum_count):
            break
    if not rows:
        raise RuntimeError("trajectory decision has no available RGB frame")
    return tuple(reversed(rows))


def _repeat_last_to_horizon(values: np.ndarray, horizon: int) -> np.ndarray:
    if values.ndim < 1 or values.shape[0] <= 0 or values.shape[0] > horizon:
        raise ValueError("tail padding requires between one and horizon values")
    if values.shape[0] == horizon:
        return values
    padded = np.empty((horizon, *values.shape[1:]), dtype=values.dtype)
    padded[: values.shape[0]] = values
    padded[values.shape[0] :] = values[-1]
    return padded


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ValueError(f"manifest field {key!r} must be an object")
    return item


def _normalize_split(split: str) -> str:
    normalized = _SPLIT_ALIASES.get(str(split), str(split))
    if normalized not in GENERIC_M1_SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return normalized


def _validate_hdf5_path(value: str, *, field: str) -> None:
    if not value or "\\" in value:
        raise ValueError(f"{field} must be a non-empty POSIX HDF5 path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.parts[0] != "data":
        raise ValueError(f"{field} must be a safe path below data/")


def _resolve_relative_file(root: Path, value: str, *, field: str) -> Path:
    if not value or "\\" in value:
        raise ValueError(f"{field} must be a relative POSIX path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{field} must not escape the manifest directory")
    resolved = (root / Path(*relative.parts)).resolve(strict=True)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} escapes the manifest directory") from exc
    if not resolved.is_file():
        raise ValueError(f"{field} is not a file: {resolved}")
    return resolved


def _positive_integer(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _positive_or_zero_integer(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def _constant_integer(dataset: h5py.Dataset, path: Path) -> int:
    values = np.asarray(dataset[:], dtype=np.int64)
    if values.ndim != 1 or values.size == 0 or not np.all(values == values[0]):
        raise ValueError(f"{path} {dataset.name} is not a constant integer field")
    return int(values[0])


def _constant_string(dataset: h5py.Dataset, path: Path) -> str:
    values = dataset.asstr()[:]
    if values.ndim != 1 or values.size == 0 or not np.all(values == values[0]):
        raise ValueError(f"{path} {dataset.name} is not a constant string field")
    return str(values[0])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


__all__ = [
    "GENERIC_M1_DATASET_PROTOCOL",
    "GENERIC_M1_MANIFEST_FORMAT",
    "GENERIC_M1_SPLITS",
    "GenericM1Episode",
    "GenericM1HDF5Metadata",
    "GenericM1ManifestIndex",
    "GenericM1SampleLineage",
    "GenericM1WindowDataset",
]
