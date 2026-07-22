"""Build audited split and normalization artifacts for RoboFactory M1 data."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import h5py
import numpy as np

from data.robofactory import ROBOFACTORY_M1_PROFILE, ROBOFACTORY_M1_SCHEMA_VERSION
from models.wam.action_codec import AffineActionCodec, AffineActionCodecConfig
from models.wam.normalizer import NormalizationStats
from train.trajectory_dataset import split_episode_paths


TRAINING_MANIFEST_VERSION = "wam.multimodal.trajectory.training_manifest/1"
NORMALIZATION_PROTOCOL = "population_moments_float64/1"
TRANSITION_SELECTIONS = (
    "all-recorded",
    "through-first-done-inclusive",
)
SPLIT_NAMES = ("train", "validation", "test")
ProgressCallback = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class PreparedTrainingArtifacts:
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str
    normalization_path: Path
    normalization_file_sha256: str
    normalization_semantic_sha256: str


@dataclass(frozen=True)
class _EpisodeRecord:
    path: Path
    relative_path: str
    hdf5_sha256: str
    hdf5_size_bytes: int
    episode_index: int
    source_episode_id: int
    seed: int
    recorded_steps: int
    selected_steps: int
    first_done_index: int | None
    post_first_done_transitions: int
    done_true_transitions: int
    success: bool
    terminated: bool
    truncated: bool
    task_id: str
    task_text: str


class _Moments:
    """Streaming population moments with deterministic float64 accumulation."""

    def __init__(self, width: int) -> None:
        self.count = 0
        self.sum = np.zeros(width, dtype=np.float64)
        self.square_sum = np.zeros(width, dtype=np.float64)
        self.minimum = np.full(width, np.inf, dtype=np.float64)
        self.maximum = np.full(width, -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.sum.shape[0]:
            raise ValueError(
                f"moment input must have shape [N,{self.sum.shape[0]}], got "
                f"{array.shape}"
            )
        if array.shape[0] == 0:
            return
        if not np.isfinite(array).all():
            raise ValueError("normalization input contains NaN or Inf")
        self.count += int(array.shape[0])
        self.sum += array.sum(axis=0, dtype=np.float64)
        self.square_sum += np.square(array, dtype=np.float64).sum(
            axis=0, dtype=np.float64
        )
        self.minimum = np.minimum(self.minimum, array.min(axis=0))
        self.maximum = np.maximum(self.maximum, array.max(axis=0))

    def finish(
        self, *, std_floor: float
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if self.count <= 0:
            raise ValueError("cannot fit statistics from zero transitions")
        mean = self.sum / self.count
        variance = np.maximum(
            self.square_sum / self.count - np.square(mean),
            0.0,
        )
        raw_std = np.sqrt(variance)
        floored = np.flatnonzero(raw_std < std_floor).astype(np.int64)
        std = np.maximum(raw_std, std_floor)
        summary = {
            "count": self.count,
            "mean": mean.astype(np.float32).tolist(),
            "std": std.astype(np.float32).tolist(),
            "minimum": self.minimum.astype(np.float32).tolist(),
            "maximum": self.maximum.astype(np.float32).tolist(),
            "std_floor_applied_indices": floored.tolist(),
        }
        return mean.astype(np.float32), std.astype(np.float32), summary


def _coerce_action_codec(
    value: AffineActionCodec | AffineActionCodecConfig | str | Path | None,
) -> AffineActionCodec | None:
    if value is None:
        return None
    if isinstance(value, AffineActionCodec):
        return value
    if isinstance(value, AffineActionCodecConfig):
        return AffineActionCodec(value)
    return AffineActionCodec(AffineActionCodecConfig.load(value))


def prepare_robofactory_m1_training_artifacts(
    dataset_dir: str | Path,
    *,
    transition_selection: str,
    conversion_manifest_path: str | Path = "manifest.json",
    training_manifest_path: str | Path = "training_manifest.json",
    normalization_path: str | Path = "normalization.npz",
    split_seed: int = 7,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    std_floor: float = 1e-3,
    expected_episodes: int | None = None,
    expected_state_dim: int | None = None,
    expected_action_dim: int | None = None,
    expected_task_id: str | None = None,
    expected_cameras: Sequence[str] | None = None,
    expected_fps: float | None = None,
    action_codec: AffineActionCodec | AffineActionCodecConfig | str | Path | None = None,
    overwrite: bool = False,
    progress: ProgressCallback | None = None,
) -> PreparedTrainingArtifacts:
    """Create a portable manifest and train-only normalization statistics.

    The split unit is a complete episode seed.  Statistics consume each selected
    raw transition exactly once and never pass through overlapping model windows.
    When an action codec is supplied, HDF5 remains losslessly raw while window
    actions and train-only action statistics use the canonical encoded domain.
    """

    if transition_selection not in TRANSITION_SELECTIONS:
        raise ValueError(
            f"transition_selection must be one of {TRANSITION_SELECTIONS}, got "
            f"{transition_selection!r}"
        )
    if std_floor <= 0.0 or not np.isfinite(std_floor):
        raise ValueError("std_floor must be finite and positive")
    root = Path(dataset_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {root}")
    conversion_path = _resolve_inside(root, conversion_manifest_path)
    output_manifest = _resolve_inside(root, training_manifest_path)
    output_normalization = _resolve_inside(root, normalization_path)
    sidecar_path = output_manifest.with_suffix(output_manifest.suffix + ".sha256")
    if output_manifest == conversion_path:
        raise ValueError("training manifest must not overwrite conversion manifest")
    for path in (output_manifest, output_normalization, sidecar_path):
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"refusing to overwrite {path}; pass overwrite=True explicitly"
            )

    conversion_bytes = conversion_path.read_bytes()
    conversion = json.loads(conversion_bytes)
    if not isinstance(conversion, dict):
        raise ValueError("conversion manifest root must be an object")
    contract = _validate_conversion_manifest(
        conversion,
        expected_episodes=expected_episodes,
        expected_state_dim=expected_state_dim,
        expected_action_dim=expected_action_dim,
        expected_task_id=expected_task_id,
        expected_cameras=expected_cameras,
        expected_fps=expected_fps,
    )
    resolved_codec = _coerce_action_codec(action_codec)
    if resolved_codec is not None and resolved_codec.action_dim != contract["action_dim"]:
        raise ValueError("action codec dimension differs from conversion manifest")

    raw_episodes = conversion.get("episodes")
    assert isinstance(raw_episodes, list)
    expected_paths = {
        root / "hdf5" / f"episode_{int(item['episode_index']):06d}.hdf5"
        for item in raw_episodes
    }
    actual_paths = {
        path.resolve()
        for path in (root / "hdf5").glob("episode_*.hdf5")
        if ".partial." not in path.name
    }
    missing = sorted(str(path) for path in expected_paths - actual_paths)
    extra = sorted(str(path) for path in actual_paths - expected_paths)
    if missing or extra:
        raise ValueError(
            "HDF5 file set differs from conversion manifest: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    records: list[_EpisodeRecord] = []
    for position, item in enumerate(
        sorted(raw_episodes, key=lambda value: int(value["episode_index"])),
        start=1,
    ):
        if not isinstance(item, dict):
            raise ValueError("conversion manifest episode entries must be objects")
        path = root / "hdf5" / f"episode_{int(item['episode_index']):06d}.hdf5"
        records.append(
            _inspect_episode(
                root,
                path,
                item,
                state_dim=contract["state_dim"],
                action_dim=contract["action_dim"],
                cameras=contract["cameras"],
                schema_profile=contract["schema_profile"],
                schema_version=contract["schema_version"],
                fps=contract["fps"],
                task_id=contract["task_id"],
                transition_selection=transition_selection,
                action_codec=resolved_codec,
            )
        )
        if progress is not None:
            progress(
                {
                    "phase": "audit_and_hash",
                    "current": position,
                    "total": len(raw_episodes),
                    "path": records[-1].relative_path,
                }
            )
    _audit_unique_records(records)

    split_paths = split_episode_paths(
        [record.path for record in records],
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=split_seed,
    )
    if any(not split_paths[name] for name in SPLIT_NAMES):
        raise ValueError("train, validation, and test splits must all be non-empty")
    split_by_path = {
        path.resolve(): name
        for name, paths in split_paths.items()
        for path in paths
    }
    if set(split_by_path) != {record.path.resolve() for record in records}:
        raise RuntimeError("split assignment does not cover every episode exactly once")

    stats, stats_summary = _fit_train_normalization(
        records,
        split_by_path=split_by_path,
        state_dim=contract["state_dim"],
        action_dim=contract["action_dim"],
        std_floor=std_floor,
        action_codec=resolved_codec,
        progress=progress,
    )
    _atomic_save_normalization(stats, output_normalization)
    loaded_stats = NormalizationStats.load(output_normalization)
    if loaded_stats.sha256() != stats.sha256():
        raise RuntimeError("normalization changed during save/reload")
    normalization_file_sha256 = _sha256_file(output_normalization)
    normalization_semantic_sha256 = stats.sha256()

    episode_entries = [
        _episode_manifest_entry(record, split_by_path[record.path.resolve()])
        for record in records
    ]
    split_counts = _split_counts(episode_entries, transition_selection)
    assignment = [
        {
            "episode_index": entry["episode_index"],
            "seed": entry["seed"],
            "split": entry["split"],
        }
        for entry in episode_entries
    ]
    source = conversion.get("source")
    source = source if isinstance(source, dict) else {}
    behavior_id = str(source.get("metadata", {}).get("source_type", "unknown"))
    action_domain = (
        "raw_pd_joint_pos_commanded"
        if resolved_codec is None
        else resolved_codec.config.encoded_domain
    )
    codec_manifest: dict[str, Any]
    if resolved_codec is None:
        codec_manifest = {
            "applied": False,
            "reason": "simulator per-dimension bounds are not frozen yet",
        }
    else:
        codec_manifest = {
            "applied": True,
            "semantic_sha256": resolved_codec.semantic_sha256,
            "config": resolved_codec.config.to_dict(),
        }
    manifest: dict[str, Any] = {
        "format_version": TRAINING_MANIFEST_VERSION,
        "dataset_protocol": "generic_multimodal_trajectory",
        "source": {
            "conversion_manifest_path": _relative_posix(root, conversion_path),
            "conversion_manifest_sha256": hashlib.sha256(
                conversion_bytes
            ).hexdigest(),
            "source_hdf5_sha256": source.get("hdf5_sha256"),
            "source_metadata_json_sha256": source.get("metadata_json_sha256"),
        },
        "schema": {
            "profile": contract["schema_profile"],
            "version": contract["schema_version"],
            "hdf5_format_version": "wam.trajectory.hdf5/1",
        },
        "task": {
            "id": contract["task_id"],
            "text": str(conversion["task"]),
            "behavior_id": behavior_id,
        },
        "state": {
            "field": "data/observation/state",
            "next_field": "data/next_observation/state",
            "dimension": contract["state_dim"],
            "dtype": "float32",
            "layout": conversion.get("field_mapping", {}).get(
                "centralized_state", []
            ),
        },
        "action": {
            "field": "data/action/commanded",
            "history_field": "data/action/commanded",
            "executed_field": "data/action/executed",
            "dimension": contract["action_dim"],
            "dtype": "float32",
            "control_mode": "pd_joint_pos",
            "storage_domain": "raw_pd_joint_pos_commanded",
            "domain": action_domain,
            "codec": codec_manifest,
            "executed_semantics": "command_echo_exact_copy",
            "independent_actuator_feedback_available": False,
            "layout": conversion.get("field_mapping", {}).get(
                "centralized_action", []
            ),
        },
        "vision": {
            "camera_order": list(contract["cameras"]),
            "current_prefix": "data/observation/images",
            "next_prefix": "data/next_observation/images",
        },
        "timing": {
            "control_hz": contract["fps"],
            "image_hz": float(
                conversion.get("data_semantics", {})
                .get("timing", {})
                .get("image_hz", contract["fps"])
            ),
        },
        "transition_selection": {
            "mode": transition_selection,
            "terminal_field": "data/done",
            "includes_first_terminal_transition": True,
            "recorded_transitions": sum(
                record.recorded_steps for record in records
            ),
            "selected_transitions": sum(
                record.selected_steps for record in records
            ),
            "post_first_done_transitions": sum(
                record.post_first_done_transitions for record in records
            ),
            "excluded_post_first_done_transitions": (
                sum(record.post_first_done_transitions for record in records)
                if transition_selection == "through-first-done-inclusive"
                else 0
            ),
        },
        "split_protocol": {
            "unit": "episode_seed",
            "claim": "seed_disjoint_only",
            "algorithm": "sorted_paths_then_numpy_default_rng_shuffle/1",
            "seed": int(split_seed),
            "fractions": {
                "train": float(train_fraction),
                "validation": float(validation_fraction),
                "test": float(test_fraction),
            },
            "assignment_sha256": _canonical_sha256(assignment),
        },
        "split_counts": split_counts,
        "normalization": {
            "path": _relative_posix(root, output_normalization),
            "file_sha256": normalization_file_sha256,
            "semantic_sha256": normalization_semantic_sha256,
            "protocol": NORMALIZATION_PROTOCOL,
            "source_split": "train",
            "sample_unit": (
                "selected_raw_transition"
                if resolved_codec is None
                else "selected_transition_after_action_codec"
            ),
            "transition_selection": transition_selection,
            "transition_count": stats_summary["state"]["count"],
            "std_floor": float(std_floor),
            "action_domain": action_domain,
            "fields": {
                "state": {
                    "source": "data/observation/state",
                    **stats_summary["state"],
                },
                "action": {
                    "source": (
                        "data/action/commanded"
                        if resolved_codec is None
                        else "action_codec.encode(data/action/commanded)"
                    ),
                    **stats_summary["action"],
                },
                "delta": {
                    "source": (
                        "data/next_observation/state - "
                        "data/observation/state"
                    ),
                    **stats_summary["delta"],
                },
            },
            "reward": {
                "available": False,
                "placeholder": True,
                "usage": "unused_model_compatibility_only",
                "mean": [0.0],
                "std": [1.0],
            },
        },
        "episodes": episode_entries,
        "totals": {
            "episodes": len(records),
            "unique_seeds": len({record.seed for record in records}),
            "recorded_transitions": sum(
                record.recorded_steps for record in records
            ),
            "selected_transitions": sum(
                record.selected_steps for record in records
            ),
        },
        "integrity": {
            "all_hdf5_sha256_recorded": True,
            "episode_paths_unique": True,
            "episode_indices_unique": True,
            "episode_seeds_unique": True,
            "split_seed_disjoint": True,
            "normalization_train_only": True,
            "normalization_counts_each_transition_once": True,
            "action_codec_bounds_verified": resolved_codec is not None,
            "generator_module_sha256": _sha256_file(Path(__file__)),
        },
    }
    _assert_json_finite(manifest)
    _atomic_json(output_manifest, manifest)
    manifest_sha256 = _sha256_file(output_manifest)
    _atomic_text(
        sidecar_path,
        f"{manifest_sha256}  {output_manifest.name}\n",
    )
    return PreparedTrainingArtifacts(
        manifest=manifest,
        manifest_path=output_manifest,
        manifest_sha256=manifest_sha256,
        normalization_path=output_normalization,
        normalization_file_sha256=normalization_file_sha256,
        normalization_semantic_sha256=normalization_semantic_sha256,
    )


def _validate_conversion_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_episodes: int | None,
    expected_state_dim: int | None,
    expected_action_dim: int | None,
    expected_task_id: str | None,
    expected_cameras: Sequence[str] | None,
    expected_fps: float | None,
) -> dict[str, Any]:
    if manifest.get("format_version") != "robofactory.conversion_manifest/2.0":
        raise ValueError("conversion manifest is not RoboFactory M1 version 2")
    profile = str(manifest.get("schema_profile", ""))
    version = str(manifest.get("schema_version", ""))
    if profile != ROBOFACTORY_M1_PROFILE or version != ROBOFACTORY_M1_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema {profile!r}/{version!r}; expected "
            f"{ROBOFACTORY_M1_PROFILE!r}/{ROBOFACTORY_M1_SCHEMA_VERSION!r}"
        )
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("conversion manifest episodes must be a non-empty list")
    if expected_episodes is not None and len(episodes) != expected_episodes:
        raise ValueError(
            f"expected {expected_episodes} episodes, conversion has {len(episodes)}"
        )
    layout = manifest.get("layout")
    if not isinstance(layout, dict):
        raise ValueError("conversion manifest is missing layout")
    state_dim = int(layout.get("state_size", -1))
    action_dim = int(layout.get("action_size", -1))
    if state_dim <= 0 or action_dim <= 0:
        raise ValueError("conversion manifest state/action dimensions are invalid")
    if expected_state_dim is not None and state_dim != expected_state_dim:
        raise ValueError(f"expected state_dim={expected_state_dim}, got {state_dim}")
    if expected_action_dim is not None and action_dim != expected_action_dim:
        raise ValueError(f"expected action_dim={expected_action_dim}, got {action_dim}")
    task_id = str(manifest.get("task_id", ""))
    if not task_id:
        raise ValueError("conversion manifest task_id is empty")
    if expected_task_id is not None and task_id != expected_task_id:
        raise ValueError(f"expected task_id={expected_task_id!r}, got {task_id!r}")
    exported = layout.get("exported_cameras")
    if not isinstance(exported, list) or not exported:
        raise ValueError("conversion manifest exported_cameras is empty")
    cameras = tuple(str(item.get("target_name", "")) for item in exported)
    if any(not camera for camera in cameras) or len(cameras) != len(set(cameras)):
        raise ValueError("conversion manifest exported camera names are invalid")
    if expected_cameras is not None and cameras != tuple(expected_cameras):
        raise ValueError(
            f"expected cameras {tuple(expected_cameras)}, got {cameras}"
        )
    fps = float(manifest.get("fps", np.nan))
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("conversion manifest fps must be finite and positive")
    if expected_fps is not None and not np.isclose(fps, expected_fps):
        raise ValueError(f"expected fps={expected_fps}, got {fps}")
    action_semantics = manifest.get("data_semantics", {}).get("action", {})
    if (
        action_semantics.get("commanded_field") != "action.commanded"
        or action_semantics.get("executed_action_source") != "command_echo"
        or action_semantics.get("independent_actuator_feedback_available") is not False
    ):
        raise ValueError("conversion manifest action semantics are not command-echo M1")
    return {
        "schema_profile": profile,
        "schema_version": version,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "task_id": task_id,
        "cameras": cameras,
        "fps": fps,
    }


def _inspect_episode(
    root: Path,
    path: Path,
    summary: Mapping[str, Any],
    *,
    state_dim: int,
    action_dim: int,
    cameras: Sequence[str],
    schema_profile: str,
    schema_version: str,
    fps: float,
    task_id: str,
    transition_selection: str,
    action_codec: AffineActionCodec | None,
) -> _EpisodeRecord:
    if not path.is_file():
        raise FileNotFoundError(path)
    episode_index = int(summary["episode_index"])
    source_episode_id = int(summary["source_episode_id"])
    seed = int(summary["seed"])
    recorded_steps = int(summary["steps"])
    with h5py.File(path, "r") as file:
        expected_attrs = {
            "schema_profile": schema_profile,
            "schema_version": schema_version,
            "episode_index": episode_index,
            "seed": seed,
            "num_steps": recorded_steps,
            "task_id": task_id,
        }
        for name, expected in expected_attrs.items():
            actual = file.attrs.get(name)
            if str(actual) != str(expected):
                raise ValueError(
                    f"{path}: attr {name}={actual!r}, expected {expected!r}"
                )
        actual_fps = float(file.attrs.get("fps", np.nan))
        if not np.isclose(actual_fps, fps):
            raise ValueError(f"{path}: fps {actual_fps} does not match {fps}")
        try:
            camera_order = tuple(json.loads(str(file.attrs["camera_order_json"])))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}: invalid camera_order_json") from exc
        if camera_order != tuple(cameras):
            raise ValueError(
                f"{path}: camera order {camera_order} does not match {tuple(cameras)}"
            )
        required = (
            "data/observation/state",
            "data/next_observation/state",
            "data/action/commanded",
            "data/action/executed",
            "data/timestamp",
            "data/frame_index",
            "data/episode_index",
            "data/seed",
            "data/task/id",
            "data/task/text",
            "data/terminated",
            "data/truncated",
            "data/done",
            "data/success",
        )
        image_fields = tuple(
            field
            for camera in cameras
            for field in (
                f"data/observation/images/{camera}",
                f"data/next_observation/images/{camera}",
                f"data/observation/image_frame_index/{camera}",
                f"data/next_observation/image_frame_index/{camera}",
                f"data/observation/image_timestamp/{camera}",
                f"data/next_observation/image_timestamp/{camera}",
            )
        )
        missing = [name for name in (*required, *image_fields) if name not in file]
        if missing:
            raise KeyError(f"{path}: missing required datasets {missing}")
        for name in (*required, *image_fields):
            if file[name].shape[0] != recorded_steps:
                raise ValueError(
                    f"{path}: {name} length {file[name].shape[0]} does not match "
                    f"{recorded_steps}"
                )
        state = file["data/observation/state"]
        next_state = file["data/next_observation/state"]
        commanded = file["data/action/commanded"]
        executed = file["data/action/executed"]
        if state.shape != (recorded_steps, state_dim):
            raise ValueError(f"{path}: state shape is {state.shape}")
        if next_state.shape != state.shape:
            raise ValueError(f"{path}: next-state shape is {next_state.shape}")
        if commanded.shape != (recorded_steps, action_dim):
            raise ValueError(f"{path}: commanded action shape is {commanded.shape}")
        if executed.shape != commanded.shape:
            raise ValueError(f"{path}: executed action shape is {executed.shape}")
        for dataset in (state, next_state, commanded, executed):
            if dataset.dtype != np.dtype(np.float32):
                raise TypeError(f"{path}: {dataset.name} must be float32")
            if not np.isfinite(dataset[:]).all():
                raise ValueError(f"{path}: {dataset.name} contains NaN or Inf")
        if not np.array_equal(commanded[:], executed[:]):
            raise ValueError(f"{path}: command echo differs from commanded action")
        if action_codec is not None:
            action_codec.encode(np.asarray(commanded[:], dtype=np.float32))
        if recorded_steps > 1 and not np.array_equal(next_state[:-1], state[1:]):
            raise ValueError(f"{path}: next_state[t] does not equal state[t+1]")
        frame_index = np.asarray(file["data/frame_index"][:], dtype=np.int64)
        if not np.array_equal(frame_index, np.arange(recorded_steps, dtype=np.int64)):
            raise ValueError(f"{path}: frame_index is not contiguous from zero")
        timestamps = np.asarray(file["data/timestamp"][:], dtype=np.float64)
        if not np.isfinite(timestamps).all() or (
            recorded_steps > 1 and not np.all(np.diff(timestamps) > 0.0)
        ):
            raise ValueError(f"{path}: timestamps are not finite and increasing")
        if not np.all(np.asarray(file["data/episode_index"][:]) == episode_index):
            raise ValueError(f"{path}: per-step episode_index is inconsistent")
        if not np.all(np.asarray(file["data/seed"][:]) == seed):
            raise ValueError(f"{path}: per-step seed is inconsistent")
        if not np.all(file["data/task/id"].asstr()[:] == task_id):
            raise ValueError(f"{path}: per-step task id is inconsistent")
        task_values = file["data/task/text"].asstr()[:]
        if not len(task_values) or not np.all(task_values == task_values[0]):
            raise ValueError(f"{path}: task text changes inside episode")
        for camera in cameras:
            current_image = file[f"data/observation/images/{camera}"]
            next_image = file[f"data/next_observation/images/{camera}"]
            if current_image.dtype != np.dtype(np.uint8) or next_image.dtype != np.dtype(
                np.uint8
            ):
                raise TypeError(f"{path}: camera {camera} RGB must be uint8")
            if current_image.ndim != 4 or current_image.shape[-1] != 3:
                raise ValueError(f"{path}: camera {camera} current RGB shape is invalid")
            if next_image.shape != current_image.shape:
                raise ValueError(f"{path}: camera {camera} current/next shapes differ")
            current_frames = np.asarray(
                file[f"data/observation/image_frame_index/{camera}"][:],
                dtype=np.int64,
            )
            next_frames = np.asarray(
                file[f"data/next_observation/image_frame_index/{camera}"][:],
                dtype=np.int64,
            )
            if not np.array_equal(current_frames, frame_index):
                raise ValueError(f"{path}: camera {camera} current frame ids drift")
            if not np.array_equal(next_frames, frame_index + 1):
                raise ValueError(f"{path}: camera {camera} next frame ids drift")
        done = np.asarray(file["data/done"][:], dtype=np.bool_)
        if file["data/done"].dtype != np.dtype(np.bool_):
            raise TypeError(f"{path}: done must be bool")
        true_indices = np.flatnonzero(done)
        first_done_index = int(true_indices[0]) if true_indices.size else None
        if first_done_index is not None and not bool(done[first_done_index:].all()):
            raise ValueError(f"{path}: done becomes false after first terminal step")
        if bool(summary["success"]) and first_done_index is None:
            raise ValueError(f"{path}: successful episode has no done transition")
        if transition_selection == "through-first-done-inclusive":
            selected_steps = (
                first_done_index + 1
                if first_done_index is not None
                else recorded_steps
            )
        else:
            selected_steps = recorded_steps
        post_first_done = (
            recorded_steps - first_done_index - 1
            if first_done_index is not None
            else 0
        )
        task_text = str(task_values[0])
    return _EpisodeRecord(
        path=path.resolve(),
        relative_path=_relative_posix(root, path),
        hdf5_sha256=_sha256_file(path),
        hdf5_size_bytes=path.stat().st_size,
        episode_index=episode_index,
        source_episode_id=source_episode_id,
        seed=seed,
        recorded_steps=recorded_steps,
        selected_steps=selected_steps,
        first_done_index=first_done_index,
        post_first_done_transitions=post_first_done,
        done_true_transitions=int(true_indices.size),
        success=bool(summary["success"]),
        terminated=bool(summary["terminated"]),
        truncated=bool(summary["truncated"]),
        task_id=task_id,
        task_text=task_text,
    )


def _audit_unique_records(records: Sequence[_EpisodeRecord]) -> None:
    for label, values in (
        ("path", [record.relative_path for record in records]),
        ("episode_index", [record.episode_index for record in records]),
        ("source_episode_id", [record.source_episode_id for record in records]),
        ("seed", [record.seed for record in records]),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"episode {label} values must be globally unique")


def _fit_train_normalization(
    records: Sequence[_EpisodeRecord],
    *,
    split_by_path: Mapping[Path, str],
    state_dim: int,
    action_dim: int,
    std_floor: float,
    action_codec: AffineActionCodec | None,
    progress: ProgressCallback | None,
) -> tuple[NormalizationStats, dict[str, dict[str, Any]]]:
    selected = [
        record
        for record in records
        if split_by_path[record.path.resolve()] == "train"
    ]
    if not selected:
        raise ValueError("train split is empty")
    state_moments = _Moments(state_dim)
    action_moments = _Moments(action_dim)
    delta_moments = _Moments(state_dim)
    for position, record in enumerate(selected, start=1):
        with h5py.File(record.path, "r") as file:
            stop = record.selected_steps
            state = np.asarray(file["data/observation/state"][:stop], dtype=np.float32)
            next_state = np.asarray(
                file["data/next_observation/state"][:stop], dtype=np.float32
            )
            action = np.asarray(file["data/action/commanded"][:stop], dtype=np.float32)
        state_moments.update(state)
        action_moments.update(
            action
            if action_codec is None
            else action_codec.encode(action)
        )
        delta_moments.update(next_state - state)
        if progress is not None:
            progress(
                {
                    "phase": "fit_train_normalization",
                    "current": position,
                    "total": len(selected),
                    "path": record.relative_path,
                }
            )
    state_mean, state_std, state_summary = state_moments.finish(std_floor=std_floor)
    action_mean, action_std, action_summary = action_moments.finish(
        std_floor=std_floor
    )
    delta_mean, delta_std, delta_summary = delta_moments.finish(std_floor=std_floor)
    expected_count = sum(record.selected_steps for record in selected)
    actual_counts = {
        state_summary["count"],
        action_summary["count"],
        delta_summary["count"],
    }
    if actual_counts != {expected_count}:
        raise RuntimeError(
            f"normalization transition count mismatch: {actual_counts}, "
            f"expected {expected_count}"
        )
    return (
        NormalizationStats(
            state_mean=state_mean,
            state_std=state_std,
            action_mean=action_mean,
            action_std=action_std,
            delta_mean=delta_mean,
            delta_std=delta_std,
            reward_mean=np.zeros(1, dtype=np.float32),
            reward_std=np.ones(1, dtype=np.float32),
        ),
        {
            "state": state_summary,
            "action": action_summary,
            "delta": delta_summary,
        },
    )


def _episode_manifest_entry(record: _EpisodeRecord, split: str) -> dict[str, Any]:
    return {
        "episode_index": record.episode_index,
        "source_episode_id": record.source_episode_id,
        "seed": record.seed,
        "split": split,
        "hdf5_path": record.relative_path,
        "hdf5_sha256": record.hdf5_sha256,
        "hdf5_size_bytes": record.hdf5_size_bytes,
        "recorded_steps": record.recorded_steps,
        "steps": record.selected_steps,
        "first_done_index": record.first_done_index,
        "post_first_done_transitions": record.post_first_done_transitions,
        "done_true_transitions": record.done_true_transitions,
        "task_id": record.task_id,
        "task_text": record.task_text,
        "success": record.success,
        "terminated": record.terminated,
        "truncated": record.truncated,
    }


def _split_counts(
    episodes: Sequence[Mapping[str, Any]], transition_selection: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    seed_sets: dict[str, set[int]] = {}
    path_sets: dict[str, set[str]] = {}
    for split in SPLIT_NAMES:
        selected = [entry for entry in episodes if entry["split"] == split]
        seeds = {int(entry["seed"]) for entry in selected}
        paths = {str(entry["hdf5_path"]) for entry in selected}
        seed_sets[split] = seeds
        path_sets[split] = paths
        result[split] = {
            "episodes": len(selected),
            "unique_seeds": len(seeds),
            "recorded_transitions": sum(
                int(entry["recorded_steps"]) for entry in selected
            ),
            "selected_transitions": sum(int(entry["steps"]) for entry in selected),
            "excluded_post_first_done_transitions": (
                sum(int(entry["post_first_done_transitions"]) for entry in selected)
                if transition_selection == "through-first-done-inclusive"
                else 0
            ),
            "successes": sum(bool(entry["success"]) for entry in selected),
            "task_ids": sorted({str(entry["task_id"]) for entry in selected}),
            "episode_indices_sha256": _canonical_sha256(
                sorted(int(entry["episode_index"]) for entry in selected)
            ),
            "seeds_sha256": _canonical_sha256(sorted(seeds)),
        }
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if seed_sets[left] & seed_sets[right] or path_sets[left] & path_sets[right]:
            raise RuntimeError(f"split leakage between {left} and {right}")
    return result


def _resolve_inside(root: Path, value: str | Path) -> Path:
    raw = Path(value)
    resolved = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact path must remain inside dataset directory: {value}") from exc
    return resolved


def _relative_posix(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assert_json_finite(value: Mapping[str, Any]) -> None:
    json.dumps(value, allow_nan=False)


def _atomic_save_normalization(stats: NormalizationStats, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        stats.save(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    _atomic_text(path, payload)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "NORMALIZATION_PROTOCOL",
    "PreparedTrainingArtifacts",
    "TRAINING_MANIFEST_VERSION",
    "TRANSITION_SELECTIONS",
    "prepare_robofactory_m1_training_artifacts",
]
