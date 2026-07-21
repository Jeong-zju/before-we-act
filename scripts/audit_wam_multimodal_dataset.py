"""Fail-closed M0-v2 audit for MuJoCo, three-camera multimodal datasets."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

# Keep MuJoCo model/provenance checks usable on headless acceptance hosts.
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import h5py
import mujoco
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.visual_required_env import (  # noqa: E402
    EVENT_DECISION_STEP,
    VISUAL_REQUIRED_TASKS,
    VisualRequiredEnv,
    VisualRequiredEnvConfig,
)


FORMAT_VERSION = "wam.multimodal.m0.dataset_audit/2"
MANIFEST_FORMAT_VERSION = "wam.multimodal.m0.dataset/2"
CONFIG_VERSION = "wam.multimodal.m0/2"
SCHEMA_PROFILE = "wam_multimodal"
SCHEMA_VERSION = "wam.multimodal/1.1"
MIN_FORMAL_DATASET_EPISODES = 2_000
CAMERA_ORDER = ("fixed", "robot_0_camera", "robot_1_camera")

_REQUIRED_BASE_PATHS = (
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
)


def _required_paths(camera_order: Sequence[str]) -> tuple[str, ...]:
    """Return the complete v1.1 on-disk contract for every configured view."""

    paths = list(_REQUIRED_BASE_PATHS)
    for camera in camera_order:
        for prefix in ("observation", "next_observation"):
            paths.extend(
                (
                    f"data/{prefix}/images/{camera}",
                    f"data/{prefix}/image_timestamp/{camera}",
                    f"data/{prefix}/image_state_timestamp/{camera}",
                    f"data/{prefix}/image_frame_index/{camera}",
                )
            )
        for prefix in ("camera", "next_camera"):
            paths.extend(
                (
                    f"data/{prefix}/intrinsics/{camera}",
                    f"data/{prefix}/extrinsics/{camera}",
                    f"data/{prefix}/resolution/{camera}",
                )
            )
    return tuple(paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/wam_multimodal/m0_data.yaml",
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--no-loader-smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = _load_config(config_path)
    canonical_data = _root_path(config["dataset"]["directory"])
    canonical_report = _root_path(config["audit"]["report"])
    data_dir = (args.data_dir or canonical_data).resolve()
    report_path = (args.report or canonical_report).resolve()
    formal_protocol = bool(
        config_path == (ROOT / "configs/wam_multimodal/m0_data.yaml").resolve()
        and data_dir == canonical_data.resolve()
        and report_path == canonical_report.resolve()
        and not args.no_loader_smoke
        and _configured_dataset_episode_total(config) >= MIN_FORMAL_DATASET_EPISODES
    )
    _require_new_file(report_path)
    report = audit_dataset(
        config,
        config_path=config_path,
        data_dir=data_dir,
        formal_protocol=formal_protocol,
        run_loader_smoke=not args.no_loader_smoke,
    )
    _atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        return 1
    return 0 if formal_protocol else 2


def audit_dataset(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    data_dir: Path,
    formal_protocol: bool,
    run_loader_smoke: bool = True,
) -> dict[str, Any]:
    """Audit one dataset without mutating it."""

    dataset_cfg = _mapping(config, "dataset")
    camera_cfg = _mapping(config, "camera")
    audit_cfg = _mapping(config, "audit")
    acceptance_cfg = _mapping(config, "acceptance")
    camera_order = _camera_order(config)
    formal_protocol = bool(
        formal_protocol
        and _configured_dataset_episode_total(config) >= MIN_FORMAL_DATASET_EPISODES
    )
    if dataset_cfg.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("canonical config has an unsupported schema version")

    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _load_json(manifest_path)
    entries = manifest.get("episodes")
    if not isinstance(entries, list):
        raise ValueError("dataset manifest episodes must be a list")

    _visual_required_tasks(manifest)
    expected_episodes = int(dataset_cfg["episodes_per_task"]) * len(
        VISUAL_REQUIRED_TASKS
    )
    formal_protocol = bool(
        formal_protocol
        and expected_episodes >= MIN_FORMAL_DATASET_EPISODES
        and len(entries) >= MIN_FORMAL_DATASET_EPISODES
    )
    errors: list[str] = []
    maximum_errors = int(audit_cfg["maximum_errors_reported"])
    episode_metrics: list[dict[str, Any]] = []
    hdf5_paths: list[Path] = []
    for manifest_index, entry in enumerate(entries):
        try:
            if not isinstance(entry, Mapping):
                raise TypeError("episode manifest entry is not a mapping")
            hdf5_path = _resolve_inside(data_dir, str(entry["hdf5_path"]))
            videos = entry.get("videos")
            if not isinstance(videos, Mapping) or tuple(videos) != camera_order:
                raise ValueError("episode videos do not use canonical camera order")
            video_paths = {
                camera: _resolve_inside(data_dir, str(_mapping(videos, camera)["path"]))
                for camera in camera_order
            }
            metric = _audit_episode(
                hdf5_path,
                video_paths,
                entry,
                camera_cfg=camera_cfg,
                audit_cfg=audit_cfg,
                camera_order=camera_order,
            )
            episode_metrics.append(metric)
            hdf5_paths.append(hdf5_path)
        except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
            if len(errors) < maximum_errors:
                errors.append(f"episode entry {manifest_index}: {exc}")

    split_audit = _audit_splits(entries, config, manifest)
    loader_audit = (
        _loader_smoke(hdf5_paths, camera_cfg=camera_cfg, camera_order=camera_order)
        if run_loader_smoke and not errors
        else _check(not run_loader_smoke, skipped=not run_loader_smoke)
    )
    sync_skews = np.asarray(
        [
            value
            for item in episode_metrics
            for camera in camera_order
            for value in item["cameras"][camera]["capture_sync_skew"]
        ],
        dtype=np.float64,
    )
    frame_ages = np.asarray(
        [
            value
            for item in episode_metrics
            for camera in camera_order
            for value in item["cameras"][camera]["action_frame_age"]
        ],
        dtype=np.float64,
    )
    p99_skew = _percentile(sync_skews, 99)
    max_age = float(frame_ages.max()) if frame_ages.size else math.inf
    camera_rows = [
        item["cameras"][camera] for item in episode_metrics for camera in camera_order
    ]
    reuse = _weighted_rate(camera_rows, "reuse_count", "reuse_denominator")
    duplicates = _weighted_rate(
        camera_rows, "unexpected_duplicate_count", "captured_comparisons"
    )
    empty = _weighted_rate(camera_rows, "empty_frame_count", "captured_frames")
    corrupt_videos = sum(not item["video_valid"] for item in camera_rows)
    boundary_crossings = sum(
        item["episode_boundary_crossings"] for item in episode_metrics
    )
    cross_camera_sync_failures = sum(
        not item["cross_camera_sync"] for item in episode_metrics
    )
    dynamic_extrinsics_failures = sum(
        not item["dynamic_extrinsics_valid"] for item in episode_metrics
    )
    raw_signal_failures = sum(
        not item["visual_signal_valid"] for item in episode_metrics
    )
    cue_pair_audit = (
        _audit_cue_pair_pixels(
            data_dir,
            entries,
            camera_order=camera_order,
            minimum_changed_pixels=int(
                audit_cfg["minimum_visual_signal_changed_pixels"]
            ),
        )
        if not errors
        else _check(False, reason="episode audit errors prevent cue-pair audit")
    )
    event_semantics_audit = _audit_visual_event_signal_and_brake_lights()

    skew_limit = _threshold(
        acceptance_cfg, "capture_sync_skew_p99_seconds", expected_operator="<"
    )
    age_limit = _threshold(
        acceptance_cfg,
        "maximum_action_frame_age_seconds",
        expected_operator="<=",
    )
    manifest_audit = _audit_manifest(
        manifest,
        config=config,
        config_path=config_path,
        data_dir=data_dir,
        entries=entries,
        expected_episodes=expected_episodes,
        formal_protocol=formal_protocol,
    )
    checks = {
        "manifest_contract": manifest_audit,
        "episode_count": _check(
            len(entries) == expected_episodes,
            actual=len(entries),
            expected=expected_episodes,
        ),
        "minimum_formal_dataset_size": _minimum_formal_size_details(
            len(entries), formal_protocol=formal_protocol
        ),
        "all_episode_contracts": _check(
            not errors,
            audited=len(episode_metrics),
            errors=errors,
        ),
        "capture_sync_skew_p99": _threshold_check(p99_skew, skew_limit),
        "maximum_action_frame_age": _threshold_check(max_age, age_limit),
        "zero_episode_boundary_crossings": _check(
            boundary_crossings == 0, actual=boundary_crossings, expected=0
        ),
        "captured_frame_reuse_rate": _range_check(
            reuse,
            center=float(audit_cfg["expected_captured_frame_reuse_fraction"]),
            tolerance=float(audit_cfg["captured_frame_reuse_tolerance"]),
        ),
        "unexpected_duplicate_rate": _upper_check(
            duplicates, float(audit_cfg["maximum_unexpected_duplicate_rate"])
        ),
        "empty_frame_rate": _upper_check(
            empty, float(audit_cfg["maximum_empty_frame_rate"])
        ),
        "corrupt_videos": _upper_check(
            float(corrupt_videos), float(audit_cfg["maximum_corrupt_videos"])
        ),
        "cross_camera_frame_sync": _check(
            cross_camera_sync_failures == 0,
            failures=cross_camera_sync_failures,
        ),
        "fixed_static_and_robot_dynamic_extrinsics": _check(
            dynamic_extrinsics_failures == 0,
            failures=dynamic_extrinsics_failures,
        ),
        "raw_visual_signal_onset_and_visibility": _check(
            raw_signal_failures == 0,
            failures=raw_signal_failures,
        ),
        "paired_cue_pixels_differ_in_every_camera": cue_pair_audit,
        "visual_event_signal_semantic_isolation": event_semantics_audit[
            "visual_event_signal_semantic_isolation"
        ],
        "brake_lights_action_causal_and_red_only": event_semantics_audit[
            "brake_lights_action_causal_and_red_only"
        ],
        "split_isolation": split_audit,
        "multimodal_loader_smoke": loader_audit,
    }
    return {
        "format_version": FORMAT_VERSION,
        "gate": "M0-data",
        "formal_protocol": bool(formal_protocol),
        "passed": all(item["passed"] for item in checks.values()),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "data_dir": str(data_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "schema_profile": SCHEMA_PROFILE,
        "schema_version": SCHEMA_VERSION,
        "camera_order": list(camera_order),
        "episodes": len(entries),
        "audited_episodes": len(episode_metrics),
        "transitions": sum(item["transitions"] for item in episode_metrics),
        "metrics": {
            "capture_sync_skew_seconds": _summary(sync_skews),
            "action_frame_age_seconds": _summary(frame_ages),
            "captured_frame_reuse_rate": reuse,
            "unexpected_duplicate_rate": duplicates,
            "empty_frame_rate": empty,
            "corrupt_videos": corrupt_videos,
            "episode_boundary_crossings": boundary_crossings,
            "cross_camera_sync_failures": cross_camera_sync_failures,
            "dynamic_extrinsics_failures": dynamic_extrinsics_failures,
            "visual_signal_failures": raw_signal_failures,
        },
        "split_audit": split_audit,
        "loader_audit": loader_audit,
        "cue_pair_audit": cue_pair_audit,
        "event_semantics_audit": event_semantics_audit,
        "provenance": manifest.get("provenance"),
        "checks": checks,
    }


def _audit_visual_event_signal_and_brake_lights() -> dict[str, dict[str, Any]]:
    """Probe that event truth is isolated from action-causal robot brake lamps."""

    environments = (
        VisualRequiredEnv(
            VisualRequiredEnvConfig(
                task_id="visual_event_stop", render_cue_mode="truth"
            )
        ),
        VisualRequiredEnv(
            VisualRequiredEnvConfig(
                task_id="visual_event_stop", render_cue_mode="truth"
            )
        ),
        VisualRequiredEnv(
            VisualRequiredEnvConfig(
                task_id="visual_event_stop", render_cue_mode="opposite"
            )
        ),
    )
    stop, passed, opposite = environments
    off = np.asarray([0.025, 0.004, 0.004, 1.0], dtype=np.float32)
    red = np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    cruise = np.asarray([0, 0.9, 0, 1, 0, 0.9, 0, 1], dtype=np.float32)
    brake_robot_0 = np.asarray([0, 0.0, 0, 1, 0, 0.9, 0, 1], dtype=np.float32)
    brake_robot_1 = np.asarray([0, 0.9, 0, 1, 0, 0.0, 0, 1], dtype=np.float32)
    brake_both = np.asarray([0, 0.0, 0, 1, 0, 0.0, 0, 1], dtype=np.float32)
    semantic_details: dict[str, Any] = {}
    brake_details: dict[str, Any] = {}
    try:
        stop.reset(seed=200, randomize=False)
        passed.reset(seed=201, randomize=False)
        opposite.reset(seed=200, randomize=False)
        brake_ids = tuple(
            int(mujoco.mj_name2id(stop.model, mujoco.mjtObj.mjOBJ_GEOM, name))
            for name in ("robot_0_brake_light", "robot_1_brake_light")
        )
        signal_id = int(
            mujoco.mj_name2id(
                stop.model, mujoco.mjtObj.mjOBJ_GEOM, "visual_event_signal"
            )
        )
        if min(*brake_ids, signal_id) < 0:
            raise ValueError("visual event signal or robot brake-light geom is missing")

        initial_lamps_off = all(
            np.allclose(env.model.geom_rgba[list(brake_ids)], off, atol=1e-7)
            for env in environments
        )
        for _ in range(EVENT_DECISION_STEP):
            stop.step(cruise)
            passed.step(cruise)
            opposite.step(cruise)

        onset_lamps_off = all(
            np.allclose(env.model.geom_rgba[list(brake_ids)], off, atol=1e-7)
            for env in environments
        )
        stop_pass_changed = set(
            np.flatnonzero(
                np.any(
                    np.abs(stop.model.geom_rgba - passed.model.geom_rgba) > 1e-7,
                    axis=1,
                )
            ).tolist()
        )
        truth_opposite_changed = set(
            np.flatnonzero(
                np.any(
                    np.abs(stop.model.geom_rgba - opposite.model.geom_rgba) > 1e-7,
                    axis=1,
                )
            ).tolist()
        )
        semantic_passed = bool(
            initial_lamps_off
            and onset_lamps_off
            and stop_pass_changed == {signal_id}
            and truth_opposite_changed == {signal_id}
        )
        semantic_details = {
            "initial_lamps_off": initial_lamps_off,
            "onset_lamps_off": onset_lamps_off,
            "expected_changed_geom_ids": [signal_id],
            "stop_pass_changed_geom_ids": sorted(stop_pass_changed),
            "truth_opposite_changed_geom_ids": sorted(truth_opposite_changed),
        }

        lamp_samples = [
            env.model.geom_rgba[list(brake_ids)].copy() for env in environments
        ]
        _, _, _, _, stop_robot_0_info = stop.step(brake_robot_0)
        opposite.step(brake_robot_0)
        robot_0_state = stop.model.geom_rgba[list(brake_ids)].copy()
        opposite_robot_0_state = opposite.model.geom_rgba[list(brake_ids)].copy()
        lamp_samples.extend((robot_0_state, opposite_robot_0_state))

        _, _, _, _, stop_robot_1_info = stop.step(brake_robot_1)
        robot_1_state = stop.model.geom_rgba[list(brake_ids)].copy()
        lamp_samples.append(robot_1_state)

        _, _, _, _, stop_both_info = stop.step(brake_both)
        both_state = stop.model.geom_rgba[list(brake_ids)].copy()
        lamp_samples.append(both_state)

        _, _, _, _, pass_info = passed.step(cruise)
        pass_state = passed.model.geom_rgba[list(brake_ids)].copy()
        lamp_samples.append(pass_state)
        never_green = all(
            bool(np.all(sample[:, 1] <= float(off[1]) + 1e-7))
            for sample in lamp_samples
        )
        brake_passed = bool(
            np.allclose(robot_0_state, [red, off], atol=1e-7)
            and np.allclose(opposite_robot_0_state, [red, off], atol=1e-7)
            and np.allclose(robot_1_state, [off, red], atol=1e-7)
            and np.allclose(both_state, [red, red], atol=1e-7)
            and np.allclose(pass_state, off, atol=1e-7)
            and stop_robot_0_info.get("brake_light_active_agents") == [0]
            and stop_robot_1_info.get("brake_light_active_agents") == [1]
            and stop_both_info.get("brake_light_active_agents") == [0, 1]
            and pass_info.get("brake_light_active_agents") == []
            and never_green
        )
        brake_details = {
            "robot_0_only_valid": bool(
                np.allclose(robot_0_state, [red, off], atol=1e-7)
            ),
            "robot_1_only_valid": bool(
                np.allclose(robot_1_state, [off, red], atol=1e-7)
            ),
            "both_valid": bool(np.allclose(both_state, [red, red], atol=1e-7)),
            "pass_cruise_lamps_off": bool(
                np.allclose(pass_state, off, atol=1e-7)
            ),
            "truth_opposite_same_action_lamps_equal": bool(
                np.allclose(robot_0_state, opposite_robot_0_state, atol=1e-7)
            ),
            "never_green": never_green,
            "semantics": stop_robot_0_info.get("brake_light_semantics"),
        }
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        reason = f"{type(exc).__name__}: {exc}"
        return {
            "visual_event_signal_semantic_isolation": _check(False, reason=reason),
            "brake_lights_action_causal_and_red_only": _check(False, reason=reason),
        }
    finally:
        for env in environments:
            env.close()
    return {
        "visual_event_signal_semantic_isolation": _check(
            semantic_passed, **semantic_details
        ),
        "brake_lights_action_causal_and_red_only": _check(
            brake_passed, **brake_details
        ),
    }


def _audit_episode(
    path: Path,
    video_paths: Mapping[str, Path],
    entry: Mapping[str, Any],
    *,
    camera_cfg: Mapping[str, Any],
    audit_cfg: Mapping[str, Any],
    camera_order: Sequence[str],
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if _sha256(path) != str(entry.get("hdf5_sha256", "")):
        raise ValueError(f"HDF5 SHA-256 mismatch: {path}")
    with h5py.File(path, "r") as file:
        profile = str(file.attrs.get("schema_profile", ""))
        version = str(file.attrs.get("schema_version", ""))
        if profile != SCHEMA_PROFILE:
            raise ValueError(f"unexpected schema profile {profile!r}")
        if version != SCHEMA_VERSION:
            raise ValueError(f"unexpected schema version {version!r}")
        file_camera_order = json.loads(str(file.attrs.get("camera_order_json", "[]")))
        if file_camera_order != list(camera_order):
            raise ValueError(f"unexpected camera order {file_camera_order!r}")
        control_hz = float(camera_cfg["control_hz"])
        if not math.isclose(float(file.attrs.get("fps", -1.0)), control_hz):
            raise ValueError("HDF5 fps does not match control_hz")
        required_paths = _required_paths(camera_order)
        missing = [name for name in required_paths if name not in file]
        if missing:
            raise KeyError(f"missing required datasets: {missing}")
        all_paths: list[str] = []
        file.visit(all_paths.append)
        forbidden = [name for name in all_paths if "privileged" in name.lower()]
        if forbidden:
            raise ValueError(f"privileged datasets are forbidden: {forbidden[:10]}")

        steps = int(file.attrs.get("num_steps", -1))
        if steps <= 0:
            raise ValueError("episode has no completed transitions")
        for name in required_paths:
            if file[name].shape[0] != steps:
                raise ValueError(f"{name} length differs from num_steps")
        _validate_shapes_and_dtypes(
            file,
            steps=steps,
            camera_cfg=camera_cfg,
            camera_order=camera_order,
        )
        environment = _validate_constant_labels(file, steps=steps, entry=entry)
        if (
            environment.get("renderer_backend") != "mujoco.Renderer"
            or environment.get("geometry_source") != "mujoco_xml"
            or environment.get("raw_unannotated") is not True
            or environment.get("camera_order") != list(camera_order)
            or environment.get("mujoco_version") != str(mujoco.__version__)
            or environment.get("mujoco_gl") != str(entry.get("mujoco_gl", ""))
            or not str(environment.get("mujoco_gl", ""))
            or environment.get("model_xml_sha256")
            != str(entry.get("model_xml_sha256", ""))
        ):
            raise ValueError(
                "HDF5 environment provenance is not canonical MuJoCo raw RGB"
            )

        timestamp = np.asarray(file["data/timestamp"][:], dtype=np.float64)
        control_dt = 1.0 / float(camera_cfg["control_hz"])
        if not np.allclose(np.diff(timestamp), control_dt, atol=1e-9, rtol=0.0):
            raise ValueError("state timestamps are not a strict control-rate grid")
        next_timestamp = timestamp + control_dt
        tolerance = 1e-6
        camera_metrics: dict[str, dict[str, Any]] = {}
        synchronization_arrays: dict[str, tuple[np.ndarray, ...]] = {}
        dynamic_extrinsics_valid = True
        for camera in camera_order:
            image_ts = np.asarray(
                file[f"data/observation/image_timestamp/{camera}"][:],
                dtype=np.float64,
            )
            image_state_ts = np.asarray(
                file[f"data/observation/image_state_timestamp/{camera}"][:],
                dtype=np.float64,
            )
            next_image_ts = np.asarray(
                file[f"data/next_observation/image_timestamp/{camera}"][:],
                dtype=np.float64,
            )
            next_image_state_ts = np.asarray(
                file[f"data/next_observation/image_state_timestamp/{camera}"][:],
                dtype=np.float64,
            )
            frame_index = np.asarray(
                file[f"data/observation/image_frame_index/{camera}"][:],
                dtype=np.int64,
            )
            next_frame_index = np.asarray(
                file[f"data/next_observation/image_frame_index/{camera}"][:],
                dtype=np.int64,
            )
            if np.any(np.diff(image_ts) < -1e-9) or np.any(np.diff(frame_index) < 0):
                raise ValueError(f"{camera} observation image mapping is not monotonic")
            if np.any(np.diff(frame_index) > 1):
                raise ValueError(f"{camera} image frame index skips captured frames")
            if np.any(next_frame_index < frame_index) or np.any(
                next_frame_index - frame_index > 1
            ):
                raise ValueError(f"{camera} next image frame mapping is invalid")
            if not np.array_equal(next_frame_index[:-1], frame_index[1:]):
                raise ValueError(f"{camera} next/current frame references disagree")
            if np.any(image_ts > timestamp + tolerance) or np.any(
                next_image_ts > next_timestamp + tolerance
            ):
                raise ValueError(f"{camera} future RGB leaked into observation")
            if not np.allclose(
                next_image_ts[:-1], image_ts[1:], atol=tolerance, rtol=0.0
            ) or not np.allclose(
                next_image_state_ts[:-1],
                image_state_ts[1:],
                atol=tolerance,
                rtol=0.0,
            ):
                raise ValueError(f"{camera} next/current timing disagrees")
            ages = np.concatenate(
                [timestamp - image_ts, next_timestamp - next_image_ts]
            )
            if np.any(ages < -tolerance):
                raise ValueError(f"{camera} has negative action frame age")
            skews = np.concatenate(
                [
                    np.abs(image_ts - image_state_ts),
                    np.abs(next_image_ts - next_image_state_ts),
                ]
            )
            image_health = _image_health(
                file[f"data/observation/images/{camera}"],
                frame_index,
                file[f"data/next_observation/images/{camera}"],
                next_frame_index,
            )
            calibration = _calibration_health(
                file,
                camera=camera,
                frame_indices=frame_index,
                next_frame_indices=next_frame_index,
                audit_cfg=audit_cfg,
            )
            dynamic_extrinsics_valid = bool(
                dynamic_extrinsics_valid and calibration["passed"]
            )
            video = _video_health(
                video_paths[camera],
                _mapping(_mapping(entry, "videos"), camera),
                expected_captured_frames=int(image_health["captured_frames"]),
            )
            camera_metrics[camera] = {
                "capture_sync_skew": skews.tolist(),
                "action_frame_age": ages.tolist(),
                "calibration": calibration,
                **image_health,
                **video,
            }
            synchronization_arrays[camera] = (
                image_ts,
                image_state_ts,
                next_image_ts,
                next_image_state_ts,
                frame_index,
                next_frame_index,
            )
        reference = synchronization_arrays[camera_order[0]]
        cross_camera_sync = (
            all(
                all(
                    np.array_equal(left, right)
                    for left, right in zip(reference, values)
                )
                for camera, values in synchronization_arrays.items()
                if camera != camera_order[0]
            )
            and len({camera_metrics[camera]["video_frames"] for camera in camera_order})
            == 1
        )
        visual_signal = _audit_signal_evidence(
            file,
            entry,
            camera_order=camera_order,
            camera_cfg=camera_cfg,
            minimum_changed_pixels=int(
                audit_cfg["minimum_visual_signal_changed_pixels"]
            ),
        )
        episode_values = np.asarray(file["data/episode_index"][:], dtype=np.int64)
        boundary_crossings = int(np.count_nonzero(episode_values != episode_values[0]))
    return {
        "path": str(path),
        "task_id": str(entry["task_id"]),
        "physical_seed": int(entry["physical_seed"]),
        "cue_id": int(entry["cue_id"]),
        "transitions": steps,
        "episode_boundary_crossings": boundary_crossings,
        "cross_camera_sync": bool(cross_camera_sync),
        "dynamic_extrinsics_valid": bool(dynamic_extrinsics_valid),
        "visual_signal_valid": bool(visual_signal["passed"]),
        "visual_signal": visual_signal,
        "cameras": camera_metrics,
    }


def _validate_shapes_and_dtypes(
    file: h5py.File,
    *,
    steps: int,
    camera_cfg: Mapping[str, Any],
    camera_order: Sequence[str],
) -> None:
    height = int(camera_cfg["height"])
    width = int(camera_cfg["width"])
    channels = int(camera_cfg["channels"])
    expected_exact: dict[str, tuple[tuple[int, ...], Any]] = {}
    for camera in camera_order:
        for prefix in ("observation", "next_observation"):
            expected_exact[f"data/{prefix}/images/{camera}"] = (
                (steps, height, width, channels),
                np.uint8,
            )
        for prefix in ("camera", "next_camera"):
            expected_exact[f"data/{prefix}/intrinsics/{camera}"] = (
                (steps, 3, 3),
                np.float32,
            )
            expected_exact[f"data/{prefix}/extrinsics/{camera}"] = (
                (steps, 4, 4),
                np.float32,
            )
            expected_exact[f"data/{prefix}/resolution/{camera}"] = (
                (steps, 2),
                np.int64,
            )
    for name, (shape, dtype) in expected_exact.items():
        dataset = file[name]
        if tuple(dataset.shape) != shape:
            raise ValueError(f"{name} has shape {dataset.shape}, expected {shape}")
        if dataset.dtype != np.dtype(dtype):
            raise TypeError(
                f"{name} has dtype {dataset.dtype}, expected {np.dtype(dtype)}"
            )
    vector_shapes = {
        "data/observation/state": np.float32,
        "data/next_observation/state": np.float32,
        "data/action/commanded": np.float32,
        "data/action/executed": np.float32,
    }
    for name, dtype in vector_shapes.items():
        dataset = file[name]
        if (
            dataset.ndim != 2
            or dataset.shape[1] <= 0
            or dataset.dtype != np.dtype(dtype)
        ):
            raise TypeError(f"{name} must be a non-empty {np.dtype(dtype)} matrix")
        if not np.isfinite(dataset[:]).all():
            raise ValueError(f"{name} contains NaN/Inf")
    if (
        file["data/observation/state"].shape
        != file["data/next_observation/state"].shape
    ):
        raise ValueError("current and next state dimensions differ")
    if file["data/action/commanded"].shape != file["data/action/executed"].shape:
        raise ValueError("commanded and executed action dimensions differ")

    float64_paths = ["data/timestamp"]
    int64_paths = ["data/frame_index", "data/episode_index", "data/seed"]
    for camera in camera_order:
        float64_paths.extend(
            (
                f"data/observation/image_timestamp/{camera}",
                f"data/observation/image_state_timestamp/{camera}",
                f"data/next_observation/image_timestamp/{camera}",
                f"data/next_observation/image_state_timestamp/{camera}",
            )
        )
        int64_paths.extend(
            (
                f"data/observation/image_frame_index/{camera}",
                f"data/next_observation/image_frame_index/{camera}",
            )
        )
    for name in float64_paths:
        if (
            file[name].dtype != np.dtype(np.float64)
            or not np.isfinite(file[name][:]).all()
        ):
            raise TypeError(f"{name} must contain finite float64 values")
    for name in int64_paths:
        if file[name].dtype != np.dtype(np.int64):
            raise TypeError(f"{name} must contain int64 values")
    for name in ("data/reward",):
        if (
            file[name].dtype != np.dtype(np.float32)
            or not np.isfinite(file[name][:]).all()
        ):
            raise TypeError(f"{name} must contain finite float32 values")
    for name in (
        "data/terminated",
        "data/truncated",
        "data/done",
        "data/success",
        "data/failure",
    ):
        if file[name].dtype != np.dtype(np.bool_):
            raise TypeError(f"{name} must contain bool values")


def _validate_constant_labels(
    file: h5py.File, *, steps: int, entry: Mapping[str, Any]
) -> Mapping[str, Any]:
    expected = {
        "data/episode_index": int(entry["episode_index"]),
        "data/seed": int(entry["seed"]),
        "data/task/id": str(entry["task_id"]),
        "data/schema_version": SCHEMA_VERSION,
        "data/behavior_id": str(entry["behavior_id"]),
    }
    for name, value in expected.items():
        dataset = file[name]
        values = (
            dataset.asstr()[:] if h5py.check_string_dtype(dataset.dtype) else dataset[:]
        )
        if len(values) != steps or not all(item == value for item in values):
            raise ValueError(f"{name} changes within episode or mismatches manifest")
    if int(entry.get("steps", -1)) != steps:
        raise ValueError("manifest steps differ from HDF5 num_steps")
    task_text = file["data/task/text"].asstr()[:]
    if any(not str(value).strip() for value in task_text):
        raise ValueError("task text cannot be empty")
    parsed_configs: dict[str, Mapping[str, Any]] = {}
    for name in ("data/environment_config", "data/randomization_config"):
        values = file[name].asstr()[:]
        parsed = [json.loads(value) for value in values]
        if any(value != parsed[0] for value in parsed[1:]):
            raise ValueError(f"{name} changes within episode")
        if not isinstance(parsed[0], Mapping):
            raise ValueError(f"{name} must encode a JSON mapping")
        parsed_configs[name] = parsed[0]
    environment = parsed_configs["data/environment_config"]
    randomization = parsed_configs["data/randomization_config"]
    if str(environment.get("task_id", "")) != str(entry["task_id"]) or str(
        environment.get("randomization_template_id", "")
    ) != str(entry["template_id"]):
        raise ValueError("environment_config disagrees with manifest entry")
    expected_randomization = {
        "physical_seed": int(entry["physical_seed"]),
        "episode_seed": int(entry["seed"]),
        "cue_id": int(entry["cue_id"]),
        "cue_variant": int(entry["cue_id"]),
        "split": str(entry["split"]),
        "template_id": str(entry["template_id"]),
        "scene_id": str(entry["scene_id"]),
        "object_combination_id": str(entry["object_combination_id"]),
    }
    if any(
        randomization.get(key) != value for key, value in expected_randomization.items()
    ):
        raise ValueError("randomization_config disagrees with manifest entry")
    done = np.asarray(file["data/done"][:], dtype=np.bool_)
    terminated = np.asarray(file["data/terminated"][:], dtype=np.bool_)
    truncated = np.asarray(file["data/truncated"][:], dtype=np.bool_)
    if not np.array_equal(done, terminated | truncated):
        raise ValueError("done differs from terminated|truncated")
    success = bool(np.asarray(file["data/success"][-1]).item())
    failure = bool(np.asarray(file["data/failure"][-1]).item())
    if success is failure:
        raise ValueError("final outcome must be exactly one of success/failure")
    failure_reason = str(file["data/failure_reason"].asstr()[-1])
    if (
        success != bool(entry.get("success"))
        or failure != bool(entry.get("failure"))
        or failure_reason != str(entry.get("failure_reason", ""))
    ):
        raise ValueError("HDF5 final outcome disagrees with manifest entry")
    if bool(file.attrs.get("terminated", False)) != bool(terminated[-1]) or bool(
        file.attrs.get("truncated", False)
    ) != bool(truncated[-1]):
        raise ValueError("HDF5 final flags disagree with episode attributes")
    return environment


def _calibration_health(
    file: h5py.File,
    *,
    camera: str,
    frame_indices: np.ndarray,
    next_frame_indices: np.ndarray,
    audit_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    current_intrinsics = np.asarray(
        file[f"data/camera/intrinsics/{camera}"][:], dtype=np.float64
    )
    next_intrinsics = np.asarray(
        file[f"data/next_camera/intrinsics/{camera}"][:], dtype=np.float64
    )
    current_extrinsics = np.asarray(
        file[f"data/camera/extrinsics/{camera}"][:], dtype=np.float64
    )
    next_extrinsics = np.asarray(
        file[f"data/next_camera/extrinsics/{camera}"][:], dtype=np.float64
    )
    current_resolution = np.asarray(
        file[f"data/camera/resolution/{camera}"][:], dtype=np.int64
    )
    next_resolution = np.asarray(
        file[f"data/next_camera/resolution/{camera}"][:], dtype=np.int64
    )

    continuity = bool(
        np.allclose(next_intrinsics[:-1], current_intrinsics[1:], atol=1e-6, rtol=0.0)
        and np.allclose(
            next_extrinsics[:-1], current_extrinsics[1:], atol=1e-6, rtol=0.0
        )
        and np.array_equal(next_resolution[:-1], current_resolution[1:])
    )
    expected_resolution = np.asarray(
        file[f"data/observation/images/{camera}"].shape[1:3], dtype=np.int64
    )
    resolution_valid = bool(
        np.all(current_resolution == expected_resolution)
        and np.all(next_resolution == expected_resolution)
    )
    intrinsics_valid = bool(
        np.isfinite(current_intrinsics).all()
        and np.isfinite(next_intrinsics).all()
        and np.all(current_intrinsics[:, 0, 0] > 0.0)
        and np.all(current_intrinsics[:, 1, 1] > 0.0)
        and np.all(next_intrinsics[:, 0, 0] > 0.0)
        and np.all(next_intrinsics[:, 1, 1] > 0.0)
        and np.allclose(current_intrinsics[:, 2, :], (0.0, 0.0, 1.0), atol=1e-6)
        and np.allclose(next_intrinsics[:, 2, :], (0.0, 0.0, 1.0), atol=1e-6)
    )
    poses_valid = _valid_pose_batch(current_extrinsics) and _valid_pose_batch(
        next_extrinsics
    )

    captured_poses: list[np.ndarray] = []
    last_index = -1
    for row, raw_index in enumerate(frame_indices):
        index = int(raw_index)
        if index != last_index:
            captured_poses.append(current_extrinsics[row])
            last_index = index
    if int(next_frame_indices[-1]) != last_index:
        captured_poses.append(next_extrinsics[-1])
    pose_stack = np.stack(captured_poses, axis=0)
    max_abs_delta = float(np.max(np.abs(pose_stack - pose_stack[0])))
    translation_travel = float(
        np.max(np.linalg.norm(pose_stack[:, :3, 3] - pose_stack[0, :3, 3], axis=1))
    )
    if camera == "fixed":
        role_valid = max_abs_delta <= float(
            audit_cfg["static_extrinsics_max_abs_delta"]
        )
        expected_role = "static"
    else:
        role_valid = translation_travel >= float(
            audit_cfg["dynamic_extrinsics_min_translation_m"]
        )
        expected_role = "dynamic"
    passed = bool(
        continuity
        and resolution_valid
        and intrinsics_valid
        and poses_valid
        and role_valid
    )
    return _check(
        passed,
        role=expected_role,
        continuity=continuity,
        resolution_valid=resolution_valid,
        intrinsics_valid=intrinsics_valid,
        poses_valid=poses_valid,
        role_valid=role_valid,
        captured_poses=len(captured_poses),
        max_abs_extrinsics_delta=max_abs_delta,
        translation_travel_m=translation_travel,
    )


def _valid_pose_batch(values: np.ndarray) -> bool:
    if values.ndim != 3 or values.shape[1:] != (4, 4) or not np.isfinite(values).all():
        return False
    rotations = values[:, :3, :3]
    identity = np.eye(3, dtype=np.float64)[None, :, :]
    return bool(
        np.allclose(values[:, 3, :], (0.0, 0.0, 0.0, 1.0), atol=1e-5)
        and np.allclose(
            np.swapaxes(rotations, 1, 2) @ rotations,
            identity,
            atol=2e-4,
            rtol=0.0,
        )
        and np.allclose(np.linalg.det(rotations), 1.0, atol=2e-4, rtol=0.0)
    )


def _audit_signal_evidence(
    file: h5py.File,
    entry: Mapping[str, Any],
    *,
    camera_order: Sequence[str],
    camera_cfg: Mapping[str, Any],
    minimum_changed_pixels: int,
) -> dict[str, Any]:
    signal = entry.get("visual_signal")
    if not isinstance(signal, Mapping):
        raise ValueError("episode visual_signal evidence must be a mapping")
    camera_evidence = signal.get("cameras")
    visibility = signal.get("visibility_expected")
    if (
        not isinstance(camera_evidence, Mapping)
        or tuple(camera_evidence) != tuple(camera_order)
        or not isinstance(visibility, Mapping)
        or tuple(visibility) != tuple(camera_order)
    ):
        raise ValueError("visual signal evidence does not cover canonical cameras")
    if signal.get("active_observed") is not True or not all(
        visibility.get(camera) is True for camera in camera_order
    ):
        raise ValueError("visual signal was not active/expected in every camera")
    onset_step = int(signal.get("onset_step", -1))
    kind = str(signal.get("kind", ""))
    if onset_step < 0 or not kind:
        raise ValueError("visual signal onset metadata is invalid")

    task_id = str(entry.get("task_id", ""))
    per_camera: dict[str, Any] = {}
    for camera in camera_order:
        evidence = _mapping(camera_evidence, camera)
        if evidence.get("visible_expected") is not True:
            raise ValueError(f"{camera} visual signal was not expected visible")
        active_index = int(evidence.get("active_frame_index", -1))
        active_timestamp, active_rgb = _captured_frame(
            file, camera=camera, frame_index=active_index
        )
        active_hash_valid = str(evidence.get("active_rgb_sha256", "")) == _array_sha256(
            active_rgb
        )
        active_time_valid = math.isclose(
            float(evidence.get("active_timestamp", math.nan)),
            active_timestamp,
            abs_tol=1e-9,
            rel_tol=0.0,
        )
        pre_index_raw = evidence.get("pre_frame_index")
        pre_valid = True
        changed_pixels: int | None = None
        if pre_index_raw is not None:
            pre_index = int(pre_index_raw)
            pre_timestamp, pre_rgb = _captured_frame(
                file, camera=camera, frame_index=pre_index
            )
            changed_pixels = int(
                np.count_nonzero(np.any(active_rgb != pre_rgb, axis=2))
            )
            pre_valid = bool(
                0 <= pre_index < active_index
                and str(evidence.get("pre_rgb_sha256", "")) == _array_sha256(pre_rgb)
                and math.isclose(
                    float(evidence.get("pre_timestamp", math.nan)),
                    pre_timestamp,
                    abs_tol=1e-9,
                    rel_tol=0.0,
                )
                and int(evidence.get("changed_pixels", -1)) == changed_pixels
            )
        elif any(
            evidence.get(name) is not None
            for name in ("pre_timestamp", "pre_rgb_sha256", "changed_pixels")
        ):
            pre_valid = False

        event_valid = True
        if task_id == "visual_event_stop":
            event_valid = bool(
                pre_index_raw is not None
                and changed_pixels is not None
                and changed_pixels >= minimum_changed_pixels
                and onset_step > 0
            )
        else:
            event_valid = onset_step == 0
        camera_valid = bool(
            active_hash_valid and active_time_valid and pre_valid and event_valid
        )
        per_camera[camera] = {
            "passed": camera_valid,
            "active_frame_index": active_index,
            "active_timestamp": active_timestamp,
            "changed_pixels": changed_pixels,
        }
    image_period = 1.0 / float(camera_cfg["image_hz"])
    onset_time = onset_step / float(camera_cfg["control_hz"])
    active_times = [item["active_timestamp"] for item in per_camera.values()]
    onset_timing_valid = bool(
        task_id != "visual_event_stop"
        or all(
            onset_time - 1e-6 <= value <= onset_time + image_period + 1e-6
            for value in active_times
        )
    )
    passed = bool(
        onset_timing_valid and all(item["passed"] for item in per_camera.values())
    )
    return _check(
        passed,
        kind=kind,
        onset_step=onset_step,
        onset_time_seconds=onset_time,
        onset_timing_valid=onset_timing_valid,
        cameras=per_camera,
    )


def _captured_frame(
    file: h5py.File, *, camera: str, frame_index: int
) -> tuple[float, np.ndarray]:
    candidates: list[tuple[float, np.ndarray]] = []
    for prefix in ("observation", "next_observation"):
        indices = np.asarray(
            file[f"data/{prefix}/image_frame_index/{camera}"][:], dtype=np.int64
        )
        rows = np.flatnonzero(indices == frame_index)
        timestamps = file[f"data/{prefix}/image_timestamp/{camera}"]
        images = file[f"data/{prefix}/images/{camera}"]
        for row in rows:
            candidates.append(
                (
                    float(timestamps[int(row)]),
                    np.asarray(images[int(row)], dtype=np.uint8),
                )
            )
    if not candidates:
        raise ValueError(f"{camera} captured frame {frame_index} is absent from HDF5")
    timestamp, rgb = candidates[0]
    if any(
        not math.isclose(value_timestamp, timestamp, abs_tol=1e-9, rel_tol=0.0)
        or not np.array_equal(value_rgb, rgb)
        for value_timestamp, value_rgb in candidates[1:]
    ):
        raise ValueError(f"{camera} captured frame {frame_index} has inconsistent rows")
    return timestamp, rgb


def _image_health(
    images: h5py.Dataset,
    frame_indices: np.ndarray,
    next_images: h5py.Dataset,
    next_frame_indices: np.ndarray,
) -> dict[str, Any]:
    for row in range(max(0, len(frame_indices) - 1)):
        if not np.array_equal(next_images[row], images[row + 1]):
            raise ValueError("next/current RGB payloads disagree")
    if next_frame_indices[-1] == frame_indices[-1]:
        if not np.array_equal(next_images[-1], images[-1]):
            raise ValueError("terminal reused RGB payload disagrees")
    elif next_frame_indices[-1] != frame_indices[-1] + 1:
        raise ValueError("terminal next RGB frame index is not contiguous")

    changed = np.concatenate(([True], np.diff(frame_indices) != 0))
    captured_indices = np.flatnonzero(changed)
    frames = [np.asarray(images[int(row)], dtype=np.uint8) for row in captured_indices]
    if next_frame_indices[-1] != frame_indices[-1]:
        frames.append(np.asarray(next_images[-1], dtype=np.uint8))
    unique_indices = [int(frame_indices[row]) for row in captured_indices]
    if next_frame_indices[-1] != frame_indices[-1]:
        unique_indices.append(int(next_frame_indices[-1]))
    if unique_indices != list(range(len(unique_indices))):
        raise ValueError("captured RGB frame indices must be contiguous from zero")
    previous_digest: str | None = None
    duplicates = 0
    empty = 0
    for frame in frames:
        empty += int(not np.any(frame))
        digest = hashlib.sha256(frame.tobytes()).hexdigest()
        duplicates += int(previous_digest is not None and digest == previous_digest)
        previous_digest = digest
    denominator = max(0, len(frame_indices) - 1)
    reuse_count = int(np.count_nonzero(np.diff(frame_indices) == 0))
    return {
        "captured_frames": len(frames),
        "empty_frame_count": int(empty),
        "unexpected_duplicate_count": int(duplicates),
        "captured_comparisons": max(0, len(frames) - 1),
        "reuse_count": reuse_count,
        "reuse_denominator": denominator,
    }


def _video_health(
    path: Path,
    entry: Mapping[str, Any],
    *,
    expected_captured_frames: int,
) -> dict[str, Any]:
    if not path.is_file() or _sha256(path) != str(entry.get("sha256", "")):
        return {"video_valid": False, "video_frames": 0}
    capture = cv2.VideoCapture(str(path))
    frames = 0
    valid = capture.isOpened()
    expected_width = int(entry.get("width", 0))
    expected_height = int(entry.get("height", 0))
    expected_fps = float(entry.get("fps", 0.0))
    decoded_fps = float(capture.get(cv2.CAP_PROP_FPS)) if valid else 0.0
    try:
        while valid:
            ok, frame = capture.read()
            if not ok:
                break
            frames += 1
            valid = valid and bool(
                frame.ndim == 3
                and frame.shape[1] == expected_width
                and frame.shape[0] == expected_height
                and np.any(frame)
            )
    finally:
        capture.release()
    manifest_frames = int(entry.get("captured_frames", -1))
    valid = bool(
        valid
        and frames > 0
        and frames == expected_captured_frames
        and manifest_frames == expected_captured_frames
        and expected_width > 0
        and expected_height > 0
        and expected_fps > 0.0
        and math.isclose(decoded_fps, expected_fps, abs_tol=0.05, rel_tol=0.01)
        and path.suffix.lower() == ".mp4"
    )
    return {
        "video_valid": bool(valid),
        "video_frames": frames,
        "video_fps": decoded_fps,
    }


def _audit_cue_pair_pixels(
    data_dir: Path,
    entries: Sequence[Mapping[str, Any]],
    *,
    camera_order: Sequence[str],
    minimum_changed_pixels: int,
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[(str(entry["task_id"]), int(entry["physical_seed"]))].append(entry)
    failures: list[dict[str, Any]] = []
    comparisons = 0
    pre_onset_comparisons = 0
    pre_onset_sequences_valid = True
    for (task_id, physical_seed), pair in grouped.items():
        cue_map = {int(entry["cue_id"]): entry for entry in pair}
        identity_valid = bool(
            len(pair) == 2
            and set(cue_map) == {0, 1}
            and len(
                {
                    (
                        str(entry["scene_id"]),
                        str(entry["object_combination_id"]),
                        str(entry["template_id"]),
                    )
                    for entry in pair
                }
            )
            == 1
        )
        if not identity_valid:
            failures.append(
                {
                    "task_id": task_id,
                    "physical_seed": physical_seed,
                    "reason": "incomplete or physically inconsistent cue pair",
                }
            )
            continue
        left_path = _resolve_inside(data_dir, str(cue_map[0]["hdf5_path"]))
        right_path = _resolve_inside(data_dir, str(cue_map[1]["hdf5_path"]))
        with h5py.File(left_path, "r") as left, h5py.File(right_path, "r") as right:
            for camera in camera_order:
                left_signal = _mapping(_mapping(cue_map[0], "visual_signal"), "cameras")
                right_signal = _mapping(
                    _mapping(cue_map[1], "visual_signal"), "cameras"
                )
                left_index = int(_mapping(left_signal, camera)["active_frame_index"])
                right_index = int(_mapping(right_signal, camera)["active_frame_index"])
                left_timestamp, left_rgb = _captured_frame(
                    left, camera=camera, frame_index=left_index
                )
                right_timestamp, right_rgb = _captured_frame(
                    right, camera=camera, frame_index=right_index
                )
                onset_alignment_valid = bool(
                    left_index == right_index
                    and math.isclose(
                        left_timestamp,
                        right_timestamp,
                        abs_tol=1e-9,
                        rel_tol=0.0,
                    )
                )
                if task_id == "visual_event_stop":
                    event_pre_valid = onset_alignment_valid and left_index > 0
                    if event_pre_valid:
                        for frame_index in range(left_index):
                            left_pre_timestamp, left_pre_rgb = _captured_frame(
                                left, camera=camera, frame_index=frame_index
                            )
                            right_pre_timestamp, right_pre_rgb = _captured_frame(
                                right, camera=camera, frame_index=frame_index
                            )
                            pre_onset_comparisons += 1
                            event_pre_valid = bool(
                                event_pre_valid
                                and math.isclose(
                                    left_pre_timestamp,
                                    right_pre_timestamp,
                                    abs_tol=1e-9,
                                    rel_tol=0.0,
                                )
                                and _array_sha256(left_pre_rgb)
                                == _array_sha256(right_pre_rgb)
                            )
                    pre_onset_sequences_valid = bool(
                        pre_onset_sequences_valid and event_pre_valid
                    )
                    if not event_pre_valid and len(failures) < 100:
                        failures.append(
                            {
                                "task_id": task_id,
                                "physical_seed": physical_seed,
                                "camera": camera,
                                "reason": "pre-onset RGB/timing sequence differs across cue pair",
                            }
                        )
                changed = int(np.count_nonzero(np.any(left_rgb != right_rgb, axis=2)))
                comparisons += 1
                if (
                    not onset_alignment_valid or changed < minimum_changed_pixels
                ) and len(failures) < 100:
                    failures.append(
                        {
                            "task_id": task_id,
                            "physical_seed": physical_seed,
                            "camera": camera,
                            "changed_pixels": changed,
                            "onset_alignment_valid": onset_alignment_valid,
                        }
                    )
    expected_comparisons = len(grouped) * len(camera_order)
    return _check(
        not failures and comparisons == expected_comparisons,
        cue_pairs=len(grouped),
        cameras_per_pair=len(camera_order),
        comparisons=comparisons,
        expected_comparisons=expected_comparisons,
        pre_onset_sequences_valid=pre_onset_sequences_valid,
        pre_onset_frame_comparisons=pre_onset_comparisons,
        minimum_changed_pixels=minimum_changed_pixels,
        failures=failures,
    )


def _audit_splits(
    entries: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    split_cfg = _mapping(_mapping(config, "dataset"), "split")
    expected_splits = ("train", "validation", "test")
    by_split: dict[str, list[Mapping[str, Any]]] = {
        split: [entry for entry in entries if entry.get("split") == split]
        for split in expected_splits
    }
    episode_sets = {
        split: {int(entry["episode_index"]) for entry in values}
        for split, values in by_split.items()
    }
    seed_sets = {
        split: {int(entry["physical_seed"]) for entry in values}
        for split, values in by_split.items()
    }
    template_sets = {
        split: {str(entry["template_id"]) for entry in values}
        for split, values in by_split.items()
    }
    scene_sets = {
        split: {str(entry["scene_id"]) for entry in values}
        for split, values in by_split.items()
    }
    object_sets = {
        split: {str(entry["object_combination_id"]) for entry in values}
        for split, values in by_split.items()
    }
    overlaps: dict[str, Any] = {}
    overlap_zero = True
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        values = {
            "episodes": sorted(episode_sets[left] & episode_sets[right]),
            "seeds": sorted(seed_sets[left] & seed_sets[right]),
            "templates": sorted(template_sets[left] & template_sets[right]),
            "scenes": sorted(scene_sets[left] & scene_sets[right]),
            "object_combinations": sorted(object_sets[left] & object_sets[right]),
        }
        overlaps[f"{left}-{right}"] = values
        overlap_zero = overlap_zero and not any(values.values())
    tasks = tuple(_visual_required_tasks(manifest))
    cues = {int(value) for value in _mapping(config, "dataset")["cue_variants"]}
    coverage = {}
    counts_valid = True
    pairing_valid = True
    identities_pair_consistent = True
    templates_match_config = True
    for split, values in by_split.items():
        expected_count = int(_mapping(split_cfg, split)["episodes"]) * len(tasks)
        counts_valid = counts_valid and len(values) == expected_count
        task_cues = {
            task: sorted(
                {int(entry["cue_id"]) for entry in values if entry["task_id"] == task}
            )
            for task in tasks
        }
        coverage[split] = {
            "episodes": len(values),
            "expected_episodes": expected_count,
            "task_cues": task_cues,
        }
        counts_valid = counts_valid and all(
            set(values_) == cues for values_ in task_cues.values()
        )
        templates_match_config = templates_match_config and template_sets[split] == {
            str(value) for value in _mapping(split_cfg, split)["template_ids"]
        }
        grouped: dict[tuple[str, int], set[int]] = defaultdict(set)
        identities: dict[tuple[str, int], set[tuple[str, str]]] = defaultdict(set)
        for entry in values:
            grouped[(str(entry["task_id"]), int(entry["physical_seed"]))].add(
                int(entry["cue_id"])
            )
            identities[(str(entry["task_id"]), int(entry["physical_seed"]))].add(
                (str(entry["scene_id"]), str(entry["object_combination_id"]))
            )
        pairing_valid = pairing_valid and all(
            values_ == cues for values_ in grouped.values()
        )
        identities_pair_consistent = identities_pair_consistent and all(
            len(values_) == 1 for values_ in identities.values()
        )
    passed = bool(
        overlap_zero
        and counts_valid
        and pairing_valid
        and identities_pair_consistent
        and templates_match_config
    )
    return _check(
        passed,
        overlap_zero=overlap_zero,
        counts_valid=counts_valid,
        physical_seed_pairs_complete=pairing_valid,
        cue_pair_scene_object_identity_consistent=identities_pair_consistent,
        templates_match_config=templates_match_config,
        overlaps=overlaps,
        coverage=coverage,
    )


def _loader_smoke(
    paths: Sequence[Path],
    *,
    camera_cfg: Mapping[str, Any],
    camera_order: Sequence[str],
) -> dict[str, Any]:
    if len(paths) < 2:
        return _check(False, reason="loader smoke requires at least two episodes")
    try:
        from train.multimodal_trajectory_dataset import MultimodalSequenceDataset

        dataset = MultimodalSequenceDataset(
            paths=paths,
            history_horizon=4,
            forecast_horizon=4,
            camera_order=tuple(camera_order),
            max_frame_age_seconds=1.0 / float(camera_cfg["image_hz"]),
            hdf5_cache_size=1,
        )
        try:
            first = dataset[0]
            boundary_index = dataset.records[0].num_steps - 1
            boundary = dataset[boundary_index]
            second = dataset[dataset.records[0].num_steps]
            required = {
                "images",
                "target_images",
                "image_timestamps",
                "image_frame_indices",
                "camera_intrinsics",
                "camera_extrinsics",
                "camera_resolution",
                "target_camera_intrinsics",
                "target_camera_extrinsics",
                "target_camera_resolution",
                "states",
                "candidate_actions",
                "episode_index",
                "task_id",
                "task_text",
            }
            passed = bool(
                required <= set(first)
                and tuple(first["images"].shape[1:])
                == (
                    len(camera_order),
                    3,
                    int(camera_cfg["height"]),
                    int(camera_cfg["width"]),
                )
                and tuple(first["camera_intrinsics"].shape[1:])
                == (len(camera_order), 3, 3)
                and tuple(first["camera_extrinsics"].shape[1:])
                == (len(camera_order), 4, 4)
                and tuple(first["camera_resolution"].shape[1:])
                == (len(camera_order), 2)
                and int(boundary["forecast_mask"].sum()) == 1
                and int(second["valid_mask"].sum()) == 1
                and int(second["episode_index"]) != int(first["episode_index"])
            )
        finally:
            dataset.close()
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return _check(False, reason=f"{type(exc).__name__}: {exc}")
    return _check(passed, sampled_episodes=2, camera_order=list(camera_order))


def _audit_manifest(
    manifest: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    config_path: Path,
    data_dir: Path,
    entries: Sequence[Mapping[str, Any]],
    expected_episodes: int,
    formal_protocol: bool,
) -> dict[str, Any]:
    try:
        dataset_cfg = _mapping(config, "dataset")
        camera_order = _camera_order(config)
        dataset_card = _repository_file(str(dataset_cfg["dataset_card"]))
        artifact_paths: list[str] = []
        entry_provenance_valid = True
        for entry in entries:
            artifact_paths.append(str(entry["hdf5_path"]))
            videos = entry.get("videos")
            if not isinstance(videos, Mapping) or tuple(videos) != camera_order:
                raise ValueError(
                    "manifest episode does not contain three canonical videos"
                )
            artifact_paths.extend(
                str(_mapping(videos, camera)["path"]) for camera in camera_order
            )
            entry_provenance_valid = bool(
                entry_provenance_valid
                and entry.get("renderer_backend") == "mujoco.Renderer"
                and entry.get("geometry_source") == "mujoco_xml"
                and entry.get("mujoco_version") == str(mujoco.__version__)
                and isinstance(entry.get("mujoco_gl"), str)
                and bool(entry.get("mujoco_gl"))
                and entry.get("raw_unannotated") is True
            )
        paths_unique = len(set(artifact_paths)) == len(artifact_paths)
        episode_indices = [int(entry.get("episode_index", -1)) for entry in entries]
        provenance = _audit_manifest_provenance(
            manifest, config=config, camera_order=camera_order
        )
        passed = bool(
            manifest.get("format_version") == MANIFEST_FORMAT_VERSION
            and manifest.get("tasks") == list(VISUAL_REQUIRED_TASKS)
            and int(manifest.get("episodes_per_task", -1))
            == expected_episodes // len(VISUAL_REQUIRED_TASKS)
            and manifest.get("schema_profile") == SCHEMA_PROFILE
            and manifest.get("schema_version") == SCHEMA_VERSION
            and manifest.get("camera_order") == list(camera_order)
            and manifest.get("raw_unannotated") is True
            and isinstance(manifest.get("formal_protocol"), bool)
            and manifest.get("formal_protocol") == formal_protocol
            and manifest.get("config_sha256") == _sha256(config_path)
            and manifest.get("dataset_card_sha256") == _sha256(dataset_card)
            and len(entries) == expected_episodes
            and sorted(episode_indices) == list(range(expected_episodes))
            and paths_unique
            and entry_provenance_valid
            and provenance["passed"]
        )
        return _check(
            passed,
            format_version=manifest.get("format_version"),
            manifest_formal_protocol=manifest.get("formal_protocol"),
            expected_formal_protocol=bool(formal_protocol),
            config_sha256=manifest.get("config_sha256"),
            dataset_card_sha256=manifest.get("dataset_card_sha256"),
            expected_episodes=expected_episodes,
            expected_tasks=list(VISUAL_REQUIRED_TASKS),
            camera_order=list(camera_order),
            artifact_paths_unique=paths_unique,
            entry_provenance_valid=entry_provenance_valid,
            provenance=provenance,
            data_dir=str(data_dir),
        )
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        return _check(False, reason=f"{type(exc).__name__}: {exc}")


def _audit_manifest_provenance(
    manifest: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    camera_order: Sequence[str],
) -> dict[str, Any]:
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("manifest provenance must be a mapping")
    model_xml_path = _repository_file(str(provenance.get("model_xml_path", "")))
    model_xml_sha_valid = str(provenance.get("model_xml_sha256", "")) == _sha256(
        model_xml_path
    )
    required_sources_raw = _mapping(config, "audit").get("required_source_files")
    if (
        not isinstance(required_sources_raw, Sequence)
        or isinstance(required_sources_raw, (str, bytes))
        or not required_sources_raw
    ):
        raise ValueError("audit.required_source_files must be a sequence")
    required_sources = tuple(
        Path(str(value)).as_posix() for value in required_sources_raw
    )
    source_sha256 = provenance.get("source_sha256")
    if not isinstance(source_sha256, Mapping) or set(source_sha256) != set(
        required_sources
    ):
        raise ValueError(
            "manifest source SHA map does not exactly cover required sources"
        )
    source_sha_valid = all(
        str(source_sha256[path]) == _sha256(_repository_file(path))
        for path in required_sources
    )
    rig = provenance.get("camera_rig")
    if not isinstance(rig, Mapping) or tuple(rig) != tuple(camera_order):
        raise ValueError("manifest camera rig does not exactly cover canonical cameras")

    model = mujoco.MjModel.from_xml_path(str(model_xml_path))
    rig_valid = True
    parent_names: dict[str, str] = {}
    camera_ids: list[int] = []
    for camera in camera_order:
        values = _mapping(rig, camera)
        actual_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera))
        if actual_id < 0:
            rig_valid = False
            continue
        body_id = int(model.cam_bodyid[actual_id])
        body_name = str(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        )
        parent_names[camera] = body_name
        camera_ids.append(actual_id)
        rig_valid = bool(
            rig_valid
            and int(values.get("camera_id", -1)) == actual_id
            and int(values.get("parent_body_id", -1)) == body_id
            and str(values.get("parent_body_name", "")) == body_name
            and math.isfinite(float(values.get("fovy_degrees", math.nan)))
            and 0.0 < float(values.get("fovy_degrees", math.nan)) < 180.0
            and values.get("convention") == "opencv_optical_camera_pose_in_world"
        )
    rig_valid = bool(
        rig_valid
        and len(set(camera_ids)) == len(camera_order)
        and parent_names.get("fixed") == "world"
        and all(
            parent_names.get(camera, "") not in {"", "world"}
            for camera in camera_order[1:]
        )
        and len({parent_names.get(camera) for camera in camera_order[1:]})
        == len(camera_order) - 1
    )
    core_valid = bool(
        provenance.get("renderer_backend") == "mujoco.Renderer"
        and provenance.get("geometry_source") == "mujoco_xml"
        and provenance.get("mujoco_version") == str(mujoco.__version__)
        and isinstance(provenance.get("mujoco_gl"), str)
        and bool(provenance.get("mujoco_gl"))
    )
    return _check(
        core_valid and model_xml_sha_valid and source_sha_valid and rig_valid,
        renderer_backend=provenance.get("renderer_backend"),
        geometry_source=provenance.get("geometry_source"),
        mujoco_version=provenance.get("mujoco_version"),
        mujoco_gl=provenance.get("mujoco_gl"),
        model_xml_path=str(model_xml_path),
        model_xml_sha256=provenance.get("model_xml_sha256"),
        model_xml_sha_valid=model_xml_sha_valid,
        source_sha_valid=source_sha_valid,
        camera_rig_valid=rig_valid,
    )


def _visual_required_tasks(manifest: Mapping[str, Any]) -> list[str]:
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks or any(not str(item) for item in tasks):
        raise ValueError("manifest tasks must be a non-empty list")
    if len(set(map(str, tasks))) != len(tasks):
        raise ValueError("manifest tasks contain duplicates")
    if tuple(map(str, tasks)) != VISUAL_REQUIRED_TASKS:
        raise ValueError(
            "manifest tasks differ from the canonical visual-required suite"
        )
    return [str(item) for item in tasks]


def _weighted_rate(
    items: Sequence[Mapping[str, Any]], numerator: str, denominator: str
) -> float:
    top = sum(int(item[numerator]) for item in items)
    bottom = sum(int(item[denominator]) for item in items)
    return top / bottom if bottom else 0.0


def _minimum_formal_size_details(
    episodes: int, *, formal_protocol: bool
) -> dict[str, Any]:
    satisfied = int(episodes) >= MIN_FORMAL_DATASET_EPISODES
    return _check(
        not formal_protocol or satisfied,
        actual=int(episodes),
        minimum=MIN_FORMAL_DATASET_EPISODES,
        formal_protocol=bool(formal_protocol),
        formal_requirement_satisfied=satisfied,
    )


def _summary(values: np.ndarray) -> dict[str, Any]:
    if not values.size:
        return {"samples": 0, "mean": None, "p95": None, "p99": None, "max": None}
    return {
        "samples": int(values.size),
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _percentile(values: np.ndarray, percentile: int) -> float:
    return float(np.percentile(values, percentile)) if values.size else math.inf


def _range_check(actual: float, *, center: float, tolerance: float) -> dict[str, Any]:
    return _check(
        abs(actual - center) <= tolerance,
        actual=actual,
        minimum=center - tolerance,
        maximum=center + tolerance,
    )


def _upper_check(actual: float, maximum: float) -> dict[str, Any]:
    return _check(actual <= maximum, actual=actual, operator="<=", threshold=maximum)


def _threshold(
    config: Mapping[str, Any], name: str, *, expected_operator: str
) -> dict[str, Any]:
    raw = config.get(name)
    if not isinstance(raw, Mapping):
        raise ValueError(f"missing threshold {name!r}")
    operator = str(raw.get("operator", ""))
    value = raw.get("value")
    if operator != expected_operator or not isinstance(value, (int, float)):
        raise ValueError(f"invalid threshold {name!r}")
    return {"operator": operator, "value": float(value)}


def _threshold_check(actual: float, threshold: Mapping[str, Any]) -> dict[str, Any]:
    operator = threshold["operator"]
    value = float(threshold["value"])
    passed = actual < value if operator == "<" else actual <= value
    return _check(passed, actual=actual, operator=operator, threshold=value)


def _check(passed: bool, **details: Any) -> dict[str, Any]:
    details.pop("passed", None)
    return {"passed": bool(passed), **details}


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ValueError(f"config field {key!r} must be a mapping")
    return item


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != CONFIG_VERSION:
        raise ValueError("unsupported Phase M0 config")
    return payload


def _configured_dataset_episode_total(config: Mapping[str, Any]) -> int:
    return len(VISUAL_REQUIRED_TASKS) * int(
        _mapping(config, "dataset")["episodes_per_task"]
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be a mapping: {path}")
    return payload


def _root_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _camera_order(config: Mapping[str, Any]) -> tuple[str, ...]:
    dataset = _mapping(config, "dataset")
    camera = _mapping(config, "camera")
    dataset_order = tuple(str(value) for value in dataset.get("camera_order", ()))
    camera_order = tuple(str(value) for value in camera.get("camera_order", ()))
    if dataset_order != CAMERA_ORDER or camera_order != CAMERA_ORDER:
        raise ValueError(f"M0-v2 canonical camera order must be {list(CAMERA_ORDER)!r}")
    if (
        camera.get("renderer_backend") != "mujoco.Renderer"
        or camera.get("geometry_source") != "mujoco_xml"
        or camera.get("raw_unannotated") is not True
        or camera.get("calibration_convention") != "opencv_optical_camera_pose_in_world"
    ):
        raise ValueError("M0-v2 camera provenance config is not canonical")
    return dataset_order


def _repository_file(value: str) -> Path:
    relative = Path(value)
    if not value or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"repository provenance path is unsafe: {value!r}")
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"repository provenance path escapes root: {value!r}") from exc
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return path


def _resolve_inside(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"artifact escapes dataset root: {value!r}") from exc
    return path


def _require_new_file(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite stale report: {path}")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    if temporary.exists():
        raise FileExistsError(f"stale partial report exists: {temporary}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
