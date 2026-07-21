"""Collect the MuJoCo-backed, three-camera Phase M0-v2 dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

# Headless MuJoCo rendering is part of the formal collection protocol.
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.exporters import ExportObserver, HDF5TrajectoryExporter  # noqa: E402
from data.trajectory import schema_profile  # noqa: E402
from envs.runtime import RenderRequest, RunnerConfig, SimulationRunner  # noqa: E402
from envs.visual_required_env import (  # noqa: E402
    VISUAL_REQUIRED_TASKS,
    VisualRequiredEnv,
    VisualRequiredEnvConfig,
)
from policies.visual_required import PrivilegedScriptedOraclePolicy  # noqa: E402


FORMAT_VERSION = "wam.multimodal.m0.dataset/2"
CONFIG_VERSION = "wam.multimodal.m0/2"
SCHEMA_VERSION = "wam.multimodal/1.1"
BEHAVIOR_ID = "privileged_scripted_oracle_mujoco_v2"
MIN_FORMAL_DATASET_EPISODES = 2_000
CAMERA_ORDER = ("fixed", "robot_0_camera", "robot_1_camera")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/wam_multimodal/m0_data.yaml",
    )
    parser.add_argument("--out-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = _load_config(config_path)
    canonical_config = (ROOT / "configs/wam_multimodal/m0_data.yaml").resolve()
    canonical_output = _root_path(config["dataset"]["directory"]).resolve()
    output = (args.out_dir or canonical_output).resolve()
    formal_protocol = bool(
        config_path == canonical_config
        and output == canonical_output
        and _configured_dataset_episode_total(config) >= MIN_FORMAL_DATASET_EPISODES
    )
    collect_dataset(
        config,
        config_path=config_path,
        output=output,
        formal_protocol=formal_protocol,
    )
    return 0


def collect_dataset(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    output: Path,
    formal_protocol: bool,
) -> dict[str, Any]:
    """Collect the configured number of paired episodes; counts are not hard-coded."""

    dataset_cfg = _mapping(config, "dataset")
    camera_cfg = _mapping(config, "camera")
    audit_cfg = _mapping(config, "audit")
    camera_order = _camera_order(config)
    tasks = tuple(str(task) for task in VISUAL_REQUIRED_TASKS)
    if len(tasks) != 3 or len(set(tasks)) != len(tasks):
        raise ValueError("Phase M0 requires exactly three unique visual tasks")
    cue_variants = tuple(int(value) for value in dataset_cfg["cue_variants"])
    if cue_variants != (0, 1):
        raise ValueError("the canonical paired-cue MuJoCo tasks require [0,1]")
    episodes_per_task = int(dataset_cfg["episodes_per_task"])
    if episodes_per_task <= 0 or episodes_per_task % len(cue_variants):
        raise ValueError("episodes_per_task must be positive and cue-pair divisible")
    split_plan = _split_plan(dataset_cfg, tasks, cue_variants)
    if len(split_plan) != episodes_per_task * len(tasks):
        raise ValueError("split plan does not match episodes_per_task")
    formal_protocol = bool(
        formal_protocol and len(split_plan) >= MIN_FORMAL_DATASET_EPISODES
    )
    if dataset_cfg.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("M0-v2 requires schema wam.multimodal/1.1")
    if output.exists():
        raise FileExistsError(f"refusing to mix with an existing dataset: {output}")
    environment_provenance = _environment_preflight(
        config,
        task_id=tasks[0],
        template_id=str(split_plan[0]["template_id"]),
        camera_order=camera_order,
    )
    source_sha256 = _source_sha256(audit_cfg)
    output.mkdir(parents=True)
    (output / "hdf5").mkdir()
    (output / "videos").mkdir()

    schema = schema_profile("wam_multimodal", cameras=camera_order)
    if schema.version != SCHEMA_VERSION:
        raise ValueError("multimodal schema implementation/version mismatch")
    _validate_schema_camera_sources(schema, camera_order)
    exporter = HDF5TrajectoryExporter(output / "hdf5", schema, stream_videos=False)
    export_observer = ExportObserver((exporter,), fps=float(camera_cfg["control_hz"]))
    entries: list[dict[str, Any]] = []
    try:
        for episode_index, planned in enumerate(split_plan):
            entry = _collect_episode(
                config,
                output=output,
                episode_index=episode_index,
                planned=planned,
                export_observer=export_observer,
                camera_order=camera_order,
                environment_provenance=environment_provenance,
            )
            entries.append(entry)
            print(
                f"[{episode_index + 1}/{len(split_plan)}] "
                f"{entry['task_id']} {entry['split']} seed={entry['seed']} "
                f"cue={entry['cue_id']} success={entry['success']}",
                file=sys.stderr,
            )
    finally:
        export_observer.close()

    dataset_card = ROOT / str(dataset_cfg["dataset_card"])
    manifest = {
        "format_version": FORMAT_VERSION,
        "phase": "M0",
        "formal_protocol": bool(formal_protocol),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "dataset_card": str(dataset_card.resolve()),
        "dataset_card_sha256": _sha256(dataset_card),
        "schema_profile": "wam_multimodal",
        "schema_version": SCHEMA_VERSION,
        "camera_order": list(camera_order),
        "raw_unannotated": True,
        "control_hz": float(camera_cfg["control_hz"]),
        "image_hz": float(camera_cfg["image_hz"]),
        "resolution": [int(camera_cfg["height"]), int(camera_cfg["width"])],
        "tasks": list(tasks),
        "cue_variants": list(cue_variants),
        "episodes_per_task": episodes_per_task,
        "episodes": entries,
        "split_counts": _split_counts(entries),
        "outcome_counts": {
            "success": sum(bool(entry["success"]) for entry in entries),
            "failure": sum(not bool(entry["success"]) for entry in entries),
        },
        "provenance": {
            **environment_provenance,
            "source_sha256": source_sha256,
        },
    }
    manifest_path = output / "manifest.json"
    _atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "output": str(output),
                "manifest": str(manifest_path),
                "episodes": len(entries),
                "successes": manifest["outcome_counts"]["success"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return manifest


def _collect_episode(
    config: Mapping[str, Any],
    *,
    output: Path,
    episode_index: int,
    planned: Mapping[str, Any],
    export_observer: ExportObserver,
    camera_order: Sequence[str],
    environment_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    camera_cfg = _mapping(config, "camera")
    collection_cfg = _mapping(config, "collection")
    task_id = str(planned["task_id"])
    physical_seed = int(planned["physical_seed"])
    cue_id = int(planned["cue_id"])
    episode_seed = physical_seed * 2 + cue_id
    control_dt = 1.0 / float(camera_cfg["control_hz"])
    env_cfg = VisualRequiredEnvConfig(
        task_id=task_id,
        control_dt=control_dt,
        episode_len=int(collection_cfg["max_steps"]),
        image_width=int(camera_cfg["width"]),
        image_height=int(camera_cfg["height"]),
        render_cue_mode="truth",
        randomization_template_id=str(planned["template_id"]),
    )
    env = VisualRequiredEnv(env_cfg)
    if tuple(str(value) for value in getattr(env, "camera_names", ())) != tuple(
        camera_order
    ):
        raise ValueError("environment camera_names differ from canonical camera_order")
    video_directory = output / "videos" / f"episode_{episode_index:06d}"
    videos = {
        camera: _CapturedFrameVideoObserver(
            video_directory / f"{camera}.mp4",
            stream=camera,
            fps=float(camera_cfg["image_hz"]),
            codec=str(camera_cfg["codec"]),
        )
        for camera in camera_order
    }
    signal = _VisualSignalEvidenceObserver(
        task_id=task_id,
        camera_order=camera_order,
        minimum_changed_pixels=int(
            _mapping(config, "audit")["minimum_visual_signal_changed_pixels"]
        ),
    )
    try:
        task_condition = dict(env.task_condition)
        if str(task_condition.get("id")) != task_id:
            raise ValueError("environment task condition/id mismatch")
        _, identity_info = env.reset(
            seed=episode_seed,
            randomize=bool(collection_cfg["randomize"]),
        )
        _validate_episode_identity(
            identity_info,
            physical_seed=physical_seed,
            cue_id=cue_id,
            template_id=str(planned["template_id"]),
        )
        scene_id = str(identity_info["scene_id"])
        object_combination_id = str(identity_info["object_combination_id"])
        calibrations = {
            camera: _camera_calibration(
                env,
                camera=camera,
                width=int(camera_cfg["width"]),
                height=int(camera_cfg["height"]),
            )
            for camera in camera_order
        }
        environment_payload = {
            **asdict(env_cfg),
            "camera_order": list(camera_order),
            "geometry_source": environment_provenance["geometry_source"],
            "renderer_backend": environment_provenance["renderer_backend"],
            "mujoco_version": environment_provenance["mujoco_version"],
            "mujoco_gl": environment_provenance["mujoco_gl"],
            "model_xml_path": environment_provenance["model_xml_path"],
            "model_xml_sha256": environment_provenance["model_xml_sha256"],
            "raw_unannotated": True,
        }
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "seed": episode_seed,
            "task_id": task_id,
            "behavior_id": BEHAVIOR_ID,
            # Kept in episode metadata as a provenance snapshot.  The v1.1
            # trajectory fields themselves resolve current/next calibration
            # from each SimulationTransition, including dynamic ego extrinsics.
            "camera_intrinsics": {
                camera: calibrations[camera]["intrinsics"] for camera in camera_order
            },
            "camera_extrinsics": {
                camera: calibrations[camera]["extrinsics"] for camera in camera_order
            },
            "camera_resolution": {
                camera: calibrations[camera]["resolution"] for camera in camera_order
            },
            "environment_config": json.dumps(environment_payload, sort_keys=True),
            "randomization_config": json.dumps(
                {
                    "enabled": bool(collection_cfg["randomize"]),
                    "physical_seed": physical_seed,
                    "episode_seed": episode_seed,
                    "cue_id": cue_id,
                    "cue_variant": cue_id,
                    "split": str(planned["split"]),
                    "template_id": str(planned["template_id"]),
                    "scene_id": scene_id,
                    "object_combination_id": object_combination_id,
                },
                sort_keys=True,
            ),
        }
        runner = SimulationRunner(
            env,
            PrivilegedScriptedOraclePolicy(env),
            RunnerConfig(
                max_steps=int(collection_cfg["max_steps"]),
                render=tuple(
                    RenderRequest(
                        camera,
                        camera,
                        width=int(camera_cfg["width"]),
                        height=int(camera_cfg["height"]),
                        fps=float(camera_cfg["image_hz"]),
                        annotator=None,
                    )
                    for camera in camera_order
                ),
                expose_privileged_state_to_policy=False,
                policy_observation_keys=("proprioception",),
                expose_rendered_images_to_policy=False,
                policy_image_streams=(),
                expose_task_to_policy=True,
                task_id=task_id,
                task=str(task_condition["text"]),
                policy_action_history=int(collection_cfg["action_history"]),
            ),
        )
        summary = runner.run_episode(
            seed=episode_seed,
            episode_index=episode_index,
            randomize=bool(collection_cfg["randomize"]),
            observers=(export_observer, *videos.values(), signal),
            metadata=metadata,
        )
    finally:
        for video in videos.values():
            video.close()
        env.close()

    hdf5_path = output / "hdf5" / f"episode_{episode_index:06d}.hdf5"
    if not hdf5_path.is_file() or any(
        not video.path.is_file() for video in videos.values()
    ):
        raise RuntimeError("episode artifacts were not finalized")
    info = summary.final_info
    if int(info.get("physical_seed", -1)) != physical_seed:
        raise ValueError("environment physical seed does not match collection plan")
    if int(info.get("cue_variant", -1)) != cue_id:
        raise ValueError("environment cue does not match collection plan")
    _validate_episode_identity(
        info,
        physical_seed=physical_seed,
        cue_id=cue_id,
        template_id=str(planned["template_id"]),
        scene_id=scene_id,
        object_combination_id=object_combination_id,
    )
    video_entries = {
        camera: {
            "path": str(videos[camera].path.relative_to(output)),
            "sha256": _sha256(videos[camera].path),
            "captured_frames": videos[camera].frames_written,
            "width": int(camera_cfg["width"]),
            "height": int(camera_cfg["height"]),
            "fps": float(camera_cfg["image_hz"]),
            "codec": str(camera_cfg["codec"]),
        }
        for camera in camera_order
    }
    return {
        "episode_index": episode_index,
        "seed": episode_seed,
        "physical_seed": physical_seed,
        "cue_id": cue_id,
        "task_id": task_id,
        "task_text": str(task_condition["text"]),
        "split": str(planned["split"]),
        "template_id": str(planned["template_id"]),
        "scene_id": scene_id,
        "object_combination_id": object_combination_id,
        "behavior_id": BEHAVIOR_ID,
        "steps": int(summary.steps),
        "success": bool(info.get("success", False)),
        "failure": bool(info.get("failure", False)),
        "failure_reason": str(info.get("failure_reason", "none")),
        "hdf5_path": str(hdf5_path.relative_to(output)),
        "hdf5_sha256": _sha256(hdf5_path),
        "videos": video_entries,
        "visual_signal": signal.evidence(),
        "renderer_backend": environment_provenance["renderer_backend"],
        "geometry_source": environment_provenance["geometry_source"],
        "mujoco_version": environment_provenance["mujoco_version"],
        "mujoco_gl": environment_provenance["mujoco_gl"],
        "model_xml_sha256": environment_provenance["model_xml_sha256"],
        "raw_unannotated": True,
    }


def _split_plan(
    dataset_cfg: Mapping[str, Any],
    tasks: Sequence[str],
    cues: Sequence[int],
) -> list[dict[str, Any]]:
    split_cfg = _mapping(dataset_cfg, "split")
    physical_seed = int(dataset_cfg["physical_seed_start"])
    planned: list[dict[str, Any]] = []
    per_task_total = 0
    for split in ("train", "validation", "test"):
        values = _mapping(split_cfg, split)
        physical_count = int(values["physical_seeds"])
        episode_count = int(values["episodes"])
        if physical_count <= 0 or episode_count != physical_count * len(cues):
            raise ValueError(f"split {split!r} must contain complete cue pairs")
        templates = tuple(str(value) for value in values["template_ids"])
        if not templates or any(not value for value in templates):
            raise ValueError(f"split {split!r} has no randomization templates")
        per_task_total += episode_count
    if per_task_total != int(dataset_cfg["episodes_per_task"]):
        raise ValueError("split episodes do not sum to episodes_per_task")

    for task in tasks:
        for split in ("train", "validation", "test"):
            values = _mapping(split_cfg, split)
            templates = tuple(str(value) for value in values["template_ids"])
            for local_seed in range(int(values["physical_seeds"])):
                template = templates[local_seed % len(templates)]
                for cue in cues:
                    planned.append(
                        {
                            "task_id": task,
                            "split": split,
                            "template_id": template,
                            "physical_seed": physical_seed,
                            "cue_id": int(cue),
                        }
                    )
                physical_seed += 1
    return planned


def _camera_calibration(
    env: Any, *, camera: str, width: int, height: int
) -> dict[str, Any]:
    method = getattr(env, "camera_calibration", None)
    if not callable(method):
        raise TypeError("M0-v2 requires environment.camera_calibration")
    raw = method(camera=camera, width=width, height=height)
    if not isinstance(raw, Mapping):
        raise TypeError("camera_calibration must return a mapping")
    required = {
        "intrinsics",
        "extrinsics",
        "resolution",
        "camera_id",
        "parent_body_id",
        "parent_body_name",
        "fovy_degrees",
        "convention",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise KeyError(f"camera {camera!r} calibration is missing {missing}")
    intrinsics = np.asarray(raw["intrinsics"], dtype=np.float32)
    extrinsics = np.asarray(raw["extrinsics"], dtype=np.float32)
    raw_resolution = np.asarray(raw["resolution"])
    if raw_resolution.dtype.kind not in {"i", "u"}:
        raise TypeError("camera resolution must contain integers")
    resolution = raw_resolution.astype(np.int64, copy=False)
    if (
        intrinsics.shape != (3, 3)
        or extrinsics.shape != (4, 4)
        or not np.isfinite(intrinsics).all()
        or not np.isfinite(extrinsics).all()
    ):
        raise ValueError("camera calibration has invalid/non-finite matrices")
    if resolution.shape != (2,) or not np.array_equal(
        resolution, np.asarray((height, width), dtype=np.int64)
    ):
        raise ValueError("camera calibration resolution mismatch")
    return {
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "resolution": resolution,
        "camera_id": int(raw["camera_id"]),
        "parent_body_id": int(raw["parent_body_id"]),
        "parent_body_name": str(raw["parent_body_name"]),
        "fovy_degrees": float(raw["fovy_degrees"]),
        "convention": str(raw["convention"]),
    }


def _environment_preflight(
    config: Mapping[str, Any],
    *,
    task_id: str,
    template_id: str,
    camera_order: Sequence[str],
) -> dict[str, Any]:
    camera_cfg = _mapping(config, "camera")
    collection_cfg = _mapping(config, "collection")
    env = VisualRequiredEnv(
        VisualRequiredEnvConfig(
            task_id=task_id,
            control_dt=1.0 / float(camera_cfg["control_hz"]),
            episode_len=int(collection_cfg["max_steps"]),
            image_width=int(camera_cfg["width"]),
            image_height=int(camera_cfg["height"]),
            render_cue_mode="truth",
            randomization_template_id=template_id,
        )
    )
    try:
        names = tuple(str(value) for value in getattr(env, "camera_names", ()))
        if names != tuple(camera_order):
            raise ValueError(f"MuJoCo camera order {names!r} is not canonical")
        renderer_backend = str(getattr(env, "renderer_backend", ""))
        if renderer_backend != "mujoco.Renderer":
            raise ValueError("M0-v2 only accepts renderer_backend='mujoco.Renderer'")
        mujoco_gl = str(os.environ.get("MUJOCO_GL", ""))
        if not mujoco_gl:
            raise ValueError("MUJOCO_GL must name the active rendering context")
        model_xml_path = Path(getattr(env, "model_xml_path", "")).resolve()
        if not model_xml_path.is_file():
            raise FileNotFoundError(model_xml_path)
        model_xml_sha256 = str(getattr(env, "model_xml_sha256", ""))
        if model_xml_sha256 != _sha256(model_xml_path):
            raise ValueError("environment model_xml_sha256 does not match XML bytes")
        _, info = env.reset(seed=0, randomize=False)
        if (
            str(info.get("geometry_source", "")) != "mujoco_xml"
            or str(info.get("renderer_backend", "")) != renderer_backend
        ):
            raise ValueError("environment info does not bind MuJoCo renderer/XML")
        calibrations = {
            camera: _camera_calibration(
                env,
                camera=camera,
                width=int(camera_cfg["width"]),
                height=int(camera_cfg["height"]),
            )
            for camera in camera_order
        }
        if calibrations["fixed"]["parent_body_name"] != "world":
            raise ValueError("fixed camera must be attached to the MuJoCo world")
        ego_parents = [
            calibrations[camera]["parent_body_name"] for camera in CAMERA_ORDER[1:]
        ]
        if any(parent == "world" or not parent for parent in ego_parents) or len(
            set(ego_parents)
        ) != len(ego_parents):
            raise ValueError("robot cameras must have distinct non-world parent bodies")
        for camera, calibration in calibrations.items():
            if calibration["convention"] != "opencv_optical_camera_pose_in_world":
                raise ValueError(f"{camera} calibration convention is not canonical")
            if not np.isfinite(calibration["fovy_degrees"]):
                raise ValueError(f"{camera} fovy is not finite")
        return {
            "renderer_backend": renderer_backend,
            "geometry_source": "mujoco_xml",
            "mujoco_version": str(mujoco.__version__),
            "mujoco_gl": mujoco_gl,
            "model_xml_path": _display_path(model_xml_path),
            "model_xml_sha256": model_xml_sha256,
            "camera_rig": {
                camera: {
                    "camera_id": calibrations[camera]["camera_id"],
                    "parent_body_id": calibrations[camera]["parent_body_id"],
                    "parent_body_name": calibrations[camera]["parent_body_name"],
                    "fovy_degrees": calibrations[camera]["fovy_degrees"],
                    "convention": calibrations[camera]["convention"],
                }
                for camera in camera_order
            },
        }
    finally:
        env.close()


def _validate_schema_camera_sources(schema: Any, cameras: Sequence[str]) -> None:
    actual = {str(field.name): str(field.source) for field in schema.fields}
    expected: dict[str, str] = {}
    for camera in cameras:
        for field in ("intrinsics", "extrinsics", "resolution"):
            plural = "resolutions" if field == "resolution" else field
            expected[f"camera.{field}.{camera}"] = f"camera_{plural}.{camera}"
            expected[f"next_camera.{field}.{camera}"] = f"next_camera_{plural}.{camera}"
    mismatches = {
        name: {"actual": actual.get(name), "expected": source}
        for name, source in expected.items()
        if actual.get(name) != source
    }
    if mismatches:
        raise ValueError(
            f"wam.multimodal/1.1 camera fields are not transition-bound: {mismatches}"
        )


def _source_sha256(audit_cfg: Mapping[str, Any]) -> dict[str, str]:
    raw = audit_cfg.get("required_source_files")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError("audit.required_source_files must be a non-empty sequence")
    result: dict[str, str] = {}
    for value in raw:
        relative = Path(str(value))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"source provenance path is not repository-relative: {value}"
            )
        path = (ROOT / relative).resolve()
        try:
            path.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError(
                f"source provenance path escapes repository: {value}"
            ) from exc
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        key = relative.as_posix()
        if key in result:
            raise ValueError(f"duplicate source provenance path: {key}")
        result[key] = _sha256(path)
    return result


def _validate_episode_identity(
    info: Mapping[str, Any],
    *,
    physical_seed: int,
    cue_id: int,
    template_id: str,
    scene_id: str | None = None,
    object_combination_id: str | None = None,
) -> None:
    actual_scene = str(info.get("scene_id", ""))
    actual_objects = str(info.get("object_combination_id", ""))
    passed = bool(
        int(info.get("physical_seed", -1)) == physical_seed
        and int(info.get("cue_variant", -1)) == cue_id
        and str(info.get("randomization_template_id", "")) == template_id
        and actual_scene
        and actual_objects
        and (scene_id is None or actual_scene == scene_id)
        and (object_combination_id is None or actual_objects == object_combination_id)
    )
    if not passed:
        raise ValueError("environment randomization identity does not match plan")


class _VisualSignalEvidenceObserver:
    """Capture raw, cross-camera evidence for the first visible cue frame."""

    def __init__(
        self,
        *,
        task_id: str,
        camera_order: Sequence[str],
        minimum_changed_pixels: int,
    ) -> None:
        if minimum_changed_pixels <= 0:
            raise ValueError("minimum_changed_pixels must be positive")
        self.task_id = str(task_id)
        self.camera_order = tuple(str(value) for value in camera_order)
        self.minimum_changed_pixels = int(minimum_changed_pixels)
        self._active = False
        self._active_observed = False
        self._onset_step: int | None = None
        self._kind: str | None = None
        self._visible_expected: dict[str, bool] = {}
        self._last_index = {camera: -1 for camera in self.camera_order}
        self._last_inactive: dict[str, tuple[int, float, np.ndarray]] = {}
        self._camera_evidence: dict[str, dict[str, Any]] = {}

    def on_episode_start(self, *, info: Mapping[str, Any], **_: Any) -> None:
        self._active = False
        self._active_observed = False
        self._onset_step = None
        self._kind = None
        self._visible_expected = {}
        self._last_index = {camera: -1 for camera in self.camera_order}
        self._last_inactive = {}
        self._camera_evidence = {}
        self._observe_info(info)

    def on_transition(self, transition: Any) -> None:
        previous_active = self._active
        self._observe_info(transition.info)
        for camera in self.camera_order:
            self._observe_frame(
                camera,
                frame_index=int(transition.image_frame_indices[camera]),
                timestamp=float(transition.image_timestamps[camera]),
                frame=transition.images[camera],
                active=previous_active,
            )
            self._observe_frame(
                camera,
                frame_index=int(transition.next_image_frame_indices[camera]),
                timestamp=float(transition.next_image_timestamps[camera]),
                frame=transition.next_images[camera],
                active=self._active,
            )

    def on_episode_end(self, summary: Any) -> None:
        if int(summary.steps) <= 0:
            raise ValueError("visual-signal evidence received an empty episode")

    def _observe_info(self, info: Mapping[str, Any]) -> None:
        required = (
            "renderer_backend",
            "geometry_source",
            "visual_signal_active",
            "visual_signal_onset_step",
            "visual_signal_kind",
            "cue_visible_expected",
        )
        missing = [name for name in required if name not in info]
        if missing:
            raise KeyError(f"visual signal info is missing {missing}")
        if (
            str(info["renderer_backend"]) != "mujoco.Renderer"
            or str(info["geometry_source"]) != "mujoco_xml"
        ):
            raise ValueError("visual signal info is not MuJoCo-backed")
        onset = int(info["visual_signal_onset_step"])
        kind = str(info["visual_signal_kind"])
        if onset < 0 or not kind:
            raise ValueError("visual signal onset/kind is invalid")
        if self._onset_step is not None and self._onset_step != onset:
            raise ValueError("visual signal onset changed within episode")
        if self._kind is not None and self._kind != kind:
            raise ValueError("visual signal kind changed within episode")
        self._onset_step = onset
        self._kind = kind
        visible = info["cue_visible_expected"]
        if isinstance(visible, Mapping):
            normalized = {
                camera: bool(visible.get(camera, False)) for camera in self.camera_order
            }
        elif type(visible) is bool:
            normalized = {camera: visible for camera in self.camera_order}
        else:
            raise TypeError("cue_visible_expected must be bool or a camera mapping")
        if self._visible_expected and self._visible_expected != normalized:
            raise ValueError("cue visibility expectation changed within episode")
        self._visible_expected = normalized
        self._active = bool(info["visual_signal_active"])
        self._active_observed = self._active_observed or self._active

    def _observe_frame(
        self,
        camera: str,
        *,
        frame_index: int,
        timestamp: float,
        frame: Any,
        active: bool,
    ) -> None:
        if frame_index == self._last_index[camera]:
            return
        if frame_index != self._last_index[camera] + 1:
            raise ValueError(f"{camera} signal evidence frame index is not contiguous")
        rgb = np.asarray(frame, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.any(rgb):
            raise ValueError(f"{camera} signal evidence frame is invalid")
        self._last_index[camera] = frame_index
        if not active:
            self._last_inactive[camera] = (frame_index, timestamp, rgb.copy())
            return
        if camera in self._camera_evidence:
            return
        previous = self._last_inactive.get(camera)
        evidence: dict[str, Any] = {
            "active_frame_index": frame_index,
            "active_timestamp": timestamp,
            "active_rgb_sha256": _array_sha256(rgb),
            "visible_expected": bool(self._visible_expected.get(camera, False)),
            "pre_frame_index": None,
            "pre_timestamp": None,
            "pre_rgb_sha256": None,
            "changed_pixels": None,
        }
        if previous is not None:
            previous_index, previous_timestamp, previous_rgb = previous
            changed = np.any(rgb != previous_rgb, axis=2)
            evidence.update(
                {
                    "pre_frame_index": previous_index,
                    "pre_timestamp": previous_timestamp,
                    "pre_rgb_sha256": _array_sha256(previous_rgb),
                    "changed_pixels": int(np.count_nonzero(changed)),
                }
            )
        self._camera_evidence[camera] = evidence

    def evidence(self) -> dict[str, Any]:
        if self._onset_step is None or self._kind is None:
            raise RuntimeError("visual signal evidence is incomplete")
        missing = sorted(set(self.camera_order) - set(self._camera_evidence))
        if missing:
            raise RuntimeError(f"cameras never captured an active cue: {missing}")
        if not self._active_observed or not all(self._visible_expected.values()):
            raise RuntimeError(
                "visual signal was not expected/observed in every camera"
            )
        if self.task_id == "visual_event_stop":
            invalid = [
                camera
                for camera, values in self._camera_evidence.items()
                if values["pre_frame_index"] is None
                or values["changed_pixels"] is None
                or int(values["changed_pixels"]) < self.minimum_changed_pixels
            ]
            if invalid:
                raise RuntimeError(
                    f"event cue onset lacks raw pixel evidence in cameras: {invalid}"
                )
        return {
            "active_observed": self._active_observed,
            "onset_step": self._onset_step,
            "kind": self._kind,
            "visibility_expected": dict(self._visible_expected),
            "cameras": dict(self._camera_evidence),
        }


class _CapturedFrameVideoObserver:
    """Encode only new camera captures, not expected 20/10 Hz row reuse."""

    def __init__(
        self,
        path: Path,
        *,
        stream: str,
        fps: float,
        codec: str,
    ) -> None:
        if fps <= 0.0 or len(codec) != 4:
            raise ValueError("invalid captured-frame video configuration")
        self.path = path
        self.stream = stream
        self.fps = float(fps)
        self.codec = codec
        self.frames_written = 0
        self._last_frame_index: int | None = None
        self._writer: cv2.VideoWriter | None = None
        self._shape: tuple[int, int, int] | None = None

    def on_episode_start(self, **_: Any) -> None:
        self.frames_written = 0
        self._last_frame_index = None
        self._writer = None
        self._shape = None

    def on_transition(self, transition: Any) -> None:
        self._write_frame(
            int(transition.image_frame_indices[self.stream]),
            transition.images[self.stream],
        )
        self._write_frame(
            int(transition.next_image_frame_indices[self.stream]),
            transition.next_images[self.stream],
        )

    def _write_frame(self, frame_index: int, value: Any) -> None:
        if self._last_frame_index == frame_index:
            return
        if (
            self._last_frame_index is not None
            and frame_index != self._last_frame_index + 1
        ):
            raise ValueError("captured video frame index is not contiguous")
        frame = np.asarray(value, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3 or not np.any(frame):
            raise ValueError("captured video received an empty/invalid RGB frame")
        if self._writer is None:
            self._open(frame)
        if tuple(frame.shape) != self._shape:
            raise ValueError("captured video resolution changed within an episode")
        assert self._writer is not None
        self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.frames_written += 1
        self._last_frame_index = frame_index

    def on_episode_end(self, summary: Any) -> None:
        if int(summary.steps) <= 0 or self.frames_written <= 0:
            raise ValueError("captured video has no episode frames")
        self.close()

    def _open(self, frame: np.ndarray) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*self.codec),
            self.fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"failed to open video writer: {self.path}")
        self._writer = writer
        self._shape = tuple(frame.shape)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def _split_counts(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        values = [entry for entry in entries if entry["split"] == split]
        result[split] = {
            "episodes": len(values),
            "physical_seeds": len({int(entry["physical_seed"]) for entry in values}),
            "tasks": sorted({str(entry["task_id"]) for entry in values}),
            "cue_counts": {
                str(cue): sum(int(entry["cue_id"]) == cue for entry in values)
                for cue in (0, 1)
            },
            "template_ids": sorted({str(entry["template_id"]) for entry in values}),
        }
    return result


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != CONFIG_VERSION:
        raise ValueError("unsupported Phase M0 config")
    return payload


def _configured_dataset_episode_total(config: Mapping[str, Any]) -> int:
    return len(VISUAL_REQUIRED_TASKS) * int(
        _mapping(config, "dataset")["episodes_per_task"]
    )


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ValueError(f"config field {key!r} must be a mapping")
    return item


def _camera_order(config: Mapping[str, Any]) -> tuple[str, ...]:
    dataset = _mapping(config, "dataset")
    camera = _mapping(config, "camera")
    first = tuple(str(value) for value in dataset.get("camera_order", ()))
    second = tuple(str(value) for value in camera.get("camera_order", ()))
    if first != CAMERA_ORDER or second != CAMERA_ORDER:
        raise ValueError(f"M0-v2 canonical camera order must be {list(CAMERA_ORDER)!r}")
    if (
        camera.get("renderer_backend") != "mujoco.Renderer"
        or camera.get("geometry_source") != "mujoco_xml"
        or camera.get("raw_unannotated") is not True
        or camera.get("calibration_convention") != "opencv_optical_camera_pose_in_world"
    ):
        raise ValueError("M0-v2 camera provenance config is not canonical")
    return first


def _root_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    if temporary.exists():
        raise FileExistsError(f"stale partial manifest exists: {temporary}")
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
