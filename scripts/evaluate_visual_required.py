"""Run the paired Phase M0 state/RGB/oracle visual-required benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

# Headless MuJoCo rendering is part of the formal benchmark protocol.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.runtime import RenderRequest, RunnerConfig, SimulationRunner  # noqa: E402
from envs.visual_required_env import (  # noqa: E402
    VISUAL_REQUIRED_TASKS,
    VisualRequiredEnv,
    VisualRequiredEnvConfig,
)
from eval.visual_required import (  # noqa: E402
    CAMERA_ORDER,
    FORMAT_VERSION,
    REQUIRED_POLICIES,
    SCRIPTED_ORACLE_POLICY,
    SHUFFLED_VISION_POLICY,
    STATE_ONLY_POLICY,
    VISION_ORACLE_POLICY,
    VisualRequiredEpisode,
    contains_privileged_path,
    mapping_key,
    observation_leaf_paths,
    visual_required_acceptance,
)
from policies.visual_required import (  # noqa: E402
    PrivilegedScriptedOraclePolicy,
    StateOnlyPolicy,
    VisionOraclePolicy,
)


MIN_FORMAL_DATASET_EPISODES = 2_000
CONFIG_VERSION = "wam.multimodal.m0/2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/wam_multimodal/m0_data.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--physical-seeds-per-task",
        type=int,
        help="Diagnostic override; requires a non-canonical output directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = _load_config(config_path)
    canonical_config = (ROOT / "configs/wam_multimodal/m0_data.yaml").resolve()
    canonical_output = _root_path(config["benchmark"]["output_directory"]).resolve()
    output = (args.output_dir or canonical_output).resolve()
    configured_seed_count = int(config["benchmark"]["physical_seeds_per_task"])
    seed_count = args.physical_seeds_per_task or configured_seed_count
    if seed_count <= 0:
        raise ValueError("physical_seeds_per_task must be positive")
    if args.physical_seeds_per_task is not None and output == canonical_output:
        raise ValueError("diagnostic seed override requires --output-dir")
    formal_protocol = bool(
        config_path == canonical_config
        and output == canonical_output
        and seed_count == configured_seed_count
        and _configured_dataset_episode_total(config) >= MIN_FORMAL_DATASET_EPISODES
    )
    report = evaluate_benchmark(
        config,
        config_path=config_path,
        output=output,
        physical_seeds_per_task=seed_count,
        formal_protocol=formal_protocol,
    )
    if not report["passed"]:
        return 1
    return 0 if formal_protocol else 2


def evaluate_benchmark(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    output: Path,
    physical_seeds_per_task: int,
    formal_protocol: bool,
) -> dict[str, Any]:
    """Run all four policies on exact task/seed/cue pairs."""

    benchmark_cfg = _mapping(config, "benchmark")
    camera_cfg = _mapping(config, "camera")
    camera_order = _camera_order(config)
    formal_protocol = bool(
        formal_protocol
        and _configured_dataset_episode_total(config) >= MIN_FORMAL_DATASET_EPISODES
    )
    tasks = tuple(str(task) for task in VISUAL_REQUIRED_TASKS)
    configured_policies = tuple(str(value) for value in benchmark_cfg["policies"])
    if configured_policies != REQUIRED_POLICIES:
        raise ValueError(
            "benchmark policies differ from the canonical four-policy order"
        )
    cue_variants = tuple(int(value) for value in benchmark_cfg["cue_variants"])
    if cue_variants != (0, 1):
        raise ValueError("opposite-cue benchmark requires [0,1]")
    if str(benchmark_cfg["shuffle_mode"]) != "opposite_cue_derangement":
        raise ValueError("only opposite_cue_derangement is accepted")
    if benchmark_cfg.get("require_identical_seed_pairs") is not True:
        raise ValueError("visual benchmark requires identical physical seed pairs")
    if (
        tuple(str(value) for value in benchmark_cfg.get("camera_order", ()))
        != camera_order
        or benchmark_cfg.get("policy_rgb_stream") != "fixed"
        or benchmark_cfg.get("record_all_view_evidence") is not True
    ):
        raise ValueError(
            "benchmark must render all canonical views and bind policy RGB"
        )
    environment_provenance = _environment_preflight(
        config, task_id=tasks[0], camera_order=camera_order
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite stale benchmark output: {output}")
    output.mkdir(parents=True)

    physical_seed_start = int(benchmark_cfg["physical_seed_start"])
    physical_seeds = tuple(
        range(physical_seed_start, physical_seed_start + physical_seeds_per_task)
    )
    records: list[VisualRequiredEpisode] = []
    total = (
        len(tasks) * len(REQUIRED_POLICIES) * len(physical_seeds) * len(cue_variants)
    )
    completed = 0
    for task_id in tasks:
        for policy_name in REQUIRED_POLICIES:
            for physical_seed in physical_seeds:
                for cue_id in cue_variants:
                    record = _run_episode(
                        task_id,
                        policy_name,
                        physical_seed=physical_seed,
                        cue_id=cue_id,
                        benchmark_cfg=benchmark_cfg,
                        camera_cfg=camera_cfg,
                        camera_order=camera_order,
                        environment_provenance=environment_provenance,
                    )
                    records.append(record)
                    completed += 1
                    if (
                        completed == total
                        or completed % max(1, len(cue_variants) * 10) == 0
                    ):
                        print(
                            f"visual benchmark {completed}/{total}: "
                            f"{task_id}/{policy_name}",
                            file=sys.stderr,
                        )

    acceptance = visual_required_acceptance(
        records,
        tasks=tasks,
        physical_seeds=physical_seeds,
        cue_variants=cue_variants,
        thresholds=_mapping(config, "acceptance"),
    )
    records_path = output / "visual_required_episodes.jsonl"
    with records_path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.as_dict(), sort_keys=True) + "\n")
    report = {
        "format_version": FORMAT_VERSION,
        "gate": "M0-visual-required",
        "formal_protocol": bool(formal_protocol),
        "passed": bool(acceptance["passed"]),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "output_directory": str(output),
        "protocol": {
            "physical_seeds_per_task": physical_seeds_per_task,
            "physical_seeds": list(physical_seeds),
            "cue_variants": list(cue_variants),
            "tasks": list(tasks),
            "policies": list(REQUIRED_POLICIES),
            "same_seed_cue_pairs_for_all_policies": True,
            "shuffle_mode": "opposite_cue_derangement",
            "rgb_intervention": (
                "same MuJoCo physical seed/truth; renderer displays the opposite cue"
            ),
            "control_hz": float(camera_cfg["control_hz"]),
            "image_hz": float(camera_cfg["image_hz"]),
            "resolution": [int(camera_cfg["height"]), int(camera_cfg["width"])],
            "camera_order": list(camera_order),
            "render_requests": list(camera_order),
            "policy_rgb_stream": "fixed",
            "record_all_view_evidence": True,
            "raw_unannotated": True,
            **environment_provenance,
        },
        "episode_records": str(records_path),
        "episode_records_sha256": _sha256(records_path),
        "acceptance": acceptance,
    }
    report_path = output / "visual_required_benchmark.json"
    _atomic_write_json(report_path, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": report["passed"],
                "formal_protocol": report["formal_protocol"],
                "by_task": acceptance["by_task"],
                "macro": acceptance["macro"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return report


def _run_episode(
    task_id: str,
    policy_name: str,
    *,
    physical_seed: int,
    cue_id: int,
    benchmark_cfg: Mapping[str, Any],
    camera_cfg: Mapping[str, Any],
    camera_order: tuple[str, ...],
    environment_provenance: Mapping[str, Any],
) -> VisualRequiredEpisode:
    opposite = policy_name == SHUFFLED_VISION_POLICY
    render_mode = "opposite" if opposite else "truth"
    control_dt = 1.0 / float(camera_cfg["control_hz"])
    env = VisualRequiredEnv(
        VisualRequiredEnvConfig(
            task_id=task_id,
            control_dt=control_dt,
            episode_len=int(benchmark_cfg["max_steps"]),
            image_width=int(camera_cfg["width"]),
            image_height=int(camera_cfg["height"]),
            render_cue_mode=render_mode,
        )
    )
    try:
        if (
            tuple(str(value) for value in getattr(env, "camera_names", ()))
            != camera_order
            or str(getattr(env, "renderer_backend", "")) != "mujoco.Renderer"
            or str(getattr(env, "model_xml_sha256", ""))
            != str(environment_provenance["model_xml_sha256"])
        ):
            raise ValueError("benchmark environment provenance changed after preflight")
        base_policy = _policy(policy_name, env)
        policy = _PolicyAudit(base_policy)
        view_evidence = _AllViewEvidenceObserver(
            task_id=task_id, camera_order=camera_order
        )
        task_condition = dict(env.task_condition)
        episode_seed = int(physical_seed) * 2 + int(cue_id)
        initial_observation, preview_info = env.reset(
            seed=episode_seed,
            randomize=bool(benchmark_cfg["randomize"]),
        )
        initial_state = np.asarray(
            initial_observation["proprioception"], dtype=np.float32
        )
        initial_proprioception_sha256 = _array_sha256(initial_state)
        task_condition_sha256 = _mapping_sha256(task_condition)
        scene_id = str(preview_info.get("scene_id", ""))
        object_combination_id = str(preview_info.get("object_combination_id", ""))
        if (
            int(preview_info.get("physical_seed", -1)) != physical_seed
            or int(preview_info.get("cue_variant", -1)) != cue_id
            or not scene_id
            or not object_combination_id
        ):
            raise ValueError("visual benchmark preview identity mismatch")
        exposes_rgb = policy_name in {VISION_ORACLE_POLICY, SHUFFLED_VISION_POLICY}
        runner = SimulationRunner(
            env,
            policy,
            RunnerConfig(
                max_steps=int(benchmark_cfg["max_steps"]),
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
                expose_rendered_images_to_policy=exposes_rgb,
                policy_image_streams=("fixed",) if exposes_rgb else (),
                expose_task_to_policy=True,
                task_id=task_id,
                task=str(task_condition["text"]),
                policy_action_history=int(benchmark_cfg["action_history"]),
            ),
        )
        summary = runner.run_episode(
            seed=episode_seed,
            randomize=bool(benchmark_cfg["randomize"]),
            observers=(view_evidence,),
        )
        info = summary.final_info
        if int(info.get("physical_seed", -1)) != physical_seed:
            raise ValueError("visual benchmark physical seed mismatch")
        if int(info.get("cue_variant", -1)) != cue_id:
            raise ValueError("visual benchmark cue mismatch")
        rendered_cue = int(info.get("rendered_cue_variant", -1))
        expected_rendered = 1 - cue_id if opposite else cue_id
        if rendered_cue != expected_rendered:
            raise ValueError("RGB intervention did not render the expected cue")
        if (
            str(info.get("scene_id", "")) != scene_id
            or str(info.get("object_combination_id", "")) != object_combination_id
        ):
            raise ValueError("visual intervention changed the physical world identity")
        success = bool(info.get("success", False))
        failure_reason = str(info.get("failure_reason", "none"))
        key = mapping_key(task_id, physical_seed, cue_id) if opposite else None
        evidence = view_evidence.evidence()
        return VisualRequiredEpisode(
            task_id=task_id,
            cue_id=cue_id,
            physical_seed=physical_seed,
            policy=policy_name,
            success=success,
            failure=not success,
            failure_reason=failure_reason,
            steps=int(summary.steps),
            total_reward=float(summary.total_reward),
            presented_observation_paths=tuple(sorted(policy.presented_paths)),
            consumed_observation_paths=tuple(sorted(policy.consumed_paths)),
            privileged_observation_seen=bool(policy.privileged_seen),
            rgb_source_cue_id=expected_rendered
            if policy_name
            in {
                VISION_ORACLE_POLICY,
                SHUFFLED_VISION_POLICY,
            }
            else None,
            rgb_mapping_key=key,
            action_source=policy.action_source,
            initial_proprioception_sha256=initial_proprioception_sha256,
            task_condition_sha256=task_condition_sha256,
            scene_id=scene_id,
            object_combination_id=object_combination_id,
            camera_order=tuple(camera_order),
            all_view_frame_counts=evidence["frame_counts"],
            all_view_first_rgb_sha256=evidence["first_rgb_sha256"],
            all_view_last_rgb_sha256=evidence["last_rgb_sha256"],
            active_rgb_sha256=evidence["active_rgb_sha256"],
            pre_signal_frame_counts=evidence["pre_signal_frame_counts"],
            pre_signal_sequence_sha256=evidence["pre_signal_sequence_sha256"],
            camera_translation_travel_m=evidence["camera_translation_travel_m"],
            fixed_extrinsics_max_abs_delta=evidence["fixed_extrinsics_max_abs_delta"],
            cross_camera_sync=evidence["cross_camera_sync"],
            renderer_backend=str(environment_provenance["renderer_backend"]),
            geometry_source=str(environment_provenance["geometry_source"]),
            mujoco_version=str(environment_provenance["mujoco_version"]),
            mujoco_gl=str(environment_provenance["mujoco_gl"]),
            model_xml_sha256=str(environment_provenance["model_xml_sha256"]),
            raw_unannotated=True,
            cue_visible_expected=evidence["cue_visible_expected"],
            visual_signal_active_observed=evidence["visual_signal_active_observed"],
            visual_signal_onset_step=evidence["visual_signal_onset_step"],
            visual_signal_kind=evidence["visual_signal_kind"],
            policy_rgb_stream="fixed" if exposes_rgb else None,
        )
    finally:
        env.close()


def _policy(name: str, env: Any) -> Any:
    if name == STATE_ONLY_POLICY:
        return StateOnlyPolicy(blind_cue_variant=0)
    if name == SCRIPTED_ORACLE_POLICY:
        return PrivilegedScriptedOraclePolicy(env)
    if name in {VISION_ORACLE_POLICY, SHUFFLED_VISION_POLICY}:
        return VisionOraclePolicy()
    raise ValueError(f"unsupported visual policy {name!r}")


class _PolicyAudit:
    """Record presented and policy-declared consumed observation leaves."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.presented_paths: set[str] = set()
        self.consumed_paths: set[str] = set()
        self.privileged_seen = False
        self.action_source: str | None = None

    def reset(self) -> None:
        self.presented_paths.clear()
        self.consumed_paths.clear()
        self.privileged_seen = False
        self.action_source = None
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        presented = observation_leaf_paths(observation)
        self.presented_paths.update(presented)
        self.privileged_seen = self.privileged_seen or contains_privileged_path(
            presented
        )
        action = np.asarray(self.policy.act(observation), dtype=np.float32)
        diagnostics = dict(getattr(self.policy, "last_diagnostics", {}) or {})
        consumed = tuple(
            str(value) for value in diagnostics.get("consumed_observation_paths", ())
        )
        self.consumed_paths.update(consumed)
        self.privileged_seen = bool(
            self.privileged_seen
            or diagnostics.get("privileged_state_seen", False)
            or contains_privileged_path(consumed)
        )
        source = diagnostics.get("action_source")
        if source is not None:
            self.action_source = str(source)
        return action

    @property
    def last_diagnostics(self) -> Mapping[str, Any]:
        return dict(getattr(self.policy, "last_diagnostics", {}) or {})


class _AllViewEvidenceObserver:
    """Record raw three-view timing, cue, and camera-motion evidence."""

    def __init__(self, *, task_id: str, camera_order: Sequence[str]) -> None:
        self.task_id = str(task_id)
        self.camera_order = tuple(str(value) for value in camera_order)
        self._reset_state()

    def _reset_state(self) -> None:
        self._last_indices = {camera: -1 for camera in self.camera_order}
        self._frames: dict[str, dict[int, tuple[float, str]]] = {
            camera: {} for camera in self.camera_order
        }
        self._first_hash: dict[str, str] = {}
        self._last_hash: dict[str, str] = {}
        self._active_hash: dict[str, str] = {}
        self._pre_signal: dict[str, list[tuple[int, float, str]]] = {
            camera: [] for camera in self.camera_order
        }
        self._first_pose: dict[str, np.ndarray] = {}
        self._translation_travel = {camera: 0.0 for camera in self.camera_order}
        self._pose_max_abs_delta = {camera: 0.0 for camera in self.camera_order}
        self._cross_camera_sync = True
        self._active = False
        self._active_observed = False
        self._onset_step: int | None = None
        self._kind: str | None = None
        self._visible: dict[str, bool] = {}

    def on_episode_start(self, *, info: Mapping[str, Any], **_: Any) -> None:
        self._reset_state()
        self._observe_info(info)

    def on_transition(self, transition: Any) -> None:
        mapping_names = (
            "images",
            "next_images",
            "image_timestamps",
            "next_image_timestamps",
            "image_state_timestamps",
            "next_image_state_timestamps",
            "image_frame_indices",
            "next_image_frame_indices",
            "camera_intrinsics",
            "next_camera_intrinsics",
            "camera_extrinsics",
            "next_camera_extrinsics",
            "camera_resolutions",
            "next_camera_resolutions",
        )
        for name in mapping_names:
            value = getattr(transition, name)
            if not isinstance(value, Mapping) or tuple(value) != self.camera_order:
                raise ValueError(
                    f"benchmark transition {name} is not canonical three-view"
                )
        reference = self.camera_order[0]
        for camera in self.camera_order[1:]:
            self._cross_camera_sync = bool(
                self._cross_camera_sync
                and int(transition.image_frame_indices[camera])
                == int(transition.image_frame_indices[reference])
                and int(transition.next_image_frame_indices[camera])
                == int(transition.next_image_frame_indices[reference])
                and float(transition.image_timestamps[camera])
                == float(transition.image_timestamps[reference])
                and float(transition.next_image_timestamps[camera])
                == float(transition.next_image_timestamps[reference])
                and float(transition.image_state_timestamps[camera])
                == float(transition.image_state_timestamps[reference])
                and float(transition.next_image_state_timestamps[camera])
                == float(transition.next_image_state_timestamps[reference])
            )
        previous_active = self._active
        for camera in self.camera_order:
            self._observe_frame(
                camera,
                frame_index=int(transition.image_frame_indices[camera]),
                timestamp=float(transition.image_timestamps[camera]),
                frame=transition.images[camera],
                intrinsics=transition.camera_intrinsics[camera],
                extrinsics=transition.camera_extrinsics[camera],
                resolution=transition.camera_resolutions[camera],
                active=previous_active,
            )
        self._observe_info(transition.info)
        for camera in self.camera_order:
            self._observe_frame(
                camera,
                frame_index=int(transition.next_image_frame_indices[camera]),
                timestamp=float(transition.next_image_timestamps[camera]),
                frame=transition.next_images[camera],
                intrinsics=transition.next_camera_intrinsics[camera],
                extrinsics=transition.next_camera_extrinsics[camera],
                resolution=transition.next_camera_resolutions[camera],
                active=self._active,
            )

    def on_episode_end(self, summary: Any) -> None:
        if int(summary.steps) <= 0:
            raise ValueError("three-view benchmark evidence has no transitions")

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
            raise KeyError(f"benchmark visual info is missing {missing}")
        if (
            info["renderer_backend"] != "mujoco.Renderer"
            or info["geometry_source"] != "mujoco_xml"
        ):
            raise ValueError("benchmark visual evidence is not MuJoCo-backed")
        onset = int(info["visual_signal_onset_step"])
        kind = str(info["visual_signal_kind"])
        if onset < 0 or not kind:
            raise ValueError("benchmark visual signal metadata is invalid")
        if self._onset_step is not None and onset != self._onset_step:
            raise ValueError("benchmark visual signal onset changed")
        if self._kind is not None and kind != self._kind:
            raise ValueError("benchmark visual signal kind changed")
        visible = info["cue_visible_expected"]
        if isinstance(visible, Mapping):
            normalized = {
                camera: bool(visible.get(camera, False)) for camera in self.camera_order
            }
        elif type(visible) is bool:
            normalized = {camera: visible for camera in self.camera_order}
        else:
            raise TypeError("cue_visible_expected must be bool or camera mapping")
        if self._visible and normalized != self._visible:
            raise ValueError("benchmark cue visibility expectation changed")
        self._onset_step = onset
        self._kind = kind
        self._visible = normalized
        self._active = bool(info["visual_signal_active"])
        self._active_observed = bool(self._active_observed or self._active)

    def _observe_frame(
        self,
        camera: str,
        *,
        frame_index: int,
        timestamp: float,
        frame: Any,
        intrinsics: Any,
        extrinsics: Any,
        resolution: Any,
        active: bool,
    ) -> None:
        rgb = np.asarray(frame, dtype=np.uint8)
        intrinsic = np.asarray(intrinsics, dtype=np.float64)
        pose = np.asarray(extrinsics, dtype=np.float64)
        size = np.asarray(resolution, dtype=np.int64)
        if (
            rgb.ndim != 3
            or rgb.shape[2] != 3
            or not np.any(rgb)
            or intrinsic.shape != (3, 3)
            or pose.shape != (4, 4)
            or size.shape != (2,)
            or not np.isfinite(intrinsic).all()
            or not np.isfinite(pose).all()
            or not np.array_equal(size, np.asarray(rgb.shape[:2], dtype=np.int64))
        ):
            raise ValueError(f"{camera} benchmark raw frame/calibration is invalid")
        digest = _array_sha256(rgb)
        previous = self._frames[camera].get(frame_index)
        if previous is not None:
            if previous != (timestamp, digest):
                raise ValueError(f"{camera} repeated benchmark frame changed")
            return
        if frame_index != self._last_indices[camera] + 1:
            raise ValueError(f"{camera} benchmark frame indices are not contiguous")
        self._frames[camera][frame_index] = (timestamp, digest)
        self._last_indices[camera] = frame_index
        self._first_hash.setdefault(camera, digest)
        self._last_hash[camera] = digest
        if not active:
            self._pre_signal[camera].append((frame_index, timestamp, digest))
        elif camera not in self._active_hash:
            self._active_hash[camera] = digest
        if camera not in self._first_pose:
            self._first_pose[camera] = pose.copy()
        first_pose = self._first_pose[camera]
        self._translation_travel[camera] = max(
            self._translation_travel[camera],
            float(np.linalg.norm(pose[:3, 3] - first_pose[:3, 3])),
        )
        self._pose_max_abs_delta[camera] = max(
            self._pose_max_abs_delta[camera],
            float(np.max(np.abs(pose - first_pose))),
        )

    def evidence(self) -> dict[str, Any]:
        frame_counts = {
            camera: len(self._frames[camera]) for camera in self.camera_order
        }
        if (
            any(value <= 0 for value in frame_counts.values())
            or len(set(frame_counts.values())) != 1
            or tuple(self._active_hash) != self.camera_order
            or not self._active_observed
            or not all(self._visible.get(camera, False) for camera in self.camera_order)
            or not self._cross_camera_sync
            or self._onset_step is None
            or self._kind is None
        ):
            raise RuntimeError("three-view benchmark evidence is incomplete")
        if self.task_id == "visual_event_stop" and any(
            not self._pre_signal[camera] for camera in self.camera_order
        ):
            raise RuntimeError("event benchmark has no pre-signal RGB evidence")
        return {
            "frame_counts": frame_counts,
            "first_rgb_sha256": dict(self._first_hash),
            "last_rgb_sha256": dict(self._last_hash),
            "active_rgb_sha256": dict(self._active_hash),
            "pre_signal_frame_counts": {
                camera: len(self._pre_signal[camera]) for camera in self.camera_order
            },
            "pre_signal_sequence_sha256": {
                camera: _sequence_sha256(self._pre_signal[camera])
                for camera in self.camera_order
            },
            "camera_translation_travel_m": dict(self._translation_travel),
            "fixed_extrinsics_max_abs_delta": self._pose_max_abs_delta["fixed"],
            "cross_camera_sync": self._cross_camera_sync,
            "cue_visible_expected": dict(self._visible),
            "visual_signal_active_observed": self._active_observed,
            "visual_signal_onset_step": self._onset_step,
            "visual_signal_kind": self._kind,
        }


def _environment_preflight(
    config: Mapping[str, Any], *, task_id: str, camera_order: Sequence[str]
) -> dict[str, Any]:
    camera_cfg = _mapping(config, "camera")
    benchmark_cfg = _mapping(config, "benchmark")
    if (
        camera_cfg.get("renderer_backend") != "mujoco.Renderer"
        or camera_cfg.get("geometry_source") != "mujoco_xml"
        or camera_cfg.get("raw_unannotated") is not True
    ):
        raise ValueError("benchmark config is not bound to raw MuJoCo RGB")
    env = VisualRequiredEnv(
        VisualRequiredEnvConfig(
            task_id=task_id,
            control_dt=1.0 / float(camera_cfg["control_hz"]),
            episode_len=int(benchmark_cfg["max_steps"]),
            image_width=int(camera_cfg["width"]),
            image_height=int(camera_cfg["height"]),
            render_cue_mode="truth",
        )
    )
    try:
        if tuple(str(value) for value in getattr(env, "camera_names", ())) != tuple(
            camera_order
        ):
            raise ValueError("benchmark MuJoCo camera order is not canonical")
        if str(getattr(env, "renderer_backend", "")) != "mujoco.Renderer":
            raise ValueError("benchmark requires renderer_backend='mujoco.Renderer'")
        mujoco_gl = str(os.environ.get("MUJOCO_GL", ""))
        if not mujoco_gl:
            raise ValueError("MUJOCO_GL must name the active rendering context")
        model_xml_path = Path(getattr(env, "model_xml_path", "")).resolve()
        if not model_xml_path.is_file():
            raise FileNotFoundError(model_xml_path)
        model_xml_sha256 = str(getattr(env, "model_xml_sha256", ""))
        if model_xml_sha256 != _sha256(model_xml_path):
            raise ValueError("benchmark XML SHA does not match XML bytes")
        _, info = env.reset(seed=0, randomize=False)
        if (
            info.get("renderer_backend") != "mujoco.Renderer"
            or info.get("geometry_source") != "mujoco_xml"
        ):
            raise ValueError("benchmark environment info lacks MuJoCo provenance")
        rig: dict[str, Any] = {}
        for camera in camera_order:
            raw = env.camera_calibration(
                camera=camera,
                width=int(camera_cfg["width"]),
                height=int(camera_cfg["height"]),
            )
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
            if not isinstance(raw, Mapping) or not required <= set(raw):
                raise ValueError(
                    f"{camera} benchmark calibration provenance is incomplete"
                )
            if raw["convention"] != "opencv_optical_camera_pose_in_world":
                raise ValueError(
                    f"{camera} benchmark calibration convention is invalid"
                )
            rig[camera] = {
                "camera_id": int(raw["camera_id"]),
                "parent_body_id": int(raw["parent_body_id"]),
                "parent_body_name": str(raw["parent_body_name"]),
                "fovy_degrees": float(raw["fovy_degrees"]),
                "convention": str(raw["convention"]),
            }
        if rig["fixed"]["parent_body_name"] != "world":
            raise ValueError("benchmark fixed camera is not world-mounted")
        ego_parents = [rig[camera]["parent_body_name"] for camera in camera_order[1:]]
        if any(value in {"", "world"} for value in ego_parents) or len(
            set(ego_parents)
        ) != len(ego_parents):
            raise ValueError("benchmark ego cameras lack distinct robot parents")
        return {
            "renderer_backend": "mujoco.Renderer",
            "geometry_source": "mujoco_xml",
            "mujoco_version": str(mujoco.__version__),
            "mujoco_gl": mujoco_gl,
            "model_xml_path": _display_path(model_xml_path),
            "model_xml_sha256": model_xml_sha256,
            "camera_rig": rig,
        }
    finally:
        env.close()


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != CONFIG_VERSION:
        raise ValueError("unsupported Phase M0 config")
    return payload


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ValueError(f"config field {key!r} must be a mapping")
    return item


def _configured_dataset_episode_total(config: Mapping[str, Any]) -> int:
    return len(VISUAL_REQUIRED_TASKS) * int(
        _mapping(config, "dataset")["episodes_per_task"]
    )


def _root_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _camera_order(config: Mapping[str, Any]) -> tuple[str, ...]:
    dataset_order = tuple(
        str(value) for value in _mapping(config, "dataset").get("camera_order", ())
    )
    camera_cfg = _mapping(config, "camera")
    camera_order = tuple(str(value) for value in camera_cfg.get("camera_order", ()))
    if dataset_order != CAMERA_ORDER or camera_order != CAMERA_ORDER:
        raise ValueError(f"M0-v2 camera order must be {list(CAMERA_ORDER)!r}")
    if camera_cfg.get("calibration_convention") != (
        "opencv_optical_camera_pose_in_world"
    ):
        raise ValueError("M0-v2 camera calibration convention is not canonical")
    return dataset_order


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    if temporary.exists():
        raise FileExistsError(f"stale partial benchmark report: {temporary}")
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


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    serialized = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _sequence_sha256(value: Sequence[tuple[int, float, str]]) -> str:
    serialized = json.dumps(list(value), separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
