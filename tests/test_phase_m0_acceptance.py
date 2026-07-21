from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import h5py
import numpy as np
import pytest
import torch

from eval.visual_required import (
    CAMERA_ORDER,
    FORMAT_VERSION as VISUAL_FORMAT_VERSION,
    REQUIRED_POLICIES,
    SCRIPTED_ORACLE_POLICY,
    SHUFFLED_VISION_POLICY,
    STATE_ONLY_POLICY,
    VISION_ORACLE_POLICY,
    VisualRequiredEpisode,
    contains_privileged_path,
    mapping_key,
    visual_required_acceptance,
)
from scripts import accept_phase_m0 as phase_m0
from scripts.accept_phase_m0 import (
    _checkpoint_tree,
    _checkpoint_tree_digest,
    _state_dict_max_abs_diff,
    phase_m0_acceptance_report,
    run_legacy_regression,
)
from scripts.audit_wam_multimodal_dataset import (
    FORMAT_VERSION as AUDIT_FORMAT_VERSION,
    _audit_cue_pair_pixels,
    _audit_visual_event_signal_and_brake_lights,
    _image_health,
    _minimum_formal_size_details,
)


TASK = "visual_event_stop"
SEEDS = tuple(range(10))
CUES = (0, 1)
COMMON_PATHS = (
    "past_executed_actions",
    "proprioception",
    "task.id",
    "task.text",
)
VISUAL_PATHS = (*COMMON_PATHS, "images.fixed")
THRESHOLDS = {
    "maximum_state_only_success_rate": {"operator": "<=", "value": 0.70},
    "minimum_scripted_oracle_success_rate": {"operator": ">=", "value": 0.95},
    "minimum_vision_oracle_success_rate": {"operator": ">=", "value": 0.95},
    "minimum_opposite_rgb_success_drop": {"operator": ">=", "value": 0.20},
}


def test_visual_required_acceptance_includes_threshold_boundaries() -> None:
    report = _accept(_records_at_boundaries())

    assert report["passed"] is True
    assert report["macro"]["policies"][STATE_ONLY_POLICY]["rate"] == 0.70
    assert report["macro"]["policies"][SCRIPTED_ORACLE_POLICY]["rate"] == 0.95
    assert report["macro"]["policies"][VISION_ORACLE_POLICY]["rate"] == 0.95
    assert report["macro"]["clean_minus_opposite_rgb_success_drop"] == pytest.approx(
        0.20
    )
    contract = report["observation_contract"]
    assert contract["per_record_violation_count"] == 0
    assert contract["required_visual_consumed_paths"] == sorted(VISUAL_PATHS)


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    (
        ("state_above_ceiling", "every_task_passes"),
        ("scripted_below_floor", "every_task_passes"),
        ("vision_below_floor", "every_task_passes"),
        ("opposite_drop_below_floor", "every_task_passes"),
        ("missing_pair", "identical_complete_seed_cue_pairs"),
        ("duplicate_pair", "unique_task_policy_seed_cue_records"),
        ("wrong_rgb_mapping", "opposite_cue_derangement_valid"),
        ("wrong_action_source", "policy_action_sources"),
        (
            "inconsistent_physical_identity",
            "identical_physical_seed_worlds_across_cue_and_policy",
        ),
        ("privileged_path", "policy_observation_contract"),
        ("state_presented_rgb", "policy_observation_contract"),
        ("one_vision_record_missing_task_text", "policy_observation_contract"),
        ("missing_camera_evidence", "mujoco_three_camera_render_evidence"),
        ("cross_camera_desync", "mujoco_three_camera_render_evidence"),
        ("wrong_renderer_provenance", "mujoco_three_camera_render_evidence"),
        ("pre_onset_future_leak", "mujoco_three_camera_render_evidence"),
    ),
)
def test_visual_required_acceptance_fails_closed(
    mutation: str, failed_check: str
) -> None:
    records = _records_at_boundaries()
    if mutation == "state_above_ceiling":
        index = _first(records, STATE_ONLY_POLICY, success=False)
        records[index] = replace(records[index], success=True, failure=False)
    elif mutation == "scripted_below_floor":
        index = _first(records, SCRIPTED_ORACLE_POLICY, success=True)
        records[index] = replace(records[index], success=False, failure=True)
    elif mutation == "vision_below_floor":
        index = _first(records, VISION_ORACLE_POLICY, success=True)
        records[index] = replace(records[index], success=False, failure=True)
    elif mutation == "opposite_drop_below_floor":
        index = _first(records, SHUFFLED_VISION_POLICY, success=False)
        records[index] = replace(records[index], success=True, failure=False)
    elif mutation == "missing_pair":
        records.pop()
    elif mutation == "duplicate_pair":
        records.append(records[-1])
    elif mutation == "wrong_rgb_mapping":
        index = _first(records, SHUFFLED_VISION_POLICY)
        item = records[index]
        records[index] = replace(item, rgb_source_cue_id=item.cue_id)
    elif mutation == "wrong_action_source":
        index = _first(records, SHUFFLED_VISION_POLICY)
        records[index] = replace(records[index], action_source=None)
    elif mutation == "inconsistent_physical_identity":
        index = _first(records, VISION_ORACLE_POLICY)
        records[index] = replace(records[index], scene_id="different-scene")
    elif mutation == "privileged_path":
        index = _first(records, VISION_ORACLE_POLICY)
        item = records[index]
        records[index] = replace(
            item,
            presented_observation_paths=(
                *item.presented_observation_paths,
                "task_truth.cue_variant",
            ),
        )
    elif mutation == "state_presented_rgb":
        index = _first(records, STATE_ONLY_POLICY)
        item = records[index]
        records[index] = replace(
            item,
            presented_observation_paths=(
                *item.presented_observation_paths,
                "images.fixed",
            ),
        )
    elif mutation == "one_vision_record_missing_task_text":
        index = _first(records, VISION_ORACLE_POLICY)
        item = records[index]
        records[index] = replace(
            item,
            consumed_observation_paths=tuple(
                path for path in item.consumed_observation_paths if path != "task.text"
            ),
        )
    elif mutation == "missing_camera_evidence":
        index = _first(records, VISION_ORACLE_POLICY)
        records[index] = replace(records[index], camera_order=("fixed",))
    elif mutation == "cross_camera_desync":
        index = _first(records, VISION_ORACLE_POLICY)
        records[index] = replace(records[index], cross_camera_sync=False)
    elif mutation == "wrong_renderer_provenance":
        index = _first(records, VISION_ORACLE_POLICY)
        records[index] = replace(records[index], renderer_backend="analytical")
    elif mutation == "pre_onset_future_leak":
        index = next(
            index
            for index, record in enumerate(records)
            if record.policy == VISION_ORACLE_POLICY and record.cue_id == 1
        )
        changed = dict(records[index].pre_signal_sequence_sha256 or {})
        changed["fixed"] = _fake_sha("future-cue-leak")
        records[index] = replace(records[index], pre_signal_sequence_sha256=changed)
    else:  # pragma: no cover - parameter exhaustiveness.
        raise AssertionError(mutation)

    report = _accept(records)

    assert report["passed"] is False
    assert report["checks"][failed_check]["passed"] is False


@pytest.mark.parametrize(
    "path",
    (
        "cue_variant",
        "rendered_cue_variant",
        "task_truth",
        "observation.task_truth.target",
    ),
)
def test_visual_privilege_denylist_covers_cue_truth(path: str) -> None:
    assert contains_privileged_path((path,)) is True


def test_state_dict_strict_reload_handles_bool_tensors() -> None:
    left = {
        "mask": torch.tensor([True, False]),
        "weight": torch.tensor([1.0, 2.0]),
    }
    right = {key: value.clone() for key, value in left.items()}

    assert _state_dict_max_abs_diff(left, right) == 0.0
    right["mask"][1] = True
    assert _state_dict_max_abs_diff(left, right) == 1.0
    right["mask"] = left["mask"].clone()
    right["weight"][0] += 0.25
    assert _state_dict_max_abs_diff(left, right) == pytest.approx(0.25)


def test_legacy_gate_uses_privileged_state_leakage_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "final.safetensors").write_bytes(b"immutable")
    experiment = {
        "runtime": {
            "fixed_actions": {},
            "anchor_residual_scale": 1.0,
            "normalized_action_clip": 1.0,
            "observation_residual_nrmse_max": 1.0,
            "risk_veto": False,
            "max_failure_probability": 1.0,
            "max_predicted_robot_distance": 10.0,
            "max_action_ood": 10.0,
            "action_ood_threshold": 1.0,
            "latency_budget_ms": 1000.0,
        },
        "action_chunk": {
            "horizon": 2,
            "execution_steps": 1,
            "solver_steps": 1,
            "warm_start_mode": "shift_repeat_last",
            "solver": "euler",
        },
        "data": {"action_dim": 8},
        "evaluation": {"challenge_environment": {}},
    }

    class FakeModule:
        def state_dict(self) -> dict[str, torch.Tensor]:
            return {
                "ready": torch.tensor(True),
                "weight": torch.tensor([1.0]),
            }

    def fake_load(*args: Any, **kwargs: Any) -> tuple[Any, Any, dict[str, Any]]:
        del args, kwargs
        return FakeModule(), FakeModule(), {"experiment_config": experiment}

    class FakeEnvironment:
        def __init__(self, config: Any) -> None:
            del config

        def close(self) -> None:
            pass

    class FakePolicy:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

    class FakeObserver:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def finish(self, summary: Any) -> Any:
            return summary

    class FakeRunner:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def run_episode(self, **kwargs: Any) -> Any:
            return SimpleNamespace(seed=kwargs["seed"])

    def fake_aggregate(records: list[Any]) -> dict[str, Any]:
        return {
            "episodes": len(records),
            "success_rate": 1.0,
            "fallback_trigger_rate": 0.0,
            "action_source_coverage": 1.0,
            "all_actions_finite_and_bounded": True,
            "privileged_state_leakage": False,
        }

    monkeypatch.setattr(phase_m0, "load_joint_wam_checkpoint", fake_load)
    monkeypatch.setattr(phase_m0, "TwoRobotCooperativeStopEnv", FakeEnvironment)
    monkeypatch.setattr(phase_m0, "JointWAMPolicy", FakePolicy)
    monkeypatch.setattr(phase_m0, "ClosedLoopEpisodeObserver", FakeObserver)
    monkeypatch.setattr(phase_m0, "SimulationRunner", FakeRunner)
    monkeypatch.setattr(phase_m0, "aggregate_closed_loop", fake_aggregate)
    config = {
        "legacy_regression": {
            "expected_schema_version": "wam.proprio/1.0",
            "standard_seed_start": 1,
            "challenge_seed_start": 2,
            "minimum_success_rate": 1.0,
            "maximum_reference_regression": 0.0,
            "reference_success_rate": {"standard": 1.0, "challenge": 1.0},
            "expected_checkpoint_tree_sha256": _checkpoint_tree_digest(
                _checkpoint_tree(checkpoint)
            ),
        }
    }

    report = run_legacy_regression(config, checkpoint=checkpoint, episodes_per_suite=1)

    assert report["passed"] is True
    assert report["strict_reload"]["passed"] is True
    assert report["checkpoint_tree_immutable"] is True
    assert all(
        suite["privileged_state_leakage"] is False
        for suite in report["suites"].values()
    )


def test_terminal_next_rgb_is_counted_as_a_captured_frame(tmp_path: Path) -> None:
    path = tmp_path / "frames.hdf5"
    current_indices = np.asarray([0, 0, 1, 1], dtype=np.int64)
    next_indices = np.asarray([0, 1, 1, 2], dtype=np.int64)
    first = np.full((8, 8, 3), 30, dtype=np.uint8)
    second = np.full((8, 8, 3), 90, dtype=np.uint8)
    terminal = np.full((8, 8, 3), 150, dtype=np.uint8)
    current = np.stack([first, first, second, second])
    following = np.stack([first, second, second, terminal])
    with h5py.File(path, "w") as file:
        images = file.create_dataset("current", data=current)
        next_images = file.create_dataset("next", data=following)
        health = _image_health(images, current_indices, next_images, next_indices)

    assert health["captured_frames"] == 3
    assert health["captured_comparisons"] == 2


def test_small_dataset_is_diagnostic_pass_but_not_formal_pass() -> None:
    diagnostic = _minimum_formal_size_details(18, formal_protocol=False)
    formal = _minimum_formal_size_details(18, formal_protocol=True)

    assert diagnostic["passed"] is True
    assert diagnostic["formal_requirement_satisfied"] is False
    assert formal["passed"] is False


def test_captured_video_observer_encodes_terminal_next_frame(tmp_path: Path) -> None:
    from scripts.collect_wam_multimodal_dataset import _CapturedFrameVideoObserver

    path = tmp_path / "terminal.mp4"
    observer = _CapturedFrameVideoObserver(path, stream="fixed", fps=10.0, codec="mp4v")
    observer.on_episode_start()
    frames = [np.full((16, 16, 3), value, dtype=np.uint8) for value in (30, 90, 150)]
    pairs = ((0, 0), (0, 1), (1, 1), (1, 2))
    for current_index, next_index in pairs:
        observer.on_transition(
            SimpleNamespace(
                image_frame_indices={"fixed": current_index},
                next_image_frame_indices={"fixed": next_index},
                images={"fixed": frames[current_index]},
                next_images={"fixed": frames[next_index]},
            )
        )
    observer.on_episode_end(SimpleNamespace(steps=len(pairs)))

    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    finally:
        capture.release()
    assert observer.frames_written == 3


def test_event_cue_pair_requires_identical_full_pre_onset_rgb_sequence(
    tmp_path: Path,
) -> None:
    entries = [_write_cue_pair_hdf5(tmp_path, cue=cue) for cue in (0, 1)]

    valid = _audit_cue_pair_pixels(
        tmp_path,
        entries,
        camera_order=CAMERA_ORDER,
        minimum_changed_pixels=1,
    )

    assert valid["passed"] is True
    assert valid["pre_onset_sequences_valid"] is True
    assert valid["pre_onset_frame_comparisons"] == 6

    leaked_path = tmp_path / str(entries[1]["hdf5_path"])
    with h5py.File(leaked_path, "r+") as file:
        leaked = np.full((8, 8, 3), 211, dtype=np.uint8)
        file["data/observation/images/robot_0_camera"][1] = leaked
        file["data/next_observation/images/robot_0_camera"][0] = leaked

    leaked = _audit_cue_pair_pixels(
        tmp_path,
        entries,
        camera_order=CAMERA_ORDER,
        minimum_changed_pixels=1,
    )

    assert leaked["passed"] is False
    assert leaked["pre_onset_sequences_valid"] is False


def test_event_signal_and_brake_light_semantic_probe_passes() -> None:
    report = _audit_visual_event_signal_and_brake_lights()

    assert report["visual_event_signal_semantic_isolation"]["passed"] is True
    assert report["brake_lights_action_causal_and_red_only"]["passed"] is True
    assert report["visual_event_signal_semantic_isolation"][
        "onset_lamps_off"
    ] is True
    assert report["brake_lights_action_causal_and_red_only"]["never_green"] is True


def test_phase_aggregate_verifies_hashes_and_formal_dataset_floor(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "dataset"
    benchmark_dir = tmp_path / "benchmark"
    data_dir.mkdir()
    benchmark_dir.mkdir()
    manifest = data_dir / "manifest.json"
    manifest.write_text("{}")
    audit_path = tmp_path / "audit.json"
    benchmark_path = benchmark_dir / "visual_required_benchmark.json"
    audit_path.write_text("{}")
    benchmark_path.write_text("{}")
    tasks = (
        "visual_event_stop",
        "visual_target_select",
        "visual_obstacle_avoid",
    )
    benchmark_records = _small_formal_benchmark_records(tasks)
    records = benchmark_dir / "visual_required_episodes.jsonl"
    records.write_text(
        "".join(
            json.dumps(record.as_dict(), sort_keys=True) + "\n"
            for record in benchmark_records
        )
    )
    config_path = tmp_path / "m0.yaml"
    source_path = Path("eval/visual_required.py")
    xml_path = Path("envs/assets/two_robot_carry.xml")
    camera_rig = {
        "fixed": {
            "camera_id": 0,
            "parent_body_id": 0,
            "parent_body_name": "world",
            "fovy_degrees": 45.0,
            "convention": "opencv_optical_camera_pose_in_world",
        },
        "robot_0_camera": {
            "camera_id": 2,
            "parent_body_id": 1,
            "parent_body_name": "robot_a",
            "fovy_degrees": 100.0,
            "convention": "opencv_optical_camera_pose_in_world",
        },
        "robot_1_camera": {
            "camera_id": 3,
            "parent_body_id": 2,
            "parent_body_name": "robot_b",
            "fovy_degrees": 100.0,
            "convention": "opencv_optical_camera_pose_in_world",
        },
    }
    provenance = {
        "renderer_backend": "mujoco.Renderer",
        "geometry_source": "mujoco_xml",
        "mujoco_version": phase_m0.mujoco.__version__,
        "mujoco_gl": "egl",
        "model_xml_path": xml_path.as_posix(),
        "model_xml_sha256": _sha256(xml_path),
        "camera_rig": camera_rig,
        "source_sha256": {source_path.as_posix(): _sha256(source_path)},
    }
    acceptance = {
        **THRESHOLDS,
        "require_mujoco_renderer_provenance": True,
        "require_three_camera_dataset": True,
        "require_three_camera_benchmark": True,
        "require_cross_camera_sync": True,
        "require_dynamic_robot_camera_extrinsics": True,
        "require_raw_unannotated_rgb": True,
        "require_visual_signal_onset_evidence": True,
        "require_visual_event_signal_semantic_isolation": True,
        "require_brake_lights_action_causal_and_red_only": True,
    }
    config = {
        "dataset": {
            "directory": str(data_dir),
            "episodes_per_task": 667,
            "schema_version": "wam.multimodal/1.1",
            "camera_order": list(CAMERA_ORDER),
        },
        "camera": {
            "camera_order": list(CAMERA_ORDER),
            "renderer_backend": "mujoco.Renderer",
            "geometry_source": "mujoco_xml",
            "raw_unannotated": True,
            "calibration_convention": "opencv_optical_camera_pose_in_world",
        },
        "audit": {
            "report": str(audit_path),
            "required_source_files": [source_path.as_posix()],
        },
        "benchmark": {
            "output_directory": str(benchmark_dir),
            "physical_seeds_per_task": 1,
            "physical_seed_start": 10,
            "cue_variants": [0, 1],
            "policies": list(REQUIRED_POLICIES),
            "camera_order": list(CAMERA_ORDER),
            "policy_rgb_stream": "fixed",
            "record_all_view_evidence": True,
        },
        "legacy_regression": {"episodes_per_suite": 20},
        "acceptance": acceptance,
    }
    config_path.write_text(json.dumps(config, sort_keys=True))
    config_sha = _sha256(config_path)
    audit = {
        "format_version": AUDIT_FORMAT_VERSION,
        "formal_protocol": True,
        "passed": True,
        "config_sha256": config_sha,
        "data_dir": str(data_dir),
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "episodes": 2_001,
        "audited_episodes": 2_001,
        "schema_version": "wam.multimodal/1.1",
        "camera_order": list(CAMERA_ORDER),
        "provenance": provenance,
        "cue_pair_audit": {
            "passed": True,
            "pre_onset_sequences_valid": True,
            "pre_onset_frame_comparisons": 1,
        },
        "checks": {
            name: {"passed": True}
            for name in (
                "manifest_contract",
                "episode_count",
                "minimum_formal_dataset_size",
                "all_episode_contracts",
                "capture_sync_skew_p99",
                "maximum_action_frame_age",
                "zero_episode_boundary_crossings",
                "corrupt_videos",
                "cross_camera_frame_sync",
                "fixed_static_and_robot_dynamic_extrinsics",
                "raw_visual_signal_onset_and_visibility",
                "paired_cue_pixels_differ_in_every_camera",
                "visual_event_signal_semantic_isolation",
                "brake_lights_action_causal_and_red_only",
                "split_isolation",
                "multimodal_loader_smoke",
            )
        },
    }
    benchmark_acceptance = visual_required_acceptance(
        benchmark_records,
        tasks=tasks,
        physical_seeds=(10,),
        cue_variants=CUES,
        thresholds=acceptance,
    )
    benchmark = {
        "format_version": VISUAL_FORMAT_VERSION,
        "formal_protocol": True,
        "passed": True,
        "config_sha256": config_sha,
        "output_directory": str(benchmark_dir),
        "episode_records": str(records),
        "episode_records_sha256": _sha256(records),
        "protocol": {
            "physical_seeds_per_task": 1,
            "physical_seeds": [10],
            "cue_variants": [0, 1],
            "tasks": list(tasks),
            "policies": list(REQUIRED_POLICIES),
            "camera_order": list(CAMERA_ORDER),
            "render_requests": list(CAMERA_ORDER),
            "policy_rgb_stream": "fixed",
            "record_all_view_evidence": True,
            "raw_unannotated": True,
            **provenance,
        },
        "acceptance": benchmark_acceptance,
    }
    legacy = {
        "passed": True,
        "episodes_per_suite": 20,
        "strict_reload": {"passed": True},
        "checkpoint_matches_pre_m0_anchor": True,
        "expected_checkpoint_tree_sha256": "a" * 64,
        "checkpoint_tree_immutable": True,
        "checkpoint_tree_sha256_before": "a" * 64,
        "checkpoint_tree_sha256_after": "a" * 64,
        "suites": {},
    }

    report = phase_m0_acceptance_report(
        config,
        config_path=config_path,
        audit=audit,
        audit_path=audit_path,
        benchmark=benchmark,
        benchmark_path=benchmark_path,
        legacy=legacy,
        formal_protocol=True,
    )
    assert report["passed"] is True
    assert report["formal_protocol"] is True

    manifest.write_text('{"tampered": true}')
    tampered = phase_m0_acceptance_report(
        config,
        config_path=config_path,
        audit=audit,
        audit_path=audit_path,
        benchmark=benchmark,
        benchmark_path=benchmark_path,
        legacy=legacy,
        formal_protocol=True,
    )
    assert tampered["passed"] is False
    assert tampered["checks"]["audit_artifacts_hashed"]["passed"] is False

    low_config_path = tmp_path / "low.yaml"
    low_config = {
        **config,
        "dataset": {"directory": str(data_dir), "episodes_per_task": 666},
    }
    low_config_path.write_text(json.dumps(low_config, sort_keys=True))
    low_sha = _sha256(low_config_path)
    low_audit = {
        **audit,
        "config_sha256": low_sha,
        "episodes": 1_998,
        "audited_episodes": 1_998,
    }
    low_benchmark = {**benchmark, "config_sha256": low_sha}
    low = phase_m0_acceptance_report(
        low_config,
        config_path=low_config_path,
        audit=low_audit,
        audit_path=audit_path,
        benchmark=low_benchmark,
        benchmark_path=benchmark_path,
        legacy=legacy,
        formal_protocol=True,
    )
    assert low["formal_protocol"] is False
    assert low["passed"] is False
    assert low["checks"]["formal_protocol"]["passed"] is False


def test_checkpoint_tree_rejects_symlinks(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    payload = checkpoint / "payload"
    payload.write_text("model")
    (checkpoint / "alias").symlink_to(payload)

    with pytest.raises(ValueError, match="symlinks"):
        _checkpoint_tree(checkpoint)


def _records_at_boundaries() -> list[VisualRequiredEpisode]:
    records: list[VisualRequiredEpisode] = []
    success_counts = {
        STATE_ONLY_POLICY: 14,
        SCRIPTED_ORACLE_POLICY: 19,
        VISION_ORACLE_POLICY: 19,
        SHUFFLED_VISION_POLICY: 15,
    }
    pairs = [(seed, cue) for seed in SEEDS for cue in CUES]
    for policy in REQUIRED_POLICIES:
        for index, (seed, cue) in enumerate(pairs):
            success = index < success_counts[policy]
            consumed = COMMON_PATHS if policy == STATE_ONLY_POLICY else VISUAL_PATHS
            if policy == SCRIPTED_ORACLE_POLICY:
                consumed = ()
            opposite = policy == SHUFFLED_VISION_POLICY
            presented = (
                (*VISUAL_PATHS, "image_frame_indices.fixed", "image_timestamps.fixed")
                if policy in {VISION_ORACLE_POLICY, SHUFFLED_VISION_POLICY}
                else COMMON_PATHS
            )
            records.append(
                VisualRequiredEpisode(
                    task_id=TASK,
                    cue_id=cue,
                    physical_seed=seed,
                    policy=policy,
                    success=success,
                    failure=not success,
                    failure_reason="none" if success else "wrong_cue",
                    steps=4,
                    total_reward=1.0 if success else 0.0,
                    presented_observation_paths=presented,
                    consumed_observation_paths=consumed,
                    privileged_observation_seen=False,
                    rgb_source_cue_id=(1 - cue if opposite else cue)
                    if policy in {VISION_ORACLE_POLICY, SHUFFLED_VISION_POLICY}
                    else None,
                    rgb_mapping_key=mapping_key(TASK, seed, cue) if opposite else None,
                    action_source=(
                        VISION_ORACLE_POLICY
                        if policy == SHUFFLED_VISION_POLICY
                        else policy
                    ),
                    initial_proprioception_sha256=_fake_sha(f"initial:{TASK}:{seed}"),
                    task_condition_sha256=_fake_sha(f"task:{TASK}"),
                    scene_id=f"scene:{TASK}:{seed}",
                    object_combination_id=f"objects:{TASK}",
                    **_render_evidence_kwargs(
                        task=TASK, policy=policy, seed=seed, cue=cue
                    ),
                )
            )
    return records


def _small_formal_benchmark_records(
    tasks: tuple[str, ...],
) -> list[VisualRequiredEpisode]:
    records: list[VisualRequiredEpisode] = []
    for task in tasks:
        for policy in REQUIRED_POLICIES:
            for cue in CUES:
                if policy == STATE_ONLY_POLICY:
                    success = cue == 0
                    consumed = COMMON_PATHS
                elif policy == SCRIPTED_ORACLE_POLICY:
                    success = True
                    consumed = ()
                elif policy == VISION_ORACLE_POLICY:
                    success = True
                    consumed = VISUAL_PATHS
                else:
                    success = False
                    consumed = VISUAL_PATHS
                opposite = policy == SHUFFLED_VISION_POLICY
                presented = (
                    (
                        *VISUAL_PATHS,
                        "image_frame_indices.fixed",
                        "image_timestamps.fixed",
                    )
                    if policy in {VISION_ORACLE_POLICY, SHUFFLED_VISION_POLICY}
                    else COMMON_PATHS
                )
                records.append(
                    VisualRequiredEpisode(
                        task_id=task,
                        cue_id=cue,
                        physical_seed=10,
                        policy=policy,
                        success=success,
                        failure=not success,
                        failure_reason="none" if success else "wrong_cue",
                        steps=4,
                        total_reward=float(success),
                        presented_observation_paths=presented,
                        consumed_observation_paths=consumed,
                        privileged_observation_seen=False,
                        rgb_source_cue_id=(1 - cue if opposite else cue)
                        if policy in {VISION_ORACLE_POLICY, SHUFFLED_VISION_POLICY}
                        else None,
                        rgb_mapping_key=mapping_key(task, 10, cue)
                        if opposite
                        else None,
                        action_source=(
                            VISION_ORACLE_POLICY
                            if policy == SHUFFLED_VISION_POLICY
                            else policy
                        ),
                        initial_proprioception_sha256=_fake_sha(f"initial:{task}:10"),
                        task_condition_sha256=_fake_sha(f"task:{task}"),
                        scene_id=f"scene:{task}:10",
                        object_combination_id=f"objects:{task}",
                        **_render_evidence_kwargs(
                            task=task, policy=policy, seed=10, cue=cue
                        ),
                    )
                )
    return records


def _accept(records: list[VisualRequiredEpisode]) -> dict[str, Any]:
    return visual_required_acceptance(
        records,
        tasks=(TASK,),
        physical_seeds=SEEDS,
        cue_variants=CUES,
        thresholds=THRESHOLDS,
    )


def _first(
    records: list[VisualRequiredEpisode],
    policy: str,
    *,
    success: bool | None = None,
) -> int:
    return next(
        index
        for index, record in enumerate(records)
        if record.policy == policy and (success is None or record.success is success)
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_cue_pair_hdf5(tmp_path: Path, *, cue: int) -> dict[str, Any]:
    relative = Path(f"episode_{cue}.hdf5")
    current_indices = np.asarray([0, 1, 2], dtype=np.int64)
    next_indices = np.asarray([1, 2, 2], dtype=np.int64)
    timestamps = np.asarray([0.0, 0.1, 0.2], dtype=np.float64)
    next_timestamps = np.asarray([0.1, 0.2, 0.2], dtype=np.float64)
    cameras: dict[str, Any] = {}
    with h5py.File(tmp_path / relative, "w") as file:
        for camera_index, camera in enumerate(CAMERA_ORDER):
            pre0 = np.full((8, 8, 3), 20 + camera_index, dtype=np.uint8)
            pre1 = np.full((8, 8, 3), 40 + camera_index, dtype=np.uint8)
            active = np.full((8, 8, 3), 100 + cue * 50 + camera_index, dtype=np.uint8)
            group = file.require_group("data/observation")
            group.require_group("images").create_dataset(
                camera, data=np.stack((pre0, pre1, active))
            )
            group.require_group("image_timestamp").create_dataset(
                camera, data=timestamps
            )
            group.require_group("image_frame_index").create_dataset(
                camera, data=current_indices
            )
            following = file.require_group("data/next_observation")
            following.require_group("images").create_dataset(
                camera, data=np.stack((pre1, active, active))
            )
            following.require_group("image_timestamp").create_dataset(
                camera, data=next_timestamps
            )
            following.require_group("image_frame_index").create_dataset(
                camera, data=next_indices
            )
            cameras[camera] = {"active_frame_index": 2}
    return {
        "task_id": "visual_event_stop",
        "physical_seed": 77,
        "cue_id": cue,
        "scene_id": "scene-77",
        "object_combination_id": "objects-77",
        "template_id": "train-a",
        "hdf5_path": relative.as_posix(),
        "visual_signal": {"cameras": cameras},
    }


def _render_evidence_kwargs(
    *, task: str, policy: str, seed: int, cue: int
) -> dict[str, Any]:
    cameras = tuple(CAMERA_ORDER)
    return {
        "camera_order": cameras,
        "all_view_frame_counts": {camera: 4 for camera in cameras},
        "all_view_first_rgb_sha256": {
            camera: _fake_sha(f"first:{task}:{seed}:{cue}:{camera}")
            for camera in cameras
        },
        "all_view_last_rgb_sha256": {
            camera: _fake_sha(f"last:{task}:{seed}:{policy}:{cue}:{camera}")
            for camera in cameras
        },
        "active_rgb_sha256": {
            camera: _fake_sha(f"active:{task}:{seed}:{cue}:{camera}")
            for camera in cameras
        },
        "pre_signal_frame_counts": {
            camera: (2 if task == "visual_event_stop" else 0) for camera in cameras
        },
        "pre_signal_sequence_sha256": {
            camera: _fake_sha(f"pre:{task}:{seed}:{policy}:{camera}")
            for camera in cameras
        },
        "camera_translation_travel_m": {
            "fixed": 0.0,
            "robot_0_camera": 0.01,
            "robot_1_camera": 0.01,
        },
        "fixed_extrinsics_max_abs_delta": 0.0,
        "cross_camera_sync": True,
        "renderer_backend": "mujoco.Renderer",
        "geometry_source": "mujoco_xml",
        "mujoco_version": phase_m0.mujoco.__version__,
        "mujoco_gl": "egl",
        "model_xml_sha256": _sha256(Path("envs/assets/two_robot_carry.xml")),
        "raw_unannotated": True,
        "cue_visible_expected": {camera: True for camera in cameras},
        "visual_signal_active_observed": True,
        "visual_signal_onset_step": 14 if task == "visual_event_stop" else 0,
        "visual_signal_kind": "event" if task == "visual_event_stop" else "cue",
        "policy_rgb_stream": (
            "fixed"
            if policy in {VISION_ORACLE_POLICY, SHUFFLED_VISION_POLICY}
            else None
        ),
    }
