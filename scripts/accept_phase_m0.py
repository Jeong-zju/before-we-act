"""Aggregate M0 data/visual evidence and run self-contained legacy regression."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping

# Keep MuJoCo provenance checks deterministic on headless acceptance hosts.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.runtime import RunnerConfig, SimulationRunner  # noqa: E402
from envs.visual_required_env import VISUAL_REQUIRED_TASKS  # noqa: E402
from envs.two_robot_carry_env import (  # noqa: E402
    CooperativeStopEnvConfig,
    TwoRobotCooperativeStopEnv,
)
from eval.closed_loop import (  # noqa: E402
    ClosedLoopEpisodeObserver,
    aggregate_closed_loop,
)
from eval.visual_required import (  # noqa: E402
    CAMERA_ORDER,
    FORMAT_VERSION as VISUAL_FORMAT_VERSION,
    REQUIRED_POLICIES,
    visual_required_acceptance,
)
from models.wam import ActionChunkConfig  # noqa: E402
from policies.joint_wam import JointWAMPolicy, JointWAMPolicyConfig  # noqa: E402
from scripts.audit_wam_multimodal_dataset import (  # noqa: E402
    FORMAT_VERSION as AUDIT_FORMAT_VERSION,
)
from train.joint_wam_checkpointing import load_joint_wam_checkpoint  # noqa: E402


FORMAT_VERSION = "wam.multimodal.m0.acceptance/2"
CONFIG_VERSION = "wam.multimodal.m0/2"
MIN_FORMAL_DATASET_EPISODES = 2_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/wam_multimodal/m0_data.yaml",
    )
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--benchmark-report", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--legacy-episodes-per-suite",
        type=int,
        help="Diagnostic override; requires --report outside the canonical path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = _load_config(config_path)
    canonical_config = (ROOT / "configs/wam_multimodal/m0_data.yaml").resolve()
    canonical_audit = _root_path(config["audit"]["report"]).resolve()
    canonical_benchmark = (
        _root_path(config["benchmark"]["output_directory"])
        / "visual_required_benchmark.json"
    ).resolve()
    canonical_checkpoint = _root_path(
        config["legacy_regression"]["checkpoint"]
    ).resolve()
    canonical_report = _root_path(config["acceptance"]["report"]).resolve()
    audit_path = (args.audit_report or canonical_audit).resolve()
    benchmark_path = (args.benchmark_report or canonical_benchmark).resolve()
    checkpoint = (args.checkpoint or canonical_checkpoint).resolve()
    report_path = (args.report or canonical_report).resolve()
    configured_episodes = int(config["legacy_regression"]["episodes_per_suite"])
    episodes = args.legacy_episodes_per_suite or configured_episodes
    if episodes <= 0:
        raise ValueError("legacy episodes_per_suite must be positive")
    if args.legacy_episodes_per_suite is not None and report_path == canonical_report:
        raise ValueError("diagnostic legacy override requires --report")
    formal_protocol = bool(
        config_path == canonical_config
        and audit_path == canonical_audit
        and benchmark_path == canonical_benchmark
        and checkpoint == canonical_checkpoint
        and report_path == canonical_report
        and episodes == configured_episodes
        and _configured_dataset_episode_total(config) >= MIN_FORMAL_DATASET_EPISODES
        and configured_episodes >= 20
    )
    _require_new_file(report_path)
    audit = _load_json(audit_path)
    benchmark = _load_json(benchmark_path)
    legacy = run_legacy_regression(
        config,
        checkpoint=checkpoint,
        episodes_per_suite=episodes,
    )
    report = phase_m0_acceptance_report(
        config,
        config_path=config_path,
        audit=audit,
        audit_path=audit_path,
        benchmark=benchmark,
        benchmark_path=benchmark_path,
        legacy=legacy,
        formal_protocol=formal_protocol,
    )
    _atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        return 1
    return 0 if formal_protocol else 2


def phase_m0_acceptance_report(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    audit: Mapping[str, Any],
    audit_path: Path,
    benchmark: Mapping[str, Any],
    benchmark_path: Path,
    legacy: Mapping[str, Any],
    formal_protocol: bool,
) -> dict[str, Any]:
    """Pure aggregation surface shared by the CLI and boundary tests."""

    formal_protocol = bool(
        formal_protocol
        and _configured_dataset_episode_total(config) >= MIN_FORMAL_DATASET_EPISODES
        and int(_mapping(config, "legacy_regression")["episodes_per_suite"]) >= 20
        and int(legacy.get("episodes_per_suite", -1)) >= 20
    )
    config_sha256 = _sha256(config_path)
    audit_format = audit.get("format_version") == AUDIT_FORMAT_VERSION
    benchmark_format = benchmark.get("format_version") == VISUAL_FORMAT_VERSION
    config_matches = bool(
        audit.get("config_sha256") == config_sha256
        and benchmark.get("config_sha256") == config_sha256
    )
    report_paths_bound = _input_report_paths_bound(
        config,
        audit_path=audit_path,
        benchmark_path=benchmark_path,
    )
    audit_artifacts_valid = _audit_artifacts_valid(audit, config=config)
    benchmark_artifacts_valid = _benchmark_artifacts_valid(
        benchmark,
        config=config,
    )
    shared_mujoco_provenance = _shared_mujoco_provenance(audit, benchmark)
    requirements_enabled = _m0_v2_requirements_enabled(config)
    checks = {
        "formal_protocol": _check(
            formal_protocol,
            minimum_dataset_episodes=MIN_FORMAL_DATASET_EPISODES,
            configured_dataset_episodes=_configured_dataset_episode_total(config),
        ),
        "audit_format_and_gate_pass": _check(
            audit_format and audit.get("passed") is True,
            format_version=audit.get("format_version"),
            gate_passed=audit.get("passed"),
        ),
        "benchmark_format_and_gate_pass": _check(
            benchmark_format and benchmark.get("passed") is True,
            format_version=benchmark.get("format_version"),
            gate_passed=benchmark.get("passed"),
        ),
        "input_reports_are_formal": _check(
            audit.get("formal_protocol") is True
            and benchmark.get("formal_protocol") is True,
            audit_formal=audit.get("formal_protocol"),
            benchmark_formal=benchmark.get("formal_protocol"),
        ),
        "input_report_paths_bound_to_config": report_paths_bound,
        "config_fingerprints_match": _check(config_matches, expected=config_sha256),
        "audit_artifacts_hashed": audit_artifacts_valid,
        "benchmark_episode_records_hashed": benchmark_artifacts_valid,
        "m0_v2_requirements_enabled": requirements_enabled,
        "dataset_and_benchmark_share_mujoco_world": shared_mujoco_provenance,
        "legacy_checkpoint_strict_reload": _check(
            legacy.get("strict_reload", {}).get("passed") is True,
            strict_reload=legacy.get("strict_reload"),
        ),
        "legacy_checkpoint_matches_pre_m0_anchor": _check(
            legacy.get("checkpoint_matches_pre_m0_anchor") is True,
            expected=legacy.get("expected_checkpoint_tree_sha256"),
            actual=legacy.get("checkpoint_tree_sha256_before"),
        ),
        "legacy_checkpoint_tree_immutable": _check(
            legacy.get("checkpoint_tree_immutable") is True,
            before=legacy.get("checkpoint_tree_sha256_before"),
            after=legacy.get("checkpoint_tree_sha256_after"),
        ),
        "legacy_standard_and_challenge_pass": _check(
            legacy.get("passed") is True,
            suites=legacy.get("suites"),
        ),
    }
    passed = all(item["passed"] for item in checks.values())
    return {
        "format_version": FORMAT_VERSION,
        "gate": "M0",
        "formal_protocol": bool(formal_protocol),
        "passed": passed,
        "config": str(config_path),
        "config_sha256": config_sha256,
        "thresholds": _plain(_mapping(config, "acceptance")),
        "inputs": {
            "audit_report": str(audit_path),
            "audit_report_sha256": _sha256(audit_path),
            "benchmark_report": str(benchmark_path),
            "benchmark_report_sha256": _sha256(benchmark_path),
        },
        "data_audit": {
            "passed": audit.get("passed"),
            "episodes": audit.get("episodes"),
            "transitions": audit.get("transitions"),
            "metrics": audit.get("metrics"),
            "checks": audit.get("checks"),
            "camera_order": audit.get("camera_order"),
            "provenance": audit.get("provenance"),
        },
        "visual_required": benchmark.get("acceptance"),
        "legacy_regression": _plain(legacy),
        "checks": checks,
        "provenance": {
            "python": sys.version,
            "torch": torch.__version__,
            "mujoco": mujoco.__version__,
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "cuda_available": torch.cuda.is_available(),
            "device": "cpu",
            "dataset": audit.get("provenance"),
            "benchmark": benchmark.get("protocol"),
        },
    }


def run_legacy_regression(
    config: Mapping[str, Any],
    *,
    checkpoint: Path,
    episodes_per_suite: int,
) -> dict[str, Any]:
    """Evaluate only the self-contained final checkpoint; no initialization assets."""

    regression_cfg = _mapping(config, "legacy_regression")
    before = _checkpoint_tree(checkpoint)
    before_digest = _checkpoint_tree_digest(before)
    expected_checkpoint_digest = str(regression_cfg["expected_checkpoint_tree_sha256"])
    if not _is_sha256(expected_checkpoint_digest):
        raise ValueError("legacy expected_checkpoint_tree_sha256 is invalid")
    checkpoint_matches_anchor = before_digest == expected_checkpoint_digest
    world, flow, metadata = load_joint_wam_checkpoint(
        checkpoint,
        device="cpu",
        expected_schema_version=str(regression_cfg["expected_schema_version"]),
    )
    world_reload, flow_reload, _ = load_joint_wam_checkpoint(
        checkpoint,
        device="cpu",
        expected_schema_version=str(regression_cfg["expected_schema_version"]),
    )
    world_diff = _state_dict_max_abs_diff(world.state_dict(), world_reload.state_dict())
    flow_diff = _state_dict_max_abs_diff(flow.state_dict(), flow_reload.state_dict())
    strict_reload = {
        "passed": world_diff == 0.0 and flow_diff == 0.0,
        "world_model_max_abs_diff": world_diff,
        "action_flow_max_abs_diff": flow_diff,
    }
    del world_reload, flow_reload

    model_config = metadata["experiment_config"]
    if not isinstance(model_config, Mapping):
        raise ValueError("self-contained checkpoint config is missing")
    runtime = _mapping(model_config, "runtime")
    action_chunk = _mapping(model_config, "action_chunk")
    data = _mapping(model_config, "data")
    evaluation = _mapping(model_config, "evaluation")
    fixed_actions = {
        int(index): float(value)
        for index, value in runtime.get("fixed_actions", {}).items()
    }
    policy_config = JointWAMPolicyConfig(
        action_chunk=ActionChunkConfig(
            action_dim=int(data["action_dim"]),
            horizon=int(action_chunk["horizon"]),
            execution_steps=int(action_chunk["execution_steps"]),
            solver_steps=int(action_chunk["solver_steps"]),
            warm_start_mode=str(action_chunk["warm_start_mode"]),
        ),
        solver=str(action_chunk["solver"]),
        anchor_residual_scale=float(runtime["anchor_residual_scale"]),
        normalized_action_clip=float(runtime["normalized_action_clip"]),
        observation_residual_nrmse_max=float(runtime["observation_residual_nrmse_max"]),
        risk_veto=bool(runtime["risk_veto"]),
        max_failure_probability=float(runtime["max_failure_probability"]),
        max_predicted_robot_distance=float(runtime["max_predicted_robot_distance"]),
        max_action_ood=float(runtime["max_action_ood"]),
        action_ood_threshold=float(runtime["action_ood_threshold"]),
        latency_budget_ms=float(runtime["latency_budget_ms"]),
        fallback_enabled=False,
    )
    suite_settings = {
        "standard": {
            "seed_start": int(regression_cfg["standard_seed_start"]),
            "environment": {},
        },
        "challenge": {
            "seed_start": int(regression_cfg["challenge_seed_start"]),
            "environment": dict(evaluation["challenge_environment"]),
        },
    }
    suites: dict[str, Any] = {}
    minimum = float(regression_cfg["minimum_success_rate"])
    maximum_regression = float(regression_cfg["maximum_reference_regression"])
    references = _mapping(regression_cfg, "reference_success_rate")
    all_passed = strict_reload["passed"] and checkpoint_matches_anchor
    for suite, settings in suite_settings.items():
        env = TwoRobotCooperativeStopEnv(
            CooperativeStopEnvConfig(
                include_camera_images=False,
                **settings["environment"],
            )
        )
        try:
            policy = JointWAMPolicy(
                world,
                flow,
                config=policy_config,
                fixed_actions=fixed_actions,
            )
            records = []
            seeds = range(
                int(settings["seed_start"]),
                int(settings["seed_start"]) + episodes_per_suite,
            )
            for episode_index, seed in enumerate(seeds):
                _seed_rollout(seed)
                observer = ClosedLoopEpisodeObserver("joint_wam_direct", policy)
                summary = SimulationRunner(
                    env,
                    policy,
                    RunnerConfig(
                        expose_privileged_state_to_policy=False,
                        policy_observation_keys=("proprioception",),
                    ),
                ).run_episode(
                    seed=seed,
                    episode_index=episode_index,
                    randomize=True,
                    observers=(observer,),
                )
                records.append(observer.finish(summary))
            metrics = aggregate_closed_loop(records)
        finally:
            env.close()
        reference = float(references[suite])
        success_rate = float(metrics["success_rate"])
        regression = reference - success_rate
        passed = bool(
            int(metrics["episodes"]) == episodes_per_suite
            and success_rate >= minimum
            and regression <= maximum_regression
            and metrics["fallback_trigger_rate"] == 0.0
            and metrics["action_source_coverage"] == 1.0
            and metrics["all_actions_finite_and_bounded"] is True
            and metrics["privileged_state_leakage"] is False
        )
        suites[suite] = {
            "passed": passed,
            "episodes": metrics["episodes"],
            "successes": int(round(success_rate * int(metrics["episodes"]))),
            "success_rate": success_rate,
            "reference_success_rate": reference,
            "regression": regression,
            "fallback_trigger_rate": metrics["fallback_trigger_rate"],
            "action_source_coverage": metrics["action_source_coverage"],
            "all_actions_finite_and_bounded": metrics["all_actions_finite_and_bounded"],
            "privileged_state_leakage": metrics["privileged_state_leakage"],
        }
        all_passed = all_passed and passed

    after = _checkpoint_tree(checkpoint)
    after_digest = _checkpoint_tree_digest(after)
    immutable = before == after
    all_passed = all_passed and immutable
    return {
        "format_version": "wam.multimodal.m0.legacy_regression/1",
        "passed": bool(all_passed),
        "checkpoint": str(checkpoint),
        "checkpoint_is_self_contained": True,
        "initialization_assets_loaded": False,
        "episodes_per_suite": episodes_per_suite,
        "strict_reload": strict_reload,
        "checkpoint_matches_pre_m0_anchor": checkpoint_matches_anchor,
        "expected_checkpoint_tree_sha256": expected_checkpoint_digest,
        "checkpoint_tree_immutable": immutable,
        "checkpoint_tree_sha256_before": before_digest,
        "checkpoint_tree_sha256_after": after_digest,
        "checkpoint_tree_files_before": before,
        "checkpoint_tree_files_after": after,
        "suites": suites,
    }


def _input_report_paths_bound(
    config: Mapping[str, Any], *, audit_path: Path, benchmark_path: Path
) -> dict[str, Any]:
    expected_audit = _root_path(_mapping(config, "audit")["report"]).resolve()
    expected_benchmark = (
        _root_path(_mapping(config, "benchmark")["output_directory"])
        / "visual_required_benchmark.json"
    ).resolve()
    passed = bool(
        audit_path.resolve() == expected_audit
        and benchmark_path.resolve() == expected_benchmark
    )
    return _check(
        passed,
        audit_report=str(audit_path.resolve()),
        expected_audit_report=str(expected_audit),
        benchmark_report=str(benchmark_path.resolve()),
        expected_benchmark_report=str(expected_benchmark),
    )


def _audit_artifacts_valid(
    audit: Mapping[str, Any], *, config: Mapping[str, Any]
) -> dict[str, Any]:
    expected_data_dir = _root_path(_mapping(config, "dataset")["directory"]).resolve()
    expected_manifest = (expected_data_dir / "manifest.json").resolve()
    manifest = Path(str(audit.get("manifest", ""))).resolve()
    data_dir = Path(str(audit.get("data_dir", ""))).resolve()
    expected = audit.get("manifest_sha256")
    expected_episodes = _configured_dataset_episode_total(config)
    checks = audit.get("checks")
    required_checks = (
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
    checks_valid = bool(
        isinstance(checks, Mapping)
        and all(
            isinstance(checks.get(name), Mapping) and checks[name].get("passed") is True
            for name in required_checks
        )
    )
    cue_pair = audit.get("cue_pair_audit")
    cue_pair_pre_onset_valid = bool(
        isinstance(cue_pair, Mapping)
        and cue_pair.get("passed") is True
        and cue_pair.get("pre_onset_sequences_valid") is True
        and int(cue_pair.get("pre_onset_frame_comparisons", 0)) > 0
    )
    provenance_valid = _current_mujoco_provenance(
        audit.get("provenance"), config=config, require_sources=True
    )
    passed = bool(
        data_dir == expected_data_dir
        and manifest == expected_manifest
        and manifest.is_file()
        and isinstance(expected, str)
        and _sha256(manifest) == expected
        and int(audit.get("episodes", -1)) == expected_episodes
        and int(audit.get("audited_episodes", -1)) == expected_episodes
        and expected_episodes >= MIN_FORMAL_DATASET_EPISODES
        and audit.get("schema_version") == "wam.multimodal/1.1"
        and audit.get("camera_order") == list(CAMERA_ORDER)
        and checks_valid
        and cue_pair_pre_onset_valid
        and provenance_valid["passed"]
    )
    return _check(
        passed,
        data_dir=str(data_dir),
        expected_data_dir=str(expected_data_dir),
        manifest=str(manifest),
        expected_manifest=str(expected_manifest),
        expected_sha256=expected,
        episodes=audit.get("episodes"),
        audited_episodes=audit.get("audited_episodes"),
        expected_episodes=expected_episodes,
        required_checks_valid=checks_valid,
        cue_pair_pre_onset_valid=cue_pair_pre_onset_valid,
        provenance=provenance_valid,
    )


def _benchmark_artifacts_valid(
    benchmark: Mapping[str, Any], *, config: Mapping[str, Any]
) -> dict[str, Any]:
    benchmark_cfg = _mapping(config, "benchmark")
    expected_output = _root_path(benchmark_cfg["output_directory"]).resolve()
    expected_records = (expected_output / "visual_required_episodes.jsonl").resolve()
    output = Path(str(benchmark.get("output_directory", ""))).resolve()
    records = Path(str(benchmark.get("episode_records", ""))).resolve()
    expected = benchmark.get("episode_records_sha256")
    protocol = benchmark.get("protocol")
    if not isinstance(protocol, Mapping):
        return _check(False, reason="benchmark protocol is missing")
    seed_count = int(benchmark_cfg["physical_seeds_per_task"])
    seed_start = int(benchmark_cfg["physical_seed_start"])
    expected_seeds = list(range(seed_start, seed_start + seed_count))
    expected_cues = [int(value) for value in benchmark_cfg["cue_variants"]]
    tasks = protocol.get("tasks")
    expected_count = (
        len(VISUAL_REQUIRED_TASKS)
        * len(REQUIRED_POLICIES)
        * seed_count
        * len(expected_cues)
    )
    try:
        payloads = _load_jsonl(records) if records.is_file() else []
        recomputed = visual_required_acceptance(
            payloads,
            tasks=tasks if isinstance(tasks, list) else (),
            physical_seeds=expected_seeds,
            cue_variants=expected_cues,
            thresholds=_mapping(config, "acceptance"),
        )
        acceptance_matches = _plain(recomputed) == _plain(benchmark.get("acceptance"))
    except (KeyError, TypeError, ValueError) as exc:
        return _check(False, reason=f"{type(exc).__name__}: {exc}")
    protocol_provenance = _current_mujoco_provenance(
        protocol, config=config, require_sources=False
    )
    render_contract = recomputed.get("render_evidence_contract")
    render_evidence_valid = bool(
        isinstance(render_contract, Mapping)
        and render_contract.get("passed") is True
        and render_contract.get("camera_order") == list(CAMERA_ORDER)
        and int(render_contract.get("records", -1)) == expected_count
        and render_contract.get("mujoco_versions") == [protocol.get("mujoco_version")]
        and render_contract.get("mujoco_gl") == [protocol.get("mujoco_gl")]
        and render_contract.get("model_xml_sha256")
        == [protocol.get("model_xml_sha256")]
        and render_contract.get("record_provenance_consistent") is True
    )
    protocol_matches = bool(
        isinstance(tasks, list)
        and tasks == list(VISUAL_REQUIRED_TASKS)
        and protocol.get("physical_seeds") == expected_seeds
        and protocol.get("cue_variants") == expected_cues
        and protocol.get("policies") == list(REQUIRED_POLICIES)
        and int(protocol.get("physical_seeds_per_task", -1)) == seed_count
        and protocol.get("camera_order") == list(CAMERA_ORDER)
        and protocol.get("render_requests") == list(CAMERA_ORDER)
        and protocol.get("policy_rgb_stream") == "fixed"
        and protocol.get("record_all_view_evidence") is True
        and protocol.get("raw_unannotated") is True
        and protocol_provenance["passed"]
    )
    passed = bool(
        output == expected_output
        and records == expected_records
        and records.is_file()
        and isinstance(expected, str)
        and _sha256(records) == expected
        and len(payloads) == expected_count
        and protocol_matches
        and acceptance_matches
        and recomputed.get("passed") is True
        and render_evidence_valid
    )
    return _check(
        passed,
        output_directory=str(output),
        expected_output_directory=str(expected_output),
        records=str(records),
        expected_records=str(expected_records),
        expected_sha256=expected,
        records_observed=len(payloads),
        records_expected=expected_count,
        protocol_matches=protocol_matches,
        acceptance_recomputed_matches=acceptance_matches,
        render_evidence_valid=render_evidence_valid,
        provenance=protocol_provenance,
    )


def _m0_v2_requirements_enabled(config: Mapping[str, Any]) -> dict[str, Any]:
    acceptance = _mapping(config, "acceptance")
    required_flags = (
        "require_mujoco_renderer_provenance",
        "require_three_camera_dataset",
        "require_three_camera_benchmark",
        "require_cross_camera_sync",
        "require_dynamic_robot_camera_extrinsics",
        "require_raw_unannotated_rgb",
        "require_visual_signal_onset_evidence",
        "require_visual_event_signal_semantic_isolation",
        "require_brake_lights_action_causal_and_red_only",
    )
    disabled = [name for name in required_flags if acceptance.get(name) is not True]
    camera = _mapping(config, "camera")
    canonical = bool(
        _mapping(config, "dataset").get("schema_version") == "wam.multimodal/1.1"
        and _mapping(config, "dataset").get("camera_order") == list(CAMERA_ORDER)
        and camera.get("camera_order") == list(CAMERA_ORDER)
        and camera.get("renderer_backend") == "mujoco.Renderer"
        and camera.get("geometry_source") == "mujoco_xml"
        and camera.get("raw_unannotated") is True
        and camera.get("calibration_convention")
        == "opencv_optical_camera_pose_in_world"
    )
    return _check(
        not disabled and canonical, disabled_flags=disabled, canonical=canonical
    )


def _shared_mujoco_provenance(
    audit: Mapping[str, Any], benchmark: Mapping[str, Any]
) -> dict[str, Any]:
    dataset = audit.get("provenance")
    protocol = benchmark.get("protocol")
    if not isinstance(dataset, Mapping) or not isinstance(protocol, Mapping):
        return _check(False, reason="dataset/benchmark MuJoCo provenance missing")
    fields = (
        "renderer_backend",
        "geometry_source",
        "mujoco_version",
        "mujoco_gl",
        "model_xml_path",
        "model_xml_sha256",
        "camera_rig",
    )
    mismatches = [
        name
        for name in fields
        if _plain(dataset.get(name)) != _plain(protocol.get(name))
    ]
    return _check(not mismatches, mismatches=mismatches)


def _current_mujoco_provenance(
    value: Any,
    *,
    config: Mapping[str, Any],
    require_sources: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return _check(False, reason="MuJoCo provenance is missing")
    try:
        xml_path = _repository_file(str(value.get("model_xml_path", "")))
        xml_valid = str(value.get("model_xml_sha256", "")) == _sha256(xml_path)
        rig = value.get("camera_rig")
        if not isinstance(rig, Mapping) or tuple(rig) != CAMERA_ORDER:
            raise ValueError("camera rig does not exactly cover canonical cameras")
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        rig_valid = True
        parent_names: dict[str, str] = {}
        camera_ids: list[int] = []
        for camera in CAMERA_ORDER:
            values = _mapping(rig, camera)
            camera_id = int(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            )
            if camera_id < 0:
                rig_valid = False
                continue
            body_id = int(model.cam_bodyid[camera_id])
            body_name = str(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            )
            camera_ids.append(camera_id)
            parent_names[camera] = body_name
            rig_valid = bool(
                rig_valid
                and int(values.get("camera_id", -1)) == camera_id
                and int(values.get("parent_body_id", -1)) == body_id
                and values.get("parent_body_name") == body_name
                and values.get("convention") == "opencv_optical_camera_pose_in_world"
                and np.isfinite(float(values.get("fovy_degrees", np.nan)))
                and 0.0 < float(values.get("fovy_degrees", np.nan)) < 180.0
            )
        rig_valid = bool(
            rig_valid
            and len(set(camera_ids)) == len(CAMERA_ORDER)
            and parent_names.get("fixed") == "world"
            and all(
                parent_names.get(camera, "") not in {"", "world"}
                for camera in CAMERA_ORDER[1:]
            )
            and len({parent_names.get(camera) for camera in CAMERA_ORDER[1:]})
            == len(CAMERA_ORDER) - 1
        )
        source_valid = not require_sources
        if require_sources:
            configured = tuple(
                Path(str(path)).as_posix()
                for path in _mapping(config, "audit")["required_source_files"]
            )
            source_sha256 = value.get("source_sha256")
            source_valid = bool(
                isinstance(source_sha256, Mapping)
                and set(source_sha256) == set(configured)
                and all(
                    str(source_sha256[path]) == _sha256(_repository_file(path))
                    for path in configured
                )
            )
        core_valid = bool(
            value.get("renderer_backend") == "mujoco.Renderer"
            and value.get("geometry_source") == "mujoco_xml"
            and value.get("mujoco_version") == mujoco.__version__
            and isinstance(value.get("mujoco_gl"), str)
            and bool(value.get("mujoco_gl"))
        )
        return _check(
            core_valid and xml_valid and rig_valid and source_valid,
            core_valid=core_valid,
            xml_valid=xml_valid,
            rig_valid=rig_valid,
            source_valid=source_valid,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return _check(False, reason=f"{type(exc).__name__}: {exc}")


def _checkpoint_tree(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"checkpoint tree cannot contain symlinks: {path}")
        if path.is_file():
            result[str(path.relative_to(root))] = _sha256(path)
    if not result:
        raise ValueError("checkpoint tree is empty")
    return result


def _checkpoint_tree_digest(tree: Mapping[str, str]) -> str:
    serialized = json.dumps(
        dict(sorted(tree.items())),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _state_dict_max_abs_diff(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> float:
    if set(left) != set(right):
        raise ValueError("strict reload state-dict keys differ")
    maximum = 0.0
    for name in left:
        if (
            left[name].shape != right[name].shape
            or left[name].dtype != right[name].dtype
        ):
            raise ValueError(f"strict reload tensor contract differs: {name}")
        if left[name].numel():
            left_tensor = left[name].detach().cpu()
            right_tensor = right[name].detach().cpu()
            if left_tensor.dtype == torch.bool:
                difference = 0.0 if torch.equal(left_tensor, right_tensor) else 1.0
            else:
                difference = float((left_tensor - right_tensor).abs().max().item())
            maximum = max(maximum, difference)
    return maximum


def _seed_rollout(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != CONFIG_VERSION:
        raise ValueError("unsupported Phase M0 config")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be a mapping: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at line {line_number}")
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL record {line_number} is not a mapping")
            payloads.append(payload)
    return payloads


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


def _require_new_file(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite stale acceptance report: {path}")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    if temporary.exists():
        raise FileExistsError(f"stale partial acceptance report: {temporary}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _check(passed: bool, **details: Any) -> dict[str, Any]:
    details.pop("passed", None)
    return {"passed": bool(passed), **details}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
