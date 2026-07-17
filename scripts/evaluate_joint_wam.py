"""Run the formal 500-seed Joint WAM evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Callable, Mapping, Sequence

# Headless video rendering is part of the formal protocol.  This must be set
# before importing the MuJoCo-backed environment modules.
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.trajectory import PROPRIO_WAM_SCHEMA_VERSION  # noqa: E402
from envs.annotations import annotate_cooperative_stop_frame  # noqa: E402
from envs.runtime import RenderRequest, RunnerConfig, SimulationRunner  # noqa: E402
from envs.two_robot_carry_env import (  # noqa: E402
    CooperativeStopEnvConfig,
    TwoRobotCooperativeStopEnv,
)
from envs.video import StreamingVideoObserver  # noqa: E402
from eval.closed_loop import (  # noqa: E402
    ClosedLoopEpisode,
    ClosedLoopEpisodeObserver,
    aggregate_closed_loop,
    episode_to_dict,
    paired_policy_statistics,
)
from eval.joint_wam import (  # noqa: E402
    joint_wam_acceptance_report,
    select_joint_wam_video_seeds,
    validate_video_evidence,
)
from models.wam import ActionChunkConfig  # noqa: E402
from policies import (  # noqa: E402
    ActionPriorPolicy,
    JointWAMPolicyConfig,
    JointWAMPolicy,
)
from train.action_prior import load_action_prior_checkpoint  # noqa: E402
from train.joint_wam_checkpointing import (  # noqa: E402
    load_joint_wam_checkpoint,
)
from train.progress import TrainingProgress  # noqa: E402
from train.rwm_u_checkpointing import load_rwm_u_member_checkpoint  # noqa: E402


FORMAL_SUITES = ("standard", "challenge")
DIRECT_POLICY = "joint_wam_direct"
FALLBACK_POLICY = "joint_wam_with_fallback"
REQUIRED_POLICIES = (
    DIRECT_POLICY,
    "action_prior",
    "stationary",
    "scripted_oracle",
)
ALL_POLICIES = (*REQUIRED_POLICIES, FALLBACK_POLICY)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/wam/joint_wam.yaml",
    )
    parser.add_argument("--world-model-checkpoint-dir", type=Path)
    parser.add_argument("--action-prior-checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--standard-seed-start", type=int)
    parser.add_argument("--challenge-seed-start", type=int)
    parser.add_argument("--policies", nargs="+", choices=ALL_POLICIES)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def _settings(config: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    evaluation = _mapping(config, "evaluation")
    initialization = _mapping(config, "initialization")
    fallback = _mapping(config, "fallback_deployment")
    formal_count = int(evaluation["episodes_per_suite"])
    count = formal_count if args.episodes is None else int(args.episodes)
    if count <= 0:
        raise ValueError("episodes must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max-steps must be positive")
    starts = {
        "standard": int(
            evaluation["standard_seed_start"]
            if args.standard_seed_start is None
            else args.standard_seed_start
        ),
        "challenge": int(
            evaluation["challenge_seed_start"]
            if args.challenge_seed_start is None
            else args.challenge_seed_start
        ),
    }
    policies = tuple(
        str(value)
        for value in (
            list(evaluation["policies"]) + [fallback["policy"]]
            if args.policies is None
            else args.policies
        )
    )
    if len(set(policies)) != len(policies):
        raise ValueError("evaluation policies must be unique")
    if not set(policies) <= set(ALL_POLICIES):
        raise ValueError("evaluation requests an unsupported policy")

    has_diagnostic_override = any(
        value is not None
        for value in (
            args.episodes,
            args.standard_seed_start,
            args.challenge_seed_start,
            args.policies,
            args.max_steps,
            args.output_dir,
            args.world_model_checkpoint_dir,
            args.action_prior_checkpoint_dir,
            args.checkpoint_dir,
        )
    ) or bool(args.skip_videos)
    if has_diagnostic_override and args.output_dir is None:
        raise ValueError(
            "evaluation overrides require a separate --output-dir"
        )
    formal_protocol = not has_diagnostic_override
    suite_seeds = {
        suite: tuple(range(starts[suite], starts[suite] + count))
        for suite in FORMAL_SUITES
    }
    if set(suite_seeds["standard"]) & set(suite_seeds["challenge"]):
        raise ValueError("standard and challenge evaluation seeds must be disjoint")

    output = (
        args.output_dir or ROOT / str(evaluation["output_directory"])
    ).resolve()
    return {
        "config_path": args.config.resolve(),
        "world_model_checkpoint": (
            args.world_model_checkpoint_dir
            or ROOT / str(initialization["world_model_checkpoint"])
        ).resolve(),
        "action_prior_checkpoint": (
            args.action_prior_checkpoint_dir
            or ROOT / str(initialization["action_prior_checkpoint"])
        ).resolve(),
        "checkpoint": (
            args.checkpoint_dir or ROOT / str(config["checkpoint"]["directory"])
        ).resolve(),
        "output_dir": output,
        "episodes": count,
        "minimum_episodes": formal_count,
        "suite_seeds": suite_seeds,
        "policies": policies,
        "max_steps": args.max_steps,
        "render_videos": not args.skip_videos,
        "formal_protocol": formal_protocol,
    }


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Joint WAM config field {key!r} must be a mapping")
    return value


class _NamedCallablePolicy:
    """Give simple baselines the same auditable action-source diagnostics."""

    def __init__(
        self, name: str, function: Callable[[Mapping[str, Any]], np.ndarray]
    ) -> None:
        self.name = str(name)
        self.function = function
        self.last_diagnostics: dict[str, Any] = {}

    def reset(self) -> None:
        self.last_diagnostics = {}

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        keys = {str(key) for key in observation}
        if "privileged_state" in observation:
            raise RuntimeError(f"privileged_state leakage into {self.name}")
        action = np.asarray(self.function(observation), dtype=np.float32)
        self.last_diagnostics = {
            "executed_mode": self.name,
            "planned_mode": "none",
            "plan_executed": False,
            "deadline_exceeded": False,
            "fallback_reason": "none",
            "observation_keys": sorted(keys),
            "privileged_state_seen": False,
        }
        return action


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_yaml(args.config)
    _validate_evaluation_config(config)
    settings = _settings(config, args)
    model_config = config

    device = _device(args.device)
    _configure_torch(config, device)
    source_before = _source_fingerprints(settings)
    joint, flow, checkpoint_metadata = load_joint_wam_checkpoint(
        settings["checkpoint"],
        device=device,
        expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION,
    )
    _validate_checkpoint_config(model_config, checkpoint_metadata["experiment_config"])
    strict_joint_evidence = _strict_joint_evidence(checkpoint_metadata["metrics"])
    if not strict_joint_evidence["passed"]:
        failed = [
            name
            for name, passed in strict_joint_evidence["checks"].items()
            if not passed
        ]
        raise RuntimeError(f"strict Joint WAM evidence failed: {failed}")

    frozen_teacher, world_model_metadata = load_rwm_u_member_checkpoint(
        settings["world_model_checkpoint"],
        0,
        device=device,
        expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION,
    )
    action_prior, action_prior_metadata = load_action_prior_checkpoint(
        settings["action_prior_checkpoint"],
        world_model_checkpoint=settings["world_model_checkpoint"],
        device=device,
        expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION,
        expected_normalization_sha256=world_model_metadata["normalization"].sha256(),
    )
    _require_identical_state_dicts(
        action_prior,
        flow.anchor_prior,
        message="evaluation action prior differs from the frozen anchor",
    )

    warmup_metrics = checkpoint_metadata["metrics"].get("action_flow_warmup")
    if not isinstance(warmup_metrics, Mapping):
        raise ValueError("Joint WAM metrics have no action-flow warm-up evidence")
    training_seeds, training_seed_evidence = _training_seed_audit(
        checkpoint_metadata["dataset_manifest"],
        world_model_manifest=_load_json(
            settings["world_model_checkpoint"] / "dataset_manifest.json"
        ),
        action_flow_metrics=warmup_metrics,
        action_prior_manifest=action_prior_metadata["dataset_manifest"],
    )
    evaluation_seeds = {
        int(seed)
        for suite_seeds in settings["suite_seeds"].values()
        for seed in suite_seeds
    }
    overlap_seeds = sorted(training_seeds & evaluation_seeds)
    held_out_overlap = len(overlap_seeds)
    training_seed_evidence = {
        **training_seed_evidence,
        "evaluation_unique_seed_count": len(evaluation_seeds),
        "held_out_overlap_count": held_out_overlap,
        "held_out_overlap_seeds": overlap_seeds,
        "passed": held_out_overlap == 0,
    }

    output = settings["output_dir"]
    _prepare_output_directory(output)
    records_by_suite = _evaluate_all_policies(
        config,
        model_config,
        settings,
        joint=joint,
        flow=flow,
        frozen_teacher=frozen_teacher,
        action_prior=action_prior,
        progress_enabled=not args.no_progress,
    )
    metrics = {
        suite: {
            policy: aggregate_closed_loop(records)
            for policy, records in policies.items()
        }
        for suite, policies in records_by_suite.items()
    }
    paired = {
        suite: paired_policy_statistics(
            records_by_suite[suite][DIRECT_POLICY],
            records_by_suite[suite]["action_prior"],
            seed=73,
        )
        for suite in FORMAL_SUITES
        if DIRECT_POLICY in records_by_suite[suite]
        and "action_prior" in records_by_suite[suite]
    }

    selection: Mapping[str, Any] = {
        "format_version": "wam.joint_wam.video_selection/1",
        "selected": (),
    }
    video_evidence: list[dict[str, Any]] = []
    video_validation: dict[str, Any] = {
        "passed": False,
        "checks": {"videos_rendered": False},
    }
    if settings["render_videos"] and DIRECT_POLICY in settings["policies"]:
        video_config = _mapping(config, "video")
        selection = select_joint_wam_video_seeds(
            records_by_suite,
            success_per_suite=int(video_config["success_per_suite"]),
            failure_global_max=int(video_config["failure_global_max"]),
        )
        video_evidence = _render_video_evidence(
            selection,
            output=output,
            evaluation_config=config,
            model_config=model_config,
            joint=joint,
            flow=flow,
        )
        video_validation = validate_video_evidence(selection, video_evidence)
        directory_validation = _validate_video_directory(output, video_evidence)
        video_validation["checks"]["video_directory_exact"] = directory_validation[
            "passed"
        ]
        video_validation["directory"] = directory_validation
        video_validation["passed"] = bool(
            video_validation["passed"] and directory_validation["passed"]
        )

    source_after = _source_fingerprints(settings)
    source_immutable = source_before == source_after
    checkpoint_reload = checkpoint_metadata["metrics"].get("checkpoint_reload", {})
    strict_reload_max_abs_diff = checkpoint_reload.get("max_abs_diff")
    limits = _mapping(config, "acceptance")
    acceptance = joint_wam_acceptance_report(
        metrics,
        records_by_suite=records_by_suite,
        held_out_seed_overlap=held_out_overlap,
        strict_joint_evidence=strict_joint_evidence,
        source_checkpoints_immutable=source_immutable,
        strict_reload_max_abs_diff=strict_reload_max_abs_diff,
        required_videos_complete=bool(video_validation.get("passed", False)),
        minimum_episodes=int(limits["minimum_episodes_per_suite"]),
        minimum_success_rate=float(limits["minimum_direct_success_rate"]),
        maximum_prior_regression=float(
            limits["maximum_prior_success_regression"]
        ),
        formal_protocol=bool(settings["formal_protocol"]),
    )

    report = {
        "format_version": "wam.joint_wam.closed_loop/1",
        "model": "joint_wam",
        "protocol": "paired_500_seed",
        "formal_protocol": bool(settings["formal_protocol"]),
        "policy_acceptable": bool(acceptance["policy_acceptable"]),
        "fallback_enabled_for_acceptance": False,
        "joint_benefit": acceptance["joint_benefit"],
        "config": str(settings["config_path"]),
        "config_sha256": _sha256(settings["config_path"]),
        "world_model_checkpoint": str(settings["world_model_checkpoint"]),
        "action_prior_checkpoint": str(settings["action_prior_checkpoint"]),
        "checkpoint": str(settings["checkpoint"]),
        "suite_seeds": {
            name: list(values) for name, values in settings["suite_seeds"].items()
        },
        "policy_contract": {
            DIRECT_POLICY: {"fallback_enabled": False, "blocking": True},
            FALLBACK_POLICY: {"fallback_enabled": True, "blocking": False},
            "scripted_oracle": {"uses_environment_oracle": True, "blocking": False},
        },
        "metrics": metrics,
        "paired_direct_vs_prior": paired,
        "offline_world_metrics": _offline_world_metrics(checkpoint_metadata["metrics"]),
        "strict_joint_evidence": strict_joint_evidence,
        "training_seed_count": len(training_seeds),
        "evaluation_seed_count": len(evaluation_seeds),
        "training_seed_evidence": training_seed_evidence,
        "held_out_seed_overlap": held_out_overlap,
        "source_fingerprints_before": source_before,
        "source_fingerprints_after": source_after,
        "source_checkpoints_immutable": source_immutable,
        "video_selection": _selection_to_json(selection),
        "video_evidence": video_evidence,
        "video_validation": video_validation,
        "acceptance": acceptance,
        "runtime": _runtime_metadata(config, device),
    }
    _write_outputs(output, report, records_by_suite)
    print(
        json.dumps(
            {
                "output": str(output),
                "policy_acceptable": report["policy_acceptable"],
                "acceptance_checks": acceptance["checks"],
                "suite_success": {
                    suite: {
                        policy: values["success_rate"]
                        for policy, values in policies.items()
                    }
                    for suite, policies in metrics.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["policy_acceptable"] else 2


def _evaluate_all_policies(
    evaluation_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    settings: Mapping[str, Any],
    *,
    joint: Any,
    flow: Any,
    frozen_teacher: Any,
    action_prior: Any,
    progress_enabled: bool,
) -> dict[str, dict[str, list[ClosedLoopEpisode]]]:
    result: dict[str, dict[str, list[ClosedLoopEpisode]]] = {}
    total_stages = len(FORMAL_SUITES) * len(settings["policies"])
    with TrainingProgress(
        enabled=progress_enabled, total_stages=total_stages
    ) as progress:
        for suite in FORMAL_SUITES:
            suite_records: dict[str, list[ClosedLoopEpisode]] = {}
            for policy_name in settings["policies"]:
                seeds = settings["suite_seeds"][suite]
                phase = progress.add_phase(
                    f"{suite} {policy_name}", len(seeds)
                )
                env = _environment(evaluation_config, model_config, suite)
                try:
                    policy = _policy(
                        policy_name,
                        env,
                        model_config,
                        joint=joint,
                        flow=flow,
                        frozen_teacher=frozen_teacher,
                        action_prior=action_prior,
                    )
                    records: list[ClosedLoopEpisode] = []
                    for episode_index, seed in enumerate(seeds):
                        _seed_rollout(int(seed))
                        observer = ClosedLoopEpisodeObserver(policy_name, policy)
                        summary = SimulationRunner(
                            env,
                            policy,
                            RunnerConfig(
                                max_steps=settings["max_steps"],
                                expose_privileged_state_to_policy=False,
                            ),
                        ).run_episode(
                            seed=int(seed),
                            episode_index=episode_index,
                            randomize=bool(
                                _mapping(evaluation_config, "evaluation")["randomize"]
                            ),
                            observers=(observer,),
                        )
                        record = observer.finish(summary)
                        records.append(record)
                        phase.advance(
                            {
                                "episode": episode_index + 1,
                                "episodes": len(seeds),
                                "success": int(record.success),
                            }
                        )
                    suite_records[policy_name] = records
                    phase.finish(
                        f"success {np.mean([record.success for record in records]):.1%}"
                    )
                finally:
                    env.close()
            result[suite] = suite_records
    return result


def _policy(
    name: str,
    env: TwoRobotCooperativeStopEnv,
    model_config: Mapping[str, Any],
    *,
    joint: Any,
    flow: Any,
    frozen_teacher: Any,
    action_prior: Any,
) -> Any:
    runtime = _mapping(model_config, "runtime")
    fixed_actions = {
        int(index): float(value)
        for index, value in runtime.get("fixed_actions", {}).items()
    }
    if name in {DIRECT_POLICY, FALLBACK_POLICY}:
        policy_config = _joint_wam_policy_config(
            model_config, fallback_enabled=name == FALLBACK_POLICY
        )
        return JointWAMPolicy(
            joint,
            flow,
            config=policy_config,
            fallback_world_model=(
                frozen_teacher if name == FALLBACK_POLICY else None
            ),
            fallback_prior=action_prior if name == FALLBACK_POLICY else None,
            fixed_actions=fixed_actions,
        )
    if name == "action_prior":
        return ActionPriorPolicy(
            frozen_teacher, action_prior, fixed_actions=fixed_actions
        )
    if name == "stationary":
        action = np.zeros(env.action_dim, dtype=np.float32)
        for index, value in fixed_actions.items():
            action[index] = value
        return _NamedCallablePolicy(
            name, lambda observation, value=action: value.copy()
        )
    if name == "scripted_oracle":
        return _NamedCallablePolicy(name, lambda observation: env.scripted_action())
    raise ValueError(f"unsupported evaluation policy {name!r}")


def _joint_wam_policy_config(
    model_config: Mapping[str, Any], *, fallback_enabled: bool
) -> JointWAMPolicyConfig:
    runtime = _mapping(model_config, "runtime")
    chunk = _mapping(model_config, "action_chunk")
    data = _mapping(model_config, "data")
    return JointWAMPolicyConfig(
        action_chunk=ActionChunkConfig(
            action_dim=int(data["action_dim"]),
            horizon=int(chunk["horizon"]),
            execution_steps=int(chunk["execution_steps"]),
            solver_steps=int(chunk["solver_steps"]),
            warm_start_mode=str(chunk["warm_start_mode"]),
        ),
        solver=str(chunk["solver"]),
        anchor_residual_scale=float(runtime["anchor_residual_scale"]),
        normalized_action_clip=float(runtime["normalized_action_clip"]),
        observation_residual_nrmse_max=float(
            runtime["observation_residual_nrmse_max"]
        ),
        risk_veto=bool(runtime["risk_veto"]),
        max_failure_probability=float(runtime["max_failure_probability"]),
        max_predicted_robot_distance=float(
            runtime["max_predicted_robot_distance"]
        ),
        max_action_ood=float(runtime["max_action_ood"]),
        action_ood_threshold=float(runtime["action_ood_threshold"]),
        latency_budget_ms=float(runtime["latency_budget_ms"]),
        fallback_enabled=fallback_enabled,
    )


def _environment(
    evaluation_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    suite: str,
) -> TwoRobotCooperativeStopEnv:
    del model_config
    evaluation = _mapping(evaluation_config, "evaluation")
    overrides = (
        {} if suite == "standard" else dict(evaluation["challenge_environment"])
    )
    return TwoRobotCooperativeStopEnv(
        CooperativeStopEnvConfig(include_camera_images=False, **overrides)
    )


def _render_video_evidence(
    selection: Mapping[str, Any],
    *,
    output: Path,
    evaluation_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    joint: Any,
    flow: Any,
) -> list[dict[str, Any]]:
    video = _mapping(evaluation_config, "video")
    evidence: list[dict[str, Any]] = []
    for episode_index, item in enumerate(selection["selected"]):
        suite = str(item.suite)
        seed = int(item.seed)
        outcome = "success" if bool(item.success) else "failure"
        video_path = output / "videos" / suite / outcome / f"seed_{seed:06d}.mp4"
        sidecar_path = video_path.with_suffix(".json")
        env = _environment(evaluation_config, model_config, suite)
        try:
            policy = _policy(
                DIRECT_POLICY,
                env,
                model_config,
                joint=joint,
                flow=flow,
                frozen_teacher=None,
                action_prior=None,
            )
            observer = ClosedLoopEpisodeObserver(DIRECT_POLICY, policy)
            render_name = "evaluation"
            request = RenderRequest(
                render_name,
                str(video["camera"]),
                width=int(video["width"]),
                height=int(video["height"]),
                annotator=(
                    annotate_cooperative_stop_frame
                    if bool(video["annotated"])
                    else None
                ),
            )
            _seed_rollout(seed)
            with ExitStack() as stack:
                stream = stack.enter_context(
                    StreamingVideoObserver(
                        video_path,
                        stream=render_name,
                        fps=1.0 / env.control_dt,
                        codec=str(video["codec"]),
                        frame_getter=(
                            lambda transition, name=render_name: transition.next_images[
                                name
                            ]
                        ),
                    )
                )
                summary = SimulationRunner(
                    env,
                    policy,
                    RunnerConfig(
                        render=(request,), expose_privileged_state_to_policy=False
                    ),
                ).run_episode(
                    seed=seed,
                    episode_index=episode_index,
                    randomize=bool(
                        _mapping(evaluation_config, "evaluation")["randomize"]
                    ),
                    observers=(observer, stream),
                )
                frames_written = int(stream.frames_written)
            replay = observer.finish(summary)
        finally:
            env.close()
        video_probe = _probe_video(video_path)
        if (
            int(video_probe["frames"]) != frames_written
            or int(video_probe["width"]) != int(video["width"])
            or int(video_probe["height"]) != int(video["height"])
        ):
            raise RuntimeError(f"encoded evaluation video contract mismatch: {video_path}")
        source_counts = dict(sorted(Counter(replay.planner_modes).items()))
        replay_matches = bool(
            replay.success is bool(item.success)
            and replay.failure_reason == str(item.failure_reason)
        )
        sidecar = {
            "format_version": "wam.joint_wam.video/1",
            "suite": suite,
            "seed": seed,
            "policy": DIRECT_POLICY,
            "success": bool(replay.success),
            "failure": bool(replay.failure),
            "failure_reason": replay.failure_reason,
            "action_source": (
                next(iter(source_counts)) if len(source_counts) == 1 else "mixed"
            ),
            "action_source_counts": source_counts,
            "fallback_enabled": False,
            "fallback_used": any(
                name not in {DIRECT_POLICY, "joint_wam_flow"}
                for name in source_counts
            ),
            "replay_matches_evaluation": replay_matches,
            "steps": int(replay.steps),
            "frames_written": frames_written,
            "terminal_frame_included": True,
            "fps": 1.0 / env.control_dt,
            "width": int(video["width"]),
            "height": int(video["height"]),
            "camera": str(video["camera"]),
            "annotated": bool(video["annotated"]),
            "video_path": str(video_path),
            "sidecar_path": str(sidecar_path),
            "video_bytes": video_path.stat().st_size,
            "video_sha256": _sha256(video_path),
            "video_probe": video_probe,
        }
        _atomic_write_text(
            sidecar_path,
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        )
        evidence.append(sidecar)
    return evidence


def _probe_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    try:
        opened = bool(capture.isOpened())
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
        first_frame_read = bool(capture.read()[0]) if opened else False
    finally:
        capture.release()
    if not opened or frames <= 0 or not first_frame_read:
        raise RuntimeError(f"encoded evaluation video is unreadable: {path}")
    return {
        "opened": opened,
        "frames": frames,
        "width": width,
        "height": height,
        "first_frame_read": first_frame_read,
    }


def _prepare_output_directory(path: Path) -> None:
    """Reject stale formal evidence instead of silently mixing or deleting it."""

    if path.exists() and not path.is_dir():
        raise FileExistsError(f"output path is not a directory: {path}")
    existing = sorted(
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file()
    ) if path.exists() else []
    if existing:
        preview = ", ".join(existing[:5])
        raise FileExistsError(
            "output directory contains stale evidence; use a fresh diagnostic "
            f"directory or archive the formal result first: {preview}"
        )
    path.mkdir(parents=True, exist_ok=True)


def _validate_video_directory(
    output: Path, evidence: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    video_root = output / "videos"
    expected = {
        Path(str(item[key])).resolve()
        for item in evidence
        for key in ("video_path", "sidecar_path")
    }
    observed = (
        {item.resolve() for item in video_root.rglob("*") if item.is_file()}
        if video_root.is_dir()
        else set()
    )
    return {
        "passed": observed == expected,
        "expected_files": len(expected),
        "observed_files": len(observed),
        "missing": sorted(str(path) for path in expected - observed),
        "unexpected": sorted(str(path) for path in observed - expected),
    }


def _strict_joint_evidence(metrics: Mapping[str, Any]) -> dict[str, Any]:
    positive = {
        "member_0_parameter_delta_nonzero": "member_0_parameter_delta",
        "shared_history_parameter_delta_nonzero": (
            "shared_history_parameter_delta"
        ),
        "world_parameter_delta_nonzero": "world_parameter_delta",
        "action_flow_parameter_delta_nonzero": "action_flow_parameter_delta",
    }
    gradients = metrics.get("branch_gradient_maxima", {})
    required_gradients = (
        "action_to_flow_gradient_norm",
        "action_to_backbone_gradient_norm",
        "world_to_backbone_gradient_norm",
        "consistency_to_flow_gradient_norm",
        "consistency_to_backbone_gradient_norm",
    )
    reload_metrics = metrics.get("checkpoint_reload", {})
    checks = {
        name: _positive_finite(metrics.get(key)) for name, key in positive.items()
    }
    checks.update(
        {
            name.replace("_norm", "_nonzero"): bool(
                isinstance(gradients, Mapping)
                and _positive_finite(gradients.get(name))
            )
            for name in required_gradients
        }
    )
    checks.update(
        {
            "anchor_prior_immutable": _exact_zero(
                metrics.get("anchor_prior_parameter_delta")
            ),
            "frozen_teacher_immutable": _exact_zero(
                metrics.get("frozen_teacher_parameter_delta")
            ),
            "source_checkpoints_immutable": metrics.get(
                "source_checkpoints_immutable"
            )
            is True,
            "strict_checkpoint_reload_exact": bool(
                isinstance(reload_metrics, Mapping)
                and reload_metrics.get("strict") is True
                and _exact_zero(reload_metrics.get("max_abs_diff"))
            ),
            "formal_run": metrics.get("formal_run") is True,
            "offline_audit_passed": metrics.get("passed") is True
            and bool(metrics.get("offline_acceptance", {}).get("passed", False)),
        }
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "parameter_deltas": {
            key: metrics.get(key) for key in positive.values()
        },
        "branch_gradient_maxima": dict(gradients),
        "checkpoint_reload": dict(reload_metrics),
    }


def _offline_world_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    splits = metrics.get("offline_acceptance", {}).get("splits", {})
    result: dict[str, Any] = {}
    if isinstance(splits, Mapping):
        for name, values in splits.items():
            if isinstance(values, Mapping):
                result[str(name)] = dict(values.get("offline_summary", {}))
    return result


def _validate_evaluation_config(config: Mapping[str, Any]) -> None:
    if config.get("name") != "wam.cooperative_stop/joint-wam":
        raise ValueError("unsupported Joint WAM config")
    evaluation = _mapping(config, "evaluation")
    fallback = _mapping(config, "fallback_deployment")
    video = _mapping(config, "video")
    if int(evaluation["episodes_per_suite"]) != 500:
        raise ValueError("formal validation requires 500 episodes per suite")
    if tuple(evaluation["policies"]) != REQUIRED_POLICIES:
        raise ValueError("formal validation requires all four baseline policies")
    if str(fallback["policy"]) != FALLBACK_POLICY or not bool(
        fallback["fallback_enabled"]
    ):
        raise ValueError("fallback deployment contract is invalid")
    if str(video["source_policy"]) != DIRECT_POLICY or bool(
        video["fallback_enabled"]
    ):
        raise ValueError("videos must use the direct no-fallback policy")
    if str(video["selection"]) != "sorted_smallest_seed":
        raise ValueError("video selection must be deterministic")

    chunk = _mapping(config, "action_chunk")
    expected = {
        "horizon": 8,
        "execution_steps": 2,
        "solver_steps": 4,
        "solver": "euler",
        "expert_count": 1,
        "warm_start_mode": "shift_repeat_last",
        "warm_start_shift_source": "actual_executed_steps",
    }
    if _canonical(chunk) != expected:
        raise ValueError("config does not satisfy the locked action-chunk contract")


def _validate_checkpoint_config(
    caller: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> None:
    for key, value in caller.items():
        if key not in checkpoint or _canonical(value) != _canonical(checkpoint[key]):
            raise ValueError(
                f"checkpoint config field {key!r} differs from the evaluation config"
            )


def _training_seeds(
    manifest: Mapping[str, Any],
    *,
    world_model_manifest: Mapping[str, Any],
    action_flow_metrics: Mapping[str, Any],
    action_prior_manifest: Mapping[str, Any],
) -> set[int]:
    """Return every ancestor training seed after a strict provenance audit."""

    seeds, _ = _training_seed_audit(
        manifest,
        world_model_manifest=world_model_manifest,
        action_flow_metrics=action_flow_metrics,
        action_prior_manifest=action_prior_manifest,
    )
    return seeds


def _training_seed_audit(
    manifest: Mapping[str, Any],
    *,
    world_model_manifest: Mapping[str, Any],
    action_flow_metrics: Mapping[str, Any],
    action_prior_manifest: Mapping[str, Any],
) -> tuple[set[int], dict[str, Any]]:
    """Prove the portable seed list covers every training source."""

    if manifest.get("smoke_subset") is not False:
        raise ValueError("training manifest must prove a non-smoke run")
    if action_prior_manifest.get("smoke_subset") is not False:
        raise ValueError("action-prior manifest must prove a non-smoke run")

    joint_partitions = _manifest_partitions(manifest, field="Joint WAM partitions")
    parent_partitions = {
        "world model": _manifest_partitions(
            world_model_manifest,
            field="world-model partitions",
        ),
        "action prior": _manifest_partitions(
            action_prior_manifest,
            field="action-prior partitions",
        ),
    }
    for name, partitions in parent_partitions.items():
        if partitions != joint_partitions:
            raise ValueError(f"Joint WAM partitions do not match {name}")

    result: set[int] = set()
    partition_seeds = manifest.get("partition_seeds")
    if not isinstance(partition_seeds, Mapping):
        raise ValueError("Joint WAM manifest requires portable partition_seeds")
    required_partitions = {"train", "validation", "test"}
    if set(partition_seeds) != required_partitions:
        raise ValueError("partition_seeds must contain train/validation/test")
    offline_counts: dict[str, int] = {}
    offline_union: set[int] = set()
    for name, values in partition_seeds.items():
        seeds = _manifest_seed_list(
            values,
            field=f"partition_seeds.{name}",
            allow_empty=False,
        )
        path_seeds = _partition_path_seeds(joint_partitions[name], field=name)
        if seeds != path_seeds:
            raise ValueError(
                f"portable seeds for partition {name} do not match its paths"
            )
        if offline_union & seeds:
            raise ValueError("offline partitions contain overlapping seeds")
        offline_union.update(seeds)
        offline_counts[str(name)] = len(seeds)
        result.update(seeds)

    inherited_action_flow = _manifest_seed_list(
        manifest.get("action_flow_on_policy_seeds"),
        field="action_flow_on_policy_seeds",
        allow_empty=False,
    )
    generated = _manifest_seed_list(
        manifest.get("generated_or_relabel_seeds"),
        field="generated_or_relabel_seeds",
        allow_empty=True,
    )
    on_policy_metrics = action_flow_metrics.get("on_policy_distillation")
    if not isinstance(on_policy_metrics, Mapping):
        raise ValueError("action-flow metrics have no on-policy seed evidence")
    metric_action_flow = _manifest_seed_list(
        on_policy_metrics.get("seeds"),
        field="action-flow metrics on-policy seeds",
        allow_empty=False,
    )
    if inherited_action_flow != metric_action_flow:
        raise ValueError(
            "action-flow seed list does not match the embedded warm-up metrics"
        )
    result.update(inherited_action_flow)
    result.update(generated)
    return result, {
        "format_version": "wam.joint_wam.training_seed_evidence/1",
        "offline_partition_counts": dict(sorted(offline_counts.items())),
        "offline_unique_seed_count": len(offline_union),
        "action_flow_on_policy_seed_count": len(inherited_action_flow),
        "generated_or_relabel_seed_count": len(generated),
        "unique_training_seed_count": len(result),
        "partition_manifests_cross_checked": [
            "world_model",
            "action_prior",
            "joint_wam",
        ],
        "action_flow_metrics_cross_checked": True,
    }


def _manifest_partitions(
    manifest: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, tuple[str, ...]]:
    raw = manifest.get("partitions")
    required = {"train", "validation", "test"}
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError(f"{field} must contain exactly train/validation/test")
    result: dict[str, tuple[str, ...]] = {}
    for name in sorted(required):
        values = raw[name]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{field}.{name} must be a non-empty array")
        paths = tuple(str(value) for value in values)
        if any(not value for value in paths) or len(paths) != len(set(paths)):
            raise ValueError(f"{field}.{name} contains invalid or duplicate paths")
        result[name] = tuple(sorted(paths))
    return result


def _partition_path_seeds(paths: Sequence[str], *, field: str) -> set[int]:
    seeds: list[int] = []
    for value in paths:
        stem = Path(value).stem
        prefix = "episode_"
        suffix = stem[len(prefix) :] if stem.startswith(prefix) else ""
        if not suffix.isdigit():
            raise ValueError(
                f"partition {field} path does not encode an episode seed: {value}"
            )
        seeds.append(int(suffix))
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"partition {field} paths contain duplicate episode seeds")
    return set(seeds)


def _manifest_seed_list(
    values: Any,
    *,
    field: str,
    allow_empty: bool,
) -> set[int]:
    if not isinstance(values, list):
        raise ValueError(f"training manifest field {field} must be an array")
    if not values and not allow_empty:
        raise ValueError(f"training manifest field {field} must be non-empty")
    seeds: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"training manifest field {field} contains an invalid seed"
            )
        seeds.append(value)
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"training manifest field {field} contains duplicate seeds")
    return set(seeds)


def _configure_torch(config: Mapping[str, Any], device: torch.device) -> None:
    runtime = _mapping(config, "runtime")
    torch.set_float32_matmul_precision(
        str(runtime.get("torch_float32_matmul_precision", "highest"))
    )
    if device.type == "cpu":
        threads = int(runtime["cpu_threads"])
        if threads <= 0:
            raise ValueError("runtime.cpu_threads must be positive")
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(threads)
        except RuntimeError:
            # A test process may already have initialized the inter-op pool.  The
            # standalone formal CLI reaches this branch before any parallel work.
            pass


def _seed_rollout(seed: int) -> None:
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _source_fingerprints(settings: Mapping[str, Any]) -> dict[str, Any]:
    source_paths = {
        "evaluation_script": Path(__file__).resolve(),
        "config": settings["config_path"],
        "acceptance": ROOT / "eval/joint_wam.py",
        "closed_loop_metrics": ROOT / "eval/closed_loop.py",
        "runtime": ROOT / "envs/runtime.py",
        "environment": ROOT / "envs/two_robot_carry_env.py",
        "video": ROOT / "envs/video.py",
        "annotations": ROOT / "envs/annotations.py",
        "joint_policy": ROOT / "policies/joint_wam.py",
        "action_prior_policy": ROOT / "policies/action_prior.py",
        "checkpointing": ROOT / "train/joint_wam_checkpointing.py",
    }
    checkpoint_paths = {
        "world_model": settings["world_model_checkpoint"],
        "action_prior": settings["action_prior_checkpoint"],
        "joint_wam": settings["checkpoint"],
    }
    return {
        "source_files": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in source_paths.items()
        },
        "checkpoint_trees": {
            name: _tree_fingerprint(path) for name, path in checkpoint_paths.items()
        },
    }


def _tree_fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"checkpoint directory is empty: {path}")
    digest = hashlib.sha256()
    total_bytes = 0
    manifest: dict[str, str] = {}
    for item in files:
        relative = item.relative_to(path).as_posix()
        item_hash = _sha256(item)
        manifest[relative] = item_hash
        total_bytes += item.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item_hash.encode("ascii"))
        digest.update(b"\n")
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "files": len(files),
        "bytes": total_bytes,
        "manifest": manifest,
    }


def _require_identical_state_dicts(
    first: torch.nn.Module,
    second: torch.nn.Module,
    *,
    message: str,
) -> None:
    first_state = first.state_dict()
    second_state = second.state_dict()
    if first_state.keys() != second_state.keys() or any(
        not torch.equal(
            first_state[name].detach().cpu(), second_state[name].detach().cpu()
        )
        for name in first_state
    ):
        raise RuntimeError(message)


def _runtime_metadata(
    config: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" and torch.cuda.is_available()
            else None
        ),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "configured_runtime": _canonical(_mapping(config, "runtime")),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
    }


def _selection_to_json(selection: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in selection.items():
        if key in {"success", "failure", "selected"}:
            result[key] = [
                item.as_dict() if hasattr(item, "as_dict") else dict(item)
                for item in value
            ]
        else:
            result[key] = _canonical(value)
    return result


def _write_outputs(
    output: Path,
    report: Mapping[str, Any],
    records_by_suite: Mapping[str, Mapping[str, Sequence[ClosedLoopEpisode]]],
) -> None:
    _atomic_write_text(
        output / "closed_loop_metrics.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    episodes_path = output / "closed_loop_episodes.jsonl"
    temporary = episodes_path.with_name(episodes_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for suite in FORMAL_SUITES:
            for policy in report["metrics"][suite]:
                for episode in records_by_suite[suite][policy]:
                    stream.write(
                        json.dumps(
                            {"suite": suite, **episode_to_dict(episode)},
                            sort_keys=True,
                        )
                        + "\n"
                    )
    temporary.replace(episodes_path)
    _atomic_write_text(
        output / "video_manifest.json",
        json.dumps(
            {
                "selection": report["video_selection"],
                "evidence": report["video_evidence"],
                "validation": report["video_validation"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    _atomic_write_text(output / "report.md", _markdown(report))


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Joint WAM validation",
        "",
        f"结论：**Joint WAM 验证{'通过' if report['policy_acceptable'] else '未通过'}**，"
        f"`policy_acceptable={str(report['policy_acceptable']).lower()}`。",
        "",
        "| Suite | Policy | Episodes | Success | Return | Response delay | "
        "Coordination error | P50/P95/P99 latency | Fallback |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for suite in FORMAL_SUITES:
        for policy, metrics in report["metrics"][suite].items():
            latency = metrics["planner_latency_ms"]
            delay = metrics["mean_response_delay_seconds"]
            lines.append(
                f"| {suite} | {policy} | {metrics['episodes']} | "
                f"{metrics['success_rate']:.1%} | {metrics['mean_episode_return']:.3f} | "
                f"{_format_optional(delay)} | {metrics['mean_coordination_error']:.5f} | "
                f"{_format_optional(latency['p50'])} / "
                f"{_format_optional(latency['p95'])} / "
                f"{_format_optional(latency['p99'])} ms | "
                f"{metrics['fallback_trigger_rate']:.1%} |"
            )
    acceptance = report["acceptance"]
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            f"- formal protocol: `{str(report['formal_protocol']).lower()}`",
            f"- policy acceptable: `{str(report['policy_acceptable']).lower()}`",
            f"- held-out seed overlap: `{report['held_out_seed_overlap']}`",
            "- source checkpoints immutable: "
            f"`{str(report['source_checkpoints_immutable']).lower()}`",
            "- required videos complete: "
            f"`{str(report['video_validation']['passed']).lower()}`",
            "- joint benefit: `not evaluated`",
            "",
        ]
    )
    for suite, values in acceptance["suites"].items():
        lines.append(
            f"- {suite}: direct `{values.get('direct_success_rate'):.1%}`, "
            f"prior `{values.get('prior_success_rate'):.1%}`, "
            f"regression `{values.get('prior_regression'):.1%}`"
        )
    if report["policy_acceptable"]:
        conclusion = "本结果证明当前 proprioceptive Joint WAM 在本任务上可接受"
    else:
        conclusion = "本次运行未形成可接受的正式 Joint WAM 结论"
    lines.extend(
        [
            "",
            conclusion
            + "；未评测未饱和任务上的同预算配对消融，因此不声称 joint world "
            "modeling 带来控制增益。",
            "",
        ]
    )
    return "\n".join(lines)


def _format_optional(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_finite(value: Any) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
        and float(value) > 0.0
    )


def _exact_zero(value: Any) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
        and float(value) == 0.0
    )


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Joint WAM config root must be a mapping")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be a mapping: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
