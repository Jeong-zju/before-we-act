"""Evaluate M1 gate-zero preservation against immutable legacy Joint WAM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

# Headless MuJoCo rendering is required for the raw fixed-camera policy input.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.runtime import RenderRequest, RunnerConfig, SimulationRunner  # noqa: E402
from envs.two_robot_carry_env import (  # noqa: E402
    CooperativeStopEnvConfig,
    TwoRobotCooperativeStopEnv,
)
from eval.m1_legacy_regression import (  # noqa: E402
    AuditedLegacyDirectPolicy,
    FORMAL_EPISODES_PER_SUITE,
    LEGACY_POLICY,
    LegacyRegressionEpisode,
    LegacyRegressionObserver,
    M1_POLICY,
    REQUIRED_SUITES,
    checkpoint_tree_sha256,
    legacy_regression_report,
    module_state_sha256,
    rotating_train_seed,
)
from eval.m1_vision_contract import (  # noqa: E402
    validate_loaded_checkpoint_vision,
    validate_training_summary_vision,
)
from models.wam import ActionChunkConfig  # noqa: E402
from policies.joint_wam import JointWAMPolicy, JointWAMPolicyConfig  # noqa: E402
from policies.multimodal_joint_wam import (  # noqa: E402
    MultimodalJointWAMPolicy,
    MultimodalJointWAMPolicyConfig,
)
from train.joint_wam_checkpointing import load_joint_wam_checkpoint  # noqa: E402
from train.m1_checkpointing import load_m1_checkpoint  # noqa: E402


CANONICAL_CONFIG = ROOT / "configs/wam_multimodal/m1_latent_wam_dinov3.yaml"
TASK_TEXT = (
    "carry the object together; when one robot slows to a stop, "
    "the other robot should gradually slow and stop"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CANONICAL_CONFIG)
    parser.add_argument("--training-summary", type=Path)
    parser.add_argument("--legacy-checkpoint-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--standard-seed-start", type=int)
    parser.add_argument("--challenge-seed-start", type=int)
    parser.add_argument("--train-seeds", nargs=3, type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=24)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = _load_yaml(config_path)
    settings = _settings(config, config_path=config_path, args=args)
    if args.torch_threads <= 0:
        raise ValueError("torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(max(1, min(4, args.torch_threads)))
    device = _device(args.device)

    training_summary = _load_json(settings["training_summary"])
    _validate_training_summary(
        training_summary,
        config,
        config_path=config_path,
        config_sha256=_file_sha256(config_path),
        train_seeds=settings["train_seeds"],
        formal_protocol=settings["formal_protocol"],
    )
    legacy_path = settings["legacy_checkpoint"]
    source_before = checkpoint_tree_sha256(legacy_path)
    expected_source = str(config["initialization"]["expected_legacy_tree_sha256"])
    if source_before != expected_source:
        raise ValueError("immutable legacy Joint WAM tree hash differs from M1 config")
    legacy_world, legacy_flow, legacy_metadata = load_joint_wam_checkpoint(
        legacy_path,
        device=device,
        expected_schema_version="wam.proprio/1.0",
    )
    legacy_world_hash = module_state_sha256(legacy_world)
    legacy_flow_hash = module_state_sha256(legacy_flow)

    reports = _primary_reports(training_summary, train_seeds=settings["train_seeds"])
    checkpoint_evidence: dict[str, Any] = {}
    m1_records: list[LegacyRegressionEpisode] = []
    for train_seed in settings["train_seeds"]:
        report = reports[train_seed]
        checkpoint_path = _resolve_path(report["checkpoint"])
        observed_tree = checkpoint_tree_sha256(checkpoint_path)
        expected_tree = str(report["checkpoint_tree_sha256"])
        summary_tree = str(
            training_summary["checkpoint_sha256"]["state_vision_future"][
                str(train_seed)
            ]
        )
        if observed_tree != expected_tree or observed_tree != summary_tree:
            raise ValueError(
                f"primary M1 checkpoint tree mismatch for seed {train_seed}"
            )
        model, flow, embedded_world, embedded_flow, metadata = load_m1_checkpoint(
            checkpoint_path,
            device=device,
            expected_schema_version=str(config["data"]["schema_version"]),
        )
        embedded_matches = bool(
            module_state_sha256(embedded_world) == legacy_world_hash
            and module_state_sha256(embedded_flow) == legacy_flow_hash
        )
        if not embedded_matches:
            raise ValueError(
                f"M1 checkpoint seed {train_seed} changed the gate-zero legacy path"
            )
        if (
            metadata["schema"].get("model_variant") != "state_vision_future"
            or int(metadata["schema"].get("train_seed", -1)) != train_seed
        ):
            raise ValueError("primary M1 schema variant/train seed mismatch")
        validate_loaded_checkpoint_vision(config, model, metadata)
        policy = MultimodalJointWAMPolicy(
            model,
            flow,
            embedded_world,
            embedded_flow,
            config=_m1_policy_config(config),
            device=device,
        )
        if policy.canonical_variant != "state_vision_future":
            raise ValueError("retained-task evaluator loaded a non-primary M1 variant")
        strict_summary = training_summary["strict_reload"]["state_vision_future"][
            str(train_seed)
        ]
        checkpoint_evidence[str(train_seed)] = {
            "checkpoint": str(checkpoint_path),
            "tree_sha256": observed_tree,
            "train_seed": train_seed,
            "model_variant": policy.canonical_variant,
            "strict_reload_passed": strict_summary.get("passed") is True,
            "embedded_legacy_matches_source": embedded_matches,
            "schema_format_version": metadata["schema"].get("format_version"),
        }
        assigned = {
            suite: tuple(
                seed
                for index, seed in enumerate(settings["suite_seeds"][suite])
                if rotating_train_seed(index, settings["train_seeds"]) == train_seed
            )
            for suite in REQUIRED_SUITES
        }
        m1_records.extend(
            _evaluate_m1_checkpoint(
                config,
                legacy_metadata["experiment_config"],
                policy=policy,
                train_seed=train_seed,
                assigned_seeds=assigned,
                max_steps=settings["max_steps"],
                progress=not args.no_progress,
            )
        )
        del policy, model, flow, embedded_world, embedded_flow
        if device.type == "cuda":
            torch.cuda.empty_cache()

    legacy_records = _evaluate_legacy_direct(
        legacy_metadata["experiment_config"],
        world=legacy_world,
        flow=legacy_flow,
        suite_seeds=settings["suite_seeds"],
        max_steps=settings["max_steps"],
        progress=not args.no_progress,
    )
    source_after = checkpoint_tree_sha256(legacy_path)
    result = legacy_regression_report(
        sorted(m1_records, key=lambda value: (value.suite, value.seed)),
        sorted(legacy_records, key=lambda value: (value.suite, value.seed)),
        suite_seeds=settings["suite_seeds"],
        train_seeds=settings["train_seeds"],
        formal_protocol=settings["formal_protocol"],
        source_checkpoint_sha256_before=source_before,
        source_checkpoint_sha256_after=source_after,
        expected_source_checkpoint_sha256=expected_source,
        checkpoint_evidence=checkpoint_evidence,
        expected_episodes_per_suite=settings["episodes"],
        maximum_regression=float(config["acceptance"]["maximum_legacy_regression"]),
    )
    result.update(
        {
            "config": str(config_path),
            "config_sha256": _file_sha256(config_path),
            "training_summary": str(settings["training_summary"]),
            "training_summary_sha256": _file_sha256(settings["training_summary"]),
            "legacy_checkpoint": str(legacy_path),
            "device": str(device),
            "visual_input": {
                "camera": "fixed",
                "width": int(config["model"]["vision_input_size"]),
                "height": int(config["model"]["vision_input_size"]),
                "refresh_hz": float(config["evaluation"]["visual_refresh_hz"]),
                "control_hz": float(config["evaluation"]["control_hz"]),
                "raw_unannotated_rgb": True,
            },
        }
    )
    _atomic_json(settings["output"], result)
    if settings["formal_protocol"] and not result["passed"]:
        return 1
    return 0


def _settings(
    config: Mapping[str, Any], *, config_path: Path, args: argparse.Namespace
) -> dict[str, Any]:
    evaluation = _mapping(config, "evaluation")
    training = _mapping(config, "training")
    initialization = _mapping(config, "initialization")
    canonical_summary = (
        ROOT / str(training["report_root"]) / "training_summary.json"
    ).resolve()
    canonical_legacy = (
        ROOT / str(initialization["legacy_joint_wam_checkpoint"])
    ).resolve()
    canonical_output = (
        ROOT / str(evaluation["output_directory"]) / "legacy_regression.json"
    ).resolve()
    training_summary = (args.training_summary or canonical_summary).resolve()
    legacy_checkpoint = (args.legacy_checkpoint_dir or canonical_legacy).resolve()
    output = (args.output or canonical_output).resolve()
    episodes = int(
        evaluation["legacy_regression_episodes_per_suite"]
        if args.episodes is None
        else args.episodes
    )
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max-steps must be positive")
    train_seeds = tuple(
        int(value) for value in (args.train_seeds or _sequence(training, "seeds"))
    )
    if len(train_seeds) != 3 or len(set(train_seeds)) != 3:
        raise ValueError("legacy regression requires three unique train seeds")
    starts = {
        "standard": int(
            evaluation["legacy_standard_seed_start"]
            if args.standard_seed_start is None
            else args.standard_seed_start
        ),
        "challenge": int(
            evaluation["legacy_challenge_seed_start"]
            if args.challenge_seed_start is None
            else args.challenge_seed_start
        ),
    }
    suite_seeds = {
        suite: tuple(range(starts[suite], starts[suite] + episodes))
        for suite in REQUIRED_SUITES
    }
    if set(suite_seeds["standard"]) & set(suite_seeds["challenge"]):
        raise ValueError("standard and challenge seed sets must be disjoint")
    semantic_override = bool(
        config_path != CANONICAL_CONFIG.resolve()
        or args.training_summary is not None
        or args.legacy_checkpoint_dir is not None
        or args.output is not None
        or args.episodes is not None
        or args.standard_seed_start is not None
        or args.challenge_seed_start is not None
        or args.train_seeds is not None
        or args.max_steps is not None
    )
    formal = not semantic_override
    if formal and episodes != FORMAL_EPISODES_PER_SUITE:
        raise ValueError(
            "canonical formal legacy regression requires 500 seeds per suite"
        )
    if semantic_override:
        if args.output is None:
            raise ValueError("diagnostic overrides require a separate --output")
        if output == canonical_output:
            raise ValueError(
                "diagnostic output cannot overwrite formal legacy evidence"
            )
    return {
        "training_summary": training_summary,
        "legacy_checkpoint": legacy_checkpoint,
        "output": output,
        "episodes": episodes,
        "suite_seeds": suite_seeds,
        "train_seeds": train_seeds,
        "max_steps": args.max_steps,
        "formal_protocol": formal,
    }


def _validate_training_summary(
    value: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    config_path: Path,
    config_sha256: str,
    train_seeds: Sequence[int],
    formal_protocol: bool,
) -> None:
    if value.get("format_version") != "wam.multimodal.m1.training/1":
        raise ValueError("unsupported M1 training summary")
    validate_training_summary_vision(value, config, project_root=ROOT)
    if value.get("config_sha256") != config_sha256:
        raise ValueError("training summary does not bind the requested M1 config")
    if Path(str(value.get("config", ""))).resolve() != config_path:
        raise ValueError("training summary config path differs from requested config")
    if tuple(int(item) for item in value.get("train_seeds", ())) != tuple(train_seeds):
        raise ValueError("training summary train seeds differ from legacy evaluation")
    if "state_vision_future" not in value.get("variants", ()):
        raise ValueError("training summary has no primary state+vision+future variant")
    if formal_protocol and (
        value.get("formal_protocol") is not True or value.get("passed") is not True
    ):
        raise ValueError("formal legacy evaluation requires passed formal M1 training")


def _primary_reports(
    summary: Mapping[str, Any], *, train_seeds: Sequence[int]
) -> dict[int, Mapping[str, Any]]:
    reports = summary.get("reports")
    if not isinstance(reports, list):
        raise ValueError("M1 training summary has no run reports")
    result: dict[int, Mapping[str, Any]] = {}
    for report in reports:
        if not isinstance(report, Mapping):
            raise ValueError("M1 training report entries must be mappings")
        if report.get("variant") != "state_vision_future":
            continue
        train_seed = int(report.get("train_seed", -1))
        if train_seed in result:
            raise ValueError(f"duplicate primary training report for seed {train_seed}")
        result[train_seed] = report
    if set(result) != set(int(value) for value in train_seeds):
        raise ValueError(
            "primary training reports do not cover exactly three train seeds"
        )
    for train_seed, report in result.items():
        strict = report.get("strict_reload")
        if not isinstance(strict, Mapping) or strict.get("passed") is not True:
            raise ValueError(
                f"primary checkpoint seed {train_seed} lacks strict reload"
            )
    return result


def _evaluate_m1_checkpoint(
    config: Mapping[str, Any],
    legacy_config: Mapping[str, Any],
    *,
    policy: MultimodalJointWAMPolicy,
    train_seed: int,
    assigned_seeds: Mapping[str, Sequence[int]],
    max_steps: int | None,
    progress: bool,
) -> list[LegacyRegressionEpisode]:
    records: list[LegacyRegressionEpisode] = []
    for suite in REQUIRED_SUITES:
        env = _environment(legacy_config, suite)
        try:
            runner = SimulationRunner(
                env,
                policy,
                _m1_runner_config(config, max_steps=max_steps),
            )
            suite_seeds = tuple(int(value) for value in assigned_seeds[suite])
            for episode_index, seed in enumerate(suite_seeds):
                _seed_rollout(seed)
                observer = LegacyRegressionObserver(
                    suite=suite,
                    policy_name=M1_POLICY,
                    policy=policy,
                    train_seed=train_seed,
                    control_hz=float(config["evaluation"]["control_hz"]),
                    visual_hz=float(config["evaluation"]["visual_refresh_hz"]),
                )
                summary = runner.run_episode(
                    seed=seed,
                    episode_index=episode_index,
                    randomize=True,
                    observers=(observer,),
                    metadata={"suite": suite, "train_seed": train_seed},
                )
                records.append(observer.finish(summary))
                _progress(
                    progress,
                    label=f"{suite} M1 seed-{train_seed}",
                    index=episode_index,
                    total=len(suite_seeds),
                )
        finally:
            env.close()
    return records


def _evaluate_legacy_direct(
    legacy_config: Mapping[str, Any],
    *,
    world: Any,
    flow: Any,
    suite_seeds: Mapping[str, Sequence[int]],
    max_steps: int | None,
    progress: bool,
) -> list[LegacyRegressionEpisode]:
    base = JointWAMPolicy(
        world,
        flow,
        config=_legacy_policy_config(legacy_config),
        fixed_actions=_legacy_fixed_actions(legacy_config),
    )
    policy = AuditedLegacyDirectPolicy(base)
    records: list[LegacyRegressionEpisode] = []
    for suite in REQUIRED_SUITES:
        env = _environment(legacy_config, suite)
        try:
            runner = SimulationRunner(
                env,
                policy,
                RunnerConfig(
                    max_steps=max_steps,
                    expose_privileged_state_to_policy=False,
                    policy_observation_keys=("proprioception",),
                ),
            )
            seeds = tuple(int(value) for value in suite_seeds[suite])
            for episode_index, seed in enumerate(seeds):
                _seed_rollout(seed)
                observer = LegacyRegressionObserver(
                    suite=suite,
                    policy_name=LEGACY_POLICY,
                    policy=policy,
                    train_seed=None,
                )
                summary = runner.run_episode(
                    seed=seed,
                    episode_index=episode_index,
                    randomize=True,
                    observers=(observer,),
                    metadata={"suite": suite},
                )
                records.append(observer.finish(summary))
                _progress(
                    progress,
                    label=f"{suite} legacy",
                    index=episode_index,
                    total=len(seeds),
                )
        finally:
            env.close()
    return records


def _m1_runner_config(
    config: Mapping[str, Any], *, max_steps: int | None
) -> RunnerConfig:
    input_size = int(config["model"]["vision_input_size"])
    visual_hz = float(config["evaluation"]["visual_refresh_hz"])
    history = int(config["data"]["state_history"]) - 1
    return RunnerConfig(
        max_steps=max_steps,
        render=(
            RenderRequest(
                name="fixed",
                camera="fixed",
                width=input_size,
                height=input_size,
                fps=visual_hz,
            ),
        ),
        expose_privileged_state_to_policy=False,
        policy_observation_keys=("proprioception",),
        expose_rendered_images_to_policy=True,
        policy_image_streams=("fixed",),
        expose_task_to_policy=True,
        task_id="cooperative_stop",
        task=TASK_TEXT,
        policy_action_history=history,
    )


def _m1_policy_config(config: Mapping[str, Any]) -> MultimodalJointWAMPolicyConfig:
    chunk = _mapping(config, "action_chunk")
    data = _mapping(config, "data")
    evaluation = _mapping(config, "evaluation")
    acceptance = _mapping(config, "acceptance")
    return MultimodalJointWAMPolicyConfig(
        action_chunk=ActionChunkConfig(
            action_dim=int(data["action_dim"]),
            horizon=int(chunk["horizon"]),
            execution_steps=int(chunk["execution_steps"]),
            solver_steps=int(chunk["solver_steps"]),
            warm_start_mode=str(chunk["warm_start_mode"]),
        ),
        solver=str(chunk["solver"]),
        normalized_action_clip=float(chunk["normalized_action_clip"]),
        visual_residual_scale=float(chunk["anchor_residual_scale_visual"]),
        cooperative_residual_scale=float(chunk["anchor_residual_scale_cooperative"]),
        replan_warm_start_enabled=bool(chunk.get("replan_warm_start_enabled", True)),
        latency_budget_ms=float(acceptance["maximum_sensor_to_action_p95_ms"]),
        maximum_visual_age_ms=float(acceptance["maximum_decimated_action_age_ms"]),
        visual_history_frames=int(data["visual_history_frames"]),
        fixed_actions=((3, 1.0), (7, 1.0)),
        fallback_enabled=bool(evaluation["fallback_enabled"]),
    )


def _legacy_policy_config(config: Mapping[str, Any]) -> JointWAMPolicyConfig:
    runtime = _mapping(config, "runtime")
    chunk = _mapping(config, "action_chunk")
    data = _mapping(config, "data")
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
        observation_residual_nrmse_max=float(runtime["observation_residual_nrmse_max"]),
        risk_veto=bool(runtime["risk_veto"]),
        max_failure_probability=float(runtime["max_failure_probability"]),
        max_predicted_robot_distance=float(runtime["max_predicted_robot_distance"]),
        max_action_ood=float(runtime["max_action_ood"]),
        action_ood_threshold=float(runtime["action_ood_threshold"]),
        latency_budget_ms=float(runtime["latency_budget_ms"]),
        fallback_enabled=False,
    )


def _legacy_fixed_actions(config: Mapping[str, Any]) -> dict[int, float]:
    runtime = _mapping(config, "runtime")
    values = runtime.get("fixed_actions", {})
    if not isinstance(values, Mapping):
        raise ValueError("legacy fixed_actions must be a mapping")
    return {int(index): float(value) for index, value in values.items()}


def _environment(
    legacy_config: Mapping[str, Any], suite: str
) -> TwoRobotCooperativeStopEnv:
    evaluation = _mapping(legacy_config, "evaluation")
    overrides = (
        {}
        if suite == "standard"
        else dict(_mapping(evaluation, "challenge_environment"))
    )
    return TwoRobotCooperativeStopEnv(
        CooperativeStopEnvConfig(include_camera_images=False, **overrides)
    )


def _seed_rollout(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _progress(enabled: bool, *, label: str, index: int, total: int) -> None:
    if not enabled:
        return
    completed = index + 1
    if completed == 1 or completed == total or completed % 25 == 0:
        print(f"{label}: {completed}/{total}", flush=True)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def _resolve_path(value: Any) -> Path:
    path = Path(str(value))
    return (path if path.is_absolute() else ROOT / path).resolve()


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"M1 config field {key!r} must be a mapping")
    return result


def _sequence(value: Mapping[str, Any], key: str) -> list[Any]:
    result = value.get(key)
    if not isinstance(result, list) or not result:
        raise ValueError(f"M1 config field {key!r} must be a non-empty list")
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    result = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("M1 config root must be a mapping")
    return result


def _load_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"JSON root must be a mapping: {path}")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
