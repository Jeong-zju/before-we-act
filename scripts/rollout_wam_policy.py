"""Run Phase 3 WAM-MPPI closed-loop baselines and emit a Gate D report."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.trajectory import PROPRIO_WAM_SCHEMA_VERSION  # noqa: E402
from envs.runtime import (  # noqa: E402
    CallablePolicy,
    RenderRequest,
    RunnerConfig,
    SimulationRunner,
)
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
    gate_d_report,
)
from policies import (  # noqa: E402
    MPPIConfig,
    MPPIRiskWeights,
    MPPISafetyConfig,
    RiskAwareMPPI,
    WAMMPPIActionPolicy,
)
from train.progress import TrainingProgress  # noqa: E402
from train.rwm_u_checkpointing import load_rwm_u_checkpoint  # noqa: E402
from train.wam_mppi_checkpointing import load_wam_mppi_heads_checkpoint  # noqa: E402

POLICIES = ("wam_mppi", "action_prior", "stationary", "scripted_oracle")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/wam/phase3_wam_mppi_v2.yaml"
    )
    parser.add_argument("--phase2-checkpoint-dir", type=Path)
    parser.add_argument("--phase3-checkpoint-dir", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--policies", nargs="+", choices=POLICIES)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--no-randomize", action="store_true")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--num-elites", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--particles", type=int)
    parser.add_argument("--planning-horizon", type=int)
    parser.add_argument("--candidate-batch-size", type=int)
    parser.add_argument("--initial-std", type=float)
    parser.add_argument("--prior-action-max-delta", type=float)
    parser.add_argument("--terminal-value-weight", type=float)
    parser.add_argument("--latency-budget-ms", type=float)
    parser.add_argument("--recovery-steps", type=int)
    parser.add_argument("--allow-latency-recovery", action="store_true")
    parser.add_argument("--execute-over-budget-plans", action="store_true")
    parser.add_argument("--max-predicted-robot-distance", type=float)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--video-episodes", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--progress-refresh-hz", type=float, default=4.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_yaml(args.config)
    settings = _settings(config, args)
    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    ensemble, phase2_metadata = load_rwm_u_checkpoint(
        settings["phase2_checkpoint"],
        device=device,
        expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION,
    )
    heads, phase3_metadata = load_wam_mppi_heads_checkpoint(
        settings["phase3_checkpoint"],
        phase2_checkpoint=settings["phase2_checkpoint"],
        device=device,
        expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION,
        expected_normalization_sha256=phase2_metadata["normalization"].sha256(),
    )
    calibration = json.loads(settings["calibration"].read_text(encoding="utf-8"))
    variance_scale = np.asarray(calibration["variance_scale"], dtype=np.float32)
    planner_config = _planner_config(config["planner"], args)
    risk_weights = MPPIRiskWeights(**dict(config["risk"]))
    safety = _safety_config(config["safety"], args)
    outcome_positive_weights = {
        name: float(values["positive_weight"])
        for name, values in phase2_metadata["metrics"][
            "outcome_label_stats"
        ].items()
    }
    fixed_actions = {
        int(index): float(value)
        for index, value in config["planner"].get("fixed_actions", {}).items()
    }
    output_dir = settings["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    all_episodes: dict[str, list[ClosedLoopEpisode]] = {}
    with TrainingProgress(
        enabled=not args.no_progress,
        total_stages=len(settings["policies"]) + 1,
        refresh_per_second=args.progress_refresh_hz,
    ) as progress:
        for policy_index, policy_name in enumerate(settings["policies"]):
            phase = progress.add_phase(
                f"closed-loop {policy_name}", settings["episodes"]
            )
            env = TwoRobotCooperativeStopEnv(
                CooperativeStopEnvConfig(include_camera_images=False)
            )
            try:
                policy = _policy(
                    policy_name,
                    env,
                    ensemble=ensemble,
                    heads=heads,
                    planner_config=planner_config,
                    risk_weights=risk_weights,
                    safety=safety,
                    variance_scale=variance_scale,
                    fixed_actions=fixed_actions,
                    outcome_positive_weights=outcome_positive_weights,
                    seed=int(config["training"]["seed"]) + 1009 * policy_index,
                )
                records: list[ClosedLoopEpisode] = []
                for episode_index in range(settings["episodes"]):
                    seed = settings["seeds"][episode_index]
                    record_video = bool(
                        args.video_dir is not None
                        and episode_index < args.video_episodes
                    )
                    render = (
                        (RenderRequest("fixed", "fixed", width=640, height=360),)
                        if record_video
                        else ()
                    )
                    runner = SimulationRunner(
                        env,
                        policy,
                        RunnerConfig(
                            max_steps=args.max_steps,
                            render=render,
                            expose_privileged_state_to_policy=False,
                        ),
                    )
                    observer = ClosedLoopEpisodeObserver(policy_name, policy)
                    with ExitStack() as stack:
                        observers: list[Any] = [observer]
                        if record_video:
                            video_path = (
                                args.video_dir
                                / policy_name
                                / f"seed_{seed:06d}.mp4"
                            )
                            observers.append(
                                stack.enter_context(
                                    StreamingVideoObserver(
                                        video_path,
                                        stream="fixed",
                                        fps=1.0 / env.control_dt,
                                    )
                                )
                            )
                        summary = runner.run_episode(
                            seed=seed,
                            episode_index=episode_index,
                            randomize=settings["randomize"],
                            observers=observers,
                        )
                    records.append(observer.finish(summary))
                    phase.advance(
                        {
                            "episode": episode_index + 1,
                            "episodes": settings["episodes"],
                            "success": int(summary.final_info.get("success", False)),
                        }
                    )
                all_episodes[policy_name] = records
                success_rate = np.mean([record.success for record in records])
                phase.finish(f"success {success_rate:.1%}")
            finally:
                env.close()

        report_phase = progress.add_phase("write Gate D report", 3)
        metrics = {
            name: aggregate_closed_loop(
                records,
                exploitation_predicted_return_min=settings[
                    "exploitation_predicted_return_min"
                ],
                exploitation_actual_return_max=settings[
                    "exploitation_actual_return_max"
                ],
            )
            for name, records in all_episodes.items()
        }
        report_phase.advance({"batch": 1})
        training_seeds = _training_seeds(phase3_metadata["dataset_manifest"])
        evaluation_seeds = set(settings["seeds"])
        seed_overlap = len(training_seeds & evaluation_seeds)
        formal = bool(
            settings["episodes"] >= settings["minimum_episodes"]
            and set(settings["policies"]) == set(POLICIES)
            and args.max_steps is None
            and settings["randomize"]
            and not settings["planner_overridden"]
            and args.seeds is None
        )
        gate_d = gate_d_report(
            metrics,
            full_evaluation=formal,
            protocol=settings["gate_d_protocol"],
            minimum_episodes=settings["minimum_episodes"],
            minimum_success_improvement=settings["minimum_success_improvement"],
            maximum_success_regression=settings["maximum_success_regression"],
            maximum_return_regression=settings["maximum_return_regression"],
            minimum_mppi_execution_rate=settings["minimum_mppi_execution_rate"],
            maximum_model_exploitation_events=settings[
                "maximum_model_exploitation_events"
            ],
            latency_budget_ms=safety.latency_budget_ms,
            held_out_seed_overlap=seed_overlap,
        )
        report = {
            "format_version": "wam.phase3.closed_loop/2",
            "gate_d_protocol": settings["gate_d_protocol"],
            "phase2_checkpoint": str(settings["phase2_checkpoint"]),
            "phase3_checkpoint": str(settings["phase3_checkpoint"]),
            "calibration": str(settings["calibration"]),
            "planner": vars(planner_config),
            "risk": vars(risk_weights),
            "safety": vars(safety),
            "metrics": metrics,
            "held_out_seed_overlap": seed_overlap,
            "gate_d": gate_d,
        }
        (output_dir / "closed_loop_metrics.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        report_phase.advance({"batch": 2})
        with (output_dir / "closed_loop_episodes.jsonl").open(
            "w", encoding="utf-8"
        ) as stream:
            for policy_name in settings["policies"]:
                for episode in all_episodes[policy_name]:
                    stream.write(json.dumps(episode_to_dict(episode), sort_keys=True))
                    stream.write("\n")
        (output_dir / "closed_loop_report.md").write_text(
            _markdown_report(report), encoding="utf-8"
        )
        report_phase.advance({"batch": 3})
        report_phase.finish("Gate D " + ("passed" if gate_d["passed"] else "failed"))

    print(json.dumps({"output": str(output_dir), "gate_d": gate_d}, indent=2))
    return 0 if (gate_d["passed"] or not formal) else 2


def _policy(
    name: str,
    env: TwoRobotCooperativeStopEnv,
    *,
    ensemble: Any,
    heads: Any,
    planner_config: MPPIConfig,
    risk_weights: MPPIRiskWeights,
    safety: MPPISafetyConfig,
    variance_scale: np.ndarray,
    fixed_actions: Mapping[int, float],
    outcome_positive_weights: Mapping[str, float],
    seed: int,
) -> Any:
    if name == "scripted_oracle":
        return CallablePolicy(lambda observation: env.scripted_action())
    if name == "stationary":
        action = np.zeros(env.action_dim, dtype=np.float32)
        for index, value in fixed_actions.items():
            action[index] = value
        return CallablePolicy(lambda observation, value=action: value.copy())
    planner = RiskAwareMPPI(
        ensemble,
        heads,
        planner_config,
        risk_weights=risk_weights,
        variance_scale=variance_scale,
        fixed_actions=fixed_actions,
        outcome_positive_weights=outcome_positive_weights,
        max_predicted_robot_distance=safety.max_predicted_robot_distance,
        seed=seed,
    )
    return WAMMPPIActionPolicy(
        planner,
        mode="mppi" if name == "wam_mppi" else "action_prior",
        safety=safety,
    )


def _planner_config(raw: Mapping[str, Any], args: argparse.Namespace) -> MPPIConfig:
    values = {
        name: raw[name]
        for name in MPPIConfig.__dataclass_fields__
        if name in raw
    }
    overrides = {
        "num_samples": args.num_samples,
        "num_elites": args.num_elites,
        "iterations": args.iterations,
        "particles_per_candidate": args.particles,
        "planning_horizon": args.planning_horizon,
        "candidate_batch_size": args.candidate_batch_size,
        "initial_std": args.initial_std,
        "prior_action_max_delta": args.prior_action_max_delta,
        "terminal_value_weight": args.terminal_value_weight,
    }
    values.update({name: value for name, value in overrides.items() if value is not None})
    values["num_policy_trajectories"] = min(
        int(values["num_policy_trajectories"]), int(values["num_samples"])
    )
    return MPPIConfig(**values)


def _safety_config(
    raw: Mapping[str, Any], args: argparse.Namespace
) -> MPPISafetyConfig:
    values = dict(raw)
    overrides = {
        "latency_budget_ms": args.latency_budget_ms,
        "recovery_steps": args.recovery_steps,
        "max_predicted_robot_distance": args.max_predicted_robot_distance,
    }
    values.update({name: value for name, value in overrides.items() if value is not None})
    if args.allow_latency_recovery:
        values["sticky_latency_fallback"] = False
    if args.execute_over_budget_plans:
        values["discard_over_budget_plans"] = False
    return MPPISafetyConfig(**values)


def _settings(config: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    phase2 = config["phase2"]
    checkpoint = config["checkpoint"]
    evaluation = config["evaluation"]
    gate_d_protocol = str(
        evaluation.get("gate_d_protocol", "success_improvement_v1")
    )
    overrides = (
        args.num_samples,
        args.num_elites,
        args.iterations,
        args.particles,
        args.planning_horizon,
        args.candidate_batch_size,
        args.initial_std,
        args.prior_action_max_delta,
        args.terminal_value_weight,
        args.latency_budget_ms,
        args.recovery_steps,
        args.max_predicted_robot_distance,
    )
    if args.seeds is not None and (args.episodes is not None or args.seed_start is not None):
        raise ValueError("--seeds cannot be combined with --episodes or --seed-start")
    seed_start = int(
        evaluation["held_out_seed_start"]
        if args.seed_start is None
        else args.seed_start
    )
    episode_count = int(args.episodes or evaluation["episodes"])
    seeds = (
        tuple(int(seed) for seed in args.seeds)
        if args.seeds is not None
        else tuple(range(seed_start, seed_start + episode_count))
    )
    result = {
        "phase2_checkpoint": (
            args.phase2_checkpoint_dir or ROOT / phase2["checkpoint"]
        ).resolve(),
        "phase3_checkpoint": (
            args.phase3_checkpoint_dir or ROOT / checkpoint["directory"]
        ).resolve(),
        "calibration": (
            args.calibration or ROOT / phase2["uncertainty_calibration"]
        ).resolve(),
        "output_dir": (
            args.output_dir or ROOT / evaluation["output_directory"]
        ).resolve(),
        "episodes": len(seeds),
        "seed_start": min(seeds) if seeds else -1,
        "seeds": seeds,
        "policies": tuple(args.policies or evaluation["policies"]),
        "randomize": not args.no_randomize,
        "minimum_episodes": int(evaluation["gate_d_minimum_episodes"]),
        "gate_d_protocol": gate_d_protocol,
        "minimum_success_improvement": float(
            evaluation.get("gate_d_success_improvement_min", 0.10)
        ),
        "maximum_success_regression": float(
            evaluation.get("gate_d_success_regression_max", 0.01)
        ),
        "maximum_return_regression": float(
            evaluation.get("gate_d_return_regression_max", 0.5)
        ),
        "minimum_mppi_execution_rate": float(
            evaluation.get("gate_d_mppi_execution_rate_min", 0.0)
        ),
        "maximum_model_exploitation_events": int(
            evaluation.get("gate_d_model_exploitation_max", 0)
        ),
        "exploitation_predicted_return_min": float(
            evaluation["exploitation_predicted_return_min"]
        ),
        "exploitation_actual_return_max": float(
            evaluation["exploitation_actual_return_max"]
        ),
        "planner_overridden": bool(
            any(value is not None for value in overrides)
            or args.allow_latency_recovery
            or args.execute_over_budget_plans
        ),
    }
    if result["episodes"] <= 0 or result["seed_start"] < 0:
        raise ValueError("episodes must be positive and seed_start non-negative")
    for path_name in ("phase2_checkpoint", "phase3_checkpoint"):
        if not result[path_name].is_dir():
            raise FileNotFoundError(result[path_name])
    if not result["calibration"].is_file():
        raise FileNotFoundError(result["calibration"])
    return result


def _training_seeds(manifest: Mapping[str, Any]) -> set[int]:
    result: set[int] = set()
    for paths in manifest.get("partitions", {}).values():
        for raw_path in paths:
            path = Path(raw_path)
            if not path.is_file():
                continue
            with h5py.File(path, "r") as file:
                seed = int(file.attrs.get("seed", -1))
            if seed >= 0:
                result.add(seed)
    return result


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 3 Closed-loop / Gate D",
        "",
        f"Protocol: `{report['gate_d_protocol']}`",
        "",
        "| Policy | Success | Return | P95 latency | MPPI execution |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metric in report["metrics"].items():
        p95 = metric["planner_latency_ms"]["p95"]
        execution = metric.get("mppi_execution_rate", 0.0)
        lines.append(
            f"| {name} | {metric['success_rate']:.3f} | "
            f"{metric['mean_episode_return']:.3f} | "
            f"{p95:.3f} ms | {execution:.1%} |" if p95 is not None else
            f"| {name} | {metric['success_rate']:.3f} | "
            f"{metric['mean_episode_return']:.3f} | n/a | {execution:.1%} |"
        )
    mppi = report["metrics"].get("wam_mppi", {})
    lines.extend(
        [
            "",
            f"Deadline misses/discarded plans: {mppi.get('deadline_misses', 0)}/"
            f"{mppi.get('discarded_plans', 0)}",
        ]
    )
    lines.extend(["", f"Gate D: **{'PASS' if report['gate_d']['passed'] else 'FAIL'}**", ""])
    for name, check in report["gate_d"]["checks"].items():
        lines.append(f"- {'PASS' if check['passed'] else 'FAIL'} `{name}`")
    return "\n".join(lines) + "\n"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase 3 config root must be a mapping")
    return payload


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


if __name__ == "__main__":
    raise SystemExit(main())
