"""Evaluate the retained action-prior baseline in closed loop."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.trajectory import PROPRIO_WAM_SCHEMA_VERSION  # noqa: E402
from envs.runtime import CallablePolicy, RenderRequest, RunnerConfig, SimulationRunner  # noqa: E402
from envs.two_robot_carry_env import CooperativeStopEnvConfig, TwoRobotCooperativeStopEnv  # noqa: E402
from envs.video import StreamingVideoObserver  # noqa: E402
from eval.closed_loop import ClosedLoopEpisode, ClosedLoopEpisodeObserver, aggregate_closed_loop, episode_to_dict  # noqa: E402
from policies import ActionPriorPolicy  # noqa: E402
from train.action_prior import load_action_prior_checkpoint  # noqa: E402
from train.progress import TrainingProgress  # noqa: E402
from train.rwm_u_checkpointing import load_rwm_u_member_checkpoint  # noqa: E402

POLICIES = ("action_prior", "stationary", "scripted_oracle")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/wam/action_prior.yaml")
    parser.add_argument("--world-model-checkpoint-dir", type=Path)
    parser.add_argument("--action-prior-checkpoint-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/action_prior_eval")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--no-randomize", action="store_true")
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--video-episodes", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_yaml(args.config)
    device = _device(args.device)
    world_model_path = (args.world_model_checkpoint_dir or ROOT / config["world_model"]["checkpoint"]).resolve()
    prior_path = (args.action_prior_checkpoint_dir or ROOT / config["checkpoint"]["directory"]).resolve()
    member, metadata = load_rwm_u_member_checkpoint(
        world_model_path,
        0,
        device=device,
        expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION,
    )
    prior, _ = load_action_prior_checkpoint(
        prior_path,
        world_model_checkpoint=world_model_path,
        device=device,
        expected_schema_version=PROPRIO_WAM_SCHEMA_VERSION,
        expected_normalization_sha256=metadata["normalization"].sha256(),
    )
    fixed_actions = {
        int(index): float(value)
        for index, value in config["runtime"].get("fixed_actions", {}).items()
    }
    seeds = args.seeds or list(range(args.seed_start, args.seed_start + args.episodes))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_by_policy: dict[str, list[ClosedLoopEpisode]] = {}
    with TrainingProgress(enabled=not args.no_progress, total_stages=len(args.policies)) as progress:
        for name in args.policies:
            phase = progress.add_phase(f"closed-loop {name}", len(seeds))
            env = TwoRobotCooperativeStopEnv(CooperativeStopEnvConfig(include_camera_images=False))
            try:
                policy = _policy(name, env, member, prior, fixed_actions)
                records: list[ClosedLoopEpisode] = []
                for episode_index, seed in enumerate(seeds):
                    record_video = args.video_dir is not None and episode_index < args.video_episodes
                    runner = SimulationRunner(
                        env,
                        policy,
                        RunnerConfig(
                            max_steps=args.max_steps,
                            render=(RenderRequest("fixed", "fixed", width=640, height=360),) if record_video else (),
                            expose_privileged_state_to_policy=False,
                        ),
                    )
                    observer = ClosedLoopEpisodeObserver(name, policy)
                    with ExitStack() as stack:
                        observers: list[Any] = [observer]
                        if record_video:
                            observers.append(
                                stack.enter_context(
                                    StreamingVideoObserver(
                                        args.video_dir / name / f"seed_{seed:06d}.mp4",
                                        stream="fixed",
                                        fps=1.0 / env.control_dt,
                                    )
                                )
                            )
                        summary = runner.run_episode(
                            seed=seed,
                            episode_index=episode_index,
                            randomize=not args.no_randomize,
                            observers=observers,
                        )
                    records.append(observer.finish(summary))
                    phase.advance(
                        {
                            "episode": episode_index + 1,
                            "success": int(summary.final_info.get("success", False)),
                        }
                    )
                records_by_policy[name] = records
                phase.finish(f"success {np.mean([record.success for record in records]):.1%}")
            finally:
                env.close()
    report = {
        "format_version": "wam.action_prior.closed_loop/1",
        "world_model_checkpoint": str(world_model_path),
        "action_prior_checkpoint": str(prior_path),
        "randomize": not args.no_randomize,
        "metrics": {name: aggregate_closed_loop(records) for name, records in records_by_policy.items()},
    }
    (output / "closed_loop_metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    with (output / "closed_loop_episodes.jsonl").open("w", encoding="utf-8") as stream:
        for name in args.policies:
            for episode in records_by_policy[name]:
                stream.write(json.dumps(episode_to_dict(episode), sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "metrics": report["metrics"]}, indent=2))
    return 0


def _policy(
    name: str,
    env: TwoRobotCooperativeStopEnv,
    world_model: Any,
    prior: Any,
    fixed_actions: Mapping[int, float],
) -> Any:
    if name == "scripted_oracle":
        return CallablePolicy(lambda observation: env.scripted_action())
    if name == "stationary":
        action = np.zeros(env.action_dim, dtype=np.float32)
        for index, value in fixed_actions.items():
            action[index] = value
        return CallablePolicy(lambda observation, value=action: value.copy())
    return ActionPriorPolicy(world_model, prior, fixed_actions=fixed_actions)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("action-prior config root must be a mapping")
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
