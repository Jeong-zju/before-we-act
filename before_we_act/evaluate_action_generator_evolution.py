"""Closed-loop R12-E1 hybrid: exact W10 fallback plus one specialist."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import gymnasium as gym
import numpy as np
import torch

import robofactory  # noqa: F401

from before_we_act.action_generator.evolution import (
    TaskConditionedActionGenerator,
    load_r12_evolution_config,
)
from before_we_act.benchmark import TASKS as BENCHMARK_TASKS, get_task
from before_we_act.contracts import TeamBeliefState
from before_we_act.data.raw_team_windows import TASKS
from before_we_act.evaluate_action_generator_r4 import (
    TeamHistory,
    TemporalChunkEnsembler,
    terminal_info,
)
from before_we_act.spatial_observation import R12SpatialObservationEncoder
from before_we_act.team_belief.base import PredictiveBeliefModel, load_r11_config
from before_we_act.upstream_components.cogact import AdaptiveEnsembler
from before_we_act.upstream_components.aac import select_joint_action_chunk



def reset_reproducibly(env, seed: int):
    """Reset the specialist evaluation without importing the W10 runtime."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return env.reset(seed=seed)


def make_env(task_name: str):
    specification = get_task(task_name)
    return gym.make(
        specification["env_id"],
        config=f"/workspace/RoboFactory/{specification['config']}",
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="sensors",
        reward_mode="dense",
        sim_backend="cpu",
        sensor_configs=dict(shader_pack="default", width=640, height=480),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
    )


class CogACTAdaptiveTeamEnsembler:
    """Per-arm adapter around the pinned CogACT inference component."""

    def __init__(self, arms, horizon: int = 2, alpha: float = 0.1):
        self.arms = tuple(arms)
        self.ensemblers = [
            AdaptiveEnsembler(horizon, alpha) for _ in self.arms
        ]

    def append_and_select(self, _step: int, chunks: np.ndarray):
        return {
            f"panda-{arm}": self.ensemblers[local_index]
            .ensemble_action(chunks[local_index])
            .copy()
            for local_index, arm in enumerate(self.arms)
        }


def expand_belief_for_samples(
    belief: TeamBeliefState, samples: int
) -> TeamBeliefState:
    if len(belief.tokens) != 1 or samples < 2:
        raise ValueError("AAC belief expansion requires one observation and >=2 samples")
    return TeamBeliefState(
        tokens=belief.tokens.expand(samples, -1, -1),
        agent_tokens=belief.agent_tokens.expand(samples, -1, -1),
        consensus_token=belief.consensus_token.expand(samples, -1),
        uncertainty=belief.uncertainty.expand(samples, *belief.uncertainty.shape[1:]),
        agent_mask=belief.agent_mask.expand(samples, -1),
    ).validate()


def aac_noise(*, seed: int, step: int, samples: int, device) -> torch.Tensor:
    base_seed = int(seed) * 1_000_003 + int(step)
    return torch.cat(
        [
            torch.randn(
                (1, 100, 32),
                generator=torch.Generator(device=device).manual_seed(
                    base_seed + sample_index * 1_000_000_007
                ),
                device=device,
            )
            for sample_index in range(samples)
        ],
        dim=0,
    )


def load_specialist(
    config,
    checkpoint_path,
    belief_config_path,
    belief_checkpoint_path,
    vision_artifact,
    vision_batch_size,
    device,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("round") != "R12-E1"
        or checkpoint.get("candidate_id") != config.candidate_id
        or not checkpoint.get("core_free_runtime")
    ):
        raise ValueError("R12-E1 specialist checkpoint identity differs")
    generator = TaskConditionedActionGenerator(config).to(device)
    generator.load_state_dict(checkpoint["model"], strict=True)
    generator.eval()
    belief_config = load_r11_config(belief_config_path)
    belief_saved = torch.load(
        belief_checkpoint_path, map_location="cpu", weights_only=False
    )
    belief = PredictiveBeliefModel(belief_config).to(device)
    belief.load_state_dict(belief_saved["model"], strict=True)
    belief.eval()
    spatial = R12SpatialObservationEncoder(
        config.observation,
        vision_artifact,
        inference_batch_size=int(vision_batch_size),
    ).to(device).eval()
    stats = {
        key: torch.as_tensor(value, device=device)
        for key, value in checkpoint["stats"].items()
    }
    return generator, belief, spatial, stats, int(checkpoint["update"])


@torch.no_grad()
def evaluate_specialist(
    config,
    checkpoint_path,
    belief_config_path,
    belief_checkpoint_path,
    vision_artifact,
    vision_batch_size,
    task_name,
    seeds,
    device,
    max_steps,
    execution_mode="act_temporal_ensemble",
):
    generator, belief, spatial, stats, update = load_specialist(
        config,
        checkpoint_path,
        belief_config_path,
        belief_checkpoint_path,
        vision_artifact,
        vision_batch_size,
        device,
    )
    specification = get_task(task_name)
    arms = specification["agents"]
    task_index = torch.tensor([TASKS.index(task_name)], device=device)
    env = make_env(task_name)
    rows, latencies = [], []
    try:
        for seed in seeds:
            observation, _ = reset_reproducibly(env, seed)
            history = TeamHistory(arms)
            ensemble = (
                CogACTAdaptiveTeamEnsembler(arms, horizon=2, alpha=0.1)
                if execution_mode == "cogact_adaptive_ensemble"
                else None
                if execution_mode == "aac_entropy_chunk"
                else TemporalChunkEnsembler(
                    arms,
                    decay={
                        "act_temporal_ensemble": 0.01,
                        "mild_temporal_ensemble": 0.02,
                        "balanced_temporal_ensemble": 0.05,
                        "recent_temporal_ensemble": 0.10,
                        "responsive_temporal_ensemble": 0.20,
                        "latest_chunk": 0.01,
                    }[execution_mode],
                )
            )
            previous_action = None
            success, info = False, {}
            aac_action_queue: list[dict[str, np.ndarray]] = []
            aac_chunk_sizes: list[int] = []
            aac_entropy_elbows: list[float] = []
            for step in range(max_steps):
                batch = history.batch(observation, previous_action, device)
                if execution_mode == "aac_entropy_chunk" and aac_action_queue:
                    action = aac_action_queue.pop(0)
                else:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter_ns()
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
                    ):
                        belief_state = belief(batch)["belief"]
                        spatial_tokens, spatial_view_mask = spatial(
                            batch["raw_fixed_rgb"], batch["spatial_view_mask"]
                        )
                        if execution_mode == "aac_entropy_chunk":
                            sample_count = 20
                            proposals = generator.sample(
                                expand_belief_for_samples(belief_state, sample_count),
                                spatial_tokens=spatial_tokens.expand(
                                    sample_count, -1, -1, -1
                                ),
                                spatial_view_mask=spatial_view_mask.expand(
                                    sample_count, -1
                                ),
                                task_index=task_index.expand(sample_count),
                                noise=aac_noise(
                                    seed=int(seed), step=step,
                                    samples=sample_count, device=device,
                                ),
                            )
                        else:
                            noise_generator = torch.Generator(device=device).manual_seed(
                                int(seed) * 1_000_003 + step
                            )
                            noise = torch.randn(
                                (1, 100, 32), generator=noise_generator, device=device
                            )
                            proposals = generator.sample(
                                belief_state,
                                spatial_tokens=spatial_tokens,
                                spatial_view_mask=spatial_view_mask,
                                task_index=task_index,
                                noise=noise,
                            )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    latencies.append((time.perf_counter_ns() - started) / 1e6)
                    if execution_mode == "aac_entropy_chunk":
                        normalized_samples = (
                            proposals.actions[:, 0, : len(arms)].float().cpu().numpy()
                        )
                        selection = select_joint_action_chunk(normalized_samples)
                        raw_samples = (
                            proposals.actions[:, 0, : len(arms)]
                            * stats["a_std"][None, None, None]
                            + stats["a_mean"][None, None, None]
                        ).float().cpu().numpy()
                        selected = raw_samples[selection.sample_index]
                        aac_action_queue = [
                            {
                                f"panda-{arm}": selected[local_index, offset].copy()
                                for local_index, arm in enumerate(arms)
                            }
                            for offset in range(selection.chunk_size)
                        ]
                        aac_chunk_sizes.append(selection.chunk_size)
                        aac_entropy_elbows.append(
                            float(selection.chunk_mean_entropy[selection.chunk_size - 1])
                        )
                        action = aac_action_queue.pop(0)
                    else:
                        normalized = proposals.actions[0, 0, : len(arms)]
                        raw = (
                            normalized * stats["a_std"][None, None]
                            + stats["a_mean"][None, None]
                        )
                        raw_actions = raw.float().cpu().numpy()
                        action = (
                            {
                                f"panda-{arm}": raw_actions[local_index, 0].copy()
                                for local_index, arm in enumerate(arms)
                            }
                            if execution_mode == "latest_chunk"
                            else ensemble.append_and_select(step, raw_actions)
                        )
                previous_action = {key: value.copy() for key, value in action.items()}
                observation, _, terminated, truncated, info = env.step(action)
                success = bool(np.asarray(info.get("success", False)).all())
                if (
                    success
                    or bool(np.asarray(terminated).all())
                    or bool(np.asarray(truncated).all())
                ):
                    break
            row = {
                "task": task_name,
                "seed": int(seed),
                "success": success,
                "steps": step + 1,
                "safety_projections": 0,
                "terminal_info": terminal_info(info),
                "route": {
                    "act_temporal_ensemble": "r12e1_high_resolution_specialist",
                    "mild_temporal_ensemble": "r15_w12_mild_decay_0p02_stack_specialist",
                    "balanced_temporal_ensemble": "r15_w12_balanced_decay_0p05_stack_specialist",
                    "recent_temporal_ensemble": "r15_w12_recent_decay_0p10_stack_specialist",
                    "responsive_temporal_ensemble": "r15_w12_responsive_decay_0p20_stack_specialist",
                    "cogact_adaptive_ensemble": "r15_cogact_adaptive_alpha0p1_h2_stack_specialist",
                    "aac_entropy_chunk": "r15_aac_entropy20_h16_stack_specialist",
                    "latest_chunk": "r15_w12_latest_chunk_stack_specialist",
                }[execution_mode],
                "aac_replans": len(aac_chunk_sizes),
                "aac_chunk_sizes": aac_chunk_sizes,
                "aac_mean_chunk_size": (
                    float(np.mean(aac_chunk_sizes)) if aac_chunk_sizes else None
                ),
                "aac_mean_selected_entropy": (
                    float(np.mean(aac_entropy_elbows)) if aac_entropy_elbows else None
                ),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        env.close()
    return rows, latencies, update


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--belief-config", required=True)
    parser.add_argument("--belief-checkpoint", required=True)
    parser.add_argument("--vision-artifact", required=True)
    parser.add_argument("--vision-batch-size", type=int, default=1)
    parser.add_argument("--task", choices=tuple(BENCHMARK_TASKS), required=True)
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-log", default="")
    parser.add_argument(
        "--execution-mode",
        choices=(
            "act_temporal_ensemble",
            "mild_temporal_ensemble",
            "balanced_temporal_ensemble",
            "recent_temporal_ensemble",
            "responsive_temporal_ensemble",
            "cogact_adaptive_ensemble",
            "aac_entropy_chunk",
            "latest_chunk",
        ),
        default="act_temporal_ensemble",
    )
    args = parser.parse_args()
    if args.vision_batch_size < 1:
        raise ValueError("R12-E1 vision batch size must be positive")
    config = load_r12_evolution_config(args.config)
    seed_path = Path(args.seed_file).resolve(strict=True)
    seed_bytes = seed_path.read_bytes()
    all_seeds = [int(seed) for seed in json.loads(seed_bytes)["seeds"]]
    if args.episodes < 1 or args.episodes > len(all_seeds):
        raise ValueError("R12-E1 episodes exceed the seed manifest")
    requested = all_seeds[: args.episodes]
    recovered = []
    if args.resume_log and Path(args.resume_log).is_file():
        for line in Path(args.resume_log).read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                row.get("task") == args.task
                and row.get("seed") in requested
                and isinstance(row.get("success"), bool)
                and isinstance(row.get("steps"), int)
            ):
                recovered.append(row)
    recovered = list({row["seed"]: row for row in recovered}.values())
    complete = {row["seed"] for row in recovered}
    remaining = [seed for seed in requested if seed not in complete]
    device = torch.device(args.device)
    if args.task in config.deployment["specialist_tasks"]:
        evaluated, latencies, update = evaluate_specialist(
            config,
            args.checkpoint,
            args.belief_config,
            args.belief_checkpoint,
            args.vision_artifact,
            args.vision_batch_size,
            args.task,
            remaining,
            device,
            args.max_steps,
            args.execution_mode,
        ) if remaining else ([], [], None)
        route = {
            "act_temporal_ensemble": "r12e1_high_resolution_specialist",
            "mild_temporal_ensemble": "r15_w12_mild_decay_0p02_stack_specialist",
            "balanced_temporal_ensemble": "r15_w12_balanced_decay_0p05_stack_specialist",
            "recent_temporal_ensemble": "r15_w12_recent_decay_0p10_stack_specialist",
            "responsive_temporal_ensemble": "r15_w12_responsive_decay_0p20_stack_specialist",
            "cogact_adaptive_ensemble": "r15_cogact_adaptive_alpha0p1_h2_stack_specialist",
            "aac_entropy_chunk": "r15_aac_entropy20_h16_stack_specialist",
            "latest_chunk": "r15_w12_latest_chunk_stack_specialist",
        }[args.execution_mode]
    elif args.task in config.deployment["protected_tasks"]:
        raise ValueError(
            "protected tasks must be evaluated by the isolated exact-W10 "
            "fallback materializer/canary, never the core-free specialist process"
        )
    else:
        raise ValueError("R12-E1 task has no deployment route")
    rows = recovered + evaluated
    rows.sort(key=lambda row: requested.index(row["seed"]))
    values = np.asarray(latencies, dtype=np.float64)
    result = {
        "schema_version": 1,
        "round": "R12-E1",
        "candidate_id": config.candidate_id,
        "task": args.task,
        "route": route,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_update": update,
        "episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "rows": rows,
        "latency_ms": {
            "samples": len(latencies),
            "p50": float(np.percentile(values, 50)) if len(values) else None,
            "p95": float(np.percentile(values, 95)) if len(values) else None,
        },
        "seed_protocol": {
            "source": str(seed_path),
            "sha256": hashlib.sha256(seed_bytes).hexdigest(),
        },
        "policy_inputs": "native 480x640 RGB first; W11 TeamBeliefState, task ID and bounded agent-slot ID supplemental on specialist route",
        "privileged_inputs": False,
        "control_cadence": (
            "AAC entropy-selected open-loop chunk with 20 samples per replan"
            if args.execution_mode == "aac_entropy_chunk"
            else "one proposal per environment step"
        ),
        "temporal_aggregation": {
            "act_temporal_ensemble": "W10 exponential chunk ensemble decay=0.01",
            "mild_temporal_ensemble": "mild exponential chunk ensemble decay=0.02",
            "balanced_temporal_ensemble": "balanced exponential chunk ensemble decay=0.05",
            "recent_temporal_ensemble": "exponential chunk ensemble decay=0.10",
            "responsive_temporal_ensemble": "responsive exponential chunk ensemble decay=0.20",
            "cogact_adaptive_ensemble": "Microsoft CogACT Adaptive Action Ensemble alpha=0.1 horizon=2",
            "aac_entropy_chunk": "CVPR 2026 AAC: 20 samples, entropy horizon=16, minimum chunk=2",
            "latest_chunk": "latest predicted chunk first action; replan every environment step",
        }[args.execution_mode],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result | {"rows": "saved"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
