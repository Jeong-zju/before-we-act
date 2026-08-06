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
from before_we_act.data.raw_team_windows import TASKS
from before_we_act.evaluate_action_generator_r4 import (
    TeamHistory,
    TemporalChunkEnsembler,
    terminal_info,
)
from before_we_act.spatial_observation import R12SpatialObservationEncoder
from before_we_act.team_belief.base import PredictiveBeliefModel, load_r11_config
from stereo_core.evaluate_no_wrist_pair import (
    TemporalChunkEnsembler as W10TemporalChunkEnsembler,
    load_model as load_w10,
    predict_all as predict_w10,
    reset_reproducibly,
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
def evaluate_w10(config, task_name, seeds, device, max_steps):
    checkpoint = Path(str(config.deployment["w10_checkpoint"])).resolve(strict=True)
    if sha256(checkpoint) != str(config.deployment["w10_checkpoint_sha256"]):
        raise ValueError("R12-E1 W10 fallback hash differs")
    model, stats, _w10_config = load_w10(str(checkpoint), device)
    specification = get_task(task_name)
    arms = specification["agents"]
    env = make_env(task_name)
    rows, latencies = [], []
    try:
        for seed in seeds:
            observation, _ = reset_reproducibly(env, seed)
            ensemble = W10TemporalChunkEnsembler(arms)
            success, info = False, {}
            for step in range(max_steps):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter_ns()
                chunks = predict_w10(model, stats, observation, arms, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                latencies.append((time.perf_counter_ns() - started) / 1e6)
                action = ensemble.append_and_select(step, chunks)
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
                "route": "exact_w10_fallback",
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        env.close()
    return rows, latencies, None


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
            ensemble = TemporalChunkEnsembler(arms)
            previous_action = None
            success, info = False, {}
            for step in range(max_steps):
                batch = history.batch(observation, previous_action, device)
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
                normalized = proposals.actions[0, 0, : len(arms)]
                raw = normalized * stats["a_std"][None, None] + stats["a_mean"][None, None]
                action = ensemble.append_and_select(step, raw.float().cpu().numpy())
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
                "route": "r12e1_high_resolution_specialist",
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
    if args.task in config.deployment["protected_tasks"]:
        evaluated, latencies, update = evaluate_w10(
            config, args.task, remaining, device, args.max_steps
        ) if remaining else ([], [], None)
        route = "exact_w10_fallback"
    elif args.task in config.deployment["specialist_tasks"]:
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
        ) if remaining else ([], [], None)
        route = "r12e1_high_resolution_specialist"
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
        "policy_inputs": "native 480x640 RGB first; W11 TeamBeliefState and task ID supplemental on specialist route",
        "privileged_inputs": False,
        "control_cadence": "one proposal per environment step",
        "temporal_aggregation": "W10 exponential chunk ensemble decay=0.01",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result | {"rows": "saved"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
