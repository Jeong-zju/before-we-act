"""Core-free closed-loop R12 evaluator on the frozen paired Gate20 seeds."""
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

from before_we_act.action_generator.base import JointActionGenerator, load_r12_config
from before_we_act.benchmark import TASKS, get_task
from before_we_act.team_belief.base import PredictiveBeliefModel, load_r11_config


def reset_reproducibly(env, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return env.reset(seed=seed)


def patch_means(image: np.ndarray, grid: int = 4) -> np.ndarray:
    height, width, channels = image.shape
    if channels != 3 or height % grid or width % grid:
        raise ValueError("R12 fixed RGB must be divisible into the frozen 4x4 grid")
    value = image.reshape(grid, height // grid, grid, width // grid, 3)
    return value.mean(axis=(1, 3), dtype=np.float32).reshape(grid * grid, 3) / 127.5 - 1.0


def observation_row(observation, arms, previous_action):
    sensors = observation["sensor_data"]
    global_rgb = np.asarray(sensors["head_camera_global"]["rgb"])
    global_rgb = global_rgb[0] if global_rgb.ndim == 4 else global_rgb
    images = [global_rgb]
    for arm in arms:
        local = np.asarray(sensors[f"head_camera_agent{arm}"]["rgb"])
        images.append(local[0] if local.ndim == 4 else local)
    visual = np.zeros((16, 15), dtype=np.float32)
    view_mask = np.zeros(5, dtype=np.float32)
    for index, image in enumerate(images):
        visual[:, index * 3 : (index + 1) * 3] = patch_means(image)
        view_mask[index] = 1.0
    qpos = np.zeros((4, 9), dtype=np.float32)
    actions = np.zeros((4, 8), dtype=np.float32)
    for index, arm in enumerate(arms):
        raw = np.asarray(observation["agent"][f"panda-{arm}"]["qpos"])
        qpos[index] = raw[0] if raw.ndim == 2 else raw
        if previous_action is not None:
            actions[index] = previous_action[f"panda-{arm}"]
    return visual, view_mask, qpos, actions


class TeamHistory:
    def __init__(self, arms):
        self.arms = tuple(arms)
        self.rows = []

    def batch(self, observation, previous_action, device):
        row = observation_row(observation, self.arms, previous_action)
        self.rows.append(row)
        self.rows = self.rows[-3:]
        padded = [self.rows[0]] * (3 - len(self.rows)) + self.rows
        agent_mask = torch.zeros((1, 4), dtype=torch.bool, device=device)
        agent_mask[:, : len(self.arms)] = True
        return {
            "visual": torch.from_numpy(np.stack([item[0] for item in padded])).unsqueeze(0).to(device),
            "view_mask": torch.from_numpy(np.stack([item[1] for item in padded])).unsqueeze(0).to(device),
            "qpos": torch.from_numpy(np.stack([item[2] for item in padded])).unsqueeze(0).to(device),
            "actions": torch.from_numpy(np.stack([item[3] for item in padded])).unsqueeze(0).to(device),
            "agent_mask": agent_mask,
        }


class TemporalChunkEnsembler:
    """Frozen append/prune/exponential aggregation used by W10."""

    def __init__(self, arms, decay: float = 0.01):
        self.arms = tuple(arms)
        self.decay = float(decay)
        self.histories = [[] for _ in self.arms]

    def append_and_select(self, step: int, chunks: np.ndarray):
        if len(chunks) != len(self.arms):
            raise ValueError("joint chunk/arm count mismatch")
        action = {}
        for local_index, arm in enumerate(self.arms):
            self.histories[local_index].append((step, chunks[local_index]))
            self.histories[local_index] = [
                item for item in self.histories[local_index]
                if step - item[0] < len(item[1])
            ]
            candidates = np.asarray(
                [chunk[step - start] for start, chunk in self.histories[local_index]]
            )
            weights = np.exp(-self.decay * np.arange(len(candidates) - 1, -1, -1))
            weights /= weights.sum()
            action[f"panda-{arm}"] = np.sum(candidates * weights[:, None], axis=0)
        return action


def load_models(config_path, checkpoint_path, belief_config_path, belief_checkpoint_path, device):
    config = load_r12_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["candidate_id"] != config.candidate_id or not checkpoint.get("core_free_runtime"):
        raise ValueError("R12 checkpoint identity/core-free declaration differs")
    generator = JointActionGenerator(config).to(device)
    generator.load_state_dict(checkpoint["model"], strict=True)
    generator.eval()
    belief_config = load_r11_config(belief_config_path)
    belief_saved = torch.load(belief_checkpoint_path, map_location="cpu", weights_only=False)
    belief = PredictiveBeliefModel(belief_config).to(device)
    belief.load_state_dict(belief_saved["model"], strict=True)
    belief.eval()
    stats = {key: torch.as_tensor(value, device=device) for key, value in checkpoint["stats"].items()}
    return config, generator, belief, stats


@torch.no_grad()
def evaluate(config_path, checkpoint_path, belief_config_path, belief_checkpoint_path, task_name, seeds, device_name, max_steps):
    torch.set_num_threads(12)
    device = torch.device(device_name)
    config, generator, belief, stats = load_models(
        config_path, checkpoint_path, belief_config_path, belief_checkpoint_path, device
    )
    specification = get_task(task_name)
    arms = specification["agents"]
    env = gym.make(
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
    rows, latencies = [], []
    for seed in seeds:
        observation, _ = reset_reproducibly(env, seed)
        history = TeamHistory(arms)
        ensembler = TemporalChunkEnsembler(arms)
        previous_action = None
        success = False
        projection_count = 0
        for step in range(max_steps):
            batch = history.batch(observation, previous_action, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter_ns()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                belief_state = belief(batch)["belief"]
                noise_generator = torch.Generator(device=device).manual_seed(
                    int(seed) * 1_000_003 + step
                )
                noise = torch.randn((1, 100, 32), generator=noise_generator, device=device)
                proposals = generator.sample(belief_state, noise=noise)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies.append((time.perf_counter_ns() - started) / 1e6)
            normalized = proposals.actions[0, 0, : len(arms)]
            raw = normalized * stats["a_std"][None, None] + stats["a_mean"][None, None]
            chunks = raw.float().cpu().numpy()
            action = ensembler.append_and_select(step, chunks)
            for key, value in action.items():
                if not np.isfinite(value).all():
                    raise FloatingPointError(f"non-finite action for {key}")
            previous_action = {key: value.copy() for key, value in action.items()}
            observation, _, terminated, truncated, info = env.step(action)
            success = bool(np.asarray(info.get("success", False)).all())
            if success or bool(np.asarray(terminated).all()) or bool(np.asarray(truncated).all()):
                break
        row = {
            "task": task_name,
            "seed": int(seed),
            "success": success,
            "steps": step + 1,
            "safety_projections": projection_count,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    env.close()
    return rows, latencies, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--belief-config", required=True)
    parser.add_argument("--belief-checkpoint", required=True)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-log", default="")
    args = parser.parse_args()
    seed_path = Path(args.seed_file).resolve(strict=True)
    raw = seed_path.read_bytes()
    seed_manifest = json.loads(raw)
    all_seeds = [int(seed) for seed in seed_manifest["seeds"]]
    if args.episodes != 20 or len(all_seeds) < 20:
        raise ValueError("R12 Gate20 requires exactly the first 20 frozen seeds")
    requested = all_seeds[:20]
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
                recovered.append(
                    {key: row[key] for key in ("task", "seed", "success", "steps", "safety_projections")}
                )
    recovered = list({row["seed"]: row for row in recovered}.values())
    complete = {row["seed"] for row in recovered}
    remaining = [seed for seed in requested if seed not in complete]
    if remaining:
        rows, latencies, config = evaluate(
            args.config,
            args.checkpoint,
            args.belief_config,
            args.belief_checkpoint,
            args.task,
            remaining,
            args.device,
            args.max_steps,
        )
    else:
        rows, latencies = [], []
        config = load_r12_config(args.config)
    rows = recovered + rows
    rows.sort(key=lambda row: requested.index(row["seed"]))
    latency_values = np.asarray(latencies, dtype=np.float64)
    result = {
        "schema_version": 1,
        "round": "R12",
        "candidate_id": config.candidate_id,
        "task": args.task,
        "episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "rows": rows,
        "latency_ms": {
            "samples": len(latencies),
            "p50": float(np.percentile(latency_values, 50)) if len(latencies) else None,
            "p95": float(np.percentile(latency_values, 95)) if len(latencies) else None,
        },
        "seed_protocol": {"source": str(seed_path), "sha256": hashlib.sha256(raw).hexdigest()},
        "policy_inputs": "fixed-view RGB patch means, qpos, executed-action history, masks, W11 belief",
        "privileged_inputs": False,
        "core_free_runtime": True,
        "control_cadence": "one proposal per environment step",
        "temporal_aggregation": "W10 exponential chunk ensemble decay=0.01",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result | {"rows": "saved"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
