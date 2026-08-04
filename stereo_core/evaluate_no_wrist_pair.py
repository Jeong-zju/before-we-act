"""Closed-loop evaluation for the fixed global+agent RGB Stereo-CoRE policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import robofactory  # noqa: F401
try:
    from .no_wrist_pair_model import NoWristPAIRRoute
    from .two_three_task_manifest import TASKS, get_task
except ImportError:
    from no_wrist_pair_model import NoWristPAIRRoute
    from two_three_task_manifest import TASKS, get_task


def reset_reproducibly(env, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return env.reset(seed=seed)


def load_model(checkpoint: str, device: torch.device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    if config.get("policy_variant") != "no_wrist_rgb_pair_route":
        raise ValueError(f"wrong checkpoint variant: {config.get('policy_variant')}")
    model = NoWristPAIRRoute(
        config.get("state_dim", 9),
        config.get("action_dim", 8),
        horizon=config.get("horizon", 100),
        d_model=config.get("d_model", 384),
        enc_layers=config.get("enc_layers", 4),
        dec_layers=config.get("dec_layers", 7),
        roles=config.get("roles", 4),
        role_rank=config.get("role_rank", 32),
        dino_model=config["dino_model"],
    ).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    stats = {
        key: torch.as_tensor(value, device=device)
        for key, value in saved["stats"].items()
    }
    return model, stats, config


def prepare_no_wrist_batch(observation, arms, stats, device):
    """Freeze camera extraction and qpos normalization for one decision."""
    sensors = observation["sensor_data"]
    global_image = np.asarray(sensors["head_camera_global"]["rgb"])
    global_image = global_image[0] if global_image.ndim == 4 else global_image
    global_images, local_images, qposes = [], [], []
    for arm in arms:
        local = np.asarray(sensors[f"head_camera_agent{arm}"]["rgb"])
        qpos = np.asarray(observation["agent"][f"panda-{arm}"]["qpos"])
        global_images.append(global_image)
        local_images.append(local[0] if local.ndim == 4 else local)
        qposes.append(qpos[0] if qpos.ndim == 2 else qpos)
    global_rgb = torch.as_tensor(np.stack(global_images)).permute(0, 3, 1, 2)
    local_rgb = torch.as_tensor(np.stack(local_images)).permute(0, 3, 1, 2)
    global_rgb = global_rgb.float().div_(255).to(device)
    local_rgb = local_rgb.float().div_(255).to(device)
    qpos = torch.as_tensor(np.stack(qposes)).float().to(device)
    qpos = (qpos - stats["q_mean"]) / stats["q_std"]
    return global_rgb, local_rgb, qpos


def denormalize_action_chunks(chunks, stats):
    """Apply the parent action statistics exactly once."""
    return chunks * stats["a_std"] + stats["a_mean"]


class TemporalChunkEnsembler:
    """Frozen append/prune/exponential aggregation used by the parent."""

    def __init__(self, arms, decay: float = 0.01):
        self.arms = tuple(arms)
        self.decay = float(decay)
        self.histories = [[] for _ in self.arms]

    def append_and_select(self, step: int, chunks: np.ndarray) -> dict[str, np.ndarray]:
        if len(chunks) != len(self.arms):
            raise ValueError("chunk/arm count mismatch")
        action = {}
        for local_index, arm in enumerate(self.arms):
            self.histories[local_index].append((step, chunks[local_index]))
            self.histories[local_index] = [
                item
                for item in self.histories[local_index]
                if step - item[0] < len(item[1])
            ]
            candidates = np.asarray(
                [
                    chunk[step - start]
                    for start, chunk in self.histories[local_index]
                ]
            )
            weights = np.exp(-self.decay * np.arange(len(candidates) - 1, -1, -1))
            weights /= weights.sum()
            action[f"panda-{arm}"] = np.sum(candidates * weights[:, None], axis=0)
        return action


@torch.no_grad()
def predict_all(model, stats, observation, arms, device):
    global_rgb, local_rgb, qpos = prepare_no_wrist_batch(
        observation, arms, stats, device
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        chunks = model(global_rgb, local_rgb, qpos)[0]
    return denormalize_action_chunks(chunks, stats).float().cpu().numpy()


def evaluate(checkpoint: str, task_name: str, seeds: list[int], device_name: str, max_steps: int):
    torch.set_num_threads(12)
    device = torch.device(device_name)
    model, stats, config = load_model(checkpoint, device)
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
    rows = []
    for seed in seeds:
        observation, _ = reset_reproducibly(env, seed)
        ensembler = TemporalChunkEnsembler(arms)
        success = False
        for step in range(max_steps):
            chunks = predict_all(model, stats, observation, arms, device)
            action = ensembler.append_and_select(step, chunks)
            observation, _, terminated, truncated, info = env.step(action)
            success = bool(np.asarray(info.get("success", False)).all())
            if bool(np.asarray(terminated).all()) or bool(np.asarray(truncated).all()):
                break
        row = {"seed": seed, "success": success, "steps": step + 1}
        rows.append(row)
        print(json.dumps({"task": task_name, **row}), flush=True)
    env.close()
    return rows, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-log", default="")
    args = parser.parse_args()

    raw = Path(args.seed_file).read_bytes()
    seed_manifest = json.loads(raw)
    all_seeds = [int(seed) for seed in seed_manifest["seeds"]]
    if args.episodes < 1 or args.episodes > len(all_seeds):
        raise ValueError(f"episodes must be in [1, {len(all_seeds)}]")
    requested = all_seeds[: args.episodes]
    recovered = []
    if args.resume_log and Path(args.resume_log).is_file():
        allowed = set(requested)
        for line in Path(args.resume_log).read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                row.get("task") == args.task
                and row.get("seed") in allowed
                and isinstance(row.get("success"), bool)
                and isinstance(row.get("steps"), int)
            ):
                recovered.append({key: row[key] for key in ("seed", "success", "steps")})
        recovered = list({row["seed"]: row for row in recovered}.values())
    completed = {row["seed"] for row in recovered}
    remaining = [seed for seed in requested if seed not in completed]
    print(
        json.dumps({"task": args.task, "recovered": len(recovered), "remaining": len(remaining)}),
        flush=True,
    )
    rows, config = evaluate(
        args.checkpoint,
        args.task,
        remaining,
        args.device,
        args.max_steps,
    ) if remaining else ([], torch.load(args.checkpoint, map_location="cpu", weights_only=False)["config"])
    rows = recovered + rows
    rows.sort(key=lambda row: requested.index(row["seed"]))
    successes = sum(row["success"] for row in rows)
    result = {
        "task": args.task,
        "env_id": get_task(args.task)["env_id"],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "rows": rows,
        "camera": "fixed global RGB + matching fixed agent RGB; no wrist or depth input",
        "policy_input": config.get("policy_input"),
        "seed_protocol": {
            "source": args.seed_file,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "selection_method": seed_manifest.get("selection_method"),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result | {"rows": "saved"}), flush=True)


if __name__ == "__main__":
    main()
