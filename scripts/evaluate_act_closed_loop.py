#!/usr/bin/env python3
"""Run ACT checkpoints in the RoboFactory simulator and write Validation20 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import torch
import yaml
import robofactory.tasks  # noqa: F401 - registers the six Gym environments

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from stereo_core.train_act import ACT

TASKS = (
    "lift_barrier",
    "camera_alignment",
    "long_pipeline_delivery",
    "take_photo",
    "pass_shoe",
    "place_food",
)


def _action_codec(stats_root: Path, task: str, arm: int):
    payload = json.loads((stats_root / task / "training_manifest.json").read_text())
    cfg = payload["action"]["codec"]["config"]
    low = np.asarray(cfg["low"], np.float32)[arm * 8:(arm + 1) * 8]
    high = np.asarray(cfg["high"], np.float32)[arm * 8:(arm + 1) * 8]
    return low, high


def _success(info) -> bool:
    value = info.get("success", False) if isinstance(info, dict) else False
    if torch.is_tensor(value):
        return bool(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        return bool(value.reshape(-1)[0])
    return bool(value)


def _make_env(config: Path):
    with config.open() as handle:
        env_id = yaml.safe_load(handle)["task_name"] + "-rf"
    return gym.make(
        env_id,
        config=str(config),
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend="gpu",
        sensor_configs={"shader_pack": "minimal"},
    )


@torch.no_grad()
def _predict(model, obs, arm: int, stats, codec, device):
    image = obs["sensor_data"][f"head_camera_agent{arm}"]["rgb"]
    image = image.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    qpos = obs["agent"][f"panda-{arm}"]["qpos"][:, :9].to(device=device)
    qpos = (qpos - torch.as_tensor(stats["q_mean"], device=device)) / torch.as_tensor(stats["q_std"], device=device)
    pred, _, _ = model(image, qpos)
    normalized = pred[:, 0].float().cpu().numpy()[0]
    normalized = normalized * stats["a_std"] + stats["a_mean"]
    low, high = codec
    return np.clip((normalized + 1.0) * 0.5 * (high - low) + low, low, high).astype(np.float32)


def evaluate(args) -> dict:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = ACT(
        int(config["state_dim"]), int(config["action_dim"]), int(config["horizon"]),
        int(config["d_model"]), int(config["enc_layers"]), int(config["dec_layers"]),
        vision_backbone=config.get("vision_backbone", "resnet18"),
        dino_model=config.get("dino_model", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    stats = {key: np.asarray(value, np.float32) for key, value in checkpoint["stats"].items()}
    stats_root = Path(args.stats_root)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for task in TASKS:
        config_path = Path(args.config_root) / f"{task}.yaml"
        # All current RoboFactory six-task configs use two local Panda policies;
        # reading the config keeps this evaluator valid if that changes.
        with config_path.open() as handle:
            agent_count = len(yaml.safe_load(handle)["agents"])
        successes = 0
        episodes = []
        for episode in range(args.episodes):
            env = _make_env(config_path)
            obs, _ = env.reset(seed=args.seed + episode)
            success = False
            steps = 0
            for steps in range(1, args.max_steps + 1):
                action = {
                    f"panda-{arm}": _predict(
                        model, obs, arm, stats, _action_codec(stats_root, task, arm), args.device
                    )
                    for arm in range(agent_count)
                }
                obs, _, terminated, truncated, info = env.step(action)
                success = _success(info)
                if success or bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
                    break
            env.close()
            successes += int(success)
            episodes.append({"episode": episode, "seed": args.seed + episode, "success": success, "steps": steps})
        summary = {"baseline": "act", "task": task, "successes": successes, "episodes": args.episodes,
                   "success_rate": successes / args.episodes, "episodes_detail": episodes}
        (output / f"{task}.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries[task] = summary
    report = {"baseline": "act", "episodes_per_task": args.episodes, "tasks": summaries,
              "macro_success_rate": float(np.mean([row["success_rate"] for row in summaries.values()]))}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats-root", required=True)
    parser.add_argument("--config-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--device", default="cuda:0")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
