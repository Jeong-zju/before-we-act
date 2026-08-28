from __future__ import annotations

import argparse
import json
import os
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml
import tasks  # noqa: F401

from .common import FROZEN_CONFIG, POLICY_CONTRACT, TASKS, atomic_json, load_frozen_config
from .policy import LocalLatentToMPolicy


def scalar(value):
    if torch.is_tensor(value): return bool(value.detach().cpu().reshape(-1)[0])
    return bool(np.asarray(value).reshape(-1)[0])


def make_env(config: Path):
    task_name = yaml.safe_load(config.read_text())["task_name"]
    return gym.make(task_name + "-rf", config=str(config), obs_mode="rgb", control_mode="pd_joint_pos",
                    render_mode="rgb_array", num_envs=1, sim_backend="cpu",
                    sensor_configs={"shader_pack": "default", "width": 320, "height": 240},
                    human_render_camera_configs={"shader_pack": "default"}, viewer_camera_configs={"shader_pack": "default"})


@torch.no_grad()
def run(args):
    frozen = load_frozen_config(args.config)
    validation = frozen["validation20"]
    expected_episodes = 1 if args.smoke else validation["episodes_per_task"]
    expected_diffusion_steps = validation["diffusion_steps"]
    expected_replan_interval = validation["replan_interval"]
    if args.episodes is None: args.episodes = expected_episodes
    if args.device is None: args.device = frozen["runtime"]["evaluation_device"]
    if args.diffusion_steps is None: args.diffusion_steps = expected_diffusion_steps
    if args.replan_interval is None: args.replan_interval = expected_replan_interval
    if args.episodes != expected_episodes and not args.smoke: raise ValueError("episodes disagree with frozen config")
    if args.diffusion_steps != expected_diffusion_steps: raise ValueError("diffusion steps disagree with frozen config")
    if args.replan_interval != expected_replan_interval: raise ValueError("replan interval disagrees with frozen config")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("contract") != POLICY_CONTRACT: raise RuntimeError("strict-local checkpoint contract mismatch")
    if payload.get("frozen_config") and payload["frozen_config"] != frozen:
        raise RuntimeError("checkpoint frozen config does not match project config")
    device = torch.device(args.device); model = LocalLatentToMPolicy.from_frozen_config(frozen).to(device); model.load_state_dict(payload.get("ema_model", payload["model"])); model.set_stats(payload["stats"]); model.eval()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); summary_rows = {}
    # RoboFactory's official MARS branch resolves scene meshes relative to the
    # repository root rather than the YAML path.
    os.chdir(Path(args.config_root).resolve().parents[1])
    for task_index, spec in enumerate(TASKS):
        rows = []
        for episode in range(args.episodes):
            seed = (990000 + task_index * 1000 + episode) if args.smoke else spec.seed_start + episode
            row = {"episode": episode, "seed": seed, "success": False, "steps": 0}; env = None
            try:
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                env = make_env(Path(args.config_root) / spec.config); obs, _ = env.reset(seed=seed)
                images = [deque(maxlen=2) for _ in range(spec.arms)]; qposes = [deque(maxlen=2) for _ in range(spec.arms)]; chunks = [None] * spec.arms; offsets = [0] * spec.arms
                for step in range(2 if args.smoke else spec.max_steps):
                    own = []
                    for arm in range(spec.arms):
                        image = np.asarray(obs["sensor_data"][f"head_camera_agent{arm}"]["rgb"])[0]
                        qpos = np.asarray(obs["agent"][f"panda-{arm}"]["qpos"])[0, :9]
                        images[arm].append(image); qposes[arm].append(qpos)
                        while len(images[arm]) < 2: images[arm].appendleft(images[arm][0]); qposes[arm].appendleft(qposes[arm][0])
                        own.append({"image": torch.from_numpy(np.moveaxis(np.asarray(images[arm]), -1, 1).copy())[None].to(device), "qpos": torch.from_numpy(np.asarray(qposes[arm], np.float32))[None].to(device)})
                    replanners = [arm for arm in range(spec.arms) if chunks[arm] is None or offsets[arm] >= args.replan_interval]
                    if replanners:
                        batch = {key: torch.cat([own[arm][key] for arm in replanners]) for key in ("image", "qpos")}
                        predicted = model.predict_chunk(batch, steps=args.diffusion_steps).float().cpu().numpy()
                        for index, arm in enumerate(replanners): chunks[arm], offsets[arm] = predicted[index], 1
                    action = {}
                    for arm in range(spec.arms):
                        space = env.action_space.spaces[f"panda-{arm}"]
                        value = np.clip(chunks[arm][offsets[arm]], np.asarray(space.low), np.asarray(space.high)).astype(np.float32); offsets[arm] += 1; action[f"panda-{arm}"] = value
                    obs, _, terminated, truncated, info = env.step(action); row["success"] = scalar(info.get("success", False)); row["steps"] = step + 1
                    if row["success"] or scalar(terminated) or scalar(truncated): break
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                if env is not None: env.close()
            rows.append(row); print(json.dumps({"task": spec.name, **row}), flush=True)
        task_result = {"schema": "mars-control.latent-tom.validation20.task.v1", "status": "failed" if any("error" in row for row in rows) else "complete", "task": spec.name, "episodes": len(rows), "target_episodes": args.episodes, "successes": sum(int(row["success"]) for row in rows), "success_rate": sum(int(row["success"]) for row in rows) / len(rows), "policy_contract": POLICY_CONTRACT, "preprocessing": {"rgb": "uint8_div_255", "qpos": "global_mean_std", "action": "global_mean_std_then_env_clip", "ddim_clip_sample": False}, "episodes_detail": rows}
        atomic_json(output / f"{spec.name}.json", task_result); summary_rows[spec.name] = task_result
    errors = [row for result in summary_rows.values() for row in result["episodes_detail"] if "error" in row]
    summary = {"schema": "mars-control.latent-tom.validation20.v1", "status": "failed" if errors else "complete", "episodes_per_task": args.episodes, "total_episodes": args.episodes * len(TASKS), "tasks": summary_rows, "macro_success_rate": float(np.mean([result["success_rate"] for result in summary_rows.values()])), "policy_contract": POLICY_CONTRACT, "diffusion_steps": args.diffusion_steps, "replan_interval": args.replan_interval, "sim_backend": "cpu"}
    atomic_json(output / "summary.json", summary)
    if errors: raise RuntimeError(f"{len(errors)} validation episodes failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", required=True); parser.add_argument("--config", type=Path, default=FROZEN_CONFIG); parser.add_argument("--config-root", default="/workspace/repos/RoboFactory/configs/table"); parser.add_argument("--episodes", type=int); parser.add_argument("--device"); parser.add_argument("--diffusion-steps", type=int); parser.add_argument("--replan-interval", type=int); parser.add_argument("--smoke", action="store_true"); run(parser.parse_args())
