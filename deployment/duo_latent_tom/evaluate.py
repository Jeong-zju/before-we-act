from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torchvision.transforms.v2 import functional as TVF

from .common import ACTION_ENCODING, FROZEN_CONFIG, POLICY_CONTRACT, TASKS, VALIDATION_MAX_STEPS, atomic_json, load_config
from .policy import LocalLatentToMPolicy
from ..duo_act.action_target import canonicalize_controller_action


def _scalar(value) -> bool:
    if torch.is_tensor(value):
        return bool(value.detach().cpu().reshape(-1)[0])
    return bool(np.asarray(value).reshape(-1)[0])


def make_env(task: str):
    module = __import__(f"duobench.tasks.{task}", fromlist=["*"])
    config_class = getattr(module, "".join(part.title() for part in task.split("_")) + "EnvConfig")
    from rcs._core.sim import SimConfig
    from rcs.envs.base import ControlMode, RelativeTo
    config = config_class().config()
    config.headless = True
    config.control_mode = ControlMode.JOINTS
    config.relative_to = RelativeTo.NONE
    config.sim_cfg = SimConfig(async_control=True, realtime=False, frequency=30)
    config.wrapper_cfg.binary_gripper = True
    return gym.make(f"duobench/{task}", cfg=config)


def _frame(value) -> np.ndarray:
    if isinstance(value, dict):
        value = value.get("rgb", value)
        if isinstance(value, dict):
            value = value.get("data", value.get("image", value))
    value = np.asarray(value)
    if value.ndim == 4:
        value = value[0]
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"expected RGB HWC, got {value.shape}")
    if value.dtype != np.uint8:
        if np.issubdtype(value.dtype, np.floating) and float(value.max(initial=0)) <= 1:
            value = value * 255
        value = np.asarray(value, dtype=np.uint8)
    return np.ascontiguousarray(value)


def local_observation(observation: dict, arm: int, image_history: deque, qpos_history: deque,
                      task_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    key = "left" if arm == 0 else "right"
    joints = np.asarray(observation[key]["joints"], dtype=np.float32).reshape(-1)
    gripper = np.asarray(observation[key]["gripper"], dtype=np.float32).reshape(-1)
    if joints.size < 7 or gripper.size < 1:
        raise ValueError("DuoBench arm proprioception shape drift")
    qpos = np.concatenate((joints[:7], [float(gripper[0] > 0.9)])).astype(np.float32)
    frames = observation["frames"]
    head = _frame(frames["head"])
    wrist = _frame(frames["left_wrist" if arm == 0 else "right_wrist"])
    views = torch.from_numpy(np.stack((head, wrist)).copy()).permute(0, 3, 1, 2)
    views = TVF.resize(views, (224, 224), antialias=True)
    image = torch.cat((views[0], views[1]), dim=2)
    image_history.append(np.asarray(image))
    qpos_history.append(qpos)
    while len(image_history) < 2:
        image_history.appendleft(image_history[0].copy())
        qpos_history.appendleft(qpos_history[0].copy())
    return {
        "image": torch.from_numpy(np.stack(tuple(image_history)).copy()).to(device).unsqueeze(0),
        "qpos": torch.from_numpy(np.stack(tuple(qpos_history)).copy()).to(device).unsqueeze(0),
        "task": torch.nn.functional.one_hot(torch.tensor([task_id], device=device), num_classes=len(TASKS)).float(),
        "arm_id": torch.nn.functional.one_hot(torch.tensor([arm], device=device), num_classes=2).float(),
    }


@torch.inference_mode()
def evaluate_task(*, checkpoint: Path, data: Path, output: Path, task: str,
                  episodes: int, seed_base: int, diffusion_steps: int,
                  replan_interval: int, device: str, max_steps_override: int | None = None) -> dict:
    config = load_config()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("policy_contract") != POLICY_CONTRACT:
        raise ValueError("checkpoint strict-local contract mismatch")
    model = LocalLatentToMPolicy.from_config(config).to(device)
    model.load_state_dict(payload.get("ema_model", payload["model"]), strict=True)
    model.set_stats(payload["stats"]); model.eval()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"{task}.json"
    recovered = {}
    if result_path.is_file():
        try:
            recovered = {int(row["episode"]): row for row in json.loads(result_path.read_text()).get("episodes_detail", [])}
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            recovered = {}
    rows = []
    env = make_env(task)
    try:
        for episode in range(episodes):
            if episode in recovered and not recovered[episode].get("error"):
                rows.append(recovered[episode]); continue
            seed = seed_base + TASKS.index(task) * 1000 + episode
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            max_steps = int(max_steps_override or VALIDATION_MAX_STEPS[task])
            row = {"task": task, "episode": episode, "seed": seed, "success": False, "steps": 0,
                   "max_steps": max_steps}
            try:
                observation, _ = env.reset(seed=seed)
                histories = [deque(maxlen=2) for _ in range(2)]
                qhistories = [deque(maxlen=2) for _ in range(2)]
                chunks: list[np.ndarray | None] = [None, None]
                offsets = [0, 0]
                for step in range(max_steps):
                    own = [local_observation(observation, arm, histories[arm], qhistories[arm], TASKS.index(task), model.q_mean.device) for arm in range(2)]
                    replanners = [arm for arm in range(2) if chunks[arm] is None or offsets[arm] >= replan_interval]
                    if replanners:
                        batch = {key: torch.cat([own[arm][key] for arm in replanners], dim=0) for key in ("image", "qpos", "task", "arm_id")}
                        predicted = model.predict_chunk(batch, steps=diffusion_steps).float().cpu().numpy()
                        for position, arm in enumerate(replanners):
                            chunks[arm] = predicted[position]
                            # This causal dataset conditions on observation[t]
                            # and predicts action[t+1] at chunk index zero.
                            offsets[arm] = 0
                    action = {}
                    for arm, key in enumerate(("left", "right")):
                        local = canonicalize_controller_action(chunks[arm][offsets[arm]])
                        offsets[arm] += 1
                        action[key] = {"joints": local[:7].astype(np.float32), "gripper": np.asarray([local[7]], dtype=np.float32)}
                    observation, _reward, terminated, truncated, info = env.step(action)
                    row["success"] = _scalar(info.get("success", False)); row["steps"] = step + 1
                    if row["success"] or _scalar(terminated) or _scalar(truncated):
                        break
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            rows.append(row)
            atomic_json(result_path, {"schema": "duobench.latent-tom.validation20.task.v1", "status": "running", "task": task, "episodes": len(rows), "target_episodes": episodes, "successes": sum(int(x["success"]) for x in rows), "episodes_detail": rows, "policy_contract": POLICY_CONTRACT, "action_encoding": ACTION_ENCODING})
            print(json.dumps(row), flush=True)
    finally:
        env.close()
    errors = [row for row in rows if row.get("error")]
    result = {"schema": "duobench.latent-tom.validation20.task.v2", "status": "failed" if errors else "complete", "task": task, "episodes": len(rows), "target_episodes": episodes, "successes": sum(int(x["success"]) for x in rows), "success_rate": float(np.mean([x["success"] for x in rows])), "max_steps": VALIDATION_MAX_STEPS[task], "episodes_detail": rows, "policy_contract": POLICY_CONTRACT, "action_encoding": ACTION_ENCODING, "preprocessing": {"rgb": "uint8_div_255_imagenet_norm", "qpos": "population_minmax_to_minus1_plus1_with_binary_gripper", "action": "population_minmax_from_minus1_plus1_then_pinned_controller_ctrlrange", "ddim_clip_sample": True, "diffusion_steps": diffusion_steps, "replan_interval": replan_interval}}
    atomic_json(result_path, result)
    if errors:
        raise RuntimeError(f"{task}: {len(errors)} rollout errors")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--episodes", type=int, default=20); parser.add_argument("--seed-base", type=int, default=20260820)
    parser.add_argument("--diffusion-steps", type=int, default=100); parser.add_argument("--replan-interval", type=int, default=20)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(); validation = config["validation20"]
    if not args.smoke and (args.episodes != validation["episodes_per_task"] or args.seed_base != validation["seed_base"] or args.diffusion_steps != validation["diffusion_steps"] or args.replan_interval != validation["replan_interval"]):
        raise ValueError("formal validation arguments differ from frozen protocol")
    episodes = 1 if args.smoke else args.episodes
    tasks = [args.task] if args.task else list(TASKS)
    results = [evaluate_task(checkpoint=args.checkpoint, data=args.data, output=args.output, task=task, episodes=episodes, seed_base=args.seed_base, diffusion_steps=args.diffusion_steps, replan_interval=args.replan_interval, device=args.device, max_steps_override=2 if args.smoke else None) for task in tasks]
    if args.task:
        return
    summary = {"schema": "duobench.latent-tom.validation20.v1", "status": "complete", "episodes_per_task": episodes, "total_episodes": sum(x["episodes"] for x in results), "successes": sum(x["successes"] for x in results), "macro_success_rate": float(np.mean([x["success_rate"] for x in results])), "tasks": {x["task"]: x for x in results}, "policy_contract": POLICY_CONTRACT, "seed_base": args.seed_base, "diffusion_steps": args.diffusion_steps, "replan_interval": args.replan_interval, "sim_backend": "cpu", "completed_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "total_episodes", "successes", "macro_success_rate")}))


if __name__ == "__main__":
    main()
