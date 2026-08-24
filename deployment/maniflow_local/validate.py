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
import yaml
import robofactory.tasks  # noqa: F401

from local_dataset import TASKS
from modeling import load_policy, model_config

MAX_STEPS = {"lift_barrier": 500, "camera_alignment": 1500, "long_pipeline_delivery": 1500,
             "take_photo": 1500, "pass_shoe": 500, "place_food": 500}


def atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def scalar_bool(value):
    if torch.is_tensor(value): return bool(value.detach().cpu().reshape(-1)[0].item())
    return bool(np.asarray(value).reshape(-1)[0])


def make_env(config: Path):
    task_name = yaml.safe_load(config.read_text())["task_name"]
    return gym.make(task_name + "-rf", config=str(config), obs_mode="rgb", control_mode="pd_joint_pos",
                    render_mode="rgb_array", num_envs=1, sim_backend="cpu",
                    sensor_configs={"shader_pack": "default", "width": 320, "height": 240},
                    human_render_camera_configs={"shader_pack": "default"},
                    viewer_camera_configs={"shader_pack": "default"})


def bounds(dataset_root: Path, task: str, agent: int):
    manifest = json.loads((dataset_root / task / "training_manifest.json").read_text())
    codec = manifest["action"]["codec"]["config"]
    low = np.asarray(codec["low"][agent * 8:(agent + 1) * 8], np.float32)
    high = np.asarray(codec["high"][agent * 8:(agent + 1) * 8], np.float32)
    return low, high


def success(info):
    return scalar_bool(info.get("success", False) if isinstance(info, dict) else False)


class Ensemble:
    def __init__(self, count: int, decay: float): self.histories = [[] for _ in range(count)]; self.decay = float(decay)
    def add(self, step: int, chunks: list[np.ndarray]):
        for agent, chunk in enumerate(chunks): self.histories[agent].append((step, chunk))
    def select(self, step: int):
        result = {}
        for agent, history in enumerate(self.histories):
            history[:] = [(s, x) for s, x in history if step - s < len(x)]
            candidates = np.asarray([x[step - s] for s, x in history], np.float32)
            weights = np.exp(-self.decay * np.arange(len(candidates) - 1, -1, -1, dtype=np.float32)); weights /= weights.sum()
            result[f"panda-{agent}"] = np.sum(candidates * weights[:, None], axis=0).astype(np.float32)
        return result


@torch.no_grad()
def evaluate(args):
    if args.formal and (args.episodes != 20 or args.seed != 20260820 or args.sim_backend != "cpu" or args.replan_interval != 8):
        raise ValueError("formal Validation20 requires episodes=20, seed=20260820, CPU sim and replan interval 8")
    torch.set_num_threads(args.cpu_threads)
    policy, payload = load_policy(args.checkpoint, args.device)
    if payload.get("contract") != model_config()["policy_contract"]: raise RuntimeError("local checkpoint contract mismatch")
    dataset_root, config_root, out = Path(args.dataset_root), Path(args.config_root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True); reports = {}
    for task in args.task or TASKS:
        result_path = out / f"{task}.json"; previous = {}
        if result_path.is_file():
            try: previous = json.loads(result_path.read_text())
            except json.JSONDecodeError: pass
        episode_map = {int(row["episode"]): row for row in previous.get("episodes_detail", []) if not row.get("error")}
        config = config_root / f"{task}.yaml"; agent_count = len(yaml.safe_load(config.read_text())["agents"])
        lows_highs = [bounds(dataset_root, task, i) for i in range(agent_count)]
        for episode in range(args.episodes):
            if episode in episode_map: continue
            env = None; row = {"episode": episode, "seed": args.seed + episode, "success": False, "steps": 0}
            try:
                env = make_env(config); obs, _ = env.reset(seed=args.seed + episode)
                image_hist = [deque(maxlen=2) for _ in range(agent_count)]; qpos_hist = [deque(maxlen=2) for _ in range(agent_count)]
                ensemble = Ensemble(agent_count, 0.01); chunks = [None] * agent_count; offsets = [args.replan_interval] * agent_count
                for step in range(MAX_STEPS[task]):
                    local = []
                    for agent in range(agent_count):
                        image = obs["sensor_data"][f"head_camera_agent{agent}"]["rgb"][0].detach().cpu().numpy()
                        qpos = obs["agent"][f"panda-{agent}"]["qpos"][0, :9].detach().cpu().numpy()
                        image_hist[agent].append(image); qpos_hist[agent].append(qpos)
                        while len(image_hist[agent]) < 2: image_hist[agent].appendleft(image_hist[agent][0])
                        while len(qpos_hist[agent]) < 2: qpos_hist[agent].appendleft(qpos_hist[agent][0])
                        local.append({"head_cam": np.asarray(image_hist[agent]), "agent_pos": np.asarray(qpos_hist[agent])})
                    if any(x is None or offsets[i] >= args.replan_interval for i, x in enumerate(chunks)):
                        images = torch.from_numpy(np.stack([x["head_cam"] for x in local])).to(args.device)
                        qposes = torch.from_numpy(np.stack([x["agent_pos"] for x in local])).to(args.device)
                        pred = policy.predict_action({"head_cam": images, "agent_pos": qposes})["action"].float().cpu().numpy()
                        chunks = [pred[i] for i in range(agent_count)]; offsets = [0] * agent_count; ensemble.add(step, chunks)
                    action = ensemble.select(step)
                    for agent, (low, high) in enumerate(lows_highs):
                        encoded = np.clip(action[f"panda-{agent}"], -1, 1)
                        action[f"panda-{agent}"] = (low + 0.5 * (encoded + 1) * (high - low)).astype(np.float32); offsets[agent] += 1
                    obs, _, terminated, truncated, info = env.step(action); row["success"], row["steps"] = success(info), step + 1
                    if row["success"] or scalar_bool(terminated) or scalar_bool(truncated): break
            except Exception as exc: row["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                if env is not None: env.close()
            episode_map[episode] = row; detail = [episode_map[k] for k in sorted(episode_map)]; successes = sum(bool(x["success"]) for x in detail)
            atomic_json(result_path, {"schema": "bwa.maniflow.validation20.task.v1", "task": task,
                "status": "failed" if any(x.get("error") for x in detail) else ("complete" if len(detail) == args.episodes else "running"),
                "successes": successes, "episodes": len(detail), "target_episodes": args.episodes,
                "success_rate": successes / len(detail), "max_steps": MAX_STEPS[task], "seed_base": args.seed,
                "sim_backend": "cpu", "policy_contract": model_config()["policy_contract"], "episodes_detail": detail,
                "updated_at": datetime.now(timezone.utc).isoformat()})
        reports[task] = json.loads(result_path.read_text())
    errors = [e for r in reports.values() for e in r["episodes_detail"] if e.get("error")]
    summary = {"schema": "bwa.maniflow.validation20.v1", "status": "failed" if errors else "complete",
               "baseline": "maniflow", "episodes_per_task": args.episodes, "total_episodes": sum(r["episodes"] for r in reports.values()),
               "tasks": reports, "macro_success_rate": float(np.mean([r["success_rate"] for r in reports.values()])),
               "seed_base": args.seed, "sim_backend": "cpu", "policy_contract": model_config()["policy_contract"],
               "completed_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(out / "summary.json", summary)
    if errors: raise RuntimeError(f"{len(errors)} validation episodes failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); p.add_argument("--dataset-root", default=os.getenv("BWA_DATASET_ROOT", "/workspace/datasets/robofactory_multitask")); p.add_argument("--config-root", default="/workspace/repos/RoboFactory/robofactory/configs/table"); p.add_argument("--output", required=True); p.add_argument("--episodes", type=int, default=20); p.add_argument("--seed", type=int, default=20260820); p.add_argument("--device", default="cuda:0"); p.add_argument("--cpu-threads", type=int, default=16); p.add_argument("--replan-interval", type=int, default=8); p.add_argument("--sim-backend", default="cpu"); p.add_argument("--task", action="append", choices=TASKS); p.add_argument("--formal", action="store_true"); evaluate(p.parse_args())
