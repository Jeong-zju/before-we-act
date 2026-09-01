#!/usr/bin/env python3
"""Closed-loop RoboFactory Validation20 client for a local VLA RPC worker."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import socket
import struct
import sys
import tempfile

import gymnasium as gym
import numpy as np
import torch
import yaml
mars_root = os.environ.get("MARS_ROBOFACTORY_ROOT")
if os.environ.get("BWA_MARS_CONTROL") == "1" and mars_root:
    # MARS task configs resolve simulator assets relative to the benchmark
    # checkout (for example assets/scenes/table/table.glb).
    os.chdir(mars_root)
try:
    import robofactory.tasks  # noqa: F401
except ModuleNotFoundError:
    mars_root = os.environ.get("MARS_ROBOFACTORY_ROOT")
    if not mars_root:
        raise
    sys.path.insert(0, mars_root)
    import tasks  # noqa: F401

TASK_PROMPTS = {
    "camera_alignment": "Align the cameras together",
    "lift_barrier": "Lift the barrier together",
    "long_pipeline_delivery": "Deliver the long pipeline together",
    "pass_shoe": "Pass the shoe between robots",
    "place_food": "Place the food together",
    "take_photo": "Take a photo together",
}
TASK_PROMPTS.update({
    "place_cube_in_cup": "place the cube in the cup together",
    "strike_cube_hard": "strike the cube hard together",
    "three_robots_place_shoes": "place the shoes together",
    "four_robots_stack_cube": "stack the cubes together",
})
MAX_STEPS = {
    "lift_barrier": 500,
    "camera_alignment": 1500,
    "long_pipeline_delivery": 1500,
    "take_photo": 1500,
    "pass_shoe": 500,
    "place_food": 500,
}
MAX_STEPS.update({
    "place_cube_in_cup": 500,
    "strike_cube_hard": 500,
    "three_robots_place_shoes": 1200,
    "four_robots_stack_cube": 800,
})
MARS_BOUNDS_LOW = np.asarray([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -1.0], np.float32)
MARS_BOUNDS_HIGH = np.asarray([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1.0], np.float32)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    blocks = []
    while size:
        block = conn.recv(size)
        if not block:
            raise EOFError("policy worker disconnected")
        blocks.append(block)
        size -= len(block)
    return b"".join(blocks)


def rpc(socket_path: str, request: dict) -> dict:
    payload = pickle.dumps(request, protocol=5)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(900)
        conn.connect(socket_path)
        conn.sendall(struct.pack("!Q", len(payload)) + payload)
        size = struct.unpack("!Q", _recv_exact(conn, 8))[0]
        response = pickle.loads(_recv_exact(conn, size))
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "policy RPC failed"))
    return response


def _success(info) -> bool:
    value = info.get("success", False) if isinstance(info, dict) else False
    if torch.is_tensor(value):
        return bool(value.detach().cpu().reshape(-1)[0].item())
    return bool(np.asarray(value).reshape(-1)[0])


def _terminal(value) -> bool:
    if torch.is_tensor(value):
        return bool(value.detach().cpu().reshape(-1)[0].item())
    return bool(np.asarray(value).reshape(-1)[0])


def _action_codec(dataset_root: Path, task: str, agent: int) -> tuple[np.ndarray, np.ndarray]:
    if os.environ.get("BWA_MARS_CONTROL") == "1":
        return MARS_BOUNDS_LOW.copy(), MARS_BOUNDS_HIGH.copy()
    manifest = json.loads((dataset_root / task / "training_manifest.json").read_text())
    codec = manifest["action"]["codec"]["config"]
    start = agent * 8
    return (
        np.asarray(codec["low"][start : start + 8], np.float32),
        np.asarray(codec["high"][start : start + 8], np.float32),
    )


class TemporalChunkEnsembler:
    def __init__(self, agent_count: int, decay: float):
        self.decay = float(decay)
        self.history = [[] for _ in range(agent_count)]

    def update(self, step: int, chunks: list[np.ndarray]) -> dict[str, np.ndarray]:
        result = {}
        for agent, chunk in enumerate(chunks):
            chunk = np.asarray(chunk, np.float32)
            if chunk.ndim != 2 or chunk.shape[1] != 8 or not np.isfinite(chunk).all():
                raise ValueError(f"invalid policy chunk for agent {agent}: {chunk.shape}")
            history = self.history[agent]
            history.append((step, chunk))
            history[:] = [(start, value) for start, value in history if step - start < len(value)]
            candidates = np.asarray([value[step - start] for start, value in history], np.float32)
            weights = np.exp(-self.decay * np.arange(len(candidates) - 1, -1, -1, dtype=np.float32))
            weights /= weights.sum()
            result[f"panda-{agent}"] = np.sum(candidates * weights[:, None], axis=0).astype(np.float32)
        return result


def evaluate(args) -> dict:
    if args.formal and (args.episodes != 20 or args.seed != 20260820 or args.sim_backend != "cpu"):
        raise ValueError("formal Validation20 requires episodes=20, seed=20260820 and sim_backend=cpu")
    if args.formal and abs(args.temporal_ensemble_decay - 0.01) > 1e-12:
        raise ValueError("formal Validation20 requires temporal ensemble decay 0.01")

    torch.set_num_threads(args.cpu_threads)
    dataset_root = Path(args.dataset_root)
    config_path = Path(args.config_root) / f"{args.task}.yaml"
    config = yaml.safe_load(config_path.read_text())
    agent_count = len(config["agents"])
    max_steps = args.max_steps_override or MAX_STEPS[args.task]
    output_path = Path(args.output)
    previous = {}
    if output_path.is_file():
        try:
            previous = json.loads(output_path.read_text())
        except json.JSONDecodeError:
            pass
    episode_map = {int(row["episode"]): row for row in previous.get("episodes_detail", []) if not row.get("error")}

    for episode in range(args.episodes):
        if episode in episode_map:
            continue
        env = None
        row = {"episode": episode, "seed": args.seed + episode, "success": False, "steps": 0}
        try:
            env = gym.make(
                config["task_name"] + "-rf",
                config=str(config_path),
                obs_mode="rgb",
                control_mode="pd_joint_pos",
                # Match the native training capture path: sensor observations
                # rendered with the default shader pack at 320x240.
                render_mode="sensors",
                reward_mode="dense",
                num_envs=1,
                sim_backend=args.sim_backend,
                render_backend="cpu",
                sensor_configs={"shader_pack": "default", "width": 320, "height": 240},
                human_render_camera_configs={"shader_pack": "default"},
                viewer_camera_configs={"shader_pack": "default"},
            )
            obs, _ = env.reset(seed=args.seed + episode)
            rpc(args.socket, {"op": "reset"})
            ensembler = TemporalChunkEnsembler(agent_count, args.temporal_ensemble_decay)
            bounds = [_action_codec(dataset_root, args.task, agent) for agent in range(agent_count)]
            for step in range(max_steps):
                local = []
                for agent in range(agent_count):
                    image = obs["sensor_data"][f"head_camera_agent{agent}"]["rgb"]
                    qpos = obs["agent"][f"panda-{agent}"]["qpos"]
                    local.append(
                        {
                            "agent": agent,
                            "task": args.task,
                            "prompt": TASK_PROMPTS[args.task],
                            "image": image.detach().cpu().numpy()[0],
                            "state": qpos.detach().cpu().numpy()[0, :9],
                        }
                    )
                # A decentralized deployment is stronger than merely avoiding
                # cross-attention: each inference request contains exactly one
                # arm's RGB and qpos.  The worker shares weights, never inputs.
                chunks = [
                    rpc(args.socket, {"op": "infer", "observations": [observation]})["chunks"][0]
                    for observation in local
                ]
                action = ensembler.update(step, chunks)
                for agent in range(agent_count):
                    low, high = bounds[agent]
                    action[f"panda-{agent}"] = np.clip(action[f"panda-{agent}"], low, high)
                obs, _, terminated, truncated, info = env.step(action)
                row["success"] = _success(info)
                row["steps"] = step + 1
                if row["success"] or _terminal(terminated) or _terminal(truncated):
                    break
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if env is not None:
                env.close()
        episode_map[episode] = row
        episodes = [episode_map[key] for key in sorted(episode_map)]
        successes = sum(bool(item["success"]) for item in episodes)
        report = {
            "schema": "bwa.vla.validation20.task.v1",
            "baseline": args.policy,
            "task": args.task,
            "status": "failed" if any(item.get("error") for item in episodes) else ("complete" if len(episodes) == args.episodes else "running"),
            "successes": successes,
            "episodes": len(episodes),
            "target_episodes": args.episodes,
            "success_rate": successes / len(episodes),
            "max_steps": max_steps,
            "seed_base": args.seed,
            "sim_backend": args.sim_backend,
            "temporal_ensemble_decay": args.temporal_ensemble_decay,
            "camera_size": [320, 240],
            "render_mode": "sensors",
            "shader_pack": "default",
            "policy_contract": "shared_weights_decentralized_local_rgb_qpos_to_local_action8",
            "episodes_detail": episodes,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(output_path, report)
    result = json.loads(output_path.read_text())
    if result["status"] != "complete":
        raise RuntimeError(f"{args.task} validation did not complete cleanly")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("rdt", "openvla", "pi05", "gaudp"), required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--task", choices=tuple(TASK_PROMPTS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-root", default="/workspace/datasets/robofactory_multitask")
    parser.add_argument("--config-root", default="/workspace/repos/RoboFactory/robofactory/configs/table")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--sim-backend", choices=("cpu", "gpu", "auto"), default="cpu")
    parser.add_argument("--temporal-ensemble-decay", type=float, default=0.01)
    parser.add_argument("--cpu-threads", type=int, default=10)
    parser.add_argument("--max-steps-override", type=int)
    parser.add_argument("--formal", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
