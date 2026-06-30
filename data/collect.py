from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from tqdm import trange

from envs.two_robot_carry_env import TwoRobotCarryNarrowPassageEnv
from data.policies import Stage2ScriptedPolicy, robot_distance_from_obs
from data.schema import failure_to_id, phase_to_id, save_stage2_episode


def empty_episode_buffers() -> Dict[str, list]:
    return {
        "obs_robot_0": [],
        "obs_robot_1": [],
        "global_state": [],
        "object_pose": [],
        "actions": [],
        "rewards": [],
        "force_proxy": [],
        "contacts": [],
        "robot_distance": [],
        "phase": [],
        "success": [],
        "failure": [],
        "failure_reason": [],
        "communication_dummy": [],
        "time": [],
    }


def append_step(buffers: Dict[str, list], obs: dict, action: np.ndarray, reward: float, info: dict, phase: str, t: int):
    buffers["obs_robot_0"].append(obs["robot_0"])
    buffers["obs_robot_1"].append(obs["robot_1"])
    buffers["global_state"].append(obs["global_state"])
    buffers["object_pose"].append(obs["object"])
    buffers["actions"].append(action.astype(np.float32))
    buffers["rewards"].append(float(reward))
    buffers["force_proxy"].append(float(info.get("force_proxy", obs["metrics"]["force_proxy"])))
    buffers["contacts"].append(int(info.get("ncon", obs["metrics"]["ncon"])))
    buffers["robot_distance"].append(robot_distance_from_obs(obs))
    buffers["phase"].append(phase_to_id(phase))
    buffers["success"].append(float(info.get("success", False)))
    buffers["failure"].append(float(info.get("failure", False)))
    buffers["failure_reason"].append(failure_to_id(info.get("failure_reason", "none")))
    buffers["communication_dummy"].append(np.zeros(8, dtype=np.float32))
    buffers["time"].append(float(t))


def finalize_episode(buffers: Dict[str, list]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in buffers.items()}


def collect_one_episode(
    env: TwoRobotCarryNarrowPassageEnv,
    policy: Stage2ScriptedPolicy,
    seed: int,
    randomize: bool = True,
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    obs = env.reset(seed=seed, randomize=randomize)
    buffers = empty_episode_buffers()

    done = False
    info = {
        "success": False,
        "failure": False,
        "failure_reason": "none",
        "force_proxy": obs["metrics"]["force_proxy"],
        "ncon": obs["metrics"]["ncon"],
    }

    while not done:
        policy_out = policy(env)
        action = policy_out.action
        phase = policy_out.phase

        next_obs, reward, done, info = env.step(action)
        append_step(buffers, obs, action, reward, info, phase, env.step_count)
        obs = next_obs

    episode = finalize_episode(buffers)
    meta = {
        "seed": int(seed),
        "success": bool(info.get("success", False)),
        "failure": bool(info.get("failure", False)),
        "failure_reason": str(info.get("failure_reason", "none")),
        "num_steps": int(len(episode["actions"])),
    }
    return episode, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["scripted", "noisy", "recovery"], required=True)
    parser.add_argument("--num_episodes", type=int, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument("--randomize", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = TwoRobotCarryNarrowPassageEnv()
    successes = 0
    failures = 0
    failure_reasons: dict[str, int] = {}
    lengths = []

    for ep_idx in trange(args.num_episodes):
        seed = args.seed_start + ep_idx
        policy = Stage2ScriptedPolicy(noise_std=args.noise_std, seed=seed, mode=args.mode)

        episode, meta = collect_one_episode(env=env, policy=policy, seed=seed, randomize=bool(args.randomize))

        meta["episode_index"] = ep_idx
        meta["mode"] = args.mode
        meta["noise_std"] = float(args.noise_std)

        successes += int(meta["success"])
        failures += int(meta["failure"])
        lengths.append(meta["num_steps"])
        reason = meta["failure_reason"]
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

        save_stage2_episode(out_dir / f"episode_{ep_idx:06d}.hdf5", episode, meta)

    summary = {
        "mode": args.mode,
        "num_episodes": args.num_episodes,
        "successes": successes,
        "failures": failures,
        "success_rate": successes / max(1, args.num_episodes),
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "failure_reasons": failure_reasons,
        "out_dir": str(out_dir),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
