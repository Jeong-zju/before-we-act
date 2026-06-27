from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import trange

from envs.two_robot_carry_env import TwoRobotCarryNarrowPassageEnv


def save_episode(path: Path, episode: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in episode.items():
            if isinstance(v, dict):
                grp = f.create_group(k)
                for kk, vv in v.items():
                    grp.create_dataset(kk, data=np.asarray(vv))
            else:
                f.create_dataset(k, data=np.asarray(v))


def collect_one(env: TwoRobotCarryNarrowPassageEnv, seed: int, noise_std: float = 0.0):
    obs = env.reset(seed=seed, randomize=True)

    robot_0_obs = []
    robot_1_obs = []
    object_obs = []
    global_states = []
    actions = []
    rewards = []
    force_proxy = []
    contacts = []
    success_flags = []
    failure_flags = []

    done = False
    info = {}

    while not done:
        action = env.scripted_action()
        if noise_std > 0:
            action[:3] += env.rng.normal(0.0, noise_std, size=3)
            action[4:7] += env.rng.normal(0.0, noise_std, size=3)
            action = np.clip(action, -1.0, 1.0)

        robot_0_obs.append(obs["robot_0"])
        robot_1_obs.append(obs["robot_1"])
        object_obs.append(obs["object"])
        global_states.append(obs["global_state"])
        actions.append(action)

        obs, reward, done, info = env.step(action)

        rewards.append(reward)
        force_proxy.append(info["force_proxy"])
        contacts.append(info["ncon"])
        success_flags.append(float(info["success"]))
        failure_flags.append(float(info["failure"]))

    episode = {
        "obs": {
            "robot_0": np.asarray(robot_0_obs, dtype=np.float32),
            "robot_1": np.asarray(robot_1_obs, dtype=np.float32),
            "object": np.asarray(object_obs, dtype=np.float32),
        },
        "global_state": np.asarray(global_states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "force_proxy": np.asarray(force_proxy, dtype=np.float32),
        "contacts": np.asarray(contacts, dtype=np.int32),
        "success": np.asarray(success_flags, dtype=np.float32),
        "failure": np.asarray(failure_flags, dtype=np.float32),
        "final_success": np.asarray([float(info.get("success", False))], dtype=np.float32),
        "failure_reason_code": np.asarray([hash(info.get("failure_reason", "none")) % 100000], dtype=np.int32),
    }
    return episode, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default="examples/stage1/scripted")
    parser.add_argument("--noise_std", type=float, default=0.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    env = TwoRobotCarryNarrowPassageEnv()

    successes = 0
    failure_reasons = {}

    for ep in trange(args.num_episodes):
        episode, info = collect_one(env, seed=ep, noise_std=args.noise_std)
        successes += int(info.get("success", False))
        reason = info.get("failure_reason", "none")
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        save_episode(out_dir / f"episode_{ep:06d}.hdf5", episode)

    print("saved dir:", out_dir)
    print("num_episodes:", args.num_episodes)
    print("successes:", successes)
    print("success_rate:", successes / args.num_episodes)
    print("failure_reasons:", failure_reasons)


if __name__ == "__main__":
    main()
