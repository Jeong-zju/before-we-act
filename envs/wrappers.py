from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.two_robot_carry_env import CarryEnvConfig, TwoRobotCarryNarrowPassageEnv


class TwoRobotCarryGymWrapper(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: CarryEnvConfig | None = None):
        super().__init__()
        self.env = TwoRobotCarryNarrowPassageEnv(cfg)

        self.observation_space = spaces.Dict({
            "robot_0": spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32),
            "robot_1": spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32),
            "object": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "global_state": spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
        })

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs = self.env.reset(seed=seed)
        return self._filter_obs(obs), obs["metrics"]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = bool(info["success"] or info["failure"])
        truncated = bool(obs["metrics"]["step_count"] >= self.env.cfg.episode_len)
        return self._filter_obs(obs), reward, terminated, truncated, info

    @staticmethod
    def _filter_obs(obs):
        return {
            "robot_0": obs["robot_0"],
            "robot_1": obs["robot_1"],
            "object": obs["object"],
            "global_state": obs["global_state"],
        }


def main():
    env = TwoRobotCarryGymWrapper()
    obs, info = env.reset(seed=0)
    print("obs keys:", obs.keys())
    print("action space:", env.action_space)
    for _ in range(5):
        action = env.env.scripted_action().astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
    print("terminated:", terminated, "truncated:", truncated, "info:", info)


if __name__ == "__main__":
    main()
