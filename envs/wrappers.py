from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.two_robot_carry_env import CarryEnvConfig, TwoRobotCarryNarrowPassageEnv


class TwoRobotCarryGymWrapper(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        cfg: CarryEnvConfig | None = None,
        *,
        render_mode: str | None = None,
        camera: str = "fixed",
        render_width: int = 640,
        render_height: int = 360,
    ):
        super().__init__()
        if render_mode not in {None, "rgb_array"}:
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.env = TwoRobotCarryNarrowPassageEnv(cfg)
        self.render_mode = render_mode
        self.camera = camera
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.metadata = dict(self.metadata, render_fps=round(1.0 / self.env.control_dt))

        self.observation_space = spaces.Dict(
            {
                "robot_0": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
                ),
                "robot_1": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
                ),
                "object": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "global_state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
                ),
            }
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs = self.env.reset(seed=seed)
        return self._filter_obs(obs), obs["metrics"]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        truncated = bool(done and info.get("failure_reason") == "timeout")
        terminated = bool(done and not truncated)
        return self._filter_obs(obs), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        return self.env.render(
            camera=self.camera,
            width=self.render_width,
            height=self.render_height,
        )

    def close(self):
        self.env.close()

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
