from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.two_robot_carry_env import (
    CooperativeStopEnvConfig,
    TwoRobotCooperativeStopEnv,
)


class TwoRobotCooperativeStopGymWrapper(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        cfg: CooperativeStopEnvConfig | None = None,
        *,
        render_mode: str | None = None,
        camera: str = "fixed",
        render_width: int = 640,
        render_height: int = 360,
    ):
        super().__init__()
        if render_mode not in {None, "rgb_array"}:
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.env = TwoRobotCooperativeStopEnv(cfg)
        self.render_mode = render_mode
        self.camera = camera
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.metadata = dict(self.metadata, render_fps=round(1.0 / self.env.control_dt))

        agent_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.env.robot_state_dim,),
                    dtype=np.float32,
                ),
                "base_pose": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "base_velocity": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "gripper": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                "base_effort": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self.env.cfg.agent_camera_height,
                        self.env.cfg.agent_camera_width,
                        3,
                    ),
                    dtype=np.uint8,
                ),
            }
        )
        privileged_space = spaces.Dict(
            {
                "state": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(self.env.privileged_state_dim,),
                    dtype=np.float32,
                ),
                "object_pose": spaces.Box(-np.inf, np.inf, (3,), np.float32),
                "object_velocity": spaces.Box(-np.inf, np.inf, (6,), np.float32),
                "task_bounds": spaces.Box(-np.inf, np.inf, (4,), np.float32),
                "object_half_size": spaces.Box(-np.inf, np.inf, (3,), np.float32),
                "task": spaces.Box(-np.inf, np.inf, (10,), np.float32),
                "braking_event": spaces.Box(-np.inf, np.inf, (10,), np.float32),
                "contact": spaces.Box(-np.inf, np.inf, (5,), np.float32),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "robot_0": agent_space,
                "robot_1": agent_space,
                "proprioception": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.env.proprioception_dim,),
                    dtype=np.float32,
                ),
                "privileged_state": privileged_space,
            }
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        randomize = True if options is None else bool(options.get("randomize", True))
        obs, info = self.env.reset(seed=seed, randomize=randomize)
        return obs, info

    def step(self, action):
        return self.env.step(action)

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


def main():
    env = TwoRobotCooperativeStopGymWrapper()
    obs, info = env.reset(seed=0)
    print("obs keys:", obs.keys())
    print("action space:", env.action_space)
    for _ in range(5):
        action = env.env.scripted_action().astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
    print("terminated:", terminated, "truncated:", truncated, "info:", info)


if __name__ == "__main__":
    main()


# Backward-compatible wrapper name for downstream code.
TwoRobotCarryGymWrapper = TwoRobotCooperativeStopGymWrapper
