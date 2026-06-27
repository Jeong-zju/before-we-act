from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from envs.two_robot_carry_env import TwoRobotCarryNarrowPassageEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="outputs/stage1/stage1_rollout.mp4")
    parser.add_argument("--camera", type=str, default="fixed")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    env = TwoRobotCarryNarrowPassageEnv()
    env.reset(seed=0, randomize=False)

    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    frames = []

    done = False
    while not done:
        action = env.scripted_action()
        obs, reward, done, info = env.step(action)

        renderer.update_scene(env.data, camera=args.camera)
        frame = renderer.render()
        frames.append(frame)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=20)
    print("saved:", out)
    print("frames:", len(frames))
    print("final info:", info)


if __name__ == "__main__":
    main()
