from __future__ import annotations

import inspect
import numpy as np

from envs.two_robot_carry_env import TwoRobotCarryNarrowPassageEnv
from data.stage2_policies import Stage2ScriptedPolicy


def run_noise(noise: float, n: int = 30):
    env = TwoRobotCarryNarrowPassageEnv()
    success = 0
    reasons = {}
    mean_action_abs = []

    for ep in range(n):
        obs = env.reset(seed=1000 + ep, randomize=True)
        policy = Stage2ScriptedPolicy(noise_std=noise, seed=2000 + ep, mode="noisy")

        done = False
        info = {}
        acts = []

        while not done:
            out = policy(env)
            acts.append(out.action.copy())
            obs, reward, done, info = env.step(out.action)

        success += int(info.get("success", False))
        reason = info.get("failure_reason", "none")
        reasons[reason] = reasons.get(reason, 0) + 1
        mean_action_abs.append(np.abs(np.asarray(acts)).mean())

    print("noise_std:", noise)
    print("  success_rate:", success / n)
    print("  reasons:", reasons)
    print("  mean_abs_action:", float(np.mean(mean_action_abs)))


def main():
    import data.stage2_policies as p
    import envs.two_robot_carry_env as e

    print("policy file:", p.__file__)
    print("env file:", e.__file__)
    print("has robot_too_far in _failure:", "robot_too_far" in inspect.getsource(e.TwoRobotCarryNarrowPassageEnv._failure))
    print("has gripper failure in noisy policy:", "Gripper failure" in inspect.getsource(p.Stage2ScriptedPolicy.apply_noisy_disturbance))

    for noise in [0.0, 0.2, 0.6, 1.0, 10.0]:
        run_noise(noise, n=30)


if __name__ == "__main__":
    main()
