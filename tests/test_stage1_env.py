import numpy as np

from envs.two_robot_carry_env import TwoRobotCarryNarrowPassageEnv
from envs.wrappers import TwoRobotCarryGymWrapper


def test_stage1_env_reset_step():
    env = TwoRobotCarryNarrowPassageEnv()
    obs = env.reset(seed=0, randomize=False)
    assert "robot_0" in obs
    assert "robot_1" in obs
    assert "object" in obs
    assert obs["robot_0"].shape == (11,)
    assert obs["robot_1"].shape == (11,)
    assert env.action_dim == 8

    action = env.scripted_action()
    obs, reward, done, info = env.step(action)
    assert np.isfinite(reward)
    assert "success" in info
    assert "force_proxy" in info


def test_stage1_scripted_rollout_finishes():
    env = TwoRobotCarryNarrowPassageEnv()
    obs = env.reset(seed=0, randomize=False)

    done = False
    info = {}
    for _ in range(env.cfg.episode_len):
        obs, reward, done, info = env.step(env.scripted_action())
        if done:
            break

    assert done
    assert "failure_reason" in info


def test_stage1_gym_wrapper():
    env = TwoRobotCarryGymWrapper()
    obs, info = env.reset(seed=0)
    assert env.action_space.shape == (8,)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
