from envs.mujoco_carry_env import MinimalTwoRobotCarryEnv


def test_env_step():
    env = MinimalTwoRobotCarryEnv()
    obs = env.reset()
    action = env.scripted_action()
    obs, reward, done, info = env.step(action)
    assert "qpos" in obs
    assert "qvel" in obs
    assert "ncon" in obs
