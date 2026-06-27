from pathlib import Path
import numpy as np
import mujoco


class MinimalTwoRobotCarryEnv:
    def __init__(self, xml_path: str | None = None, seed: int = 0):
        root = Path(__file__).resolve().parent
        self.xml_path = xml_path or str(root / "assets" / "two_robot_carry_minimal.xml")
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.max_steps = 400

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

        # qpos: robot_a 3 + robot_b 3 + object freejoint 7
        self.data.qpos[:] = 0.0
        self.data.qpos[0:3] = [-0.6, 0.0, 0.0]
        self.data.qpos[3:6] = [0.6, 0.0, 0.0]

        # object freejoint: x y z qw qx qy qz
        obj_start = 6
        self.data.qpos[obj_start:obj_start + 7] = [0.0, 0.35, 0.15, 1.0, 0.0, 0.0, 0.0]

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.get_obs()

    def get_obs(self):
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "ncon": int(self.data.ncon),
            "time": float(self.data.time),
        }

    def get_state_vector(self):
        return np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            self.data.ctrl.copy(),
            np.array([self.data.time, self.data.ncon], dtype=np.float64),
        ])

    def scripted_action(self):
        # 让两个机器人简单向 +y 方向移动，测试 step 和 logging
        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        ctrl[1] = 0.4
        ctrl[4] = 0.4
        return ctrl

    def step(self, action):
        self.data.ctrl[:] = np.asarray(action, dtype=np.float64)
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        obs = self.get_obs()
        reward = 0.0
        done = self.step_count >= self.max_steps
        info = {"ncon": int(self.data.ncon)}
        return obs, reward, done, info


def main():
    env = MinimalTwoRobotCarryEnv()
    obs = env.reset()
    print("reset obs keys:", obs.keys())
    print("nu:", env.model.nu, "nq:", env.model.nq, "nv:", env.model.nv)

    for _ in range(100):
        action = env.scripted_action()
        obs, reward, done, info = env.step(action)

    print("final time:", obs["time"])
    print("contacts:", obs["ncon"])
    print("state dim:", env.get_state_vector().shape[0])


if __name__ == "__main__":
    main()
