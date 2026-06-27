from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import mujoco
import numpy as np


@dataclass
class CarryEnvConfig:
    xml_path: str | None = None
    sim_dt: float = 0.002
    control_dt: float = 0.05
    episode_len: int = 500
    goal_y: float = 3.05
    goal_tol_xy: float = 0.28
    passage_half_width: float = 0.90
    robot_radius: float = 0.22
    object_half_length: float = 0.65
    object_half_width: float = 0.07
    max_force_proxy: float = 1.0
    max_action_v: float = 0.7
    max_action_w: float = 1.2
    grip_distance: float = 0.42
    seed: int = 0


class TwoRobotCarryNarrowPassageEnv:
    """
    Stage-1 MVP environment.

    Action:
        shape = (8,)
        [a_vx, a_vy, a_wz, a_grip, b_vx, b_vy, b_wz, b_grip]

    Observation:
        Dict with robot_0, robot_1, object, global_state, metrics.

    Notes:
        This MVP uses a geometric carrying approximation:
        if both robots grip, the object pose is updated from the midpoint and heading
        of two robot gripper sites. This keeps Stage 1 stable and makes scripted data
        collection reliable. Later stages can replace this with real grasp/contact dynamics.
    """

    def __init__(self, cfg: CarryEnvConfig | None = None):
        self.cfg = cfg or CarryEnvConfig()
        root = Path(__file__).resolve().parent
        xml_path = self.cfg.xml_path or str(root / "assets" / "two_robot_carry_stage1.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.rng = np.random.default_rng(self.cfg.seed)
        self.frame_skip = max(1, int(round(self.cfg.control_dt / self.cfg.sim_dt)))
        self.step_count = 0

        self.robot_a_qpos_addr = 0
        self.robot_b_qpos_addr = 3
        self.object_qpos_addr = 6

        self.last_action = np.zeros(8, dtype=np.float64)
        self.grasped = False
        self.failure_reason = "none"

    @property
    def action_dim(self) -> int:
        return 8

    def reset(self, seed: int | None = None, randomize: bool = True) -> Dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.last_action[:] = 0.0
        self.grasped = False
        self.failure_reason = "none"

        if randomize:
            ax = -0.45 + self.rng.uniform(-0.08, 0.08)
            bx = 0.45 + self.rng.uniform(-0.08, 0.08)
            y0 = -1.20 + self.rng.uniform(-0.05, 0.05)
            yaw_a = self.rng.uniform(-0.05, 0.05)
            yaw_b = self.rng.uniform(-0.05, 0.05)
        else:
            ax, bx, y0, yaw_a, yaw_b = -0.45, 0.45, -1.20, 0.0, 0.0

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        self.data.qpos[self.robot_a_qpos_addr:self.robot_a_qpos_addr + 3] = [ax, y0, yaw_a]
        self.data.qpos[self.robot_b_qpos_addr:self.robot_b_qpos_addr + 3] = [bx, y0, yaw_b]

        obj_x = 0.0 + self.rng.uniform(-0.04, 0.04) if randomize else 0.0
        obj_y = -0.95 + self.rng.uniform(-0.04, 0.04) if randomize else -0.95
        self._set_object_pose(obj_x, obj_y, 0.18, 0.0)

        mujoco.mj_forward(self.model, self.data)
        return self.get_obs()

    def _set_object_pose(self, x: float, y: float, z: float, yaw: float):
        qw = np.cos(yaw / 2.0)
        qz = np.sin(yaw / 2.0)
        self.data.qpos[self.object_qpos_addr:self.object_qpos_addr + 7] = [x, y, z, qw, 0.0, 0.0, qz]
        self.data.qvel[6:12] = 0.0

    def _robot_pose(self, robot: int) -> np.ndarray:
        addr = self.robot_a_qpos_addr if robot == 0 else self.robot_b_qpos_addr
        return self.data.qpos[addr:addr + 3].copy()

    def _object_pose_xy_yaw(self) -> np.ndarray:
        q = self.data.qpos[self.object_qpos_addr:self.object_qpos_addr + 7]
        x, y = q[0], q[1]
        qw, qz = q[3], q[6]
        yaw = 2.0 * np.arctan2(qz, qw)
        return np.array([x, y, yaw], dtype=np.float64)

    def _grip_points(self) -> Tuple[np.ndarray, np.ndarray]:
        a = self._robot_pose(0)
        b = self._robot_pose(1)

        def front_point(p):
            x, y, yaw = p
            return np.array([x + 0.23 * np.sin(yaw), y + 0.23 * np.cos(yaw)], dtype=np.float64)

        return front_point(a), front_point(b)

    def _apply_geometric_carry(self):
        a_grip, b_grip = self._grip_points()
        mid = 0.5 * (a_grip + b_grip)
        diff = b_grip - a_grip
        yaw = np.arctan2(diff[1], diff[0])
        self._set_object_pose(mid[0], mid[1], 0.18, yaw)

    def _compute_force_proxy(self) -> float:
        obj = self._object_pose_xy_yaw()
        x, y = obj[0], obj[1]

        wall_violation = 0.0
        if 0.15 < y < 2.65:
            clearance = self.cfg.passage_half_width - abs(x) - self.cfg.object_half_width
            wall_violation = max(0.0, -clearance)

        a = self._robot_pose(0)
        b = self._robot_pose(1)
        robot_dist = np.linalg.norm(a[:2] - b[:2])
        robot_violation = max(0.0, 2 * self.cfg.robot_radius - robot_dist)

        return float(10.0 * wall_violation + 5.0 * robot_violation)

    def _success(self) -> bool:
        obj = self._object_pose_xy_yaw()
        return bool(abs(obj[0]) < self.cfg.goal_tol_xy and obj[1] > self.cfg.goal_y - self.cfg.goal_tol_xy)

    def _failure(self) -> bool:
        obj = self._object_pose_xy_yaw()
        force_proxy = self._compute_force_proxy()
        a = self._robot_pose(0)
        b = self._robot_pose(1)

        if force_proxy > self.cfg.max_force_proxy:
            self.failure_reason = "force_violation"
            return True
        if obj[1] < -2.0 or obj[1] > 3.7 or abs(obj[0]) > 2.2:
            self.failure_reason = "object_out_of_bounds"
            return True
        if np.any(np.abs(a[:2]) > np.array([2.25, 3.7])) or np.any(np.abs(b[:2]) > np.array([2.25, 3.7])):
            self.failure_reason = "robot_out_of_bounds"
            return True
        if self.step_count >= self.cfg.episode_len:
            self.failure_reason = "timeout"
            return True

        self.failure_reason = "none"
        return False

    def get_obs(self) -> Dict:
        a = self._robot_pose(0)
        b = self._robot_pose(1)
        obj = self._object_pose_xy_yaw()
        force_proxy = self._compute_force_proxy()
        success = self._success()

        robot_0 = np.array([
            a[0], a[1], a[2],
            obj[0] - a[0], obj[1] - a[1], obj[2] - a[2],
            b[0] - a[0], b[1] - a[1], b[2] - a[2],
            float(self.grasped), force_proxy,
        ], dtype=np.float32)

        robot_1 = np.array([
            b[0], b[1], b[2],
            obj[0] - b[0], obj[1] - b[1], obj[2] - b[2],
            a[0] - b[0], a[1] - b[1], a[2] - b[2],
            float(self.grasped), force_proxy,
        ], dtype=np.float32)

        global_state = np.concatenate([
            a, b, obj,
            np.array([force_proxy, float(success), float(self.step_count)], dtype=np.float64),
        ]).astype(np.float32)

        return {
            "robot_0": robot_0,
            "robot_1": robot_1,
            "object": obj.astype(np.float32),
            "global_state": global_state,
            "metrics": {
                "success": bool(success),
                "force_proxy": float(force_proxy),
                "grasped": bool(self.grasped),
                "ncon": int(self.data.ncon),
                "failure_reason": self.failure_reason,
                "step_count": int(self.step_count),
            },
        }

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got {action.shape[0]}")

        action = np.clip(action, -1.0, 1.0)
        self.last_action = action.copy()

        a_vx, a_vy, a_wz, a_grip = action[:4]
        b_vx, b_vy, b_wz, b_grip = action[4:]

        # Stage-1 MVP uses kinematic base control.
        # The action represents normalized velocity commands rather than motor force.
        dt = self.cfg.control_dt

        a = self._robot_pose(0)
        b = self._robot_pose(1)

        a[0] += float(a_vx) * self.cfg.max_action_v * dt
        a[1] += float(a_vy) * self.cfg.max_action_v * dt
        a[2] += float(a_wz) * self.cfg.max_action_w * dt

        b[0] += float(b_vx) * self.cfg.max_action_v * dt
        b[1] += float(b_vy) * self.cfg.max_action_v * dt
        b[2] += float(b_wz) * self.cfg.max_action_w * dt

        # Keep yaw in [-pi, pi].
        a[2] = (a[2] + np.pi) % (2 * np.pi) - np.pi
        b[2] = (b[2] + np.pi) % (2 * np.pi) - np.pi

        self.data.qpos[self.robot_a_qpos_addr:self.robot_a_qpos_addr + 3] = a
        self.data.qpos[self.robot_b_qpos_addr:self.robot_b_qpos_addr + 3] = b

        # Store normalized control for logging compatibility.
        self.data.ctrl[:] = np.array([a_vx, a_vy, a_wz, b_vx, b_vy, b_wz], dtype=np.float64)

        self.grasped = bool(a_grip > 0.5 and b_grip > 0.5)

        if self.grasped:
            self._apply_geometric_carry()

        mujoco.mj_forward(self.model, self.data)

        self.step_count += 1

        obs = self.get_obs()
        success = self._success()
        failure = self._failure()
        done = bool(success or failure)

        obj = self._object_pose_xy_yaw()
        force_proxy = obs["metrics"]["force_proxy"]

        reward = 0.0
        reward += 0.5 * (obj[1] + 1.2)
        reward -= 2.0 * force_proxy
        reward -= 0.01 * float(np.sum(action[:3] ** 2) + np.sum(action[4:7] ** 2))
        if success:
            reward += 100.0
        if failure and not success:
            reward -= 20.0

        info = {
            "success": success,
            "failure": failure and not success,
            "failure_reason": self.failure_reason,
            "force_proxy": force_proxy,
            "ncon": int(self.data.ncon),
            "object_xy_yaw": obj.copy(),
        }
        return obs, float(reward), done, info

    def scripted_action(self) -> np.ndarray:
        """
        A simple oracle-like controller:
        - close both grippers;
        - move both robots forward through the passage;
        - keep x separated and roughly centered.
        """
        a = self._robot_pose(0)
        b = self._robot_pose(1)
        obj = self._object_pose_xy_yaw()

        target_a_x = -0.42
        target_b_x = 0.42
        target_y = 3.15

        def ctrl_for_robot(p, target_x):
            x, y, yaw = p
            vx = np.clip(1.5 * (target_x - x), -0.55, 0.55)
            vy = np.clip(0.9 * (target_y - y), -0.70, 0.70)
            wz = np.clip(-1.2 * yaw, -0.8, 0.8)
            return vx, vy, wz

        a_vx, a_vy, a_wz = ctrl_for_robot(a, target_a_x)
        b_vx, b_vy, b_wz = ctrl_for_robot(b, target_b_x)

        # Slow down near walls in passage.
        if 0.2 < obj[1] < 2.4:
            a_vy *= 0.75
            b_vy *= 0.75

        return np.array([a_vx, a_vy, a_wz, 1.0, b_vx, b_vy, b_wz, 1.0], dtype=np.float64)

    def get_state_vector(self) -> np.ndarray:
        obs = self.get_obs()
        return obs["global_state"].astype(np.float64)


def main():
    env = TwoRobotCarryNarrowPassageEnv()
    obs = env.reset(seed=0, randomize=False)
    print("action_dim:", env.action_dim)
    print("obs robot_0:", obs["robot_0"].shape)
    print("global_state:", obs["global_state"].shape)

    total_reward = 0.0
    done = False
    info = {}
    while not done:
        action = env.scripted_action()
        obs, reward, done, info = env.step(action)
        total_reward += reward

    print("done:", done)
    print("info:", info)
    print("total_reward:", round(total_reward, 3))
    print("final object:", obs["object"])


if __name__ == "__main__":
    main()
