from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import mujoco
import numpy as np


@dataclass
class CarryEnvConfig:
    xml_path: str | None = None
    scenario: str = "nominal"
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
    max_robot_distance: float = 1.25
    max_y_desync: float = 0.55
    max_object_yaw_abs: float = 0.75
    max_object_midpoint_error: float = 0.35
    occlusion_prob: float = 0.0
    teammate_delay_steps: int = 0
    asymmetric_obstacle: bool = False
    narrow_width_scale: float = 1.0
    blocked_passage_prob: float = 0.0
    false_belief_prob: float = 0.0
    force_limit: float = 1.0
    min_robot_distance_limit: float = 0.25
    local_force_scale_newtons: float = 1000.0
    object_z: float = 0.061
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
        self._apply_scenario_preset()
        root = Path(__file__).resolve().parent
        xml_path = self.cfg.xml_path or str(root / "assets" / "two_robot_carry.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self._align_robot_joint_frames()
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
        self.robot_pose_history: list[np.ndarray] = []
        self.blocked_passage_active = False
        self.blocked_passage_side = 1
        self.false_belief_active = False
        self.occlusion_active = False
        self.teammate_visible = {0: True, 1: True}
        self.last_visible_teammate_pose: dict[int, np.ndarray | None] = {0: None, 1: None}
        self.min_robot_distance = float("inf")
        self.max_contact_force = 0.0

    def _apply_scenario_preset(self):
        scenario = str(getattr(self.cfg, "scenario", "nominal") or "nominal").lower()
        self.cfg.scenario = scenario

        if scenario == "nominal":
            return
        if scenario == "narrow":
            self.cfg.narrow_width_scale = min(self.cfg.narrow_width_scale, 0.76)
            self.cfg.force_limit = min(self.cfg.force_limit, 0.80)
            return
        if scenario == "occlusion":
            self.cfg.occlusion_prob = max(self.cfg.occlusion_prob, 0.60)
            return
        if scenario == "asymmetric_obstacle":
            self.cfg.asymmetric_obstacle = True
            return
        if scenario == "delayed_teammate":
            self.cfg.teammate_delay_steps = max(int(self.cfg.teammate_delay_steps), 4)
            return
        if scenario == "blocked_passage":
            self.cfg.blocked_passage_prob = max(self.cfg.blocked_passage_prob, 0.45)
            return
        if scenario == "false_belief":
            self.cfg.false_belief_prob = max(self.cfg.false_belief_prob, 1.0)
            return
        if scenario == "hard_comm":
            self.cfg.narrow_width_scale = min(self.cfg.narrow_width_scale, 0.76)
            self.cfg.occlusion_prob = max(self.cfg.occlusion_prob, 0.65)
            self.cfg.asymmetric_obstacle = True
            self.cfg.false_belief_prob = max(self.cfg.false_belief_prob, 1.0)
            self.cfg.force_limit = min(self.cfg.force_limit, 0.80)
            self.cfg.min_robot_distance_limit = max(self.cfg.min_robot_distance_limit, 0.35)
            return
        raise ValueError(f"Unknown carry scenario: {scenario}")

    @property
    def action_dim(self) -> int:
        return 8

    def _align_robot_joint_frames(self):
        for name in ("robot_a", "robot_b"):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                self.model.body_pos[body_id, 0:2] = 0.0

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
        self._set_object_pose(obj_x, obj_y, self.cfg.object_z, 0.0)

        mujoco.mj_forward(self.model, self.data)
        self._reset_scenario_state()
        self._record_pose_history()
        self._update_scenario_state()
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
        self._set_object_pose(mid[0], mid[1], self.cfg.object_z, yaw)

    def _effective_passage_half_width(self) -> float:
        return float(self.cfg.passage_half_width * self.cfg.narrow_width_scale)

    def _force_limit(self) -> float:
        return float(min(self.cfg.force_limit, self.cfg.max_force_proxy))

    def _object_lateral_extent(self, yaw: float) -> float:
        return float(
            self.cfg.object_half_length * abs(np.cos(yaw))
            + self.cfg.object_half_width * abs(np.sin(yaw))
        )

    def _reset_scenario_state(self):
        self.robot_pose_history = []
        self.blocked_passage_active = bool(self.rng.random() < self.cfg.blocked_passage_prob)
        self.blocked_passage_side = -1 if self.rng.random() < 0.5 else 1
        self.false_belief_active = bool(self.rng.random() < self.cfg.false_belief_prob)
        self.occlusion_active = False
        self.teammate_visible = {0: True, 1: True}
        self.last_visible_teammate_pose = {0: self._robot_pose(1), 1: self._robot_pose(0)}
        self.min_robot_distance = float("inf")
        self.max_contact_force = 0.0

    def _record_pose_history(self):
        poses = np.stack([self._robot_pose(0), self._robot_pose(1)], axis=0)
        self.robot_pose_history.append(poses)
        keep = max(2, int(self.cfg.teammate_delay_steps) + 2)
        if len(self.robot_pose_history) > keep:
            self.robot_pose_history = self.robot_pose_history[-keep:]

    def _delayed_robot_pose(self, robot: int) -> np.ndarray:
        delay = max(0, int(self.cfg.teammate_delay_steps))
        if delay <= 0 or not self.robot_pose_history:
            return self._robot_pose(robot)
        idx = max(0, len(self.robot_pose_history) - 1 - delay)
        return self.robot_pose_history[idx][robot].copy()

    def _update_scenario_state(self):
        a = self._robot_pose(0)
        b = self._robot_pose(1)
        obj = self._object_pose_xy_yaw()
        robot_distance = float(np.linalg.norm(a[:2] - b[:2]))
        self.min_robot_distance = min(self.min_robot_distance, robot_distance)

        in_shared_space = -0.25 < obj[1] < 2.80 or -0.25 < a[1] < 2.80 or -0.25 < b[1] < 2.80
        # Consume a fixed common-random-number tape every step.  Paired policy
        # modes may enter the shared space at different times; conditional RNG
        # draws would otherwise desynchronize all later exogenous occlusions.
        occlusion_draw, hide_robot_0_draw, hide_robot_1_draw = self.rng.random(3)
        occlusion_sample = bool(occlusion_draw < self.cfg.occlusion_prob)
        self.occlusion_active = bool(occlusion_sample and in_shared_space)

        if self.occlusion_active:
            # Keep at least one viewpoint degraded so the local observations disagree.
            hide_robot_0 = bool(hide_robot_0_draw < 0.55)
            hide_robot_1 = bool(hide_robot_1_draw < 0.55)
            if not hide_robot_0 and not hide_robot_1:
                hide_robot_0 = True
            self.teammate_visible = {0: not hide_robot_0, 1: not hide_robot_1}
        else:
            self.teammate_visible = {0: True, 1: True}

        if self.teammate_visible[0]:
            self.last_visible_teammate_pose[0] = b.copy()
        if self.teammate_visible[1]:
            self.last_visible_teammate_pose[1] = a.copy()

        self.max_contact_force = max(self.max_contact_force, self._compute_force_proxy())

    def _compute_force_components(self) -> Dict[str, float]:
        obj = self._object_pose_xy_yaw()
        x, y, yaw = obj[0], obj[1], obj[2]

        wall_violation = 0.0
        if 0.15 < y < 2.65:
            clearance = self._effective_passage_half_width() - abs(x) - self._object_lateral_extent(yaw)
            wall_violation = max(0.0, -clearance)

        blocked_violation = 0.0
        if self.blocked_passage_active and 0.85 < y < 1.90:
            open_lane_center = -0.28 * float(self.blocked_passage_side)
            usable_half_width = 0.62 * self._effective_passage_half_width()
            clearance = usable_half_width - abs(x - open_lane_center) - self._object_lateral_extent(yaw)
            blocked_violation = max(0.0, -clearance)

        a = self._robot_pose(0)
        b = self._robot_pose(1)
        robot_dist = np.linalg.norm(a[:2] - b[:2])
        robot_clearance = max(2 * self.cfg.robot_radius, self.cfg.min_robot_distance_limit)
        robot_violation = max(0.0, robot_clearance - robot_dist)

        force_proxy = 10.0 * wall_violation + 8.0 * blocked_violation + 5.0 * robot_violation
        return {
            "force_proxy": float(force_proxy),
            "wall_violation": float(wall_violation),
            "blocked_violation": float(blocked_violation),
            "robot_violation": float(robot_violation),
        }

    def _compute_force_proxy(self) -> float:
        return self._compute_force_components()["force_proxy"]

    def _local_contact_agents(self) -> np.ndarray:
        """Return per-robot tactile contact flags from MuJoCo contact pairs."""

        robot_geom_ids = (
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "robot_a_base"
            ),
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "robot_b_base"
            ),
        )
        flags = np.zeros(2, dtype=np.float32)
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            for agent_id, geom_id in enumerate(robot_geom_ids):
                if geom_id >= 0 and geom_id in pair:
                    flags[agent_id] = 1.0
        return flags

    def _local_force_agents(self) -> np.ndarray:
        """Return per-robot contact-force magnitudes from local geom contacts."""

        robot_geom_ids = (
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "robot_a_base"
            ),
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "robot_b_base"
            ),
        )
        forces = np.zeros(2, dtype=np.float32)
        contact_force = np.zeros(6, dtype=np.float64)
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            contact_force.fill(0.0)
            mujoco.mj_contactForce(
                self.model, self.data, contact_index, contact_force
            )
            magnitude = float(np.linalg.norm(contact_force[:3]))
            for agent_id, geom_id in enumerate(robot_geom_ids):
                if geom_id >= 0 and geom_id in pair:
                    forces[agent_id] += magnitude
        scale = float(self.cfg.local_force_scale_newtons)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("local_force_scale_newtons must be finite and positive")
        return np.clip(forces / scale, 0.0, 1.0).astype(np.float32)

    def _success(self) -> bool:
        obj = self._object_pose_xy_yaw()
        return bool(abs(obj[0]) < self.cfg.goal_tol_xy and obj[1] > self.cfg.goal_y - self.cfg.goal_tol_xy)

    def _failure(self) -> bool:
        obj = self._object_pose_xy_yaw()
        force_proxy = self._compute_force_proxy()
        a = self._robot_pose(0)
        b = self._robot_pose(1)

        robot_dist = float(np.linalg.norm(a[:2] - b[:2]))
        y_desync = float(abs(a[1] - b[1]))
        obj_yaw_abs = float(abs(((obj[2] + np.pi) % (2 * np.pi)) - np.pi))

        grip_a, grip_b = self._grip_points()
        midpoint = 0.5 * (grip_a + grip_b)
        midpoint_error = float(np.linalg.norm(obj[:2] - midpoint))

        if force_proxy > self._force_limit():
            self.failure_reason = "force_violation"
            return True

        if robot_dist > self.cfg.max_robot_distance:
            self.failure_reason = "robot_too_far"
            return True

        if 0.0 < obj[1] < 2.65 and y_desync > self.cfg.max_y_desync:
            self.failure_reason = "desync_in_passage"
            return True

        if 0.0 < obj[1] < 2.65 and obj_yaw_abs > self.cfg.max_object_yaw_abs:
            self.failure_reason = "object_yaw_too_large"
            return True

        if self.grasped and midpoint_error > self.cfg.max_object_midpoint_error:
            self.failure_reason = "object_dropped"
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

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    def _relative_pose(self, target_pose: np.ndarray, reference_pose: np.ndarray) -> np.ndarray:
        return np.array([
            target_pose[0] - reference_pose[0],
            target_pose[1] - reference_pose[1],
            self._wrap_angle(target_pose[2] - reference_pose[2]),
        ], dtype=np.float32)

    def _observed_teammate_pose(self, agent_id: int) -> np.ndarray:
        teammate_id = 1 - agent_id
        if self.teammate_visible.get(agent_id, True):
            pose = self._delayed_robot_pose(teammate_id)
        else:
            stale_pose = self.last_visible_teammate_pose.get(agent_id)
            pose = stale_pose.copy() if stale_pose is not None else self._delayed_robot_pose(teammate_id)

        if self.false_belief_active:
            sign = -1.0 if agent_id == 0 else 1.0
            pose = pose + np.array([-0.12 * sign, 0.16, -0.08 * sign], dtype=np.float64)
            pose[2] = self._wrap_angle(pose[2])
        return pose

    def _observed_object_pose(self, agent_id: int, obj: np.ndarray) -> np.ndarray:
        pose = obj.copy()
        sign = -1.0 if agent_id == 0 else 1.0
        if self.cfg.asymmetric_obstacle and -0.25 < obj[1] < 2.80:
            pose[0] += 0.10 * sign
        if self.false_belief_active:
            pose += np.array([0.16 * sign, -0.10, 0.10 * sign], dtype=np.float64)
        pose[2] = self._wrap_angle(pose[2])
        return pose

    def _observed_teammate_rel_pose(self, agent_id: int, self_pose: np.ndarray) -> np.ndarray:
        if not self.teammate_visible.get(agent_id, True):
            return np.zeros(3, dtype=np.float32)
        return self._relative_pose(self._observed_teammate_pose(agent_id), self_pose)

    def _observed_object_rel_pose(self, agent_id: int, self_pose: np.ndarray, obj: np.ndarray) -> np.ndarray:
        return self._relative_pose(self._observed_object_pose(agent_id, obj), self_pose)

    def _perceived_force_proxy(self, agent_id: int, force_proxy: float, obj: np.ndarray) -> float:
        perceived = float(force_proxy)
        if self.cfg.asymmetric_obstacle:
            perceived += 0.08 if agent_id == 0 else 0.14
            if -0.25 < obj[1] < 2.80:
                perceived += 0.10 if agent_id == 0 else 0.18
                if self.blocked_passage_active:
                    perceived += 0.20 if self.blocked_passage_side == (-1 if agent_id == 0 else 1) else 0.08
        if self.false_belief_active:
            perceived += 0.15
        if not self.teammate_visible.get(agent_id, True):
            perceived += 0.05
        return float(max(0.0, perceived))

    def _local_obs(self, agent_id: int, self_pose: np.ndarray, obj: np.ndarray, force_proxy: float) -> np.ndarray:
        object_rel = self._observed_object_rel_pose(agent_id, self_pose, obj)
        teammate_rel = self._observed_teammate_rel_pose(agent_id, self_pose)
        perceived_force = self._perceived_force_proxy(agent_id, force_proxy, obj)

        return np.array([
            self_pose[0], self_pose[1], self_pose[2],
            object_rel[0], object_rel[1], object_rel[2],
            teammate_rel[0], teammate_rel[1], teammate_rel[2],
            float(self.grasped), perceived_force,
        ], dtype=np.float32)

    def _local_pose_context_agents(self, obj: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rel_target = np.stack([
            self._observed_teammate_rel_pose(0, a),
            self._observed_teammate_rel_pose(1, b),
        ], axis=0).astype(np.float32)
        object_rel = np.stack([
            self._observed_object_rel_pose(0, a, obj),
            self._observed_object_rel_pose(1, b, obj),
        ], axis=0).astype(np.float32)
        return rel_target, object_rel

    def _model_local_obs_agents(
        self,
        obj: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        force_proxy: float,
        contacts: int,
    ) -> np.ndarray:
        proprio = np.stack([
            self._local_obs(0, a, obj, force_proxy),
            self._local_obs(1, b, obj, force_proxy),
        ], axis=0).astype(np.float32)
        actions = self.last_action.astype(np.float32).reshape(2, 4)
        local_force = proprio[:, 10:11]
        contact = np.full((2, 1), float(contacts > 0), dtype=np.float32)
        return np.concatenate([proprio, actions, local_force, contact], axis=-1).astype(np.float32)

    def _object_goal_distance(self, obj: np.ndarray | None = None) -> float:
        obj = self._object_pose_xy_yaw() if obj is None else obj
        return float(np.linalg.norm(np.array([obj[0], obj[1] - self.cfg.goal_y], dtype=np.float64)))

    def _progress(self, obj: np.ndarray | None = None) -> float:
        obj = self._object_pose_xy_yaw() if obj is None else obj
        start_y = -1.20
        denom = max(1e-6, self.cfg.goal_y - start_y)
        return float(np.clip((obj[1] - start_y) / denom, 0.0, 1.0))

    def _build_info(self, success: bool | None = None, failure: bool | None = None) -> Dict:
        obj = self._object_pose_xy_yaw()
        a = self._robot_pose(0)
        b = self._robot_pose(1)
        force_components = self._compute_force_components()
        force_proxy = force_components["force_proxy"]
        robot_distance = float(np.linalg.norm(a[:2] - b[:2]))
        min_robot_distance = float(min(self.min_robot_distance, robot_distance))
        contacts = int(self.data.ncon)
        local_contact_agents = self._local_contact_agents()
        local_force_agents = self._local_force_agents()
        virtual_collision = bool(
            force_components["wall_violation"] > 0.0
            or force_components["blocked_violation"] > 0.0
            or force_components["robot_violation"] > 0.0
        )
        collision_count = contacts + int(virtual_collision)
        force_violation = bool(force_proxy > self._force_limit())
        success = self._success() if success is None else bool(success)
        failure = bool(self.failure_reason != "none") if failure is None else bool(failure)
        communication_required = bool(
            self.occlusion_active
            or (not self.teammate_visible.get(0, True))
            or (not self.teammate_visible.get(1, True))
            or self.blocked_passage_active
            or self.false_belief_active
            or min_robot_distance < self.cfg.min_robot_distance_limit
            or force_violation
        )
        local_obs_agents = self._model_local_obs_agents(obj, a, b, force_proxy, contacts)
        rel_target_pose_agents, object_rel_pose_agents = self._local_pose_context_agents(obj, a, b)

        return {
            "scenario": self.cfg.scenario,
            "success": success,
            "failure": failure and not success,
            "failure_reason": self.failure_reason,
            "collision": bool(collision_count > 0),
            "collision_count": int(collision_count),
            "contact_force": float(force_proxy),
            "max_contact_force": float(max(self.max_contact_force, force_proxy)),
            "force_proxy": float(force_proxy),
            "force_violation": force_violation,
            "force_limit": self._force_limit(),
            "robot_distance": robot_distance,
            "inter_robot_distance": robot_distance,
            "min_robot_distance": min_robot_distance,
            "object_goal_distance": self._object_goal_distance(obj),
            "progress": self._progress(obj),
            "occlusion_active": bool(self.occlusion_active),
            "teammate_visible_robot_0": bool(self.teammate_visible.get(0, True)),
            "teammate_visible_robot_1": bool(self.teammate_visible.get(1, True)),
            "communication_required": communication_required,
            "blocked_passage_active": bool(self.blocked_passage_active),
            "blocked_passage_current": bool(self.blocked_passage_active and 0.85 < obj[1] < 1.90),
            "false_belief_active": bool(self.false_belief_active),
            "asymmetric_obstacle": bool(self.cfg.asymmetric_obstacle),
            "effective_passage_half_width": self._effective_passage_half_width(),
            "narrow_width_scale": float(self.cfg.narrow_width_scale),
            "wall_violation": force_components["wall_violation"],
            "blocked_violation": force_components["blocked_violation"],
            "robot_violation": force_components["robot_violation"],
            "ncon": contacts,
            "contacts": contacts,
            "local_contact_agents": local_contact_agents,
            "local_force_agents": local_force_agents,
            "local_force_units": "normalized_0_1",
            "local_force_scale_newtons": float(
                self.cfg.local_force_scale_newtons
            ),
            "object_xy_yaw": obj.copy(),
            "grasped": bool(self.grasped),
            "step_count": int(self.step_count),
            "local_obs_agents": local_obs_agents,
            "rel_target_pose_agents": rel_target_pose_agents,
            "object_rel_pose_agents": object_rel_pose_agents,
        }

    def get_obs(self) -> Dict:
        a = self._robot_pose(0)
        b = self._robot_pose(1)
        obj = self._object_pose_xy_yaw()
        force_proxy = self._compute_force_proxy()
        success = self._success()

        robot_0 = self._local_obs(0, a, obj, force_proxy)
        robot_1 = self._local_obs(1, b, obj, force_proxy)

        global_state = np.concatenate([
            a, b, obj,
            np.array([force_proxy, float(success), float(self.step_count)], dtype=np.float64),
        ]).astype(np.float32)
        metrics = self._build_info(success=success, failure=False)

        return {
            "robot_0": robot_0,
            "robot_1": robot_1,
            "object": obj.astype(np.float32),
            "global_state": global_state,
            "metrics": metrics,
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
        self._record_pose_history()
        self._update_scenario_state()

        success = self._success()
        failure = self._failure()
        done = bool(success or failure)
        obs = self.get_obs()
        info = self._build_info(success=success, failure=failure)
        obs["metrics"] = info

        obj = self._object_pose_xy_yaw()
        force_proxy = info["force_proxy"]

        reward = 0.0
        reward += 0.5 * (obj[1] + 1.2)
        reward -= 2.0 * force_proxy
        reward -= 0.01 * float(np.sum(action[:3] ** 2) + np.sum(action[4:7] ** 2))
        if success:
            reward += 100.0
        if failure and not success:
            reward -= 20.0

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
