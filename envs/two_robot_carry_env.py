from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import mujoco
import numpy as np


DEFAULT_TASK_INSTRUCTION = (
    "carry the object together; when one robot slows to a stop, "
    "the other robot should gradually slow and stop"
)


@dataclass
class CooperativeStopEnvConfig:
    """Configuration for the standard cooperative stopping task."""

    xml_path: str | None = None
    scenario: str = "standard"
    control_dt: float = 0.05
    episode_len: int = 300
    max_action_v: float = 0.7
    max_action_w: float = 1.2
    max_linear_acceleration: float = 0.8
    max_angular_acceleration: float = 2.0
    brake_start_time_min: float = 2.0
    brake_start_time_max: float = 5.0
    max_response_time: float = 5.0
    min_cruise_forward_speed: float = 0.20
    max_pre_brake_speed_error: float = 0.12
    response_speed_delta: float = 0.05
    stop_linear_speed: float = 0.04
    stop_angular_speed: float = 0.08
    min_gradual_brake_steps: int = 5
    stop_hold_steps: int = 8
    max_robot_distance: float = 1.25
    agent_camera_width: int = 128
    agent_camera_height: int = 128
    include_camera_images: bool = True
    seed: int = 0
    reward_cruise_scale: float = 0.5
    reward_speed_match_scale: float = 1.0
    reward_response_progress_scale: float = 5.0
    reward_speed_tracking_scale: float = 2.0
    reward_acceleration_tracking_scale: float = 0.05
    reward_stop_hold: float = 0.25
    reward_time_cost: float = 0.01
    reward_energy_scale: float = 0.005
    reward_ungrasped_cost: float = 0.05
    reward_success: float = 50.0
    reward_failure: float = 20.0


@dataclass(frozen=True)
class CooperativeStopGeometry:
    """Task geometry resolved exclusively from named MuJoCo XML elements."""

    home_qpos: np.ndarray
    floor_half_size: np.ndarray
    task_center: np.ndarray
    task_half_size: np.ndarray
    robot_half_sizes: np.ndarray
    object_half_size: np.ndarray
    object_height: float
    grip_offsets: np.ndarray


class TwoRobotCooperativeStopEnv:
    """Geometric two-robot cooperative stopping environment.

    Both planar robots geometrically carry a horizontal bar while moving in
    the world-frame ``+Y`` direction.  At a seeded random time, one seeded
    random robot becomes the braking robot.  Its base target velocity is
    overridden to zero and its executed velocity is reduced by the same
    acceleration-limited kinematic controller used by both robots.  The task
    succeeds when the responding robot also decelerates and both robots hold a
    stable stop.

    Action (8):
        [r0_vx, r0_vy, r0_wz, r0_grip, r1_vx, r1_vy, r1_wz, r1_grip]

    The task remains geometric: poses are updated directly and MuJoCo is used
    for named geometry, contact queries, generalized actuator effort, and RGB
    rendering.  No ``mj_step`` motor-dynamics integration is performed.
    """

    def __init__(self, cfg: CooperativeStopEnvConfig | None = None):
        self.cfg = cfg or CooperativeStopEnvConfig()
        self._validate_config()

        root = Path(__file__).resolve().parent
        xml_path = self.cfg.xml_path or str(root / "assets" / "two_robot_carry.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.geometry = self._read_geometry_from_model()
        self.data = mujoco.MjData(self.model)

        self.robot_qpos_addrs = (
            self._joint_qpos_addr("robot_a_x"),
            self._joint_qpos_addr("robot_b_x"),
        )
        self.robot_a_qpos_addr, self.robot_b_qpos_addr = self.robot_qpos_addrs
        self.robot_dof_addrs = (
            self._joint_dof_addr("robot_a_x"),
            self._joint_dof_addr("robot_b_x"),
        )
        self.object_qpos_addr = self._joint_qpos_addr("object_free")
        self.object_dof_addr = self._joint_dof_addr("object_free")
        self.robot_camera_names = ("robot_0_camera", "robot_1_camera")
        for camera_name in self.robot_camera_names:
            self._named_id(mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

        self.rng = np.random.default_rng(self.cfg.seed)
        self.step_count = 0
        self.last_action = np.zeros(8, dtype=np.float64)
        self.executed_action = np.zeros(8, dtype=np.float64)
        self.base_velocities = np.zeros((2, 3), dtype=np.float64)
        self.base_accelerations = np.zeros((2, 3), dtype=np.float64)
        self.linear_decelerations = np.zeros(2, dtype=np.float64)
        self.grasped = False
        self.has_grasped = False
        self.failure_reason = "none"
        self.min_robot_distance = float("inf")

        self.braking_agent = 0
        self.responding_agent = 1
        self.brake_start_step = 0
        self.brake_event_active = False
        self.brake_event_step = -1
        self.pre_brake_motion_valid = False
        self.follower_speed_at_brake = 0.0
        self.response_started = False
        self.response_start_step = -1
        self.follower_brake_steps = 0
        self.stop_hold_count = 0
        self._best_response_progress = 0.0
        self._previous_speeds = np.zeros(2, dtype=np.float64)
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}

    def _validate_config(self) -> None:
        if self.cfg.scenario != "standard":
            raise ValueError(
                f"Unknown cooperative-stop scenario {self.cfg.scenario!r}; "
                "only 'standard' is supported"
            )
        positive_values = {
            "control_dt": self.cfg.control_dt,
            "episode_len": self.cfg.episode_len,
            "max_action_v": self.cfg.max_action_v,
            "max_action_w": self.cfg.max_action_w,
            "max_linear_acceleration": self.cfg.max_linear_acceleration,
            "max_angular_acceleration": self.cfg.max_angular_acceleration,
            "max_response_time": self.cfg.max_response_time,
            "stop_hold_steps": self.cfg.stop_hold_steps,
            "min_gradual_brake_steps": self.cfg.min_gradual_brake_steps,
            "agent_camera_width": self.cfg.agent_camera_width,
            "agent_camera_height": self.cfg.agent_camera_height,
        }
        for name, value in positive_values.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.cfg.brake_start_time_min < 0.0:
            raise ValueError("brake_start_time_min cannot be negative")
        if self.cfg.brake_start_time_max < self.cfg.brake_start_time_min:
            raise ValueError(
                "brake_start_time_max must be at least brake_start_time_min"
            )
        min_step = int(np.ceil(self.cfg.brake_start_time_min / self.cfg.control_dt))
        if min_step >= self.cfg.episode_len:
            raise ValueError("braking event must start before the episode timeout")

    @property
    def control_dt(self) -> float:
        return float(self.cfg.control_dt)

    @property
    def action_dim(self) -> int:
        return 8

    @property
    def robot_state_dim(self) -> int:
        return 11

    @property
    def proprioception_dim(self) -> int:
        return 2 * self.robot_state_dim

    @property
    def privileged_state_dim(self) -> int:
        return 34

    def _named_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(
                f"MuJoCo XML is missing required {object_type.name} {name!r}"
            )
        return int(object_id)

    def _joint_qpos_addr(self, name: str) -> int:
        joint_id = self._named_id(mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.model.jnt_qposadr[joint_id])

    def _joint_dof_addr(self, name: str) -> int:
        joint_id = self._named_id(mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.model.jnt_dofadr[joint_id])

    def _read_geometry_from_model(self) -> CooperativeStopGeometry:
        home_id = self._named_id(mujoco.mjtObj.mjOBJ_KEY, "home")
        floor_id = self._named_id(mujoco.mjtObj.mjOBJ_GEOM, "floor")
        bounds_id = self._named_id(mujoco.mjtObj.mjOBJ_GEOM, "task_bounds")
        object_id = self._named_id(mujoco.mjtObj.mjOBJ_GEOM, "carry_object_geom")
        robot_ids = (
            self._named_id(mujoco.mjtObj.mjOBJ_GEOM, "robot_a_base"),
            self._named_id(mujoco.mjtObj.mjOBJ_GEOM, "robot_b_base"),
        )
        grip_ids = (
            self._named_id(mujoco.mjtObj.mjOBJ_SITE, "robot_a_grip_site"),
            self._named_id(mujoco.mjtObj.mjOBJ_SITE, "robot_b_grip_site"),
        )
        object_qpos_addr = self._joint_qpos_addr("object_free")
        home_qpos = self.model.key_qpos[home_id].copy()
        return CooperativeStopGeometry(
            home_qpos=home_qpos,
            floor_half_size=self.model.geom_size[floor_id, :2].copy(),
            task_center=self.model.geom_pos[bounds_id, :2].copy(),
            task_half_size=self.model.geom_size[bounds_id, :2].copy(),
            robot_half_sizes=np.stack(
                [self.model.geom_size[geom_id].copy() for geom_id in robot_ids]
            ),
            object_half_size=self.model.geom_size[object_id].copy(),
            object_height=float(home_qpos[object_qpos_addr + 2]),
            grip_offsets=np.stack(
                [self.model.site_pos[site_id].copy() for site_id in grip_ids]
            ),
        )

    def _sample_brake_start_step(self) -> int:
        min_step = max(
            1, int(np.ceil(self.cfg.brake_start_time_min / self.cfg.control_dt))
        )
        max_step = int(np.floor(self.cfg.brake_start_time_max / self.cfg.control_dt))
        max_step = min(max_step, self.cfg.episode_len - 1)
        if max_step < min_step:
            max_step = min_step
        return int(self.rng.integers(min_step, max_step + 1))

    def reset(
        self, seed: int | None = None, randomize: bool = True
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        home_id = self._named_id(mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, home_id)

        self.step_count = 0
        self.last_action[:] = 0.0
        self.executed_action[:] = 0.0
        self.base_velocities[:] = 0.0
        self.base_accelerations[:] = 0.0
        self.linear_decelerations[:] = 0.0
        self.grasped = False
        self.has_grasped = False
        self.failure_reason = "none"
        self.min_robot_distance = float("inf")

        # Event randomness is sampled before pose jitter so a seed identifies
        # the same braking agent and time in randomized and deterministic reset.
        self.braking_agent = int(self.rng.integers(0, 2))
        self.responding_agent = 1 - self.braking_agent
        self.brake_start_step = self._sample_brake_start_step()
        self.brake_event_active = False
        self.brake_event_step = -1
        self.pre_brake_motion_valid = False
        self.follower_speed_at_brake = 0.0
        self.response_started = False
        self.response_start_step = -1
        self.follower_brake_steps = 0
        self.stop_hold_count = 0
        self._best_response_progress = 0.0
        self._previous_speeds[:] = 0.0

        home_a = self.geometry.home_qpos[
            self.robot_a_qpos_addr : self.robot_a_qpos_addr + 3
        ]
        home_b = self.geometry.home_qpos[
            self.robot_b_qpos_addr : self.robot_b_qpos_addr + 3
        ]
        if randomize:
            ax = float(home_a[0] + self.rng.uniform(-0.08, 0.08))
            bx = float(home_b[0] + self.rng.uniform(-0.08, 0.08))
            y0 = float(0.5 * (home_a[1] + home_b[1]) + self.rng.uniform(-0.05, 0.05))
            yaw_a = float(self.rng.uniform(-0.05, 0.05))
            yaw_b = float(self.rng.uniform(-0.05, 0.05))
        else:
            ax, bx = float(home_a[0]), float(home_b[0])
            y0 = float(0.5 * (home_a[1] + home_b[1]))
            yaw_a, yaw_b = float(home_a[2]), float(home_b[2])

        self.data.qpos[:] = self.geometry.home_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qpos[self.robot_a_qpos_addr : self.robot_a_qpos_addr + 3] = [
            ax,
            y0,
            yaw_a,
        ]
        self.data.qpos[self.robot_b_qpos_addr : self.robot_b_qpos_addr + 3] = [
            bx,
            y0,
            yaw_b,
        ]

        home_object = self.geometry.home_qpos[
            self.object_qpos_addr : self.object_qpos_addr + 7
        ]
        obj_x = (
            float(home_object[0] + self.rng.uniform(-0.04, 0.04))
            if randomize
            else float(home_object[0])
        )
        obj_y = (
            float(home_object[1] + self.rng.uniform(-0.04, 0.04))
            if randomize
            else float(home_object[1])
        )
        self._set_object_pose(obj_x, obj_y, self.geometry.object_height, 0.0)

        mujoco.mj_forward(self.model, self.data)
        self._update_distance_statistics()
        observation = self.get_obs()
        info = self._build_info(success=False, failure=False)
        return observation, info

    def _set_object_pose(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        velocity: np.ndarray | None = None,
    ) -> None:
        qw = np.cos(yaw / 2.0)
        qz = np.sin(yaw / 2.0)
        self.data.qpos[self.object_qpos_addr : self.object_qpos_addr + 7] = [
            x,
            y,
            z,
            qw,
            0.0,
            0.0,
            qz,
        ]
        object_velocity = np.zeros(6, dtype=np.float64)
        if velocity is not None:
            object_velocity[:] = np.asarray(velocity, dtype=np.float64)
        self.data.qvel[self.object_dof_addr : self.object_dof_addr + 6] = (
            object_velocity
        )

    def _robot_pose(self, robot: int) -> np.ndarray:
        address = self.robot_qpos_addrs[robot]
        return self.data.qpos[address : address + 3].copy()

    def _object_pose_xy_yaw(self) -> np.ndarray:
        qpos = self.data.qpos[self.object_qpos_addr : self.object_qpos_addr + 7]
        yaw = 2.0 * np.arctan2(qpos[6], qpos[3])
        return np.asarray([qpos[0], qpos[1], yaw], dtype=np.float64)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _grip_points(self) -> Tuple[np.ndarray, np.ndarray]:
        def site_point(pose: np.ndarray, offset: np.ndarray) -> np.ndarray:
            x, y, yaw = pose
            cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
            return np.asarray(
                [
                    x + cos_yaw * offset[0] - sin_yaw * offset[1],
                    y + sin_yaw * offset[0] + cos_yaw * offset[1],
                ],
                dtype=np.float64,
            )

        return (
            site_point(self._robot_pose(0), self.geometry.grip_offsets[0]),
            site_point(self._robot_pose(1), self.geometry.grip_offsets[1]),
        )

    def _apply_geometric_carry(self, previous_pose: np.ndarray) -> None:
        a_grip, b_grip = self._grip_points()
        midpoint = 0.5 * (a_grip + b_grip)
        difference = b_grip - a_grip
        yaw = float(np.arctan2(difference[1], difference[0]))
        velocity = np.zeros(6, dtype=np.float64)
        velocity[:2] = (midpoint - previous_pose[:2]) / self.cfg.control_dt
        velocity[5] = (
            self._wrap_angle(yaw - float(previous_pose[2])) / self.cfg.control_dt
        )
        self._set_object_pose(
            midpoint[0],
            midpoint[1],
            self.geometry.object_height,
            yaw,
            velocity,
        )

    def _activate_brake_event(self) -> None:
        speeds = self._linear_speeds()
        self.brake_event_active = True
        self.brake_event_step = self.step_count
        self.follower_speed_at_brake = float(speeds[self.responding_agent])
        forward_speeds = self.base_velocities[:, 1]
        self.pre_brake_motion_valid = bool(
            self.grasped
            and self.has_grasped
            and np.all(forward_speeds >= self.cfg.min_cruise_forward_speed)
            and abs(float(speeds[0] - speeds[1])) <= self.cfg.max_pre_brake_speed_error
        )

    def _limit_velocity_change(
        self, current: np.ndarray, desired: np.ndarray
    ) -> np.ndarray:
        result = current.copy()
        linear_delta = desired[:2] - current[:2]
        max_linear_delta = self.cfg.max_linear_acceleration * self.cfg.control_dt
        delta_norm = float(np.linalg.norm(linear_delta))
        if delta_norm > max_linear_delta:
            linear_delta *= max_linear_delta / delta_norm
        result[:2] += linear_delta
        max_angular_delta = self.cfg.max_angular_acceleration * self.cfg.control_dt
        result[2] += float(
            np.clip(desired[2] - current[2], -max_angular_delta, max_angular_delta)
        )
        return result

    def _integrate_bases(self, action: np.ndarray) -> None:
        previous_velocities = self.base_velocities.copy()
        desired = np.asarray(
            [
                [
                    action[0] * self.cfg.max_action_v,
                    action[1] * self.cfg.max_action_v,
                    action[2] * self.cfg.max_action_w,
                ],
                [
                    action[4] * self.cfg.max_action_v,
                    action[5] * self.cfg.max_action_v,
                    action[6] * self.cfg.max_action_w,
                ],
            ],
            dtype=np.float64,
        )
        if self.brake_event_active:
            desired[self.braking_agent] = 0.0

        for agent_id in range(2):
            self.base_velocities[agent_id] = self._limit_velocity_change(
                previous_velocities[agent_id], desired[agent_id]
            )
            pose = self._robot_pose(agent_id)
            pose += self.base_velocities[agent_id] * self.cfg.control_dt
            pose[2] = self._wrap_angle(float(pose[2]))
            qpos_address = self.robot_qpos_addrs[agent_id]
            dof_address = self.robot_dof_addrs[agent_id]
            self.data.qpos[qpos_address : qpos_address + 3] = pose
            self.data.qvel[dof_address : dof_address + 3] = self.base_velocities[
                agent_id
            ]

        self.base_accelerations = (
            self.base_velocities - previous_velocities
        ) / self.cfg.control_dt
        self.executed_action[:] = action
        for agent_id, offset in ((0, 0), (1, 4)):
            self.executed_action[offset] = (
                self.base_velocities[agent_id, 0] / self.cfg.max_action_v
            )
            self.executed_action[offset + 1] = (
                self.base_velocities[agent_id, 1] / self.cfg.max_action_v
            )
            self.executed_action[offset + 2] = (
                self.base_velocities[agent_id, 2] / self.cfg.max_action_w
            )
        self.data.ctrl[:] = np.asarray(
            [
                self.executed_action[0],
                self.executed_action[1],
                self.executed_action[2],
                self.executed_action[4],
                self.executed_action[5],
                self.executed_action[6],
            ],
            dtype=np.float64,
        )

    def _linear_speeds(self) -> np.ndarray:
        return np.linalg.norm(self.base_velocities[:, :2], axis=1)

    def _agent_stopped(self, agent_id: int) -> bool:
        return bool(
            np.linalg.norm(self.base_velocities[agent_id, :2])
            <= self.cfg.stop_linear_speed
            and abs(float(self.base_velocities[agent_id, 2]))
            <= self.cfg.stop_angular_speed
        )

    def _both_stopped(self) -> bool:
        return self._agent_stopped(0) and self._agent_stopped(1)

    def _update_response_state(self) -> None:
        speeds = self._linear_speeds()
        self.linear_decelerations = np.maximum(
            0.0, (self._previous_speeds - speeds) / self.cfg.control_dt
        )
        if not self.brake_event_active:
            self._previous_speeds = speeds
            return

        follower_speed = float(speeds[self.responding_agent])
        if (
            not self.response_started
            and self.follower_speed_at_brake - follower_speed
            >= self.cfg.response_speed_delta
        ):
            self.response_started = True
            self.response_start_step = self.step_count

        if self.response_started and not self._agent_stopped(self.responding_agent):
            self.follower_brake_steps += 1

        valid_stop = bool(
            self.pre_brake_motion_valid
            and self.response_started
            and self.follower_brake_steps >= self.cfg.min_gradual_brake_steps
            and self._both_stopped()
            and self.grasped
        )
        if valid_stop:
            self.stop_hold_count += 1
        else:
            self.stop_hold_count = 0
        self._previous_speeds = speeds

    def _response_progress(self) -> float:
        if not self.brake_event_active or self.follower_speed_at_brake <= 1e-6:
            return 0.0
        follower_speed = float(self._linear_speeds()[self.responding_agent])
        return float(
            np.clip(
                (self.follower_speed_at_brake - follower_speed)
                / self.follower_speed_at_brake,
                0.0,
                1.0,
            )
        )

    def _success(self) -> bool:
        return bool(
            self.brake_event_active
            and self.pre_brake_motion_valid
            and self.response_started
            and self.follower_brake_steps >= self.cfg.min_gradual_brake_steps
            and self.stop_hold_count >= self.cfg.stop_hold_steps
            and self._both_stopped()
            and self.grasped
        )

    def _failure(self) -> bool:
        if self.has_grasped and not self.grasped:
            self.failure_reason = "grasp_lost"
            return True

        if self._robot_distance() > self.cfg.max_robot_distance:
            self.failure_reason = "robot_too_far"
            return True

        object_pose = self._object_pose_xy_yaw()
        if np.any(
            np.abs(object_pose[:2] - self.geometry.task_center)
            > self.geometry.task_half_size
        ):
            self.failure_reason = "object_out_of_bounds"
            return True

        for agent_id in range(2):
            pose = self._robot_pose(agent_id)
            if np.any(
                np.abs(pose[:2] - self.geometry.task_center)
                > self.geometry.task_half_size
            ):
                self.failure_reason = "robot_out_of_bounds"
                return True

        if self.brake_event_active and self.step_count - self.brake_event_step >= int(
            np.ceil(self.cfg.max_response_time / self.cfg.control_dt)
        ):
            self.failure_reason = "response_timeout"
            return True

        if self.step_count >= self.cfg.episode_len:
            self.failure_reason = "episode_timeout"
            return True

        self.failure_reason = "none"
        return False

    def _robot_distance(self) -> float:
        return float(np.linalg.norm(self._robot_pose(0)[:2] - self._robot_pose(1)[:2]))

    def _update_distance_statistics(self) -> None:
        self.min_robot_distance = min(self.min_robot_distance, self._robot_distance())

    def _local_contact_agents(self) -> np.ndarray:
        robot_geom_ids = (
            self._named_id(mujoco.mjtObj.mjOBJ_GEOM, "robot_a_base"),
            self._named_id(mujoco.mjtObj.mjOBJ_GEOM, "robot_b_base"),
        )
        flags = np.zeros(2, dtype=np.float32)
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            for agent_id, geom_id in enumerate(robot_geom_ids):
                if geom_id in pair:
                    flags[agent_id] = 1.0
        return flags

    def _robot_base_effort(self, agent_id: int) -> np.ndarray:
        """Return geometric-controller generalized effort ``[Fx, Fy, Tz]``."""

        address = self.robot_dof_addrs[agent_id]
        return self.data.qfrc_actuator[address : address + 3].astype(
            np.float32, copy=True
        )

    def _robot_gripper_state(self, agent_id: int) -> np.ndarray:
        command_index = 3 if agent_id == 0 else 7
        command = float(self.last_action[command_index])
        return np.asarray([command, float(command > 0.5)], dtype=np.float32)

    def _robot_state(self, agent_id: int) -> np.ndarray:
        return np.concatenate(
            [
                self._robot_pose(agent_id).astype(np.float32),
                self.base_velocities[agent_id].astype(np.float32),
                self._robot_gripper_state(agent_id),
                self._robot_base_effort(agent_id),
            ]
        ).astype(np.float32)

    def _robot_observation(self, agent_id: int) -> Dict[str, np.ndarray]:
        state = self._robot_state(agent_id)
        if self.cfg.include_camera_images:
            image = self.render(
                camera=self.robot_camera_names[agent_id],
                width=self.cfg.agent_camera_width,
                height=self.cfg.agent_camera_height,
            )
        else:
            image = np.zeros(
                (self.cfg.agent_camera_height, self.cfg.agent_camera_width, 3),
                dtype=np.uint8,
            )
        return {
            "state": state,
            "base_pose": state[:3].copy(),
            "base_velocity": state[3:6].copy(),
            "gripper": state[6:8].copy(),
            "base_effort": state[8:11].copy(),
            "image": image,
        }

    def _task_phase(self, success: bool = False, failure: bool = False) -> str:
        if success:
            return "success"
        if failure:
            return "failure"
        if not self.has_grasped:
            return "waiting_for_grasp"
        if not self.brake_event_active:
            return "cruise"
        if not self._agent_stopped(self.braking_agent):
            return "braking_robot_stopping"
        if not self._agent_stopped(self.responding_agent):
            return "responding_robot_stopping"
        return "stable_stop_hold"

    def _acceleration_tracking_error(self) -> float:
        return float(
            abs(
                self.linear_decelerations[self.responding_agent]
                - self.linear_decelerations[self.braking_agent]
            )
        )

    def _build_info(
        self, success: bool | None = None, failure: bool | None = None
    ) -> Dict[str, Any]:
        success = self._success() if success is None else bool(success)
        failure = self.failure_reason != "none" if failure is None else bool(failure)
        speeds = self._linear_speeds()
        speed_error = float(
            abs(speeds[self.responding_agent] - speeds[self.braking_agent])
        )
        response_delay_steps = (
            self.response_start_step - self.brake_event_step
            if self.response_started
            else -1
        )
        return {
            "scenario": "standard",
            "control_dt": float(self.cfg.control_dt),
            "geometry_source": "mujoco_xml",
            "task_phase": self._task_phase(success=success, failure=failure),
            "success": success,
            "failure": bool(failure and not success),
            "failure_reason": self.failure_reason,
            "braking_agent": int(self.braking_agent),
            "responding_agent": int(self.responding_agent),
            "brake_start_step": int(self.brake_start_step),
            "brake_start_time": float(self.brake_start_step * self.cfg.control_dt),
            "brake_event_active": bool(self.brake_event_active),
            "brake_event_step": int(self.brake_event_step),
            "steps_since_brake": int(
                self.step_count - self.brake_event_step
                if self.brake_event_active
                else -1
            ),
            "pre_brake_motion_valid": bool(self.pre_brake_motion_valid),
            "response_started": bool(self.response_started),
            "response_start_step": int(self.response_start_step),
            "response_delay_steps": int(response_delay_steps),
            "response_delay_seconds": float(
                response_delay_steps * self.cfg.control_dt
                if response_delay_steps >= 0
                else -1.0
            ),
            "follower_brake_steps": int(self.follower_brake_steps),
            "stop_hold_steps": int(self.stop_hold_count),
            "speed_agents": speeds.astype(np.float32),
            "base_velocity_agents": self.base_velocities.astype(np.float32, copy=True),
            "base_acceleration_agents": self.base_accelerations.astype(
                np.float32, copy=True
            ),
            "braking_agent_speed": float(speeds[self.braking_agent]),
            "responding_agent_speed": float(speeds[self.responding_agent]),
            "speed_error": speed_error,
            "coordination_error": speed_error,
            "acceleration_tracking_error": self._acceleration_tracking_error(),
            "response_progress": self._response_progress(),
            "both_stopped": bool(self._both_stopped()),
            "grasped": bool(self.grasped),
            "has_grasped": bool(self.has_grasped),
            "robot_distance": self._robot_distance(),
            "min_robot_distance": float(self.min_robot_distance),
            "object_xy_yaw": self._object_pose_xy_yaw(),
            "contacts": int(self.data.ncon),
            "local_contact_agents": self._local_contact_agents(),
            "base_effort_agents": np.stack(
                [self._robot_base_effort(0), self._robot_base_effort(1)], axis=0
            ),
            "step_count": int(self.step_count),
        }

    def _privileged_state(self) -> Dict[str, np.ndarray]:
        info = self._build_info()
        success = float(info["success"])
        failure = float(info["failure"])
        timeout = float(self.failure_reason in {"response_timeout", "episode_timeout"})
        task_state = np.asarray(
            [
                info["response_progress"],
                info["speed_agents"][0],
                info["speed_agents"][1],
                info["speed_error"],
                info["acceleration_tracking_error"],
                float(info["both_stopped"]),
                success,
                failure,
                timeout,
                float(self.step_count),
            ],
            dtype=np.float32,
        )
        event_state = np.asarray(
            [
                float(self.brake_event_active),
                float(self.braking_agent),
                float(self.responding_agent),
                float(self.brake_start_step),
                float(info["steps_since_brake"]),
                float(self.pre_brake_motion_valid),
                float(self.response_started),
                float(info["response_delay_steps"]),
                float(self.follower_brake_steps),
                float(self.stop_hold_count),
            ],
            dtype=np.float32,
        )
        contact_state = np.asarray(
            [
                float(self.data.ncon),
                self._robot_distance(),
                float(self.min_robot_distance),
                float(self.grasped),
                float(self.has_grasped),
            ],
            dtype=np.float32,
        )
        object_pose = self._object_pose_xy_yaw().astype(np.float32)
        object_velocity = self.data.qvel[
            self.object_dof_addr : self.object_dof_addr + 6
        ].astype(np.float32, copy=True)
        state = np.concatenate(
            [object_pose, object_velocity, task_state, event_state, contact_state]
        ).astype(np.float32)
        if state.shape != (self.privileged_state_dim,):
            raise RuntimeError(
                f"privileged state shape drifted to {state.shape}, "
                f"expected {(self.privileged_state_dim,)}"
            )
        return {
            "state": state,
            "object_pose": object_pose,
            "object_velocity": object_velocity,
            "task_bounds": np.concatenate(
                [self.geometry.task_center, self.geometry.task_half_size]
            ).astype(np.float32),
            "object_half_size": self.geometry.object_half_size.astype(
                np.float32, copy=True
            ),
            "task": task_state,
            "braking_event": event_state,
            "contact": contact_state,
        }

    def get_obs(self) -> Dict[str, Any]:
        robot_0 = self._robot_observation(0)
        robot_1 = self._robot_observation(1)
        proprioception = np.concatenate([robot_0["state"], robot_1["state"]]).astype(
            np.float32
        )
        return {
            "robot_0": robot_0,
            "robot_1": robot_1,
            "proprioception": proprioception,
            "privileged_state": self._privileged_state(),
        }

    def _compute_reward(self, success: bool, failure: bool) -> float:
        speeds = self._linear_speeds()
        reward = -self.cfg.reward_time_cost
        reward -= self.cfg.reward_energy_scale * float(
            np.sum(self.executed_action[:3] ** 2)
            + np.sum(self.executed_action[4:7] ** 2)
        )
        if not self.grasped:
            reward -= self.cfg.reward_ungrasped_cost

        if not self.brake_event_active:
            cruise_speed = max(
                0.0,
                min(
                    float(self.base_velocities[0, 1]),
                    float(self.base_velocities[1, 1]),
                ),
            )
            reward += self.cfg.reward_cruise_scale * cruise_speed
            reward -= self.cfg.reward_speed_match_scale * abs(
                float(speeds[0] - speeds[1])
            )
        else:
            progress = self._response_progress()
            progress_delta = max(0.0, progress - self._best_response_progress)
            self._best_response_progress = max(self._best_response_progress, progress)
            speed_error = abs(
                float(speeds[self.responding_agent] - speeds[self.braking_agent])
            )
            reward += self.cfg.reward_response_progress_scale * progress_delta
            reward -= self.cfg.reward_speed_tracking_scale * speed_error
            reward -= (
                self.cfg.reward_acceleration_tracking_scale
                * self._acceleration_tracking_error()
            )
            if self._both_stopped():
                reward += self.cfg.reward_stop_hold

        if success:
            reward += self.cfg.reward_success
        if failure and not success:
            reward -= self.cfg.reward_failure
        return float(reward)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (self.action_dim,):
            raise ValueError(
                f"Expected action shape {(self.action_dim,)}, got {action.shape}"
            )
        action = np.clip(action, -1.0, 1.0)
        self.last_action = action.copy()
        self._previous_speeds = self._linear_speeds()

        self.grasped = bool(action[3] > 0.5 and action[7] > 0.5)
        self.has_grasped = bool(self.has_grasped or self.grasped)
        if not self.brake_event_active and self.step_count >= self.brake_start_step:
            self._activate_brake_event()

        previous_object_pose = self._object_pose_xy_yaw()
        self._integrate_bases(action)
        if self.grasped:
            self._apply_geometric_carry(previous_object_pose)
        else:
            self.data.qvel[self.object_dof_addr : self.object_dof_addr + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)
        self.step_count += 1
        self._update_distance_statistics()
        self._update_response_state()

        success = self._success()
        failure = False if success else self._failure()
        truncated = bool(failure and self.failure_reason == "episode_timeout")
        terminated = bool(success or (failure and not truncated))
        reward = self._compute_reward(success, failure)
        observation = self.get_obs()
        info = self._build_info(success=success, failure=failure)
        return observation, reward, terminated, truncated, info

    def scripted_action(self) -> np.ndarray:
        """Oracle baseline that cruises and then tracks the braking robot speed."""

        home_a_x = float(self.geometry.home_qpos[self.robot_a_qpos_addr])
        home_b_x = float(self.geometry.home_qpos[self.robot_b_qpos_addr])
        target_x = (home_a_x, home_b_x)
        controls: list[float] = []
        leader_forward = max(0.0, float(self.base_velocities[self.braking_agent, 1]))

        for agent_id in range(2):
            pose = self._robot_pose(agent_id)
            vx = float(np.clip(1.5 * (target_x[agent_id] - pose[0]), -0.55, 0.55))
            wz = float(np.clip(-1.2 * pose[2], -0.8, 0.8))
            if not self.brake_event_active:
                vy = 0.70
            elif agent_id == self.braking_agent:
                vy = 0.0
            else:
                vy = float(np.clip(leader_forward / self.cfg.max_action_v, 0.0, 1.0))
            controls.extend([vx, vy, wz, 1.0])
        return np.asarray(controls, dtype=np.float64)

    def get_state_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                self._robot_state(0),
                self._robot_state(1),
                self._privileged_state()["state"],
            ]
        ).astype(np.float64)

    def render(
        self,
        *,
        camera: str = "fixed",
        width: int = 640,
        height: int = 360,
    ) -> np.ndarray:
        if width <= 0 or height <= 0:
            raise ValueError("render dimensions must be positive")
        key = (int(height), int(width))
        renderer = self._renderers.get(key)
        if renderer is None:
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._renderers[key] = renderer
        renderer.update_scene(self.data, camera=camera)
        return np.asarray(renderer.render(), dtype=np.uint8).copy()

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "step_count": self.step_count,
            "last_action": self.last_action.copy(),
            "executed_action": self.executed_action.copy(),
            "base_velocities": self.base_velocities.copy(),
            "base_accelerations": self.base_accelerations.copy(),
            "linear_decelerations": self.linear_decelerations.copy(),
            "grasped": self.grasped,
            "has_grasped": self.has_grasped,
            "failure_reason": self.failure_reason,
            "min_robot_distance": self.min_robot_distance,
            "braking_agent": self.braking_agent,
            "responding_agent": self.responding_agent,
            "brake_start_step": self.brake_start_step,
            "brake_event_active": self.brake_event_active,
            "brake_event_step": self.brake_event_step,
            "pre_brake_motion_valid": self.pre_brake_motion_valid,
            "follower_speed_at_brake": self.follower_speed_at_brake,
            "response_started": self.response_started,
            "response_start_step": self.response_start_step,
            "follower_brake_steps": self.follower_brake_steps,
            "stop_hold_count": self.stop_hold_count,
            "best_response_progress": self._best_response_progress,
            "previous_speeds": self._previous_speeds.copy(),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.data.qpos[:] = state["qpos"]
        self.data.qvel[:] = state["qvel"]
        self.data.ctrl[:] = state["ctrl"]
        self.step_count = int(state["step_count"])
        self.last_action = np.asarray(state["last_action"], dtype=np.float64).copy()
        self.executed_action = np.asarray(
            state["executed_action"], dtype=np.float64
        ).copy()
        self.base_velocities = np.asarray(
            state["base_velocities"], dtype=np.float64
        ).copy()
        self.base_accelerations = np.asarray(
            state["base_accelerations"], dtype=np.float64
        ).copy()
        self.linear_decelerations = np.asarray(
            state["linear_decelerations"], dtype=np.float64
        ).copy()
        self.grasped = bool(state["grasped"])
        self.has_grasped = bool(state["has_grasped"])
        self.failure_reason = str(state["failure_reason"])
        self.min_robot_distance = float(state["min_robot_distance"])
        self.braking_agent = int(state["braking_agent"])
        self.responding_agent = int(state["responding_agent"])
        self.brake_start_step = int(state["brake_start_step"])
        self.brake_event_active = bool(state["brake_event_active"])
        self.brake_event_step = int(state["brake_event_step"])
        self.pre_brake_motion_valid = bool(state["pre_brake_motion_valid"])
        self.follower_speed_at_brake = float(state["follower_speed_at_brake"])
        self.response_started = bool(state["response_started"])
        self.response_start_step = int(state["response_start_step"])
        self.follower_brake_steps = int(state["follower_brake_steps"])
        self.stop_hold_count = int(state["stop_hold_count"])
        self._best_response_progress = float(state["best_response_progress"])
        self._previous_speeds = np.asarray(
            state["previous_speeds"], dtype=np.float64
        ).copy()
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        mujoco.mj_forward(self.model, self.data)


# Compatibility aliases keep external imports working while the public task
# name and all in-repository entry points use the cooperative-stop definition.
CarryEnvConfig = CooperativeStopEnvConfig
CarryGeometry = CooperativeStopGeometry
TwoRobotCarryNarrowPassageEnv = TwoRobotCooperativeStopEnv


def main() -> None:
    env = TwoRobotCooperativeStopEnv()
    try:
        observation, info = env.reset(seed=0)
        del observation
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            observation, reward, terminated, truncated, info = env.step(
                env.scripted_action()
            )
            del observation
            total_reward += reward
        print(
            {
                "return": total_reward,
                "steps": env.step_count,
                "success": info["success"],
                "braking_agent": info["braking_agent"],
                "brake_start_step": info["brake_start_step"],
            }
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
