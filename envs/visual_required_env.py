"""MuJoCo-native environments for the Phase M0 visual-required gate.

The three tasks share the retained cooperative-stop environment's 22-D
proprioception and 8-D action contracts.  An episode seed is interpreted as
``2 * physical_seed + cue_variant``: paired cue episodes therefore reset to
identical physical state while raw MuJoCo RGB carries the task truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np

from envs.mujoco_camera import camera_calibration as mujoco_camera_calibration
from envs.two_robot_carry_env import (
    CooperativeStopEnvConfig,
    TwoRobotCooperativeStopEnv,
)


VISUAL_REQUIRED_TASKS: tuple[str, ...] = (
    "visual_event_stop",
    "visual_target_select",
    "visual_obstacle_avoid",
)

VISUAL_REQUIRED_TASK_TEXTS: Mapping[str, str] = {
    "visual_event_stop": (
        "carry together and obey the fixed-camera stop-or-pass signal at the "
        "decision marker"
    ),
    "visual_target_select": (
        "carry together to the target selected by the fixed-camera visual marker"
    ),
    "visual_obstacle_avoid": (
        "carry together through the unblocked lane visible in the fixed camera"
    ),
}

EVENT_DECISION_STEP = 14
LANE_CENTER_X = 0.62
COMMITMENT_Y = -0.12
GOAL_Y = 0.56
STOP_LINE_Y = 0.08

CAMERA_NAMES: tuple[str, ...] = (
    "fixed",
    "robot_0_camera",
    "robot_1_camera",
)

_NEUTRAL = np.asarray([0.10, 0.10, 0.12, 1.0], dtype=np.float32)
_OFF = np.asarray([0.025, 0.004, 0.004, 1.0], dtype=np.float32)
_STOP_RED = np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
_PASS_GREEN = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
_TARGET_ACTIVE = np.asarray([1.0, 0.82, 0.0, 1.0], dtype=np.float32)
_TARGET_IDLE = np.asarray([0.12, 0.12, 0.14, 1.0], dtype=np.float32)
_OBSTACLE = np.asarray([1.0, 0.34, 0.0, 1.0], dtype=np.float32)
_INVISIBLE = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


@dataclass(frozen=True)
class VisualRequiredEnvConfig:
    """Configuration for one deterministic M0 visual-required task."""

    task_id: str = "visual_event_stop"
    control_dt: float = 0.05
    episode_len: int = 64
    image_width: int = 96
    image_height: int = 96
    render_cue_mode: str = "truth"
    randomization_template_id: str = "template_default"
    max_linear_speed: float = 0.72
    max_angular_speed: float = 1.2
    max_linear_acceleration: float = 2.4
    max_angular_acceleration: float = 4.0

    def __post_init__(self) -> None:
        if self.task_id not in VISUAL_REQUIRED_TASKS:
            raise ValueError(
                f"unknown visual-required task {self.task_id!r}; "
                f"expected one of {VISUAL_REQUIRED_TASKS}"
            )
        if self.render_cue_mode not in {"truth", "opposite"}:
            raise ValueError("render_cue_mode must be 'truth' or 'opposite'")
        if not self.randomization_template_id:
            raise ValueError("randomization_template_id cannot be empty")
        positive = {
            "control_dt": self.control_dt,
            "episode_len": self.episode_len,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "max_linear_speed": self.max_linear_speed,
            "max_angular_speed": self.max_angular_speed,
            "max_linear_acceleration": self.max_linear_acceleration,
            "max_angular_acceleration": self.max_angular_acceleration,
        }
        for name, value in positive.items():
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.episode_len <= EVENT_DECISION_STEP + 6:
            raise ValueError("episode_len is too short for the visual event task")
        if self.image_width < 32 or self.image_height < 32:
            raise ValueError("visual-required RGB must be at least 32x32")


class VisualRequiredEnv:
    """Paired-cue benchmark backed by the repository's MuJoCo carry model."""

    action_dim = 8
    proprioception_dim = 22
    robot_state_dim = 11
    camera_names = CAMERA_NAMES

    def __init__(self, cfg: VisualRequiredEnvConfig | None = None) -> None:
        self.cfg = cfg or VisualRequiredEnvConfig()
        xml_path = (Path(__file__).resolve().parent / "assets" / "two_robot_carry.xml")
        self.model_xml_path = str(xml_path.resolve())
        self.model_xml_sha256 = hashlib.sha256(xml_path.read_bytes()).hexdigest()
        self.renderer_backend = "mujoco.Renderer"
        self.geometry_source = "mujoco_xml"
        self._env = TwoRobotCooperativeStopEnv(
            CooperativeStopEnvConfig(
                xml_path=self.model_xml_path,
                control_dt=self.cfg.control_dt,
                episode_len=max(self.cfg.episode_len, EVENT_DECISION_STEP + 8),
                max_action_v=self.cfg.max_linear_speed,
                max_action_w=self.cfg.max_angular_speed,
                max_linear_acceleration=self.cfg.max_linear_acceleration,
                max_angular_acceleration=self.cfg.max_angular_acceleration,
                include_camera_images=False,
                agent_camera_width=self.cfg.image_width,
                agent_camera_height=self.cfg.image_height,
            )
        )
        self.model = self._env.model
        self.data = self._env.data
        self._configure_visual_camera_rig()
        self._geom_ids = {
            name: self._named_geom(name)
            for name in (
                "visual_event_signal",
                "visual_target_left",
                "visual_target_right",
                "visual_obstacle_left",
                "visual_obstacle_right",
                "visual_obstacle_collision_left",
                "visual_obstacle_collision_right",
                "robot_0_brake_light",
                "robot_1_brake_light",
                "robot_a_base",
                "robot_b_base",
                "carry_object_geom",
            )
        }
        self._success = False
        self._failure = False
        self._failure_reason = "none"
        self._stop_hold_count = 0
        self._slow_hold_count = 0
        self._committed = False
        self._committed_lane = 0
        self._visual_signal_active = False
        self._brake_light_hold = np.zeros(2, dtype=bool)
        self._brake_light_active = np.zeros(2, dtype=bool)
        self.step_count = 0
        self.episode_seed = 0
        self.physical_seed = 0
        self.cue_variant = 0
        self.scene_id = ""
        self.object_combination_id = ""

    @property
    def control_dt(self) -> float:
        return float(self.cfg.control_dt)

    @property
    def task_condition(self) -> dict[str, str]:
        return {
            "id": self.cfg.task_id,
            "text": str(VISUAL_REQUIRED_TASK_TEXTS[self.cfg.task_id]),
        }

    @property
    def rendered_cue_variant(self) -> int:
        if self.cfg.render_cue_mode == "opposite":
            return 1 - int(self.cue_variant)
        return int(self.cue_variant)

    @property
    def visual_signal_onset_step(self) -> int:
        return EVENT_DECISION_STEP if self.cfg.task_id == "visual_event_stop" else 0

    def _named_geom(self, name: str) -> int:
        geom_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name))
        if geom_id < 0:
            raise ValueError(f"MuJoCo XML is missing required geom {name!r}")
        return geom_id

    def reset(
        self, seed: int | None = None, randomize: bool = True
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        episode_seed = 0 if seed is None else int(seed)
        if episode_seed < 0:
            raise ValueError("episode seed cannot be negative")
        self.episode_seed = episode_seed
        self.physical_seed, self.cue_variant = divmod(episode_seed, 2)

        template_digest = hashlib.sha256(
            self.cfg.randomization_template_id.encode("utf-8")
        ).digest()
        template_seed = int.from_bytes(template_digest[:8], "little", signed=False)
        task_index = VISUAL_REQUIRED_TASKS.index(self.cfg.task_id)
        seed_rng = np.random.default_rng(
            np.random.SeedSequence(
                [self.physical_seed, template_seed & 0xFFFFFFFF, task_index]
            )
        )
        backend_seed = int(seed_rng.integers(0, np.iinfo(np.int32).max))
        self._env.reset(seed=backend_seed, randomize=bool(randomize))
        self._env.brake_event_active = False
        self._env.brake_event_step = -1
        self._env.step_count = 0
        self._env.base_velocities[:] = 0.0
        self._env.base_accelerations[:] = 0.0
        self._env.linear_decelerations[:] = 0.0
        self._env.last_action[:] = 0.0
        self._env.executed_action[:] = 0.0

        self._success = False
        self._failure = False
        self._failure_reason = "none"
        self._stop_hold_count = 0
        self._slow_hold_count = 0
        self._committed = False
        self._committed_lane = 0
        self._brake_light_hold[:] = False
        self._brake_light_active[:] = False
        self.step_count = 0
        self._visual_signal_active = self.visual_signal_onset_step == 0
        self.scene_id = (
            f"{self.cfg.task_id}:{self.cfg.randomization_template_id}:"
            f"{self.physical_seed:08d}"
        )
        object_bucket = int.from_bytes(template_digest[12:14], "little") % 17
        self.object_combination_id = (
            f"{self.cfg.task_id}:{self.cfg.randomization_template_id}:"
            f"objects:{object_bucket:02d}"
        )
        self._configure_task_geometry()
        mujoco.mj_forward(self.model, self.data)
        return self._observation(), self._info()

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._success or self._failure:
            raise RuntimeError("step called after episode termination")
        command = np.asarray(action, dtype=np.float64).reshape(-1)
        if command.shape != (self.action_dim,):
            raise ValueError(
                f"expected action shape {(self.action_dim,)}, got {command.shape}"
            )
        if not np.isfinite(command).all():
            raise ValueError("action must be finite")
        command = np.clip(command, -1.0, 1.0)
        self._env.last_action = command.copy()
        self._env.grasped = bool(command[3] > 0.5 and command[7] > 0.5)
        self._env.has_grasped = bool(self._env.has_grasped or self._env.grasped)
        previous_object_pose = self._env._object_pose_xy_yaw()
        previous_forward_speeds = self._env.base_velocities[:, 1].copy()
        # The visual event never forces a cue-dependent robot velocity.  This is
        # the causal boundary that keeps paired proprioception/action history
        # identical until a policy acts on the newly visible signal.
        self._env.brake_event_active = False
        self._env._integrate_bases(command)
        if self._env.grasped:
            self._env._apply_geometric_carry(previous_object_pose)
        else:
            start = self._env.object_dof_addr
            self.data.qvel[start : start + 6] = 0.0

        self.step_count += 1
        self._env.step_count = self.step_count
        if (
            self.cfg.task_id == "visual_event_stop"
            and not self._visual_signal_active
            and self.step_count >= self.visual_signal_onset_step
        ):
            # Runtime renders this next state before the following policy.act.
            # Task commitment is deliberately deferred until that next action.
            self._visual_signal_active = True
            self._configure_task_geometry()
        self._update_brake_lights_from_action(command, previous_forward_speeds)
        mujoco.mj_forward(self.model, self.data)
        self._evaluate_task()

        truncated = False
        if (
            not self._success
            and not self._failure
            and self.step_count >= self.cfg.episode_len
        ):
            self._failure = True
            self._failure_reason = "episode_timeout"
            truncated = True
        terminated = bool(self._success or (self._failure and not truncated))
        reward = 1.0 if self._success else (-1.0 if self._failure else -0.01)
        return self._observation(), float(reward), terminated, truncated, self._info()

    def scripted_action(self) -> np.ndarray:
        """Return the privileged truth-controller action for the current state."""

        return control_action(
            self.cfg.task_id,
            int(self.cue_variant),
            self._observation()["proprioception"],
            step_count=self.step_count,
        )

    def render(
        self,
        *,
        camera: str = "fixed",
        width: int | None = None,
        height: int | None = None,
    ) -> np.ndarray:
        if camera not in self.camera_names:
            raise ValueError(
                f"visual-required environment exposes cameras {self.camera_names}, "
                f"not {camera!r}"
            )
        return self._env.render(
            camera=camera,
            width=self.cfg.image_width if width is None else int(width),
            height=self.cfg.image_height if height is None else int(height),
        )

    def camera_calibration(
        self,
        *,
        camera: str = "fixed",
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]:
        if camera not in self.camera_names:
            raise ValueError(
                f"visual-required environment exposes cameras {self.camera_names}, "
                f"not {camera!r}"
            )
        return mujoco_camera_calibration(
            self.model,
            self.data,
            camera=camera,
            width=self.cfg.image_width if width is None else int(width),
            height=self.cfg.image_height if height is None else int(height),
        )

    def close(self) -> None:
        self._env.close()

    def _configure_visual_camera_rig(self) -> None:
        """Apply the visual benchmark's chase rig to this model instance only.

        The shared XML keeps the standard environment's legacy onboard cameras.
        ``MjModel`` arrays are instance-local, so changing them here cannot alter
        the standard environment or the XML provenance identified by its hash.
        """

        camera_x = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        camera_y = np.asarray([0.0, 0.55, 0.84], dtype=np.float64)
        camera_x /= np.linalg.norm(camera_x)
        camera_y -= camera_x * float(np.dot(camera_x, camera_y))
        camera_y /= np.linalg.norm(camera_y)
        camera_z = np.cross(camera_x, camera_y)
        rotation = np.column_stack((camera_x, camera_y, camera_z))
        chase_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(chase_quaternion, rotation.ravel())

        for camera_name in ("robot_0_camera", "robot_1_camera"):
            camera_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
            )
            if camera_id < 0:
                raise ValueError(f"MuJoCo XML is missing camera {camera_name!r}")
            self.model.cam_pos[camera_id] = np.asarray(
                [0.0, -0.90, 0.75], dtype=np.float64
            )
            self.model.cam_quat[camera_id] = chase_quaternion
            self.model.cam_fovy[camera_id] = 100.0
        mujoco.mj_forward(self.model, self.data)

    def _configure_task_geometry(self) -> None:
        rgba = self.model.geom_rgba
        rgba[self._geom_ids["visual_event_signal"]] = _INVISIBLE
        rgba[self._geom_ids["visual_target_left"]] = _INVISIBLE
        rgba[self._geom_ids["visual_target_right"]] = _INVISIBLE
        rgba[self._geom_ids["visual_obstacle_left"]] = _INVISIBLE
        rgba[self._geom_ids["visual_obstacle_right"]] = _INVISIBLE
        rgba[self._geom_ids["robot_0_brake_light"]] = _OFF
        rgba[self._geom_ids["robot_1_brake_light"]] = _OFF
        for name in (
            "visual_obstacle_collision_left",
            "visual_obstacle_collision_right",
        ):
            geom_id = self._geom_ids[name]
            rgba[geom_id] = _INVISIBLE
            self.model.geom_contype[geom_id] = 0
            self.model.geom_conaffinity[geom_id] = 0

        rendered = self.rendered_cue_variant
        if self.cfg.task_id == "visual_event_stop":
            rgba[self._geom_ids["visual_event_signal"]] = (
                (_STOP_RED if rendered == 0 else _PASS_GREEN)
                if self._visual_signal_active
                else _NEUTRAL
            )
            return

        if self.cfg.task_id == "visual_target_select":
            rgba[self._geom_ids["visual_target_left"]] = (
                _TARGET_ACTIVE if rendered == 0 else _TARGET_IDLE
            )
            rgba[self._geom_ids["visual_target_right"]] = (
                _TARGET_ACTIVE if rendered == 1 else _TARGET_IDLE
            )
            return

        rendered_side = "left" if rendered == 0 else "right"
        rgba[self._geom_ids[f"visual_obstacle_{rendered_side}"]] = _OBSTACLE
        truth_side = "left" if self.cue_variant == 0 else "right"
        collision_id = self._geom_ids[f"visual_obstacle_collision_{truth_side}"]
        self.model.geom_contype[collision_id] = 1
        self.model.geom_conaffinity[collision_id] = 1

    def _update_brake_lights_from_action(
        self,
        command: np.ndarray,
        previous_forward_speeds: np.ndarray,
    ) -> None:
        """Drive each red brake light from that robot's deceleration request.

        The stop/pass cue is carried only by ``visual_event_signal``.  In
        particular, the cue (including the opposite-RGB intervention) never
        colors a robot light.  A light turns red after its agent requests a
        lower forward velocity, and stays red while a zero-speed brake command
        is held.  No brake-light state can ever be green.
        """

        desired_forward_speeds = (
            np.asarray(command, dtype=np.float64)[[1, 5]]
            * float(self.cfg.max_linear_speed)
        )
        previous = np.asarray(previous_forward_speeds, dtype=np.float64)
        newly_braking = (previous > 0.02) & (
            desired_forward_speeds < previous - 1e-6
        )
        holding_brake = self._brake_light_hold & (
            np.abs(desired_forward_speeds) <= 1e-6
        )
        self._brake_light_active = np.asarray(
            newly_braking | holding_brake, dtype=bool
        )
        self._brake_light_hold = self._brake_light_active.copy()
        for agent_id, name in enumerate(
            ("robot_0_brake_light", "robot_1_brake_light")
        ):
            self.model.geom_rgba[self._geom_ids[name]] = (
                _STOP_RED if self._brake_light_active[agent_id] else _OFF
            )

    def _observation(self) -> dict[str, np.ndarray]:
        proprioception = np.concatenate(
            [self._env._robot_state(0), self._env._robot_state(1)]
        ).astype(np.float32)
        if proprioception.shape != (self.proprioception_dim,):
            raise RuntimeError("visual-required proprioception shape drift")
        return {"proprioception": proprioception}

    def _info(self) -> dict[str, Any]:
        center = self._center_pose()
        return {
            "task_id": self.cfg.task_id,
            "task_text": VISUAL_REQUIRED_TASK_TEXTS[self.cfg.task_id],
            "episode_seed": int(self.episode_seed),
            "physical_seed": int(self.physical_seed),
            "cue_variant": int(self.cue_variant),
            "cue_name": self._cue_name(int(self.cue_variant)),
            "rendered_cue_variant": int(self.rendered_cue_variant),
            "rgb_intervention": self.cfg.render_cue_mode,
            "visual_signal_active": bool(self._visual_signal_active),
            "visual_signal_onset_step": int(self.visual_signal_onset_step),
            "visual_signal_onset_time": float(
                self.visual_signal_onset_step * self.control_dt
            ),
            "visual_signal_kind": self._visual_signal_kind(),
            "brake_light_active_agents": [
                int(agent_id)
                for agent_id, active in enumerate(self._brake_light_active)
                if active
            ],
            "brake_light_semantics": "per_agent_deceleration_command_red_only",
            "cue_visible_expected": {
                camera: True for camera in self.camera_names
            },
            "randomization_template_id": self.cfg.randomization_template_id,
            "scene_id": self.scene_id,
            "object_combination_id": self.object_combination_id,
            "object_nuisance_scale": 1.0,
            "geometry_source": self.geometry_source,
            "renderer_backend": self.renderer_backend,
            "model_xml_path": self.model_xml_path,
            "model_xml_sha256": self.model_xml_sha256,
            "camera_names": self.camera_names,
            "step_count": int(self.step_count),
            "success": bool(self._success),
            "failure": bool(self._failure and not self._success),
            "failure_reason": self._failure_reason,
            "center_xy": center[:2].astype(np.float32),
            "center_velocity": self._env.base_velocities.mean(axis=0).astype(
                np.float32
            ),
            "commitment_made": bool(self._committed),
            "committed_lane": int(self._committed_lane),
            "commanded_action": self._env.last_action.astype(np.float32, copy=True),
            "executed_action": self._env.executed_action.astype(
                np.float32, copy=True
            ),
        }

    def _center_pose(self) -> np.ndarray:
        return 0.5 * (self._env._robot_pose(0) + self._env._robot_pose(1))

    def _evaluate_task(self) -> None:
        if self.step_count >= 2 and not self._env.grasped:
            self._failure = True
            self._failure_reason = "grip_released"
            return
        if self.cfg.task_id == "visual_event_stop":
            self._evaluate_event()
        elif self.cfg.task_id == "visual_target_select":
            self._evaluate_lane_task(target_lane=-1 if self.cue_variant == 0 else 1)
        else:
            safe_lane = 1 if self.cue_variant == 0 else -1
            self._evaluate_lane_task(target_lane=safe_lane, obstacle=True)

    def _evaluate_event(self) -> None:
        center_y = float(self._center_pose()[1])
        forward_speed = float(self._env.base_velocities[:, 1].mean())
        speed = float(np.linalg.norm(self._env.base_velocities[:, :2], axis=1).mean())
        if 4 <= self.step_count < self.visual_signal_onset_step and forward_speed < 0.18:
            self._failure = True
            self._failure_reason = "premature_stop"
            return
        # The action that produces the onset state was selected from a neutral
        # current image.  Only the next policy action may commit to stop/pass.
        if self.step_count <= self.visual_signal_onset_step:
            return
        if self.cue_variant == 0:
            self._stop_hold_count = self._stop_hold_count + 1 if speed <= 0.045 else 0
            if center_y >= STOP_LINE_Y:
                self._failure = True
                self._failure_reason = "missed_stop_line"
            elif self._stop_hold_count >= 3:
                self._success = True
                self._failure_reason = "none"
        else:
            self._slow_hold_count = self._slow_hold_count + 1 if speed <= 0.06 else 0
            if self._slow_hold_count >= 3 and center_y < GOAL_Y:
                self._failure = True
                self._failure_reason = "stopped_on_pass"
            elif center_y >= GOAL_Y:
                if forward_speed >= 0.24:
                    self._success = True
                    self._failure_reason = "none"
                else:
                    self._failure = True
                    self._failure_reason = "passed_too_slowly"

    def _evaluate_lane_task(self, *, target_lane: int, obstacle: bool = False) -> None:
        center = self._center_pose()
        center_x, center_y = float(center[0]), float(center[1])
        if obstacle and self._active_obstacle_contact():
            self._failure = True
            self._failure_reason = "obstacle_collision"
            return
        if not self._committed and center_y >= COMMITMENT_Y:
            if abs(center_x) < 0.28:
                self._failure = True
                self._failure_reason = (
                    "unsafe_obstacle_clearance"
                    if obstacle
                    else "ambiguous_target_choice"
                )
                return
            self._committed = True
            self._committed_lane = -1 if center_x < 0.0 else 1
            if self._committed_lane != target_lane:
                self._failure = True
                self._failure_reason = (
                    "obstacle_collision" if obstacle else "wrong_target"
                )
                return
        if center_y >= GOAL_Y:
            target_x = target_lane * LANE_CENTER_X
            if self._committed and abs(center_x - target_x) <= 0.24:
                self._success = True
                self._failure_reason = "none"
            else:
                self._failure = True
                self._failure_reason = (
                    "unsafe_obstacle_clearance" if obstacle else "missed_target"
                )

    def _active_obstacle_contact(self) -> bool:
        side = "left" if self.cue_variant == 0 else "right"
        obstacle_id = self._geom_ids[f"visual_obstacle_collision_{side}"]
        moving_ids = {
            self._geom_ids["robot_a_base"],
            self._geom_ids["robot_b_base"],
            self._geom_ids["carry_object_geom"],
        }
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if obstacle_id in pair and pair.intersection(moving_ids):
                return True
        return False

    def _cue_name(self, cue: int) -> str:
        if self.cfg.task_id == "visual_event_stop":
            return "stop" if cue == 0 else "pass"
        if self.cfg.task_id == "visual_target_select":
            return "left_target" if cue == 0 else "right_target"
        return "left_blocked" if cue == 0 else "right_blocked"

    def _visual_signal_kind(self) -> str:
        return {
            "visual_event_stop": "stop_pass",
            "visual_target_select": "target_selection",
            "visual_obstacle_avoid": "obstacle_blockage",
        }[self.cfg.task_id]


def control_action(
    task_id: str,
    cue_variant: int,
    proprioception: np.ndarray,
    *,
    step_count: int,
) -> np.ndarray:
    """Shared deterministic controller used by truth and observation policies."""

    if task_id not in VISUAL_REQUIRED_TASKS:
        raise ValueError(f"unknown visual-required task {task_id!r}")
    if cue_variant not in {0, 1}:
        raise ValueError("cue_variant must be 0 or 1")
    state = np.asarray(proprioception, dtype=np.float32)
    if state.shape != (22,) or not np.isfinite(state).all():
        raise ValueError("proprioception must be finite float data with shape (22,)")
    center_x = 0.5 * float(state[0] + state[11])
    action = np.zeros(8, dtype=np.float32)
    action[[3, 7]] = 1.0
    forward = 0.90
    lateral = 0.0
    if task_id == "visual_event_stop":
        if step_count >= EVENT_DECISION_STEP and cue_variant == 0:
            forward = 0.0
    else:
        if task_id == "visual_target_select":
            lane = -1 if cue_variant == 0 else 1
        else:
            lane = 1 if cue_variant == 0 else -1
        target_x = lane * LANE_CENTER_X
        lateral = float(np.clip(2.6 * (target_x - center_x), -0.9, 0.9))
    action[[0, 4]] = lateral
    action[[1, 5]] = forward
    return action


__all__ = [
    "CAMERA_NAMES",
    "EVENT_DECISION_STEP",
    "VISUAL_REQUIRED_TASKS",
    "VISUAL_REQUIRED_TASK_TEXTS",
    "VisualRequiredEnv",
    "VisualRequiredEnvConfig",
    "control_action",
]
