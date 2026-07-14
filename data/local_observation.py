"""Deployable, ego-local observation contract for FE-PC-WAM.

The objects in this module deliberately separate two concepts:

* :class:`PrivilegedAgentState` is a simulator-side value used only by the
  measurement generator and by privileged training targets.
* :class:`LocalObservationPacket` is the only observation that may be passed
  to a decentralized policy at deployment time.

In particular, a packet contains neither the ego robot's global pose nor the
  other robot's private observation or state.  The object pose is a noisy
  local-perception estimate expressed in the ego SE(2) frame.  Teammate belief
  is formed later from local interaction history and selectively received plan
  messages; no explicit teammate pose is part of this contract.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, Mapping

import numpy as np

if TYPE_CHECKING:
    import torch


SCHEMA_NAME = "fe_pc_wam_local_observation"
OBJECT_ESTIMATE_FIELDS = (
    "estimates/object/pose",
    "estimates/object/valid",
    "estimates/object/confidence",
    "estimates/object/age",
)


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap radians to ``[-pi, pi)`` while preserving scalar inputs."""

    wrapped = (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi
    if np.ndim(angle) == 0:
        return float(wrapped)
    return wrapped


def ego_relative_pose(target_world: np.ndarray, ego_world: np.ndarray) -> np.ndarray:
    """Transform an SE(2) world pose into the ego robot's coordinate frame.

    The legacy environment used a world-axis subtraction for translation.
     uses ``R(-ego_yaw) (target_xy - ego_xy)`` so the resulting estimate is
    a sensor-compatible, robot-relative quantity.
    """

    target = np.asarray(target_world, dtype=np.float64).reshape(3)
    ego = np.asarray(ego_world, dtype=np.float64).reshape(3)
    dx, dy = target[0] - ego[0], target[1] - ego[1]
    c, s = np.cos(ego[2]), np.sin(ego[2])
    return np.asarray(
        [
            c * dx + s * dy,
            -s * dx + c * dy,
            wrap_angle(target[2] - ego[2]),
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class LocalObservationSpec:
    """Shape metadata for one deployable observation packet.

    The real platform supplies a three-dimensional planar base twist plus
    joint position, velocity, and torque.  ``joint_dim=0`` is valid for the
    current mobile-base-only simulator.
    """

    joint_dim: int = 0
    force_dim: int = 1
    base_twist_dim: int = 3
    private_event_cue_dim: int = 3

    def __post_init__(self):
        if self.joint_dim < 0:
            raise ValueError("joint_dim must be non-negative")
        if self.force_dim <= 0:
            raise ValueError("force_dim must be positive")
        if self.base_twist_dim != 3:
            raise ValueError(" currently requires planar base twist [vx, vy, wz]")
        if self.private_event_cue_dim != 3:
            raise ValueError("private event cue must encode left/hold/right with 3 values")

    @property
    def flat_dim(self) -> int:
        # sensors + object estimate + goal + private cue/status + next-gate context
        return 3 + 3 * self.joint_dim + self.force_dim + 2 + 6 + 3 + 3 + 1 + 1 + 3

    @property
    def model_observation_dim(self) -> int:
        """Onboard/task features passed to the temporal encoder (object excluded)."""

        return self.flat_dim - 6

    def field_shapes(self) -> Dict[str, tuple[int, ...]]:
        return {
            "self/base_twist": (3,),
            "self/joint_position": (self.joint_dim,),
            "self/joint_velocity": (self.joint_dim,),
            "self/joint_torque": (self.joint_dim,),
            "local/force": (self.force_dim,),
            "local/contact": (1,),
            "local/grasp": (1,),
            "estimates/object/pose": (3,),
            "estimates/object/valid": (1,),
            "estimates/object/confidence": (1,),
            "estimates/object/age": (1,),
            "task/goal": (3,),
            "task/private_event_cue": (self.private_event_cue_dim,),
            "task/private_event_valid": (1,),
            "task/private_event_age": (1,),
            "task/next_gate_context": (3,),
        }

    def feature_names(self) -> list[str]:
        names = ["base_vx", "base_vy", "base_wz"]
        for prefix in ("joint_position", "joint_velocity", "joint_torque"):
            names.extend(f"{prefix}_{i}" for i in range(self.joint_dim))
        names.extend(f"local_force_{i}" for i in range(self.force_dim))
        names.extend(["local_contact", "local_grasp"])
        names.extend(
            [
                "object_ego_x",
                "object_ego_y",
                "object_ego_yaw",
                "object_valid",
                "object_confidence",
                "object_age_s",
            ]
        )
        names.extend(["goal_ego_x", "goal_ego_y", "goal_ego_yaw"])
        names.extend(
            [
                "private_event_left",
                "private_event_hold",
                "private_event_right",
                "private_event_valid",
                "private_event_age_s",
                "next_gate_distance_y",
                "next_gate_index",
                "next_gate_active",
            ]
        )
        return names

    def model_field_names(self) -> list[str]:
        """Canonical packet fields excluding the separately encoded object estimate."""

        return [name for name in self.field_shapes() if name not in OBJECT_ESTIMATE_FIELDS]

    def model_feature_names(self) -> list[str]:
        names = ["base_vx", "base_vy", "base_wz"]
        for prefix in ("joint_position", "joint_velocity", "joint_torque"):
            names.extend(f"{prefix}_{i}" for i in range(self.joint_dim))
        names.extend(f"local_force_{i}" for i in range(self.force_dim))
        names.extend(["local_contact", "local_grasp"])
        names.extend(["goal_ego_x", "goal_ego_y", "goal_ego_yaw"])
        names.extend(
            [
                "private_event_left",
                "private_event_hold",
                "private_event_right",
                "private_event_valid",
                "private_event_age_s",
                "next_gate_distance_y",
                "next_gate_index",
                "next_gate_active",
            ]
        )
        return names


@dataclass
class PoseEstimate:
    pose: np.ndarray
    valid: np.ndarray
    confidence: np.ndarray
    age: np.ndarray

    def validate(self, name: str) -> None:
        _require_shape(self.pose, (3,), f"{name}.pose")
        _require_shape(self.valid, (1,), f"{name}.valid")
        _require_shape(self.confidence, (1,), f"{name}.confidence")
        _require_shape(self.age, (1,), f"{name}.age")
        if not 0.0 <= float(self.valid[0]) <= 1.0:
            raise ValueError(f"{name}.valid must be in [0, 1]")
        if not 0.0 <= float(self.confidence[0]) <= 1.0:
            raise ValueError(f"{name}.confidence must be in [0, 1]")
        if float(self.age[0]) < 0.0:
            raise ValueError(f"{name}.age must be non-negative")


@dataclass
class LocalObservationPacket:
    """One deployable observation produced from local sensors/perception."""

    base_twist: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    joint_torque: np.ndarray
    local_force: np.ndarray
    contact: np.ndarray
    grasp: np.ndarray
    object_estimate: PoseEstimate
    task_goal: np.ndarray
    private_event_cue: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    private_event_valid: np.ndarray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float32)
    )
    private_event_age: np.ndarray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float32)
    )
    next_gate_context: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )

    def validate(self, spec: LocalObservationSpec) -> None:
        _require_shape(self.base_twist, (3,), "base_twist")
        _require_shape(self.joint_position, (spec.joint_dim,), "joint_position")
        _require_shape(self.joint_velocity, (spec.joint_dim,), "joint_velocity")
        _require_shape(self.joint_torque, (spec.joint_dim,), "joint_torque")
        _require_shape(self.local_force, (spec.force_dim,), "local_force")
        _require_shape(self.contact, (1,), "contact")
        _require_shape(self.grasp, (1,), "grasp")
        _require_shape(self.task_goal, (3,), "task_goal")
        _require_shape(self.private_event_cue, (3,), "private_event_cue")
        _require_shape(self.private_event_valid, (1,), "private_event_valid")
        _require_shape(self.private_event_age, (1,), "private_event_age")
        _require_shape(self.next_gate_context, (3,), "next_gate_context")
        if float(self.private_event_valid[0]) not in (0.0, 1.0):
            raise ValueError("private_event_valid must be binary")
        if float(self.private_event_age[0]) < 0.0:
            raise ValueError("private_event_age must be non-negative")
        if float(self.private_event_valid[0]) == 0.0 and np.any(self.private_event_cue != 0.0):
            raise ValueError("invalid private event cue must be zeroed")
        self.object_estimate.validate("object_estimate")

        for name, value in self.as_mapping().items():
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")

    def as_mapping(self) -> Dict[str, np.ndarray]:
        """Return schema paths mapped to float32 arrays."""

        return {
            "self/base_twist": _float32(self.base_twist),
            "self/joint_position": _float32(self.joint_position),
            "self/joint_velocity": _float32(self.joint_velocity),
            "self/joint_torque": _float32(self.joint_torque),
            "local/force": _float32(self.local_force),
            "local/contact": _float32(self.contact),
            "local/grasp": _float32(self.grasp),
            "estimates/object/pose": _float32(self.object_estimate.pose),
            "estimates/object/valid": _float32(self.object_estimate.valid),
            "estimates/object/confidence": _float32(self.object_estimate.confidence),
            "estimates/object/age": _float32(self.object_estimate.age),
            "task/goal": _float32(self.task_goal),
            "task/private_event_cue": _float32(self.private_event_cue),
            "task/private_event_valid": _float32(self.private_event_valid),
            "task/private_event_age": _float32(self.private_event_age),
            "task/next_gate_context": _float32(self.next_gate_context),
        }

    def to_flat(self, spec: LocalObservationSpec) -> np.ndarray:
        self.validate(spec)
        values = self.as_mapping()
        return np.concatenate([values[name].reshape(-1) for name in spec.field_shapes()]).astype(np.float32)


@dataclass
class PrivilegedAgentState:
    """Simulator truth consumed by measurement generation, never policy input."""

    ego_pose_world: np.ndarray
    object_pose_world: np.ndarray
    task_goal_world: np.ndarray
    base_twist: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    joint_position: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    joint_velocity: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    joint_torque: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    local_force: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))
    contact: bool = False
    grasp: bool = False
    private_event_cue: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    private_event_valid: bool = False
    private_event_age: float = 0.0
    next_gate_context: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))


@dataclass(frozen=True)
class SensorSimulationConfig:
    """Noise/dropout parameters for simulator-side local perception."""

    control_dt: float = 0.05
    base_twist_std: float = 0.01
    joint_position_std: float = 0.002
    joint_velocity_std: float = 0.005
    joint_torque_std: float = 0.01
    force_std: float = 0.02
    object_position_std: float = 0.025
    object_yaw_std: float = 0.035
    object_dropout_prob: float = 0.05
    stale_confidence_tau: float = 0.50
    missing_age: float = 10.0

    def __post_init__(self):
        if self.control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        for name in ("object_dropout_prob",):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.stale_confidence_tau <= 0.0:
            raise ValueError("stale_confidence_tau must be positive")


class LocalObservationSimulator:
    """Generate noisy/occluded local packets from privileged simulator state.

    On an occluded frame the last local estimate is retained, marked invalid,
    aged, and assigned decaying confidence.  No hidden truth is used to update
    the stale estimate.
    """

    def __init__(
        self,
        spec: LocalObservationSpec | None = None,
        config: SensorSimulationConfig | None = None,
        seed: int = 0,
    ):
        self.spec = spec or LocalObservationSpec()
        self.config = config or SensorSimulationConfig()
        self.rng = np.random.default_rng(seed)
        self._cache: Dict[tuple[int, str], PoseEstimate] = {}

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._cache.clear()

    def snapshot(self) -> dict:
        """Capture RNG and stale-estimate state for matched interventions."""

        return {
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "cache": {key: _copy_estimate(value) for key, value in self._cache.items()},
        }

    def restore(self, state: Mapping[str, object]) -> None:
        """Restore :meth:`snapshot` without leaking branch observations."""

        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        cache = state["cache"]
        if not isinstance(cache, Mapping):
            raise TypeError("sensor snapshot cache must be a mapping")
        self._cache = {key: _copy_estimate(value) for key, value in cache.items()}

    def observe(
        self,
        agent_id: int,
        truth: PrivilegedAgentState,
        *,
        object_visible: bool = True,
    ) -> LocalObservationPacket:
        if agent_id < 0:
            raise ValueError("agent_id must be non-negative")
        self._validate_truth(truth)
        cfg = self.config

        object_visible = bool(object_visible) and self.rng.random() >= cfg.object_dropout_prob

        object_estimate = self._pose_measurement(
            agent_id,
            "object",
            truth.object_pose_world,
            truth.ego_pose_world,
            object_visible,
            cfg.object_position_std,
            cfg.object_yaw_std,
        )
        packet = LocalObservationPacket(
            base_twist=self._noisy(truth.base_twist, cfg.base_twist_std),
            joint_position=self._noisy(truth.joint_position, cfg.joint_position_std),
            joint_velocity=self._noisy(truth.joint_velocity, cfg.joint_velocity_std),
            joint_torque=self._noisy(truth.joint_torque, cfg.joint_torque_std),
            local_force=np.clip(
                self._noisy(truth.local_force, cfg.force_std), 0.0, 1.0
            ).astype(np.float32),
            contact=np.asarray([float(truth.contact)], dtype=np.float32),
            grasp=np.asarray([float(truth.grasp)], dtype=np.float32),
            object_estimate=object_estimate,
            task_goal=ego_relative_pose(truth.task_goal_world, truth.ego_pose_world),
            private_event_cue=(
                np.asarray(truth.private_event_cue, dtype=np.float32)
                if truth.private_event_valid
                else np.zeros(3, dtype=np.float32)
            ),
            private_event_valid=np.asarray([float(truth.private_event_valid)], dtype=np.float32),
            private_event_age=np.asarray([float(truth.private_event_age)], dtype=np.float32),
            next_gate_context=np.asarray(truth.next_gate_context, dtype=np.float32),
        )
        packet.validate(self.spec)
        return packet

    def _validate_truth(self, truth: PrivilegedAgentState) -> None:
        _require_shape(truth.ego_pose_world, (3,), "ego_pose_world")
        _require_shape(truth.object_pose_world, (3,), "object_pose_world")
        _require_shape(truth.task_goal_world, (3,), "task_goal_world")
        _require_shape(truth.base_twist, (3,), "base_twist")
        _require_shape(truth.joint_position, (self.spec.joint_dim,), "joint_position")
        _require_shape(truth.joint_velocity, (self.spec.joint_dim,), "joint_velocity")
        _require_shape(truth.joint_torque, (self.spec.joint_dim,), "joint_torque")
        _require_shape(truth.local_force, (self.spec.force_dim,), "local_force")
        _require_shape(truth.private_event_cue, (3,), "private_event_cue")
        _require_shape(truth.next_gate_context, (3,), "next_gate_context")
        if truth.private_event_age < 0.0:
            raise ValueError("private_event_age must be non-negative")

    def _noisy(self, value: np.ndarray, std: float) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        if value.size == 0 or std <= 0.0:
            return value.copy()
        return (value + self.rng.normal(0.0, std, size=value.shape)).astype(np.float32)

    def _pose_measurement(
        self,
        agent_id: int,
        entity: str,
        target_world: np.ndarray,
        ego_world: np.ndarray,
        visible: bool,
        position_std: float,
        yaw_std: float,
    ) -> PoseEstimate:
        cache_key = (agent_id, entity)
        if visible:
            pose = ego_relative_pose(target_world, ego_world)
            pose[:2] += self.rng.normal(0.0, position_std, size=2).astype(np.float32)
            pose[2] = wrap_angle(pose[2] + float(self.rng.normal(0.0, yaw_std)))
            # Confidence describes measurement quality, not hidden ground-truth error.
            scale = max(1e-6, position_std + yaw_std)
            confidence = float(np.clip(np.exp(-scale), 0.0, 1.0))
            estimate = PoseEstimate(
                pose=pose.astype(np.float32),
                valid=np.ones(1, dtype=np.float32),
                confidence=np.asarray([confidence], dtype=np.float32),
                age=np.zeros(1, dtype=np.float32),
            )
            self._cache[cache_key] = _copy_estimate(estimate)
            return estimate

        cached = self._cache.get(cache_key)
        if cached is None:
            return PoseEstimate(
                pose=np.zeros(3, dtype=np.float32),
                valid=np.zeros(1, dtype=np.float32),
                confidence=np.zeros(1, dtype=np.float32),
                age=np.asarray([self.config.missing_age], dtype=np.float32),
            )

        age = float(cached.age[0]) + self.config.control_dt
        confidence = float(cached.confidence[0]) * np.exp(
            -self.config.control_dt / self.config.stale_confidence_tau
        )
        stale = PoseEstimate(
            pose=cached.pose.copy(),
            valid=np.zeros(1, dtype=np.float32),
            confidence=np.asarray([confidence], dtype=np.float32),
            age=np.asarray([age], dtype=np.float32),
        )
        self._cache[cache_key] = _copy_estimate(stale)
        return stale


def stack_packets(
    packets: Iterable[LocalObservationPacket], spec: LocalObservationSpec
) -> Dict[str, np.ndarray]:
    """Stack packets into schema-path arrays with leading time dimension."""

    packet_list = list(packets)
    if not packet_list:
        raise ValueError("cannot stack an empty packet sequence")
    mappings = []
    for packet in packet_list:
        packet.validate(spec)
        mappings.append(packet.as_mapping())
    return {
        name: np.stack([mapping[name] for mapping in mappings], axis=0).astype(np.float32)
        for name in spec.field_shapes()
    }


def flatten_observation_mapping(
    mapping: Mapping[str, np.ndarray], spec: LocalObservationSpec
) -> np.ndarray:
    """Flatten one or many observations according to the canonical field order."""

    arrays = [np.asarray(mapping[name], dtype=np.float32) for name in spec.field_shapes()]
    leading = arrays[0].shape[:-1]
    for name, value in zip(spec.field_shapes(), arrays):
        if value.shape[:-1] != leading:
            raise ValueError(f"inconsistent leading shape for {name}: {value.shape[:-1]} != {leading}")
    return np.concatenate(arrays, axis=-1).astype(np.float32)


class LocalHistoryBuffer:
    """Online counterpart of ``DecentralizedTransitionDataset`` history.

    Call ``append(o_0, previous_action=None)`` after reset.  After executing
    ``a_t`` and receiving ``o_(t+1)``, call
    ``append(o_(t+1), previous_action=a_t)``.  The returned tensors therefore
    have exactly the same ``(observation_tau, action_(tau-1))`` semantics as
    offline training windows.
    """

    def __init__(
        self,
        spec: LocalObservationSpec,
        *,
        action_dim: int = 4,
        history: int = 8,
    ) -> None:
        if action_dim <= 0 or history <= 0:
            raise ValueError("action_dim and history must be positive")
        self.spec = spec
        self.action_dim = int(action_dim)
        self.history = int(history)
        self.reset()

    @property
    def local_dim(self) -> int:
        return self.spec.model_observation_dim + self.action_dim

    def reset(self) -> None:
        self._model_observations: list[np.ndarray] = []
        self._previous_actions: list[np.ndarray] = []
        self._previous_action_valid: list[bool] = []
        self._object_observations: list[np.ndarray] = []
        self._object_valid: list[bool] = []
        self._object_confidence: list[float] = []
        self._object_age: list[float] = []

    def append(
        self,
        packet: LocalObservationPacket,
        *,
        previous_action: np.ndarray | None,
    ) -> None:
        packet.validate(self.spec)
        self.append_mapping(packet.as_mapping(), previous_action=previous_action)

    def append_mapping(
        self,
        mapping: Mapping[str, np.ndarray],
        *,
        previous_action: np.ndarray | None,
    ) -> None:
        missing = set(self.spec.field_shapes()) - set(mapping)
        if missing:
            raise KeyError(f"local observation mapping is missing fields: {sorted(missing)}")
        for name, shape in self.spec.field_shapes().items():
            value = np.asarray(mapping[name])
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")

        if previous_action is None:
            action = np.zeros(self.action_dim, dtype=np.float32)
            action_valid = False
        else:
            action = np.asarray(previous_action, dtype=np.float32)
            if action.shape != (self.action_dim,):
                raise ValueError(
                    f"previous_action must have shape ({self.action_dim},), got {action.shape}"
                )
            if not np.all(np.isfinite(action)):
                raise ValueError("previous_action contains non-finite values")
            action_valid = True

        model_observation = np.concatenate(
            [np.asarray(mapping[name], dtype=np.float32).reshape(-1) for name in self.spec.model_field_names()]
        )
        self._model_observations.append(model_observation)
        self._previous_actions.append(action.copy())
        self._previous_action_valid.append(action_valid)
        self._object_observations.append(
            np.asarray(mapping["estimates/object/pose"], dtype=np.float32).copy()
        )
        self._object_valid.append(
            bool(float(np.asarray(mapping["estimates/object/valid"]).reshape(-1)[0]) > 0.5)
        )
        self._object_confidence.append(
            float(np.asarray(mapping["estimates/object/confidence"]).reshape(-1)[0])
        )
        self._object_age.append(
            float(np.asarray(mapping["estimates/object/age"]).reshape(-1)[0])
        )

        # Ring-buffer truncation happens only after constructing a causal row.
        for values in (
            self._model_observations,
            self._previous_actions,
            self._previous_action_valid,
            self._object_observations,
            self._object_valid,
            self._object_confidence,
            self._object_age,
        ):
            if len(values) > self.history:
                del values[0]

    def as_arrays(self) -> Dict[str, np.ndarray]:
        if not self._model_observations:
            raise RuntimeError("history buffer is empty; append the reset observation first")
        count = len(self._model_observations)
        pad = self.history - count
        model_observation = np.zeros(
            (self.history, self.spec.model_observation_dim), dtype=np.float32
        )
        previous_action = np.zeros((self.history, self.action_dim), dtype=np.float32)
        object_observation = np.zeros((self.history, 3), dtype=np.float32)
        object_valid = np.zeros(self.history, dtype=np.bool_)
        object_confidence = np.zeros(self.history, dtype=np.float32)
        object_age = np.zeros(self.history, dtype=np.float32)
        history_mask = np.zeros(self.history, dtype=np.bool_)
        previous_action_valid = np.zeros(self.history, dtype=np.bool_)

        model_observation[pad:] = np.stack(self._model_observations)
        previous_action[pad:] = np.stack(self._previous_actions)
        object_observation[pad:] = np.stack(self._object_observations)
        object_valid[pad:] = np.asarray(self._object_valid, dtype=np.bool_)
        object_confidence[pad:] = np.asarray(self._object_confidence, dtype=np.float32)
        object_age[pad:] = np.asarray(self._object_age, dtype=np.float32)
        history_mask[pad:] = True
        previous_action_valid[pad:] = np.asarray(
            self._previous_action_valid, dtype=np.bool_
        )
        local_history = np.concatenate(
            [model_observation, previous_action], axis=-1
        ).astype(np.float32)
        return {
            "model_observation_history": model_observation,
            "prev_action_history": previous_action,
            "local_history": local_history,
            "model_history": local_history,
            "history_mask": history_mask,
            "history_valid_mask": history_mask.copy(),
            "padding_mask": ~history_mask,
            "prev_action_valid_mask": previous_action_valid,
            "object_observation_history": object_observation,
            "object_valid_history": object_valid,
            "object_confidence_history": object_confidence,
            "object_age_history": object_age,
        }

    def as_torch(
        self,
        *,
        device: str | None = None,
        add_batch_dimension: bool = True,
    ) -> Dict[str, "torch.Tensor"]:
        """Return model-ready tensors without importing torch at module load."""

        import torch

        result: Dict[str, torch.Tensor] = {}
        for name, value in self.as_arrays().items():
            tensor = torch.from_numpy(value)
            if add_batch_dimension:
                tensor = tensor.unsqueeze(0)
            if device is not None:
                tensor = tensor.to(device)
            result[name] = tensor
        return result


def _copy_estimate(value: PoseEstimate) -> PoseEstimate:
    return PoseEstimate(
        pose=value.pose.copy(),
        valid=value.valid.copy(),
        confidence=value.confidence.copy(),
        age=value.age.copy(),
    )


def _float32(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _require_shape(value: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
