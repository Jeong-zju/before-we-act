"""HDF5 schema for decentralized FE-PC-WAM episodes.

The schema encodes transitions without overloading a single row:

``deployable_observation[t] --action[t]--> deployable_observation[t + 1]``

Consequently, every episode with ``T`` actions stores ``T + 1`` observations.
Simulator-only values live below ``/privileged`` and are never mixed into the
deployable observation group.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import h5py
import numpy as np

from data.local_observation import (
    LocalObservationSpec,
    flatten_observation_mapping,
)


SCHEMA_VERSION = "fe_pc_wam/private_gates_v1"
STRICT_LOCAL_CONTACT_SEMANTICS = "per_agent_mujoco_geom_contact"
LEGACY_CONTACT_SEMANTICS = "shared_global_contact_proxy/legacy"
STRICT_LOCAL_FORCE_SEMANTICS = (
    "per_agent_mujoco_contact_force_normalized_clip_0_1"
)
LEGACY_FORCE_SEMANTICS = "scenario_global_force_proxy/legacy"
STRICT_LOCAL_SENSOR_PROVENANCE = "two_robot_carry_per_agent_mujoco_sensors"
LOCAL_FORCE_UNITS = "normalized_0_1"
TRANSITION_SEMANTICS = "observation[t], action[t], observation[t+1]"
DEPLOYABLE_POLICY = (
    "ego local sensors/perception + previous ego action + selectively received plan messages only"
)
EGO_FRAME = "SE2 ego frame: x forward/right follows simulator body convention; yaw wrapped [-pi,pi)"
ACTION_FRAME = "sender ego/base frame; each agent stores and transmits only its own command"

PHASE_TO_ID = {
    "approach": 0,
    "align": 1,
    "grasp": 2,
    "carry_to_passage": 3,
    "passage": 4,
    "carry_to_goal": 5,
    "release": 6,
    "done": 7,
    "failure": 8,
}

FAILURE_TO_ID = {
    "none": 0,
    "timeout": 1,
    "force_violation": 2,
    "object_out_of_bounds": 3,
    "robot_out_of_bounds": 4,
    "object_dropped": 5,
    "robot_too_far": 6,
    "desync_in_passage": 7,
    "object_yaw_too_large": 8,
    "private_event_mismatch": 9,
    "unknown": 99,
}


def phase_to_id(phase: str) -> int:
    """Map an environment phase label to its stored integer identifier."""

    return PHASE_TO_ID.get(phase, PHASE_TO_ID["failure"])


def failure_to_id(reason: str) -> int:
    """Map an environment failure label to its stored integer identifier."""

    return FAILURE_TO_ID.get(reason, FAILURE_TO_ID["unknown"])


@dataclass
class Episode:
    """In-memory representation accepted by :func:`save_episode`."""

    local_observations: Dict[int, Dict[str, np.ndarray]]
    actions: Dict[int, np.ndarray]
    privileged_observations: Dict[str, np.ndarray]
    privileged_transitions: Dict[str, np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)
    research_v2_branch_groups: list[Any] = field(default_factory=list)


def save_episode(
    path: str | Path,
    episode: Episode,
    spec: LocalObservationSpec,
    *,
    compression: str | None = "gzip",
) -> None:
    """Validate and atomically describe one  episode in HDF5.

    This function is intentionally strict: deployable groups must contain
    exactly the fields in :class:`LocalObservationSpec`.  Extra simulator truth
    therefore cannot silently become policy input.
    """

    summary = validate_episode(episode, spec)
    _validate_strict_sensor_provenance(episode.metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["transition_semantics"] = TRANSITION_SEMANTICS
        f.attrs["deployable_input_policy"] = DEPLOYABLE_POLICY
        f.attrs["observation_coordinate_frame"] = EGO_FRAME
        f.attrs["action_coordinate_frame"] = ACTION_FRAME
        f.attrs["local_contact_semantics"] = episode.metadata[
            "local_contact_semantics"
        ]
        f.attrs["local_force_semantics"] = episode.metadata[
            "local_force_semantics"
        ]
        f.attrs["local_force_units"] = episode.metadata["local_force_units"]
        f.attrs["local_force_scale_newtons"] = float(
            episode.metadata["local_force_scale_newtons"]
        )
        f.attrs["local_sensor_provenance"] = episode.metadata[
            "local_sensor_provenance"
        ]
        f.attrs["num_agents"] = summary["num_agents"]
        f.attrs["num_observations"] = summary["num_observations"]
        f.attrs["num_transitions"] = summary["num_transitions"]
        f.attrs["joint_dim"] = spec.joint_dim
        f.attrs["force_dim"] = spec.force_dim
        f.attrs["base_twist_dim"] = spec.base_twist_dim

        metadata = f.create_group("metadata")
        for key, value in episode.metadata.items():
            metadata.attrs[str(key)] = _attribute_value(value)

        schema = f.create_group("schema")
        local_schema = schema.create_group("local_observation")
        local_schema.attrs["flat_dim"] = spec.flat_dim
        local_schema.attrs["model_observation_dim"] = spec.model_observation_dim
        local_schema.attrs["field_order_json"] = json.dumps(list(spec.field_shapes()))
        local_schema.attrs["model_field_order_json"] = json.dumps(spec.model_field_names())
        local_schema.attrs["field_shapes_json"] = json.dumps(
            {name: list(shape) for name, shape in spec.field_shapes().items()}, sort_keys=True
        )
        local_schema.attrs["feature_names_json"] = json.dumps(spec.feature_names())
        local_schema.attrs["model_feature_names_json"] = json.dumps(
            spec.model_feature_names()
        )
        local_schema.attrs["field_sources_json"] = json.dumps(
            {
                "self/base_twist": "onboard base velocity estimate",
                "self/joint_position": "joint encoders",
                "self/joint_velocity": "joint encoders/differentiation",
                "self/joint_torque": "joint torque sensors",
                "local/force": "local force estimate derived from onboard sensing",
                "local/contact": "local tactile/force contact estimate",
                "local/grasp": "local gripper state",
                "estimates/object/*": "multi-view RGB estimator; synthetic corruption in simulation",
                "task/goal": "task command transformed to ego frame",
                "task/private_event_*": "agent-local task cue; invalid cues are zeroed",
                "task/next_gate_context": "public local context for the next decision gate",
            },
            sort_keys=True,
        )
        local_schema.attrs["explicit_teammate_state_allowed"] = False
        local_schema.attrs["teammate_belief_source"] = (
            "local temporal coupling and selectively received plan latent messages"
        )

        # RGB is a real deployable sensor but  does not yet train a perception
        # encoder.  Reserve a stable location and calibration metadata without
        # adding pixels to the current low-dimensional policy packet.
        rgb_schema = schema.create_group("rgb")
        rgb_schema.attrs["enabled_in_current_model"] = False
        rgb_schema.attrs["storage_root"] = "/raw_sensors/agent_{agent_id}/rgb/{camera_name}"
        rgb_schema.attrs["timestamp_path"] = "/raw_sensors/agent_{agent_id}/rgb_timestamps"
        rgb_schema.attrs["camera_names_json"] = json.dumps(
            episode.metadata.get("rgb_camera_names", [])
        )
        rgb_schema.attrs["calibration_reference"] = str(
            episode.metadata.get("rgb_calibration_reference", "")
        )

        observations = f.create_group("observations")
        actions_group = f.create_group("transitions").create_group("actions")
        raw_sensors = f.create_group("raw_sensors")
        for agent_id in sorted(episode.local_observations):
            agent_name = f"agent_{agent_id}"
            deployable = observations.create_group(agent_name).create_group("deployable")
            for field_name in spec.field_shapes():
                _write_path(deployable, field_name, episode.local_observations[agent_id][field_name], compression)
            _write_array(actions_group, agent_name, episode.actions[agent_id], compression)

            rgb_group = raw_sensors.create_group(agent_name).create_group("rgb")
            rgb_group.attrs["available"] = False
            rgb_group.attrs["description"] = "reserved for synchronized multi-view RGB frames"

        privileged = f.create_group("privileged")
        privileged.attrs["policy_input_allowed"] = False
        privileged_obs = privileged.create_group("observations")
        privileged_tr = privileged.create_group("transitions")
        for name, value in episode.privileged_observations.items():
            _write_path(privileged_obs, name, value, compression)
        for name, value in episode.privileged_transitions.items():
            _write_path(privileged_tr, name, value, compression)


def validate_episode(episode: Episode, spec: LocalObservationSpec) -> Dict[str, int]:
    if not episode.local_observations:
        raise ValueError("local_observations must contain at least one agent")
    agent_ids = sorted(episode.local_observations)
    if agent_ids != list(range(len(agent_ids))):
        raise ValueError(f"agent ids must be contiguous from zero, got {agent_ids}")
    if sorted(episode.actions) != agent_ids:
        raise ValueError("actions and local_observations must contain the same agent ids")

    required_fields = list(spec.field_shapes())
    observation_count: int | None = None
    transition_count: int | None = None
    action_dim: int | None = None

    for agent_id in agent_ids:
        fields = episode.local_observations[agent_id]
        if set(fields) != set(required_fields):
            missing = sorted(set(required_fields) - set(fields))
            extra = sorted(set(fields) - set(required_fields))
            raise ValueError(
                f"agent {agent_id} deployable fields mismatch; missing={missing}, extra={extra}"
            )
        for name, shape in spec.field_shapes().items():
            value = np.asarray(fields[name])
            if value.ndim != 1 + len(shape) or tuple(value.shape[1:]) != shape:
                raise ValueError(
                    f"agent {agent_id} field {name} must have shape [N,{shape}], got {value.shape}"
                )
            if not np.all(np.isfinite(value)):
                raise ValueError(f"agent {agent_id} field {name} contains non-finite values")
            _validate_deployable_field_range(value, name, agent_id)
            if observation_count is None:
                observation_count = int(value.shape[0])
            elif value.shape[0] != observation_count:
                raise ValueError("all observation arrays must have the same leading length")

        private_valid = np.asarray(fields["task/private_event_valid"]).reshape(-1)
        private_cue = np.asarray(fields["task/private_event_cue"])
        if np.any(private_cue[private_valid < 0.5] != 0.0):
            raise ValueError(
                f"agent {agent_id} invalid private event cue must be zeroed"
            )

        action = np.asarray(episode.actions[agent_id])
        if action.ndim != 2 or action.shape[1] != 4:
            raise ValueError(
                f"agent {agent_id} actions must have  shape [T,4], got {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError(f"agent {agent_id} actions contain non-finite values")
        # World-frame planar components are individually clipped to [-1, 1]
        # before collection, then rotated into the sender frame.  Rotation can
        # make one ego component exceed 1 while preserving the <=sqrt(2) norm.
        planar_norm = np.linalg.norm(action[:, :2], axis=-1)
        if np.any(planar_norm > np.sqrt(2.0) + 1e-6):
            raise ValueError(
                f"agent {agent_id} normalized planar action norm exceeds sqrt(2)"
            )
        if np.any((action[:, 2:] < -1.0) | (action[:, 2:] > 1.0)):
            raise ValueError(f"agent {agent_id} wz/grip actions must lie in [-1, 1]")
        if transition_count is None:
            transition_count = int(action.shape[0])
            action_dim = int(action.shape[1])
        elif action.shape != (transition_count, action_dim):
            raise ValueError("all agents must use the same action shape")

    assert observation_count is not None and transition_count is not None
    if observation_count != transition_count + 1:
        raise ValueError(
            "strict transition semantics require N_observations == N_actions + 1; "
            f"got {observation_count} and {transition_count}"
        )

    _validate_timed_mapping(
        episode.privileged_observations,
        observation_count,
        "privileged_observations",
    )
    _validate_timed_mapping(
        episode.privileged_transitions,
        transition_count,
        "privileged_transitions",
    )

    return {
        "num_agents": len(agent_ids),
        "num_observations": observation_count,
        "num_transitions": transition_count,
        "action_dim": int(action_dim or 0),
    }


def _validate_strict_sensor_provenance(metadata: Mapping[str, Any]) -> None:
    required = {
        "local_contact_semantics": STRICT_LOCAL_CONTACT_SEMANTICS,
        "local_force_semantics": STRICT_LOCAL_FORCE_SEMANTICS,
        "local_force_units": LOCAL_FORCE_UNITS,
        "local_sensor_provenance": STRICT_LOCAL_SENSOR_PROVENANCE,
    }
    for name, expected in required.items():
        if metadata.get(name) != expected:
            raise ValueError(
                f"strict  episode metadata requires {name}={expected!r}"
            )
    if "local_force_scale_newtons" not in metadata:
        raise ValueError("strict  episode metadata requires local_force_scale_newtons")
    scale = float(metadata["local_force_scale_newtons"])
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("local_force_scale_newtons must be finite and positive")


def _validate_deployable_field_range(
    value: np.ndarray, name: str, agent_id: int
) -> None:
    if name == "local/force" and np.any((value < 0.0) | (value > 1.0)):
        raise ValueError(f"agent {agent_id} local/force must lie in [0, 1]")
    if name in {
        "local/contact",
        "local/grasp",
        "estimates/object/valid",
        "task/private_event_valid",
    } and not np.all(np.isin(value, (0.0, 1.0))):
        raise ValueError(f"agent {agent_id} field {name} must contain binary flags")
    if name == "estimates/object/confidence" and np.any(
        (value < 0.0) | (value > 1.0)
    ):
        raise ValueError(
            f"agent {agent_id} estimates/object/confidence must lie in [0, 1]"
        )
    if name == "estimates/object/age" and np.any(value < 0.0):
        raise ValueError(f"agent {agent_id} estimates/object/age cannot be negative")
    if name == "task/private_event_age" and np.any(value < 0.0):
        raise ValueError(f"agent {agent_id} task/private_event_age cannot be negative")
    if name == "task/private_event_cue" and np.any((value < 0.0) | (value > 1.0)):
        raise ValueError(f"agent {agent_id} task/private_event_cue must lie in [0, 1]")


def read_local_observations(
    file: h5py.File,
    agent_id: int,
    spec: LocalObservationSpec,
    selection=slice(None),
) -> Dict[str, np.ndarray]:
    """Read only one ego robot's deployable observation stream."""

    root = file[f"observations/agent_{agent_id}/deployable"]
    return {name: root[name][selection] for name in spec.field_shapes()}


def read_flat_local_observations(
    file: h5py.File,
    agent_id: int,
    spec: LocalObservationSpec,
    selection=slice(None),
) -> np.ndarray:
    return flatten_observation_mapping(
        read_local_observations(file, agent_id, spec, selection), spec
    )


def spec_from_hdf5(
    file: h5py.File, *, expected_schema_version: str = SCHEMA_VERSION
) -> LocalObservationSpec:
    if str(file.attrs.get("schema_version", "")) != expected_schema_version:
        raise ValueError(
            f"expected schema_version={expected_schema_version}, got {file.attrs.get('schema_version', '')}"
        )
    return LocalObservationSpec(
        joint_dim=int(file.attrs["joint_dim"]),
        force_dim=int(file.attrs["force_dim"]),
        base_twist_dim=int(file.attrs["base_twist_dim"]),
    )


def _validate_timed_mapping(mapping: Mapping[str, np.ndarray], length: int, name: str) -> None:
    if not mapping:
        raise ValueError(f"{name} must not be empty")
    for key, value in mapping.items():
        array = np.asarray(value)
        if array.ndim < 1 or array.shape[0] != length:
            raise ValueError(
                f"{name}/{key} must have leading length {length}, got {array.shape}"
            )
        if array.dtype.kind not in {"b", "i", "u", "f", "S", "U", "O"}:
            raise ValueError(f"unsupported dtype for {name}/{key}: {array.dtype}")
        if array.dtype.kind in {"f"} and not np.all(np.isfinite(array)):
            raise ValueError(f"{name}/{key} contains non-finite values")


def _write_path(
    root: h5py.Group,
    path: str,
    value: np.ndarray,
    compression: str | None,
) -> None:
    parts = path.strip("/").split("/")
    group = root
    for part in parts[:-1]:
        group = group.require_group(part)
    _write_array(group, parts[-1], value, compression)


def _write_array(
    group: h5py.Group,
    name: str,
    value: np.ndarray,
    compression: str | None,
) -> None:
    array = np.asarray(value)
    if array.dtype.kind in {"U", "O"}:
        dtype = h5py.string_dtype(encoding="utf-8")
        group.create_dataset(name, data=array.astype(object), dtype=dtype)
        return
    kwargs = {"compression": compression} if compression and array.size else {}
    group.create_dataset(name, data=array, **kwargs)


def _attribute_value(value: Any):
    if isinstance(value, (str, bytes, bool, int, float, np.integer, np.floating)):
        return value
    return json.dumps(value, sort_keys=True)
