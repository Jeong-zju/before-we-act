"""The DuoBench controller-equivalent absolute-action contract.

The released DuoBench demonstrations contain the output of the RCS Cartesian
to joint Pinocchio converter.  That converter does not solve a constrained IK
problem: a successful solution may lie outside the conservative joint limits
exposed by RCS' Gym ``Box``.  In the formal DuoBench evaluator the action is
sent to a MuJoCo position actuator with ``inheritrange=1``.  MuJoCo therefore
enforces the actuator ``ctrlrange`` before integrating the simulation.

This module makes that last, observable controller operation explicit.  It is
intentionally dependency-light so the exact same function can be used while
preparing data, constructing CARE candidates, and emitting a closed-loop
command.  No periodic-angle unwrapping, next-qpos substitution, or silent
``RCS joint_limits`` clipping is performed here.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np


JOINT_DIM = 7
ACTION_DIM = 8

# These are the ctrlrange values obtained from the pinned FR3 MJCF used by
# DuoBench/RCS.  The seven entries are in the converter's canonical order
# fr3_joint1 ... fr3_joint7.  Keep these separate from RCS.ROBOTS[FR3].joint_limits:
# the latter is an API-level safety Box and is deliberately narrower for some
# joints (notably j1, j2, j3, j5, j6 and j7).
CONTROLLER_JOINT_LOW = np.asarray(
    [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159],
    dtype=np.float32,
)
CONTROLLER_JOINT_HIGH = np.asarray(
    [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159],
    dtype=np.float32,
)

# Kept in the contract as evidence of the distinction above.  These values
# are *not* used for canonicalization.
RCS_API_JOINT_LOW = np.asarray(
    [-2.3093, -1.5133, -2.4937, -2.7478, -2.4800, 0.8521, -2.6895],
    dtype=np.float32,
)
RCS_API_JOINT_HIGH = np.asarray(
    [2.3093, 1.5133, 2.4937, -0.4461, 2.4800, 4.2094, 2.6895],
    dtype=np.float32,
)

CONTROLLER_XML_SHA256 = "e8923c05a357b1239378aa262e9d391b4fb9f49f8b0e1fe771e43cad98ef4112"
ACTION_TARGET_CONTRACT_ID = "duobench_controller_equivalent_absolute_v1"
ACTION_TARGET_CONTRACT_SCHEMA = "before-we-act.duobench.action-target-contract/1"

_CONTRACT_PAYLOAD: dict[str, Any] = {
    "schema": ACTION_TARGET_CONTRACT_SCHEMA,
    "id": ACTION_TARGET_CONTRACT_ID,
    "benchmark": "DuoBench",
    "robot_type": "FR3",
    "joint_names": [
        "fr3_joint1",
        "fr3_joint2",
        "fr3_joint3",
        "fr3_joint4",
        "fr3_joint5",
        "fr3_joint6",
        "fr3_joint7",
    ],
    "joint_order": "RCS_FR3_converter_order",
    "controller": {
        "backend": "mujoco_position_actuator",
        "range_source": "pinned_fr3_mjcf_actuator_ctrlrange",
        "xml_sha256": CONTROLLER_XML_SHA256,
        "inheritrange": 1,
        "joint_low": CONTROLLER_JOINT_LOW.tolist(),
        "joint_high": CONTROLLER_JOINT_HIGH.tolist(),
    },
    "api_limits_reference_only": {
        "joint_low": RCS_API_JOINT_LOW.tolist(),
        "joint_high": RCS_API_JOINT_HIGH.tolist(),
    },
    "gripper": {
        "encoding": "binary",
        "threshold": 0.5,
        "closed": 0.0,
        "open": 1.0,
        "source_semantics": "RCS_GripperWrapper_binary",
    },
    "source": "RCS_Pin_inverse_successful_joint_output",
    "canonicalization": {
        "joints": "clip_to_controller_ctrlrange",
        "gripper": "threshold_at_0.5_to_binary",
        "periodic_angle_unwrap": False,
        "next_qpos_substitution": False,
        "silent_clipping": False,
    },
    "action_encoding": "absolute_joint7_binary_gripper1",
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


ACTION_TARGET_CONTRACT_SHA256 = hashlib.sha256(_canonical_json(_CONTRACT_PAYLOAD)).hexdigest()


def action_target_contract() -> dict[str, Any]:
    """Return a JSON-safe copy of the immutable contract description."""

    result = json.loads(json.dumps(_CONTRACT_PAYLOAD))
    result["sha256"] = ACTION_TARGET_CONTRACT_SHA256
    return result


def controller_joint_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Return copies of the pinned MuJoCo controller bounds."""

    return CONTROLLER_JOINT_LOW.copy(), CONTROLLER_JOINT_HIGH.copy()


def _as_action_array(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim < 1 or result.shape[-1] != ACTION_DIM:
        raise ValueError(f"{name} must have a final dimension of {ACTION_DIM}, got {result.shape}")
    if result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def summarize_action_canonicalization(raw: np.ndarray, canonical: np.ndarray) -> dict[str, Any]:
    """Build a deterministic, auditable summary for one action tensor.

    Both arrays must have shape ``(..., 8)`` and represent the same rows.  The
    summary deliberately reports the pre-canonicalization extrema and every
    reason a value changed; it does not retain raw trajectories in JSON.
    """

    source = _as_action_array(raw, "raw action")
    target = _as_action_array(canonical, "canonical action")
    if source.shape != target.shape:
        raise ValueError(f"raw/canonical action shapes differ: {source.shape}/{target.shape}")

    flat = source.reshape(-1, ACTION_DIM)
    out = target.reshape(-1, ACTION_DIM)
    changed = ~np.equal(flat, out)
    below = flat[:, :JOINT_DIM] < CONTROLLER_JOINT_LOW[None, :]
    above = flat[:, :JOINT_DIM] > CONTROLLER_JOINT_HIGH[None, :]
    rcs_below = flat[:, :JOINT_DIM] < RCS_API_JOINT_LOW[None, :]
    rcs_above = flat[:, :JOINT_DIM] > RCS_API_JOINT_HIGH[None, :]
    nonbinary_gripper = ~np.isin(flat[:, JOINT_DIM], (0.0, 1.0))
    delta = np.abs(out.astype(np.float64) - flat.astype(np.float64))
    reasons: dict[str, int] = {
        "joint_below_controller_low": int(below.sum()),
        "joint_above_controller_high": int(above.sum()),
        "gripper_nonbinary_thresholded": int(nonbinary_gripper.sum()),
    }
    changed_joint = changed[:, :JOINT_DIM]
    changed_gripper = changed[:, JOINT_DIM]
    result: dict[str, Any] = {
        "raw_values": int(flat.size),
        "rows": int(len(flat)),
        "raw_min": flat.min(axis=0).astype(float).tolist(),
        "raw_max": flat.max(axis=0).astype(float).tolist(),
        "canonical_min": out.min(axis=0).astype(float).tolist(),
        "canonical_max": out.max(axis=0).astype(float).tolist(),
        "changed_values": int(changed.sum()),
        "changed_joint_values": int(changed_joint.sum()),
        "changed_gripper_values": int(changed_gripper.sum()),
        "max_abs_delta": float(delta.max(initial=0.0)),
        "max_abs_joint_delta": float(delta[:, :JOINT_DIM].max(initial=0.0)),
        "out_of_controller_range_entries": int((below | above).sum()),
        "out_of_controller_range_by_joint": (below | above).sum(axis=0).astype(int).tolist(),
        "outside_rcs_api_limits_diagnostic_entries": int((rcs_below | rcs_above).sum()),
        "outside_rcs_api_limits_diagnostic_by_joint": (
            rcs_below | rcs_above
        ).sum(axis=0).astype(int).tolist(),
        "rcs_api_limits_used_for_canonicalization": False,
        "nonbinary_gripper_entries": int(nonbinary_gripper.sum()),
        "reasons": reasons,
    }
    if source.ndim >= 3 and source.shape[-2] == 2:
        paired_raw = source.reshape(-1, 2, ACTION_DIM)
        paired_out = target.reshape(-1, 2, ACTION_DIM)
        by_arm: dict[str, Any] = {}
        for arm, name in enumerate(("left", "right")):
            arm_raw = paired_raw[:, arm]
            arm_out = paired_out[:, arm]
            arm_below = arm_raw[:, :JOINT_DIM] < CONTROLLER_JOINT_LOW
            arm_above = arm_raw[:, :JOINT_DIM] > CONTROLLER_JOINT_HIGH
            arm_delta = np.abs(
                arm_out.astype(np.float64) - arm_raw.astype(np.float64)
            )
            by_arm[name] = {
                "out_of_controller_range_by_joint": (
                    arm_below | arm_above
                ).sum(axis=0).astype(int).tolist(),
                "changed_joint_values_by_joint": np.not_equal(
                    arm_raw[:, :JOINT_DIM], arm_out[:, :JOINT_DIM]
                ).sum(axis=0).astype(int).tolist(),
                "max_abs_joint_delta_by_joint": arm_delta[
                    :, :JOINT_DIM
                ].max(axis=0, initial=0.0).astype(float).tolist(),
            }
        result["by_arm"] = by_arm
    return result


def canonicalize_controller_action_with_audit(value: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Canonicalize raw converted actions and return an audit summary.

    The operation is exactly the one applied by a MuJoCo position actuator at
    the controller boundary: finite joint commands are saturated to
    ``ctrlrange`` and gripper commands are converted to binary.  Inputs are
    never modified in place.
    """

    raw = _as_action_array(value, "raw action")
    canonical = raw.copy()
    canonical[..., :JOINT_DIM] = np.clip(
        canonical[..., :JOINT_DIM], CONTROLLER_JOINT_LOW, CONTROLLER_JOINT_HIGH
    )
    canonical[..., JOINT_DIM] = (
        canonical[..., JOINT_DIM] >= 0.5
    ).astype(np.float32)
    return canonical, summarize_action_canonicalization(raw, canonical)


def canonicalize_controller_action(value: np.ndarray) -> np.ndarray:
    """Return only the controller-equivalent action tensor."""

    return canonicalize_controller_action_with_audit(value)[0]


# Short aliases make the contract convenient at deployment boundaries while
# retaining the explicit name above for provenance code and documentation.
canonicalize_action = canonicalize_controller_action
canonicalize_action_with_audit = canonicalize_controller_action_with_audit


def validate_controller_action(value: np.ndarray, *, atol: float = 1e-6) -> None:
    """Raise if ``value`` is not already controller-equivalent and binary."""

    action = _as_action_array(value, "controller action")
    if np.any(action[..., :JOINT_DIM] < CONTROLLER_JOINT_LOW - atol) or np.any(
        action[..., :JOINT_DIM] > CONTROLLER_JOINT_HIGH + atol
    ):
        raise ValueError("controller action exceeds pinned MuJoCo ctrlrange")
    if not np.isin(action[..., JOINT_DIM], (0.0, 1.0)).all():
        raise ValueError("controller action gripper is not binary")


def validate_action_target_contract(value: Mapping[str, Any]) -> None:
    """Validate a serialized contract identity without accepting substitutions."""

    if value.get("id") != ACTION_TARGET_CONTRACT_ID:
        raise ValueError(f"unexpected action target contract id: {value.get('id')!r}")
    if value.get("sha256") != ACTION_TARGET_CONTRACT_SHA256:
        raise ValueError("unexpected action target contract sha256")
    candidate = dict(value)
    candidate.pop("sha256", None)
    if candidate != _CONTRACT_PAYLOAD:
        raise ValueError("serialized action target contract differs from the pinned contract")


__all__ = [
    "ACTION_DIM",
    "ACTION_TARGET_CONTRACT_ID",
    "ACTION_TARGET_CONTRACT_SCHEMA",
    "ACTION_TARGET_CONTRACT_SHA256",
    "CONTROLLER_JOINT_HIGH",
    "CONTROLLER_JOINT_LOW",
    "CONTROLLER_XML_SHA256",
    "JOINT_DIM",
    "RCS_API_JOINT_HIGH",
    "RCS_API_JOINT_LOW",
    "action_target_contract",
    "canonicalize_action",
    "canonicalize_action_with_audit",
    "canonicalize_controller_action",
    "canonicalize_controller_action_with_audit",
    "controller_joint_bounds",
    "summarize_action_canonicalization",
    "validate_action_target_contract",
    "validate_controller_action",
]
