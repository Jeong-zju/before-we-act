from __future__ import annotations

import copy

import numpy as np
import pytest

from deployment.duo_act.action_target import (
    ACTION_TARGET_CONTRACT_ID,
    ACTION_TARGET_CONTRACT_SHA256,
    CONTROLLER_JOINT_HIGH,
    CONTROLLER_JOINT_LOW,
    RCS_API_JOINT_HIGH,
    RCS_API_JOINT_LOW,
    action_target_contract,
    canonicalize_controller_action,
    canonicalize_controller_action_with_audit,
    validate_action_target_contract,
    validate_controller_action,
)


def test_controller_equivalent_clip_and_binary_gripper_without_mutating_input():
    raw = np.zeros((2, 2, 8), dtype=np.float32)
    raw[..., :7] = (CONTROLLER_JOINT_LOW + CONTROLLER_JOINT_HIGH) / 2
    raw[0, 0, :7] = CONTROLLER_JOINT_LOW - 1.0
    raw[0, 1, :7] = CONTROLLER_JOINT_HIGH + 1.0
    raw[..., 7] = np.asarray([[0.49, 0.5], [0.9, 0.0]], dtype=np.float32)
    before = raw.copy()

    canonical, audit = canonicalize_controller_action_with_audit(raw)

    np.testing.assert_array_equal(raw, before)
    np.testing.assert_array_equal(canonical[0, 0, :7], CONTROLLER_JOINT_LOW)
    np.testing.assert_array_equal(canonical[0, 1, :7], CONTROLLER_JOINT_HIGH)
    np.testing.assert_array_equal(canonical[..., 7], [[0, 1], [1, 0]])
    validate_controller_action(canonical)
    assert audit["out_of_controller_range_entries"] == 14
    assert audit["changed_gripper_values"] == 3
    assert audit["rcs_api_limits_used_for_canonicalization"] is False
    assert audit["by_arm"]["left"]["out_of_controller_range_by_joint"] == [1] * 7


def test_values_outside_narrow_rcs_box_but_inside_xml_ctrlrange_are_preserved():
    # These values are deliberately outside the API safety Box but physically
    # representable by the pinned MuJoCo position actuators.
    raw = np.array(
        [
            [2.5, -1.6, -2.7, -0.2, 2.6, 0.6, 2.8, 0.0],
            [-2.5, 1.6, 2.7, -2.9, -2.6, 4.4, -2.8, 1.0],
        ],
        dtype=np.float32,
    )
    assert np.any(raw[:, :7] < RCS_API_JOINT_LOW) or np.any(
        raw[:, :7] > RCS_API_JOINT_HIGH
    )
    assert np.all(raw[:, :7] >= CONTROLLER_JOINT_LOW)
    assert np.all(raw[:, :7] <= CONTROLLER_JOINT_HIGH)
    canonical = canonicalize_controller_action(raw)
    np.testing.assert_array_equal(canonical, raw)
    validate_controller_action(canonical)


def test_contract_is_pinned_and_tampering_is_rejected():
    contract = action_target_contract()
    assert contract["id"] == ACTION_TARGET_CONTRACT_ID
    assert contract["sha256"] == ACTION_TARGET_CONTRACT_SHA256
    validate_action_target_contract(contract)

    tampered = copy.deepcopy(contract)
    tampered["controller"]["joint_low"][0] -= 0.01
    with pytest.raises(ValueError, match="contract"):
        validate_action_target_contract(tampered)

    missing_hash = copy.deepcopy(contract)
    del missing_hash["sha256"]
    with pytest.raises(ValueError):
        validate_action_target_contract(missing_hash)


def test_periodic_wrap_and_next_qpos_substitution_are_not_performed():
    raw = np.array(
        [[
            CONTROLLER_JOINT_HIGH[0] + 0.01,
            CONTROLLER_JOINT_LOW[1] - 0.01,
            CONTROLLER_JOINT_HIGH[2] + 0.01,
            -0.2,  # outside RCS high, inside XML range
            0.0,
            0.6,  # outside RCS low, inside XML range
            CONTROLLER_JOINT_LOW[6] - 0.01,
            0.3,
        ]], dtype=np.float32
    )
    canonical = canonicalize_controller_action(raw)
    # Only the XML-overflow values and gripper threshold may change.  In
    # particular, no +/-2pi wrapping and no replacement by a hypothetical
    # next-qpos is allowed.
    np.testing.assert_allclose(canonical[0, 3], raw[0, 3])
    np.testing.assert_allclose(canonical[0, 5], raw[0, 5])
    np.testing.assert_allclose(canonical[0, 0], CONTROLLER_JOINT_HIGH[0])
    np.testing.assert_allclose(canonical[0, 1], CONTROLLER_JOINT_LOW[1])
    np.testing.assert_allclose(canonical[0, 2], CONTROLLER_JOINT_HIGH[2])
    np.testing.assert_allclose(canonical[0, 6], CONTROLLER_JOINT_LOW[6])
    assert canonical[0, 7] == 0.0


def test_nonfinite_and_noncanonical_targets_fail_closed():
    with pytest.raises(ValueError, match="non-finite"):
        canonicalize_controller_action(np.array([[np.nan] * 8], dtype=np.float32))
    with pytest.raises(ValueError, match="gripper"):
        joints = ((CONTROLLER_JOINT_LOW + CONTROLLER_JOINT_HIGH) / 2).tolist()
        validate_controller_action(np.array([joints + [0.25]], dtype=np.float32))
    with pytest.raises(ValueError, match="ctrlrange"):
        validate_controller_action(
            np.array([[float(CONTROLLER_JOINT_HIGH[0]) + 0.1] + [0.0] * 7], dtype=np.float32)
        )
