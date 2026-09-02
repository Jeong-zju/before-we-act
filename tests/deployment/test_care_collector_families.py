"""Every benchmark collector must be able to reach the behavior family.

The wiring differs per benchmark -- BiCoord acts on 7 dimensions with its own
native transforms, DuoBench works in anchor-relative encoded coordinates -- so
these tests check the properties that must hold in all three rather than the
mechanics of any one.
"""
from __future__ import annotations

import numpy as np
import pytest

from before_we_act.care_behavior_candidates import (
    CANDIDATE_NAMES,
    BehaviorCandidateConfig,
)


def _reference(horizon: int, action_dim: int) -> np.ndarray:
    steps = np.arange(horizon, dtype=np.float32)[:, None]
    plan = np.zeros((horizon, action_dim), dtype=np.float32)
    plan[:, : action_dim - 1] = 0.005 * steps
    plan[:, action_dim - 1] = (steps[:, 0] >= 5).astype(np.float32)
    return plan


def _duo_reference(horizon: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """An *encoded* chunk whose decoded pose stays inside the Panda bounds.

    DuoBench chunks are anchor-relative: joints are offsets from the proposal
    pose, the gripper is absolute. Several Panda joints exclude zero, so the
    anchor is each joint's midpoint and the chunk is a small offset from it --
    otherwise canonicalization clips the plan and a round-trip check fails for
    the wrong reason.
    """
    from deployment.duo_act.action_target import (
        CONTROLLER_JOINT_HIGH,
        CONTROLLER_JOINT_LOW,
    )

    low = np.asarray(CONTROLLER_JOINT_LOW, dtype=np.float32)
    high = np.asarray(CONTROLLER_JOINT_HIGH, dtype=np.float32)
    anchor = (low + high) / 2.0
    span = np.minimum(0.2, (high - low) / 8.0)
    steps = np.arange(horizon, dtype=np.float32)[:, None]

    encoded = np.zeros((horizon, 8), dtype=np.float32)
    encoded[:, :7] = (steps / max(1, horizon - 1)) * span[None]
    encoded[:, 7] = (steps[:, 0] >= 5).astype(np.float32)
    qpos = np.zeros(8, dtype=np.float32)
    qpos[:7] = anchor
    return encoded, qpos


class TestBiCoord:
    def test_fixed_family_keeps_the_native_seven_dimensional_transforms(self) -> None:
        """The archived corpus must stay reproducible byte-for-byte."""
        from deployment.bicoord_care.branch_collection import _behavior_config

        assert _behavior_config("fixed", 1) is None

    def test_behavior_config_matches_the_native_action_space(self) -> None:
        from deployment.bicoord_care.branch_collection import _behavior_config
        from deployment.bicoord_care.config import ACTION_DIM, ACTION_HORIZON

        config = _behavior_config("behavior", 8)
        assert config is not None
        assert config.action_dim == ACTION_DIM == 7
        assert config.action_horizon == ACTION_HORIZON
        assert config.joints == 6 and config.gripper == 6

    def test_behavior_magnitudes_scale_with_the_commitment_window(self) -> None:
        from deployment.bicoord_care.branch_collection import _behavior_config

        assert _behavior_config("behavior", 8).wait_steps == 4
        assert _behavior_config("behavior", 16).wait_steps == 8

    def test_a_window_too_short_for_a_behavior_is_refused(self) -> None:
        from deployment.bicoord_care.branch_collection import _behavior_config

        with pytest.raises(ValueError, match="at least four"):
            _behavior_config("behavior", 1)


class TestDuoBench:
    def test_default_config_preserves_the_archived_one_step_protocol(self) -> None:
        from deployment.duo_care.branch_collection_v2 import KernelConfig

        config = KernelConfig()
        assert config.candidate_family == "fixed"
        assert config.intervention_steps == 1
        assert config.behavior_config() is None

    def test_behavior_config_matches_the_duo_action_space(self) -> None:
        from deployment.duo_care.branch_collection_v2 import KernelConfig

        config = KernelConfig(candidate_family="behavior", intervention_steps=8)
        behavior = config.behavior_config()
        assert behavior is not None
        assert behavior.action_dim == 8 and behavior.action_horizon == 100
        assert behavior.intervention_steps == 8

    def test_a_window_too_short_for_a_behavior_is_refused(self) -> None:
        from deployment.duo_care.branch_collection_v2 import KernelConfig

        with pytest.raises(ValueError, match="at least four"):
            KernelConfig(candidate_family="behavior", intervention_steps=1)

    def test_commitment_cannot_exceed_the_branch_horizon(self) -> None:
        from deployment.duo_care.branch_collection_v2 import KernelConfig

        with pytest.raises(ValueError, match="inside the branch horizon"):
            KernelConfig(candidate_family="behavior", intervention_steps=999)

    def test_encoded_family_round_trips_through_the_controller_boundary(self) -> None:
        from deployment.duo_act.action_target import (
            CONTROLLER_JOINT_HIGH,
            CONTROLLER_JOINT_LOW,
        )
        from deployment.duo_care.candidates import behavior_candidate_family

        config = BehaviorCandidateConfig(action_horizon=100, action_dim=8)
        reference, qpos = _duo_reference()

        stack, audits = behavior_candidate_family(
            reference,
            reference,
            qpos,
            joint_low=CONTROLLER_JOINT_LOW,
            joint_high=CONTROLLER_JOINT_HIGH,
            config=config,
        )

        assert stack.shape == (len(CANDIDATE_NAMES), 100, 8)
        assert all(row.valid for row in audits)
        np.testing.assert_allclose(stack[0], reference, atol=1e-6)

    def test_each_behavior_leaves_its_own_signature_on_the_arm(self) -> None:
        """Grip-timing candidates move only the gripper; wait and retreat move joints."""
        from deployment.duo_act.action_target import (
            CONTROLLER_JOINT_HIGH,
            CONTROLLER_JOINT_LOW,
        )
        from deployment.duo_care.candidates import behavior_candidate_family

        config = BehaviorCandidateConfig(action_horizon=100, action_dim=8)
        reference, qpos = _duo_reference()
        stack, _audits = behavior_candidate_family(
            reference,
            reference,
            qpos,
            joint_low=CONTROLLER_JOINT_LOW,
            joint_high=CONTROLLER_JOINT_HIGH,
            config=config,
        )

        window = config.intervention_steps
        joints = {
            CANDIDATE_NAMES[k]: float(
                np.abs(stack[k, :window, :7] - stack[0, :window, :7]).max()
            )
            for k in range(len(CANDIDATE_NAMES))
        }

        assert joints["grip_delay"] == pytest.approx(0.0, abs=1e-6)
        assert joints["grip_advance"] == pytest.approx(0.0, abs=1e-6)
        assert joints["retreat"] > joints["wait"] > 0.0
        assert joints["slow"] > 0.0

        grippers = {
            CANDIDATE_NAMES[k]: float(
                np.abs(stack[k, :window, 7] - stack[0, :window, 7]).max()
            )
            for k in (3, 4)
        }
        assert min(grippers.values()) > 0.0
