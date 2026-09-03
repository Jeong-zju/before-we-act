"""Behavior-level CARE candidates must differ where they are actually executed.

The family this replaces perturbed only the first action of a 100-step chunk, so
under a one-step intervention the alternatives were nearly indistinguishable
from the nominal action and the realized advantage stayed around 1e-4. These
tests pin the properties that failure violated.
"""
from __future__ import annotations

import numpy as np
import pytest

from before_we_act.care_behavior_candidates import (
    CANDIDATE_COUNT,
    CANDIDATE_NAMES,
    BehaviorCandidateConfig,
    behavior_candidate_plan,
    behavior_candidate_set,
    validate_behavior_candidate,
)


ROBOFACTORY = BehaviorCandidateConfig(action_horizon=100, action_dim=8)
SEVEN_DOF = BehaviorCandidateConfig(action_horizon=100, action_dim=7)
CONFIGS = [ROBOFACTORY, SEVEN_DOF]
IDS = ["robofactory_mars_8d", "generic_7d"]


def _reference(config: BehaviorCandidateConfig, *, rate: float = 0.01) -> np.ndarray:
    """A steadily moving arm whose gripper closes midway through the window."""
    steps = np.arange(config.action_horizon, dtype=np.float32)[:, None]
    plan = np.zeros((config.action_horizon, config.action_dim), dtype=np.float32)
    plan[:, : config.joints] = rate * steps * np.arange(1, config.joints + 1, dtype=np.float32)
    plan[:, config.gripper] = (steps[:, 0] >= 5).astype(np.float32)
    return plan


def _pose(config: BehaviorCandidateConfig) -> np.ndarray:
    return np.zeros(config.joints, dtype=np.float32)


@pytest.mark.parametrize("config", CONFIGS, ids=IDS)
def test_reference_candidate_is_returned_element_for_element(config) -> None:
    reference = _reference(config)
    plan = behavior_candidate_plan(0, reference, _pose(config), 0.0, config)
    np.testing.assert_array_equal(plan, reference)


@pytest.mark.parametrize("config", CONFIGS, ids=IDS)
def test_every_alternative_differs_inside_the_executed_window(config) -> None:
    """The defect being fixed: alternatives that only moved the first action."""
    reference = _reference(config)
    window = config.intervention_steps
    family = behavior_candidate_set(reference, _pose(config), 0.0, config)

    assert family.shape == (CANDIDATE_COUNT, config.action_horizon, config.action_dim)
    for candidate in range(1, CANDIDATE_COUNT):
        executed = np.abs(family[candidate, :window] - reference[:window])
        assert executed.max() > 1e-3, (
            f"{CANDIDATE_NAMES[candidate]} is indistinguishable from the nominal "
            "action inside the executed commitment window"
        )
        # Differing on more than the first row is what separates a behavior from
        # a first-step nudge.
        assert int((executed.max(axis=1) > 1e-6).sum()) > 1


@pytest.mark.parametrize("config", CONFIGS, ids=IDS)
def test_candidates_are_mutually_distinct(config) -> None:
    family = behavior_candidate_set(_reference(config), _pose(config), 0.0, config)
    window = config.intervention_steps
    for left in range(CANDIDATE_COUNT):
        for right in range(left + 1, CANDIDATE_COUNT):
            gap = np.abs(family[left, :window] - family[right, :window]).max()
            assert gap > 1e-4, (
                f"{CANDIDATE_NAMES[left]} and {CANDIDATE_NAMES[right]} collapse "
                "to the same executed behavior"
            )


@pytest.mark.parametrize("config", CONFIGS, ids=IDS)
def test_wait_holds_the_current_pose_then_resumes(config) -> None:
    reference = _reference(config)
    pose = _pose(config)
    plan = behavior_candidate_plan(1, reference, pose, 0.25, config)

    held = plan[: config.wait_steps]
    np.testing.assert_allclose(held[:, : config.joints], np.tile(pose, (config.wait_steps, 1)))
    np.testing.assert_allclose(held[:, config.gripper], 0.25)
    np.testing.assert_allclose(plan[config.wait_steps :], reference[: -config.wait_steps])


@pytest.mark.parametrize("config", CONFIGS, ids=IDS)
def test_retreat_moves_away_from_the_nominal_target(config) -> None:
    reference = _reference(config)
    pose = _pose(config)
    plan = behavior_candidate_plan(2, reference, pose, 0.0, config)
    window = config.intervention_steps

    direction = reference[window - 1, : config.joints] - pose
    displacement = plan[window - 1, : config.joints] - pose
    # The executed retreat opposes the nominal direction.
    assert float(displacement @ direction) < 0.0
    np.testing.assert_allclose(
        displacement, -config.retreat_scale * direction, rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("config", CONFIGS, ids=IDS)
def test_retreat_stays_within_the_nominal_rate_envelope(config) -> None:
    """Yielding must not be rejected as an illegal lunge."""
    reference = _reference(config)
    pose = _pose(config)
    plan = behavior_candidate_plan(2, reference, pose, 0.0, config)
    bounds = np.full(config.action_dim, 1e3, dtype=np.float32)

    valid, failures = validate_behavior_candidate(
        2, plan, reference, pose, -bounds, bounds, config
    )
    assert valid, failures


@pytest.mark.parametrize("config", CONFIGS, ids=IDS)
def test_grip_shifts_move_only_the_gripper_channel(config) -> None:
    reference = _reference(config)
    pose = _pose(config)
    delayed = behavior_candidate_plan(3, reference, pose, 0.0, config)
    advanced = behavior_candidate_plan(4, reference, pose, 0.0, config)

    for plan in (delayed, advanced):
        np.testing.assert_array_equal(
            plan[:, : config.joints], reference[:, : config.joints]
        )

    shift = config.grip_shift_steps
    # The nominal gripper closes at step 5; delaying pushes that later.
    nominal_close = int(np.argmax(reference[:, config.gripper] > 0.5))
    delayed_close = int(np.argmax(delayed[:, config.gripper] > 0.5))
    advanced_close = int(np.argmax(advanced[:, config.gripper] > 0.5))
    assert delayed_close == nominal_close + shift
    assert advanced_close == max(0, nominal_close - shift)


@pytest.mark.parametrize("config", CONFIGS, ids=IDS)
def test_slow_preserves_the_path_but_lags_the_nominal_plan(config) -> None:
    reference = _reference(config)
    pose = _pose(config)
    plan = behavior_candidate_plan(5, reference, pose, 0.0, config)
    window = config.intervention_steps

    executed = plan[:window, : config.joints]
    nominal = reference[:window, : config.joints]
    # Same direction of travel, less distance covered.
    assert np.linalg.norm(executed[-1] - pose) < np.linalg.norm(nominal[-1] - pose)
    assert np.all(np.diff(executed, axis=0) >= -1e-6)


@pytest.mark.parametrize("config", CONFIGS, ids=IDS)
def test_a_stationary_arm_yields_a_degenerate_retreat(config) -> None:
    """With no nominal motion there is nothing to yield, and retreat says so."""
    reference = np.zeros((config.action_horizon, config.action_dim), dtype=np.float32)
    pose = _pose(config)
    plan = behavior_candidate_plan(2, reference, pose, 0.0, config)
    np.testing.assert_allclose(plan[: config.intervention_steps, : config.joints], 0.0)


@pytest.mark.parametrize("config", CONFIGS, ids=IDS)
def test_out_of_bounds_candidates_are_rejected(config) -> None:
    reference = _reference(config)
    pose = _pose(config)
    plan = behavior_candidate_plan(1, reference, pose, 0.0, config)
    tight = np.full(config.action_dim, 1e-6, dtype=np.float32)

    valid, failures = validate_behavior_candidate(
        1, plan, reference, pose, -tight, tight, config
    )
    assert not valid and "action_domain" in failures


def test_config_rejects_a_window_that_cannot_hold_its_own_transforms() -> None:
    with pytest.raises(ValueError, match="wait must fit"):
        BehaviorCandidateConfig(
            action_horizon=100, action_dim=8, intervention_steps=2, wait_steps=4
        )


def test_unknown_candidate_is_rejected() -> None:
    reference = _reference(ROBOFACTORY)
    with pytest.raises(ValueError, match="unknown CARE behavior candidate"):
        behavior_candidate_plan(
            CANDIDATE_COUNT, reference, _pose(ROBOFACTORY), 0.0, ROBOFACTORY
        )


@pytest.mark.parametrize("config", CONFIGS, ids=IDS)
def test_retreat_is_bounded_where_it_is_generated(config) -> None:
    """Retreat is the one transform that leaves the reference's range.

    Every other candidate resamples values the nominal plan already visits, so
    it cannot go out of range. Retreat extrapolates away from them, and some
    benchmarks execute candidates without clipping, so the bound has to be
    applied to the retreat target rather than to the finished chunk.
    """
    reference = _reference(config, rate=0.05)
    pose = _pose(config)
    low = np.full(config.action_dim, -0.05, dtype=np.float32)
    high = np.full(config.action_dim, 10.0, dtype=np.float32)
    window = config.intervention_steps

    unbounded = behavior_candidate_plan(2, reference, pose, 0.0, config)
    bounded = behavior_candidate_plan(
        2, reference, pose, 0.0, config, joint_low=low, joint_high=high
    )

    assert unbounded[:window, : config.joints].min() < -0.05
    assert bounded[:window, : config.joints].min() >= -0.05 - 1e-6


@pytest.mark.parametrize("config", CONFIGS, ids=IDS)
def test_a_bounded_retreat_still_opposes_the_nominal_direction(config) -> None:
    """Clamping must shorten the yield, not reverse or cancel it."""
    reference = _reference(config, rate=0.05)
    pose = _pose(config)
    low = np.full(config.action_dim, -0.05, dtype=np.float32)
    high = np.full(config.action_dim, 10.0, dtype=np.float32)
    window = config.intervention_steps

    plan = behavior_candidate_plan(
        2, reference, pose, 0.0, config, joint_low=low, joint_high=high
    )
    direction = reference[window - 1, : config.joints] - pose
    displacement = plan[window - 1, : config.joints] - pose

    assert float(displacement @ direction) < 0.0


@pytest.mark.parametrize("config", CONFIGS, ids=IDS)
def test_no_room_to_yield_degenerates_to_holding_position(config) -> None:
    """Against a joint limit there is nowhere to retreat, and that is correct."""
    reference = _reference(config, rate=0.05)
    pose = _pose(config)
    tight = np.zeros(config.action_dim, dtype=np.float32)
    high = np.full(config.action_dim, 10.0, dtype=np.float32)
    window = config.intervention_steps

    plan = behavior_candidate_plan(
        2, reference, pose, 0.0, config, joint_low=tight, joint_high=high
    )
    np.testing.assert_allclose(plan[:window, : config.joints], 0.0, atol=1e-6)


@pytest.mark.parametrize("config", CONFIGS, ids=IDS)
def test_bounds_must_be_consistent(config) -> None:
    reference = _reference(config)
    pose = _pose(config)
    low = np.full(config.action_dim, 1.0, dtype=np.float32)
    high = np.full(config.action_dim, -1.0, dtype=np.float32)

    with pytest.raises(ValueError, match="matching finite joint limits"):
        behavior_candidate_plan(
            2, reference, pose, 0.0, config, joint_low=low, joint_high=high
        )
