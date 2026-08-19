from __future__ import annotations

import numpy as np

from before_we_act.care_branch_collector import (
    ConsolidatedChunkEnsembler,
    annotate_common_replay_support,
    canonicalize_plan,
    candidate_is_active,
    candidate_plan,
    deadlock_mask,
    freeze_common_replay_support,
    time_warp_plan,
    validate_candidate,
)


class _Space:
    low = np.asarray([-10.0] * 7 + [-1.0], dtype=np.float32)
    high = np.asarray([10.0] * 7 + [1.0], dtype=np.float32)


def reference_plan() -> np.ndarray:
    time = np.arange(1, 101, dtype=np.float32)[:, None]
    arm = np.repeat(time * 0.001, 7, axis=1)
    grip = np.where(np.arange(100) < 50, -1.0, 1.0).astype(np.float32)[:, None]
    return np.concatenate((arm, grip), axis=1)


def test_consolidated_plan_matches_existing_current_step_weighting() -> None:
    ensembler = ConsolidatedChunkEnsembler((0,))
    first = np.zeros((1, 100, 8), dtype=np.float32)
    second = np.ones((1, 100, 8), dtype=np.float32)
    ensembler.append_and_plan(0, first)
    plan = ensembler.append_and_plan(1, second)["panda-0"]
    weights = np.exp(-0.01 * np.asarray([1.0, 0.0]))
    weights /= weights.sum()
    assert np.allclose(plan[0], weights[1])


def test_frozen_candidates_have_expected_first_step_semantics() -> None:
    reference = reference_plan()
    base = reference + 0.0001
    current = np.zeros(9, dtype=np.float32)
    assert np.array_equal(candidate_plan(0, reference, base, current, -1.0), reference)
    assert np.array_equal(candidate_plan(1, reference, base, current, -1.0), base)
    wait = candidate_plan(2, reference, base, current, -1.0)
    assert np.array_equal(wait[0, :7], current[:7])
    assert wait[0, 7] == -1.0
    hold = candidate_plan(5, reference, base, current, -1.0)
    assert np.all(hold[:, 7] == -1.0)


def test_receding_horizon_candidate_schedule_is_exact() -> None:
    assert [candidate_is_active(step, 4) for step in range(7)] == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    try:
        candidate_is_active(0, 0)
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("zero-step intervention must be rejected")


def test_time_warp_is_anchored_and_gripper_is_zero_order_hold() -> None:
    reference = reference_plan()
    current = np.zeros(9, dtype=np.float32)
    slow = time_warp_plan(reference, current, 0.75, -1.0)
    commit = time_warp_plan(reference, current, 1.25, -1.0)
    assert np.allclose(slow[0, :7], 0.00075)
    assert np.allclose(commit[0, :7], 0.00125)
    assert slow[0, 7] == -1.0
    assert commit[0, 7] == reference[0, 7]


def test_candidate_legality_rejects_projection_and_accepts_reference() -> None:
    reference = reference_plan()
    base = reference.copy()
    current = np.zeros(9, dtype=np.float32)
    valid, failures = validate_candidate(
        0, reference, reference, base, current, -1.0, _Space()
    )
    assert valid and failures == []
    invalid = reference.copy()
    invalid[0, 0] = 20.0
    valid, failures = validate_candidate(
        0, invalid, reference, base, current, -1.0, _Space()
    )
    assert not valid
    assert "action_domain" in failures


def test_physical_canonicalization_makes_frozen_safe_transforms_legal() -> None:
    raw_reference = reference_plan()
    raw_reference[:, 7] *= 1.02
    reference, diagnostics = canonicalize_plan(raw_reference, _Space())
    raw_base = reference_plan() + 0.0001
    base, _ = canonicalize_plan(raw_base, _Space())
    current = np.zeros(9, dtype=np.float32)
    assert diagnostics["changed_values"] == 100
    assert diagnostics["max_abs_change"] > 0.019
    for candidate_id in (0, 2, 3, 4, 5):
        plan = candidate_plan(
            candidate_id, reference, base, current, -1.0
        )
        valid, failures = validate_candidate(
            candidate_id,
            plan,
            reference,
            base,
            current,
            -1.0,
            _Space(),
        )
        assert valid, (candidate_id, failures)


def test_belief_off_uses_its_own_canonical_gripper_knots() -> None:
    reference = reference_plan()
    raw_base = reference.copy()
    raw_base[:, 0] = 1.0
    raw_base[:, 7] = np.linspace(-1.02, 1.02, 100, dtype=np.float32)
    base, _ = canonicalize_plan(raw_base, _Space())
    current = np.zeros(9, dtype=np.float32)
    plan = candidate_plan(1, reference, base, current, -1.0)
    valid, failures = validate_candidate(
        1, plan, reference, base, current, -1.0, _Space()
    )
    assert valid, failures


def test_deadlock_marks_only_runs_of_at_least_eight_steps() -> None:
    rows = [
        {"progress": 0.2, "all_joint_changes_below_0_02": True}
        for _ in range(10)
    ]
    mask = deadlock_mask(rows)
    assert mask[0] is False
    assert sum(mask) == 9


def test_early_reference_success_freezes_common_replay_support() -> None:
    reference = {
        "steps": 22,
        "outcomes": {str(value): {"utility_main": float(value)} for value in (8, 16, 32, 64)},
    }
    teammate_log = [{"panda-1": np.zeros(8, dtype=np.float32)} for _ in range(22)]
    support = freeze_common_replay_support(reference, teammate_log)
    sibling = {"steps": 22, "outcomes": {"8": {}, "16": {}}}
    annotate_common_replay_support(sibling, support)

    assert support == 22
    assert sorted(reference["outcomes"]) == ["16", "8"]
    for branch in (reference, sibling):
        assert branch["common_replay_support_steps"] == 22
        assert branch["supported_outcome_horizons"] == [8, 16]
        assert branch["unsupported_outcome_horizons"] == [32, 64]


def test_common_replay_support_rejects_log_length_mismatch() -> None:
    reference = {"steps": 22, "outcomes": {"8": {}, "16": {}}}
    teammate_log = [{"panda-1": np.zeros(8, dtype=np.float32)} for _ in range(21)]
    try:
        freeze_common_replay_support(reference, teammate_log)
    except RuntimeError as error:
        assert "disagree" in str(error)
    else:
        raise AssertionError("mismatched replay support must be rejected")
