from __future__ import annotations

from scripts.before_we_act.rescore_care_branch_pilot import (
    SCIENTIFIC_RESOLUTION,
    discrete_projection,
    outcome_discrete_equal,
    percentile,
)


def test_scientific_resolution_is_inherited_gate_b_floor() -> None:
    assert SCIENTIFIC_RESOLUTION == 0.02


def test_percentile_interpolates_and_empty_is_not_accepted() -> None:
    assert percentile([0.0, 0.01], 0.95) == 0.0095
    assert percentile([], 0.95) == float("inf")


def test_discrete_projection_ignores_continuous_drift_only() -> None:
    left = {"custodian": 0, "complete": False, "distance": 1.00001}
    right = {"custodian": 0, "complete": False, "distance": 1.00002}
    assert discrete_projection(left) == discrete_projection(right)
    assert discrete_projection({**right, "complete": True}) != discrete_projection(left)


def test_outcome_equivalence_preserves_safety_success_and_stage() -> None:
    left = {
        "hard_safety_violation": False,
        "first_success_step": None,
        "final_stage_id": "handoff",
        "observed_steps": 64,
        "final_factorized_predicates": {"complete": False, "distance": 1.00001},
    }
    right = {
        **left,
        "final_factorized_predicates": {"complete": False, "distance": 1.00002},
    }
    assert outcome_discrete_equal(left, right)
    right["hard_safety_violation"] = True
    assert not outcome_discrete_equal(left, right)
