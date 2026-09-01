from __future__ import annotations

import math

from deployment.bicoord_care.branch_fidelity import (
    FIDELITY_SCHEMA,
    FIDELITY_STEPS,
    FIDELITY_TOLERANCE,
    strict_fidelity_receipts_valid,
    strict_fidelity_row_valid,
)


def _row(repeat_id: int) -> dict[str, object]:
    return {
        "schema": FIDELITY_SCHEMA,
        "tolerance": FIDELITY_TOLERANCE,
        "repeat_id": repeat_id,
        "passed": True,
        "utility_max_abs_error": 0.0,
        "bounded_utility_max_abs_error": 0.0,
        "executed_action_max_abs_error": 0.0,
        "qpos_max_abs_error": 0.0,
        "progress_max_abs_error": 0.0,
        "executed_action_max_abs_error_by_step": [0.0] * FIDELITY_STEPS,
        "qpos_max_abs_error_by_step": [0.0] * FIDELITY_STEPS,
        "progress_max_abs_error_by_step": [0.0] * FIDELITY_STEPS,
        "metric_length_equal": True,
        "trajectory_complete": True,
        "branch_contract_equal": True,
        "active_labels_equal": True,
        "stagnant_labels_equal": True,
        "success_labels_equal": True,
        "discrete_labels_equal": True,
        "safety_labels_equal": True,
        "outcome_discrete_labels_equal": True,
        "branch_contract_difference_fields": [],
        "active_label_difference_steps": [],
        "all_joint_changes_label_difference_steps": [],
        "success_label_difference_steps": [],
        "discrete_label_difference_steps": {"success": []},
        "safety_label_difference_steps": {"drop": []},
        "outcome_discrete_difference_horizons": {"8": [], "16": [], "32": [], "64": []},
    }


def test_strict_fidelity_requires_both_unique_repeats() -> None:
    assert strict_fidelity_receipts_valid([_row(0), _row(1)])
    assert not strict_fidelity_receipts_valid([_row(0), _row(0)])
    assert not strict_fidelity_receipts_valid([_row(0)])


def test_strict_fidelity_rejects_nan_infinity_negative_and_missing_fields() -> None:
    for value in (math.nan, math.inf, -1e-9):
        row = _row(0)
        row["qpos_max_abs_error"] = value
        assert not strict_fidelity_row_valid(row)
    row = _row(0)
    del row["success_labels_equal"]
    assert not strict_fidelity_row_valid(row)
