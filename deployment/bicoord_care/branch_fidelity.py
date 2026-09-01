"""Fail-closed validation for candidate-0 BiCoord branch fidelity receipts."""
from __future__ import annotations

import math
from typing import Any, Mapping


FIDELITY_SCHEMA = "before-we-act.bicoord.care-reactive-replay-fidelity/1"
FIDELITY_TOLERANCE = 1e-6
FIDELITY_STEPS = 64
EXPECTED_DISCRETE_DIFFERENCE_KEYS = frozenset(
    {
        "branch_step",
        "success",
        "active",
        "all_joint_changes_below_0_02",
        "hard_safety_violation",
        "collision_or_drop",
        "robot_conflict",
        "duplicate_work",
    }
)
EXPECTED_SAFETY_DIFFERENCE_KEYS = frozenset(
    {
        "drop",
        "robot_collision",
        "hard_safety_violation",
        "dropped_actor_names",
        "robot_contact_bodies",
    }
)
EXPECTED_OUTCOME_DIFFERENCE_KEYS = frozenset({"8", "16", "32", "64"})

_ERROR_FIELDS = (
    "utility_max_abs_error",
    "bounded_utility_max_abs_error",
    "executed_action_max_abs_error",
    "qpos_max_abs_error",
    "progress_max_abs_error",
)
_EQUALITY_FIELDS = (
    "metric_length_equal",
    "trajectory_complete",
    "branch_contract_equal",
    "active_labels_equal",
    "stagnant_labels_equal",
    "success_labels_equal",
    "discrete_labels_equal",
    "safety_labels_equal",
    "outcome_discrete_labels_equal",
)
_STEP_ERROR_FIELDS = (
    "executed_action_max_abs_error_by_step",
    "qpos_max_abs_error_by_step",
    "progress_max_abs_error_by_step",
)
_EMPTY_DIFFERENCE_FIELDS = (
    "discrete_label_difference_steps",
    "safety_label_difference_steps",
    "outcome_discrete_difference_horizons",
)
_EMPTY_LIST_FIELDS = (
    "branch_contract_difference_fields",
    "active_label_difference_steps",
    "all_joint_changes_label_difference_steps",
    "success_label_difference_steps",
)


def strict_fidelity_row_valid(value: Any) -> bool:
    """Return whether one repeat proves the complete strict fidelity contract.

    In particular, NaN and negative forged error values are rejected.  Plain
    ``error > tolerance`` checks are unsafe because comparisons with NaN are
    false.
    """

    if not isinstance(value, Mapping):
        return False
    if value.get("schema") != FIDELITY_SCHEMA:
        return False
    if value.get("passed") is not True:
        return False
    repeat_id = value.get("repeat_id")
    if not isinstance(repeat_id, int) or isinstance(repeat_id, bool) or repeat_id not in (0, 1):
        return False
    try:
        tolerance = float(value.get("tolerance"))
    except (TypeError, ValueError):
        return False
    if tolerance != FIDELITY_TOLERANCE:
        return False
    for field in _ERROR_FIELDS:
        try:
            error = float(value.get(field))
        except (TypeError, ValueError):
            return False
        if (
            not math.isfinite(error)
            or error < 0.0
            or error > FIDELITY_TOLERANCE
        ):
            return False
    for field in _STEP_ERROR_FIELDS:
        errors = value.get(field)
        if not isinstance(errors, list) or len(errors) != FIDELITY_STEPS:
            return False
        try:
            numeric = [float(error) for error in errors]
        except (TypeError, ValueError):
            return False
        if any(isinstance(error, bool) for error in errors):
            return False
        if any(
            not math.isfinite(error)
            or error < 0.0
            or error > FIDELITY_TOLERANCE
            for error in numeric
        ):
            return False
    expected_difference_keys = {
        "discrete_label_difference_steps": EXPECTED_DISCRETE_DIFFERENCE_KEYS,
        "safety_label_difference_steps": EXPECTED_SAFETY_DIFFERENCE_KEYS,
        "outcome_discrete_difference_horizons": EXPECTED_OUTCOME_DIFFERENCE_KEYS,
    }
    for field in _EMPTY_DIFFERENCE_FIELDS:
        differences = value.get(field)
        if (
            not isinstance(differences, Mapping)
            or not differences
            or set(differences) != expected_difference_keys[field]
            or any(
                not isinstance(item, list) or len(item) != 0
                for item in differences.values()
            )
        ):
            return False
    for field in _EMPTY_LIST_FIELDS:
        differences = value.get(field)
        if not isinstance(differences, list) or differences:
            return False
    return all(value.get(field) is True for field in _EQUALITY_FIELDS)


def strict_fidelity_receipts_valid(value: Any) -> bool:
    """Validate the exact two-repeat family receipt without silent duplicates."""

    return bool(
        isinstance(value, list)
        and len(value) == 2
        and {row.get("repeat_id") for row in value if isinstance(row, Mapping)}
        == {0, 1}
        and all(strict_fidelity_row_valid(row) for row in value)
    )


__all__ = [
    "FIDELITY_SCHEMA",
    "FIDELITY_STEPS",
    "FIDELITY_TOLERANCE",
    "EXPECTED_DISCRETE_DIFFERENCE_KEYS",
    "EXPECTED_SAFETY_DIFFERENCE_KEYS",
    "EXPECTED_OUTCOME_DIFFERENCE_KEYS",
    "strict_fidelity_receipts_valid",
    "strict_fidelity_row_valid",
]
