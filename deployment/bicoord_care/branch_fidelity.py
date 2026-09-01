"""Fail-closed validation for candidate-0 BiCoord branch fidelity receipts."""
from __future__ import annotations

import math
from typing import Any, Mapping

from .config import ACTION_DIM


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
EXPECTED_HORIZONS = (8, 16, 32, 64)
EXPECTED_BRANCH_KEYS = frozenset(
    (candidate, regime, repeat)
    for candidate in range(6)
    for regime in ("reactive", "replay")
    for repeat in (0, 1)
)

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
    "bounded_utility_contract_valid",
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


def finite_error_within(value: Any, tolerance: float = FIDELITY_TOLERANCE) -> bool:
    """Reject missing, boolean, negative, and non-finite error evidence."""

    if isinstance(value, bool):
        return False
    try:
        error = float(value)
    except (TypeError, ValueError):
        return False
    return bool(math.isfinite(error) and 0.0 <= error <= tolerance)


def seed_replay_probe_valid(value: Any) -> bool:
    """Validate the exact seeded-prefix reconstruction evidence."""

    if not isinstance(value, Mapping):
        return False
    if value.get("schema") != "before-we-act.bicoord.seed-replay-probe/1":
        return False
    if value.get("restore_mode") != "official_seed_plus_reference_prefix_replay":
        return False
    if value.get("passed") is not True or value.get("rebuilt_anchor_state_exact_match") is not True:
        return False
    repeats = value.get("repeats")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats != 2:
        return False
    try:
        tolerance = float(value.get("tolerance"))
    except (TypeError, ValueError):
        return False
    if tolerance != FIDELITY_TOLERANCE or not finite_error_within(
        value.get("max_abs_error"), tolerance
    ):
        return False
    expected = value.get("expected_anchor_state_sha256")
    rebuilt = value.get("rebuilt_anchor_state_sha256")
    return bool(
        isinstance(expected, str)
        and len(expected) == 64
        and _hex_digest(expected)
        and isinstance(rebuilt, list)
        and len(rebuilt) == 2
        and all(item == expected for item in rebuilt)
    )


def _hex_digest(value: str) -> bool:
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_matrix(value: Any, rows: int, columns: int) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == rows
        and all(
            isinstance(row, list)
            and len(row) == columns
            and all(_finite_number(item) for item in row)
            for row in value
        )
    )


def _physical_metric_valid(value: Any, step: int) -> bool:
    if not isinstance(value, Mapping) or value.get("branch_step") != step:
        return False
    for field in (
        "success",
        "hard_safety_violation",
        "collision_or_drop",
        "robot_conflict",
        "duplicate_work",
        "all_joint_changes_below_0_02",
    ):
        if not isinstance(value.get(field), bool):
            return False
    active = value.get("active")
    if not isinstance(active, list) or len(active) != 2 or any(
        not isinstance(item, bool) for item in active
    ):
        return False
    if not _finite_number(value.get("progress")) or not _finite_matrix(
        value.get("qpos"), 2, ACTION_DIM
    ):
        return False
    safety = value.get("safety")
    if not isinstance(safety, Mapping):
        return False
    for field in ("drop", "robot_collision", "hard_safety_violation"):
        if not isinstance(safety.get(field), bool):
            return False
    dropped = safety.get("dropped_actors")
    contacts = safety.get("robot_robot_contacts")
    if (
        not isinstance(dropped, list)
        or any(not isinstance(item, str) or not item for item in dropped)
        or len(set(dropped)) != len(dropped)
        or not isinstance(contacts, list)
    ):
        return False
    for contact in contacts:
        if not isinstance(contact, Mapping):
            return False
        bodies = contact.get("bodies")
        if (
            not isinstance(bodies, list)
            or len(bodies) != 2
            or any(not isinstance(body, str) or not body for body in bodies)
            or not _finite_number(contact.get("impulse_l2_sum"))
            or float(contact["impulse_l2_sum"]) <= 0.0
        ):
            return False
    drop = bool(dropped)
    collision = bool(contacts)
    hard = bool(drop or collision)
    return bool(
        safety.get("drop") is drop
        and safety.get("robot_collision") is collision
        and safety.get("hard_safety_violation") is hard
        and value.get("hard_safety_violation") is hard
        and value.get("collision_or_drop") is hard
        and value.get("robot_conflict") is collision
    )


def _physical_outcomes_valid(value: Any, metrics: list[Mapping[str, Any]]) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        str(horizon) for horizon in EXPECTED_HORIZONS
    }:
        return False
    for horizon in EXPECTED_HORIZONS:
        outcome = value.get(str(horizon))
        if not isinstance(outcome, Mapping):
            return False
        if (
            outcome.get("requested_steps") != horizon
            or outcome.get("observed_steps") != horizon
            or outcome.get("physical_simulator_outcome") is not True
            or not isinstance(outcome.get("hard_safety_violation"), bool)
            or not _finite_number(outcome.get("utility_main"))
            or not _finite_number(outcome.get("final_progress"))
        ):
            return False
        vector = outcome.get("bounded_utility_vector")
        fraction = outcome.get("active_fraction")
        if (
            not isinstance(vector, list)
            or len(vector) != 8
            or any(not _finite_number(item) for item in vector)
            or not isinstance(fraction, list)
            or len(fraction) != 2
            or any(not _finite_number(item) or not 0.0 <= float(item) <= 1.0 for item in fraction)
        ):
            return False
        first_success = outcome.get("first_success_step")
        if first_success is not None and (
            not isinstance(first_success, int)
            or isinstance(first_success, bool)
            or not 1 <= first_success <= horizon
        ):
            return False
        observed = metrics[:horizon]
        derived_success = next(
            (index + 1 for index, metric in enumerate(observed) if metric["success"]),
            None,
        )
        if (
            first_success != derived_success
            or outcome.get("hard_safety_violation")
            is not any(metric["hard_safety_violation"] for metric in observed)
            or float(outcome["final_progress"]) != float(observed[-1]["progress"])
            or any(
                abs(
                    float(fraction[arm])
                    - sum(bool(metric["active"][arm]) for metric in observed) / horizon
                )
                > 1e-12
                for arm in range(2)
            )
        ):
            return False
    return True


def physical_branch_row_valid(value: Any) -> bool:
    """Validate one branch row's physical schema and simulator-derived labels."""

    if not isinstance(value, Mapping):
        return False
    candidate = value.get("candidate_id")
    repeat = value.get("repeat_id")
    regime = value.get("regime")
    if (
        not isinstance(candidate, int)
        or isinstance(candidate, bool)
        or not 0 <= candidate < 6
        or not isinstance(repeat, int)
        or isinstance(repeat, bool)
        or repeat not in (0, 1)
        or regime not in ("reactive", "replay")
        or not isinstance(value.get("branch_seed"), int)
        or isinstance(value.get("branch_seed"), bool)
        or value["branch_seed"] < 0
    ):
        return False
    expected = {
        "status": "VALID",
        "physical_simulator_outcome": True,
        "simulator_steps": 64,
        "intervention_steps": 1,
        "candidate_transform_clipped": False,
        "action_clipped": False,
        "focal_policy_output_used": True,
        "peer_action_source": "reactive_policy" if regime == "reactive" else "candidate0_reactive_replay_log",
        "peer_policy_output_used": regime == "reactive",
    }
    if any(value.get(key) != observed for key, observed in expected.items()):
        return False
    replay_error = value.get("replay_peer_action_max_abs_error")
    if (
        not _finite_number(replay_error)
        or not 0.0 <= float(replay_error) <= FIDELITY_TOLERANCE
    ):
        return False
    actions = value.get("executed_actions")
    metrics = value.get("metrics")
    if (
        not isinstance(actions, list)
        or len(actions) != 64
        or any(not _finite_matrix(action, 2, ACTION_DIM) for action in actions)
        or not isinstance(metrics, list)
        or len(metrics) != 64
        or any(not _physical_metric_valid(metric, step) for step, metric in enumerate(metrics))
    ):
        return False
    return _physical_outcomes_valid(value.get("outcomes"), metrics)


def _nested_within(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if left is None or right is None:
        return left is right
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    if _finite_number(left) and _finite_number(right):
        return abs(float(left) - float(right)) <= FIDELITY_TOLERANCE
    if isinstance(left, list) or isinstance(right, list):
        return bool(
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(_nested_within(a, b) for a, b in zip(left, right))
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return bool(
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_nested_within(left[key], right[key]) for key in left)
        )
    return left == right


def physical_branch_family_rows_valid(branches: Any) -> bool:
    """Validate all 24 unique physical rows and repeat seed isolation."""

    if not isinstance(branches, list) or len(branches) != len(EXPECTED_BRANCH_KEYS):
        return False
    if any(not physical_branch_row_valid(row) for row in branches):
        return False
    keyed = {
        (row["candidate_id"], row["regime"], row["repeat_id"]): row
        for row in branches
    }
    if set(keyed) != EXPECTED_BRANCH_KEYS:
        return False
    repeat_seeds = []
    for repeat in (0, 1):
        seeds = {row["branch_seed"] for row in branches if row["repeat_id"] == repeat}
        if len(seeds) != 1:
            return False
        repeat_seeds.append(next(iter(seeds)))
        reactive = keyed[(0, "reactive", repeat)]
        replay = keyed[(0, "replay", repeat)]
        if not all(
            _nested_within(reactive[field], replay[field])
            for field in ("executed_actions", "metrics", "outcomes")
        ):
            return False
    return repeat_seeds[0] != repeat_seeds[1]




__all__ = [
    "FIDELITY_SCHEMA",
    "FIDELITY_STEPS",
    "FIDELITY_TOLERANCE",
    "EXPECTED_DISCRETE_DIFFERENCE_KEYS",
    "EXPECTED_SAFETY_DIFFERENCE_KEYS",
    "EXPECTED_OUTCOME_DIFFERENCE_KEYS",
    "EXPECTED_HORIZONS",
    "EXPECTED_BRANCH_KEYS",
    "finite_error_within",
    "seed_replay_probe_valid",
    "physical_branch_row_valid",
    "physical_branch_family_rows_valid",
    "strict_fidelity_receipts_valid",
    "strict_fidelity_row_valid",
]
