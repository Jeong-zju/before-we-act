from __future__ import annotations

import math

from deployment.bicoord_care.branch_fidelity import (
    FIDELITY_SCHEMA,
    FIDELITY_STEPS,
    EXPECTED_DISCRETE_DIFFERENCE_KEYS,
    EXPECTED_SAFETY_DIFFERENCE_KEYS,
    EXPECTED_OUTCOME_DIFFERENCE_KEYS,
    FIDELITY_TOLERANCE,
    physical_branch_family_rows_valid,
    seed_replay_probe_valid,
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
        "bounded_utility_contract_valid": True,
        "branch_contract_difference_fields": [],
        "active_label_difference_steps": [],
        "all_joint_changes_label_difference_steps": [],
        "success_label_difference_steps": [],
        "discrete_label_difference_steps": {
            key: [] for key in EXPECTED_DISCRETE_DIFFERENCE_KEYS
        },
        "safety_label_difference_steps": {
            key: [] for key in EXPECTED_SAFETY_DIFFERENCE_KEYS
        },
        "outcome_discrete_difference_horizons": {
            key: [] for key in EXPECTED_OUTCOME_DIFFERENCE_KEYS
        },
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


def test_seed_replay_probe_rejects_nonfinite_error_or_hash_drift() -> None:
    digest = "a" * 64
    probe = {
        "schema": "before-we-act.bicoord.seed-replay-probe/1",
        "restore_mode": "official_seed_plus_reference_prefix_replay",
        "repeats": 2,
        "tolerance": FIDELITY_TOLERANCE,
        "max_abs_error": 0.0,
        "passed": True,
        "expected_anchor_state_sha256": digest,
        "rebuilt_anchor_state_sha256": [digest, digest],
        "rebuilt_anchor_state_exact_match": True,
    }
    assert seed_replay_probe_valid(probe)
    assert not seed_replay_probe_valid({**probe, "max_abs_error": math.nan})
    assert not seed_replay_probe_valid(
        {**probe, "rebuilt_anchor_state_sha256": [digest, "b" * 64]}
    )


def _physical_row(candidate: int, regime: str, repeat: int) -> dict[str, object]:
    metrics = [
        {
            "branch_step": step,
            "progress": 0.0,
            "success": False,
            "hard_safety_violation": False,
            "collision_or_drop": False,
            "robot_conflict": False,
            "duplicate_work": False,
            "active": [False, False],
            "all_joint_changes_below_0_02": True,
            "qpos": [[0.0] * 7, [0.0] * 7],
            "safety": {
                "dropped_actors": [],
                "robot_robot_contacts": [],
                "drop": False,
                "robot_collision": False,
                "hard_safety_violation": False,
            },
        }
        for step in range(FIDELITY_STEPS)
    ]
    outcomes = {
        str(horizon): {
            "requested_steps": horizon,
            "observed_steps": horizon,
            "bounded_utility_vector": [0.0] * 8,
            "utility_main": 0.0,
            "hard_safety_violation": False,
            "first_success_step": None,
            "final_progress": 0.0,
            "active_fraction": [0.0, 0.0],
            "physical_simulator_outcome": True,
        }
        for horizon in (8, 16, 32, 64)
    }
    return {
        "candidate_id": candidate,
        "regime": regime,
        "repeat_id": repeat,
        "branch_seed": 100 + repeat,
        "status": "VALID",
        "physical_simulator_outcome": True,
        "simulator_steps": FIDELITY_STEPS,
        "intervention_steps": 1,
        "candidate_transform_clipped": False,
        "action_clipped": False,
        "peer_action_source": (
            "reactive_policy"
            if regime == "reactive"
            else "candidate0_reactive_replay_log"
        ),
        "peer_policy_output_used": regime == "reactive",
        "focal_policy_output_used": True,
        "replay_peer_action_max_abs_error": 0.0,
        "executed_actions": [[[0.0] * 7, [0.0] * 7] for _ in range(64)],
        "metrics": metrics,
        "outcomes": outcomes,
    }


def test_physical_family_rows_bind_role_schema_and_repeat_seeds() -> None:
    rows = [
        _physical_row(candidate, regime, repeat)
        for candidate in range(6)
        for regime in ("reactive", "replay")
        for repeat in (0, 1)
    ]
    assert physical_branch_family_rows_valid(rows)

    rows[0]["peer_action_source"] = "candidate0_reactive_replay_log"
    assert not physical_branch_family_rows_valid(rows)
    rows[0]["peer_action_source"] = "reactive_policy"
    rows[0]["metrics"][0]["progress"] = math.nan
    assert not physical_branch_family_rows_valid(rows)
