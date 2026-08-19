from __future__ import annotations

import math

from scripts.before_we_act.evaluate_care_gate_a_separated import (
    analyze_horizon,
    candidate_records,
    separated_weight_profiles,
)


SOURCE_PROFILES = {
    "main_weights": [0.3, 0.3, 0.12, 0.08, 0.06, 0.06, 0.03, 0.05],
    "weight_profiles": {
        "progress_heavy": [0.4, 0.3, 0.08, 0.06, 0.04, 0.05, 0.02, 0.05],
        "safety_heavy": [0.2, 0.25, 0.25, 0.1, 0.05, 0.05, 0.02, 0.08],
        "coordination_heavy": [0.25, 0.25, 0.08, 0.12, 0.1, 0.1, 0.07, 0.03],
    },
}
NAMES = {
    0: "reference",
    1: "belief_off",
    2: "wait",
    3: "slow_yield",
    4: "commit",
    5: "hold_grip",
}


def gate_contract() -> dict:
    return {"team_outcome_contract": {"utility": SOURCE_PROFILES}}


def separated_contract() -> dict:
    return {
        "unchanged_scope": {
            "tie_tolerance": 1e-6,
            "bootstrap_draws": 1000,
            "bootstrap_seed": 17,
        },
        "separated_gate_a": {
            "strict_nonreference_best_fraction_min": 0.2,
            "oracle_mean_ordinary_benefit_gain_min": 0.05,
            "oracle_group_bootstrap_95_lower_min": 0.0,
            "tasks_with_positive_point_gain_min": 1,
            "coordination_candidates_required": 1,
            "coordination_candidate_ids": [2, 3, 5],
            "each_required_candidate_strict_win_fraction_in_one_distinct_task_min": 0.05,
        },
    }


def outcome(
    ordinary_progress_value: float,
    *,
    hard_safety: bool,
    collision_rate: float,
) -> dict:
    return {
        "hard_safety_violation": hard_safety,
        "bounded_utility_vector": [
            ordinary_progress_value,
            0.0,
            -collision_rate,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    }


def family() -> dict:
    profiles = separated_weight_profiles(gate_contract())
    target_progress = 0.01 / profiles["main"][0]
    branches = []
    reference_collision = [1.0 / 64.0, 2.0 / 64.0]
    for candidate_id in range(6):
        for repeat_id in (0, 1):
            progress = target_progress if candidate_id == 2 else 0.0
            hard = True
            collision = reference_collision[repeat_id]
            if candidate_id == 5 and repeat_id == 0:
                hard = False
                collision = 0.0
            branches.append(
                {
                    "candidate_id": candidate_id,
                    "regime": "reactive",
                    "repeat_id": repeat_id,
                    "outcomes": {
                        "64": outcome(
                            progress,
                            hard_safety=hard,
                            collision_rate=collision,
                        )
                    },
                }
            )
    return {
        "snapshot_id": "camera-like",
        "scenario_group_id": "group-camera-like",
        "task": "camera_alignment",
        "sampling_stratum": "critical",
        "quality_horizons": {"64": {"use_for_gate_analysis": True}},
        "branches": branches,
    }


def test_separated_profiles_remove_collision_and_renormalize() -> None:
    profiles = separated_weight_profiles(gate_contract())
    assert profiles["main"][2] == 0.0
    assert math.isclose(sum(profiles["main"]), 1.0)
    assert math.isclose(profiles["main"][0], 0.3 / 0.88)


def test_repeat_unstable_safety_candidate_is_ineligible() -> None:
    profiles = separated_weight_profiles(gate_contract())
    records = candidate_records(family(), 64, profiles)
    assert records[0]["eligible"] is True
    assert records[2]["eligible"] is True
    assert records[5]["eligible"] is False
    assert "HARD_SAFETY_REPEAT_UNSTABLE" in records[5]["ineligibility_reasons"]


def test_unstable_safety_flip_cannot_create_large_ordinary_benefit_gain() -> None:
    profiles = separated_weight_profiles(gate_contract())
    result = analyze_horizon(
        [family()],
        horizon=64,
        separated_contract=separated_contract(),
        candidate_names=NAMES,
        profiles=profiles,
    )
    gain = result["ordinary_benefit"]["family_weighted_oracle_gain"]["mean"]
    assert math.isclose(gain, 0.01)
    assert result["ordinary_benefit"]["largest_gain_snapshot_id"] == "camera-like"
    assert result["candidate_eligibility"]["exclusion_reason_counts"][
        "hold_grip|HARD_SAFETY_REPEAT_UNSTABLE"
    ] == 1
    assert result["conditions"]["mean_ordinary_benefit_gain"] is False
    assert result["diagnostic_numeric_gate_pass"] is False
