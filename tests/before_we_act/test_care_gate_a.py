from __future__ import annotations

import math

from scripts.before_we_act.evaluate_care_gate_a import (
    analyze_horizon,
    group_bootstrap_mean,
    maximum_distinct_task_matching,
    wilson_interval,
)


PROFILES = {
    "main": [0.3, 0.3, 0.12, 0.08, 0.06, 0.06, 0.03, 0.05],
    "progress_heavy": [0.4, 0.3, 0.08, 0.06, 0.04, 0.05, 0.02, 0.05],
    "safety_heavy": [0.2, 0.25, 0.25, 0.1, 0.05, 0.05, 0.02, 0.08],
    "coordination_heavy": [0.25, 0.25, 0.08, 0.12, 0.1, 0.1, 0.07, 0.03],
}
NAMES = {
    "0": "reference",
    "1": "belief_off",
    "2": "wait",
    "3": "slow_yield",
    "4": "commit",
    "5": "hold_grip",
}
THRESHOLDS = {
    "strict_nonreference_best_fraction_min": 0.2,
    "tie_tolerance": 1e-6,
    "oracle_mean_utility_gain_min": 0.05,
    "oracle_group_bootstrap_95_lower_min": 0.0,
    "tasks_with_positive_point_gain_min": 2,
    "coordination_candidates_required": 2,
    "coordination_candidate_ids": [2, 3, 5],
    "each_required_candidate_strict_win_fraction_in_one_distinct_task_min": 0.05,
    "hard_safety_rate_delta_aggregate_max": 0.005,
    "hard_safety_rate_delta_per_task_max": 0.01,
}


def outcome(value: float) -> dict:
    vector = [value / 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return {
        "utility_main": value,
        "hard_safety_violation": False,
        "bounded_utility_vector": vector,
    }


def family(snapshot_id: str, task: str, winner: int, gain: float) -> dict:
    branches = []
    for candidate_id in range(6):
        value = gain if candidate_id == winner else 0.0
        for repeat_id in (0, 1):
            branches.append(
                {
                    "candidate_id": candidate_id,
                    "regime": "reactive",
                    "repeat_id": repeat_id,
                    "outcomes": {"64": outcome(value)},
                }
            )
    return {
        "snapshot_id": snapshot_id,
        "scenario_group_id": f"group-{snapshot_id}",
        "task": task,
        "sampling_stratum": "critical",
        "quality_horizons": {"64": {"use_for_gate_analysis": True}},
        "branches": branches,
    }


def test_wilson_interval_and_bootstrap_are_well_formed_and_deterministic() -> None:
    lower, upper = wilson_interval(5, 10)
    assert 0.0 < lower < 0.5 < upper < 1.0
    first = group_bootstrap_mean([0.0, 1.0], ["a", "b"], draws=1000, seed=7)
    second = group_bootstrap_mean([0.0, 1.0], ["a", "b"], draws=1000, seed=7)
    assert first == second
    assert first["mean"] == 0.5


def test_coordination_matching_requires_distinct_tasks() -> None:
    matched = maximum_distinct_task_matching(
        {2: ["task-a", "task-b"], 3: ["task-a"], 5: ["task-a"]}
    )
    assert len(matched) == 2
    assert len(set(matched.values())) == 2


def test_gate_a_does_not_pass_when_headroom_frequency_is_high_but_gain_is_small() -> None:
    rows = [
        family("a0", "camera_alignment", 2, 0.01),
        family("a1", "camera_alignment", 2, 0.01),
        family("b0", "pass_shoe", 3, 0.02),
        family("b1", "pass_shoe", 3, 0.02),
    ]
    result = analyze_horizon(
        rows,
        horizon=64,
        thresholds=THRESHOLDS,
        candidate_names=NAMES,
        profiles=PROFILES,
        draws=1000,
        seed=11,
    )
    assert result["strict_nonreference_best"]["fraction"] == 1.0
    assert math.isclose(result["oracle_gain_family_weighted"]["mean"], 0.015)
    assert result["coordination_candidate_diversity"]["matched_candidate_count"] == 2
    assert result["numeric_conditions"]["oracle_family_weighted_mean_gain"] is False
    assert result["numeric_gate_pass"] is False
    assert result["claimable_gate_pass"] is False
