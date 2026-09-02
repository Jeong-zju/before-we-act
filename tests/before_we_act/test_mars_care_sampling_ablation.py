from __future__ import annotations

import numpy as np

from scripts.before_we_act.prepare_mars_care_sampling_ablation import (
    bootstrap_mean_interval,
    empirical_percentile,
    moving_max,
    phase_bin,
    signal_counts,
)


SPEC = {
    "sampling": {
        "critical_phase_range": [0.35, 0.80],
        "uniform_phase_range": [0.10, 0.85],
    }
}


def test_event_rank_and_phase_bins_are_deterministic_and_bounded() -> None:
    assert np.array_equal(empirical_percentile(np.asarray([3.0, 1.0, 2.0])), [1.0, 1 / 3, 2 / 3])
    assert np.array_equal(moving_max(np.asarray([0.0, 2.0, 1.0, 0.0]), 1), [2.0, 2.0, 2.0, 1.0])
    bins = [phase_bin("critical", index, 20, SPEC) for index in range(20)]
    assert bins[0][0] == 0.35
    assert bins[-1][1] == 0.80
    assert all(left[1] == right[0] for left, right in zip(bins, bins[1:]))


def branch(candidate: int, regime: str, repeat: int, signal: float) -> dict:
    outcomes = {}
    for horizon in (8, 16, 32, 64):
        value = signal if candidate else 0.0
        if regime == "reactive":
            value *= 2.0
        outcomes[str(horizon)] = {
            "bounded_utility_vector": [value, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "hard_safety_violation": False,
        }
    return {
        "candidate_id": candidate,
        "regime": regime,
        "repeat_id": repeat,
        "candidate_valid": True,
        "status": "VALID",
        "outcomes": outcomes,
    }


def family(signal: float) -> dict:
    return {
        "branches": [
            branch(candidate, regime, repeat, signal * candidate)
            for repeat in (0, 1)
            for regime in ("reactive", "replay")
            for candidate in range(6)
        ]
    }


def test_signal_count_is_zero_for_flat_family_and_positive_for_separated_family() -> None:
    flat = signal_counts(family(0.0), 0.01)
    separated = signal_counts(family(0.1), 0.01)
    assert flat["branch_signal_density"] == 0.0
    assert flat["effective_pairs"] == 0
    assert separated["branch_signal_density"] > 0.0
    assert separated["effective_pairs"] > 0


def test_paired_bootstrap_interval_preserves_strict_positive_support() -> None:
    lower, upper = bootstrap_mean_interval([0.1, 0.2, 0.3, 0.4], 1000, 7)
    assert 0.0 < lower <= upper
