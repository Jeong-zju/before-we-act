from __future__ import annotations

from before_we_act.formal_care_sampling import (
    TASKS,
    compact_gate_targets,
    empirical_percentile,
    fitted_references,
    formal_targets,
    gate_first_targets,
    pool_size,
    score_critical,
    split_name,
)


def test_formal_targets_have_exact_counts_and_mixes() -> None:
    targets = formal_targets()
    assert sum(targets.values()) == 14_800
    assert sum(v for (s, k, _), v in targets.items() if s == "train" and k == "uniform") == 5_000
    assert sum(v for (s, k, _), v in targets.items() if s == "train" and k == "critical") == 5_000
    assert sum(v for (s, _, _), v in targets.items() if s == "validation") == 1_200
    assert sum(v for (s, _, _), v in targets.items() if s == "calibration") == 1_200
    assert sum(v for (s, k, _), v in targets.items() if s == "test" and k == "uniform") == 1_200
    assert sum(v for (s, k, _), v in targets.items() if s == "test" and k == "critical") == 1_200


def test_pool_sizes_leave_frozen_spares() -> None:
    assert pool_size(100, "uniform") == 106
    assert pool_size(100, "critical") == 358
    assert int(0.30 * pool_size(100, "critical")) >= 100


def test_gate_first_targets_are_exactly_150_per_task() -> None:
    targets = gate_first_targets()
    assert sum(targets.values()) == 900
    for task in TASKS:
        assert targets[("test", "critical", task)] == 100
        assert targets[("test", "uniform", task)] == 50
        assert sum(
            value
            for (_split, _stratum, row_task), value in targets.items()
            if row_task == task
        ) == 150


def test_compact_gate_targets_are_exactly_30_per_task() -> None:
    targets = compact_gate_targets()
    assert sum(targets.values()) == 180
    for task in TASKS:
        assert targets[("test", "critical", task)] == 20
        assert targets[("test", "uniform", task)] == 10


def test_split_hash_is_deterministic_and_total() -> None:
    for task in TASKS:
        value = split_name(task, "example-group")
        assert value in {"train", "validation", "calibration", "test"}
        assert split_name(task, "example-group") == value


def test_empirical_percentile_and_critical_score() -> None:
    train = [
        {
            "features": {
                "residual_norm": value,
                "belief_entropy": value,
                "reliability": 1.0 - value,
            }
        }
        for value in (0.0, 0.5, 1.0)
    ]
    references = fitted_references(train)
    assert empirical_percentile(references["residual_norm"], 0.5) == 2 / 3
    row = {
        "features": {
            "residual_norm": 1.0,
            "belief_entropy": 1.0,
            "reliability": 0.0,
            "partner_occlusion": True,
            "contact_or_phase_change": True,
            "paired_inactivity": True,
        }
    }
    score, percentiles = score_critical(row, references)
    assert score == 1.0
    assert all(value == 1.0 for value in percentiles.values())
