from __future__ import annotations

import torch

from scripts.before_we_act.analyze_mars_care_scorer_v2 import (
    DEFAULT_CONDITIONS,
    hard_safety_nonzero_count,
    metrics,
    parse_conditions,
    safety_supervision_status,
)


def _zero_safety_row(family_index: int = 0) -> dict[str, object]:
    quantiles = torch.zeros(6, 3, 5)
    # Give the diagnostic a clearly positive lower-bound proposal.  A random
    # safety head with large positive logits would otherwise mask every slot.
    for candidate in range(1, 6):
        quantiles[candidate, 2] = torch.tensor(
            (0.1 * candidate, 0.2 * candidate, 0.3 * candidate,
             0.4 * candidate, 0.5 * candidate)
        )
    target = torch.zeros(6, 3)
    target[1:, 2] = torch.arange(1, 6, dtype=torch.float32)
    return {
        "family_index": family_index,
        "task_id": 0,
        "quantiles": quantiles,
        "hard_safety_logit": torch.full((6,), 100.0),
        "target": target,
        "hard_safety": torch.zeros(6),
    }


def test_zero_hard_safety_labels_use_legality_only_not_random_mask() -> None:
    rows = [_zero_safety_row(0), _zero_safety_row(1)]
    assert hard_safety_nonzero_count(rows) == 0
    result = metrics(rows)
    assert result["safety_supervision_degenerate"] is True
    assert result["safety_gate_mode"] == "legality_only"
    assert result["learned_safety_mask_applied"] is False
    assert result["unsafe_candidate_count"] == 0
    assert result["unsafe_candidate_rate"] == 0.0
    assert result["raw_predicted_unsafe_candidate_rate"] == 1.0
    assert result["effective_safety_probability_max"] == 1.0
    # The positive proposal remains observable rather than being forced back
    # to the structural reference by an untrained safety head.
    assert result["override_rate"] == 1.0


def test_safety_status_is_degenerate_if_training_or_calibration_has_no_positive() -> None:
    status = safety_supervision_status(0, 4)
    assert status["safety_supervision_degenerate"] is True
    assert status["safety_gate_mode"] == "legality_only"
    status = safety_supervision_status(4, 4)
    assert status["safety_supervision_degenerate"] is False
    assert status["safety_gate_mode"] == "learned_probability_uncalibrated"
    assert status["safety_probability_calibrated"] is False


def test_default_conditions_are_orthogonal_and_physical_unit() -> None:
    conditions = parse_conditions(DEFAULT_CONDITIONS)
    assert [condition.name for condition in conditions] == [
        "legacy",
        "full_robust",
        "prefix1_robust_no_ref",
        "prefix1_robust_ref",
    ]
    assert [(condition.prefix, condition.robust_scaling,
             condition.include_reference_ranking, condition.legacy)
            for condition in conditions] == [
        (100, False, False, True),
        (100, True, False, False),
        (1, True, False, False),
        (1, True, True, False),
    ]


def test_metrics_separate_median_ranking_from_lower_bound_selector() -> None:
    row = _zero_safety_row()
    quantiles = row["quantiles"]
    assert isinstance(quantiles, torch.Tensor)
    # Median ranks candidate 1 highest, while q05 selects candidate 2.  The
    # physical target makes candidate 2 optimal, so the two regrets differ.
    quantiles[:, 2] = 0.0
    quantiles[1, 2] = torch.tensor((0.1, 0.2, 0.9, 1.0, 1.1))
    quantiles[2, 2] = torch.tensor((0.8, 0.81, 0.82, 0.83, 0.84))
    target = row["target"]
    assert isinstance(target, torch.Tensor)
    target[:, 2] = 0.0
    target[1, 2] = 1.0
    target[2, 2] = 2.0
    result = metrics([row])
    assert result["median_ranking_top1_accuracy"] == 0.0
    assert result["median_ranking_mean_regret"] == 1.0
    assert result["selector_top1_accuracy"] == 1.0
    assert result["selector_mean_regret"] == 0.0
