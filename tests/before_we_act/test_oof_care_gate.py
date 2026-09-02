from __future__ import annotations

import numpy as np
import pytest

from scripts.before_we_act.oof_care_gate import (
    AdmissionThresholds,
    admission_gate,
    fit_oof_calibration,
    make_family_folds,
    oof_metrics,
    validate_oof_rows,
)


def _row(family: int, fold: int, fit: list[int], *, task: int = 0) -> dict[str, object]:
    quantiles = np.zeros((6, 3, 5), dtype=np.float64)
    target = np.zeros((6, 3), dtype=np.float64)
    # A non-reference candidate has a finite, positive total advantage.  The
    # reference row remains structurally zero, as it must in deployed CARE.
    quantiles[1, 2] = np.asarray((0.05, 0.10, 0.15, 0.20, 0.25))
    target[1, 2] = 0.10
    return {
        "family_index": family,
        "fold": fold,
        "fit_families": fit,
        "task_id": task,
        "quantiles": quantiles,
        "target": target,
        "hard_safety_logit": np.full(6, -20.0),
        "hard_safety": np.zeros(6),
        "candidate_legality": np.ones(6, dtype=bool),
    }


def test_family_folds_are_reproducible_and_task_stratified() -> None:
    task_ids = [0] * 6 + [1] * 6
    snapshots = [f"s-{value}" for value in range(12)]
    first = make_family_folds(task_ids, snapshots, n_splits=3, seed=17)
    shuffled = [9, 1, 7, 0, 11, 4, 3, 8, 2, 10, 5, 6]
    second = make_family_folds(
        [task_ids[index] for index in shuffled],
        [snapshots[index] for index in shuffled],
        n_splits=3,
        seed=17,
    )
    # The returned keys are local family indices; compare by immutable id.
    first_by_snapshot = {snapshots[index]: fold for index, fold in first.items()}
    second_by_snapshot = {
        snapshots[shuffled[index]]: fold for index, fold in second.items()
    }
    assert first_by_snapshot == second_by_snapshot
    for task in (0, 1):
        counts = [
            sum(first[index] == fold for index, value in enumerate(task_ids) if value == task)
            for fold in range(3)
        ]
        assert counts == [2, 2, 2]


def test_validate_oof_rows_rejects_any_held_out_family_in_fit_set() -> None:
    assignments = {0: 0, 1: 1, 2: 0, 3: 1}
    # Family 0 is held out in fold 0; family 2 is a sibling in that same fold.
    # Including sibling 2 in the model fit is leakage even though row 0 itself
    # is not listed in fit_families.
    rows = [_row(0, 0, [1, 2, 3])]
    with pytest.raises(ValueError, match="OOF leakage"):
        validate_oof_rows(rows, assignments, require_complete_family_coverage=False)


def test_oof_calibration_uses_other_folds_and_reports_crossfit_coverage() -> None:
    assignments = {0: 0, 1: 1, 2: 0, 3: 1}
    rows = []
    for family, fold in assignments.items():
        held_out = [key for key, value in assignments.items() if value == fold]
        fit = [key for key in assignments if key not in held_out]
        rows.append(_row(family, fold, fit, task=family // 2))
    calibration = fit_oof_calibration(rows, assignments, nominal=0.90)
    assert calibration.family_count == 4
    assert set(calibration.fold_corrections) == {"0", "1"}
    assert 0.0 <= calibration.crossfit_family_coverage <= 1.0
    # The global correction is a diagnostic artifact; the OOF report keeps the
    # per-fold corrections needed to audit that no family calibrated itself.
    assert calibration.to_dict()["fold_corrections"]
    assert calibration.to_dict()["task_lower_corrections"]
    assert calibration.to_dict()["crossfit_correction_by_family"]
    scored = oof_metrics(
        rows,
        family_corrections=calibration.crossfit_correction_by_family,
    )
    assert scored["families"] == 4
    assert scored["safety_gate_mode"] == "legality_only"
    assert scored["family_crossfit_corrections_applied"] is True


def test_selector_metrics_mask_illegal_top_candidate_before_argmax() -> None:
    row = _row(0, 0, [])
    quantiles = np.asarray(row["quantiles"])
    target = np.asarray(row["target"])
    quantiles[1, 2] = np.asarray((0.9, 0.9, 0.9, 0.9, 0.9))
    quantiles[2, 2] = np.asarray((0.4, 0.4, 0.4, 0.4, 0.4))
    target[1, 2] = -1.0
    target[2, 2] = 0.5
    row["quantiles"] = quantiles
    row["target"] = target
    legality = np.ones(6, dtype=bool)
    legality[1] = False
    row["candidate_legality"] = legality
    companion = _row(1, 1, [0])
    scored = oof_metrics([row, companion], correction=0.0)
    assert scored["selector_selected_candidate_counts"]["2"] == 1
    assert scored["selector_mean_regret"] == 0.0
    assert scored["illegal_candidate_rate"] == 0.1


def test_admission_gate_is_fail_closed_and_requires_both_closed_loop_claims() -> None:
    rejected = admission_gate({}, require_closed_loop=True)
    assert rejected["status"] == "REJECTED"
    assert "contract.candidate0_bit_exact" in rejected["failed_checks"]

    complete = {
        "contract": {
            "candidate0_bit_exact": True,
            "strict_local": True,
            "normalization_action_decode_parity": True,
            "snapshot_restore_parity": True,
            "branch_horizon_complete": True,
            "candidate_legality_rate": 1.0,
        },
        "provenance": {
            "family_disjoint": True,
            "calibration_independent": True,
            "no_validation20_tuning": True,
        },
        "metrics": {
            "pairwise_accuracy_including_reference": 0.70,
            "top1_accuracy": 0.45,
            "candidate_vs_reference_sign_accuracy": 0.70,
            "beneficial_override_rate": 0.10,
            "harmful_override_rate": 0.01,
            "safety_gate_mode": "legality_only",
            "safety_positive_label_count": 0,
        },
        "crossfit_family_coverage": 0.95,
        "care_minus_selector_off": 0.06,
        "care_minus_act": 0.07,
        "paired_bootstrap_lower_95": 0.01,
        "care_vs_act_bootstrap_lower_95": 0.02,
    }
    accepted = admission_gate(complete, thresholds=AdmissionThresholds())
    assert accepted["status"] == "ADMITTED"
    assert accepted["failed_checks"] == []
