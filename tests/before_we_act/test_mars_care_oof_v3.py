from __future__ import annotations

import json
import hashlib
from types import SimpleNamespace

import pytest
import torch

from before_we_act.care_belief import CARE_HORIZONS
from scripts.before_we_act.run_mars_care_oof_v3 import (
    FoldTrainingConfig,
    _horizon_family_maxima,
    _json_value,
    PREDICTION_FORMAT_VERSION,
    aggregate_jobs,
    aggregate_repeat_rows,
    ensemble_seed_rows,
    family_folds,
    fit_oof_horizon_calibration,
    _read_prediction_rows,
    validate_oof_horizon_rows,
)


def test_oof_training_config_accepts_registered_h8_prefix() -> None:
    config = FoldTrainingConfig(action_prefix_steps=8)
    assert config.action_prefix_steps == 8


def test_oof_training_config_rejects_unregistered_prefix() -> None:
    with pytest.raises(ValueError, match="registered intervention"):
        FoldTrainingConfig(action_prefix_steps=3)


def repeat_row(
    family: int,
    task: int,
    target_offset: float,
    *,
    horizon_index: int = 1,
) -> dict[str, object]:
    quantiles = torch.zeros(6, 3, 5)
    quantiles[1:, 2] = 0.25
    target = torch.zeros(6, 3)
    target[1:, 2] = target_offset
    return {
        "family_index": family,
        "task_id": task,
        "horizon_index": horizon_index,
        "quantiles": quantiles,
        "hard_safety_logit": torch.arange(6).float(),
        "target": target,
        "hard_safety": torch.zeros(6),
    }


def test_repeat_aggregation_keeps_family_level_provenance_and_worst_safety() -> None:
    first = repeat_row(2, 1, 1.0)
    second = repeat_row(2, 1, 3.0)
    second["hard_safety"] = torch.tensor([0, 0, 1, 0, 0, 0]).float()
    rows = aggregate_repeat_rows(
        [first, second],
        fold=3,
        fit_families=[0, 1, 4],
        snapshot_ids=["a", "b", "family-two", "d", "e"],
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["family_index"] == 2
    assert row["fold"] == 3
    assert row["fit_families"] == [0, 1, 4]
    assert row["snapshot_id"] == "family-two"
    torch.testing.assert_close(row["target"][1:, 2], torch.full((5,), 2.0))
    assert row["hard_safety"][2] == 1
    assert torch.equal(row["quantiles"][0], torch.zeros(3, 5))


def test_seed_ensemble_averages_only_predictions_after_target_parity_check() -> None:
    base = aggregate_repeat_rows(
        [repeat_row(0, 0, 1.0), repeat_row(0, 0, 1.0)],
        fold=0,
        fit_families=[1, 2],
        snapshot_ids=["zero", "one", "two"],
    )[0]
    other = dict(base)
    other["quantiles"] = torch.as_tensor(base["quantiles"]) + 2.0
    other["quantiles"][0] = 0.0
    other["hard_safety_logit"] = torch.as_tensor(base["hard_safety_logit"]) + 4.0
    result = ensemble_seed_rows([[base], [other]], expected_seeds=[11, 12])
    assert len(result) == 1
    assert result[0]["ensemble_seeds"] == [11, 12]
    expected = torch.ones(5, 3, 5)
    expected[:, 2] = 1.25
    torch.testing.assert_close(result[0]["quantiles"][1:], expected)
    torch.testing.assert_close(
        result[0]["hard_safety_logit"],
        torch.as_tensor(base["hard_safety_logit"]) + 2.0,
    )
    assert torch.equal(result[0]["target"], base["target"])


def _strict_horizon_row(
    family: int,
    fold: int,
    fit_families: list[int],
    horizon_index: int,
    *,
    normalized_error: float,
    task: int = 0,
    scale: float = 1.0,
) -> dict[str, object]:
    row = repeat_row(
        family,
        task,
        0.0,
        horizon_index=horizon_index,
    )
    quantiles = torch.as_tensor(row["quantiles"]).clone()
    quantiles[1:, 2, :] = float(normalized_error) * float(scale)
    row.update(
        {
            "snapshot_id": f"family-{family}",
            "fold": fold,
            "fit_families": fit_families,
            "horizon_steps": CARE_HORIZONS[horizon_index],
            "total_utility_scale": float(scale),
            "quantiles": quantiles,
            "candidate_legality": torch.ones(6, dtype=torch.bool),
        }
    )
    return row


def _strict_horizon_rows(
    assignments: dict[int, int],
    errors: dict[int, float],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for family, fold in assignments.items():
        heldout = {
            other for other, other_fold in assignments.items() if other_fold == fold
        }
        fit = sorted(set(assignments).difference(heldout))
        for horizon_index in range(len(CARE_HORIZONS)):
            result.append(
                _strict_horizon_row(
                    family,
                    fold,
                    fit,
                    horizon_index,
                    normalized_error=errors[family],
                )
            )
    return result


def test_repeat_aggregation_never_averages_different_horizons() -> None:
    repeats: list[dict[str, object]] = []
    for horizon_index in range(len(CARE_HORIZONS)):
        repeats.extend(
            [
                repeat_row(
                    0, 0, float(horizon_index + 1), horizon_index=horizon_index
                ),
                repeat_row(
                    0, 0, float(horizon_index + 1), horizon_index=horizon_index
                ),
            ]
        )
    scales = torch.ones(1, len(CARE_HORIZONS), 3)
    rows = aggregate_repeat_rows(
        repeats,
        fold=0,
        fit_families=[1],
        snapshot_ids=["zero", "one"],
        task_horizon_component_scales=scales,
    )
    assert len(rows) == len(CARE_HORIZONS)
    assert [row["horizon_steps"] for row in rows] == list(CARE_HORIZONS)
    for horizon_index, row in enumerate(rows):
        assert row["horizon_index"] == horizon_index
        torch.testing.assert_close(
            row["target"][1:, 2],
            torch.full((5,), float(horizon_index + 1)),
        )


def test_all_horizon_validation_fails_closed_when_one_horizon_is_missing() -> None:
    assignments = {0: 0, 1: 1, 2: 0, 3: 1}
    rows = _strict_horizon_rows(assignments, {family: 0.1 for family in assignments})
    rows = [
        row
        for row in rows
        if not (row["family_index"] == 0 and row["horizon_index"] == 3)
    ]
    with pytest.raises(ValueError, match="omit horizons"):
        validate_oof_horizon_rows(rows, assignments)


def test_family_calibration_maximum_is_dominated_by_worst_horizon() -> None:
    values = (0.1, 0.2, 1.75, 0.4)
    rows = [
        _strict_horizon_row(
            0,
            0,
            [1],
            horizon_index,
            normalized_error=value,
        )
        for horizon_index, value in enumerate(values)
    ]
    maximum = _horizon_family_maxima(rows)[0]
    assert maximum["maximum"] == pytest.approx(1.75)
    assert maximum["horizons"]["2"] == pytest.approx(1.75)


def test_family_calibration_excludes_illegal_candidates_from_maximum() -> None:
    rows = [
        _strict_horizon_row(
            0,
            0,
            [1],
            horizon_index,
            normalized_error=0.1,
        )
        for horizon_index in range(len(CARE_HORIZONS))
    ]
    for row in rows:
        quantiles = torch.as_tensor(row["quantiles"]).clone()
        quantiles[5, 2, 0] = 100.0
        row["quantiles"] = quantiles
        row["candidate_legality"] = torch.tensor([True, True, True, True, True, False])
    maximum = _horizon_family_maxima(rows)[0]
    assert maximum["maximum"] == pytest.approx(0.1)


def _prediction_payload(checkpoint: Path) -> dict[str, object]:
    return {
        "format_version": PREDICTION_FORMAT_VERSION,
        "prepared_data_sha256": "prepared-hash",
        "fold": 0,
        "seed": 11,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "rows": [],
    }


@pytest.mark.parametrize(
    ("field", "wrong_value", "message"),
    (
        ("prepared_data_sha256", "wrong", "prepared-data hash mismatch"),
        ("fold", 1, "top-level fold mismatch"),
        ("seed", 12, "top-level seed mismatch"),
        ("checkpoint", "/tmp/wrong-checkpoint.pt", "checkpoint path mismatch"),
        ("checkpoint_sha256", "0" * 64, "checkpoint hash mismatch"),
    ),
)
def test_prediction_reader_rejects_top_level_provenance_mismatch(
    tmp_path, field, wrong_value, message
) -> None:
    path = tmp_path / "predictions.json"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    payload = _prediction_payload(checkpoint)
    payload[field] = wrong_value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        _read_prediction_rows(
            path,
            expected_prepared_data_sha256="prepared-hash",
            expected_fold=0,
            expected_seed=11,
        )


def test_horizon_crossfit_correction_never_uses_its_own_fold() -> None:
    assignments = {0: 0, 1: 1, 2: 0, 3: 1}
    rows = _strict_horizon_rows(
        assignments,
        {
            0: 100.0,
            1: 1.0,
            2: 50.0,
            3: 0.5,
        },
    )
    calibration = fit_oof_horizon_calibration(rows, assignments, nominal=0.90)
    # Fold zero's extreme values cannot influence its own correction; only
    # fold one's family maxima (1.0 and 0.5) are available to fit it.
    assert calibration["crossfit_fold_corrections"]["0"] == pytest.approx(1.0)
    assert calibration["crossfit_correction_by_family"]["0"] == pytest.approx(1.0)
    assert calibration["crossfit_correction_by_family"]["2"] == pytest.approx(1.0)
    # Conversely, fold one is evaluated against the maximum from fold zero.
    assert calibration["crossfit_fold_corrections"]["1"] == pytest.approx(100.0)


def test_seed_ensemble_requires_identical_family_horizon_coverage() -> None:
    assignments = {0: 0, 1: 1, 2: 0, 3: 1}
    complete = _strict_horizon_rows(
        assignments, {family: 0.1 for family in assignments}
    )
    missing_one_horizon = complete[:-1]
    with pytest.raises(ValueError, match="family/horizon coverage differs"):
        ensemble_seed_rows(
            [complete, missing_one_horizon],
            expected_seeds=[11, 12],
        )


def test_aggregate_payload_makes_all_horizon_calibration_canonical(tmp_path) -> None:
    prepared = SimpleNamespace(
        task_id=torch.zeros(4, dtype=torch.long),
        snapshot_ids=tuple(f"family-{family}" for family in range(4)),
    )
    fold_seed = 17
    assignments = family_folds(prepared, n_splits=2, seed=fold_seed)
    rows = _strict_horizon_rows(
        assignments, {family: 0.1 for family in assignments}
    )
    # H16 alone looks easy, while H64 is deliberately the simultaneous
    # calibration bottleneck.  The canonical payload must retain H64's 2.0
    # correction instead of exposing the optimistic H16-only value as if it
    # were the deployment contract.
    for row in rows:
        if row["horizon_index"] == 3:
            quantiles = torch.as_tensor(row["quantiles"]).clone()
            quantiles[1:, 2, :] = 2.0
            row["quantiles"] = quantiles
    seeds = (11, 12)
    for seed in seeds:
        for fold in range(2):
            output = tmp_path / f"fold_{fold}" / f"seed_{seed}"
            output.mkdir(parents=True)
            fold_rows = [row for row in rows if row["fold"] == fold]
            checkpoint = output / "checkpoint.pt"
            checkpoint.write_bytes(b"test-checkpoint")
            (output / "predictions.json").write_text(
                json.dumps(
                    _json_value(
                        {
                            "format_version": PREDICTION_FORMAT_VERSION,
                            "prepared_data_sha256": "prepared-data-hash",
                            "fold": fold,
                            "seed": seed,
                            "checkpoint": str(checkpoint.resolve()),
                            "checkpoint_sha256": hashlib.sha256(
                                checkpoint.read_bytes()
                            ).hexdigest(),
                            "rows": fold_rows,
                        }
                    )
                ),
                encoding="utf-8",
            )
    payload = aggregate_jobs(
        prepared,
        tmp_path,
        seeds=seeds,
        n_splits=2,
        fold_seed=fold_seed,
        nominal=0.90,
        prepared_data_sha256="prepared-data-hash",
    )
    assert len(payload["rows"]) == 4
    assert len(payload["horizon_rows"]) == 4 * len(CARE_HORIZONS)
    assert payload["calibration"] == payload["horizon_calibration"]
    assert payload["calibration"]["normalized_familywise_correction"] == pytest.approx(
        2.0
    )
    assert payload["legacy_primary_horizon_calibration"][
        "lower_correction"
    ] == pytest.approx(0.1)
