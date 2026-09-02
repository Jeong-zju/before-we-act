from __future__ import annotations

import pytest
import torch

from before_we_act.care_belief_v3 import (
    CAREBeliefV3Config,
    CAREBeliefV3Head,
    robust_task_horizon_component_scales,
)
from scripts.before_we_act.analyze_mars_care_scorer_v3 import (
    normalized_task_horizon_conformal_report,
    q05_family_horizon_residuals,
    q05_extreme_family_diagnostics,
    taskwise_conformal_report,
)


def small_config(**kwargs: object) -> CAREBeliefV3Config:
    values: dict[str, object] = {
        "d_model": 32,
        "action_horizon": 20,
        "action_tokens": 4,
        "action_width": 16,
        "heads": 4,
        "layers": 1,
        "dropout": 0.0,
        "action_prefix_steps": 1,
        "task_count": 4,
    }
    values.update(kwargs)
    return CAREBeliefV3Config(**values)


def inputs(config: CAREBeliefV3Config) -> tuple[torch.Tensor, ...]:
    memory = torch.randn(2, 7, config.d_model)
    mask = torch.ones(2, 7, dtype=torch.bool)
    chunks = torch.randn(2, 6, config.action_horizon, config.action_dim)
    horizon = torch.tensor([0, 1])
    task_id = torch.tensor([0, 3])
    return memory, mask, chunks, horizon, task_id


def test_v3_preserves_structural_reference_and_physical_unit_contract() -> None:
    torch.manual_seed(13)
    config = small_config()
    model = CAREBeliefV3Head(config).eval()
    memory, mask, chunks, horizon, task_id = inputs(config)
    scale = torch.tensor([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]])
    raw = model(memory, mask, chunks, horizon, task_id)
    physical = model(
        memory,
        mask,
        chunks,
        horizon,
        task_id,
        utility_scale=scale,
    )
    assert physical.quantiles.shape == (2, 6, 3, 5)
    assert physical.hard_safety_logit.shape == (2, 6)
    assert torch.equal(
        physical.quantiles[:, 0], torch.zeros_like(physical.quantiles[:, 0])
    )
    torch.testing.assert_close(
        physical.quantiles[:, 1:], raw.quantiles[:, 1:] * scale[:, None, :, None]
    )


def test_candidate_slot_identity_separates_identical_candidate_actions() -> None:
    torch.manual_seed(17)
    config = small_config(
        use_candidate_slot_embedding=True,
        use_task_embedding=False,
    )
    model = CAREBeliefV3Head(config).eval()
    memory, mask, chunks, horizon, task_id = inputs(config)
    chunks[:, 1:] = chunks[:, 1:2]
    output = model(memory, mask, chunks, horizon, task_id)
    # The action encoder receives identical deltas for slots 1..5.  Their
    # scorer states can still differ only because the slot identity is stable.
    assert not torch.allclose(
        output.candidate_state[:, 1], output.candidate_state[:, 2]
    )


def test_task_embedding_is_explicit_and_can_be_disabled_for_ablation() -> None:
    torch.manual_seed(19)
    enabled = CAREBeliefV3Head(
        small_config(use_task_embedding=True)
    ).eval()
    memory, mask, chunks, horizon, _task_id = inputs(enabled.config)
    first = enabled(memory, mask, chunks, horizon, torch.tensor([0, 0]))
    second = enabled(memory, mask, chunks, horizon, torch.tensor([1, 1]))
    assert not torch.allclose(first.candidate_state, second.candidate_state)

    torch.manual_seed(19)
    disabled = CAREBeliefV3Head(
        small_config(use_task_embedding=False)
    ).eval()
    first_disabled = disabled(
        memory, mask, chunks, horizon, torch.tensor([0, 0])
    )
    second_disabled = disabled(
        memory, mask, chunks, horizon, torch.tensor([1, 1])
    )
    torch.testing.assert_close(
        first_disabled.candidate_state, second_disabled.candidate_state
    )


def test_v3_rejects_missing_or_out_of_range_task_identity() -> None:
    config = small_config()
    model = CAREBeliefV3Head(config).eval()
    memory, mask, chunks, horizon, _task_id = inputs(config)
    with pytest.raises(ValueError, match="task id"):
        model(memory, mask, chunks, horizon, torch.tensor([0]))
    with pytest.raises(ValueError, match="task id"):
        model(memory, mask, chunks, horizon, torch.tensor([0, 4]))


def _row(
    family: int,
    task: int,
    lower_error: float,
    *,
    horizon_index: int = 0,
) -> dict[str, object]:
    quantiles = torch.zeros(6, 3, 5)
    target = torch.zeros(6, 3)
    quantiles[1, 2, 0] = float(lower_error)
    return {
        "family_index": family,
        "task_id": task,
        "horizon_index": horizon_index,
        "quantiles": quantiles,
        "target": target,
        "hard_safety_logit": torch.full((6,), -20.0),
        "hard_safety": torch.zeros(6),
    }


def test_taskwise_conformal_report_keeps_nominal_familywise_coverage() -> None:
    calibration = [
        _row(0, 0, 0.10),
        _row(1, 0, 0.20),
        _row(2, 1, 0.30),
        _row(3, 1, 0.40),
    ]
    validation = [
        _row(4, 0, 0.15),
        _row(5, 1, 0.35),
    ]
    report = taskwise_conformal_report(
        calibration,
        validation,
        nominal=0.90,
    )
    assert report["nominal_simultaneous_coverage"] == 0.90
    assert report["familywise_within_task"] is True
    assert report["admission_thresholds_unchanged"] is True
    assert report["correction_by_task_id"] == pytest.approx(
        {"0": 0.20, "1": 0.40}
    )
    assert report["validation_family_coverage_by_task_id"] == {
        "0": 1.0,
        "1": 1.0,
    }


def test_horizon_aware_scale_does_not_mix_short_and_long_outcome_units() -> None:
    targets = torch.zeros(4, 2, 6, 2, 3)
    usable = torch.ones(4, 2, dtype=torch.bool)
    task_id = torch.tensor([0, 0, 1, 1])
    # Candidate zero remains a structural zero.  Each task/horizon cell has a
    # deliberately distinct physical unit for the five learned candidates.
    targets[task_id == 0, 0, 1:] = 1.0
    targets[task_id == 0, 1, 1:] = 10.0
    targets[task_id == 1, 0, 1:] = 100.0
    targets[task_id == 1, 1, 1:] = 1000.0
    scales = robust_task_horizon_component_scales(
        targets,
        usable,
        task_id,
        quantile=0.90,
        floor=1e-4,
    )
    assert scales.shape == (2, 2, 3)
    torch.testing.assert_close(
        scales[:, :, 0], torch.tensor([[1.0, 10.0], [100.0, 1000.0]])
    )
    assert torch.all(scales > 0)


def test_q05_extreme_family_report_names_outlier_without_trimming_it() -> None:
    rows = [
        _row(0, 0, 0.10),
        _row(1, 0, 0.20),
        _row(2, 1, 7.00),
    ]
    report = q05_extreme_family_diagnostics(
        rows,
        snapshot_ids=("normal-a", "normal-b", "extreme-c"),
        nominal=0.90,
        top_k=2,
    )
    assert report["full_familywise_correction"] == 7.0
    assert report["correction_used_without_trimming"] is True
    assert report["top_families"][0]["family_index"] == 2
    assert report["top_families"][0]["snapshot_id"] == "extreme-c"
    assert report["top_families"][0]["q05_overestimate"] == 7.0
    assert report["leave_one_extreme_out_diagnostic_only"][
        "correction"
    ] == pytest.approx(0.20)


def test_normalized_task_horizon_conformal_keeps_full_family_max_and_nominal() -> None:
    calibration = [
        _row(0, 0, 0.10, horizon_index=0),
        _row(0, 0, 2.00, horizon_index=1),
        _row(1, 0, 0.20, horizon_index=0),
        _row(1, 0, 1.00, horizon_index=1),
        _row(2, 1, 3.00, horizon_index=0),
        _row(2, 1, 8.00, horizon_index=1),
        _row(3, 1, 4.00, horizon_index=0),
        _row(3, 1, 6.00, horizon_index=1),
    ]
    validation = [
        _row(4, 0, 0.15, horizon_index=0),
        _row(4, 0, 1.50, horizon_index=1),
        _row(5, 1, 3.50, horizon_index=0),
        _row(5, 1, 7.00, horizon_index=1),
    ]
    # Only the total-utility component (index 2) is used by the CARE selector.
    scales = torch.ones(2, 2, 3)
    scales[0, 0, 2] = 1.0
    scales[0, 1, 2] = 5.0
    scales[1, 0, 2] = 10.0
    scales[1, 1, 2] = 20.0
    report = normalized_task_horizon_conformal_report(
        calibration,
        validation,
        task_horizon_component_scales=scales,
        horizons=(8, 16),
        nominal=0.90,
    )
    # Family 3 contributes max(4/10, 6/20)=0.4.  Finite-sample 90%
    # calibration over four complete families therefore keeps 0.4.
    assert report["normalized_familywise_correction"] == pytest.approx(0.4)
    assert report["validation_family_coverage"] == 1.0
    assert report["nominal_simultaneous_coverage"] == 0.90
    assert report["family_max_includes_all_requested_horizons"] is True
    assert report["scale_fitted_on_training_families_only"] is True
    assert report["eligible_for_admission_or_deployment"] is False
    assert report["physical_correction_by_task_horizon"]["0"]["8"] == pytest.approx(0.4)
    assert report["physical_correction_by_task_horizon"]["1"]["16"] == pytest.approx(8.0)
    # The raw all-family/all-horizon scalar correction is 8.0.  Normalized
    # decoding is more efficient for small-unit cells without dropping rows.
    assert report["raw_familywise_physical_correction"] == pytest.approx(8.0)
    assert report["physical_efficiency_ratio_vs_raw"]["0"]["8"] == pytest.approx(0.05)


def test_per_family_horizon_q05_residuals_record_candidate_and_units() -> None:
    rows = [
        _row(0, 0, 0.10, horizon_index=0),
        _row(0, 0, 2.00, horizon_index=1),
        _row(1, 1, 3.00, horizon_index=0),
    ]
    scales = torch.ones(2, 2, 3)
    scales[0, 1, 2] = 5.0
    scales[1, 0, 2] = 10.0
    residuals = q05_family_horizon_residuals(
        rows,
        task_horizon_component_scales=scales,
        snapshot_ids=("family-zero", "family-one"),
        horizons=(8, 16),
    )
    assert [(row["family_index"], row["horizon_steps"]) for row in residuals] == [
        (0, 8),
        (0, 16),
        (1, 8),
    ]
    assert residuals[1]["q05_overestimate_physical"] == pytest.approx(2.0)
    assert residuals[1]["q05_overestimate_normalized"] == pytest.approx(0.4)
    assert residuals[1]["worst_candidate_id"] == 1
    assert residuals[1]["snapshot_id"] == "family-zero"


def test_v3_config_rejects_invalid_task_contract() -> None:
    with pytest.raises(ValueError, match="task count"):
        small_config(task_count=0)
