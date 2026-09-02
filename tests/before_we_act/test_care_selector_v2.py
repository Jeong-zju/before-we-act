from __future__ import annotations

import pytest
import torch

from before_we_act.care_belief import CAREBeliefOutput, CARECalibration
from before_we_act.care_selector_v2 import (
    REASON_NO_ELIGIBLE_NONREFERENCE,
    REASON_OVERRIDE,
    REASON_REFERENCE_BELOW_DELTA,
    REASON_SELECTOR_DISABLED,
    select_care_candidate_v2,
)


def _calibration(**kwargs: object) -> CARECalibration:
    values: dict[str, object] = {
        "lower_correction": 0.0,
        "selector_delta": 0.0,
        "hard_safety_probability_max": 0.25,
        "nominal_simultaneous_coverage": 0.9,
        "primary_horizon": 16,
    }
    values.update(kwargs)
    return CARECalibration(**values)


def _output(lower: list[list[float]], safety: float = -20.0) -> CAREBeliefOutput:
    values = torch.as_tensor(lower, dtype=torch.float32)
    quantiles = torch.zeros(values.shape[0], values.shape[1], 3, 5)
    quantiles[:, :, 2, 0] = values
    # Keep candidate zero structurally exact, matching the CARE head contract.
    quantiles[:, 0] = 0.0
    return CAREBeliefOutput(
        quantiles=quantiles,
        hard_safety_logit=torch.full(values.shape, safety),
        candidate_state=torch.empty(values.shape[0], values.shape[1], 0),
    )


def test_legality_is_applied_before_argmax_instead_of_posthoc_fallback() -> None:
    output = _output([[0.0, 0.9, 0.7, -0.1, -0.2, -0.3]])
    legality = torch.tensor([[True, False, True, True, True, True]])
    result = select_care_candidate_v2(
        output,
        _calibration(),
        legality,
        safety_gate_mode="legality_only",
    )
    assert result.selected.tolist() == [2]
    assert result.reason_code.tolist() == [REASON_OVERRIDE]
    assert result.rejected_illegal_count.tolist() == [1]
    assert torch.isneginf(result.masked_lower[0, 1])


def test_degenerate_safety_ignores_random_head_and_uses_legality_only() -> None:
    output = _output([[0.0, 0.4, 0.3, 0.2, 0.1, -0.1]], safety=20.0)
    legality = torch.ones(1, 6, dtype=torch.bool)
    result = select_care_candidate_v2(
        output,
        _calibration(),
        legality,
        safety_gate_mode="legality_only",
    )
    assert result.selected.tolist() == [1]
    assert result.rejected_safety_count.tolist() == [0]

    learned = select_care_candidate_v2(
        output,
        _calibration(),
        legality,
        safety_gate_mode="learned_probability",
    )
    assert learned.selected.tolist() == [0]
    assert learned.reason_code.tolist() == [REASON_NO_ELIGIBLE_NONREFERENCE]
    assert learned.rejected_safety_count.tolist() == [5]


def test_task_conditioned_corrections_change_only_the_corresponding_row() -> None:
    output = _output(
        [
            [0.0, 0.06, -0.1, -0.1, -0.1, -0.1],
            [0.0, 0.06, -0.1, -0.1, -0.1, -0.1],
        ]
    )
    result = select_care_candidate_v2(
        output,
        _calibration(),
        torch.ones(2, 6, dtype=torch.bool),
        safety_gate_mode="legality_only",
        lower_correction=torch.tensor([0.01, 0.10]),
    )
    assert result.selected.tolist() == [1, 0]
    assert result.reason_code.tolist() == [
        REASON_OVERRIDE,
        REASON_REFERENCE_BELOW_DELTA,
    ]


def test_reference_is_fail_closed_and_selector_off_is_explicit() -> None:
    output = _output([[0.0, 1.0, 0.5, 0.4, 0.3, 0.2]], safety=20.0)
    result = select_care_candidate_v2(
        output,
        _calibration(),
        torch.ones(1, 6, dtype=torch.bool),
        safety_gate_mode="learned_probability",
        selector_enabled=False,
    )
    assert result.selected.tolist() == [0]
    assert result.masked_lower[0, 0] == 0.0
    assert result.reason_code.tolist() == [REASON_SELECTOR_DISABLED]


def test_selector_rejects_illegal_reference_and_invalid_calibration() -> None:
    output = _output([[0.0] * 6])
    legality = torch.ones(1, 6, dtype=torch.bool)
    legality[:, 0] = False
    with pytest.raises(ValueError, match="reference candidate"):
        select_care_candidate_v2(
            output,
            _calibration(),
            legality,
            safety_gate_mode="legality_only",
        )
    with pytest.raises(ValueError, match="correction"):
        select_care_candidate_v2(
            output,
            _calibration(),
            torch.ones(1, 6, dtype=torch.bool),
            safety_gate_mode="legality_only",
            lower_correction=float("nan"),
        )
