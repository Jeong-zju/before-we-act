from __future__ import annotations

from pathlib import Path

import pytest
import torch

from before_we_act.care_belief_v3 import CAREBeliefV3Config, CAREBeliefV3Head
from before_we_act.care_training_data import sha256_file
from before_we_act.mars_care_v3_deployment import (
    DEPLOYMENT_FORMAT_VERSION,
    FINAL_TRAINING_FORMAT_VERSION,
    OOF_FORMAT_VERSION,
    TASK_NAMES,
    validate_v3_deployment_payload,
)
from before_we_act.care_chunk_commitment import (
    advance_chunk_commitments,
    apply_chunk_commitments,
)


def _payload(reference: Path) -> dict[str, object]:
    config = CAREBeliefV3Config(
        d_model=32,
        action_horizon=100,
        action_tokens=4,
        action_width=16,
        heads=4,
        layers=1,
        dropout=0.0,
        action_prefix_steps=8,
        task_count=4,
    )
    model = CAREBeliefV3Head(config)
    return {
        "format_version": DEPLOYMENT_FORMAT_VERSION,
        "source_training_format_version": FINAL_TRAINING_FORMAT_VERSION,
        "reference_checkpoint_sha256": sha256_file(reference),
        "task_names": list(TASK_NAMES),
        "config": config.to_dict(),
        "model": model.state_dict(),
        "intervention_steps": 8,
        "calibration": {
            "lower_correction": 0.1,
            "selector_delta": 0.0,
            "hard_safety_probability_max": 0.25,
            "nominal_simultaneous_coverage": 0.9,
            "primary_horizon": 16,
        },
        "task_horizon_component_scales": torch.ones(4, 4, 3),
        "task_horizon_lower_corrections": torch.ones(4, 4) * 0.1,
        "safety_gate_mode": "legality_only",
        "safety_positive_label_count": 0,
        "provenance": {
            "oof_format_version": OOF_FORMAT_VERSION,
            "admission_passed": True,
            "family_disjoint": True,
            "calibration_independent": True,
            "no_validation20_tuning": True,
            "physical_unit_runtime_parity": True,
            "horizon_oof_complete": True,
            "promotion_scope": "smoke",
        },
    }


def test_v3_payload_accepts_h8_and_all_horizon_calibration(tmp_path: Path) -> None:
    reference = tmp_path / "reference.pt"
    reference.write_bytes(b"reference")
    config, calibration, scales, corrections, mode = validate_v3_deployment_payload(
        _payload(reference), reference
    )
    assert config.action_prefix_steps == 8
    assert calibration.primary_horizon == 16
    assert scales.shape == (4, 4, 3)
    assert corrections.shape == (4, 4)
    assert mode == "legality_only"


def test_v3_payload_rejects_prefix_and_oof_drift(tmp_path: Path) -> None:
    reference = tmp_path / "reference.pt"
    reference.write_bytes(b"reference")
    payload = _payload(reference)
    payload["intervention_steps"] = 1
    with pytest.raises(ValueError, match="intervention"):
        validate_v3_deployment_payload(payload, reference)
    payload = _payload(reference)
    payload["provenance"] = dict(payload["provenance"])
    payload["provenance"]["horizon_oof_complete"] = False
    with pytest.raises(ValueError, match="horizon_oof_complete"):
        validate_v3_deployment_payload(payload, reference)


def test_h8_commitment_consumes_rows_without_repeating_row_zero() -> None:
    reference = torch.zeros(100, 8).numpy()
    plan = torch.arange(800, dtype=torch.float32).reshape(100, 8).numpy()
    candidates = [torch.stack((torch.zeros(100, 8), torch.from_numpy(plan))).numpy()]
    selected = [1]
    lower = [0.4]
    commitments = {0: {"candidate_id": 1, "plan": plan.copy(), "next_step": 0, "best_lower": 0.4}}
    active = apply_chunk_commitments(candidates, selected, lower, commitments, intervention_steps=8)
    assert active == {0}
    assert np_equal(candidates[0][1, 0], plan[0])
    decisions, committed = advance_chunk_commitments(
        candidates, selected, lower, [0], commitments, active, intervention_steps=8
    )
    assert decisions == 0 and committed == 1 and commitments[0]["next_step"] == 1
    candidates[0][1, 0] = plan[1]
    assert np_equal(candidates[0][1, 0], plan[1])


def np_equal(left: object, right: object) -> bool:
    import numpy as np

    return bool(np.array_equal(left, right))
