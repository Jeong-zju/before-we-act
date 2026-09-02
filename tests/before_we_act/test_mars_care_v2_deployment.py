from __future__ import annotations

from pathlib import Path

import pytest
import torch

from before_we_act.care_belief import CARECalibration
from before_we_act.care_belief_v2 import CAREBeliefV2Config, CAREBeliefV2Head
from before_we_act.care_training_data import sha256_file
from before_we_act.mars_care_v2_deployment import (
    DEPLOYMENT_FORMAT_VERSION,
    OOF_FORMAT_VERSION,
    TASK_NAMES,
    TRAINING_FORMAT_VERSION,
    validate_v2_deployment_payload,
)
from before_we_act.evaluate_mars_care_closed_loop_v2 import assemble_actions_v2


def _payload(reference: Path) -> dict[str, object]:
    config = CAREBeliefV2Config(
        d_model=32,
        action_horizon=100,
        action_tokens=4,
        action_width=16,
        heads=4,
        layers=1,
        dropout=0.0,
        action_prefix_steps=1,
    )
    model = CAREBeliefV2Head(config)
    return {
        "format_version": DEPLOYMENT_FORMAT_VERSION,
        "source_training_format_version": TRAINING_FORMAT_VERSION,
        "reference_checkpoint_sha256": sha256_file(reference),
        "task_names": list(TASK_NAMES),
        "config": config.to_dict(),
        "prepared_intervention_steps": 1,
        "calibration": {
            "lower_correction": 0.01,
            "selector_delta": 0.0,
            "hard_safety_probability_max": 0.25,
            "nominal_simultaneous_coverage": 0.9,
            "primary_horizon": 16,
        },
        "task_component_scales": [[1.0, 1.0, 1.0] for _ in TASK_NAMES],
        "safety_gate_mode": "legality_only",
        "safety_positive_label_count": 0,
        "provenance": {
            "oof_format_version": OOF_FORMAT_VERSION,
            "admission_passed": True,
            "family_disjoint": True,
            "calibration_independent": True,
            "no_validation20_tuning": True,
            "physical_unit_runtime_parity": True,
            "promotion_scope": "smoke",
        },
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }


def test_v2_deployment_payload_requires_oof_and_reference_hash(tmp_path: Path) -> None:
    reference = tmp_path / "reference.pt"
    reference.write_bytes(b"reference")
    payload = _payload(reference)
    config, calibration, scales, corrections, mode = validate_v2_deployment_payload(
        payload, reference
    )
    assert config.action_prefix_steps == 1
    assert calibration.primary_horizon == 16
    assert scales.shape == (len(TASK_NAMES), 3)
    assert corrections.shape == (len(TASK_NAMES),)
    assert mode == "legality_only"

    broken = dict(payload)
    broken["provenance"] = dict(payload["provenance"])
    broken["provenance"]["oof_format_version"] = "wrong"
    with pytest.raises(ValueError, match="OOF"):
        validate_v2_deployment_payload(broken, reference)


def test_v2_deployment_rejects_uncalibrated_learned_safety(tmp_path: Path) -> None:
    reference = tmp_path / "reference.pt"
    reference.write_bytes(b"reference")
    payload = _payload(reference)
    payload["safety_gate_mode"] = "learned_probability"
    payload["safety_positive_label_count"] = 3
    with pytest.raises(ValueError, match="calibration"):
        validate_v2_deployment_payload(payload, reference)
    payload["safety_threshold_calibrated"] = True
    config, *_ = validate_v2_deployment_payload(payload, reference)
    assert config.action_prefix_steps == 1


def test_v2_deployment_rejects_task_scale_and_prefix_drift(tmp_path: Path) -> None:
    reference = tmp_path / "reference.pt"
    reference.write_bytes(b"reference")
    payload = _payload(reference)
    payload["task_component_scales"] = [[1.0, 1.0, 1.0]]
    with pytest.raises(ValueError, match="scales"):
        validate_v2_deployment_payload(payload, reference)
    payload = _payload(reference)
    payload["prepared_intervention_steps"] = 4
    with pytest.raises(ValueError, match="prefix"):
        validate_v2_deployment_payload(payload, reference)


def test_action_assembly_keeps_each_arm_local_and_exposes_arbitration() -> None:
    reference = {
        "panda-0": torch.zeros(100, 8).numpy(),
        "panda-1": torch.ones(100, 8).numpy(),
    }
    candidates = [
        torch.stack((torch.zeros(100, 8), torch.full((100, 8), 2.0))).numpy(),
        torch.stack((torch.ones(100, 8), torch.full((100, 8), 3.0))).numpy(),
    ]
    decentralized, report = assemble_actions_v2(
        reference, candidates, [1, 1], [0.4, 0.3], [0, 1], mode="decentralized"
    )
    assert decentralized["panda-0"][0] == 2.0
    assert decentralized["panda-1"][0] == 3.0
    assert report["simultaneous_overrides"] == 2
    assert report["central_arbitration_suppressions"] == 0

    single, report = assemble_actions_v2(
        reference, candidates, [1, 1], [0.4, 0.3], [0, 1], mode="single_focal"
    )
    assert single["panda-0"][0] == 2.0
    assert single["panda-1"][0] == 1.0
    assert report["simultaneous_overrides"] == 1
    assert report["central_arbitration_suppressions"] == 1
