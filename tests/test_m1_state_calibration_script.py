from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.calibrate_m1_state_pairs import (
    ROOT,
    _allowed_parameter_changes,
    _state_only_stage,
    _training_scope_evidence,
    _validate_controls,
    _validate_diagnostic_output_paths,
    build_parser,
)


def _config() -> dict:
    path = ROOT / "configs/wam_multimodal/m1_latent_wam.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_state_calibration_cli_and_stage_are_low_dose_flow_only() -> None:
    args = build_parser().parse_args(
        [
            "--input-checkpoint",
            "/tmp/parent",
            "--output-checkpoint-root",
            "/tmp/state-checkpoints",
            "--output-root",
            "/tmp/state-report",
            "--steps",
            "16",
            "--learning-rate",
            "0.00003",
        ]
    )
    assert args.steps == 16
    assert args.learning_rate == pytest.approx(3e-5)
    stage = _state_only_stage(_config(), steps=args.steps, learning_rate=args.learning_rate)
    assert stage.train_action_flow
    assert not stage.train_visual_adapter
    assert not stage.train_fusion
    assert not stage.train_future_head
    assert not stage.train_world_model


def test_state_calibration_rejects_rgb_scope_and_forbidden_changes() -> None:
    clean_training = {
        "causal_pairs_enabled": False,
        "state_causal_pairs_enabled": True,
        "state_causal_pair_gradient_scope": "non_anchor_flow_only_model_frozen",
        "optimizer_groups": [
            {"role": "adapter_action", "lr": 3e-5, "parameters": 123}
        ],
    }
    assert _training_scope_evidence(clean_training)["passed"]
    assert _allowed_parameter_changes(["flow.velocity.0.weight"])["passed"]

    rgb_training = dict(clean_training, causal_pairs_enabled=True)
    assert not _training_scope_evidence(rgb_training)["passed"]
    forbidden = _allowed_parameter_changes(
        ["flow.velocity.0.weight", "model.fusion.projection.weight"]
    )
    assert not forbidden["passed"]
    assert forbidden["complete_model_changed"]


def test_state_calibration_outputs_cannot_overlap_formal_or_parent(
    tmp_path: Path,
) -> None:
    config = _config()
    parent = tmp_path / "parent"
    parent.mkdir()
    _validate_diagnostic_output_paths(
        input_checkpoint=parent,
        output_checkpoint_root=tmp_path / "diagnostic-checkpoints",
        output_root=tmp_path / "diagnostic-report",
        config=config,
    )
    with pytest.raises(ValueError, match="outside formal M1 roots"):
        _validate_diagnostic_output_paths(
            input_checkpoint=parent,
            output_checkpoint_root=ROOT / config["training"]["checkpoint_root"],
            output_root=tmp_path / "diagnostic-report",
            config=config,
        )
    with pytest.raises(ValueError, match="overlap the parent"):
        _validate_diagnostic_output_paths(
            input_checkpoint=parent,
            output_checkpoint_root=parent / "child",
            output_root=tmp_path / "diagnostic-report",
            config=config,
        )


@pytest.mark.parametrize(
    ("steps", "learning_rate", "threads"),
    [(0, 3e-5, 1), (1, 0.0, 1), (1, 3e-5, 0)],
)
def test_state_calibration_controls_fail_closed(
    steps: int, learning_rate: float, threads: int
) -> None:
    with pytest.raises(ValueError):
        _validate_controls(steps, learning_rate, threads)
