from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from scripts.train_research_v2_pipeline import _validate_dataset_manifest_for_training
from train.train_research_v2 import (
    ResearchV2TrainingConfig,
    _FileGroupedBatchSampler,
    _fit_binary_temperature,
    _fit_multiclass_temperature,
    _fit_robust_return_affine,
    _hard_code_metrics,
    _decode_plan_batch_differentiable,
    _precision_runtime,
    _safe_branch_oracle,
    _trajectory_constraint_target,
)
from models.research_v2 import PlanTokenizerV2, PlanTokenizerV2Config


def _config(tmp_path, **overrides):
    values = {
        "stage": "plan",
        "train_dir": str(tmp_path / "train"),
        "val_dir": str(tmp_path / "val"),
        "output_dir": str(tmp_path / "out"),
        "num_workers": 0,
    }
    values.update(overrides)
    return ResearchV2TrainingConfig(**values)


def test_cpu_auto_precision_remains_fp32_and_config_rejects_cpu_unsafe_values(tmp_path):
    runtime = _precision_runtime(_config(tmp_path), torch.device("cpu"))
    assert runtime.dtype is None
    assert runtime.scaler.is_enabled() is False
    with pytest.raises(ValueError, match="precision"):
        _config(tmp_path, precision="float8")
    with pytest.raises(ValueError, match="mutually exclusive"):
        _config(tmp_path, resume=True, force_retrain=True)


@dataclass(frozen=True)
class _Index:
    file_idx: int


class _Dataset:
    def __init__(self):
        self.index = [_Index(file_idx) for file_idx in range(4) for _ in range(5)]

    def __len__(self):
        return len(self.index)


def test_file_grouped_sampler_is_deterministic_local_and_epoch_shuffleable():
    dataset = _Dataset()
    validation = _FileGroupedBatchSampler(
        dataset,
        batch_size=4,
        shuffle=False,
        seed=9,
        max_batches=2,
        drop_last=False,
    )
    first = list(validation)
    assert first == list(validation)
    assert len(first) == 2
    assert all(len({dataset.index[index].file_idx for index in batch}) <= 2 for batch in first)

    training = _FileGroupedBatchSampler(
        dataset,
        batch_size=4,
        shuffle=True,
        seed=9,
        max_batches=3,
        drop_last=True,
    )
    epoch_zero = list(training)
    training.set_epoch(1)
    assert epoch_zero != list(training)


def test_hard_code_metrics_report_actual_argmin_usage():
    metrics = _hard_code_metrics(torch.tensor([8, 8, 0, 0]))
    assert metrics["hard_code_samples"] == 16
    assert metrics["hard_codes_used"] == 2
    assert metrics["hard_usage_ratio"] == pytest.approx(0.5)
    assert metrics["hard_perplexity"] == pytest.approx(2.0)
    assert metrics["hard_max_code_fraction"] == pytest.approx(0.5)
    assert metrics["hard_code_counts"] == [8, 8, 0, 0]


def test_formal_training_rejects_legacy_all_hold_manifest():
    legacy = {
        "splits": {
            "train": {"episodes": 6400},
            "val": {"episodes": 800},
        }
    }
    with pytest.raises(ValueError, match="Recollect into a new dataset root"):
        _validate_dataset_manifest_for_training(legacy, smoke=False)
    _validate_dataset_manifest_for_training(legacy, smoke=True)


def test_formal_training_accepts_private_event_quality_manifest():
    quality = {
        "private_gate_episodes": 400,
        "event_type_observation_counts": {"a": 1, "b": 1, "c": 1},
        "informed_agent_observation_counts": {"0": 1, "1": 1},
        "maneuver_observation_counts": {"left": 1, "hold": 1, "right": 1},
        "active_observations": 1,
        "cued_agent_observations": 1,
    }
    manifest = {
        "formal_scenario_mixture": ["nominal", "private_gates"],
        "splits": {
            "train": {"episodes": 6400, "private_event_quality": quality},
            "val": {"episodes": 800, "private_event_quality": quality},
        },
    }
    _validate_dataset_manifest_for_training(manifest, smoke=False)


def test_constraint_and_proposal_oracle_are_safe_first():
    batch = {
        "ego_id": torch.tensor([0, 1]),
        "target_force_violation": torch.tensor([[0.0, 0.0], [0.0, 1.0]]),
        "target_collision": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        "target_private_event_error": torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
    }
    torch.testing.assert_close(_trajectory_constraint_target(batch), torch.ones(2))

    returns = torch.tensor([[100.0, 4.0, 5.0], [1.0, 3.0, 2.0]])
    constraints = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    valid = torch.ones(2, 3, 2, dtype=torch.bool)
    assert _safe_branch_oracle(returns, constraints, valid).tolist() == [2, 1]


def test_validation_calibration_fits_known_affine_and_reduces_nll():
    prediction = torch.linspace(-2.0, 2.0, 101)
    scale, bias = _fit_robust_return_affine(prediction, 2.5 * prediction - 0.75)
    assert scale == pytest.approx(2.5, rel=1e-5)
    assert bias == pytest.approx(-0.75, abs=1e-5)

    binary_logits = torch.tensor([-3.0, -1.0, 1.0, 3.0])
    binary_target = torch.tensor([0.0, 0.0, 1.0, 1.0])
    temperature, fitted_bias = _fit_binary_temperature(binary_logits, binary_target)
    before = torch.nn.functional.binary_cross_entropy_with_logits(
        binary_logits, binary_target
    )
    after = torch.nn.functional.binary_cross_entropy_with_logits(
        binary_logits / temperature + fitted_bias, binary_target
    )
    assert temperature > 0
    assert after <= before

    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [2.0, 0.0], [0.0, 2.0]])
    target = torch.tensor([0, 1, 0, 1])
    posterior_temperature = _fit_multiclass_temperature(logits, target)
    assert posterior_temperature > 0
    assert torch.nn.functional.cross_entropy(logits / posterior_temperature, target) <= (
        torch.nn.functional.cross_entropy(logits, target)
    )


def test_ensemble_constraint_calibration_matches_runtime_probability_aggregation():
    logits = torch.tensor(
        [[-4.0, -2.0], [-2.0, -1.0], [1.0, 2.0], [2.0, 4.0]]
    )
    target = torch.tensor([0.0, 0.0, 1.0, 1.0])
    temperature, bias = _fit_binary_temperature(logits, target)
    before = torch.nn.functional.binary_cross_entropy(
        logits.sigmoid().mean(dim=-1), target
    )
    after = torch.nn.functional.binary_cross_entropy(
        (logits / temperature + bias).sigmoid().mean(dim=-1), target
    )
    assert after <= before


def test_training_plan_decode_keeps_gradient_to_proposal_residuals():
    tokenizer = PlanTokenizerV2(
        PlanTokenizerV2Config(
            horizon=4, codebook_size=8, residual_dim=16, hidden_dim=32
        )
    )
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    residuals = torch.zeros(2, 3, 16, requires_grad=True)
    decoded = _decode_plan_batch_differentiable(
        tokenizer,
        torch.tensor([[0, 1, 2], [3, 4, 5]]),
        residuals,
        torch.zeros(4),
        torch.ones(4),
    )
    decoded.square().sum().backward()
    assert residuals.grad is not None
    assert torch.isfinite(residuals.grad).all()
    assert residuals.grad.abs().sum() > 0
