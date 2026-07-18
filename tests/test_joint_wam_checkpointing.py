from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from models.wam import (
    NormalizationStats,
    RWMARConfig,
    RWMARWorldModel,
    StatefulActionFlow,
    StatefulActionFlowConfig,
)
from train.joint_wam_checkpointing import (
    ACTION_FLOW_WEIGHTS,
    CHECKPOINT_FORMAT_VERSION,
    GENERATED_ACTION_WORLD_TARGET_SOURCE,
    NORMALIZATION_FILE,
    WORLD_MODEL_WEIGHTS,
    load_joint_wam_checkpoint,
    save_joint_wam_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


def test_joint_wam_config_locks_runtime_and_checkpoint_contract() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/wam/joint_wam.yaml").read_text(encoding="utf-8")
    )
    assert config["name"] == "wam.cooperative_stop/joint-wam"
    assert set(config["initialization"]) == {
        "world_model_checkpoint",
        "action_prior_checkpoint",
        "generated_action_teacher",
    }
    assert config["action_chunk"] == {
        "horizon": 8,
        "execution_steps": 2,
        "solver_steps": 4,
        "solver": "euler",
        "expert_count": 1,
        "warm_start_mode": "shift_repeat_last",
        "warm_start_shift_source": "actual_executed_steps",
    }
    assert [stage["scope"] for stage in config["joint_training"]["stages"]] == [
        "flow_only",
        "world_heads",
        "full_joint",
    ]
    assert [stage["steps"] for stage in config["joint_training"]["stages"]] == [
        64,
        128,
        512,
    ]
    assert config["runtime"]["anchor_residual_scale"] == 0.1
    assert config["checkpoint"]["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert config["checkpoint"]["directory"] == "checkpoints/joint_wam"


def test_joint_wam_checkpoint_is_strict_and_self_contained(tmp_path: Path) -> None:
    checkpoint, world_model, flow = _save_checkpoint(tmp_path / "joint_wam")

    loaded_world, loaded_flow, metadata = load_joint_wam_checkpoint(
        checkpoint,
        expected_schema_version="wam.proprio/1.0",
    )

    assert {path.name for path in checkpoint.iterdir() if path.is_file()} == {
        WORLD_MODEL_WEIGHTS,
        ACTION_FLOW_WEIGHTS,
        NORMALIZATION_FILE,
        "config.yaml",
        "schema.json",
        "dataset_manifest.json",
        "metrics.json",
        "provenance.json",
    }
    assert metadata["schema"]["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert metadata["schema"]["generated_action_world_target_source"] == (
        GENERATED_ACTION_WORLD_TARGET_SOURCE
    )
    assert metadata["schema"]["source_fingerprints"] == {"world_model": "a" * 64}
    _assert_state_equal(world_model, loaded_world)
    _assert_state_equal(flow, loaded_flow)
    assert all(
        not parameter.requires_grad
        for parameter in loaded_flow.anchor_prior.parameters()
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model", "not_joint"),
        ("action_prior_fallback_required", True),
        ("generated_action_world_target_source", "dataset_demo_future"),
        ("generated_action_demo_state_is_ground_truth", True),
    ),
)
def test_joint_wam_checkpoint_rejects_schema_tampering(
    tmp_path: Path, field: str, value: object
) -> None:
    checkpoint, _, _ = _save_checkpoint(tmp_path / "joint_wam")
    schema_path = checkpoint / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema[field] = value
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    with pytest.raises(ValueError):
        load_joint_wam_checkpoint(checkpoint)


def test_joint_wam_checkpoint_rejects_artifact_tampering(tmp_path: Path) -> None:
    checkpoint, _, _ = _save_checkpoint(tmp_path / "joint_wam")
    metrics_path = checkpoint / "metrics.json"
    metrics_path.write_text('{"passed": false}', encoding="utf-8")

    with pytest.raises(ValueError, match="artifact fingerprint mismatch: metrics"):
        load_joint_wam_checkpoint(checkpoint)


def test_joint_wam_checkpoint_rejects_missing_normalization(tmp_path: Path) -> None:
    checkpoint, _, _ = _save_checkpoint(tmp_path / "joint_wam")
    (checkpoint / NORMALIZATION_FILE).unlink()

    with pytest.raises(FileNotFoundError, match="normalization.npz"):
        load_joint_wam_checkpoint(checkpoint)


def _save_checkpoint(
    path: Path,
) -> tuple[Path, RWMARWorldModel, StatefulActionFlow]:
    torch.manual_seed(7)
    stats = _stats()
    world_model = RWMARWorldModel(
        RWMARConfig(
            encoder_hidden_dim=16,
            gru_hidden_dim=12,
            gru_layers=1,
            train_forecast_horizon=4,
            planning_horizon=4,
        ),
        stats,
    )
    flow = StatefulActionFlow(
        StatefulActionFlowConfig(
            feature_dim=world_model.planning_feature_dim,
            horizon=4,
            hidden_dim=16,
            hidden_layers=1,
            time_embedding_dim=8,
            anchor_hidden_dim=16,
            anchor_hidden_layers=1,
        ),
        stats,
    )
    checkpoint = save_joint_wam_checkpoint(
        path,
        world_model,
        flow,
        stats,
        experiment_config={"name": "test/joint-wam"},
        dataset_manifest={"partitions": {}},
        metrics={"passed": True},
        provenance={"seed": 7},
        schema_version="wam.proprio/1.0",
        source_fingerprints={"world_model": "a" * 64},
    )
    return checkpoint, world_model, flow


def _stats() -> NormalizationStats:
    return NormalizationStats(
        state_mean=np.zeros(22, dtype=np.float32),
        state_std=np.ones(22, dtype=np.float32),
        action_mean=np.zeros(8, dtype=np.float32),
        action_std=np.ones(8, dtype=np.float32),
        delta_mean=np.zeros(22, dtype=np.float32),
        delta_std=np.ones(22, dtype=np.float32),
        reward_mean=np.zeros(1, dtype=np.float32),
        reward_std=np.ones(1, dtype=np.float32),
    )


def _assert_state_equal(first: torch.nn.Module, second: torch.nn.Module) -> None:
    assert first.state_dict().keys() == second.state_dict().keys()
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
