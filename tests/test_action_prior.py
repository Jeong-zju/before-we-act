from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from models.wam import ActionPrior, ActionPriorConfig, NormalizationStats, RWMARConfig, RWMARWorldModel
from policies import ActionPriorPolicy
from train.action_prior import load_action_prior_checkpoint, save_action_prior_checkpoint
from train.rwm_ar_checkpointing import save_wam_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def test_action_prior_config_is_independent_of_mppi():
    payload = yaml.safe_load((ROOT / "configs/wam/action_prior.yaml").read_text(encoding="utf-8"))
    assert payload["pipeline"] == "action_prior"
    assert payload["checkpoint"]["format_version"] == "wam.action_prior/1"
    assert "planner" not in payload


def test_action_prior_is_finite_and_policy_rejects_privileged_state():
    model = _world_model()
    prior = _prior(model)
    features = torch.zeros(3, model.planning_feature_dim)
    actions = prior.sample_action(features)
    assert actions.shape == (3, 8)
    assert bool(torch.isfinite(prior.nll(features, actions)).all())
    policy = ActionPriorPolicy(model, prior, fixed_actions={3: 1.0, 7: 1.0})
    observation = {"proprioception": np.zeros(22, dtype=np.float32)}
    action = policy.act(observation)
    assert action.shape == (8,)
    assert np.isfinite(action).all()
    assert action[3] == 1.0 and action[7] == 1.0
    with pytest.raises(RuntimeError, match="privileged_state"):
        policy.act({**observation, "privileged_state": {}})


def test_action_prior_checkpoint_roundtrip_is_strict(tmp_path: Path):
    world_model = tmp_path / "world_model"
    stats = _stats()
    model = _world_model(stats)
    save_wam_checkpoint(
        world_model,
        model,
        stats,
        experiment_config={"pipeline": "world_model"},
        dataset_manifest={"partitions": {}},
        metrics={"evaluation": {}},
        provenance={"seed": 7},
        schema_version="wam.proprio/1.0",
    )
    prior = _prior(model)
    checkpoint = tmp_path / "action_prior"
    save_action_prior_checkpoint(
        checkpoint,
        prior,
        world_model_checkpoint=world_model,
        experiment_config={"phase": "action_prior_baseline"},
        dataset_manifest={"partitions": {}},
        metrics={"evaluation": {}},
        provenance={"seed": 17},
        schema_version="wam.proprio/1.0",
        normalization_sha256="normalization",
    )
    loaded, metadata = load_action_prior_checkpoint(
        checkpoint,
        world_model_checkpoint=world_model,
        expected_schema_version="wam.proprio/1.0",
        expected_normalization_sha256="normalization",
    )
    for name, value in prior.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name])
    assert metadata["schema"]["format_version"] == "wam.action_prior/1"


def _world_model(stats: NormalizationStats | None = None) -> RWMARWorldModel:
    config = RWMARConfig(
        history_horizon=3,
        train_forecast_horizon=2,
        planning_horizon=2,
        encoder_hidden_dim=16,
        gru_hidden_dim=16,
        gru_layers=1,
    )
    return RWMARWorldModel(config, stats or _stats())


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


def _prior(model: RWMARWorldModel) -> ActionPrior:
    return ActionPrior(
        ActionPriorConfig(
            feature_dim=model.planning_feature_dim,
            hidden_dim=16,
            hidden_layers=1,
        )
    )
