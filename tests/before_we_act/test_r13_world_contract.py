from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from before_we_act.contracts import ConsequencePrediction
from before_we_act.data.world_windows import (
    CachedWorldWindows,
    LEGAL_INPUT_KEYS,
    TARGET_KEYS,
    legal_model_inputs,
)
from before_we_act.world_model.base import load_r13_config


def config_payload(candidate="p0"):
    kinds = {
        "p0": "tdmpc2_latent_dynamics",
        "p1": "lpwm_particle_dynamics",
        "p2": "vjepa2_action_predictor",
        "p3": "dino_wm_feature_dynamics",
    }
    return {
        "schema_version": 1,
        "round": "R13",
        "candidate_id": candidate,
        "parent_commit": "a" * 40,
        "belief_checkpoint_sha256": "b" * 64,
        "action_checkpoint_sha256": "c" * 64,
        "component": {"kind": kinds[candidate]},
        "world": {
            "belief_dim": 96,
            "belief_tokens": 16,
            "max_agents": 4,
            "action_horizon": 100,
            "action_dim": 8,
            "action_prefix_steps": 16,
            "prediction_horizons": [1, 5, 15],
            "target_tokens": 1,
            "future_inputs_forbidden": True,
            "planner_enabled": False,
            "rerank_enabled": False,
        },
        "training": {
            "batch_size": 64,
            "updates": 10000,
            "seed": 20260806,
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "precision": "bfloat16",
            "checkpoint_every": 1000,
            "progress_every": 20,
            "grad_clip": 1.0,
        },
        "loss_weights": {"latent": 0.60, "qpos": 0.20, "progress": 0.15, "failure": 0.05},
        "selection_rule": {
            "latent_gain": 0.50,
            "qpos_gain": 0.20,
            "progress_r2": 0.20,
            "throughput": 0.10,
            "throughput_saturation_windows_per_second": 1024,
            "minimum_score": None,
            "winner_rule": "highest_world_screen_score_among_valid_candidates",
        },
    }


def test_r13_config_is_fail_closed(tmp_path: Path):
    path = tmp_path / "p0.yaml"
    path.write_text(yaml.safe_dump(config_payload()), encoding="utf-8")
    assert load_r13_config(path).candidate_id == "p0"
    payload = config_payload()
    payload["world"]["planner_enabled"] = True
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="world contract"):
        load_r13_config(path)


def test_consequence_contract_and_no_action_field():
    prediction = ConsequencePrediction(
        latent_by_horizon=torch.zeros(2, 1, 3, 1, 96),
        qpos_delta_by_horizon=torch.zeros(2, 1, 3, 4, 9),
        progress_by_horizon=torch.zeros(2, 1, 3),
        failure_logits_by_horizon=torch.zeros(2, 1, 3),
        uncertainty_by_horizon=torch.zeros(2, 1, 3),
        valid_mask=torch.ones(2, 1, dtype=torch.bool),
    ).validate()
    assert not hasattr(prediction, "actions")


def test_cache_separates_legal_inputs_and_future_targets(tmp_path: Path):
    assert not set(LEGAL_INPUT_KEYS) & set(TARGET_KEYS)
    n = 2
    tensors = {
        "belief_tokens": torch.zeros(n, 16, 96),
        "belief_agent_tokens": torch.zeros(n, 4, 96),
        "belief_consensus": torch.zeros(n, 96),
        "belief_uncertainty": torch.zeros(n, 1),
        "agent_mask": torch.ones(n, 4, dtype=torch.bool),
        "candidate_actions": torch.zeros(n, 1, 4, 100, 8),
        "candidate_valid_mask": torch.ones(n, 1, dtype=torch.bool),
        "current_latent": torch.zeros(n, 96),
        "future_latent": torch.zeros(n, 3, 1, 96),
        "future_qpos_delta": torch.zeros(n, 3, 4, 9),
        "future_progress": torch.zeros(n, 3),
        "future_failure": torch.zeros(n, 3),
        "horizon_mask": torch.ones(n, 3, dtype=torch.bool),
        "task_index": torch.zeros(n, dtype=torch.long),
    }
    path = tmp_path / "cache.pt"
    torch.save(
        {
            "schema_version": 1,
            "round": "R13",
            "metadata": {"future_targets_are_model_inputs": False},
            "train": tensors,
            "validation": tensors,
        },
        path,
    )
    batch = CachedWorldWindows(path, "train").data
    assert set(legal_model_inputs(batch)) == set(LEGAL_INPUT_KEYS)
    assert not set(legal_model_inputs(batch)) & set(TARGET_KEYS)
