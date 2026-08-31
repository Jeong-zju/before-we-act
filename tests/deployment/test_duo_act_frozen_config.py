from __future__ import annotations

import copy
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

from deployment.duo_act.dataset import TASKS
from deployment.duo_act.model import ACT
from deployment.duo_act.train import (
    FROZEN_CONFIG_SCHEMA,
    FROZEN_CONFIG_SHA256,
    apply_frozen_config,
    validate_frozen_config,
    validate_manifest_against_config,
    validate_frozen_policy_sources,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "duobench_act_causal_lag1_prior_v1.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_config_is_self_consistent_and_hash_bound():
    config = _config()
    validate_frozen_config(config)
    validate_frozen_policy_sources(config)
    assert config["schema"] == FROZEN_CONFIG_SCHEMA
    assert config["data"]["tasks"] == list(TASKS)
    assert config["data"]["total_demonstrations"] == 11 * 50
    assert config["training"]["samples_drawn"] == (
        config["training"]["updates"] * config["training"]["batch_size"]
    )
    assert config["training"]["scheduler"]["T_max"] == config["training"]["updates"]
    assert config["data"]["indexed_local_arm_samples"] == 2 * config["data"]["causal_state_action_pairs"]
    assert _sha256(CONFIG_PATH) == FROZEN_CONFIG_SHA256


def _formal_args(config_path: Path) -> Namespace:
    return Namespace(
        config=config_path,
        data=None,
        output=None,
        updates=120000,
        batch_size=64,
        workers=12,
        horizon=100,
        action_lag=0,
        lr=2e-4,
        beta=1e-3,
        seed=20260829,
        save_every=5000,
        init_checkpoint=None,
        prior_loss_weight=0.0,
        prior_loss_frequency=1,
        smoke=False,
    )


def test_config_resolver_replaces_defaults_and_rejects_cli_drift():
    config = _config()
    args = _formal_args(CONFIG_PATH)
    resolved, digest = apply_frozen_config(args, ["--config", str(CONFIG_PATH)])
    assert resolved == config
    assert digest == _sha256(CONFIG_PATH)
    assert args.data == Path(config["paths"]["data_root"])
    assert args.output == Path(config["paths"]["output_root"])
    assert args.updates == 50000
    assert args.action_lag == 1
    assert args.prior_loss_weight == pytest.approx(0.1)

    drifting = _formal_args(CONFIG_PATH)
    with pytest.raises(ValueError, match="--updates"):
        apply_frozen_config(
            drifting,
            ["--config", str(CONFIG_PATH), "--updates", "1"],
        )


def test_config_resolver_rejects_any_byte_level_config_drift(tmp_path: Path):
    tampered = _config()
    tampered["training"]["updates"] -= 1
    path = tmp_path / CONFIG_PATH.name
    path.write_text(json.dumps(tampered, indent=2) + "\n")
    with pytest.raises(ValueError, match="config hash"):
        apply_frozen_config(_formal_args(path), ["--config", str(path)])


def test_manifest_validation_pins_revision_alignment_contract_and_statistics():
    config = _config()
    manifest = {
        "dataset_revision": config["data"]["dataset_revision"],
        "total_episodes": config["data"]["total_demonstrations"],
        "total_frames": config["data"]["total_frames"],
        "recording_alignment": {"action_lag_rows": config["data"]["action_lag_rows"]},
        "action_target_contract": {
            "sha256": config["data"]["action_target_contract_sha256"]
        },
        "normalization": copy.deepcopy(config["data"]["normalization"]),
    }
    validate_manifest_against_config(manifest, config)

    stale = copy.deepcopy(manifest)
    stale["normalization"]["action_std"][0] += 1e-6
    with pytest.raises(ValueError, match="action_std"):
        validate_manifest_against_config(stale, config)


def test_model_constructor_and_parameter_count_match_frozen_contract():
    config = _config()
    model = ACT(**config["model"]["constructor"])
    assert model.config["vision_backbone"] == "resnet18_scratch"
    assert sum(parameter.numel() for parameter in model.parameters()) == config["model"]["trainable_parameters"]
