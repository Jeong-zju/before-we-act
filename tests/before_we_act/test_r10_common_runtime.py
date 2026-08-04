from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from stereo_core.bwa_perception import (
    BRIDGE_REGISTRY,
    EXPECTED_PARENT_SHA256,
    build_perception_extension,
    load_r10_config,
)


ROOT = Path(__file__).resolve().parents[2]


def _load_runtime():
    path = ROOT / "scripts/before_we_act/r10_runtime.py"
    spec = importlib.util.spec_from_file_location("r10_runtime_test_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parent_registry_contains_no_candidate_and_fails_closed():
    assert BRIDGE_REGISTRY == {}
    with pytest.raises(ValueError, match="not registered"):
        build_perception_extension({"kind": "calibrated_crossview"})


def test_common_config_locks_budget_seed_precision_and_unknown_keys(tmp_path):
    config = {
        "schema_version": 1,
        "candidate_id": "p0",
        "parent_commit": "a" * 40,
        "checkpoint_sha256": EXPECTED_PARENT_SHA256,
        "bridge": {"kind": "calibrated_crossview"},
        "training": {
            "batch_size": 40,
            "seed": 20260803,
            "screen_updates": 10_000,
            "selection_updates": 30_000,
            "precision": "bfloat16",
        },
        "loss_weights": {"action": 1.0},
        "calibration": {},
        "intervention": {"name": "ray_shuffle"},
    }
    path = tmp_path / "p0.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert load_r10_config(path)["training"]["screen_updates"] == 10_000
    config["training"]["batch_size"] = 39
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="locked training protocol"):
        load_r10_config(path)
    config["training"]["batch_size"] = 40
    config["leak"] = True
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="config keys differ"):
        load_r10_config(path)


def test_runtime_heartbeat_is_real_and_refreshes_training_metrics(tmp_path):
    runtime = _load_runtime()
    candidate_root = tmp_path / "candidates/p0"
    progress = candidate_root / "train/screen/progress.jsonl"
    progress.parent.mkdir(parents=True)
    progress.write_text(
        json.dumps({"update": 50, "target_updates": 10_000, "loss": 0.25}) + "\n",
        encoding="utf-8",
    )
    runtime.atomic_json(
        candidate_root / "status.json",
        {"state": "TRAINING", "stage": "screen", "program": "trainer"},
    )
    runtime.heartbeat(tmp_path, "p0", pid=123, child_pid=456)
    beat = json.loads((candidate_root / "heartbeat.json").read_text())
    status = json.loads((candidate_root / "status.json").read_text())
    assert beat["pid"] == 123 and beat["child_pid"] == 456
    assert status["update"] == 50
    assert status["total_updates"] == 10_000
    assert status["loss"] == 0.25


def test_future_targets_are_not_put_in_deployment_context():
    source = (ROOT / "stereo_core/train_bwa_perception.py").read_text(encoding="utf-8")
    context_start = source.index("deployment_context = CoreDeploymentContext(")
    context_end = source.index("targets.update(", context_start)
    deployment_block = source[context_start:context_end]
    assert "future_qpos" not in deployment_block
    assert "future_view_features" not in deployment_block
    assert "future_qpos_horizons" in source
    assert "future_feature_horizons" in source
