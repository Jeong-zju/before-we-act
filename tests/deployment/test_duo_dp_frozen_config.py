from __future__ import annotations

import hashlib
import json
from pathlib import Path

from deployment.duo_dp import supervisor as sup
from deployment.duo_dp.common import (
    ACTION_LAG_ROWS,
    EXECUTION_STEPS,
    HORIZON,
    OBS_STEPS,
    POLICY_CONTRACT,
    TASKS,
    TEMPORAL_CONTRACT,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "duobench_dp_formal_v1.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_config_pins_formal_training_and_validation_contract() -> None:
    config = _config()
    assert config["schema"] == "before-we-act.duobench-dp.frozen-training/1"
    assert config["protocol"] == "duobench_dp_formal_v1"
    assert config["policy"]["policy_contract"] == POLICY_CONTRACT
    assert config["data"]["tasks"] == list(TASKS)
    assert config["data"]["task_count"] == 11
    assert config["data"]["demonstrations_per_task"] == 50
    assert config["data"]["total_demonstrations"] == 550
    assert config["data"]["split"] == "none_all_demonstrations_used_for_training"

    temporal = config["temporal"]
    assert temporal["contract"] == TEMPORAL_CONTRACT
    assert temporal["observation_steps"] == OBS_STEPS == 3
    assert temporal["action_lag_rows"] == ACTION_LAG_ROWS == 1
    assert temporal["prediction_horizon"] == HORIZON == 8
    assert temporal["executable_steps"] == EXECUTION_STEPS == 6

    optimization = config["optimization"]
    assert optimization["updates"] == 60_000
    assert optimization["batch_size"] == 64
    assert optimization["seed"] == 20260831
    assert optimization["samples_drawn"] == optimization["updates"] * optimization["batch_size"]
    assert optimization["task_conditioning"] is False
    assert optimization["transition_fraction"] == 0.0
    assert optimization["gripper_loss_weight"] == 1.0

    normalization = config["state_action_codec"]["normalization"]
    for key in ("q_min", "q_max", "a_min", "a_max"):
        assert len(normalization[key]) == 8
    assert all(lo < hi for lo, hi in zip(normalization["q_min"], normalization["q_max"]))
    assert all(lo < hi for lo, hi in zip(normalization["a_min"], normalization["a_max"]))

    assert config["checkpointing"]["formal_checkpoint_sha256"] == (
        "7d71c2877fa90b53ca4cf2c48d9f22d14875f995c99649b374ef2219ba2f341e"
    )
    assert config["validation20"]["total_episodes"] == 220
    assert config["validation20"]["successes"] == 1


def test_current_sync_hashes_match_the_committed_implementation() -> None:
    hashes = _config()["artifacts"]["source_file_sha256_current_sync"]
    assert "deployment/duo_dp/supervisor.py" in hashes
    for relative, expected in hashes.items():
        assert _sha256(ROOT / relative) == expected, relative


def test_supervisor_loads_the_frozen_config(monkeypatch) -> None:
    monkeypatch.setattr(sup, "CONFIG", CONFIG_PATH)
    assert sup.load_config() == _config()
