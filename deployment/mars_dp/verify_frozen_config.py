"""Fail fast if the checked-in MARS DP v3 manifest or source has drifted."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .common import ACTION_HIGH, ACTION_LOW, TASKS


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "dp" / "mars_control_full_data_v3.json"
SOURCE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify() -> dict:
    config = json.loads(CONFIG.read_text())
    assert config["schema"] == "before-we-act.dp.mars-control.full-data/3"
    assert config["protocol_status"] == "frozen; used for mars_dp_v3 formal run"
    assert tuple(config["data"]["tasks"]) == TASKS
    assert config["data"]["total_episodes"] == 600
    assert config["data"]["total_local_streams"] == 1650
    assert config["data"]["split"].startswith("none;")
    assert config["data"]["formal_shards_per_task"] == 10
    assert config["temporal_contract"]["observation_steps"] == 3
    assert config["temporal_contract"]["prediction_horizon"] == 8
    assert config["temporal_contract"]["action_steps"] == 8
    assert config["temporal_contract"]["executable_steps"] == 6
    assert config["temporal_contract"]["replan_interval"] == 6
    codec = config["state_action_codec"]
    assert tuple(codec["action_low"]) == ACTION_LOW
    assert tuple(codec["action_high"]) == ACTION_HIGH
    model = config["model"]
    assert model["down_dims"] == [256, 512, 1024]
    assert model["diffusion_train_steps"] == 100
    assert model["diffusion_inference_steps"] == 20
    assert model["prediction_type"] == "epsilon"
    optimization = config["optimization"]
    assert optimization["updates"] == 60000
    assert optimization["batch_size"] == 64
    assert optimization["learning_rate"] == 0.0001
    assert optimization["seed"] == 20260827
    validation = config["validation20"]
    assert validation["episodes_per_task"] == 20
    assert validation["seed_start_by_task"] == [20260820, 20261820, 20262820, 20263820]
    assert validation["maximum_steps"] == [500, 500, 1200, 800]
    assert validation["evaluator_revision"] == "dp-official-obs3-horizon8-exec6-command-state-topp-v6"
    mismatches = {
        name: {"expected": expected, "actual": digest(SOURCE / name)}
        for name, expected in config["artifacts"]["source_sha256"].items()
        if digest(SOURCE / name) != expected
    }
    if mismatches:
        raise RuntimeError(f"MARS DP v3 source drift: {mismatches}")
    return {
        "status": "complete",
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": digest(CONFIG),
        "checkpoint_sha256": config["artifacts"]["checkpoint_sha256"],
    }


def main() -> None:
    print(json.dumps(verify(), sort_keys=True))


if __name__ == "__main__":
    main()
