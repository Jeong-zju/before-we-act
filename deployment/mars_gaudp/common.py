from __future__ import annotations
import json, os, tempfile
import hashlib
from pathlib import Path

TASKS = ("place_cube_in_cup", "strike_cube_hard", "three_robots_place_shoes", "four_robots_stack_cube")
ARMS = {"place_cube_in_cup": 2, "strike_cube_hard": 2, "three_robots_place_shoes": 3, "four_robots_stack_cube": 4}
ENVS = {
    "place_cube_in_cup": ("PlaceCubeInCup-rf", "place_cube_in_cup.yaml", 500, 20260820),
    "strike_cube_hard": ("StrikeCubeHard-rf", "strike_cube_hard.yaml", 500, 20261820),
    "three_robots_place_shoes": ("ThreeRobotsPlaceShoes-rf", "three_robots_place_shoes.yaml", 1200, 20262820),
    "four_robots_stack_cube": ("FourRobotsStackCube-rf", "four_robots_stack_cube.yaml", 800, 20263820),
}
ACTION_LOW = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -1.0)
ACTION_HIGH = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1.0)
POLICY_CONTRACT = "shared_weights_decentralized_local_rgb_gaussian_qpos_to_absolute_action8"
FROZEN_CONFIG = Path(__file__).with_name("mars_control_gaudp_v1.json")

def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def load_frozen_config(path: str | Path = FROZEN_CONFIG) -> dict:
    config_path = Path(path)
    value = json.loads(config_path.read_text())
    if value.get("schema") != "mars-control.gaudp.frozen-config.v1":
        raise ValueError(f"unexpected GauDP config schema: {config_path}")
    if value.get("status") != "frozen":
        raise ValueError(f"GauDP config is not frozen: {config_path}")
    if value["policy_contract"]["name"] != POLICY_CONTRACT:
        raise ValueError("GauDP policy contract drift")
    if value["data"]["tasks"] != list(TASKS):
        raise ValueError("MARS-Control task order drift")
    return value

def atomic_json(path: str | Path, value: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
