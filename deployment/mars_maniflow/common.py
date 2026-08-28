from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskSpec:
    name: str
    env_id: str
    config: str
    arms: int
    max_steps: int
    seed_start: int


TASKS = (
    TaskSpec("place_cube_in_cup", "PlaceCubeInCup-rf", "place_cube_in_cup.yaml", 2, 500, 20260820),
    TaskSpec("strike_cube_hard", "StrikeCubeHard-rf", "strike_cube_hard.yaml", 2, 500, 20261820),
    TaskSpec("three_robots_place_shoes", "ThreeRobotsPlaceShoes-rf", "three_robots_place_shoes.yaml", 3, 1200, 20262820),
    TaskSpec("four_robots_stack_cube", "FourRobotsStackCube-rf", "four_robots_stack_cube.yaml", 4, 800, 20263820),
)
TASK_BY_NAME = {task.name: task for task in TASKS}

# RoboFactory's pd_joint_pos action space is shared by all MARS-Control Panda
# arms. Actions are clipped to this contract before statistics and targets.
ACTION_LOW = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -1.0)
ACTION_HIGH = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1.0)

DATASET_REPOS = {
    "place_cube_in_cup": ("Jeong-zju/mars-control-place-cube-in-cup-rf", "3878150bec8f4830e1a57a01a13762a10abc8d52"),
    "strike_cube_hard": ("Jeong-zju/mars-control-strike-cube-hard-rf", "bc7051cb0560058bf426e792871faa1ca8a4f78f"),
    "three_robots_place_shoes": ("Jeong-zju/mars-control-three-robots-place-shoes-rf", "ad231c7eff530f71f0c5302b6c03c7164bbcc896"),
    "four_robots_stack_cube": ("Jeong-zju/mars-control-four-robots-stack-cube-rf", "3fa4833f5e34c3565da04af99c62d516e048fcfc"),
}

DATASET_REPOS = {
    "place_cube_in_cup": ("Jeong-zju/mars-control-place-cube-in-cup-rf", "3878150bec8f4830e1a57a01a13762a10abc8d52"),
    "strike_cube_hard": ("Jeong-zju/mars-control-strike-cube-hard-rf", "bc7051cb0560058bf426e792871faa1ca8a4f78f"),
    "three_robots_place_shoes": ("Jeong-zju/mars-control-three-robots-place-shoes-rf", "ad231c7eff530f71f0c5302b6c03c7164bbcc896"),
    "four_robots_stack_cube": ("Jeong-zju/mars-control-four-robots-stack-cube-rf", "3fa4833f5e34c3565da04af99c62d516e048fcfc"),
}
POLICY_CONTRACT = "shared_checkpoint_strict_local_rgb_qpos_to_local_action8"
UPSTREAM_REPO = "https://github.com/geyan21/ManiFlow_Policy"
UPSTREAM_COMMIT = "ef2f116f1f90163ed36e657b8c5503740bb468af"
FROZEN_CONFIG = Path(__file__).with_name("mars_control_maniflow_v1.json")


def load_frozen_config(path: str | Path = FROZEN_CONFIG) -> dict:
    config_path = Path(path)
    value = json.loads(config_path.read_text())
    if value.get("schema") != "mars-control.maniflow.frozen-config.v2":
        raise ValueError(f"unexpected ManiFlow config schema: {config_path}")
    if value.get("status") != "frozen":
        raise ValueError(f"ManiFlow config is not frozen: {config_path}")
    if value["upstream"]["commit"] != UPSTREAM_COMMIT:
        raise ValueError("ManiFlow upstream commit drift")
    if value["policy_contract"]["name"] != POLICY_CONTRACT:
        raise ValueError("ManiFlow policy contract drift")
    expected_tasks = [task.name for task in TASKS]
    if value["data"]["tasks"] != expected_tasks:
        raise ValueError("MARS-Control task order drift")
    return value


def atomic_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
