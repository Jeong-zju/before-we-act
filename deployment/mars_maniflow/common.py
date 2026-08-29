from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
MAX_STEPS_BY_TASK = {
    "place_cube_in_cup": 500,
    "strike_cube_hard": 500,
    "three_robots_place_shoes": 1200,
    "four_robots_stack_cube": 800,
}
REPLAN_INTERVAL = 8
TEMPORAL_ENSEMBLE_DECAY = 0.01
EVALUATOR_REVISION = "maniflow-mars-temporal-ensemble-v3"


class TemporalEnsemble:
    """Blend every still-valid action prediction, favoring newer chunks."""

    def __init__(self, count: int, decay: float = TEMPORAL_ENSEMBLE_DECAY):
        if count <= 0 or decay < 0:
            raise ValueError("TemporalEnsemble requires count > 0 and decay >= 0")
        self.histories: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(count)]
        self.decay = float(decay)

    def add(self, step: int, chunks: np.ndarray) -> None:
        if len(chunks) != len(self.histories):
            raise ValueError("one action chunk is required for every arm")
        for history, chunk in zip(self.histories, chunks):
            value = np.asarray(chunk, dtype=np.float32)
            if value.ndim != 2 or len(value) == 0:
                raise ValueError("action chunks must have shape [time, action_dim]")
            history.append((int(step), value))

    def select(self, step: int) -> list[np.ndarray]:
        selected = []
        for index, history in enumerate(self.histories):
            history = [(start, chunk) for start, chunk in history if 0 <= step - start < len(chunk)]
            self.histories[index] = history
            if not history:
                raise RuntimeError(f"no valid action prediction at step {step} for arm {index}")
            candidates = np.asarray([chunk[step - start] for start, chunk in history], np.float32)
            # Histories are oldest to newest. Match ManiFlow's reference evaluator:
            # the newest prediction has weight 1 and older predictions decay.
            weights = np.exp(-self.decay * np.arange(len(candidates) - 1, -1, -1, dtype=np.float32))
            weights /= weights.sum()
            selected.append(np.sum(candidates * weights[:, None], axis=0))
        return selected

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
    validation = value["validation20"]
    if validation["episodes_per_task"] != 20:
        raise ValueError("Validation20 episode count drift")
    if validation["max_steps"] != MAX_STEPS_BY_TASK:
        raise ValueError("MARS-Control maximum-step contract drift")
    if {task.name: task.max_steps for task in TASKS} != MAX_STEPS_BY_TASK:
        raise ValueError("TaskSpec maximum steps drift")
    if validation["replan_interval"] != REPLAN_INTERVAL:
        raise ValueError("ManiFlow replan interval drift")
    if validation["chunk_aggregation"] != "temporal_ensemble":
        raise ValueError("ManiFlow chunk aggregation drift")
    if validation["temporal_ensemble_decay"] != TEMPORAL_ENSEMBLE_DECAY:
        raise ValueError("ManiFlow temporal ensemble decay drift")
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
