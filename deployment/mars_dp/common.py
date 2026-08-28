from __future__ import annotations

import json, os, tempfile
from pathlib import Path

TASKS = ("place_cube_in_cup", "strike_cube_hard", "three_robots_place_shoes", "four_robots_stack_cube")
ARMS = {"place_cube_in_cup": 2, "strike_cube_hard": 2, "three_robots_place_shoes": 3, "four_robots_stack_cube": 4}
# RoboFactory's pd_joint_pos bounds are identical for every Panda arm. v2
# applies them before computing action statistics and training targets.
ACTION_LOW = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -1.0)
ACTION_HIGH = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1.0)
ENVS = {
    "place_cube_in_cup": ("PlaceCubeInCup-rf", "place_cube_in_cup.yaml", 500),
    "strike_cube_hard": ("StrikeCubeHard-rf", "strike_cube_hard.yaml", 500),
    "three_robots_place_shoes": ("ThreeRobotsPlaceShoes-rf", "three_robots_place_shoes.yaml", 1200),
    "four_robots_stack_cube": ("FourRobotsStackCube-rf", "four_robots_stack_cube.yaml", 800),
}

def atomic_json(path: str | Path, value: dict):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)
