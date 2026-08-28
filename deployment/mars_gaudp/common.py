from __future__ import annotations
import json, os, tempfile
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

def atomic_json(path: str | Path, value: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
