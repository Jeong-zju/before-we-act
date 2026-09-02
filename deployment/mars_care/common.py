from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class TaskSpec:
    name: str
    env_id: str
    config: str
    arms: int
    max_steps: int


TASKS = (
    TaskSpec("place_cube_in_cup", "PlaceCubeInCup-rf", "configs/table/place_cube_in_cup.yaml", 2, 500),
    TaskSpec("strike_cube_hard", "StrikeCubeHard-rf", "configs/table/strike_cube_hard.yaml", 2, 500),
    TaskSpec("three_robots_place_shoes", "ThreeRobotsPlaceShoes-rf", "configs/table/three_robots_place_shoes.yaml", 3, 1200),
    TaskSpec("four_robots_stack_cube", "FourRobotsStackCube-rf", "configs/table/four_robots_stack_cube.yaml", 4, 800),
)
TASK_BY_NAME = {task.name: task for task in TASKS}


def as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def local_observation(observation: dict[str, Any], arm: int) -> tuple[np.ndarray, np.ndarray]:
    """The only policy input path: this arm's camera and this arm's proprioception."""
    image = as_numpy(observation["sensor_data"][f"head_camera_agent{arm}"]["rgb"])
    qpos = as_numpy(observation["agent"][f"panda-{arm}"]["qpos"])
    if image.ndim == 4:
        image = image[0]
    if qpos.ndim == 2:
        qpos = qpos[0]
    return image[..., :3].astype(np.uint8, copy=False), qpos.astype(np.float32, copy=False)


def make_env(
    task: TaskSpec,
    robofactory_root: Path,
    render_device: str | None = None,
):
    import os
    import sys
    import gymnasium as gym

    root = Path(robofactory_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import tasks  # noqa: F401 - registers official MARS environments

    os.chdir(root)

    return gym.make(
        task.env_id,
        config=str(root / task.config),
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="sensors",
        reward_mode="dense",
        sim_backend="cpu",
        render_backend=render_device or "gpu",
        sensor_configs={"shader_pack": "default"},
        human_render_camera_configs={"shader_pack": "default"},
        viewer_camera_configs={"shader_pack": "default"},
    )
