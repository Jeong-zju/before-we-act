"""Canonical CARE/RoboFactory reproduction settings and drift checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "configs" / "before_we_act" / "care_robofactory_reproduction.json"
EXPECTED_TASKS = (
    "lift_barrier",
    "camera_alignment",
    "long_pipeline_delivery",
    "take_photo",
    "pass_shoe",
    "place_food",
)
EXPECTED_TASK_SETTINGS = {
    "lift_barrier": ("LiftBarrier-rf", "robofactory/configs/table/lift_barrier.yaml", (0, 1), 500),
    "camera_alignment": ("CameraAlignment-rf", "robofactory/configs/table/camera_alignment.yaml", (0, 1, 2), 1500),
    "long_pipeline_delivery": ("LongPipelineDelivery-rf", "robofactory/configs/table/long_pipeline_delivery.yaml", (0, 1, 2, 3), 1500),
    "take_photo": ("TakePhoto-rf", "robofactory/configs/table/take_photo.yaml", (0, 1, 2, 3), 1500),
    "pass_shoe": ("PassShoe-rf", "robofactory/configs/table/pass_shoe.yaml", (0, 1), 500),
    "place_food": ("PlaceFood-rf", "robofactory/configs/table/place_food.yaml", (0, 1), 500),
}


def load_frozen_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> dict[str, Any]:
    """Load and validate the repository-local reproduction contract."""

    target = Path(path).resolve(strict=True)
    value = json.loads(target.read_text(encoding="utf-8"))
    validate_frozen_settings(value)
    return value


def validate_frozen_settings(settings: Mapping[str, Any]) -> None:
    if settings.get("format_version") != "before-we-act.care-robofactory-reproduction/1":
        raise ValueError("unsupported CARE/RoboFactory reproduction settings")
    tasks = settings.get("tasks")
    if not isinstance(tasks, Mapping) or tuple(tasks) != EXPECTED_TASKS:
        raise ValueError(f"task order differs from frozen order: {tuple(tasks or ())}")
    for task in EXPECTED_TASKS:
        row = tasks[task]
        expected_env, expected_config, expected_agents, expected_steps = EXPECTED_TASK_SETTINGS[task]
        if row.get("env_id") != expected_env or row.get("config") != expected_config:
            raise ValueError(f"environment mapping differs for {task}")
        if tuple(row.get("agents", ())) != expected_agents:
            raise ValueError(f"agent layout differs for {task}")
        if int(row.get("max_steps", 0)) != expected_steps:
            raise ValueError(f"max_steps differs for {task}")

    robofactory = settings.get("robofactory", {})
    required_runtime = {
        "obs_mode": "rgb",
        "control_mode": "pd_joint_pos",
        "render_mode": "sensors",
        "reward_mode": "dense",
        "sim_backend": "cpu",
    }
    for key, expected in required_runtime.items():
        if robofactory.get(key) != expected:
            raise ValueError(f"RoboFactory {key} drift: {robofactory.get(key)!r}")
    sensor = robofactory.get("sensor", {})
    if (sensor.get("width"), sensor.get("height")) != (640, 480):
        raise ValueError("RoboFactory sensor resolution drift")

    dataset = settings.get("dataset", {})
    expected_dataset = {
        "episodes_per_task": 120,
        "action_dim_per_robot": 8,
        "image_width": 640,
        "image_height": 480,
        "history_steps": 16,
        "action_horizon": 100,
    }
    for key, expected in expected_dataset.items():
        if dataset.get(key) != expected:
            raise ValueError(f"dataset {key} drift: {dataset.get(key)!r}")

    care = settings.get("care", {})
    expected_care = {
        "candidates": 6,
        "candidate_ids": [0, 1, 2, 3, 4, 5],
        "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
        "horizons": [8, 16, 32, 64],
        "primary_horizon": 16,
        "variants": ["care", "reactive_only", "replay_only", "capacity"],
        "seeds": [20260818, 20260819, 20260820],
        "updates": 4000,
        "batch_size": 48,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "evaluation_every_updates": 200,
        "calibration_nominal_coverage": 0.9,
        "one_focal_override_per_control_step": True,
        "candidate_zero_is_reference": True,
        "fallback_is_reference": True,
    }
    for key, expected in expected_care.items():
        if care.get(key) != expected:
            raise ValueError(f"CARE {key} drift: {care.get(key)!r}")

    closed_loop = settings.get("closed_loop", {})
    if closed_loop.get("episodes_per_task") != 20:
        raise ValueError("closed-loop episode count drift")
    if closed_loop.get("modes") != ["selector_off", "care"]:
        raise ValueError("closed-loop modes drift")
    if closed_loop.get("paired") is not True:
        raise ValueError("closed-loop pairing must remain enabled")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb", buffering=0) as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_robofactory_checkout(root: str | Path, settings: Mapping[str, Any]) -> str:
    """Require the exact RoboFactory source revision used by the runner."""

    target = Path(root).resolve(strict=True)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot inspect RoboFactory checkout: {target}") from error
    expected = str(settings["robofactory"]["commit"])
    if commit != expected:
        raise RuntimeError(f"RoboFactory commit drift: {commit} != {expected}")
    for task, row in settings["tasks"].items():
        config = target / str(row["config"])
        if not config.is_file():
            raise FileNotFoundError(f"missing frozen RoboFactory config for {task}: {config}")
    return commit


def verify_dependency_lock(repo_root: str | Path, settings: Mapping[str, Any]) -> str:
    lockfile = Path(repo_root).resolve(strict=True) / str(settings["dependency_lock"])
    if not lockfile.is_file():
        raise FileNotFoundError(lockfile)
    return sha256_file(lockfile)


__all__ = [
    "DEFAULT_SETTINGS_PATH",
    "EXPECTED_TASKS",
    "EXPECTED_TASK_SETTINGS",
    "load_frozen_settings",
    "sha256_file",
    "validate_frozen_settings",
    "verify_dependency_lock",
    "verify_robofactory_checkout",
]
