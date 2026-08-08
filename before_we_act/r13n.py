"""Frozen six-task contract for the R13N no-stack baseline reset."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import numpy as np


TASKS = (
    "lift_barrier",
    "camera_alignment",
    "long_pipeline_delivery",
    "take_photo",
    "pass_shoe",
    "place_food",
)

TASK_SPECS: Mapping[str, Mapping[str, object]] = {
    "lift_barrier": {
        "repo": "zeno-ai/robofactory-lift-barrier-multiview",
        "revision": "6ab620091677e69370412f08cd7adecacc28c146",
        "agents": 2,
        "camera_order": ("global", "agent_0", "agent_1"),
        "train_steps": 8_255,
        "max_steps": 500,
    },
    "camera_alignment": {
        "repo": "zeno-ai/robofactory-camera-alignment-multiview",
        "revision": "e204af13f7191dfd86dab3da529316a51558f479",
        "agents": 3,
        "camera_order": ("global", "agent_0", "agent_1", "agent_2"),
        "train_steps": 11_764,
        "max_steps": 1_500,
    },
    "long_pipeline_delivery": {
        "repo": "zeno-ai/robofactory-long-pipeline-delivery-multiview",
        "revision": "fee628311ff52a3ae0ddfddf82379c63d28f7533",
        "agents": 4,
        "camera_order": ("global", "agent_0", "agent_1", "agent_2", "agent_3"),
        "train_steps": 88_493,
        "max_steps": 1_500,
    },
    "take_photo": {
        "repo": "zeno-ai/robofactory-take-photo-multiview",
        "revision": "3966385a4c688a5610d4b6cde044150f6b73d320",
        "agents": 4,
        "camera_order": ("global", "agent_0", "agent_1", "agent_2", "agent_3"),
        "train_steps": 23_044,
        "max_steps": 1_500,
    },
    "pass_shoe": {
        "repo": "zeno-ai/robofactory-pass-shoe-multiview",
        "revision": "646bbfec792ed46c78e452acfc06b423ca1410af",
        "agents": 2,
        "camera_order": ("global", "agent_0", "agent_1"),
        "train_steps": 43_501,
        "max_steps": 500,
    },
    "place_food": {
        "repo": "zeno-ai/robofactory-place-food-multiview",
        "revision": "2237d907f0b28d3f2e19fa4ea03b4048be2de27d",
        "agents": 2,
        "camera_order": ("global",),
        "train_steps": 25_975,
        "max_steps": 500,
    },
}

SPLIT_EPISODES = {"train": 120, "validation": 15, "test": 15}
FULL_CACHE_PROTOCOL = "r13n_six_task_native_480x640_dinov3_30x40_to_6x8_v1"

CAMERA_SENSOR_KEYS = {
    "global": "head_camera_global",
    "agent_0": "head_camera_agent0",
    "agent_1": "head_camera_agent1",
    "agent_2": "head_camera_agent2",
    "agent_3": "head_camera_agent3",
}


def camera_sensor_key(view: str) -> str:
    """Map immutable dataset view names to RoboFactory runtime sensor keys."""

    try:
        return CAMERA_SENSOR_KEYS[view]
    except KeyError as exc:
        raise ValueError(f"unknown R13N camera view {view!r}") from exc


def clamp_action_to_space(action_space, action: dict[str, np.ndarray]):
    """Apply the simulator's physical joint limits after denormalization."""

    spaces = getattr(action_space, "spaces", None)
    if not isinstance(spaces, dict) or set(action) - set(spaces):
        raise ValueError("R13N action space does not cover every active robot")
    bounded = {}
    clipped_elements = 0
    total_elements = 0
    for key, value in action.items():
        array = np.asarray(value, dtype=np.float32)
        box = spaces[key]
        low = np.asarray(box.low, dtype=np.float32)
        high = np.asarray(box.high, dtype=np.float32)
        if array.shape != low.shape or low.shape != high.shape:
            raise ValueError(f"R13N physical action shape differs for {key}")
        clipped = np.clip(array, low, high).astype(np.float32, copy=False)
        clipped_elements += int(np.count_nonzero(clipped != array))
        total_elements += int(array.size)
        bounded[key] = clipped
    return bounded, clipped_elements, total_elements


def observation_contract() -> dict[str, object]:
    return {
        "mode": "r13n_native_480x640_dinov3_30x40_to_6x8_plus_causal_team_v1",
        "encoder_name": "dinov3_vitb16_lvd",
        "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "weights_sha256": "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b",
        "config_sha256": "69256c4c142d59b0c0ccf5746542d9f2415f6c7db03bd7835a1f7b3afedb77fe",
        "preprocess": "dinov3_imagenet_rgb_resize_rectangular_antialias_v2",
        "input_height": 480,
        "input_width": 640,
        "require_native_input_shape": True,
        "feature_dim": 768,
        "encoder_patch_grid": [30, 40],
        "spatial_grid": [6, 8],
        "max_views": 5,
        "history_frames": 3,
        "compression_stage": "adaptive_average_after_full_resolution_dinov3",
        "primary_input": "manifest_selected_current_fixed_view_rgb_uint8_480x640",
        "supplemental_input": "from_scratch_causal_qpos_executed_action_and_coarse_rgb",
        "training_cache_equivalence": "frozen_dinov3_post_encoder_6x8_tokens",
        "fusion": "direct_query_bridge_no_scalar_gate",
        "fusion_heads": 4,
    }


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(data_root: str | Path, task: str, *, require_files: bool) -> dict:
    """Validate one pinned task manifest without assuming agent views exist."""

    if task not in TASK_SPECS:
        raise ValueError(f"unknown R13N task {task!r}")
    root = Path(data_root).resolve()
    path = root / task / "training_manifest.json"
    raw = path.read_bytes()
    payload = json.loads(raw)
    spec = TASK_SPECS[task]
    if payload.get("task", {}).get("id") != task:
        raise ValueError(f"R13N manifest task differs: {path}")
    if tuple(payload.get("vision", {}).get("camera_order", ())) != tuple(
        spec["camera_order"]
    ):
        raise ValueError(f"R13N camera order differs: {path}")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 150:
        raise ValueError(f"R13N episode count differs: {path}")
    split_counts = {
        split: sum(row.get("split") == split for row in episodes)
        for split in SPLIT_EPISODES
    }
    if split_counts != SPLIT_EPISODES:
        raise ValueError(f"R13N split counts differ: {path}")
    train_steps = sum(int(row["steps"]) for row in episodes if row["split"] == "train")
    if train_steps != int(spec["train_steps"]):
        raise ValueError(f"R13N train transition count differs: {path}")
    episode_bytes = 0
    if require_files:
        for row in episodes:
            episode = (path.parent / str(row["hdf5_path"])).resolve(strict=True)
            if not episode.is_relative_to(path.parent.resolve()):
                raise ValueError(f"R13N episode escaped task root: {episode}")
            if episode.stat().st_size <= 0 or len(str(row.get("hdf5_sha256", ""))) != 64:
                raise ValueError(f"R13N episode identity differs: {episode}")
            episode_bytes += episode.stat().st_size
        normalization = (path.parent / payload["normalization"]["path"]).resolve(strict=True)
        if not normalization.is_relative_to(path.parent.resolve()):
            raise ValueError("R13N normalization escaped task root")
    return {
        "task": task,
        "repo": spec["repo"],
        "revision": spec["revision"],
        "manifest": str(path),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "camera_order": list(spec["camera_order"]),
        "agents": int(spec["agents"]),
        "episodes": len(episodes),
        "split_counts": split_counts,
        "train_steps": train_steps,
        "episode_bytes": episode_bytes,
        "files_required": bool(require_files),
    }


__all__ = [
    "CAMERA_SENSOR_KEYS",
    "FULL_CACHE_PROTOCOL",
    "SPLIT_EPISODES",
    "TASKS",
    "TASK_SPECS",
    "camera_sensor_key",
    "clamp_action_to_space",
    "observation_contract",
    "sha256",
    "validate_manifest",
]
