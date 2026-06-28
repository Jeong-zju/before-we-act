from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np


PHASE_TO_ID = {
    "approach": 0,
    "align": 1,
    "grasp": 2,
    "carry_to_passage": 3,
    "passage": 4,
    "carry_to_goal": 5,
    "release": 6,
    "done": 7,
    "failure": 8,
}

ID_TO_PHASE = {v: k for k, v in PHASE_TO_ID.items()}

FAILURE_TO_ID = {
    "none": 0,
    "timeout": 1,
    "force_violation": 2,
    "object_out_of_bounds": 3,
    "robot_out_of_bounds": 4,
    "object_dropped": 5,
    "robot_too_far": 6,
    "desync_in_passage": 7,
    "object_yaw_too_large": 8,
    "unknown": 99,
}

ID_TO_FAILURE = {v: k for k, v in FAILURE_TO_ID.items()}


def phase_to_id(phase: str) -> int:
    return PHASE_TO_ID.get(phase, PHASE_TO_ID["failure"])


def failure_to_id(reason: str) -> int:
    return FAILURE_TO_ID.get(reason, FAILURE_TO_ID["unknown"])


def ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x[:, None]
    return x


def write_dataset(group: h5py.Group, name: str, array: np.ndarray, compression: str | None = "gzip"):
    array = np.asarray(array)
    if array.dtype.kind in {"U", "O"}:
        dt = h5py.string_dtype(encoding="utf-8")
        group.create_dataset(name, data=array.astype(object), dtype=dt)
    else:
        if array.size == 0:
            group.create_dataset(name, data=array)
        else:
            group.create_dataset(name, data=array, compression=compression)


def save_stage2_episode(path: str | Path, episode: Dict[str, Any], meta: Dict[str, Any]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "stage2_v2"
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool, np.integer, np.floating)):
                f.attrs[k] = v

        obs = f.create_group("obs")
        r0 = obs.create_group("robot_0")
        r1 = obs.create_group("robot_1")
        write_dataset(r0, "proprio", episode["obs_robot_0"].astype(np.float32))
        write_dataset(r1, "proprio", episode["obs_robot_1"].astype(np.float32))

        actions = f.create_group("actions")
        joint_actions = episode["actions"].astype(np.float32)
        write_dataset(actions, "joint", joint_actions)
        write_dataset(actions, "robot_0", joint_actions[:, 0:4])
        write_dataset(actions, "robot_1", joint_actions[:, 4:8])

        global_group = f.create_group("global")
        write_dataset(global_group, "global_state", episode["global_state"].astype(np.float32))
        write_dataset(global_group, "object_pose", episode["object_pose"].astype(np.float32))
        write_dataset(global_group, "force_proxy", ensure_2d(episode["force_proxy"]).astype(np.float32))
        write_dataset(global_group, "contacts", ensure_2d(episode["contacts"]).astype(np.int32))
        write_dataset(global_group, "robot_distance", ensure_2d(episode["robot_distance"]).astype(np.float32))

        labels = f.create_group("labels")
        write_dataset(labels, "phase", ensure_2d(episode["phase"]).astype(np.int32))
        write_dataset(labels, "success", ensure_2d(episode["success"]).astype(np.float32))
        write_dataset(labels, "failure", ensure_2d(episode["failure"]).astype(np.float32))
        write_dataset(labels, "failure_reason", ensure_2d(episode["failure_reason"]).astype(np.int32))
        write_dataset(labels, "communication_dummy", episode["communication_dummy"].astype(np.float32))

        rewards = f.create_group("rewards")
        write_dataset(rewards, "reward", ensure_2d(episode["rewards"]).astype(np.float32))

        meta_group = f.create_group("meta")
        write_dataset(meta_group, "time", ensure_2d(episode["time"]).astype(np.float32))
        write_dataset(meta_group, "episode_index", np.asarray([meta.get("episode_index", -1)], dtype=np.int32))
        write_dataset(meta_group, "seed", np.asarray([meta.get("seed", -1)], dtype=np.int32))


def load_stage2_episode(path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    with h5py.File(path, "r") as f:
        return {
            "obs_robot_0": f["obs/robot_0/proprio"][:],
            "obs_robot_1": f["obs/robot_1/proprio"][:],
            "actions": f["actions/joint"][:],
            "global_state": f["global/global_state"][:],
            "object_pose": f["global/object_pose"][:],
            "force_proxy": f["global/force_proxy"][:].reshape(-1),
            "contacts": f["global/contacts"][:].reshape(-1),
            "robot_distance": f["global/robot_distance"][:].reshape(-1),
            "phase": f["labels/phase"][:].reshape(-1),
            "success": f["labels/success"][:].reshape(-1),
            "failure": f["labels/failure"][:].reshape(-1),
            "failure_reason": f["labels/failure_reason"][:].reshape(-1),
            "communication_dummy": f["labels/communication_dummy"][:],
            "rewards": f["rewards/reward"][:].reshape(-1),
            "time": f["meta/time"][:].reshape(-1),
        }
