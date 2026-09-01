"""Audit the frozen MARS-Control train/validation policy contract."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

from .download import REPOS

ROOT = Path(os.environ.get("MARS_OPENVLA_DATA_ROOT", "/workspace/datasets/mars_control"))
OUT = Path(os.environ.get("MARS_OPENVLA_RUN_ROOT", "/workspace/bwa_mars_openvla_runs")) / "audit/contract.json"
ARMS = {"place_cube_in_cup": 2, "strike_cube_hard": 2,
        "three_robots_place_shoes": 3, "four_robots_stack_cube": 4}
MAX_STEPS = {"place_cube_in_cup": 500, "strike_cube_hard": 500,
             "three_robots_place_shoes": 1200, "four_robots_stack_cube": 800}
LOW = np.asarray([-2.8973, -1.7628, -2.8973, -3.0718,
                  -2.8973, -0.0175, -2.8973, -1.0], np.float32)
HIGH = np.asarray([2.8973, 1.7628, 2.8973, -0.0698,
                   2.8973, 3.7525, 2.8973, 1.0], np.float32)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    report = {"schema": "mars-control.openvla.contract.v1", "status": "complete",
              "episodes": 0, "local_streams": 0, "local_transitions": 0,
              "tasks": {}, "validation_max_steps": MAX_STEPS,
              "policy_contract": "shared_weights_decentralized_local_rgb_qpos9_to_local_action8",
              "training_split_policy": "all_600_successful_episodes_no_split",
              "image_contract": "head_camera_agent{i}/rgb uint8 [0,255] -> OpenVLA processor",
              "action_contract": "clip absolute Panda joint7+gripper1 to environment bounds, then encode joint residual from chunk-start qpos plus absolute gripper",
              "action_encoding": "joint_residual_gripper_absolute",
              "residual_reference": "qpos_at_chunk_start_t"}
    all_qpos, all_actions = [], []
    encoded_min = np.full(8, np.inf, np.float32)
    encoded_max = np.full(8, -np.inf, np.float32)
    for task in REPOS:
        shards = sorted((ROOT / task / "motionplanning").glob("*.shard*.h5"))
        if len(shards) != 10:
            raise RuntimeError(f"{task}: expected 10 shards, got {len(shards)}")
        task_episodes = task_streams = task_steps = clipped = 0
        image_shapes, image_dtypes = set(), set()
        for path in shards:
            with h5py.File(path, "r") as h:
                trajectories = sorted(k for k in h if k.startswith("traj_"))
                for trajectory in trajectories:
                    group = h[trajectory]
                    if not bool(np.asarray(group["success"])[-1]):
                        raise RuntimeError(f"unsuccessful formal trajectory: {path}:{trajectory}")
                    lengths = []
                    for arm in range(ARMS[task]):
                        image = group[f"obs/sensor_data/head_camera_agent{arm}/rgb"]
                        qpos = group[f"obs/agent/panda-{arm}/qpos"]
                        action = group[f"actions/panda-{arm}"]
                        lengths.append(min(len(image), len(qpos), len(action)))
                    n = min(lengths)
                    if n < 1:
                        raise RuntimeError(f"empty trajectory: {path}:{trajectory}")
                    for arm in range(ARMS[task]):
                        image = group[f"obs/sensor_data/head_camera_agent{arm}/rgb"]
                        qpos = np.asarray(group[f"obs/agent/panda-{arm}/qpos"][:n], np.float32)
                        action = np.asarray(group[f"actions/panda-{arm}"][:n], np.float32)
                        if qpos.shape != (n, 9) or action.shape != (n, 8):
                            raise RuntimeError(f"shape drift: {task}:{trajectory}:arm{arm}")
                        sample = np.asarray(image[0])
                        if sample.dtype != np.uint8 or sample.ndim != 3 or sample.shape[-1] not in (3, 4):
                            raise RuntimeError(f"image codec drift: {task}:{sample.shape}:{sample.dtype}")
                        if not np.isfinite(qpos).all() or not np.isfinite(action).all():
                            raise RuntimeError(f"non-finite state/action: {task}:{trajectory}:arm{arm}")
                        clipped += int(np.count_nonzero(action != np.clip(action, LOW, HIGH)))
                        image_shapes.add(tuple(sample.shape)); image_dtypes.add(str(sample.dtype))
                        clipped_action = np.clip(action, LOW, HIGH)
                        all_qpos.append(qpos); all_actions.append(clipped_action)
                        # Match the actual eight-step training labels: every
                        # future joint command is relative to qpos at chunk
                        # start, with repeat-last terminal padding.
                        for block_start in range(0, n, 4096):
                            block_end = min(n, block_start + 4096)
                            t_idx = np.arange(block_start, block_end)[:, None]
                            h_idx = np.arange(8)[None, :]
                            idx = np.minimum(t_idx + h_idx, n - 1)
                            encoded = np.concatenate(
                                [
                                    clipped_action[idx, :7] - qpos[block_start:block_end, None, :7],
                                    clipped_action[idx, 7:8],
                                ],
                                axis=2,
                            )
                            encoded_min = np.minimum(encoded_min, encoded.min(axis=(0, 1)))
                            encoded_max = np.maximum(encoded_max, encoded.max(axis=(0, 1)))
                        task_streams += 1; task_steps += n
                    task_episodes += 1
        if task_episodes != 150:
            raise RuntimeError(f"{task}: expected 150 episodes, got {task_episodes}")
        report["tasks"][task] = {"episodes": task_episodes, "arms": ARMS[task],
            "local_streams": task_streams, "local_transitions": task_steps,
            "raw_action_values_outside_bounds": clipped,
            "image_shapes": [list(x) for x in sorted(image_shapes)], "image_dtypes": sorted(image_dtypes)}
        report["episodes"] += task_episodes
        report["local_streams"] += task_streams
        report["local_transitions"] += task_steps
    qpos, action = np.concatenate(all_qpos), np.concatenate(all_actions)
    report["normalization"] = {
        "type": "full clipped corpus min/max (no validation split)",
        "qpos_min": qpos.min(0).tolist(), "qpos_max": qpos.max(0).tolist(),
        "action_min": encoded_min.tolist(), "action_max": encoded_max.tolist(),
        "absolute_action_min": action.min(0).tolist(), "absolute_action_max": action.max(0).tolist(),
        "environment_action_low": LOW.tolist(), "environment_action_high": HIGH.tolist(),
    }
    if report["episodes"] != 600 or report["local_streams"] != 1650:
        raise RuntimeError("global episode/local-stream count drift")
    atomic_json(OUT, report)
    print(json.dumps({k: report[k] for k in ("status", "episodes", "local_streams", "local_transitions")}))


if __name__ == "__main__":
    main()
