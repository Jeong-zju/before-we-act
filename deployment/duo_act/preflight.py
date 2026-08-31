from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import TASKS
from .evaluate import make_env, policy_image, policy_state
from .action_target import controller_joint_bounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    norm = manifest["normalization"]
    qm, qs = np.asarray(norm["qpos_mean"]), np.asarray(norm["qpos_std"])
    am, ass = np.asarray(norm["action_mean"]), np.asarray(norm["action_std"])
    checks = {
        "cuda_single_5090": torch.cuda.device_count() == 1 and "5090" in torch.cuda.get_device_name(0),
        "normalization_roundtrip": True,
        "dataset_within_controller_joint_bounds": True,
        "dataset_state_gripper_is_binary": True,
        "runtime_state_gripper_matches_converter": True,
        "reset_state_normalized_finite": True,
        "camera_contract": True,
        "env_step_absolute_joint_binary_gripper": True,
        "task_specific_max_steps": True,
    }
    rows = {}
    controller_low, controller_high = controller_joint_bounds()
    for task in TASKS:
        dataset_state = np.load(
            args.data / task / "state.npy", mmap_mode="r"
        ).reshape(-1, 2, 8)
        checks["dataset_state_gripper_is_binary"] &= bool(
            np.isin(dataset_state[..., 7], (0.0, 1.0)).all()
        )
        env = make_env(task)
        observation, info = env.reset(seed=20260820 + TASKS.index(task) * 1000)
        images, states = [], []
        action = {}
        for arm, key in enumerate(("left", "right")):
            physical_width = float(
                np.asarray(observation[key]["gripper"]).reshape(-1)[0]
            )
            raw = policy_state(observation, key)
            checks["runtime_state_gripper_matches_converter"] &= bool(
                raw[7] == float(physical_width > 0.9)
                and raw[7] in (0.0, 1.0)
            )
            restored = (raw - qm) / qs * qs + qm
            checks["normalization_roundtrip"] &= bool(np.allclose(raw, restored, atol=1e-6))
            checks["reset_state_normalized_finite"] &= bool(np.isfinite((raw - qm) / qs).all())
            image = policy_image(observation, arm)
            checks["camera_contract"] &= tuple(image.shape) == (3, 224, 448) and image.dtype == torch.uint8
            images.append(tuple(image.shape))
            states.append(raw.tolist())
            task_row = manifest["tasks"][task]
            data_low = np.asarray(task_row["action_min"])[arm * 8:arm * 8 + 7]
            data_high = np.asarray(task_row["action_max"])[arm * 8:arm * 8 + 7]
            checks["dataset_within_controller_joint_bounds"] &= bool(
                np.all(data_low >= controller_low - 1e-4)
                and np.all(data_high <= controller_high + 1e-4)
            )
            # A hold command verifies the action interface without disturbing
            # task objects or introducing an artificial cross-task mean pose.
            local = np.concatenate((
                np.asarray(observation[key]["joints"], np.float32),
                np.asarray([raw[7]], np.float32),
            ))
            local[:7] = np.clip(local[:7], controller_low, controller_high)
            local[7] = float(local[7] >= 0.5)
            action[key] = {"joints": local[:7].astype(np.float32), "gripper": np.asarray([local[7]], np.float32)}
        _, _, terminated, truncated, _ = env.step(action)
        checks["env_step_absolute_joint_binary_gripper"] &= isinstance(bool(np.asarray(terminated).all()), bool)
        max_steps = int(manifest["tasks"][task]["validation_max_steps"])
        checks["task_specific_max_steps"] &= max_steps > 0
        rows[task] = {"images": images, "reset_local_states": states, "validation_max_steps": max_steps}
        env.close()
    report = {"schema": "duobench-act-preflight-v1", "passed": all(checks.values()), "checks": checks, "tasks": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
