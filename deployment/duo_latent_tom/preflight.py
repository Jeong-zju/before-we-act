from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .common import ACTION_ENCODING, DATASET_REVISION, POLICY_CONTRACT, TASKS, VALIDATION_MAX_STEPS, atomic_json, load_config
from .evaluate import make_env, local_observation


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(); manifest = json.loads((args.data / "manifest.json").read_text())
    checks = {
        "cuda_available_single_gpu": bool(torch.cuda.is_available() and torch.cuda.device_count() == 1),
        "all_tasks": list(manifest.get("tasks", {})) == list(TASKS),
        "dataset_revision": manifest.get("dataset_revision") == DATASET_REVISION,
        "all_episodes": manifest.get("total_episodes") == 550,
        "all_frames": manifest.get("total_frames") == 285988,
        "causal_pairs": manifest.get("total_policy_samples") == 285438,
        "normalization_finite": True,
        "normalization_std_positive": True,
        "array_contract": True,
        "task_horizons": True,
        "runtime_contract": True,
    }
    norm = manifest.get("normalization", {})
    for key in ("qpos_mean", "qpos_std", "qpos_min", "qpos_max", "action_mean", "action_std", "action_min", "action_max"):
        values = np.asarray(norm.get(key, []), dtype=np.float64)
        checks["normalization_finite"] &= bool(values.shape == (8,) and np.isfinite(values).all())
    q_std = np.asarray(norm.get("qpos_std", []), dtype=float)
    a_std = np.asarray(norm.get("action_std", []), dtype=float)
    checks["normalization_std_positive"] &= bool(q_std.shape == (8,) and q_std.min() >= 1e-4)
    checks["normalization_std_positive"] &= bool(a_std.shape == (8,) and a_std.min() >= 1e-4)
    task_rows = {}
    for task in TASKS:
        task_dir = args.data / task
        arrays = {name: np.load(task_dir / f"{name}.npy", mmap_mode="r") for name in ("state", "action", "head", "left", "right", "episodes")}
        n = len(arrays["state"])
        checks["array_contract"] &= arrays["state"].shape == (n, 16) and arrays["action"].shape == (n, 16)
        checks["array_contract"] &= all(arrays[name].shape[0] == n for name in ("head", "left", "right", "episodes"))
        checks["array_contract"] &= all(arrays[name].dtype == np.uint8 for name in ("head", "left", "right"))
        checks["array_contract"] &= bool(np.isin(np.asarray(arrays["action"][:, (7, 15)]), (0.0, 1.0)).all())
        checks["task_horizons"] &= manifest["tasks"][task]["validation_max_steps"] == VALIDATION_MAX_STEPS[task]
        task_rows[task] = {"frames": n, "episodes": int(len(np.unique(arrays["episodes"]))), "shape_head": list(arrays["head"].shape[1:]), "validation_max_steps": VALIDATION_MAX_STEPS[task]}
    # Reset/hold-step every task to catch simulator assets, action semantics,
    # gripper binarization, camera layout, and task-specific horizon wiring.
    runtime_rows = {}
    os.environ.setdefault("MUJOCO_GL", "egl")
    for task in TASKS:
        env = None
        try:
            env = make_env(task); observation, _ = env.reset(seed=20260820 + TASKS.index(task) * 1000)
            own = [local_observation(observation, arm, __import__("collections").deque(maxlen=2), __import__("collections").deque(maxlen=2), TASKS.index(task), torch.device("cpu")) for arm in range(2)]
            action = {}
            for arm, key in enumerate(("left", "right")):
                q = own[arm]["qpos"][0, -1].numpy()
                local = np.concatenate((q[:7], [q[7]])).astype(np.float32)
                action[key] = {"joints": local[:7], "gripper": np.asarray([local[7]], dtype=np.float32)}
            env.step(action)
            runtime_rows[task] = {"image_shape": list(own[0]["image"].shape), "qpos_shape": list(own[0]["qpos"].shape), "task_shape": list(own[0]["task"].shape), "arm_id_shape": list(own[0]["arm_id"].shape)}
            checks["runtime_contract"] &= own[0]["image"].shape == (1, 2, 3, 224, 448) and own[0]["qpos"].shape == (1, 2, 8) and own[0]["arm_id"].shape == (1, 2)
        except Exception as error:
            checks["runtime_contract"] = False; runtime_rows[task] = {"error": f"{type(error).__name__}: {error}"}
        finally:
            if env is not None: env.close()
    report = {"schema": "duobench.latent-tom.preflight.v1", "status": "complete" if all(checks.values()) else "failed", "passed": all(checks.values()), "checks": checks, "policy_contract": POLICY_CONTRACT, "action_encoding": ACTION_ENCODING, "tasks": task_rows, "runtime": runtime_rows, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    atomic_json(args.output, report); print(json.dumps(report))
    if not report["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
