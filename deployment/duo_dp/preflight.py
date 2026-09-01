from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from deployment.duo_act.action_target import (
    ACTION_TARGET_CONTRACT_SHA256,
    canonicalize_controller_action,
    validate_controller_action,
)
from deployment.duo_act.protocol import (
    FORMAL_DATASET_REVISION,
    VALIDATION_MAX_STEPS,
)

from .common import HORIZON, IMAGE_SIZE, OBS_STEPS, TASKS, atomic_json
from .dataset import DuoDPDataset, compute_corpus_stats, episode_bounds
from .modeling import build_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-model", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    checks = {
        "dataset_revision": manifest.get("dataset_revision") == FORMAL_DATASET_REVISION,
        "eleven_tasks": tuple(manifest.get("tasks", ())) == TASKS,
        "all_550_episodes": manifest.get("total_episodes") == 550,
        "all_285988_frames": manifest.get("total_frames") == 285988,
        "causal_lag_one": manifest.get("recording_alignment", {}).get("action_lag_rows") == 1,
        "controller_action_contract": manifest.get("action_target_contract", {}).get("sha256")
        == ACTION_TARGET_CONTRACT_SHA256,
        "validation_horizons": manifest.get("validation_horizon", {}).get("per_task_max_steps")
        == VALIDATION_MAX_STEPS,
        "arrays": True,
        "episode_boundaries": True,
        "binary_gripper": True,
        "controller_action_range": True,
        "rgb_contract": True,
        "corpus_minmax": True,
        "dataset_sample_contract": True,
        "model_loss_finite": not args.check_model,
        "model_inference_contract": not args.check_model,
    }
    for task in TASKS:
        arrays = {
            key: np.load(args.data / task / f"{key}.npy", mmap_mode="r")
            for key in ("state", "action", "head", "left", "right", "episodes")
        }
        n = len(arrays["state"])
        checks["arrays"] &= arrays["state"].shape == (n, 16) and arrays["action"].shape == (n, 16)
        checks["arrays"] &= all(len(arrays[key]) == n for key in ("head", "left", "right", "episodes"))
        starts, ends = episode_bounds(arrays["episodes"])
        checks["episode_boundaries"] &= len(starts) == 50 and bool(np.all(ends > starts + 1))
        state = arrays["state"].reshape(-1, 2, 8)
        action = arrays["action"].reshape(-1, 2, 8)
        checks["binary_gripper"] &= bool(np.isin(state[..., 7], (0, 1)).all())
        checks["binary_gripper"] &= bool(np.isin(action[..., 7], (0, 1)).all())
        try:
            validate_controller_action(np.asarray(action))
        except ValueError:
            checks["controller_action_range"] = False
        checks["rgb_contract"] &= all(
            arrays[key].dtype == np.uint8 and arrays[key].shape[1:] == (IMAGE_SIZE, IMAGE_SIZE, 3)
            for key in ("head", "left", "right")
        )
    stats = compute_corpus_stats(args.data)
    checks["corpus_minmax"] &= stats["episodes"] == 550
    checks["corpus_minmax"] &= stats["indexed_local_samples"] == 570876
    checks["corpus_minmax"] &= bool(
        np.all(np.asarray(stats["q_max"]) > np.asarray(stats["q_min"]))
        and np.all(np.asarray(stats["a_max"]) > np.asarray(stats["a_min"]))
    )
    dataset = DuoDPDataset(args.data)
    samples = []
    for task_id in range(len(TASKS)):
        stream = dataset.task_streams[task_id][0]
        sample = dataset[(*stream, stream[2])]
        samples.append(sample)
        checks["dataset_sample_contract"] &= sample["head_wrist"].shape == (
            OBS_STEPS,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE * 2,
        )
        checks["dataset_sample_contract"] &= sample["agent_pos"].shape == (OBS_STEPS, 8)
        checks["dataset_sample_contract"] &= sample["action"].shape == (HORIZON, 8)
    model_details = {}
    if args.check_model:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("DuoBench DP preflight requires the reserved single GPU")
        device = torch.device("cuda:0")
        policy = build_policy(stats, device, inference_steps=2)
        chosen = samples[:2]
        batch = {
            "obs": {
                "head_wrist": torch.stack([x["head_wrist"] for x in chosen]).to(device).float().div_(255),
                "agent_pos": torch.stack([x["agent_pos"] for x in chosen]).to(device),
            },
            "action": torch.stack([x["action"] for x in chosen]).to(device),
        }
        policy.train()
        loss = policy.compute_loss(batch)
        checks["model_loss_finite"] = bool(torch.isfinite(loss))
        policy.eval()
        with torch.inference_mode():
            prediction = policy.predict_action(batch["obs"])["action"]
        decoded = prediction.float().cpu().numpy()
        checks["model_inference_contract"] = decoded.shape == (2, 6, 8) and bool(np.isfinite(decoded).all())
        for row in decoded.reshape(-1, 8):
            validate_controller_action(canonicalize_controller_action(row))
        model_details = {
            "loss": float(loss),
            "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
            "prediction_shape": list(decoded.shape),
            "gpu": torch.cuda.get_device_name(0),
        }
    report = {
        "schema": "duobench.dp.preflight.v1",
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "passed": all(checks.values()),
        "checks": checks,
        "stats": stats,
        "model": model_details,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
