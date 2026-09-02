#!/usr/bin/env python3
"""Fail-closed data, normalization, interface, and live simulator audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np
from transformers import AutoImageProcessor

from before_we_act.mars_temporal_data import MARS_TASKS, PD_ACTION_HIGH, PD_ACTION_LOW, clip_pd_action, compute_normalization, load_mars_episodes
from deployment.mars_care.common import TASK_BY_NAME, local_observation, make_env


EXPECTED_COMMIT = "2d34fb38c80cb06550a5dbf99abac2c89f4336ed"
EXPECTED_STEPS = {"place_cube_in_cup": 500, "strike_cube_hard": 500, "three_robots_place_shoes": 1200, "four_robots_stack_cube": 800}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--dino-model", type=Path, required=True)
    parser.add_argument("--robofactory-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "-C", str(args.robofactory_root), "rev-parse", "HEAD"], text=True).strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"RoboFactory revision drift: {commit}")
    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    if {key: int(value["max_steps"]) for key, value in settings["tasks"].items()} != EXPECTED_STEPS:
        raise RuntimeError("per-task maximum-step contract drift")
    if settings["dataset"].get("policy_training_split") != "all episodes" or settings["deployment"].get("shared_policy") is not True:
        raise RuntimeError("all-data/shared-policy contract drift")
    episodes = load_mars_episodes(args.raw_root)
    computed = compute_normalization(episodes, args.normalization.with_suffix(".recomputed.json"))
    if args.normalization.exists():
        existing = json.loads(args.normalization.read_text(encoding="utf-8"))
        for key in ("q_mean", "q_std", "a_mean", "a_std"):
            if not np.allclose(existing[key], computed[key], atol=1e-10, rtol=1e-10):
                raise RuntimeError(f"normalization drift: {key}")
    else:
        os.replace(args.normalization.with_suffix(".recomputed.json"), args.normalization)
    recomputed = args.normalization.with_suffix(".recomputed.json")
    if recomputed.exists():
        recomputed.unlink()
    stats = json.loads(args.normalization.read_text(encoding="utf-8"))
    if stats.get("action_encoding") != "absolute_pd_joint_pos" or int(stats.get("episodes", -1)) != 600:
        raise RuntimeError("normalization is not the all-data absolute-action contract")
    qmean, qstd = np.asarray(stats["q_mean"]), np.asarray(stats["q_std"])
    amean, astd = np.asarray(stats["a_mean"]), np.asarray(stats["a_std"])
    if qmean.shape != (9,) or amean.shape != (8,) or np.any(qstd <= 0) or np.any(astd <= 0):
        raise RuntimeError("normalization dimension/range drift")
    action_ranges = {task: [np.full(8, np.inf), np.full(8, -np.inf)] for task in MARS_TASKS}
    qpos_ranges = {task: [np.full(9, np.inf), np.full(9, -np.inf)] for task in MARS_TASKS}
    rgb_min, rgb_max = 255, 0
    for episode in episodes:
        with h5py.File(episode.path, "r", swmr=True) as handle:
            group = handle[episode.trajectory]
            for arm in episode.arms:
                action = np.asarray(group[f"actions/panda-{arm}"], dtype=np.float32)
                qpos = np.asarray(group[f"obs/agent/panda-{arm}/qpos"], dtype=np.float32)
                if action.ndim != 2 or action.shape[1] != 8 or qpos.ndim != 2 or qpos.shape[1] != 9:
                    raise RuntimeError(f"state/action shape drift: {episode.path}:{episode.trajectory}")
                if not np.isfinite(action).all() or not np.isfinite(qpos).all():
                    raise RuntimeError("non-finite state/action in corpus")
                clipped = clip_pd_action(action)
                action_ranges[episode.task][0] = np.minimum(action_ranges[episode.task][0], clipped.min(0))
                action_ranges[episode.task][1] = np.maximum(action_ranges[episode.task][1], clipped.max(0))
                qpos_ranges[episode.task][0] = np.minimum(qpos_ranges[episode.task][0], qpos.min(0))
                qpos_ranges[episode.task][1] = np.maximum(qpos_ranges[episode.task][1], qpos.max(0))
                image = np.asarray(group[f"obs/sensor_data/head_camera_agent{arm}/rgb"][0])
                if image.shape != (240, 320, 3) or image.dtype != np.uint8:
                    raise RuntimeError(f"RGB contract drift: {image.shape}/{image.dtype}")
                rgb_min, rgb_max = min(rgb_min, int(image.min())), max(rgb_max, int(image.max()))
    roundtrip = max(
        float(np.max(np.abs(((action_ranges[t][0] - amean) / astd) * astd + amean - action_ranges[t][0])))
        for t in MARS_TASKS
    )
    if roundtrip > 1e-6:
        raise RuntimeError(f"action normalization roundtrip error: {roundtrip}")
    live = {}
    for task_name in MARS_TASKS:
        spec = TASK_BY_NAME[task_name]
        env = make_env(spec, args.robofactory_root, render_device="gpu")
        try:
            observation, _ = env.reset(seed=20269999)
            bounds = {}
            for arm in range(spec.arms):
                image, qpos = local_observation(observation, arm)
                if image.shape != (240, 320, 3) or image.dtype != np.uint8 or qpos.shape != (9,):
                    raise RuntimeError(f"live local observation drift: {task_name}/{arm}")
                space = env.action_space.spaces[f"panda-{arm}"]
                low, high = np.asarray(space.low), np.asarray(space.high)
                if low.shape != (8,) or high.shape != (8,):
                    raise RuntimeError(f"live action-space drift: {task_name}/{arm}")
                observed_low, observed_high = action_ranges[task_name]
                if np.any(PD_ACTION_LOW < low - 1e-4) or np.any(PD_ACTION_HIGH > high + 1e-4):
                    raise RuntimeError(f"canonical Panda action limits exceed live bounds: {task_name}/{arm}")
                bounds[str(arm)] = {"low": low.tolist(), "high": high.tolist()}
            live[task_name] = {"arms": spec.arms, "max_steps": spec.max_steps, "action_bounds": bounds}
        finally:
            env.close()
    processor = AutoImageProcessor.from_pretrained(args.dino_model)
    if len(processor.image_mean) != 3 or len(processor.image_std) != 3 or np.any(np.asarray(processor.image_std) <= 0):
        raise RuntimeError("DINO image normalization drift")
    result = {
        "format_version": "before-we-act.care-mars-contract-audit/1",
        "status": "PASSED",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "robofactory_commit": commit,
        "episodes": len(episodes),
        "training_split": "all 600 episodes; none held out",
        "policy_interface": "shared strict-local weights; own RGB/qpos/action history only",
        "normalization": {"action_encoding": "absolute_pd_joint_pos", "decode_exactly_once": True, "clip_to_live_pd_joint_pos_before_fit": True, "roundtrip_max_abs": roundtrip},
        "image": {"source_dtype": "uint8", "source_shape": [240, 320, 3], "source_observed_range": [rgb_min, rgb_max], "model_scale": "float32 / 255", "dino_mean": processor.image_mean, "dino_std": processor.image_std},
        "action_ranges": {task: {"min": rows[0].tolist(), "max": rows[1].tolist()} for task, rows in action_ranges.items()},
        "qpos_ranges": {task: {"min": rows[0].tolist(), "max": rows[1].tolist()} for task, rows in qpos_ranges.items()},
        "live_simulator": live,
    }
    atomic_json(args.output, result)
    print("MARS_CARE_CONTRACT_AUDIT_PASSED", flush=True)


if __name__ == "__main__":
    main()
