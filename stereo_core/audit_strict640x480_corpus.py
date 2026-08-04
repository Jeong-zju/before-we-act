"""Reject any corpus that is not the formal v2 local-wrist RGB-D protocol.

The report is intentionally self-contained and uploaded alongside every task on
the Hub.  It makes it impossible to silently mix historical 320x240 data with
the native 640x480 five-task corpus again.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


RGB_SHAPE = (480, 640, 3)
DEPTH_SHAPE = (480, 640, 1)
GRID = (30, 40)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--data", required=True, help="HDF5 glob")
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    files = [Path(p) for p in sorted(glob.glob(args.data, recursive=True))]
    if not files:
        raise FileNotFoundError(args.data)
    trajectories, streams, per_file = [], 0, []
    for path in files:
        listed = []
        with h5py.File(path, "r") as h5:
            for key in sorted(k for k in h5 if k.startswith("traj_")):
                tr = h5[key]
                sensors = tr["obs"]["sensor_data"]
                if "head_camera_global" in sensors:
                    raise AssertionError(f"forbidden global observation: {path}:{key}")
                arms = sorted(k.removeprefix("panda-") for k in tr["actions"] if k.startswith("panda-"))
                if not arms:
                    raise AssertionError(f"no local arm streams: {path}:{key}")
                local = [f"head_camera_agent{arm}" for arm in arms]
                if sorted(sensors.keys()) != local:
                    raise AssertionError(f"not exactly local wrist cameras: {path}:{key}")
                for arm in arms:
                    sensor = sensors[f"head_camera_agent{arm}"]
                    rgb, depth = sensor["rgb"], sensor["depth"]
                    if tuple(rgb.shape[1:]) != RGB_SHAPE or tuple(depth.shape[1:]) != DEPTH_SHAPE:
                        raise AssertionError(
                            f"native 640x480 required at {path}:{key}:panda-{arm}; "
                            f"rgb={tuple(rgb.shape[1:])}, depth={tuple(depth.shape[1:])}"
                        )
                    if rgb.shape[0] != depth.shape[0] or rgb.shape[0] < len(tr["actions"][f"panda-{arm}"]):
                        raise AssertionError(f"RGB-D/action temporal mismatch: {path}:{key}:panda-{arm}")
                    probe = np.asarray(depth[: min(8, len(depth))], dtype=np.float32)
                    if not np.isfinite(probe).all() or not (probe > 0).any():
                        raise AssertionError(f"invalid metric depth: {path}:{key}:panda-{arm}")
                    streams += 1
                trajectories.append({"file": path.name, "trajectory": key, "arms": arms})
                listed.append(key)
        per_file.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path), "trajectories": listed})

    if len(trajectories) != args.expected_episodes:
        raise AssertionError(f"expected {args.expected_episodes} successful trajectories, found {len(trajectories)}")
    report = {
        "status": "PASS_STRICT640X480_V2",
        "task": args.task,
        "episodes": len(trajectories),
        "local_streams": streams,
        "policy_observation": "one local panda_hand RGB-D camera plus own qpos only",
        "native_rgb_shape": list(RGB_SHAPE),
        "native_depth_shape": list(DEPTH_SHAPE),
        "dino_vit_b16_grid": list(GRID),
        "defm_s14_input": [420, 560],
        "defm_s14_grid": list(GRID),
        "aligned_tokens": 1200,
        "files": per_file,
        "trajectories": trajectories,
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "task", "episodes", "local_streams", "aligned_tokens")}, indent=2))


if __name__ == "__main__":
    main()
