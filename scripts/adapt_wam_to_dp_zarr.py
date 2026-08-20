#!/usr/bin/env python3
"""Stream the shared WAM HDF5 contract into RoboFactory DP zarr format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import zarr


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--agent", type=int, default=0)
    p.add_argument("--allow-missing", action="store_true",
                   help="Emit zero placeholders and a valid mask for tasks without this agent")
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    root = zarr.open_group(str(args.output), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    images = data.create_dataset("head_camera", shape=(0, args.height, args.width, 3), chunks=(32, args.height, args.width, 3), dtype="u1", compressor=compressor)
    states = data.create_dataset("state", shape=(0, 9), chunks=(256, 9), dtype="f4", compressor=compressor)
    actions = data.create_dataset("action", shape=(0, 8), chunks=(256, 8), dtype="f4", compressor=compressor)
    episode_ends: list[int] = []
    task_names: list[str] = []
    valid_agents: list[bool] = []
    total = 0
    for task_root in sorted(args.data_root.iterdir()):
        manifest_path = task_root / "training_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        codec = manifest["action"]["codec"]["config"]
        low = np.asarray(codec["low"], np.float32)[args.agent * 8:(args.agent + 1) * 8]
        high = np.asarray(codec["high"], np.float32)[args.agent * 8:(args.agent + 1) * 8]
        records = [record for record in manifest["episodes"] if record.get("split") == "train"]
        for record in records:
            path = task_root / record["hdf5_path"]
            with h5py.File(path, "r") as file:
                group = file["data"]
                image_key = f"agent_{args.agent}"
                panda_key = f"panda_{args.agent}"
                available = (image_key in group["observation"]["images"]
                             and panda_key in group["observation"]["agents"]
                             and panda_key in group["action"]["agents"])
                steps = int(record.get("recorded_steps", record["steps"]))
                if not available:
                    if not args.allow_missing:
                        continue
                    image = np.zeros((steps, args.height, args.width, 3), dtype=np.uint8)
                    state = np.zeros((steps, 9), dtype=np.float32)
                    action = np.zeros((steps, 8), dtype=np.float32)
                else:
                    steps = min(steps, len(group["observation"]["images"][image_key]))
                    image = group["observation"]["images"][image_key][:steps]
                    state = group["observation"]["agents"][panda_key]["qpos"][:steps].astype(np.float32)
                    action = group["action"]["agents"][panda_key]["commanded"][:steps].astype(np.float32)
            if (image.shape[1], image.shape[2]) != (args.height, args.width):
                image = image[:, :: max(image.shape[1] // args.height, 1), :: max(image.shape[2] // args.width, 1)]
                image = image[:, :args.height, :args.width]
            if available:
                action = np.clip(2.0 * (action - low) / (high - low) - 1.0, -1.0, 1.0)
            start = total
            end = start + len(image)
            images.resize((end, args.height, args.width, 3))
            states.resize((end, 9))
            actions.resize((end, 8))
            images[start:end] = image
            states[start:end] = state
            actions[start:end] = action
            total = end
            episode_ends.append(total)
            task_names.append(task_root.name)
            valid_agents.append(bool(available))
            if len(episode_ends) % 10 == 0:
                print(f"episodes={len(episode_ends)} frames={total}", flush=True)
    meta.create_dataset("episode_ends", data=np.asarray(episode_ends, dtype=np.int64), compressor=compressor)
    meta.attrs["tasks"] = task_names
    meta.attrs["valid_agent"] = valid_agents
    meta.attrs["format"] = "before-we-act.dp.robofactory/1"
    print(json.dumps({"episodes": len(episode_ends), "frames": total, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
