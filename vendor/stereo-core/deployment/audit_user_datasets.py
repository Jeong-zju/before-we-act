#!/usr/bin/env python3
"""Validate downloaded WAM manifests, file sizes, and no-wrist HDF5 contracts."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py


TASKS = (
    "lift_barrier",
    "camera_alignment",
    "three_robots_stack_cube",
    "long_pipeline_delivery",
    "take_photo",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/datasets/robofactory_multitask")
    parser.add_argument("--output", default="/workspace/logs/user_dataset_audit.json")
    parser.add_argument("--hash", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    report = {"root": str(root), "full_sha256": args.hash, "tasks": {}}
    for task in TASKS:
        task_root = root / task
        if not (task_root / ".download-complete").is_file():
            raise FileNotFoundError(f"download completion marker missing: {task}")
        manifest_path = task_root / "training_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["task"]["id"] != task:
            raise ValueError(f"task mismatch: {task}")
        split_counts = Counter()
        total_bytes = 0
        for index, row in enumerate(manifest["episodes"], 1):
            path = task_root / row["hdf5_path"]
            if path.stat().st_size != int(row["hdf5_size_bytes"]):
                raise ValueError(f"size mismatch: {path}")
            if args.hash and sha256(path) != row["hdf5_sha256"]:
                raise ValueError(f"sha256 mismatch: {path}")
            with h5py.File(path, "r") as handle:
                data = handle["data"]
                length = int(row["steps"])
                cameras = data["observation"]["images"]
                if cameras["global"].shape[1:] != (480, 640, 3):
                    raise ValueError(f"bad global RGB shape: {path}")
                arm_count = manifest["action"]["dimension"] // 8
                for arm in range(arm_count):
                    if cameras[f"agent_{arm}"].shape[1:] != (480, 640, 3):
                        raise ValueError(f"bad agent RGB shape: {path}:agent_{arm}")
                    if data["observation"]["agents"][f"panda_{arm}"]["qpos"].shape[1:] != (9,):
                        raise ValueError(f"bad qpos shape: {path}:panda_{arm}")
                    if data["action"]["agents"][f"panda_{arm}"]["commanded"].shape[1:] != (8,):
                        raise ValueError(f"bad action shape: {path}:panda_{arm}")
                if length < 1 or length > len(data["done"]):
                    raise ValueError(f"bad selected length: {path}")
            split_counts[row["split"]] += 1
            total_bytes += path.stat().st_size
            if index % 25 == 0:
                print(json.dumps({"task": task, "audited": index, "total": 150}), flush=True)
        if dict(split_counts) != {"train": 120, "validation": 15, "test": 15}:
            raise ValueError(f"wrong split counts for {task}: {dict(split_counts)}")
        report["tasks"][task] = {
            "episodes": len(manifest["episodes"]),
            "split_counts": dict(split_counts),
            "bytes": total_bytes,
            "manifest_sha256": sha256(manifest_path),
            "cameras": manifest["vision"]["camera_order"],
        }
    report["total_bytes"] = sum(item["bytes"] for item in report["tasks"].values())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
