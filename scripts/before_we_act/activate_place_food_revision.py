#!/usr/bin/env python3
"""Audit and atomically activate the corrected Place Food HF revision."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py


ROOT = Path(__file__).resolve().parents[2]
HF_REPO = "zeno-ai/robofactory-place-food-multiview"
HF_REVISION = "c912342823d41e3b1969311ec8c34e20aab22ea4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging",
        type=Path,
        default=Path(
            "/workspace/datasets/robofactory_multitask/"
            f".place_food-{HF_REVISION}"
        ),
    )
    parser.add_argument(
        "--active",
        type=Path,
        default=Path("/workspace/datasets/robofactory_multitask/place_food"),
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("/workspace/datasets/robofactory_multitask/.archive"),
    )
    parser.add_argument(
        "--python", default="/venv/robofactory-act/bin/python"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    staging = args.staging.resolve()
    active = args.active.resolve()
    already = active / "place_food_hf_activation_receipt.json"
    if not staging.exists() and already.is_file():
        receipt = json.loads(already.read_text(encoding="utf-8"))
        if receipt.get("hf_revision") == HF_REVISION and receipt.get("status") == "PASSED":
            print("PLACE_FOOD_REVISION_ALREADY_ACTIVE", HF_REVISION, flush=True)
            return
    if not staging.is_dir() or not active.is_dir():
        raise FileNotFoundError("staging and current active Place Food directories are required")

    hdf5_paths = sorted((staging / "hdf5").glob("episode_*.hdf5"))
    if len(hdf5_paths) != 150:
        raise RuntimeError(f"staging HDF5 count is {len(hdf5_paths)}, expected 150")
    upstream_manifest = staging / "training_manifest.json"
    upstream_copy = staging / f"upstream_training_manifest_{HF_REVISION}.json"
    shutil.copy2(upstream_manifest, upstream_copy)
    upstream = json.loads(upstream_copy.read_text(encoding="utf-8"))

    command = [
        args.python,
        str(ROOT / "scripts/prepare_robofactory_m1_training_artifacts.py"),
        "--dataset-dir", str(staging),
        "--transition-selection", "through-first-done-inclusive",
        "--split-seed", "7",
        "--expected-episodes", "150",
        "--expected-state-dim", "36",
        "--expected-action-dim", "16",
        "--expected-task-id", "place_food",
        "--expected-camera", "global",
        "--expected-camera", "agent_0",
        "--expected-camera", "agent_1",
        "--expected-fps", "20",
        "--action-codec",
        str(ROOT / "configs/action_codecs/robofactory_2panda_pd_joint_pos_16d.json"),
        "--overwrite",
    ]
    subprocess.run(command, cwd=ROOT, check=True)

    repaired = json.loads(upstream_manifest.read_text(encoding="utf-8"))
    rows = repaired.get("episodes", [])
    train = [row for row in rows if row.get("split") == "train"]
    if len(rows) != 150 or len(train) != 120:
        raise RuntimeError("repaired Place Food split is not 120/15/15")
    if repaired.get("vision", {}).get("camera_order") != [
        "global", "agent_0", "agent_1"
    ]:
        raise RuntimeError("repaired Place Food manifest does not expose three views")

    shape_audit = 0
    for row in rows:
        path = staging / str(row["hdf5_path"])
        with h5py.File(path, "r") as source:
            images = source["data/observation/images"]
            if sorted(images.keys()) != ["agent_0", "agent_1", "global"]:
                raise RuntimeError(f"unexpected Place Food views: {path}")
            for key in ("global", "agent_0", "agent_1"):
                shape = images[key].shape
                if shape[0] < int(row["steps"]) or shape[1:] != (480, 640, 3):
                    raise RuntimeError(f"invalid original RGB shape: {path}/{key}={shape}")
                shape_audit += 1

    old_by_index = {int(row["episode_index"]): row for row in upstream["episodes"]}
    changed_hashes = sum(
        old_by_index[int(row["episode_index"])].get("hdf5_sha256")
        != row.get("hdf5_sha256")
        for row in rows
    )
    if changed_hashes == 0:
        raise RuntimeError("HF training manifest was expected to be stale but no hash changed")

    tree_files = sorted((active / ".cache/huggingface/trees").glob("*.json"))
    previous_revision = tree_files[-1].stem if tree_files else "unknown"
    archive = args.archive_root.resolve() / f"place_food-hf-{previous_revision}"
    if archive.exists():
        raise FileExistsError(f"recoverable archive target already exists: {archive}")

    receipt = {
        "format_version": "before-we-act.place_food_hf_activation/1",
        "status": "PASSED",
        "hf_repo": HF_REPO,
        "hf_revision": HF_REVISION,
        "previous_hf_revision": previous_revision,
        "upstream_training_manifest_sha256": sha256_file(upstream_copy),
        "repaired_training_manifest_sha256": sha256_file(upstream_manifest),
        "upstream_stale_hdf5_hash_entries": changed_hashes,
        "episodes": len(rows),
        "train_episodes": len(train),
        "views": ["global", "agent_0", "agent_1"],
        "original_rgb_shape": [480, 640, 3],
        "image_datasets_audited": shape_audit,
        "hdf5_hashes_and_schema_verified_by": (
            "prepare_robofactory_m1_training_artifacts.py"
        ),
        "source_hdf5_bytes": sum(int(row["hdf5_size_bytes"]) for row in rows),
        "activated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "recoverable_previous_path": str(archive),
    }
    atomic_json(staging / "place_food_hf_activation_receipt.json", receipt)

    args.archive_root.mkdir(parents=True, exist_ok=True)
    os.rename(active, archive)
    try:
        os.rename(staging, active)
    except BaseException:
        os.rename(archive, active)
        raise
    print("PLACE_FOOD_REVISION_ACTIVATED", HF_REVISION, flush=True)


if __name__ == "__main__":
    main()

