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

import h5py
import numpy as np


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

    active_manifest = active / "training_manifest.json"
    if sha256_file(active_manifest) != sha256_file(upstream_copy):
        raise RuntimeError(
            "HF stale manifest does not exactly describe the active pre-update dataset"
        )
    repaired = json.loads(upstream_copy.read_text(encoding="utf-8"))
    conversion_manifest_sha256 = sha256_file(staging / "manifest.json")
    upstream_conversion_manifest_sha256 = repaired["source"][
        "conversion_manifest_sha256"
    ]
    if sha256_file(staging / "normalization.npz") != sha256_file(
        active / "normalization.npz"
    ):
        raise RuntimeError("Place Food non-visual normalization unexpectedly changed")
    rows = repaired.get("episodes", [])
    train = [row for row in rows if row.get("split") == "train"]
    if len(rows) != 150 or len(train) != 120:
        raise RuntimeError("repaired Place Food split is not 120/15/15")
    shape_audit = 0
    non_visual_datasets_compared = 0
    for row in rows:
        path = staging / str(row["hdf5_path"])
        old_path = active / str(row["hdf5_path"])
        with h5py.File(path, "r") as source, h5py.File(old_path, "r") as old:
            def non_visual_names(handle: h5py.File) -> list[str]:
                result: list[str] = []

                def visitor(name: str, value: object) -> None:
                    components = name.split("/")
                    visual = any(
                        part == "images" or part.startswith("image_")
                        for part in components
                    )
                    if isinstance(value, h5py.Dataset) and not visual:
                        result.append(name)

                handle.visititems(visitor)
                return sorted(result)

            new_names = non_visual_names(source)
            old_names = non_visual_names(old)
            if new_names != old_names:
                raise RuntimeError(f"non-visual schema changed: {path}")
            for name in new_names:
                current = source[name]
                previous = old[name]
                if (
                    current.shape != previous.shape
                    or current.dtype != previous.dtype
                    or not np.array_equal(current[...], previous[...])
                ):
                    raise RuntimeError(f"non-visual data changed: {path}/{name}")
                non_visual_datasets_compared += 1
            for prefix in (
                "data/observation/images",
                "data/next_observation/images",
            ):
                images = source[prefix]
                if sorted(images.keys()) != ["agent_0", "agent_1", "global"]:
                    raise RuntimeError(f"unexpected Place Food views: {path}/{prefix}")
                for key in ("global", "agent_0", "agent_1"):
                    shape = images[key].shape
                    if shape[0] < int(row["steps"]) or shape[1:] != (480, 640, 3):
                        raise RuntimeError(
                            f"invalid original RGB shape: {path}/{prefix}/{key}={shape}"
                        )
                    shape_audit += 1
        row["hdf5_sha256"] = sha256_file(path)
        row["hdf5_size_bytes"] = path.stat().st_size

    repaired["vision"]["camera_order"] = ["global", "agent_0", "agent_1"]
    repaired["source"]["conversion_manifest_sha256"] = conversion_manifest_sha256
    repaired["integrity"].update(
        {
            "hf_upstream_conversion_manifest_identity_stale": (
                upstream_conversion_manifest_sha256 != conversion_manifest_sha256
            ),
            "hf_upstream_training_manifest_stale": True,
            "visual_revision_hdf5_hashes_recomputed": True,
            "visual_revision_non_visual_data_exact": True,
        }
    )
    atomic_json(upstream_manifest, repaired)
    repaired_manifest_sha256 = sha256_file(upstream_manifest)
    sidecar = upstream_manifest.with_suffix(upstream_manifest.suffix + ".sha256")
    temporary_sidecar = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    temporary_sidecar.write_text(
        f"{repaired_manifest_sha256}  {upstream_manifest.name}\n",
        encoding="utf-8",
    )
    os.replace(temporary_sidecar, sidecar)

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
        "repaired_training_manifest_sha256": repaired_manifest_sha256,
        "upstream_stale_hdf5_hash_entries": changed_hashes,
        "upstream_conversion_manifest_sha256": upstream_conversion_manifest_sha256,
        "repaired_conversion_manifest_sha256": conversion_manifest_sha256,
        "episodes": len(rows),
        "train_episodes": len(train),
        "views": ["global", "agent_0", "agent_1"],
        "original_rgb_shape": [480, 640, 3],
        "image_datasets_audited": shape_audit,
        "non_visual_datasets_exactly_compared": non_visual_datasets_compared,
        "hdf5_hashes_recomputed": len(rows),
        "normalization_unchanged_sha256": sha256_file(staging / "normalization.npz"),
        "hdf5_hashes_and_schema_verified_by": Path(__file__).name,
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
