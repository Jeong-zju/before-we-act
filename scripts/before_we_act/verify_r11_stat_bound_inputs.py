#!/usr/bin/env python3
"""Fast deploy-time revalidation against the prior 900-file full hash audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from before_we_act.train_r11_candidate import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--baseline-provenance", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.run_manifest.read_text())
    provenance = json.loads(args.baseline_provenance.read_text())
    if provenance.get("status") != "PASSED":
        raise ValueError("full baseline provenance receipt is not PASSED")
    expected_checkpoint = manifest["baseline"]["checkpoint_sha256"]
    if provenance.get("checkpoint", {}).get("sha256") != expected_checkpoint:
        raise ValueError("baseline provenance/checkpoint identity differs")
    checkpoint = Path(manifest["baseline"]["checkpoint"])
    if sha256_file(checkpoint) != expected_checkpoint:
        raise ValueError("baseline checkpoint changed after the full provenance audit")

    verified = {}
    for line in args.progress.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "hdf5_verified":
            verified[row["path"]] = row
    total = 0
    for task, expected in manifest["dataset"]["tasks"].items():
        manifest_path = Path(manifest["dataset"]["root"]) / task / "training_manifest.json"
        raw = manifest_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected["manifest_sha256"]:
            raise ValueError(f"{task} manifest changed after freeze")
        payload = json.loads(raw)
        for episode in payload["episodes"]:
            path = (manifest_path.parent / episode["hdf5_path"]).resolve(strict=True)
            row = verified.get(str(path))
            stat = path.stat()
            if not row or any(
                row.get(key) != value
                for key, value in {
                    "sha256": episode["hdf5_sha256"],
                    "device": stat.st_dev,
                    "inode": stat.st_ino,
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }.items()
            ):
                raise ValueError(f"HDF5 stat identity changed after full audit: {path}")
            total += 1
    if total != provenance.get("dataset", {}).get("verified_files"):
        raise ValueError("stat-bound HDF5 count differs from full audit")
    for task, expected_hash in manifest["validation20_seeds"]["sha256"].items():
        path = Path(manifest["validation20_seeds"]["root"]) / f"{task}.json"
        if sha256_file(path) != expected_hash:
            raise ValueError(f"{task} validation seed file changed after freeze")
    print(
        json.dumps(
            {
                "status": "PASSED",
                "checkpoint_sha256": expected_checkpoint,
                "stat_bound_hdf5_files": total,
                "seed_files": len(manifest["validation20_seeds"]["sha256"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
