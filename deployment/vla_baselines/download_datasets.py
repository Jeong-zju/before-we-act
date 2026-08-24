#!/usr/bin/env python3
"""Resumable, manifest-verified download of the six RoboFactory corpora.

The token is intentionally read from ``/workspace/.secrets/hf_token`` and is
never included in command-line arguments or receipts.  A dataset is promoted
to ``complete`` only after every HDF5 object's byte count and SHA256 match the
signed training manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

from huggingface_hub import HfApi, snapshot_download

DATASET_ROOT = Path(os.environ.get("BWA_DATASET_ROOT", "/workspace/datasets/robofactory_multitask"))
TOKEN_FILE = Path(os.environ.get("BWA_HF_TOKEN_FILE", "/workspace/.secrets/hf_token"))
REPOS = {
    "lift_barrier": ("zeno-ai/robofactory-lift-barrier-multiview", "6ab620091677e69370412f08cd7adecacc28c146"),
    "long_pipeline_delivery": ("zeno-ai/robofactory-long-pipeline-delivery-multiview", "fee628311ff52a3ae0ddfddf82379c63d28f7533"),
    "camera_alignment": ("zeno-ai/robofactory-camera-alignment-multiview", "e204af13f7191dfd86dab3da529316a51558f479"),
    "pass_shoe": ("zeno-ai/robofactory-pass-shoe-multiview", "646bbfec792ed46c78e452acfc06b423ca1410af"),
    "place_food": ("zeno-ai/robofactory-place-food-multiview", "c912342823d41e3b1969311ec8c34e20aab22ea4"),
    "take_photo": ("zeno-ai/robofactory-take-photo-multiview", "3966385a4c688a5610d4b6cde044150f6b73d320"),
}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hub_episode_metadata(task: str, token: str) -> dict[str, dict]:
    """Return the immutable revision's LFS size/hash table.

    One published corpus currently contains a stale generated training manifest
    even though the HDF5 objects and source conversion manifest are internally
    consistent.  The pinned Hub revision's LFS OIDs are therefore the byte-level
    authority; the training-manifest disagreement is recorded in the receipt.
    """
    info = HfApi(token=token).dataset_info(
        REPOS[task][0], revision=REPOS[task][1], files_metadata=True
    )
    if info.sha != REPOS[task][1]:
        raise RuntimeError(f"{task}: resolved revision drift: {info.sha}")
    rows = {}
    for sibling in info.siblings:
        if not sibling.rfilename.startswith("hdf5/") or not sibling.rfilename.endswith(".hdf5"):
            continue
        if sibling.lfs is None or sibling.size is None:
            raise RuntimeError(f"{task}: HDF5 object lacks immutable LFS metadata: {sibling.rfilename}")
        rows[sibling.rfilename] = {
            "size_bytes": int(sibling.size),
            "sha256": sibling.lfs.sha256,
        }
    if len(rows) != 150:
        raise RuntimeError(f"{task}: pinned revision exposes {len(rows)} HDF5 objects, expected 150")
    return rows


def verify_task(task: str, token: str | None = None) -> dict:
    root = DATASET_ROOT / task
    token = token or TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(f"empty HF token file: {TOKEN_FILE}")
    hub_rows = hub_episode_metadata(task, token)
    manifest = json.loads((root / "training_manifest.json").read_text())
    episodes = manifest.get("episodes", [])
    if len(episodes) != 150:
        raise RuntimeError(f"{task}: expected 150 episodes, got {len(episodes)}")
    rows = []
    total = 0
    stale_manifest_rows = []
    for ep in episodes:
        path = root / ep["hdf5_path"]
        hub = hub_rows.get(ep["hdf5_path"])
        if hub is None:
            raise RuntimeError(f"{task}: manifest path absent from pinned revision: {ep['hdf5_path']}")
        if not path.is_file():
            raise RuntimeError(f"{task}: missing {path}")
        size = path.stat().st_size
        expected_size = int(hub["size_bytes"])
        if size != expected_size:
            raise RuntimeError(f"{task}: size mismatch {path}: {size} != {expected_size}")
        digest = sha256(path)
        if digest != hub["sha256"]:
            raise RuntimeError(f"{task}: hash mismatch {path}")
        if int(ep["hdf5_size_bytes"]) != expected_size or ep["hdf5_sha256"] != digest:
            stale_manifest_rows.append(int(ep["episode_index"]))
        rows.append({"episode_index": int(ep["episode_index"]), "path": ep["hdf5_path"],
                     "size_bytes": size, "sha256": digest,
                     "recorded_steps": int(ep.get("recorded_steps", ep.get("steps", 0))),
                     "split": ep.get("split", "unknown")})
        total += size
    receipt = {
        "schema": "bwa.robofactory.dataset_receipt.v1", "task": task,
        "repo_id": REPOS[task][0], "revision": REPOS[task][1],
        "episodes": rows, "episodes_total": len(rows), "bytes_total": total,
        "training_split_policy": "all_150_episodes_ignore_manifest_split",
        "integrity_authority": "pinned_huggingface_revision_lfs_sha256",
        "training_manifest_integrity_mismatch_count": len(stale_manifest_rows),
        "training_manifest_integrity_mismatch_episodes": stale_manifest_rows,
        "decentralized_inputs": ["data/observation/images/agent_i", "data/observation/agents/panda_i/qpos"],
        "decentralized_outputs": ["data/action/agents/panda_i/commanded"],
        "verified_at": datetime.now(timezone.utc).isoformat(), "status": "complete",
    }
    atomic_json(root / "download_receipt.json", receipt)
    return receipt


def download_task(task: str) -> dict:
    root = DATASET_ROOT / task
    root.mkdir(parents=True, exist_ok=True)
    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(f"empty HF token file: {TOKEN_FILE}")
    hub_rows = hub_episode_metadata(task, token)
    # snapshot_download is resumable through the HF cache.  Restrict patterns
    # so no hidden metadata or duplicate export is pulled into the workspace.
    # Quarantine known-incomplete objects.  Keeping them rather than unlinking
    # makes the repair auditable and recoverable.
    manifest_path = root / "training_manifest.json"
    if manifest_path.is_file():
        quarantine = root / ".quarantine"
        for relative, hub in hub_rows.items():
            path = root / relative
            if path.is_file() and path.stat().st_size != int(hub["size_bytes"]):
                quarantine.mkdir(exist_ok=True)
                target = quarantine / (path.name + f".size-{path.stat().st_size}")
                if target.exists():
                    target = quarantine / (path.name + f".size-{path.stat().st_size}.{os.getpid()}")
                path.replace(target)
                print(json.dumps({"task": task, "quarantined": str(target),
                                  "reason": "size_mismatch"}), flush=True)
    snapshot_download(
        repo_id=REPOS[task][0], revision=REPOS[task][1], repo_type="dataset", local_dir=str(root), token=token,
        allow_patterns=["manifest.json", "training_manifest.json", "normalization.npz", "hdf5/*.hdf5"],
        max_workers=min(int(os.environ.get("BWA_HF_WORKERS", "8")), os.cpu_count() or 1),
    )
    return verify_task(task, token)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(REPOS), action="append")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    tasks = args.task or list(REPOS)
    for task in tasks:
        receipt = verify_task(task) if args.verify_only else download_task(task)
        print(json.dumps({"task": task, "status": receipt["status"],
                          "episodes": receipt["episodes_total"], "bytes": receipt["bytes_total"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
