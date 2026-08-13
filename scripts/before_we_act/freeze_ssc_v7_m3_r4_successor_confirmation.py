#!/usr/bin/env python3
"""Validate fresh collection receipts and freeze the successor confirmation set."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
PREPARATION_GATE = (
    REPOSITORY / "docs/experiments/ssc_v7/m3_r4_successor_preparation_gate.json"
)
RUN_ROOT = Path(
    "/workspace/bwa_runs/ssc-v7-social-state-cooperation-v2/"
    "measurement/m3_r4_successor_a1_v1"
)
COLLECTION_ROOT = RUN_ROOT / "confirmation_collections"
OUTPUT_ROOT = RUN_ROOT / "confirmation_data"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    receipt_output = OUTPUT_ROOT / "confirmation_manifest_receipt.json"
    if receipt_output.exists() or OUTPUT_ROOT.exists():
        raise FileExistsError(f"fresh confirmation output required: {OUTPUT_ROOT}")
    gate = json.loads(PREPARATION_GATE.read_text(encoding="utf-8"))
    preparation_gate_sha = sha256_file(PREPARATION_GATE)
    base_path = Path(str(gate["data"]["training_manifest"]))
    if sha256_file(base_path) != str(gate["data"]["training_manifest_sha256"]):
        raise RuntimeError("training manifest hash mismatch")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    base_seeds = {int(item["seed"]) for item in base["episodes"]}
    base_hdf5 = {str(item["hdf5_sha256"]) for item in base["episodes"]}
    tasks = tuple(gate["confirmation_collection"]["first_unused_candidate_index_by_task"])
    required = int(gate["confirmation_collection"]["successful_episodes_per_task"])
    episodes: list[dict] = []
    source_receipts: list[dict] = []
    for task in tasks:
        receipt_path = COLLECTION_ROOT / task / "task_collection_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("stage_id") != gate["stage_id"]
            or receipt.get("task") != task
            or receipt.get("purpose") != "fresh_successor_confirmation"
            or receipt.get("gate_sha256") != preparation_gate_sha
            or int(receipt.get("test_paths_opened", -1)) != 0
            or receipt.get("r4_b_started") is not False
        ):
            raise RuntimeError(f"wrong collection receipt identity: {receipt_path}")
        if len(receipt["episodes"]) != required:
            raise RuntimeError(f"wrong confirmation episode count for {task}")
        source_receipts.append(
            {"task": task, "path": str(receipt_path), "sha256": sha256_file(receipt_path)}
        )
        for rank, raw in enumerate(receipt["episodes"]):
            item = deepcopy(raw)
            if int(item["success_rank"]) != rank:
                raise RuntimeError("non-canonical success order")
            if int(item["seed"]) in base_seeds:
                raise RuntimeError("confirmation seed overlaps existing data")
            if str(item["hdf5_sha256"]) in base_hdf5:
                raise RuntimeError("confirmation HDF5 overlaps existing data")
            hdf5_path = Path(str(item["hdf5_path"]))
            sidecar_path = Path(str(item["sidecar_path"]))
            if sha256_file(hdf5_path) != str(item["hdf5_sha256"]):
                raise RuntimeError("confirmation HDF5 hash mismatch")
            if sha256_file(sidecar_path) != str(item["sidecar_sha256"]):
                raise RuntimeError("confirmation sidecar hash mismatch")
            item["split"] = "confirmation"
            item["source_stage_id"] = gate["stage_id"]
            episodes.append(item)
    manifest = {
        "format_version": "ssc-v7.m3_r4_successor.confirmation_manifest/1",
        "stage_id": gate["stage_id"],
        "purpose": "fresh_successor_confirmation",
        "created_at_utc": utc_now(),
        "preparation_gate": str(PREPARATION_GATE),
        "preparation_gate_sha256": preparation_gate_sha,
        "source_receipts": source_receipts,
        "split_counts": {task: {"confirmation": required} for task in tasks},
        "episodes": episodes,
        "non_overlapping_with_training_manifest": True,
        "existing_tune_used": False,
        "read_only_test_used": False,
        "test_paths_opened": 0,
    }
    manifest_path = OUTPUT_ROOT / "confirmation_manifest.json"
    write_json(manifest_path, manifest)
    write_json(
        receipt_output,
        {
            "decision_code": "SSC_V7_M3_R4_SUCCESSOR_CONFIRMATION_DATA_FROZEN",
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "episode_count": len(episodes),
            "per_task_episode_count": required,
            "existing_tune_used": False,
            "test_paths_opened": 0,
            "r4_b_started": False,
        },
    )
    print("SSC_V7_M3_R4_SUCCESSOR_CONFIRMATION_DATA_FROZEN")


if __name__ == "__main__":
    main()
