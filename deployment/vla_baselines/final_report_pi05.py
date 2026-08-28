#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path("/workspace/bwa_pi05_runs")
TASKS = ("lift_barrier", "camera_alignment", "long_pipeline_delivery", "take_photo", "pass_shoe", "place_food")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    validation = json.loads((ROOT / "formal/pi05/validation20/summary.json").read_text())
    checkpoint = (ROOT / "formal/pi05/final").resolve()
    receipts = {}
    for task in TASKS:
        path = Path("/workspace/datasets/robofactory_multitask") / task / "download_receipt.json"
        row = json.loads(path.read_text())
        receipts[task] = {"repo_id": row["repo_id"], "revision": row["revision"], "episodes": row["episodes_total"], "bytes": row["bytes_total"]}
    payload = {
        "schema": "bwa.pi05.h200.delivery.v1",
        "status": "complete",
        "policy": "pi0.5_lora",
        "training": {
            "updates": 120000,
            "episodes": 900,
            "split": "all_episodes_ignore_manifest_split",
            "checkpoint": str(checkpoint),
            "protocol": "shared_weights_decentralized_local_rgb_qpos_to_local_action8",
        },
        "datasets": receipts,
        "smoke": json.loads((ROOT / "smoke/pi05/closed_loop/summary.json").read_text()),
        "validation20": validation,
        "supervisor_receipts": sorted(path.name for path in (ROOT / "supervisor/receipts").glob("*.json")),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    output = ROOT / "final_report.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (ROOT / "final_report.sha256").write_text(f"{sha256(output)}  {output.name}\n")


if __name__ == "__main__":
    main()
