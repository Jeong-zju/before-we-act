#!/usr/bin/env python3
"""Freeze 20 fresh CARE closed-loop seeds per task before evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from before_we_act.care_training_data import atomic_json, canonical_sha256, sha256_file


TASKS = (
    "lift_barrier",
    "camera_alignment",
    "long_pipeline_delivery",
    "take_photo",
    "pass_shoe",
    "place_food",
)
NAMESPACE = "before-we-act/A6R1/owner-authorized-care-validation20-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def seeds_from(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    return {int(value) for value in json.loads(path.read_text(encoding="utf-8"))["seeds"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-validation-root", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.output_root / "receipt.json"
    if receipt_path.exists():
        print(json.dumps({"status": "PRESERVED", "receipt": str(receipt_path)}))
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    files = {}
    for task in TASKS:
        excluded = seeds_from(args.old_validation_root / f"{task}.json")
        excluded.update(seeds_from(args.confirmation_root / f"{task}.json"))
        selected: list[int] = []
        counter = 0
        while len(selected) < 20:
            digest = hashlib.sha256(f"{NAMESPACE}|{task}|{counter}".encode()).digest()
            seed = int.from_bytes(digest[:8], "big") % 2_147_483_646 + 1
            counter += 1
            if seed in excluded or seed in selected:
                continue
            selected.append(seed)
        value = {
            "format_version": "before-we-act.a6r1-care-validation20-seeds/1",
            "stage": "A7R1-CARE-OWNER-AUTHORIZED-VALIDATION20",
            "task": task,
            "namespace": NAMESPACE,
            "count": 20,
            "seeds": selected,
            "disjoint_from_old_validation20": True,
            "disjoint_from_bcore_confirmation50": True,
        }
        path = args.output_root / f"{task}.json"
        atomic_json(path, value)
        files[task] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    receipt = {
        "format_version": "before-we-act.a6r1-care-validation20-seed-receipt/1",
        "created_at_utc": utc_now(),
        "namespace": NAMESPACE,
        "closed_loop_results_seen": False,
        "files": files,
        "manifest_sha256": canonical_sha256(files),
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps({"status": "A6R1_VALIDATION20_SEEDS_FROZEN", "receipt": str(receipt_path)}))


if __name__ == "__main__":
    main()
