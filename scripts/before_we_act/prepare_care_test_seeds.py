#!/usr/bin/env python3
"""Create the frozen CARE/RoboFactory test seeds for every configured task."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from before_we_act.care_training_data import atomic_json, canonical_sha256, sha256_file
from before_we_act.frozen_settings import load_frozen_settings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def deterministic_seeds(namespace: str, task: str, count: int) -> list[int]:
    selected: list[int] = []
    counter = 0
    while len(selected) < count:
        digest = hashlib.sha256(f"{namespace}|{task}|{counter}".encode()).digest()
        seed = int.from_bytes(digest[:8], "big") % 2_147_483_646 + 1
        counter += 1
        if seed not in selected:
            selected.append(seed)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.output_root / "receipt.json"
    if receipt_path.exists():
        print(json.dumps({"status": "PRESERVED", "receipt": str(receipt_path)}))
        return
    settings = load_frozen_settings(args.settings) if args.settings else load_frozen_settings()
    namespace = str(settings["closed_loop"]["test_seed_namespace"])
    count = int(settings["closed_loop"]["episodes_per_task"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    files = {}
    for task in settings["tasks"]:
        path = args.output_root / f"{task}.json"
        value = {
            "format_version": "before-we-act.care-robofactory-test-seeds/1",
            "task": task,
            "namespace": namespace,
            "count": count,
            "seeds": deterministic_seeds(namespace, task, count),
        }
        atomic_json(path, value)
        files[task] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    receipt = {
        "format_version": "before-we-act.care-robofactory-test-seed-receipt/1",
        "created_at_utc": utc_now(),
        "namespace": namespace,
        "episodes_per_task": count,
        "files": files,
        "manifest_sha256": canonical_sha256(files),
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps({"status": "CARE_TEST_SEEDS_READY", "receipt": str(receipt_path)}))


if __name__ == "__main__":
    main()
