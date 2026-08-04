#!/usr/bin/env python3
"""Freeze paired Gate20 seed files from the immutable S10 frozen100 reports."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


TASKS = (
    "lift_barrier",
    "camera_alignment",
    "three_robots_stack_cube",
    "long_pipeline_delivery",
    "take_photo",
)


def atomic_json(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline = Path(args.baseline_dir).resolve(strict=True)
    output = Path(args.output).resolve()
    seeds_dir = output / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": 1, "protocol": "R10-paired-Gate20", "tasks": {}}
    for task in TASKS:
        path = baseline / f"{task}.json"
        raw = path.read_bytes()
        payload = json.loads(raw)
        rows = payload.get("rows", [])
        if len(rows) != 100 or len({int(row["seed"]) for row in rows}) != 100:
            raise ValueError(f"{task} is not an immutable frozen100 report")
        selected = rows[:20]
        seed_payload = {
            "schema_version": 1,
            "task": task,
            "seeds": [int(row["seed"]) for row in selected],
            "selection_method": "first 20 rows of immutable S10 frozen100 report",
            "source": str(path),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
        }
        atomic_json(seed_payload, seeds_dir / f"{task}.json")
        summary["tasks"][task] = {
            "baseline_successes": sum(bool(row["success"]) for row in selected),
            "episodes": 20,
            "source_sha256": seed_payload["source_sha256"],
        }
    summary["macro_success_rate"] = (
        sum(value["baseline_successes"] for value in summary["tasks"].values()) / 100
    )
    atomic_json(summary, output / "baseline_gate20.json")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
