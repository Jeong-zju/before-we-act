#!/usr/bin/env python3
"""Freeze disjoint Stack discovery/validation/final seed manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


TASK = "three_robots_stack_cube"
SPLITS = {
    "gate20": (0, 20),
    "discovery20": (20, 40),
    "validation20": (40, 60),
    "reserve20": (60, 80),
    "final20": (80, 100),
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def prepare(frozen100_path: Path, gate20_path: Path, output: Path) -> dict:
    frozen_raw = frozen100_path.read_bytes()
    gate_raw = gate20_path.read_bytes()
    frozen = json.loads(frozen_raw)
    gate = json.loads(gate_raw)
    rows = frozen.get("rows", [])
    seeds = [int(row["seed"]) for row in rows]
    if frozen.get("task") != TASK or len(rows) != 100 or len(set(seeds)) != 100:
        raise ValueError("Stack source is not the immutable 100-episode report")
    gate_seeds = [int(seed) for seed in gate.get("seeds", [])]
    if gate.get("task") != TASK or gate_seeds != seeds[:20]:
        raise ValueError("existing Stack Gate20 is not frozen100 rows 0:20")

    source_sha = sha256(frozen_raw)
    gate_sha = sha256(gate_raw)
    manifest = {
        "schema_version": 1,
        "round": "R15-Evolution",
        "task": TASK,
        "source": str(frozen100_path.resolve()),
        "source_sha256": source_sha,
        "gate20_source": str(gate20_path.resolve()),
        "gate20_source_sha256": gate_sha,
        "selection_rule": "preserve frozen100 row order; five fixed non-overlapping blocks",
        "model_selection_policy": (
            "discovery20 may guide design; validation20 may select one recipe; "
            "reserve20 is held for confirmation and final20 remains blind until promotion"
        ),
        "splits": {},
    }
    all_selected: list[int] = []
    for name, (start, stop) in SPLITS.items():
        selected_rows = rows[start:stop]
        selected = seeds[start:stop]
        payload = {
            "schema_version": 1,
            "round": "R15-Evolution",
            "task": TASK,
            "split": name,
            "seeds": selected,
            "selection_method": f"immutable frozen100 rows {start}:{stop}",
            "source": str(frozen100_path.resolve()),
            "source_sha256": source_sha,
            "source_row_range": [start, stop],
            "source_baseline_successes": sum(
                bool(row.get("success")) for row in selected_rows
            ),
        }
        split_path = output / f"{name}.json"
        atomic_json(payload, split_path)
        manifest["splits"][name] = {
            "path": str(split_path.resolve()),
            "sha256": sha256(split_path.read_bytes()),
            "episodes": len(selected),
            "source_baseline_successes": payload["source_baseline_successes"],
        }
        all_selected.extend(selected)
    if len(all_selected) != 100 or len(set(all_selected)) != 100:
        raise AssertionError("R15 Stack protocol splits are not exhaustive and disjoint")
    atomic_json(manifest, output / "protocol.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen100", type=Path, required=True)
    parser.add_argument("--gate20", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(
        args.frozen100.resolve(strict=True),
        args.gate20.resolve(strict=True),
        args.output.resolve(),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
