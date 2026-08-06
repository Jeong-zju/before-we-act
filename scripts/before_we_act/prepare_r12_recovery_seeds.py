#!/usr/bin/env python3
"""Create immutable training-only rollout seeds disjoint from Gate20."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from before_we_act.data.raw_team_windows import TASKS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate20-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=4)
    args = parser.parse_args()
    if args.per_task < 1:
        raise ValueError("recovery seeds per task must be positive")
    if args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if payload.get("protocol") != "training_only_recovery_seeds_v1":
            raise ValueError("existing recovery seed identity differs")
        print(json.dumps({"reused": str(args.output)}, sort_keys=True))
        return
    tasks = {}
    for task_index, task in enumerate(TASKS):
        gate_path = args.gate20_root / "seeds" / f"{task}.json"
        gate_raw = gate_path.read_bytes()
        gate = set(map(int, json.loads(gate_raw)["seeds"]))
        rng = random.Random(20260806 + 1_000_003 * task_index)
        values = []
        while len(values) < args.per_task:
            seed = rng.randrange(1, 2**31 - 1)
            if seed not in gate and seed not in values:
                values.append(seed)
        tasks[task] = {
            "seeds": values,
            "gate20_seed_file": str(gate_path.resolve()),
            "gate20_seed_sha256": hashlib.sha256(gate_raw).hexdigest(),
            "overlap": sorted(set(values) & gate),
        }
    payload = {
        "schema_version": 1,
        "protocol": "training_only_recovery_seeds_v1",
        "base_seed": 20260806,
        "per_task": args.per_task,
        "tasks": tasks,
        "forbidden_use": "selection, validation or Gate20",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload | {"tasks": {key: value["seeds"] for key, value in tasks.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
