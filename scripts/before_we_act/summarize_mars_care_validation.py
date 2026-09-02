#!/usr/bin/env python3
"""Summarize four-task MARS CARE Validation20 results."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from before_we_act.mars_temporal_data import MARS_TASKS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("selector_off", "care", "decentralized"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = {}
    successes = episodes = 0
    for task in MARS_TASKS:
        row = json.loads((args.root / f"{task}.json").read_text())
        if row.get("status") != "complete" or int(row.get("episodes", -1)) != 20:
            raise RuntimeError(f"incomplete Validation20: {task}")
        if row.get("mode") != args.mode:
            raise RuntimeError(f"Validation20 mode drift: {task}")
        tasks[task] = {
            "successes": int(row["successes"]),
            "episodes": 20,
            "success_rate": float(row["success_rate"]),
        }
        successes += int(row["successes"])
        episodes += 20
    result = {
        "format_version": "before-we-act.care-mars-validation20-summary/1",
        "status": "complete",
        "policy": "official_care_mars_bench_port",
        "mode": args.mode,
        "strict_local": True,
        "successes": successes,
        "episodes": episodes,
        "success_rate": successes / episodes,
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
