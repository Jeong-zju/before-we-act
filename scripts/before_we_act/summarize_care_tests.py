#!/usr/bin/env python3
"""Write a hashable, paired CARE/RoboFactory test record without conclusions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from before_we_act.care_training_data import atomic_json, sha256_file
from before_we_act.frozen_settings import load_frozen_settings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_task(root: Path, mode: str, task: str, episodes: int) -> dict[str, Any]:
    path = root / mode / f"{task}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("mode") != mode or value.get("task") != task or value.get("episodes") != episodes:
        raise RuntimeError(f"incomplete CARE test artifact: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--offline-report", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--seed-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        print(json.dumps({"status": "PRESERVED", "output": str(args.output)}))
        return
    settings = load_frozen_settings(args.settings) if args.settings else load_frozen_settings()
    episodes = int(settings["closed_loop"]["episodes_per_task"])
    modes = list(settings["closed_loop"]["modes"])
    task_results: dict[str, Any] = {}
    totals = {mode: {"successes": 0, "steps": 0, "override_steps": 0} for mode in modes}
    for task in settings["tasks"]:
        values = {mode: load_task(args.test_root, mode, task, episodes) for mode in modes}
        seeds = [{int(row["seed"]) for row in values[mode]["rows"]} for mode in modes]
        if any(seed_set != seeds[0] for seed_set in seeds[1:]):
            raise RuntimeError(f"paired test seeds differ: {task}")
        task_results[task] = {}
        for mode, value in values.items():
            successes = int(value["successes"])
            steps = int(value["steps"])
            overrides = int(value["override_steps"])
            totals[mode]["successes"] += successes
            totals[mode]["steps"] += steps
            totals[mode]["override_steps"] += overrides
            task_results[task][mode] = {
                "successes": successes,
                "episodes": episodes,
                "steps": steps,
                "override_steps": overrides,
                "artifact_sha256": sha256_file(args.test_root / mode / f"{task}.json"),
            }
    for values in totals.values():
        values["episodes"] = episodes * len(settings["tasks"])
        values["success_rate"] = values["successes"] / values["episodes"]
        values["override_rate"] = values["override_steps"] / max(values["steps"], 1)
    report = {
        "format_version": "before-we-act.care-robofactory-test-summary/1",
        "completed_at_utc": utc_now(),
        "episodes_per_task": episodes,
        "modes": modes,
        "task_results": task_results,
        "aggregate": totals,
        "offline_report_sha256": sha256_file(args.offline_report),
        "seed_receipt_sha256": sha256_file(args.seed_receipt),
    }
    atomic_json(args.output, report)
    print(json.dumps({"status": "CARE_TESTS_SUMMARIZED", "output": str(args.output)}))


if __name__ == "__main__":
    main()
