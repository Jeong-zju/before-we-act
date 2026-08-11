#!/usr/bin/env python3
"""Apply the pre-registered R11 Discovery continuation conditions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from before_we_act.train_r11_candidate import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", type=Path, nargs="+", required=True)
    parser.add_argument("--causal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    by_update = {}
    for progress_path in args.progress:
        for line in progress_path.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "optimizer_update" and isinstance(row.get("action_loss"), (int, float)):
                by_update[int(row["update"])] = row
    updates = [by_update[index] for index in sorted(by_update)]
    if len(updates) != 1000 or updates[0].get("update") != 1 or updates[-1].get("update") != 1000:
        raise ValueError("Discovery progress does not contain exact updates 1..1000")
    initial = statistics.median(float(row["action_loss"]) for row in updates[:100])
    final = statistics.median(float(row["action_loss"]) for row in updates[-100:])
    causal = json.loads(args.causal.read_text())
    shuffle_macro = sum(
        float(row["action_shuffle_degradation"])
        for row in causal.get("tasks", {}).values()
    ) / 6
    checks = [
        {"id": "action_loss_declines", "passed": final < initial, "initial100_median": initial, "last100_median": final},
        {"id": "prediction_beats_persistence", "passed": causal.get("macro_prediction_gain", 0) > 0, "value": causal.get("macro_prediction_gain")},
        {"id": "shuffled_action_worsens_prediction", "passed": shuffle_macro > 0, "value": shuffle_macro},
    ]
    result = {
        "format_version": "before-we-act.r11.discovery_gate/1",
        "status": "PASSED" if all(row["passed"] for row in checks) else "FAILED",
        "passed": all(row["passed"] for row in checks),
        "checks": checks,
        "completed_at_epoch": time.time(),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 10)


if __name__ == "__main__":
    main()
