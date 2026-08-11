#!/usr/bin/env python3
"""Selection continuation: both dynamics gates plus offline or hard-task action gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from before_we_act.train_r11_candidate import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal", type=Path, required=True)
    parser.add_argument("--hard-task-gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    causal = json.loads(args.causal.read_text())
    by_id = {row["id"]: row for row in causal.get("checks", [])}
    hard = json.loads(args.hard_task_gate.read_text()) if args.hard_task_gate else {}
    checks = [
        by_id.get("future_vs_persistence", {"id": "future_vs_persistence", "passed": False}),
        by_id.get("action_shuffle_to_future", {"id": "action_shuffle_to_future", "passed": False}),
        {
            "id": "prediction_to_action",
            "passed": bool(by_id.get("prediction_to_action_offline", {}).get("passed")) or bool(hard.get("passed")),
            "offline": by_id.get("prediction_to_action_offline", {}),
            "hard_task": hard or {"status": "not_run"},
        },
    ]
    result = {
        "format_version": "before-we-act.r11.selection_gate/1",
        "status": "PASSED" if all(row.get("passed") for row in checks) else "FAILED",
        "passed": all(row.get("passed") for row in checks),
        "checks": checks,
        "completed_at_epoch": time.time(),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 10)


if __name__ == "__main__":
    main()
