#!/usr/bin/env python3
"""Record a prerequisite/runtime failure without inventing unrun R11 results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from before_we_act.train_r11_candidate import atomic_json


CHECK_IDS = (
    "complete_stable_validation20",
    "all_six_at_least_80_of_120",
    "protected_four_at_least_72_each_at_least_16",
    "camera_and_food_floor",
    "future_vs_persistence",
    "action_shuffle_to_future",
    "prediction_to_action",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--failed-stage", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        # Never replace the formal decision for this exact code identity with a
        # later wrapper failure.  Prerequisite failures are retry receipts, and
        # an older commit's receipt must not remain authoritative after a
        # fast-forward deployment.
        current = json.loads(args.output.read_text())
        if (
            current.get("complete")
            and current.get("commit") == args.commit
            and "failed_stage" not in current
        ):
            return
    payload = {
        "format_version": "before-we-act.r11.acceptance/1",
        "complete": True,
        "status": "FAILED",
        "passed": False,
        "candidate": args.candidate,
        "branch": args.branch,
        "commit": args.commit,
        "failed_stage": args.failed_stage,
        "failure_reason": args.reason,
        "exit_code": args.exit_code,
        "checks": [
            {
                "id": check_id,
                "passed": False,
                "not_evaluated": check_id != "complete_stable_validation20",
                "reason": args.reason,
            }
            for check_id in CHECK_IDS
        ],
        "score": None,
        "completed_at_epoch": time.time(),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
