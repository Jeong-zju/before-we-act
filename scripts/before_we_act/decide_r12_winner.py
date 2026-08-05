#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


TASKS = ("lift_barrier", "camera_alignment", "three_robots_stack_cube", "long_pipeline_delivery", "take_photo")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceptance", action="append", default=[])
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--baseline-checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.acceptance) != 4:
        raise ValueError("R12 decision requires four acceptance reports")
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.acceptance]
    if {row["candidate_id"] for row in reports} != {"p0", "p1", "p2", "p3"}:
        raise ValueError("R12 decision candidate set differs")
    qualified = [row for row in reports if row.get("qualified")]

    def rank(row):
        gate = row["gate20"]
        tasks = gate["tasks"]
        paired_wins = sum(max(0, task["delta"]) for task in tasks.values())
        camera_stack = tasks["camera_alignment"]["candidate"] + tasks["three_robots_stack_cube"]["candidate"]
        worst = min(task["candidate"] for task in tasks.values())
        latency = row.get("latency_p95_ms_max_task")
        latency_score = -float(latency) if latency is not None else float("-inf")
        return (
            gate["candidate_total_successes"],
            paired_wins,
            camera_stack,
            worst,
            latency_score,
            -int(row["candidate_id"][1:]),
        )

    winner = max(qualified, key=rank) if qualified else None
    result = {
        "schema_version": 1,
        "round": "R12",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "baseline_merge_commit": args.baseline_commit,
        "baseline_checkpoint_sha256": args.baseline_checkpoint_sha256,
        "qualified_set": [row["candidate_id"] for row in sorted(qualified, key=rank, reverse=True)],
        "unique_winner": winner["candidate_id"] if winner else None,
        "winner_source_commit": winner["commit"] if winner else None,
        "winner_checkpoint": winner["checkpoint"] if winner else None,
        "merge_performed": False,
        "baseline_after": args.baseline_commit,
        "decision": "winner_identified_no_merge_without_separate_authorization" if winner else "no_winner_no_merge",
        "candidate_results": {
            row["candidate_id"]: {
                "valid_component": row["valid_component"],
                "qualified": row["qualified"],
                "gate20_total": row["gate20"]["candidate_total_successes"],
                "rejection_reasons": [item["id"] for item in row["acceptance"] if not item["passed"]],
            }
            for row in reports
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
