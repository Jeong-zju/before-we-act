#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for candidate in ("p0", "p1", "p2", "p3"):
        payload = json.loads((args.run_root / "candidates" / candidate / "acceptance.json").read_text())
        gate = payload["gate20"]
        tasks = gate["tasks"]
        rows.append({
            "candidate_id": candidate,
            "status": payload["status"],
            "successes": gate["candidate_total_successes"],
            "paired_wins": gate["paired_wins"],
            "camera_plus_stack": tasks["camera_alignment"]["candidate"] + tasks["three_robots_stack_cube"]["candidate"],
            "worst_task": min(value["candidate"] for value in tasks.values()),
            "p95_latency_ms": gate["p95_latency_ms"] if gate["p95_latency_ms"] is not None else float("inf"),
        })
    eligible = [row for row in rows if row["status"] == "PASSED"]
    eligible.sort(key=lambda row: (-row["successes"], -row["paired_wins"], -row["camera_plus_stack"], -row["worst_task"], row["p95_latency_ms"], row["candidate_id"]))
    winner = eligible[0]["candidate_id"] if eligible else None
    result = {
        "schema_version": 1,
        "round": "R14",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "selection_rule_frozen_before_gate20": True,
        "rows": rows,
        "winner": winner,
        "decision": "winner_selected" if winner else "no_winner_no_merge",
        "merge_performed": False,
        "next_stage_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"winner": winner, "decision": result["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
