#!/usr/bin/env python3
"""Materialize a protected task's exact W12 Gate20 report for R14."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from before_we_act.planner.base import load_r14_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--w12-report", required=True)
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_r14_config(args.config)
    if args.task not in config.deployment["protected_tasks"]:
        raise ValueError("R14 materializer only accepts protected tasks")
    source_path = Path(args.w12_report).resolve(strict=True)
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    seed_path = Path(args.seed_file).resolve(strict=True)
    seed_bytes = seed_path.read_bytes()
    expected = [int(value) for value in json.loads(seed_bytes)["seeds"][:20]]
    rows = source.get("rows", [])
    if (
        source.get("task") != args.task or source.get("episodes") != 20
        or [row.get("seed") for row in rows] != expected
        or not all(isinstance(row.get("success"), bool) for row in rows)
    ):
        raise ValueError("frozen W12 report does not match the R14 Gate20 protocol")
    copied = []
    for row in rows:
        value = dict(row)
        value["route"] = "exact_w12_fallback"
        value["candidate_source"] = "W12 exact protected route"
        value["planner_calls"] = 0
        value["interventions"] = 0
        value["fallbacks"] = 0
        value["planner_exceptions"] = 0
        value["planner_timeouts"] = 0
        copied.append(value)
    result = {
        "schema_version": 1,
        "round": "R14",
        "candidate_id": config.candidate_id,
        "task": args.task,
        "route": "exact_w12_fallback",
        "episodes": 20,
        "successes": sum(row["success"] for row in copied),
        "rows": copied,
        "planner": {"calls": 0, "interventions": 0, "fallbacks": 0, "exceptions": 0, "timeouts": 0},
        "latency_ms": source.get("latency_ms", {}),
        "source_w12_report": str(source_path),
        "source_w12_report_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "seed_protocol": {"source": str(seed_path), "sha256": hashlib.sha256(seed_bytes).hexdigest()},
        "action_hash_equal_to_w12": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"task": args.task, "successes": result["successes"], "route": result["route"]}, sort_keys=True))


if __name__ == "__main__":
    main()
