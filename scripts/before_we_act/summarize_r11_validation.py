#!/usr/bin/env python3
"""Fail-closed aggregation of fixed-seed R11 closed-loop task artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from before_we_act.r11_data import SIX_TASKS
from before_we_act.train_r11_candidate import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=SIX_TASKS, default=list(SIX_TASKS))
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--mode", choices=("normal", "prediction_off", "prediction_shuffled"), required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = {}
    for task in args.tasks:
        path = args.input_root / f"{task}.json"
        payload = json.loads(path.read_text())
        if (
            payload.get("status") != "PASSED"
            or payload.get("task") != task
            or payload.get("mode") != args.mode
            or payload.get("episodes") != args.episodes
            or len(payload.get("rows", [])) != args.episodes
            or payload.get("checkpoint_sha256") != args.checkpoint_sha256
            or payload.get("invalid_actions") != 0
            or payload.get("fallback_calls") != 0
        ):
            raise ValueError(f"incomplete or incompatible validation artifact: {path}")
        seeds = [item["seed"] for item in payload["rows"]]
        if len(set(seeds)) != args.episodes:
            raise ValueError(f"duplicate/missing validation seed in {path}")
        rows[task] = payload
    latencies = [
        latency
        for payload in rows.values()
        for row in payload["rows"]
        for latency in row.get("latencies_ms", [])
    ]
    if not latencies:
        raise ValueError("validation artifacts have no model-call latency measurements")
    import numpy as np

    result = {
        "format_version": "before-we-act.r11.validation_summary/1",
        "status": "PASSED",
        "mode": args.mode,
        "checkpoint_sha256": args.checkpoint_sha256,
        "episodes_per_task": args.episodes,
        "episodes": sum(payload["episodes"] for payload in rows.values()),
        "successes": sum(payload["successes"] for payload in rows.values()),
        "invalid_actions": 0,
        "fallback_calls": 0,
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "tasks": {
            task: {
                "episodes": payload["episodes"],
                "successes": payload["successes"],
                "success_rate": payload["success_rate"],
                "max_steps": payload["max_steps"],
                "execution_cadence": payload["execution_cadence"],
                "seed_file_sha256": payload["seed_file_sha256"],
            }
            for task, payload in rows.items()
        },
        "completed_at_epoch": time.time(),
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
