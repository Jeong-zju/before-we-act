#!/usr/bin/env python3
"""Materialize a protected-task report from immutable paired W10 results.

R12-E1 calls the unchanged W10 implementation on protected tasks.  Reusing the
already executed frozen report avoids rerunning identical closed-loop episodes;
the final selected hybrid still receives a direct fallback-route canary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from before_we_act.action_generator.evolution import load_r12_evolution_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_r12_evolution_config(args.config)
    if args.task not in config.deployment["protected_tasks"]:
        raise ValueError("only protected R12-E1 tasks may reuse exact W10 results")
    seed_path = Path(args.seed_file).resolve(strict=True)
    seed_bytes = seed_path.read_bytes()
    seeds = [int(value) for value in json.loads(seed_bytes)["seeds"][:20]]
    if len(seeds) != 20:
        raise ValueError("R12-E1 fallback materialization requires Gate20 seeds")
    baseline_path = Path(args.baseline).resolve(strict=True)
    baseline = json.loads(baseline_path.read_text())
    by_seed = {int(row["seed"]): row for row in baseline["rows"]}
    if set(seeds) - set(by_seed):
        raise ValueError("frozen W10 baseline lacks a protected Gate20 seed")
    rows = []
    for seed in seeds:
        source = by_seed[seed]
        rows.append(
            {
                "task": args.task,
                "seed": seed,
                "success": bool(source["success"]),
                "steps": int(source["steps"]),
                "safety_projections": 0,
                "terminal_info": source.get("terminal_info", {}),
                "route": "exact_w10_fallback",
            }
        )
    result = {
        "schema_version": 1,
        "round": "R12-E1",
        "candidate_id": config.candidate_id,
        "task": args.task,
        "route": "exact_w10_fallback",
        "episodes": 20,
        "successes": sum(row["success"] for row in rows),
        "rows": rows,
        "latency_ms": {"samples": 0, "p50": None, "p95": None},
        "seed_protocol": {
            "source": str(seed_path),
            "sha256": hashlib.sha256(seed_bytes).hexdigest(),
        },
        "policy_inputs": "exact unchanged W10 native 480x640 RGB fallback",
        "privileged_inputs": False,
        "reused_exact_w10_baseline": True,
        "baseline_source": str(baseline_path),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result | {"rows": "saved"}, sort_keys=True))


if __name__ == "__main__":
    main()
