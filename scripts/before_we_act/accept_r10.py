#!/usr/bin/env python3
"""Fail-closed R10 Gate20, causal, gate-zero and latency acceptance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


TASKS = (
    "lift_barrier",
    "camera_alignment",
    "three_robots_stack_cube",
    "long_pipeline_delivery",
    "take_photo",
)


def parse_mapping(values: list[str], label: str):
    result = {}
    for value in values:
        task, separator, path = value.partition("=")
        if separator != "=" or task not in TASKS or task in result:
            raise ValueError(f"invalid {label} mapping: {value}")
        result[task] = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(result) != set(TASKS):
        raise ValueError(f"{label} requires all five tasks")
    return result


def rows_by_seed(payload):
    return {int(row["seed"]): bool(row["success"]) for row in payload["rows"]}


def bootstrap_lower(differences: np.ndarray, draws: int = 20_000) -> float:
    generator = np.random.default_rng(20260804)
    indices = generator.integers(0, len(differences), size=(draws, len(differences)))
    means = differences[indices].mean(axis=1)
    return float(np.percentile(means, 2.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", choices=("p0", "p1", "p2", "p3"), required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gate-audit", required=True)
    parser.add_argument("--baseline", action="append", default=[])
    parser.add_argument("--normal", action="append", default=[])
    parser.add_argument("--intervention", action="append", default=[])
    parser.add_argument("--mode", choices=("screen", "formal"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = parse_mapping(args.baseline, "baseline")
    normal = parse_mapping(args.normal, "normal")
    intervention = parse_mapping(args.intervention, "intervention")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    gate_audit = json.loads(Path(args.gate_audit).read_text(encoding="utf-8"))
    task_results, causal_values = {}, []
    for task in TASKS:
        normal_rows = rows_by_seed(normal[task])
        intervention_rows = rows_by_seed(intervention[task])
        if len(normal_rows) != 20 or set(intervention_rows) != set(normal_rows):
            raise ValueError(f"{task} normal/intervention must be the same 20 seeds")
        base_all = rows_by_seed(baseline[task])
        if not set(normal_rows).issubset(base_all):
            raise ValueError(f"{task} baseline is missing paired Gate20 seeds")
        base_count = sum(base_all[seed] for seed in normal_rows)
        normal_count = sum(normal_rows.values())
        intervention_count = sum(intervention_rows.values())
        task_results[task] = {
            "baseline": base_count,
            "candidate": normal_count,
            "intervention": intervention_count,
            "delta": normal_count - base_count,
            "episodes": 20,
        }
        causal_values.extend(
            float(normal_rows[seed]) - float(intervention_rows[seed])
            for seed in sorted(normal_rows)
        )
    base_total = sum(row["baseline"] for row in task_results.values())
    candidate_total = sum(row["candidate"] for row in task_results.values())
    macro_base = base_total / 100.0
    macro_candidate = candidate_total / 100.0
    camera_stack_gain = (
        task_results["camera_alignment"]["delta"]
        + task_results["three_robots_stack_cube"]["delta"]
    )
    other_gain = sum(
        task_results[task]["delta"]
        for task in ("lift_barrier", "long_pipeline_delivery", "take_photo")
    )
    causal_array = np.asarray(causal_values, dtype=np.float64)
    causal_mean = float(causal_array.mean())
    causal_lower = bootstrap_lower(causal_array)
    acceptance = [
        {
            "id": "gate_zero_exact",
            "rule": "gate=0 base/forced-role chunks, routes and temporal output elementwise exact",
            "passed": bool(gate_audit.get("gate_zero_passed")),
        },
        {
            "id": "paired_gate20",
            "rule": "macro strictly above B9 and no task declines by more than 1/20",
            "passed": macro_candidate > macro_base
            and all(row["delta"] >= -1 for row in task_results.values()),
        },
        {
            "id": "camera_stack_and_other_tasks",
            "rule": "Camera+Stack gain >=4/40 and Lift+LPD+Photo total does not decline",
            "passed": camera_stack_gain >= 4 and other_gain >= 0,
        },
        {
            "id": "causal_intervention",
            "rule": "preregistered intervention has correct direction and episode-bootstrap 95% lower bound >0",
            "passed": causal_mean > 0 and causal_lower > 0,
        },
        {
            "id": "latency_and_inputs",
            "rule": "P95 control latency <=1.15x B9 and no privileged input",
            "passed": bool(gate_audit.get("latency_passed"))
            and gate_audit.get("privileged_inputs") is False,
        },
    ]
    passed = all(item["passed"] for item in acceptance)
    screen_continue = passed or (
        (candidate_total > base_total or camera_stack_gain > 0)
        and np.isfinite(float(gate_audit.get("trained_gate", float("nan"))))
        and abs(float(gate_audit.get("trained_gate", 0.0))) > 0
    )
    result = {
        "schema_version": 1,
        "round": "R10",
        "candidate_id": args.candidate_id,
        "branch": args.branch,
        "commit": args.commit,
        "status": "PASSED" if passed else "FAILED",
        "mode": args.mode,
        "training": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "update": int(checkpoint["update"]),
            "metrics": checkpoint.get("last_metrics", {}),
        },
        "gate_zero": gate_audit.get("checks", {}),
        "gate20": {
            "baseline_macro": macro_base,
            "candidate_macro": macro_candidate,
            "macro_delta": macro_candidate - macro_base,
            "tasks": task_results,
            "camera_stack_gain": camera_stack_gain,
            "other_three_gain": other_gain,
        },
        "causal_intervention": {
            "paired_episodes": len(causal_array),
            "mean_delta": causal_mean,
            "bootstrap_95_lower": causal_lower,
        },
        "latency": gate_audit.get("latency", {}),
        "privileged_input_audit": {"passed": gate_audit.get("privileged_inputs") is False},
        "acceptance": acceptance,
        "screen_continue": screen_continue,
        "passed": passed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    if args.mode == "formal":
        raise SystemExit(0 if passed else 1)
    raise SystemExit(0 if screen_continue else 10)


if __name__ == "__main__":
    main()
