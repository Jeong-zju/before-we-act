#!/usr/bin/env python3
"""Apply the immutable R11 section-11 qualification gates and score formula."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from before_we_act.train_r11_candidate import atomic_json


PROTECTED = (
    "lift_barrier",
    "long_pipeline_delivery",
    "take_photo",
    "pass_shoe",
)


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--train-status", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--causal", type=Path, required=True)
    parser.add_argument("--hard-task-gate", type=Path)
    parser.add_argument("--w10-latency", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = load(args.train_status)
    validation = load(args.validation)
    causal = load(args.causal)
    baseline_latency = load(args.w10_latency)
    hard = load(args.hard_task_gate) if args.hard_task_gate else {}
    if validation.get("checkpoint_sha256") != args.checkpoint_sha256:
        raise ValueError("validation checkpoint differs from acceptance target")
    if causal.get("checkpoint_sha256") != args.checkpoint_sha256:
        raise ValueError("causal checkpoint differs from acceptance target")
    if train.get("checkpoint_sha256") != args.checkpoint_sha256:
        raise ValueError("formal training status checkpoint differs")
    tasks = validation.get("tasks", {})
    protected_values = [int(tasks.get(task, {}).get("successes", -1)) for task in PROTECTED]
    protected_total = sum(protected_values)
    camera = int(tasks.get("camera_alignment", {}).get("successes", -1))
    food = int(tasks.get("place_food", {}).get("successes", -1))
    total = int(validation.get("successes", -1))
    causal_by_id = {row["id"]: row for row in causal.get("checks", [])}
    future_gate = causal_by_id.get("future_vs_persistence", {})
    action_future_gate = causal_by_id.get("action_shuffle_to_future", {})
    prediction_action_offline = causal_by_id.get("prediction_to_action_offline", {})
    hard_pass = bool(hard.get("passed"))
    checks = [
        {
            "id": "complete_stable_validation20",
            "passed": (
                train.get("status") == "PASSED"
                and train.get("update") == 120000
                and validation.get("status") == "PASSED"
                and validation.get("episodes") == 120
                and set(tasks) == {
                    "lift_barrier", "camera_alignment", "long_pipeline_delivery",
                    "take_photo", "pass_shoe", "place_food"
                }
                and validation.get("invalid_actions") == 0
                and validation.get("fallback_calls") == 0
            ),
            "evidence": {
                "train_status": train.get("status"),
                "update": train.get("update"),
                "episodes": validation.get("episodes"),
                "invalid_actions": validation.get("invalid_actions"),
                "fallback_calls": validation.get("fallback_calls"),
            },
        },
        {
            "id": "all_six_at_least_80_of_120",
            "passed": total >= 80,
            "value": total,
            "threshold": 80,
        },
        {
            "id": "protected_four_at_least_72_each_at_least_16",
            "passed": protected_total >= 72 and min(protected_values) >= 16,
            "total": protected_total,
            "per_task": dict(zip(PROTECTED, protected_values)),
            "threshold": {"total": 72, "each": 16},
        },
        {
            "id": "camera_and_food_floor",
            "passed": camera >= 6 and camera + food >= 8,
            "camera": camera,
            "food": food,
            "combined": camera + food,
            "threshold": {"camera": 6, "combined": 8},
        },
        {
            "id": "future_vs_persistence",
            "passed": bool(future_gate.get("passed")),
            "evidence": future_gate,
        },
        {
            "id": "action_shuffle_to_future",
            "passed": bool(action_future_gate.get("passed")),
            "evidence": action_future_gate,
        },
        {
            "id": "prediction_to_action",
            "passed": bool(prediction_action_offline.get("passed")) or hard_pass,
            "offline": prediction_action_offline,
            "hard_task_validation5": hard or {"status": "not_run"},
        },
    ]
    complete = len(checks) == 7
    passed = complete and all(row["passed"] for row in checks)
    macro_gain = float(causal.get("macro_prediction_gain", 0.0))
    causal_fraction = sum(row["passed"] for row in checks[4:7]) / 3
    candidate_latency = float(validation["latency_ms_p95"])
    w10_latency = float(baseline_latency["latency_ms_p95"])
    if candidate_latency <= 0 or w10_latency <= 0:
        raise ValueError("latency receipts must be positive")
    score_components = {
        "closed_loop": 60 * total / 120,
        "protected_four": 10 * protected_total / 80,
        "camera_plus_food": 10 * (camera + food) / 40,
        "prediction_gain": 8 * min(max(macro_gain / 0.20, 0.0), 1.0),
        "causal": 7 * causal_fraction,
        "latency": 5 * min(w10_latency / candidate_latency, 1.0),
    }
    result = {
        "format_version": "before-we-act.r11.acceptance/1",
        "complete": complete,
        "status": "PASSED" if passed else "FAILED",
        "passed": passed,
        "candidate": args.candidate,
        "branch": args.branch,
        "commit": args.commit,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": args.checkpoint_sha256,
        "checks": checks,
        "raw": {
            "all_six_successes": total,
            "protected_four_successes": protected_total,
            "camera_plus_food_successes": camera + food,
            "macro_prediction_gain": macro_gain,
            "causal_gate_pass_fraction": causal_fraction,
            "w10_p95_action_latency_ms": w10_latency,
            "candidate_p95_action_latency_ms": candidate_latency,
        },
        "score_components": score_components,
        "score": sum(score_components.values()) if passed else None,
        "ineligible_score_preview": sum(score_components.values()),
        "completed_at_epoch": time.time(),
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if passed else 10)


if __name__ == "__main__":
    main()
