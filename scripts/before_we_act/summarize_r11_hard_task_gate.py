#!/usr/bin/env python3
"""Prediction-to-action closed-loop alternative on frozen hard-task Validation5."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from before_we_act.train_r11_candidate import atomic_json


TASKS = ("camera_alignment", "place_food")
MODES = ("prediction_off", "prediction_shuffled")


def load_task(root: Path, task: str, mode: str, checkpoint: str) -> dict:
    payload = json.loads((root / f"{task}.json").read_text())
    if (
        payload.get("status") != "PASSED"
        or payload.get("task") != task
        or payload.get("mode") != mode
        or payload.get("checkpoint_sha256") != checkpoint
        or payload.get("invalid_actions") != 0
        or payload.get("fallback_calls") != 0
    ):
        raise ValueError(f"invalid hard-task artifact {root / f'{task}.json'}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal-root", type=Path, required=True)
    parser.add_argument("--prediction-off-root", type=Path, required=True)
    parser.add_argument("--prediction-shuffled-root", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    normal = {
        task: load_task(args.normal_root, task, "normal", args.checkpoint_sha256)
        for task in TASKS
    }
    interventions = {
        "prediction_off": {
            task: load_task(
                args.prediction_off_root, task, "prediction_off", args.checkpoint_sha256
            )
            for task in TASKS
        },
        "prediction_shuffled": {
            task: load_task(
                args.prediction_shuffled_root,
                task,
                "prediction_shuffled",
                args.checkpoint_sha256,
            )
            for task in TASKS
        },
    }
    normal_seeds = {}
    normal_successes = 0
    for task, payload in normal.items():
        first = payload["rows"][:5]
        if len(first) != 5:
            raise ValueError("formal normal validation has fewer than five frozen episodes")
        normal_seeds[task] = [row["seed"] for row in first]
        normal_successes += sum(row["success"] for row in first)
    rows = {}
    for mode, by_task in interventions.items():
        successes = 0
        task_rows = {}
        for task, payload in by_task.items():
            if payload.get("episodes") != 5:
                raise ValueError("hard-task intervention must contain exactly five episodes")
            seeds = [row["seed"] for row in payload["rows"]]
            if seeds != normal_seeds[task]:
                raise ValueError("hard-task intervention seed/order differs from normal")
            count = sum(row["success"] for row in payload["rows"])
            successes += count
            task_rows[task] = count
        rows[mode] = {
            "successes": successes,
            "loss_vs_normal": normal_successes - successes,
            "tasks": task_rows,
        }
    passed_mode = max(rows, key=lambda mode: rows[mode]["loss_vs_normal"])
    result = {
        "format_version": "before-we-act.r11.hard_task_causal/1",
        "status": "PASSED" if rows[passed_mode]["loss_vs_normal"] >= 1 else "FAILED",
        "passed": rows[passed_mode]["loss_vs_normal"] >= 1,
        "checkpoint_sha256": args.checkpoint_sha256,
        "tasks": list(TASKS),
        "episodes_per_task": 5,
        "normal_successes": normal_successes,
        "interventions": rows,
        "best_intervention": passed_mode,
        "required_success_loss": 1,
        "task_text_held_fixed": True,
        "completed_at_epoch": time.time(),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
