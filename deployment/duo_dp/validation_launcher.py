from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .common import POLICY_CONTRACT, TASKS, atomic_json, sha256_file
from .common import EXECUTION_STEPS
from .evaluate import evaluator_revision_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--weights", choices=("ema", "online"), default="ema")
    parser.add_argument("--replan-steps", type=int, default=6)
    parser.add_argument("--revision")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if not 1 <= args.replan_steps <= EXECUTION_STEPS:
        raise ValueError(f"replan-steps must be in [1, {EXECUTION_STEPS}]")
    if args.inference_steps <= 0:
        raise ValueError("inference-steps must be positive")
    evaluator_revision = args.revision or evaluator_revision_for(
        args.replan_steps, args.inference_steps, args.weights
    )
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_sha256 = sha256_file(args.checkpoint)
    pending = list(TASKS)
    active = []
    results = []
    while pending or active:
        while pending and len(active) < args.workers:
            task = pending.pop(0)
            output = args.output / f"{task}.json"
            if output.is_file():
                try:
                    saved = json.loads(output.read_text())
                except json.JSONDecodeError:
                    saved = {}
                if (
                    saved.get("episodes") == args.episodes
                    and saved.get("checkpoint_sha256") == checkpoint_sha256
                    and saved.get("evaluator_revision") == evaluator_revision
                ):
                    results.append(saved)
                    continue
            command = [
                sys.executable,
                "-m",
                "deployment.duo_dp.evaluate",
                "--checkpoint",
                str(args.checkpoint),
                "--data",
                str(args.data),
                "--output",
                str(output),
                "--episodes",
                str(args.episodes),
                "--task",
                task,
                "--inference-steps",
                str(args.inference_steps),
                "--weights",
                args.weights,
                "--replan-steps",
                str(args.replan_steps),
                "--revision",
                evaluator_revision,
            ]
            if args.max_steps is not None:
                command += ["--max-steps", str(args.max_steps)]
            if args.smoke:
                command.append("--smoke")
            log = (args.output / f"{task}.log").open("a", encoding="utf-8")
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                start_new_session=True,
            )
            active.append((task, output, process, log))
        task, output, process, log = active.pop(0)
        returncode = process.wait()
        log.close()
        if returncode:
            raise RuntimeError(f"{task} evaluator exited {returncode}")
        results.append(json.loads(output.read_text()))
    rows = [row for result in results for row in result["rows"]]
    by_task = {task: [row for row in rows if row["task"] == task] for task in TASKS}
    if any(len(by_task[task]) != args.episodes for task in TASKS):
        raise RuntimeError("validation result is missing task episodes")
    summary = {
        "schema": "duobench.dp.validation-summary.v1",
        "status": "complete",
        "episodes_per_task": args.episodes,
        "total_episodes": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "macro_success_rate": float(
            np.mean([np.mean([row["success"] for row in by_task[task]]) for task in TASKS])
        ),
        "normalized_final_stage_progress": float(np.mean([row["final_stage_progress"] for row in rows])),
        "normalized_max_stage_progress": float(np.mean([row["max_stage_progress"] for row in rows])),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "evaluator_revision": evaluator_revision,
        "policy_contract": results[0]["policy_contract"],
        "task_conditioning": bool(results[0].get("task_conditioning", False)),
        "gpu_schedule": f"one RTX 5090, {args.workers} concurrent CPU simulators sharing GPU inference",
        "weights": args.weights,
        "inference_steps": args.inference_steps,
        "replan_steps": args.replan_steps,
        "tasks": {
            task: {
                "episodes": len(by_task[task]),
                "successes": sum(int(row["success"]) for row in by_task[task]),
                "success_rate": float(np.mean([row["success"] for row in by_task[task]])),
                "mean_final_stage_progress": float(
                    np.mean([row["final_stage_progress"] for row in by_task[task]])
                ),
                "mean_max_stage_progress": float(
                    np.mean([row["max_stage_progress"] for row in by_task[task]])
                ),
                "mean_emitted_gripper_transitions": float(
                    np.mean([row.get("emitted_gripper_transitions", 0) for row in by_task[task]])
                ),
                "max_steps": by_task[task][0]["max_steps"],
            }
            for task in TASKS
        },
        "rows": rows,
        "smoke": args.smoke,
    }
    atomic_json(args.output / "summary.json", summary)
    print(
        json.dumps(
            {key: summary[key] for key in ("status", "total_episodes", "successes", "macro_success_rate")}
        )
    )


if __name__ == "__main__":
    main()
