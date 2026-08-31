from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from .dataset import TASKS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--mode", choices=("first", "open30", "ensemble"), default="ensemble"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pending = list(TASKS)
    active = []
    results = []
    while pending or active:
        while pending and len(active) < args.workers:
            task = pending.pop(0)
            output = args.output / f"{task}.json"
            if output.is_file():
                saved = json.loads(output.read_text())
                if (
                    saved.get("episodes") == args.episodes
                    and saved.get("execution_mode", "ensemble") == args.mode
                ):
                    results.append(saved)
                    continue
            command = [
                "/venv/main/bin/python", "-m", "deployment.duo_act.evaluate",
                "--checkpoint", str(args.checkpoint), "--data", str(args.data),
                "--output", str(output), "--episodes", str(args.episodes), "--task", task,
                "--mode", args.mode,
            ]
            if args.max_steps:
                command += ["--max-steps", str(args.max_steps)]
            log = (args.output / f"{task}.log").open("a")
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(), start_new_session=True)
            active.append((task, output, process, log))
        task, output, process, log = active.pop(0)
        returncode = process.wait()
        log.close()
        if returncode:
            raise RuntimeError(f"{task} evaluator exited {returncode}")
        results.append(json.loads(output.read_text()))
    rows = [row for result in results for row in result["rows"]]
    by_task = {task: [row for row in rows if row["task"] == task] for task in TASKS}
    summary = {
        "status": "complete", "schema": "duobench-act-validation20-v1",
        "episodes_per_task": args.episodes, "total_episodes": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "macro_success_rate": float(np.mean([np.mean([row["success"] for row in by_task[task]]) for task in TASKS])),
        "normalized_final_stage_progress": float(np.mean([row["final_stage_progress"] for row in rows])),
        "execution_mode": args.mode,
        "tasks": {
            task: {
                "episodes": len(by_task[task]), "successes": sum(int(row["success"]) for row in by_task[task]),
                "success_rate": float(np.mean([row["success"] for row in by_task[task]])),
                "mean_final_stage_progress": float(np.mean([row["final_stage_progress"] for row in by_task[task]])),
            }
            for task in TASKS
        },
        "rows": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "total_episodes", "successes", "macro_success_rate")}))


if __name__ == "__main__":
    main()
