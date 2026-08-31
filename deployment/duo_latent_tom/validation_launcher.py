from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import POLICY_CONTRACT, TASKS, atomic_json, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--workers", type=int, default=3); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    protocol = load_config()["validation20"]
    episodes = 1 if args.smoke else args.episodes
    pending = list(TASKS); active = []; results = []
    while pending or active:
        while pending and len(active) < args.workers:
            task = pending.pop(0); result_path = args.output / f"{task}.json"
            if result_path.is_file():
                try:
                    saved = json.loads(result_path.read_text())
                    if saved.get("status") == "complete" and saved.get("episodes") == episodes:
                        results.append(saved); continue
                except (OSError, json.JSONDecodeError): pass
            command = ["/venv/main/bin/python", "-u", "-m", "deployment.duo_latent_tom.evaluate", "--checkpoint", str(args.checkpoint), "--data", str(args.data), "--output", str(args.output), "--task", task, "--episodes", str(args.episodes), "--diffusion-steps", str(protocol["diffusion_steps"]), "--replan-interval", str(protocol["replan_interval"])]
            if args.smoke: command.append("--smoke")
            log = (args.output / f"{task}.log").open("a")
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(), start_new_session=True)
            active.append((task, result_path, process, log))
        task, result_path, process, log = active.pop(0)
        returncode = process.wait(); log.close()
        if returncode: raise RuntimeError(f"{task} evaluator exited {returncode}")
        results.append(json.loads(result_path.read_text()))
    by_task = {row["task"]: row for row in results}
    summary = {"schema": "duobench.latent-tom.validation20.v2", "status": "complete", "episodes_per_task": episodes, "total_episodes": sum(row["episodes"] for row in results), "successes": sum(row["successes"] for row in results), "macro_success_rate": float(np.mean([row["success_rate"] for row in results])), "tasks": {task: by_task[task] for task in TASKS}, "policy_contract": POLICY_CONTRACT, "seed_base": 20260820, "diffusion_steps": protocol["diffusion_steps"], "replan_interval": protocol["replan_interval"], "sim_backend": "cpu", "completed_at": datetime.now(timezone.utc).isoformat()}
    expected = len(TASKS) if args.smoke else len(TASKS) * 20
    if summary["total_episodes"] != expected: raise RuntimeError("validation episode count drift")
    atomic_json(args.output / "summary.json", summary); print(json.dumps({key: summary[key] for key in ("status", "total_episodes", "successes", "macro_success_rate")}))


if __name__ == "__main__": main()
