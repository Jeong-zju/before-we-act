from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .common import ENVS, TASKS, atomic_json
from .evaluate import REVISION, episode
from .model import load_model
from .precompute import load_encoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--noposplat-weight", required=True)
    parser.add_argument("--robofactory-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ensemble-decay", type=float, default=0.01)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    gaussian = load_encoder(Path(args.noposplat_weight), device)
    results = {}
    for inference_steps in (20, 100):
        policy, _ = load_model(args.checkpoint, device, inference_steps)
        task_rows = {}
        for task in TASKS:
            _, _, max_steps, seed = ENVS[task]
            row = episode(policy, gaussian, task, args.robofactory_root, seed, device, max_steps, args.ensemble_decay)
            if not np.isfinite(row["mean_inference_seconds"]):
                raise RuntimeError(f"non-finite inference timing for {task}/{inference_steps}")
            if not 1 <= int(row["steps"]) <= max_steps:
                raise RuntimeError(f"episode step contract failed for {task}/{inference_steps}")
            task_rows[task] = row
            print(json.dumps({"inference_steps": inference_steps, "task": task, **row}), flush=True)
        results[str(inference_steps)] = {
            "successes": sum(int(row["success"]) for row in task_rows.values()),
            "tasks": task_rows,
        }
        del policy
        torch.cuda.empty_cache()
    success20, success100 = results["20"]["successes"], results["100"]["successes"]
    selected = 20 if success20 > success100 else 100
    output = {
        "schema": "mars-control.gaudp.inference-comparison.v1",
        "status": "complete",
        "gate_passed": True,
        "episodes": 8,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "evaluator_revision": REVISION,
        "temporal_ensemble_decay": args.ensemble_decay,
        "selection_rule": "higher_successes_then_official_100_step_tiebreak",
        "selected_inference_steps": selected,
        "results": results,
    }
    atomic_json(args.output, output)
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
