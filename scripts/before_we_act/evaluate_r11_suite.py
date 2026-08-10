#!/usr/bin/env python3
"""Load one R11 checkpoint once, then evaluate a fixed task suite sequentially."""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from before_we_act.evaluate_r11_candidate import evaluate, load_candidate
from before_we_act.r11_data import SIX_TASKS


MAX_STEPS = {
    "lift_barrier": 500,
    "camera_alignment": 1500,
    "long_pipeline_delivery": 1500,
    "take_photo": 1500,
    "pass_shoe": 500,
    "place_food": 500,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=SIX_TASKS, default=list(SIX_TASKS))
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--mode", choices=("normal", "prediction_off", "prediction_shuffled"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--robofactory-root", default="/workspace/RoboFactory")
    parser.add_argument("--resume-log", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    device = torch.device(args.device)
    loaded = load_candidate(
        checkpoint, expected_sha256=args.checkpoint_sha256, device=device
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        evaluate(
            SimpleNamespace(
                checkpoint=str(checkpoint),
                checkpoint_sha256=args.checkpoint_sha256,
                task=task,
                seed_file=str(args.seed_root / f"{task}.json"),
                episodes=args.episodes,
                max_steps=MAX_STEPS[task],
                mode=args.mode,
                device=args.device,
                robofactory_root=args.robofactory_root,
                resume_log=args.resume_log,
                output=str(args.output_root / f"{task}.json"),
            ),
            loaded=loaded,
        )


if __name__ == "__main__":
    main()
