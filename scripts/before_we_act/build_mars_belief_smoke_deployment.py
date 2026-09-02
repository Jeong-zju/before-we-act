#!/usr/bin/env python3
"""Export a four-update predictive B-core checkpoint for smoke inference."""
from __future__ import annotations
import argparse
from pathlib import Path
from before_we_act.train_mars_predictive_team_belief import export_deployment

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-checkpoint", type=Path, required=True)
    parser.add_argument("--b0h-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.exists():
        export_deployment(args.training_checkpoint, args.b0h_checkpoint, args.output)

if __name__ == "__main__":
    main()
