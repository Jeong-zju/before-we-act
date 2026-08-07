#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch

from before_we_act.planner.base import load_r14_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_r14_config(args.config)
    module = importlib.import_module("before_we_act.planner.candidate")
    core = module.build_decision_core(config)
    device = torch.device(args.device)
    base = torch.zeros(4, 100, 8, device=device)

    def score(values):
        target = torch.full_like(values, 0.02)
        return -(values - target).square().mean(dim=(1, 2, 3))

    torch.manual_seed(1400 + int(config.candidate_id[1:]))
    refined, diagnostics = core.refine(base, score, seed=1400, step=0)
    shape = tuple(refined.shape) == tuple(base.shape)
    finite = bool(torch.isfinite(refined).all())
    within = float((refined - base).abs().max()) <= float(config.planner["max_delta"]) + 1e-6
    changed = not torch.equal(refined, base)
    result = {
        "schema_version": 1,
        "round": "R14",
        "candidate_id": config.candidate_id,
        "strict_restore": "N/A (R14 has no trainable planner checkpoint)",
        "shape_valid": shape,
        "finite": finite,
        "within_trust_region": within,
        "synthetic_objective_changes_action": changed,
        "diagnostics": diagnostics,
        "passed": shape and finite and within and changed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
