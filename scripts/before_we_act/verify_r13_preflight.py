#!/usr/bin/env python3
"""Strict R13 two-update restore and legal-input/action-effect preflight."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from before_we_act.data.world_windows import CachedWorldWindows, legal_model_inputs
from before_we_act.world_model.base import CandidateConditionedWorldModel, load_r13_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_r13_config(args.config)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    model = CandidateConditionedWorldModel(config).to(device).eval()
    incompatible = model.load_state_dict(saved["model"], strict=True)
    dataset = CachedWorldWindows(args.cache, "validation")
    batch = {
        key: value[:2].to(device) for key, value in dataset.data.items()
    }
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        first = model(**legal_model_inputs(batch))
        changed = dict(legal_model_inputs(batch))
        changed["candidate_actions"] = changed["candidate_actions"].clone()
        changed["candidate_actions"][:, :, :, :16] += 0.25
        second = model(**changed)
    future_rejected = False
    try:
        model(**legal_model_inputs(batch), future_latent=batch["future_latent"])
    except TypeError:
        future_rejected = True
    delta = float(
        (first.latent_by_horizon.float() - second.latent_by_horizon.float())
        .abs()
        .mean()
    )
    checks = {
        "checkpoint_identity": (
            saved.get("round") == "R13"
            and saved.get("candidate_id") == config.candidate_id
            and int(saved.get("update", -1)) == 2
        ),
        "strict_restore": not incompatible.missing_keys and not incompatible.unexpected_keys,
        "prediction_finite": bool(torch.isfinite(first.latent_by_horizon).all()),
        "future_target_argument_rejected": future_rejected,
        "candidate_action_changes_prediction": delta > 0,
        "planner_and_rerank_disabled": (
            first.diagnostics.get("planner_enabled") is False
            and first.diagnostics.get("rerank_enabled") is False
        ),
    }
    result = {
        "schema_version": 1,
        "round": "R13",
        "candidate_id": config.candidate_id,
        "action_condition_delta": delta,
        "checks": checks,
        "passed": all(checks.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
