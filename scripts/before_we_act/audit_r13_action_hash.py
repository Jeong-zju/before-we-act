#!/usr/bin/env python3
"""Prove R13 consumes but cannot mutate or select the frozen W12 proposal."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from before_we_act.data.world_windows import CachedWorldWindows, legal_model_inputs
from before_we_act.world_model.base import CandidateConditionedWorldModel, load_r13_config


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--action-checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_r13_config(args.config)
    action_path = Path(args.action_checkpoint).resolve(strict=True)
    before_checkpoint_sha = file_sha(action_path)
    if before_checkpoint_sha != config.raw["action_checkpoint_sha256"]:
        raise ValueError("frozen W12 checkpoint digest differs before R13 audit")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    model = CandidateConditionedWorldModel(config).to(device).eval()
    model.load_state_dict(saved["model"], strict=True)
    dataset = CachedWorldWindows(args.cache, "validation")
    batch = {key: value[:64].to(device) for key, value in dataset.data.items()}
    actions = batch["candidate_actions"]
    baseline = actions.detach().clone()
    before = tensor_sha(actions)
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        prediction = model(**legal_model_inputs(batch))
    after = tensor_sha(actions)
    after_checkpoint_sha = file_sha(action_path)
    checks = {
        "action_tensor_bit_exact_after_world_forward": (
            before == after and torch.equal(actions, baseline)
        ),
        "w12_checkpoint_bit_exact": before_checkpoint_sha == after_checkpoint_sha,
        "w12_checkpoint_matches_config": before_checkpoint_sha == config.raw["action_checkpoint_sha256"],
        "world_output_has_no_action_field": not hasattr(prediction, "actions"),
        "planner_disabled": prediction.diagnostics.get("planner_enabled") is False,
        "rerank_disabled": prediction.diagnostics.get("rerank_enabled") is False,
    }
    result = {
        "schema_version": 1,
        "round": "R13",
        "candidate_id": config.candidate_id,
        "action_hash_before": before,
        "action_hash_after": after,
        "w12_checkpoint_sha256_before": before_checkpoint_sha,
        "w12_checkpoint_sha256_after": after_checkpoint_sha,
        "action_hash_equal": checks["action_tensor_bit_exact_after_world_forward"],
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
