#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from before_we_act.action_generator.base import JointActionGenerator, load_r12_config
from before_we_act.contracts import TeamBeliefState


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_r12_config(args.config)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    left = JointActionGenerator(config).to(device).eval()
    right = JointActionGenerator(config).to(device).eval()
    left.load_state_dict(saved["model"], strict=True)
    right.load_state_dict(saved["model"], strict=True)
    generator = torch.Generator(device=device).manual_seed(1200)
    agent_mask = torch.tensor([[True, True, False, False]], device=device)
    belief = TeamBeliefState(
        tokens=torch.randn((1, 16, 96), generator=generator, device=device),
        agent_tokens=torch.randn((1, 4, 96), generator=generator, device=device),
        consensus_token=torch.randn((1, 96), generator=generator, device=device),
        uncertainty=torch.zeros((1, 1), device=device),
        agent_mask=agent_mask,
    )
    noise = torch.randn((1, 100, 32), generator=generator, device=device)
    with torch.no_grad():
        first = left.sample(belief, noise=noise).actions
        second = right.sample(belief, noise=noise).actions
    stats = {key: torch.as_tensor(value, device=device) for key, value in saved["stats"].items()}
    normalized = torch.linspace(-2, 2, 800, device=device).reshape(1, 100, 8)
    raw = normalized * stats["a_std"] + stats["a_mean"]
    restored = (raw - stats["a_mean"]) / stats["a_std"]
    checks = {
        "checkpoint_update_two": int(saved["update"]) == 2,
        "strict_restore_exact": bool(torch.equal(first, second)),
        "finite": bool(torch.isfinite(first).all()),
        "normalized_range": float(first.abs().max()) <= 5.0,
        "absent_agent_mask": bool(first[:, :, 2:].eq(0).all()),
        "normalization_roundtrip": bool(torch.allclose(normalized, restored, atol=1e-6, rtol=1e-6)),
        "core_free_checkpoint": bool(saved.get("core_free_runtime")) and not any(
            any(forbidden in key.lower() for forbidden in (
                "arca", "pair_route", "forced_role", "role_prototype",
                "perception_extension", "rgbd_patch_fusion",
            ))
            for key in saved["model"]
        ),
    }
    result = {
        "schema_version": 1,
        "round": "R12",
        "candidate_id": config.candidate_id,
        "checks": checks,
        "normalized_abs_max": float(first.abs().max()),
        "restore_max_abs": float((first - second).abs().max()),
        "passed": all(checks.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
