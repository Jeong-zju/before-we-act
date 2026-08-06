#!/usr/bin/env python3
"""Strict R12-R4 restore, bridge-gradient and action-effect preflight."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from before_we_act.action_generator.r4_base import (
    R4JointActionGenerator,
    load_r12_r4_config,
)
from before_we_act.contracts import TeamBeliefState


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    config = load_r12_r4_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("round") != "R12-R4"
        or checkpoint.get("candidate_id") != config.candidate_id
        or int(checkpoint.get("update", -1)) != 2
    ):
        raise ValueError("R12-R4 preflight checkpoint identity differs")
    model = R4JointActionGenerator(config).to(device)
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("R12-R4 preflight strict restore differs")
    model.set_training_stage("bridge")
    generator = torch.Generator(device=device).manual_seed(20260806)
    belief = TeamBeliefState(
        tokens=torch.randn((2, 16, 96), generator=generator, device=device),
        agent_tokens=torch.randn((2, 4, 96), generator=generator, device=device),
        consensus_token=torch.randn((2, 96), generator=generator, device=device),
        uncertainty=torch.zeros((2, 1), device=device),
        agent_mask=torch.tensor(
            [[True, True, False, False], [True, True, True, True]], device=device
        ),
    ).validate()
    spatial = torch.randn(
        (2, 5, 48, 768), generator=generator, device=device
    )
    view_mask = torch.ones((2, 5), dtype=torch.bool, device=device)
    actions = torch.randn(
        (2, 100, 4, 8), generator=generator, device=device
    )
    step_mask = torch.ones((2, 100), dtype=torch.bool, device=device)
    model.zero_grad(set_to_none=True)
    with torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        loss = model.training_loss(
            belief, spatial, view_mask, actions, step_mask
        )["loss"]
    loss.backward()
    bridge_gradients = {}
    for name, parameter in model.bridge.named_parameters():
        gradient = parameter.grad
        bridge_gradients[name] = {
            "present": gradient is not None,
            "finite": bool(torch.isfinite(gradient).all()) if gradient is not None else False,
            "l1": float(gradient.abs().sum()) if gradient is not None else 0.0,
        }
    model.eval()
    noise = torch.randn(
        (2, 100, 32), generator=generator, device=device
    )
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        normal = model.sample(
            belief,
            spatial_tokens=spatial,
            spatial_view_mask=view_mask,
            noise=noise,
        )
        shuffled = model.sample(
            belief,
            spatial_tokens=spatial.flip(2),
            spatial_view_mask=view_mask,
            noise=noise,
        )
    values = normal.actions
    action_delta = float((values - shuffled.actions).abs().mean())
    checks = {
        "strict_restore": True,
        "loss_finite": bool(torch.isfinite(loss)),
        "all_bridge_gradients_present_finite_nonzero": all(
            row["present"] and row["finite"] and row["l1"] > 0
            for row in bridge_gradients.values()
        ),
        "sample_finite": bool(torch.isfinite(values).all()),
        "normalized_range": bool(values.abs().max() <= 5.0),
        "absent_agents_exact_zero": bool((values[0, :, 2:] == 0).all()),
        "spatial_row_order_changes_actions": action_delta > 0,
        "core_free_runtime": bool(checkpoint.get("core_free_runtime")),
    }
    result = {
        "schema_version": 1,
        "round": "R12-R4",
        "candidate_id": config.candidate_id,
        "passed": all(checks.values()),
        "checks": checks,
        "bridge_gradients": bridge_gradients,
        "spatial_shuffle_action_l1": action_delta,
        "checkpoint": str(Path(args.checkpoint).resolve()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
