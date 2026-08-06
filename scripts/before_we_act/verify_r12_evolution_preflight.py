#!/usr/bin/env python3
"""Strict R12-E1 restore, gradient and action-effect preflight."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from before_we_act.action_generator.evolution import (
    TaskConditionedActionGenerator,
    load_r12_evolution_config,
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
    config = load_r12_evolution_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("round") != "R12-E1"
        or checkpoint.get("candidate_id") != config.candidate_id
        or int(checkpoint.get("update", -1)) != 2
    ):
        raise ValueError("R12-E1 preflight checkpoint identity differs")
    model = TaskConditionedActionGenerator(config).to(device)
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("R12-E1 preflight strict restore differs")
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
    spatial = torch.randn((2, 5, 48, 768), generator=generator, device=device)
    view_mask = torch.ones((2, 5), dtype=torch.bool, device=device)
    actions = torch.randn((2, 100, 4, 8), generator=generator, device=device)
    step_mask = torch.ones((2, 100), dtype=torch.bool, device=device)
    task_index = torch.tensor([1, 2], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    with torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        loss = model.training_loss(
            belief,
            spatial,
            view_mask,
            task_index,
            actions,
            step_mask,
        )["loss"]
    loss.backward()
    gradients = {}
    for prefix, module in (
        ("bridge", model.bridge),
        ("task_embedding", model.task_embedding),
        ("task_film", model.task_film),
    ):
        for name, parameter in module.named_parameters():
            gradient = parameter.grad
            gradients[f"{prefix}.{name}"] = {
                "present": gradient is not None,
                "finite": bool(torch.isfinite(gradient).all())
                if gradient is not None
                else False,
                "l1": float(gradient.abs().sum()) if gradient is not None else 0.0,
            }
    model.eval()
    noise = torch.randn((2, 100, 32), generator=generator, device=device)
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        normal = model.sample(
            belief,
            spatial_tokens=spatial,
            spatial_view_mask=view_mask,
            task_index=task_index,
            noise=noise,
        )
        shuffled = model.sample(
            belief,
            spatial_tokens=spatial.flip(2),
            spatial_view_mask=view_mask,
            task_index=task_index,
            noise=noise,
        )
        changed_task = model.sample(
            belief,
            spatial_tokens=spatial,
            spatial_view_mask=view_mask,
            task_index=task_index.flip(0),
            noise=noise,
        )
    values = normal.actions
    spatial_delta = float((values - shuffled.actions).abs().mean())
    task_delta = float((values - changed_task.actions).abs().mean())
    checks = {
        "strict_restore": True,
        "loss_finite": bool(torch.isfinite(loss)),
        "all_bridge_and_task_gradients_present_finite": all(
            row["present"] and row["finite"] for row in gradients.values()
        ),
        "sample_finite": bool(torch.isfinite(values).all()),
        "normalized_range": bool(values.abs().max() <= 5.0),
        "absent_agents_exact_zero": bool((values[0, :, 2:] == 0).all()),
        "spatial_row_order_changes_actions": spatial_delta > 0,
        "task_condition_changes_actions": task_delta > 0,
        "native_image_primary_contract": (
            config.observation.get("input_height") == 480
            and config.observation.get("input_width") == 640
            and config.observation.get("compression_stage")
            == "adaptive_average_after_full_resolution_dinov3"
        ),
        "exact_w10_fallback_partition": (
            config.deployment.get("routing")
            == "explicit_task_id_exact_w10_fallback"
            and config.deployment.get("specialist_tasks")
            == ["three_robots_stack_cube"]
        ),
        "core_free_runtime": bool(checkpoint.get("core_free_runtime")),
    }
    result = {
        "schema_version": 1,
        "round": "R12-E1",
        "candidate_id": config.candidate_id,
        "passed": all(checks.values()),
        "checks": checks,
        "gradients": gradients,
        "spatial_shuffle_action_l1": spatial_delta,
        "task_change_action_l1": task_delta,
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
