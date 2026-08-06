#!/usr/bin/env python3
"""Measure whether an R12-R3 policy uses spatial and recovery evidence.

The formal Gate20 result remains the only quality gate.  This diagnostic is
deliberately read-only and attributes failures by comparing identical held-out
inputs with the learned spatial gate enabled, disabled, and sample-shuffled.
It also reports policy error separately on demonstration and on-policy recovery
rows; those sources must not be averaged together when diagnosing covariate
shift.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from before_we_act.action_generator.base import JointActionGenerator, load_r12_config
from before_we_act.data.action_windows import CachedActionWindows
from before_we_act.data.raw_team_windows import TASKS
from before_we_act.team_belief.base import PredictiveBeliefModel, load_r11_config


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_batch(batch: dict[str, torch.Tensor], device: torch.device):
    result = {}
    for key, value in batch.items():
        value = value.to(device, non_blocking=device.type == "cuda")
        if device.type == "cpu" and torch.is_floating_point(value):
            value = value.float()
        result[key] = value
    return result


def selected_indices(
    task_index: torch.Tensor,
    task: int,
    limit: int,
    *,
    source_index: torch.Tensor | None = None,
    source: int | None = None,
) -> list[int]:
    mask = task_index.eq(task)
    if source_index is not None and source is not None:
        mask &= source_index.eq(source)
    indices = mask.nonzero(as_tuple=False).flatten().tolist()
    if len(indices) < limit:
        raise ValueError(
            f"task {TASKS[task]} source {source} has only {len(indices)} rows; "
            f"requested {limit}"
        )
    return indices[:limit]


def normalized_prediction(
    model: JointActionGenerator,
    belief,
    batch: dict[str, torch.Tensor],
    noise: torch.Tensor,
) -> torch.Tensor:
    proposals = model.sample(
        belief,
        spatial_tokens=batch["spatial_tokens"],
        spatial_view_mask=batch["spatial_view_mask"],
        noise=noise,
    )
    return proposals.actions[:, 0].permute(0, 2, 1, 3).contiguous()


def add_metrics(
    totals: dict[str, float],
    prediction: torch.Tensor,
    target: torch.Tensor,
    agent_mask: torch.Tensor,
    step_mask: torch.Tensor,
) -> None:
    mask = (
        agent_mask[:, None, :, None]
        & step_mask[:, :, None, None]
    ).expand_as(prediction)
    first_mask = agent_mask[:, :, None].expand(-1, -1, prediction.shape[-1])
    totals["rows"] += len(prediction)
    totals["full_squared_error"] += float(
        ((prediction - target).square() * mask).sum()
    )
    totals["full_elements"] += int(mask.sum())
    totals["first_squared_error"] += float(
        ((prediction[:, 0] - target[:, 0]).square() * first_mask).sum()
    )
    totals["first_elements"] += int(first_mask.sum())
    totals["saturated"] += int(((prediction.abs() >= 4.999) & mask).sum())


def finish_metrics(totals: dict[str, float]) -> dict[str, float | int]:
    return {
        "rows": int(totals["rows"]),
        "first_step_normalized_mse": totals["first_squared_error"]
        / max(totals["first_elements"], 1),
        "full_chunk_normalized_mse": totals["full_squared_error"]
        / max(totals["full_elements"], 1),
        "normalized_clip_saturation_fraction": totals["saturated"]
        / max(totals["full_elements"], 1),
    }


def empty_totals() -> dict[str, float]:
    return {
        "rows": 0.0,
        "full_squared_error": 0.0,
        "full_elements": 0.0,
        "first_squared_error": 0.0,
        "first_elements": 0.0,
        "saturated": 0.0,
    }


@torch.inference_mode()
def evaluate_rows(
    model: JointActionGenerator,
    belief_model: PredictiveBeliefModel,
    dataset: CachedActionWindows,
    indices: Iterable[int],
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
    spatial_interventions: bool,
) -> dict:
    indices = list(indices)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    normal_totals = empty_totals()
    gate_zero_totals = empty_totals()
    shuffle_totals = empty_totals()
    spatial_absolute_delta = 0.0
    spatial_delta_elements = 0
    autocast = (
        lambda: torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    for batch_index, cpu_batch in enumerate(loader):
        batch = device_batch(cpu_batch, device)
        with autocast():
            belief = belief_model(batch)["belief"]
            generator = torch.Generator(device=device).manual_seed(
                seed + 1_000_003 * batch_index
            )
            noise = torch.randn(
                (len(batch["visual"]), model.horizon, model.max_agents * model.action_dim),
                generator=generator,
                device=device,
            )
            normal = normalized_prediction(model, belief, batch, noise)
        add_metrics(
            normal_totals,
            normal,
            batch["joint_actions"],
            batch["agent_mask"].bool(),
            batch["action_step_mask"].bool(),
        )
        if not spatial_interventions:
            continue
        saved_gate = model.spatial_gate.detach().clone()
        model.spatial_gate.zero_()
        try:
            with autocast():
                gate_zero = normalized_prediction(model, belief, batch, noise)
        finally:
            model.spatial_gate.copy_(saved_gate)
        permutation = torch.arange(
            len(batch["visual"]) - 1, -1, -1, device=device
        )
        shuffled_batch = dict(batch)
        shuffled_batch["spatial_tokens"] = batch["spatial_tokens"].index_select(
            0, permutation
        )
        shuffled_batch["spatial_view_mask"] = batch[
            "spatial_view_mask"
        ].index_select(0, permutation)
        with autocast():
            shuffled = normalized_prediction(model, belief, shuffled_batch, noise)
        add_metrics(
            gate_zero_totals,
            gate_zero,
            batch["joint_actions"],
            batch["agent_mask"].bool(),
            batch["action_step_mask"].bool(),
        )
        add_metrics(
            shuffle_totals,
            shuffled,
            batch["joint_actions"],
            batch["agent_mask"].bool(),
            batch["action_step_mask"].bool(),
        )
        action_mask = batch["agent_mask"][:, None, :, None].expand_as(normal)
        spatial_absolute_delta += float(((normal - gate_zero).abs() * action_mask).sum())
        spatial_delta_elements += int(action_mask.sum())
    result = {"normal": finish_metrics(normal_totals)}
    if spatial_interventions:
        result.update(
            {
                "gate_zero": finish_metrics(gate_zero_totals),
                "spatial_row_shuffle": finish_metrics(shuffle_totals),
                "normal_vs_gate_zero_action_l1": spatial_absolute_delta
                / max(spatial_delta_elements, 1),
            }
        )
    return result


def aggregate_normal(results: dict[str, dict]) -> dict:
    rows = sum(value["normal"]["rows"] for value in results.values())
    if not rows:
        raise ValueError("cannot aggregate empty R12 diagnostic")
    keys = (
        "first_step_normalized_mse",
        "full_chunk_normalized_mse",
        "normalized_clip_saturation_fraction",
    )
    return {
        "rows": rows,
        **{
            key: sum(
                value["normal"][key] * value["normal"]["rows"]
                for value in results.values()
            )
            / rows
            for key in keys
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--belief-config", required=True)
    parser.add_argument("--belief-checkpoint", required=True)
    parser.add_argument("--action-cache", required=True)
    parser.add_argument("--spatial-cache", required=True)
    parser.add_argument("--recovery-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows-per-task-source", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    if args.rows_per_task_source < 2 or args.batch_size < 2:
        raise ValueError("R12 diagnostic row and batch counts must be at least two")
    seed_everything(args.seed)
    torch.set_num_threads(min(12, os.cpu_count() or 1))
    device = torch.device(args.device)

    config = load_r12_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("candidate_id") != config.candidate_id:
        raise ValueError("R12 diagnostic checkpoint candidate differs")
    model = JointActionGenerator(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    belief_config = load_r11_config(args.belief_config)
    belief_payload = torch.load(
        args.belief_checkpoint, map_location="cpu", weights_only=False
    )
    belief_model = PredictiveBeliefModel(belief_config).to(device)
    belief_model.load_state_dict(belief_payload["model"], strict=True)
    belief_model.eval()

    validation = CachedActionWindows(
        args.action_cache,
        "validation",
        spatial_cache_path=args.spatial_cache,
    )
    train = CachedActionWindows(
        args.action_cache,
        "train",
        spatial_cache_path=args.spatial_cache,
        recovery_cache_path=args.recovery_cache,
    )
    validation_by_task = {}
    source_by_task: dict[str, dict] = {"demonstration": {}, "recovery": {}}
    for task_index, task in enumerate(TASKS):
        validation_by_task[task] = evaluate_rows(
            model,
            belief_model,
            validation,
            selected_indices(
                validation.task_index,
                task_index,
                args.rows_per_task_source,
            ),
            batch_size=args.batch_size,
            device=device,
            seed=args.seed + task_index,
            spatial_interventions=True,
        )
        for source, source_name in enumerate(("demonstration", "recovery")):
            source_by_task[source_name][task] = evaluate_rows(
                model,
                belief_model,
                train,
                selected_indices(
                    train.task_index,
                    task_index,
                    args.rows_per_task_source,
                    source_index=train.source_index,
                    source=source,
                ),
                batch_size=args.batch_size,
                device=device,
                seed=args.seed + 100 + 10 * source + task_index,
                spatial_interventions=False,
            )

    training = config.training
    recovery_probability = float(training["recovery_sampling_probability"])
    history_augmentation = float(training["history_augmentation_probability"])
    state = checkpoint["model"]
    result = {
        "schema_version": 1,
        "round": "R12-R3",
        "diagnostic": "spatial_gate_and_recovery_source_attribution",
        "created_at": now(),
        "candidate_id": config.candidate_id,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_update": int(checkpoint["update"]),
        "device": str(device),
        "sample_protocol": {
            "rows_per_task_source": args.rows_per_task_source,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "selection": "first deterministic rows within every task/source bucket",
        },
        "spatial_adapter": {
            "tanh_gate": float(torch.tanh(model.spatial_gate.detach()).cpu()),
            "projection_weight_l2": float(state["spatial_projection.weight"].float().norm()),
            "cross_attention_input_weight_l2": float(
                state["spatial_cross_attention.in_proj_weight"].float().norm()
            ),
            "cross_attention_output_weight_l2": float(
                state["spatial_cross_attention.out_proj.weight"].float().norm()
            ),
        },
        "validation_spatial_intervention_by_task": validation_by_task,
        "training_source_by_task": source_by_task,
        "training_source_aggregate": {
            source: aggregate_normal(values) for source, values in source_by_task.items()
        },
        "recovery_history_exposure": {
            "configured_recovery_sampling_probability": recovery_probability,
            "final_history_augmentation_probability_applied_to_both_sources": history_augmentation,
            "final_expected_recovery_rows_with_unmodified_history_per_task_batch": (
                recovery_probability * (1.0 - history_augmentation)
            ),
            "interpretation": (
                "recovery rows are not source-exempt in R12-R3 robustify_action_history"
            ),
        },
        "quality_gate": "diagnostic_only; formal Gate20 is authoritative",
    }
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
