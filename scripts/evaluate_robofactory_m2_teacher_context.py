#!/usr/bin/env python3
"""Phase-stratified teacher-context validation for RoboFactory M2."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Subset
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_robofactory_m2_inference import _build_vision  # noqa: E402
from train.m2_checkpointing import load_m2_checkpoint  # noqa: E402
from train.m2_training import (  # noqa: E402
    _agent_activity_mask,
    encode_m2_vision,
    m2_model_context,
)
from train.robofactory_multitask_dataset import (  # noqa: E402
    RoboFactoryMultitaskDataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a trained M2 checkpoint per task and per agent on exact "
            "expert histories. This diagnoses hidden handoff failures before "
            "expensive closed-loop rollout."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/wam_multimodal/m2_liftbarrier_longpipeline_joint.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "checkpoints/phase_m2_liftbarrier_longpipeline_multiview_640x480_seed101",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/phase_m2_liftbarrier_longpipeline_multiview_640x480/"
        "teacher_context_validation.json",
    )
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--samples-per-task", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples_per_task <= 0 or args.batch_size <= 0:
        raise ValueError("validation sample and batch counts must be positive")
    config_path = args.config.expanduser().resolve(strict=True)
    config = _load_yaml(config_path)
    data = _mapping(config, "data")
    manifests = [
        (ROOT / str(value)).resolve(strict=True)
        for value in _sequence(data, "manifests")
    ]
    dataset = RoboFactoryMultitaskDataset(
        manifests,
        split=args.split,
        state_history=int(data.get("state_history", 16)),
        action_horizon=int(data.get("action_horizon", 16)),
        task_action_horizons=_mapping(data, "task_action_horizons"),
        visual_history=int(data.get("visual_history_frames", 4)),
        future_horizons=tuple(data.get("future_visual_horizons", [1, 4, 8, 16])),
        cameras=tuple(data.get("camera_order", ["global"])),
        max_state_dim=int(data.get("max_state_dim", 72)),
        max_action_dim=int(data.get("max_action_dim", 32)),
        max_agents=int(data.get("max_agents", 4)),
        max_text_tokens=int(data.get("max_text_tokens", 96)),
        stride=int(data.get("stride", 1)),
        hdf5_cache_size=int(data.get("hdf5_cache_size", 16)),
        verify_hdf5_sha256=True,
        verify_hdf5_contract=True,
        verify_normalization=True,
    )
    expected_shape = (
        int(data.get("image_height", 480)),
        int(data.get("image_width", 640)),
        3,
    )
    if tuple(dataset.image_shape_hwc) != expected_shape:
        dataset.close()
        raise ValueError(
            f"validation source RGB must be {expected_shape}, "
            f"got {dataset.image_shape_hwc}"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA validation was requested but is unavailable")
    if args.precision == "bf16" and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("BF16 validation requires a CUDA device with native BF16")
    model, runtime, schema = load_m2_checkpoint(
        args.checkpoint.expanduser().resolve(strict=True),
        device=device,
    )
    if tuple(dataset.task_vocabulary) != tuple(schema["task_vocabulary"]):
        raise ValueError("validation task order differs from the checkpoint")
    for contract, checkpoint_task in zip(
        dataset.contracts,
        runtime,
        strict=True,
    ):
        if any(
            checkpoint_task.get(name) != getattr(contract, name)
            for name in (
                "task_id",
                "manifest_sha256",
                "action_codec_sha256",
                "normalization_sha256",
                "source_conversion_sha256",
                "action_horizon",
            )
        ):
            raise ValueError(
                f"validation lineage differs for task {contract.task_id!r}"
            )
    vision = _build_vision(config, schema=schema).to(device).eval()
    model.eval()
    for module in (model, vision):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    selected: list[int] = []
    selected_by_task: dict[str, int] = {}
    for task_index, (contract, task_dataset) in enumerate(
        zip(dataset.contracts, dataset.datasets, strict=True)
    ):
        count = min(args.samples_per_task, len(task_dataset))
        if count <= 0:
            raise ValueError(
                f"validation split has no windows for {contract.task_id!r}"
            )
        if count == 1:
            local_indices = [len(task_dataset) // 2]
        else:
            local_indices = [
                round(position * (len(task_dataset) - 1) / (count - 1))
                for position in range(count)
            ]
        selected.extend(dataset._offsets[task_index] + value for value in local_indices)
        selected_by_task[contract.task_id] = len(local_indices)

    loader = DataLoader(
        Subset(dataset, selected),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    threshold = float(
        _mapping(config, "training").get("active_agent_delta_threshold", 0.005)
    )
    execution_steps = int(_mapping(schema, "action_generation")["execution_steps"])
    accumulators: dict[str, list[dict[str, dict[str, float]]]] = {
        contract.task_id: [
            {
                name: {"squared_error": 0.0, "scalar_count": 0.0}
                for name in ("all", "active", "inactive", "executed_prefix")
            }
            for _ in range(contract.agent_count)
        ]
        for contract in dataset.contracts
    }
    generation = _mapping(schema, "action_generation")
    completed = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = {
                name: value.to(device, non_blocking=device.type == "cuda")
                for name, value in raw_batch.items()
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=args.precision == "bf16",
            ):
                visual, _ = encode_m2_vision(
                    vision,
                    batch,
                    spatial_grid_height=model.config.visual_grid_height,
                    spatial_grid_width=model.config.visual_grid_width,
                )
                context = m2_model_context(batch, visual)
                predicted = model.generate_actions(
                    context,
                    solver_steps=int(generation["solver_steps"]),
                    solver=str(generation["solver"]),
                    normalized_clip=float(generation["normalized_action_clip"]),
                )
            targets = batch["action_targets"]
            active, agent_valid = _agent_activity_mask(
                targets,
                target_valid=batch["action_target_valid_mask"].bool(),
                action_dimension_mask=batch["action_dimension_mask"].bool(),
                past_actions=batch["past_actions"],
                past_action_valid_mask=batch["past_action_valid_mask"].bool(),
                max_agents=model.config.max_agents,
                delta_threshold=threshold,
            )
            per_agent_dim = model.config.max_action_dim // model.config.max_agents
            squared = (predicted.float() - targets.float()).square().reshape(
                targets.shape[0],
                targets.shape[1],
                model.config.max_agents,
                per_agent_dim,
            )
            prefix = (
                torch.arange(targets.shape[1], device=device)[None, :, None]
                < execution_steps
            ) & agent_valid
            for sample in range(targets.shape[0]):
                task_index = int(batch["task_index"][sample])
                contract = dataset.contracts[task_index]
                for agent in range(contract.agent_count):
                    for name, mask in (
                        ("all", agent_valid[sample, :, agent]),
                        ("active", active[sample, :, agent]),
                        (
                            "inactive",
                            agent_valid[sample, :, agent]
                            & ~active[sample, :, agent],
                        ),
                        ("executed_prefix", prefix[sample, :, agent]),
                    ):
                        values = squared[sample, :, agent][mask]
                        if values.numel():
                            bucket = accumulators[contract.task_id][agent][name]
                            bucket["squared_error"] += float(values.sum().cpu())
                            bucket["scalar_count"] += float(values.numel())
            completed += targets.shape[0]
            print(
                f"\rteacher-context validation {completed}/{len(selected)}",
                end="",
                flush=True,
            )
    print()

    tasks: dict[str, Any] = {}
    for contract in dataset.contracts:
        agents = []
        for agent_index, buckets in enumerate(accumulators[contract.task_id]):
            metrics: dict[str, Any] = {"agent_index": agent_index}
            for name, bucket in buckets.items():
                count = int(bucket["scalar_count"])
                metrics[f"{name}_scalar_count"] = count
                metrics[f"{name}_normalized_rmse"] = (
                    None
                    if count == 0
                    else math.sqrt(bucket["squared_error"] / count)
                )
            agents.append(metrics)
        tasks[contract.task_id] = {
            "samples": selected_by_task[contract.task_id],
            "agents": agents,
        }
    report = {
        "format_version": "wam.robofactory.m2.teacher_context_validation/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "split": args.split,
        "metric_space": "per_task_zscore_canonical_unit_action",
        "activity_delta_threshold": threshold,
        "execution_steps": execution_steps,
        "spatial_visual_grid": [
            model.config.visual_grid_height,
            model.config.visual_grid_width,
        ],
        "tasks": tasks,
        "passed": all(
            agent["active_scalar_count"] > 0
            and agent["active_normalized_rmse"] is not None
            and math.isfinite(agent["active_normalized_rmse"])
            for task in tasks.values()
            for agent in task["agents"]
        ),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({"output": str(output), "passed": report["passed"]}))
    dataset.close()
    return 0 if report["passed"] else 1


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("M2 config root must be an object")
    return payload


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"M2 config/checkpoint {key!r} must be an object")
    return result


def _sequence(value: Mapping[str, Any], key: str) -> list[Any]:
    result = value.get(key)
    if not isinstance(result, list):
        raise ValueError(f"M2 config {key!r} must be a list")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
