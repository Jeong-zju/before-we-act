#!/usr/bin/env python3
"""Train the RoboFactory-only block-causal Phase M2 model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.wam_multimodal import (  # noqa: E402
    BlockCausalWAM,
    BlockCausalWAMConfig,
    FrozenDINOv3Config,
    FrozenDINOv3Encoder,
)
from train.m2_checkpointing import (  # noqa: E402
    load_m2_checkpoint,
    save_m2_checkpoint,
)
from train.m2_resume import (  # noqa: E402
    load_latest_m2_resume_checkpoint,
    save_m2_resume_checkpoint,
)
from train.m2_training import (  # noqa: E402
    DevicePrefetcher,
    M2LossWeights,
    RGBStatisticsVisionEncoder,
    encode_m2_vision,
    m2_batch_loss,
)
from train.progress import TrainingProgress  # noqa: E402
from train.robofactory_multitask_dataset import (  # noqa: E402
    CoverageTemperatureDistributedSampler,
    RoboFactoryMultitaskDataset,
)


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    enabled: bool

    @property
    def primary(self) -> bool:
        return self.rank == 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a task-balanced block-causal WAM exclusively from audited "
            "RoboFactory task manifests."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/wam_multimodal/m2_causal_wam.yaml",
    )
    parser.add_argument(
        "--manifests",
        type=Path,
        nargs="+",
        help="Override config data.manifests (used by the one-task local smoke).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=16)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--progress-log", type=Path)
    parser.add_argument("--checkpoint-interval-steps", type=int)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Require a fresh run and fail if a published resume snapshot exists.",
    )
    return parser


def _emit_stage(stage: str, detail: str, *, primary: bool = True) -> None:
    if not primary:
        return
    payload = {
        "event": "startup_stage",
        "stage": stage,
        "detail": detail,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    stage_log = os.environ.get("LPD_STAGE_LOG")
    if stage_log:
        _append_progress_event(Path(stage_log).expanduser().resolve(), payload)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.torch_threads <= 0:
        raise ValueError("--torch-threads must be positive")
    config_path = args.config.expanduser().resolve(strict=True)
    _emit_stage("config", f"loading {config_path}")
    config = _load_yaml(config_path)
    distributed = _distributed_context()
    device = _select_device(args.device, distributed)
    _emit_stage(
        "cuda_preflight",
        f"checking {device} with world_size={distributed.world_size}",
        primary=distributed.primary,
    )
    _configure_compute(config, args.torch_threads, device=device)
    training = _mapping(config, "training")
    action_generation = _action_generation(config, training=training)
    train_seed = int(training.get("seed", 101) if args.seed is None else args.seed)
    if train_seed < 0:
        raise ValueError("M2 train seed must be non-negative")
    _seed_all(train_seed + distributed.rank)

    data_config = _mapping(config, "data")
    manifests = (
        [path.expanduser().resolve(strict=True) for path in args.manifests]
        if args.manifests
        else [
            (ROOT / str(path)).resolve(strict=True)
            for path in _sequence(data_config, "manifests")
        ]
    )
    if not args.smoke and len(manifests) < int(data_config.get("minimum_tasks", 2)):
        raise ValueError("formal M2 training requires the configured RoboFactory task count")
    _emit_stage(
        "dataset_validation",
        "verifying manifests, HDF5 identities and normalization statistics",
        primary=distributed.primary,
    )
    dataset = _build_dataset(
        manifests,
        data_config=data_config,
        smoke=args.smoke,
    )
    _emit_stage(
        "dataset_ready",
        f"{len(dataset)} train windows across {len(dataset.contracts)} tasks",
        primary=distributed.primary,
    )
    incompatible_horizons = {
        contract.task_id: contract.action_horizon
        for contract in dataset.contracts
        if contract.action_horizon <= int(action_generation["execution_steps"])
    }
    if incompatible_horizons:
        raise ValueError(
            "every task action horizon must exceed execution_steps; got "
            f"{incompatible_horizons}"
        )
    model_config = _model_config(
        config,
        num_tasks=len(dataset.contracts),
        smoke=args.smoke,
    )
    _emit_stage(
        "dinov3_load",
        "loading and verifying frozen DINOv3 weights",
        primary=distributed.primary,
    )
    vision, vision_identity = _build_vision(config, model_config=model_config, smoke=args.smoke)
    vision = vision.to(device).eval()
    for parameter in vision.parameters():
        parameter.requires_grad_(False)
    _emit_stage(
        "model_build",
        "allocating the rectified-flow policy and optimizer",
        primary=distributed.primary,
    )
    model = BlockCausalWAM(model_config).to(device)
    stages = _stages(training, smoke=args.smoke)
    if not args.smoke:
        batch_size = int(training.get("batch_size", 16))
        coverage_steps = math.ceil(
            len(dataset) / (batch_size * distributed.world_size)
        )
        if sum(int(stage["steps"]) for stage in stages) < coverage_steps:
            raise ValueError(
                "formal M2 training budget cannot complete the required first "
                f"coverage pass: need at least {coverage_steps} optimizer steps"
            )
    checkpoint_config = _mapping(config, "checkpoint")
    checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else (ROOT / str(checkpoint_config["output"])).resolve()
    )
    report_path = (
        args.report.expanduser().resolve()
        if args.report is not None
        else (ROOT / str(checkpoint_config["report"])).resolve()
    )
    resume_dir = (
        args.resume_dir.expanduser().resolve()
        if args.resume_dir is not None
        else (
            (ROOT / str(checkpoint_config["resume"])).resolve()
            if checkpoint_config.get("resume") is not None
            else checkpoint.with_name(f"{checkpoint.name}.resume")
        )
    )
    progress_log = (
        args.progress_log.expanduser().resolve()
        if args.progress_log is not None
        else (
            (ROOT / str(checkpoint_config["progress_log"])).resolve()
            if checkpoint_config.get("progress_log") is not None
            else report_path.with_suffix(".progress.jsonl")
        )
    )
    checkpoint_interval_steps = (
        int(args.checkpoint_interval_steps)
        if args.checkpoint_interval_steps is not None
        else int(checkpoint_config.get("interval_steps", 0 if args.smoke else 100))
    )
    checkpoint_keep_last = int(checkpoint_config.get("keep_last", 2))
    if checkpoint_interval_steps < 0:
        raise ValueError("M2 checkpoint interval cannot be negative")
    if checkpoint_keep_last <= 0:
        raise ValueError("M2 checkpoint keep_last must be positive")
    # Every rank performs the same shared-filesystem preflight.  If only rank 0
    # raised here, the remaining ranks would block forever at the next barrier.
    if checkpoint.exists() and any(checkpoint.iterdir()):
        raise FileExistsError(f"M2 checkpoint already exists: {checkpoint}")
    if report_path.exists():
        raise FileExistsError(f"M2 report already exists: {report_path}")
    if args.no_resume and (resume_dir / "latest.json").exists():
        raise FileExistsError(
            f"M2 resume snapshot exists but --no-resume was requested: {resume_dir}"
        )
    _barrier(distributed)

    sampling = _sampling_config(
        training,
        dataset=dataset,
        smoke=args.smoke,
        world_size=distributed.world_size,
    )
    sampler = CoverageTemperatureDistributedSampler(
        dataset,
        samples_per_epoch=int(sampling["samples_per_epoch"]),
        coverage_epochs=int(sampling["coverage_epochs"]),
        temperature_alpha=float(sampling["temperature_alpha"]),
        seed=train_seed,
        rank=distributed.rank,
        replicas=distributed.world_size,
    )
    loader_memory_plan = _loader_memory_plan(
        dataset,
        training=training,
        smoke=args.smoke,
    )
    loader_generator = torch.Generator().manual_seed(
        train_seed + distributed.rank + 100_000
    )
    _emit_stage(
        "dataloader_build",
        f"configuring {_loader_workers(training, smoke=args.smoke)} workers",
        primary=distributed.primary,
    )
    loader = _build_loader(
        dataset,
        sampler=sampler,
        training=training,
        smoke=args.smoke,
        generator=loader_generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(stages[0]["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 1e-5)),
        fused=device.type == "cuda",
    )
    train_model: nn.Module = model
    if distributed.enabled:
        train_model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=True,
            gradient_as_bucket_view=True,
        )
    precision = "fp32" if args.smoke else str(training.get("precision", "bf16"))
    if precision not in {"fp32", "bf16"}:
        raise ValueError("M2 precision must be fp32 or bf16")
    if precision == "bf16" and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("M2 BF16 training requires a CUDA GPU with native BF16")

    resume_identity = {
        "config_sha256": _sha256(config_path),
        "dataset_lineage_sha256": dataset.lineage_sha256(),
        "model_config": model.config.to_dict(),
        "stages": stages,
        "train_seed": train_seed,
        "world_size": distributed.world_size,
        "precision": precision,
        "batch_size": 2 if args.smoke else int(training.get("batch_size", 16)),
    }
    resume_state = (
        None
        if args.no_resume
        else load_latest_m2_resume_checkpoint(
            resume_dir,
            model=model,
            optimizer=optimizer,
            expected_identity=resume_identity,
            device=device,
        )
    )
    _emit_stage(
        "resume_ready",
        (
            "no resume checkpoint; starting from update 0"
            if resume_state is None
            else f"resuming from global_step={resume_state['global_step']}"
        ),
        primary=distributed.primary,
    )
    total_stages = len(stages) + 2
    history: list[dict[str, Any]] = (
        [] if resume_state is None else list(resume_state.get("history", []))
    )
    preload_report: dict[str, Any]
    coverage_seen = (
        torch.zeros(len(dataset), dtype=torch.bool)
        if resume_state is None
        else resume_state["coverage_seen"].detach().to("cpu", dtype=torch.bool)
    )
    if coverage_seen.shape != (len(dataset),):
        raise ValueError("M2 resume coverage tensor differs from the dataset")
    resume_stage_index = (
        0 if resume_state is None else int(resume_state.get("stage_index", -1))
    )
    resume_stage_step = (
        0 if resume_state is None else int(resume_state.get("stage_step", -1))
    )
    epoch = 0 if resume_state is None else int(resume_state.get("epoch", -1))
    samples_consumed_in_epoch = (
        0
        if resume_state is None
        else int(resume_state.get("samples_consumed_in_epoch", -1))
    )
    global_step = (
        0 if resume_state is None else int(resume_state.get("global_step", -1))
    )
    elapsed_before_resume = (
        0.0
        if resume_state is None
        else float(resume_state.get("elapsed_seconds", 0.0))
    )
    resumed_stage_last = (
        None if resume_state is None else resume_state.get("current_stage_last")
    )
    _validate_resume_progress(
        stages=stages,
        history=history,
        stage_index=resume_stage_index,
        stage_step=resume_stage_step,
        global_step=global_step,
        epoch=epoch,
        samples_consumed_in_epoch=samples_consumed_in_epoch,
        local_samples=sampler.local_samples,
        batch_size=2 if args.smoke else int(training.get("batch_size", 16)),
        current_stage_last=resumed_stage_last,
    )
    started = time.perf_counter()
    if distributed.primary:
        _append_progress_event(
            progress_log,
            {
                "event": "run_started",
                "resume_generation": (
                    None if resume_state is None else resume_state["generation"]
                ),
                "global_step": global_step,
                "stage_index": resume_stage_index,
                "stage_step": resume_stage_step,
                "epoch": epoch,
                "samples_consumed_in_epoch": samples_consumed_in_epoch,
                "num_workers": _loader_workers(training, smoke=args.smoke),
                "prefetch_factor": (
                    int(training.get("prefetch_factor", 4))
                    if _loader_workers(training, smoke=args.smoke)
                    else None
                ),
                "batch_size": (
                    2 if args.smoke else int(training.get("batch_size", 16))
                ),
                "image_shape_hwc": list(dataset.image_shape_hwc),
                "loader_memory_plan": loader_memory_plan,
            },
        )
    with TrainingProgress(
        enabled=distributed.primary and not args.no_progress,
        total_stages=total_stages,
    ) as progress:
        preload = bool(training.get("preload_to_ram", True)) and not args.smoke
        if preload:
            estimate = dataset.estimate_ram_preload_bytes()
            available = _available_memory_bytes()
            fraction = float(training.get("preload_max_available_fraction", 0.5))
            per_rank_budget = int(available * fraction / distributed.world_size)
            if estimate > per_rank_budget:
                raise MemoryError(
                    "M2 RAM preload exceeds the per-rank safety budget: "
                    f"estimate={estimate}, budget={per_rank_budget}"
                )
            if bool(training.get("preload_shared_memory", False)):
                shared_budget = int(
                    _shared_memory_available_bytes() * 0.9 / distributed.world_size
                )
                if estimate > shared_budget:
                    raise MemoryError(
                        "M2 shared-RAM preload exceeds /dev/shm per-rank budget: "
                        f"estimate={estimate}, budget={shared_budget}"
                    )
            _emit_stage(
                "ram_preload",
                f"loading {estimate} bytes from the train split before update 1",
                primary=distributed.primary,
            )
            phase = progress.add_phase(
                "preload RoboFactory train split to RAM",
                sum(len(value.records) for value in dataset.datasets),
                show_loss_chart=False,
            )
            previous = 0

            def update(current: int, _total: int, loaded: int) -> None:
                nonlocal previous
                for _ in range(max(0, current - previous)):
                    phase.advance({"bytes": loaded})
                previous = current

            preload_report = dataset.preload_to_ram(
                shared_memory=bool(training.get("preload_shared_memory", False)),
                progress_callback=update,
            )
            phase.finish(f"{preload_report['bytes']} bytes")
            _emit_stage(
                "ram_preload_ready",
                f"loaded {preload_report['bytes']} bytes",
                primary=distributed.primary,
            )
        else:
            phase = progress.add_phase("RAM preload policy", 1, show_loss_chart=False)
            preload_report = {
                "enabled": False,
                "reason": "smoke" if args.smoke else "disabled_by_config",
            }
            phase.advance({"batch": 1})
            phase.finish(preload_report["reason"])

        _emit_stage(
            "dataloader_start",
            "starting workers and waiting for the first HDF5 batch",
            primary=distributed.primary,
        )
        sampler.set_epoch(
            epoch,
            start_offset=samples_consumed_in_epoch,
        )
        iterator = DevicePrefetcher(iter(loader), device)
        first_batch_announced = False
        for stage_index, stage in enumerate(stages):
            if stage_index < resume_stage_index:
                continue
            initial_stage_step = (
                resume_stage_step if stage_index == resume_stage_index else 0
            )
            stage_steps = int(stage["steps"])
            stage_last: dict[str, float] | None = (
                resumed_stage_last
                if stage_index == resume_stage_index
                and isinstance(resumed_stage_last, dict)
                else None
            )
            if initial_stage_step == stage_steps:
                assert stage_last is not None
                history.append(
                    {
                        "name": str(stage["name"]),
                        "steps": stage_steps,
                        "learning_rate": float(stage["learning_rate"]),
                        "final": stage_last,
                    }
                )
                resumed_stage_last = None
                continue
            for group in optimizer.param_groups:
                group["lr"] = float(stage["learning_rate"])
            weights = M2LossWeights(**_mapping(stage, "losses"))
            phase = progress.add_phase(
                (
                    f"train {stage['name']}"
                    if initial_stage_step == 0
                    else f"resume {stage['name']} after step {initial_stage_step}"
                ),
                stage_steps - initial_stage_step,
                show_loss_chart=True,
            )
            train_model.train()
            vision.eval()
            for completed in range(initial_stage_step + 1, stage_steps + 1):
                batch = iterator.next()
                if batch is None:
                    epoch += 1
                    samples_consumed_in_epoch = 0
                    sampler.set_epoch(epoch, start_offset=0)
                    iterator = DevicePrefetcher(iter(loader), device)
                    batch = iterator.next()
                    if batch is None:
                        raise RuntimeError("M2 DataLoader yielded no batches")
                if not first_batch_announced:
                    _emit_stage(
                        "optimizer_training",
                        f"first batch ready; entering stage {stage['name']}",
                        primary=distributed.primary,
                    )
                    first_batch_announced = True
                samples_consumed_in_epoch += int(batch["dataset_index"].shape[0])
                coverage_seen[
                    batch["dataset_index"].detach().to("cpu", dtype=torch.long)
                ] = True
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=precision == "bf16",
                ):
                    visual, future_visual = encode_m2_vision(
                        vision,
                        batch,
                        spatial_grid_height=model.config.visual_grid_height,
                        spatial_grid_width=model.config.visual_grid_width,
                    )
                    loss = m2_batch_loss(
                        train_model,  # type: ignore[arg-type]
                        batch,
                        visual_features=visual,
                        future_visual_targets=future_visual,
                        weights=weights,
                        warm_start_probability=float(
                            training.get("warm_start_probability", 0.5)
                        ),
                        warm_start_noise_std=float(
                            training.get("warm_start_noise_std", 0.01)
                        ),
                        execution_steps=int(action_generation["execution_steps"]),
                        executed_prefix_weight=float(
                            training.get("executed_prefix_weight", 1.0)
                        ),
                        past_action_history_dropout_probability=float(
                            training.get(
                                "past_action_history_dropout_probability",
                                0.0,
                            )
                        ),
                        state_history_noise_std=float(
                            training.get("state_history_noise_std", 0.0)
                        ),
                        past_action_history_noise_std=float(
                            training.get("past_action_history_noise_std", 0.0)
                        ),
                        solver_steps=int(action_generation["solver_steps"]),
                        solver=str(action_generation["solver"]),
                        normalized_action_clip=float(
                            action_generation["normalized_action_clip"]
                        ),
                    )
                if not bool(torch.isfinite(loss.total)):
                    raise FloatingPointError(f"M2 stage {stage['name']} produced non-finite loss")
                loss.total.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(training.get("gradient_clip_norm", 1.0)),
                    error_if_nonfinite=True,
                )
                optimizer.step()
                metrics = loss.detached_metrics()
                metrics["gradient_norm"] = float(gradient_norm.detach().float().cpu())
                metrics["step"] = float(completed)
                stage_last = metrics
                global_step += 1
                phase.advance({"loss": metrics["total"], "step": completed})
                if distributed.primary:
                    _append_progress_event(
                        progress_log,
                        {
                            "event": "optimizer_step",
                            "global_step": global_step,
                            "stage_index": stage_index,
                            "stage_name": str(stage["name"]),
                            "stage_step": completed,
                            "stage_steps": stage_steps,
                            "epoch": epoch,
                            "samples_consumed_in_epoch": (
                                samples_consumed_in_epoch
                            ),
                            "loss": metrics["total"],
                            "gradient_norm": metrics["gradient_norm"],
                            "memory_available_bytes": _available_memory_bytes(),
                        },
                    )
                if (
                    checkpoint_interval_steps > 0
                    and global_step % checkpoint_interval_steps == 0
                ):
                    _barrier(distributed)
                    resume_summary: dict[str, Any] | None = None
                    if distributed.primary:
                        resume_summary = save_m2_resume_checkpoint(
                            resume_dir,
                            model=model,
                            optimizer=optimizer,
                            identity=resume_identity,
                            progress={
                                "stage_index": stage_index,
                                "stage_step": completed,
                                "global_step": global_step,
                                "epoch": epoch,
                                "samples_consumed_in_epoch": (
                                    samples_consumed_in_epoch
                                ),
                                "history": history,
                                "current_stage_last": metrics,
                                "elapsed_seconds": (
                                    elapsed_before_resume
                                    + time.perf_counter()
                                    - started
                                ),
                            },
                            coverage_seen=coverage_seen,
                            keep_last=checkpoint_keep_last,
                        )
                        _append_progress_event(
                            progress_log,
                            {
                                "event": "resume_checkpoint_saved",
                                **resume_summary,
                            },
                        )
                    _barrier(distributed)
            assert stage_last is not None
            phase.finish(f"loss={stage_last['total']:.5f}")
            history.append(
                {
                    "name": str(stage["name"]),
                    "steps": stage_steps,
                    "learning_rate": float(stage["learning_rate"]),
                    "final": stage_last,
                }
            )
            resumed_stage_last = None

        save_phase = progress.add_phase("save and strict reload M2", 2, show_loss_chart=False)
        _barrier(distributed)
        checkpoint_summary: dict[str, Any] | None = None
        reload_max_difference = 0.0
        if distributed.primary:
            runtime = _task_runtime(dataset)
            checkpoint_summary = save_m2_checkpoint(
                checkpoint,
                model=model,
                task_runtime=runtime,
                vision_identity=vision_identity,
                action_generation=action_generation,
                action_objective={
                    "tail_windows": "repeat_last_with_validity_masks",
                    "visual_prefix_windows": (
                        "left_zero_pad_with_validity_mask"
                    ),
                    "task_horizons": (
                        "max_tensor_with_task_validity_masks"
                    ),
                    "loss_reduction": "per_sample_valid_element_mean",
                    "executed_prefix_weight": float(
                        training.get("executed_prefix_weight", 1.0)
                    ),
                },
                training={
                    "formal_protocol": not args.smoke,
                    "smoke": bool(args.smoke),
                    "distributed_world_size": distributed.world_size,
                    "train_seed": train_seed,
                    "precision": precision,
                    "dataset_lineage_sha256": dataset.lineage_sha256(),
                    "config_path": str(config_path),
                    "sampling": sampler.summary(),
                    "past_action_history_dropout_probability": float(
                        training.get(
                            "past_action_history_dropout_probability",
                            0.0,
                        )
                    ),
                    "state_history_noise_std": float(
                        training.get("state_history_noise_std", 0.0)
                    ),
                    "past_action_history_noise_std": float(
                        training.get("past_action_history_noise_std", 0.0)
                    ),
                },
                metrics={"stages": history},
            )
            save_phase.advance({"batch": 1})
            reloaded, _, _ = load_m2_checkpoint(checkpoint, device=device)
            differences = []
            reloaded_state = reloaded.state_dict()
            for name, value in model.state_dict().items():
                restored = reloaded_state[name]
                if value.dtype == torch.bool:
                    difference = float(torch.logical_xor(value, restored).any().cpu())
                else:
                    difference = float(
                        (value.detach() - restored).abs().max().cpu()
                    )
                differences.append(difference)
            reload_max_difference = max(differences, default=0.0)
            if reload_max_difference != 0.0:
                raise RuntimeError("strict M2 checkpoint reload changed parameters")
            save_phase.advance({"batch": 2})
            save_phase.finish("strict reload max_diff=0")
        else:
            save_phase.advance({"batch": 1})
            save_phase.advance({"batch": 2})
            save_phase.finish("rank checkpoint barrier complete")
        _barrier(distributed)

    coverage = _coverage_summary(
        coverage_seen,
        dataset=dataset,
        distributed=distributed,
        device=device,
    )
    if distributed.primary:
        report = {
            "format_version": "wam.robofactory.m2.training_report/5",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "formal_protocol": not args.smoke,
            "smoke": bool(args.smoke),
            "passed": True,
            "data_source_policy": "robofactory_only_no_custom_scenes",
            "dataset": dataset.summary(),
            "dataset_lineage_sha256": dataset.lineage_sha256(),
            "sampling": {
                **sampler.summary(),
                "actual_coverage": coverage,
            },
            "ram_preload": preload_report,
            "model": {
                "config": model.config.to_dict(),
                "trainable_parameters": model.trainable_parameters,
                "block_causal_mask_verified_by_tests": True,
            },
            "action_space": "per_task_zscore_canonical_unit_action",
            "action_generation": action_generation,
            "action_objective": {
                "tail_windows": "repeat_last_with_validity_masks",
                "visual_prefix_windows": "left_zero_pad_with_validity_mask",
                "task_horizons": "max_tensor_with_task_validity_masks",
                "loss_reduction": "per_sample_valid_element_mean",
                "executed_prefix_weight": float(
                    training.get("executed_prefix_weight", 1.0)
                ),
                "past_action_history_dropout_probability": float(
                    training.get(
                        "past_action_history_dropout_probability",
                        0.0,
                    )
                ),
                "state_history_noise_std": float(
                    training.get("state_history_noise_std", 0.0)
                ),
                "past_action_history_noise_std": float(
                    training.get("past_action_history_noise_std", 0.0)
                ),
            },
            "vision": vision_identity,
            "compute": {
                "device": str(device),
                "world_size": distributed.world_size,
                "train_seed": train_seed,
                "precision": precision,
                "torch_threads": args.torch_threads,
                "pin_memory": bool(training.get("pin_memory", True)) and device.type == "cuda",
                "num_workers": _loader_workers(training, smoke=args.smoke),
                "prefetch_factor": int(training.get("prefetch_factor", 4)),
                "checkpoint_interval_steps": checkpoint_interval_steps,
                "resume_directory": str(resume_dir),
                "progress_log": str(progress_log),
                "loader_memory_plan": loader_memory_plan,
            },
            "stages": history,
            "checkpoint": checkpoint_summary,
            "strict_reload_max_abs_difference": reload_max_difference,
            "elapsed_seconds": (
                elapsed_before_resume + time.perf_counter() - started
            ),
            "acceptance_scope": (
                "engineering_smoke_only_closed_loop_rollout_required_for_model_quality"
                if args.smoke
                else "formal_training_complete_closed_loop_rollout_still_required"
            ),
        }
        _write_json_atomic(report_path, report)
        _append_progress_event(
            progress_log,
            {
                "event": "training_completed",
                "global_step": global_step,
                "report": str(report_path),
                "checkpoint": str(checkpoint),
            },
        )
        print(json.dumps({"report": str(report_path), "checkpoint": str(checkpoint), "passed": True}))
    dataset.close()
    _shutdown_distributed(distributed)
    return 0


def _build_dataset(
    manifests: Sequence[Path],
    *,
    data_config: Mapping[str, Any],
    smoke: bool,
) -> RoboFactoryMultitaskDataset:
    dataset = RoboFactoryMultitaskDataset(
        manifests,
        split="train",
        state_history=int(data_config.get("state_history", 16)),
        action_horizon=int(data_config.get("action_horizon", 16)),
        task_action_horizons=(
            None
            if data_config.get("task_action_horizons") is None
            else _mapping(data_config, "task_action_horizons")
        ),
        visual_history=int(data_config.get("visual_history_frames", 4)),
        future_horizons=tuple(data_config.get("future_visual_horizons", [1, 4, 8, 16])),
        cameras=tuple(data_config.get("camera_order", ["global"])),
        max_state_dim=int(data_config.get("max_state_dim", 72)),
        max_action_dim=int(data_config.get("max_action_dim", 32)),
        max_agents=int(data_config.get("max_agents", 4)),
        max_text_tokens=int(data_config.get("max_text_tokens", 96)),
        stride=int(data_config.get("smoke_stride" if smoke else "stride", 1)),
        hdf5_cache_size=int(data_config.get("hdf5_cache_size", 16)),
        verify_hdf5_sha256=not smoke,
        verify_hdf5_contract=not smoke,
        verify_normalization=True,
    )
    expected_shape = (
        int(data_config.get("image_height", 480)),
        int(data_config.get("image_width", 640)),
        3,
    )
    if not smoke and tuple(dataset.image_shape_hwc) != expected_shape:
        dataset.close()
        raise ValueError(
            "M2 source RGB resolution is invalid: "
            f"expected HWC={expected_shape}, got {dataset.image_shape_hwc}. "
            "Recollect and reconvert the native RoboFactory sensor data; "
            "encoder-side resizing is not a substitute."
        )
    return dataset


def _model_config(
    config: Mapping[str, Any], *, num_tasks: int, smoke: bool
) -> BlockCausalWAMConfig:
    data = _mapping(config, "data")
    model = dict(_mapping(config, "model"))
    if smoke:
        model.update(
            d_model=64,
            num_layers=2,
            num_heads=4,
            ffn_dim=128,
            text_layers=1,
            visual_feature_dim=64,
            dropout=0.0,
        )
    return BlockCausalWAMConfig(
        max_state_dim=int(data.get("max_state_dim", 72)),
        max_action_dim=int(data.get("max_action_dim", 32)),
        num_tasks=num_tasks,
        max_agents=int(data.get("max_agents", 4)),
        max_cameras=len(tuple(data.get("camera_order", ["global"]))),
        history_steps=int(data.get("state_history", 16)),
        visual_history_steps=int(data.get("visual_history_frames", 4)),
        action_horizon=int(data.get("action_horizon", 16)),
        future_visual_horizons=tuple(data.get("future_visual_horizons", [1, 4, 8, 16])),
        max_text_tokens=int(data.get("max_text_tokens", 96)),
        **model,
    )


def _build_vision(
    config: Mapping[str, Any], *, model_config: BlockCausalWAMConfig, smoke: bool
) -> tuple[nn.Module, dict[str, Any]]:
    if smoke:
        encoder = RGBStatisticsVisionEncoder(model_config.visual_feature_dim)
        return encoder, {
            "family": encoder.family,
            "output_dim": encoder.output_dim,
            "smoke_only": True,
            "frozen": True,
        }
    vision = _mapping(config, "vision")
    rectangular = "input_height" in vision or "input_width" in vision
    encoder_config = FrozenDINOv3Config(
        encoder_name=str(vision["encoder_name"]),
        model_id=str(vision["model_id"]),
        revision=str(vision["revision"]),
        config_path=(ROOT / str(vision["config_path"])).resolve(strict=True),
        weights_path=(ROOT / str(vision["weights_path"])).resolve(strict=True),
        expected_config_sha256=str(vision["expected_config_sha256"]),
        expected_weights_sha256=str(vision["expected_weights_sha256"]),
        preprocess_id=str(vision["preprocess_id"]),
        input_size=(
            None
            if rectangular
            else int(vision.get("input_size", 256))
        ),
        input_height=(
            int(vision["input_height"]) if rectangular else None
        ),
        input_width=(
            int(vision["input_width"]) if rectangular else None
        ),
        inference_batch_size=int(vision.get("inference_batch_size", 64)),
    )
    encoder = FrozenDINOv3Encoder(encoder_config)
    if encoder.output_dim != model_config.visual_feature_dim:
        raise ValueError("DINO output dimension differs from M2 visual_feature_dim")
    return encoder, {
        "family": "FrozenDINOv3Encoder",
        "encoder_name": encoder.encoder_name,
        "output_dim": encoder.output_dim,
        "artifact_sha256": encoder.artifact_sha256,
        "config_sha256": encoder.config_sha256,
        "preprocess_id": encoder_config.preprocess_id,
        "input_size": encoder_config.input_size,
        "input_height": encoder_config.image_height,
        "input_width": encoder_config.image_width,
        "frozen": True,
        "smoke_only": False,
    }


def _build_loader(
    dataset: RoboFactoryMultitaskDataset,
    *,
    sampler: CoverageTemperatureDistributedSampler,
    training: Mapping[str, Any],
    smoke: bool,
    generator: torch.Generator | None = None,
) -> DataLoader[dict[str, torch.Tensor]]:
    workers = _loader_workers(training, smoke=smoke)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": 2 if smoke else int(training.get("batch_size", 16)),
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": (bool(training.get("pin_memory", True)) and torch.cuda.is_available()),
        "drop_last": True,
        "generator": generator,
    }
    if workers:
        kwargs.update(
            prefetch_factor=int(training.get("prefetch_factor", 4)),
            persistent_workers=bool(training.get("persistent_workers", True)),
            multiprocessing_context=str(training.get("multiprocessing_context", "fork")),
            in_order=bool(training.get("in_order", False)),
        )
    return DataLoader(**kwargs)


def _loader_memory_plan(
    dataset: RoboFactoryMultitaskDataset,
    *,
    training: Mapping[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    workers = _loader_workers(training, smoke=smoke)
    prefetch_factor = (
        int(training.get("prefetch_factor", 4)) if workers else 0
    )
    if prefetch_factor < 0 or (workers and prefetch_factor == 0):
        raise ValueError("M2 prefetch_factor must be positive when workers are used")
    batch_size = 2 if smoke else int(training.get("batch_size", 16))
    height, width, channels = map(int, dataset.image_shape_hwc)
    frames_per_camera = int(dataset.visual_history) + len(
        dataset.future_horizons
    )
    raw_image_bytes_per_batch = (
        batch_size
        * frames_per_camera
        * len(dataset.camera_order)
        * height
        * width
        * channels
    )
    # Account for worker queues plus the batch being collated/pinned and the
    # batch currently consumed by DevicePrefetcher.
    resident_batches = 1 if workers == 0 else workers * prefetch_factor + 2
    estimated_resident_image_bytes = (
        raw_image_bytes_per_batch * resident_batches
    )
    available = _available_memory_bytes()
    fraction = float(
        training.get("loader_queue_max_available_fraction", 0.4)
    )
    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            "M2 loader_queue_max_available_fraction must lie in (0,1]"
        )
    budget = int(available * fraction)
    if not smoke and estimated_resident_image_bytes > budget:
        raise MemoryError(
            "M2 native-RGB loader queues exceed the host-memory safety "
            f"budget: estimate={estimated_resident_image_bytes}, "
            f"budget={budget}, workers={workers}, "
            f"prefetch_factor={prefetch_factor}, batch_size={batch_size}"
        )
    return {
        "batch_size": batch_size,
        "num_workers": workers,
        "prefetch_factor": prefetch_factor,
        "resident_batches_upper_bound": resident_batches,
        "raw_image_bytes_per_batch": raw_image_bytes_per_batch,
        "estimated_resident_image_bytes": estimated_resident_image_bytes,
        "available_memory_bytes_at_start": available,
        "max_available_fraction": fraction,
        "budget_bytes": budget,
    }


def _loader_workers(training: Mapping[str, Any], *, smoke: bool) -> int:
    if smoke:
        return 0
    value = training.get("num_workers", "auto")
    if value == "auto":
        return min(max((os.cpu_count() or 1) // 2, 1), 16)
    workers = int(value)
    if workers < 0:
        raise ValueError("M2 num_workers cannot be negative")
    return workers


def _sampling_config(
    training: Mapping[str, Any],
    *,
    dataset: RoboFactoryMultitaskDataset,
    smoke: bool,
    world_size: int,
) -> dict[str, Any]:
    raw = training.get("sampling", {})
    if not isinstance(raw, Mapping):
        raise ValueError("M2 training.sampling must be an object")
    strategy = str(
        raw.get(
            "strategy",
            "coverage_then_temperature_without_replacement_cycles",
        )
    )
    if strategy != "coverage_then_temperature_without_replacement_cycles":
        raise ValueError(f"unsupported M2 sampling strategy {strategy!r}")
    batch_size = 2 if smoke else int(training.get("batch_size", 16))
    if batch_size <= 0:
        raise ValueError("M2 batch size must be positive")
    global_batch = batch_size * int(world_size)
    requested = raw.get(
        "samples_per_epoch",
        training.get("samples_per_epoch", "auto"),
    )
    if requested == "auto":
        samples_per_epoch = math.ceil(len(dataset) / global_batch) * global_batch
    else:
        samples_per_epoch = int(requested)
        if samples_per_epoch <= 0 or samples_per_epoch % global_batch:
            raise ValueError(
                "explicit M2 samples_per_epoch must be a positive multiple "
                "of the distributed global batch"
            )
    coverage_epochs = int(raw.get("coverage_epochs", 1))
    temperature_alpha = float(raw.get("temperature_alpha", 0.5))
    return {
        "strategy": strategy,
        "samples_per_epoch": samples_per_epoch,
        "coverage_epochs": coverage_epochs,
        "temperature_alpha": temperature_alpha,
    }


def _coverage_summary(
    seen: torch.Tensor,
    *,
    dataset: RoboFactoryMultitaskDataset,
    distributed: DistributedContext,
    device: torch.device,
) -> dict[str, Any]:
    combined = seen.to(device=device, dtype=torch.uint8)
    if distributed.enabled:
        torch.distributed.all_reduce(combined, op=torch.distributed.ReduceOp.MAX)
    combined = combined.to("cpu", dtype=torch.bool)
    by_task: dict[str, dict[str, Any]] = {}
    for task_index, contract in enumerate(dataset.contracts):
        start = dataset._offsets[task_index]
        stop = dataset._offsets[task_index + 1]
        observed = int(combined[start:stop].sum())
        total = stop - start
        by_task[contract.task_id] = {
            "seen_unique_windows": observed,
            "total_windows": total,
            "coverage": observed / total,
        }
    observed = int(combined.sum())
    return {
        "seen_unique_windows": observed,
        "total_windows": len(dataset),
        "coverage": observed / len(dataset),
        "complete": observed == len(dataset),
        "by_task": by_task,
    }


def _stages(training: Mapping[str, Any], *, smoke: bool) -> list[dict[str, Any]]:
    raw = _sequence(training, "stages")
    stages: list[dict[str, Any]] = []
    for value in raw:
        if not isinstance(value, Mapping):
            raise ValueError("M2 training stages must be objects")
        stage = dict(value)
        stage["steps"] = 1 if smoke else int(stage["steps"])
        if stage["steps"] <= 0 or float(stage["learning_rate"]) <= 0.0:
            raise ValueError("M2 stage steps/LR must be positive")
        stages.append(stage)
    if not stages:
        raise ValueError("M2 requires at least one training stage")
    return stages


def _task_runtime(dataset: RoboFactoryMultitaskDataset) -> list[dict[str, Any]]:
    runtime: list[dict[str, Any]] = []
    for task_index, (contract, child) in enumerate(
        zip(dataset.contracts, dataset.datasets, strict=True)
    ):
        stats = child.manifest.load_normalization()
        codec = child.manifest.action_codec
        assert codec is not None
        runtime.append(
            {
                **contract.to_dict(),
                "task_index": task_index,
                "camera_slot_indices": [
                    dataset.camera_order.index(camera)
                    for camera in contract.camera_order
                ],
                "camera_agent_indices": [
                    int(dataset.camera_agent_index[
                        dataset.camera_order.index(camera)
                    ])
                    for camera in contract.camera_order
                ],
                "state_mean": stats.state_mean.astype(float).tolist(),
                "state_std": stats.state_std.astype(float).tolist(),
                "action_codec": codec.config.to_dict(),
                "action_mean": stats.action_mean.astype(float).tolist(),
                "action_std": stats.action_std.astype(float).tolist(),
            }
        )
    return runtime


def _action_generation(
    config: Mapping[str, Any], *, training: Mapping[str, Any]
) -> dict[str, Any]:
    raw = _mapping(config, "action_generation")
    generation = {
        "solver_steps": int(raw.get("solver_steps", 0)),
        "solver": str(raw.get("solver", "")),
        "normalized_action_clip": float(raw.get("normalized_action_clip", 0.0)),
        "execution_steps": int(raw.get("execution_steps", 0)),
        "warm_start": raw.get("warm_start"),
    }
    if (
        generation["solver_steps"] <= 0
        or generation["execution_steps"] <= 0
        or generation["normalized_action_clip"] <= 0.0
        or not math.isfinite(generation["normalized_action_clip"])
        or generation["solver"] not in {"euler", "heun"}
        or not isinstance(generation["warm_start"], bool)
    ):
        raise ValueError("M2 action_generation contract is invalid")
    warm_probability = float(training.get("warm_start_probability", 0.0))
    if not generation["warm_start"] and warm_probability != 0.0:
        raise ValueError(
            "cold-only deployment requires training.warm_start_probability=0"
        )
    return generation


def _distributed_context() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if enabled:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed M2 training requires CUDA/NCCL")
        torch.distributed.init_process_group(backend="nccl")
    return DistributedContext(rank, local_rank, world_size, enabled)


def _select_device(value: str, distributed: DistributedContext) -> torch.device:
    if distributed.enabled:
        device = torch.device("cuda", distributed.local_rank)
        torch.cuda.set_device(device)
        return device
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _configure_compute(
    config: Mapping[str, Any], threads: int, *, device: torch.device
) -> None:
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(max(1, min(4, threads)))
    training = _mapping(config, "training")
    torch.set_float32_matmul_precision(
        str(training.get("torch_float32_matmul_precision", "high"))
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(training.get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(training.get("allow_tf32", True))
        torch.backends.cudnn.benchmark = bool(training.get("cudnn_benchmark", True))


def _barrier(context: DistributedContext) -> None:
    if context.enabled:
        torch.distributed.barrier()


def _shutdown_distributed(context: DistributedContext) -> None:
    if context.enabled:
        torch.distributed.destroy_process_group()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _available_memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("cannot read MemAvailable from /proc/meminfo")


def _shared_memory_available_bytes() -> int:
    statistics = os.statvfs("/dev/shm")
    return int(statistics.f_bavail * statistics.f_frsize)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("M2 config root must be an object")
    return value


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    selected = value.get(key)
    if not isinstance(selected, Mapping):
        raise ValueError(f"M2 config {key!r} must be an object")
    return selected


def _sequence(value: Mapping[str, Any], key: str) -> list[Any]:
    selected = value.get(key)
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"M2 config {key!r} must be a non-empty list")
    return selected


def _validate_resume_progress(
    *,
    stages: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    stage_index: int,
    stage_step: int,
    global_step: int,
    epoch: int,
    samples_consumed_in_epoch: int,
    local_samples: int,
    batch_size: int,
    current_stage_last: Any,
) -> None:
    if not 0 <= stage_index < len(stages):
        raise ValueError("M2 resume stage_index is outside the training schedule")
    stage_steps = int(stages[stage_index]["steps"])
    if not 0 <= stage_step <= stage_steps:
        raise ValueError("M2 resume stage_step is outside the current stage")
    expected_global_step = (
        sum(int(stage["steps"]) for stage in stages[:stage_index])
        + stage_step
    )
    if global_step != expected_global_step:
        raise ValueError("M2 resume global_step disagrees with stage progress")
    if len(history) != stage_index:
        raise ValueError("M2 resume completed-stage history is inconsistent")
    for completed, expected in zip(
        history,
        stages[:stage_index],
        strict=True,
    ):
        if (
            completed.get("name") != expected.get("name")
            or int(completed.get("steps", -1)) != int(expected["steps"])
        ):
            raise ValueError("M2 resume stage history differs from the schedule")
    if stage_step > 0 and not isinstance(current_stage_last, Mapping):
        raise ValueError("M2 resume current stage has no final step metrics")
    if epoch < 0:
        raise ValueError("M2 resume epoch cannot be negative")
    if (
        batch_size <= 0
        or not 0 <= samples_consumed_in_epoch <= local_samples
        or samples_consumed_in_epoch % batch_size
    ):
        raise ValueError("M2 resume sample offset is not batch aligned")


def _append_progress_event(path: Path, payload: Mapping[str, Any]) -> None:
    event = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        **dict(payload),
    }
    encoded = (
        json.dumps(event, sort_keys=True, allow_nan=False, separators=(",", ":"))
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
