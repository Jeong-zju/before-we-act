#!/usr/bin/env python3
"""Train the S1-R1 per-agent cold-start Rectified Flow candidate."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.static_rgb_act import StaticRGBMoEACTConfig  # noqa: E402
from models.wam_multimodal import AgentFactorizedFlowWAM  # noqa: E402
from scripts.train_static_rgb_act_moe import (  # noqa: E402
    _TaskBalancedBatchSampler,
    _append_jsonl,
    _atomic_torch_save,
    _capture_rng,
    _dataset,
    _frozen_vision_tokens,
    _git_commit,
    _load_yaml,
    _local_batch,
    _mapping,
    _restore_rng,
    _seed_everything,
    _sha256,
    _task_runtime,
    _vision,
)
from train.agent_factorized_flow_training import (  # noqa: E402
    make_flow_matching_batch,
    uniform_masked_flow_mse,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/wam_flow/s1_r1_f1_flow_cold.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--progress-log", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve(strict=True)
    raw = _load_yaml(config_path)
    training = _mapping(raw, "training")
    generation = _mapping(raw, "generation")
    if (
        generation.get("source_distribution") != "standard_normal"
        or generation.get("solver") != "euler"
        or int(generation.get("solver_steps", 0)) != 4
    ):
        raise ValueError("S1-R1 F1 training requires cold Gaussian 4-step Euler")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal S1-R1 Flow training requires one CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "expose exactly one GPU (for example CUDA_VISIBLE_DEVICES=1)"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("S1-R1 Flow training requires native BF16")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    seed = int(training.get("seed", 101))
    _seed_everything(seed)

    dataset = _dataset(raw)
    batch_size = int(args.batch_size or training.get("batch_size", 4))
    updates = int(args.updates or training.get("updates", 80000))
    if batch_size <= 0 or updates <= 0:
        raise ValueError("batch size and updates must be positive")
    model_config = StaticRGBMoEACTConfig.from_dict(_mapping(raw, "model"))
    if model_config.horizon != dataset.action_horizon:
        raise ValueError("model and dataset action horizons differ")
    vision = _vision(raw).to(device).eval()
    model = AgentFactorizedFlowWAM(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 2e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, updates)

    checkpoint = _mapping(raw, "checkpoint")
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (ROOT / str(checkpoint["output"])).resolve()
    )
    resume = (
        args.resume.expanduser().resolve()
        if args.resume is not None
        else (ROOT / str(checkpoint["resume"])).resolve()
    )
    progress_log = (
        args.progress_log.expanduser().resolve()
        if args.progress_log is not None
        else (
            (ROOT / str(checkpoint["progress_log"])).resolve()
            if checkpoint.get("progress_log") is not None
            else None
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resume.parent.mkdir(parents=True, exist_ok=True)
    if progress_log is not None:
        progress_log.parent.mkdir(parents=True, exist_ok=True)
    start = 0
    if not args.no_resume and resume.is_file():
        saved = torch.load(resume, map_location=device, weights_only=False)
        if saved.get("format_version") != (
            "wam.robofactory.agent_factorized_flow.resume/1"
        ):
            raise ValueError("resume file is not S1-R1 AgentFactorizedFlow")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start = int(saved["update"])
        _restore_rng(_mapping(saved, "rng"))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite completed checkpoint {output}")

    save_interval = int(training.get("save_interval", 1000))
    router_weight = float(training.get("router_aux_weight", 1e-2))
    workers = int(training.get("num_workers", 4))
    loader = DataLoader(
        dataset,
        batch_sampler=_TaskBalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            first_update=start + 1,
            final_update=updates,
            seed=seed,
        ),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed + 10_000_000),
    )
    iterator = iter(loader)
    model.train()
    for update in range(start + 1, updates + 1):
        batch = _local_batch(next(iterator))
        images = batch["images"].to(device, non_blocking=True)
        state = batch["state"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        vision_tokens = _frozen_vision_tokens(vision, images)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sampled = make_flow_matching_batch(actions)
            prediction, router_aux = model(
                vision_tokens,
                state,
                sampled.action_inputs,
                sampled.flow_time,
            )
            flow_matching = uniform_masked_flow_mse(
                prediction,
                sampled.target_velocity,
                valid,
            )
            loss = flow_matching + router_weight * (router_aux - 1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(training.get("gradient_clip_norm", 1.0)),
        )
        optimizer.step()
        scheduler.step()
        if update % 100 == 0 or update == 1:
            progress = {
                "update": update,
                "updates": updates,
                "loss": float(loss.detach()),
                "flow_matching": float(flow_matching.detach()),
                "router_aux": float(router_aux.detach()),
            }
            print(progress, flush=True)
            if progress_log is not None:
                _append_jsonl(progress_log, progress)
        if update % save_interval == 0 and update < updates:
            _atomic_torch_save(
                {
                    "format_version": (
                        "wam.robofactory.agent_factorized_flow.resume/1"
                    ),
                    "update": update,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng": _capture_rng(),
                },
                resume,
            )

    payload = {
        "format_version": "wam.robofactory.agent_factorized_flow.checkpoint/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "update": updates,
        "method": {
            "round_id": "s1-r1",
            "candidate_id": "F1",
            "action_generator": "rectified_flow_cold",
            "future_path": False,
            "active_agent_loss_weighting": False,
        },
        "model_config": model_config.to_dict(),
        "model": model.state_dict(),
        "generation": dict(generation),
        "task_runtime": _task_runtime(dataset),
        "vision": dict(_mapping(raw, "vision")),
        "training": dict(training),
        "data": {
            "manifests": [
                {
                    "path": str(contract.manifest_path),
                    "sha256": contract.manifest_sha256,
                    "task_id": contract.task_id,
                }
                for contract in dataset.contracts
            ],
            "camera_protocol": (
                "existing world-fixed per-agent RGB; no wrist camera; no depth"
            ),
        },
        "source": {
            "git_commit": _git_commit(),
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
        },
    }
    _atomic_torch_save(payload, output)
    if resume.exists():
        resume.unlink()
    print({"checkpoint": str(output), "sha256": _sha256(output)}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
