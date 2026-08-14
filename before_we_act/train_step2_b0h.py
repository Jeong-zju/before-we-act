"""Distributed training for the frozen Step-2 history-only/hidden-residual B0-H."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Mapping

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from before_we_act.b0h_model import B0HPolicy
from before_we_act.step2_temporal_data import (
    EFFECTIVE_BATCH,
    ExactSixTaskDistributedBatchSampler,
    SIX_TASKS,
    TeamTemporalDataset,
    load_step2_episodes,
    sha256_file,
)


FORMAL_UPDATES = 120_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_stats(path: Path) -> dict[str, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source = payload.get("stats", payload)
    return {
        key: np.asarray(source[key], dtype=np.float32)
        for key in ("q_mean", "q_std", "a_mean", "a_std")
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=B0HPolicy.VARIANTS, required=True)
    parser.add_argument(
        "--stage", choices=("f1", "discovery", "formal"), required=True
    )
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--dino-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--protocol-updates", type=int, default=FORMAL_UPDATES)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--router-lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--capability-weight", type=float, default=0.05)
    parser.add_argument("--counterfactual-every", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def distributed_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    result = value.detach().clone()
    if world_size > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= world_size
    return result


def distributed_sum(value: torch.Tensor, world_size: int) -> torch.Tensor:
    result = value.detach().clone()
    if world_size > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def model_inputs(batch: Mapping[str, object], device: torch.device) -> dict:
    keys = TeamTemporalDataset.MODEL_INPUT_FIELDS
    result = {
        key: batch[key].to(device, non_blocking=True)
        for key in keys
    }
    result["global_rgb"] = result["global_rgb"].float().div_(255)
    result["local_rgb"] = result["local_rgb"].float().div_(255)
    return result


def main() -> None:
    args = parse_args()
    if args.protocol_updates != FORMAL_UPDATES:
        raise ValueError("Step-2 sample cursor protocol is fixed at 120000 updates")
    if not 1 <= args.updates <= args.protocol_updates:
        raise ValueError("invalid Step-2 update target")
    if args.stage == "formal" and args.updates != FORMAL_UPDATES:
        raise ValueError("formal hidden-residual training requires 120000 updates")
    if args.stage == "formal" and args.variant != "hidden_residual":
        raise ValueError("only hidden-residual consumes the formal budget")
    if args.stage == "discovery" and args.updates != 5_000:
        raise ValueError("history-only Discovery is frozen at 5000 updates")
    if args.stage == "discovery" and args.variant != "history_only":
        raise ValueError("Discovery short funnel is reserved for history-only")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if EFFECTIVE_BATCH % world_size:
        raise ValueError(f"world size must divide effective batch {EFFECTIVE_BATCH}")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.set_num_threads(max(1, min(12, (os.cpu_count() or 12) // world_size)))

    contract_raw = args.contract.read_bytes()
    contract = json.loads(contract_raw)
    if contract.get("status") != "FROZEN_BEFORE_F0_F1":
        raise RuntimeError("Step-2 contract is not frozen")
    if contract["dataset"].get("episodes") != 720:
        raise RuntimeError("Step-2 contract is not the original 720-episode revision")
    cache_receipt_path = args.visual_cache / "cache_receipt.json"
    cache_receipt = json.loads(cache_receipt_path.read_text(encoding="utf-8"))
    if cache_receipt.get("status") != "PASSED" or cache_receipt.get("episodes") != 720:
        raise RuntimeError("Step-2 visual cache is incomplete")
    episodes = load_step2_episodes(args.manifests)
    stats = load_stats(args.normalization)

    resume_path = Path(args.resume).resolve() if args.resume else None
    saved = None
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        provenance = saved.get("provenance", {})
        expected = {
            "variant": args.variant,
            "seed": args.seed,
            "protocol_updates": args.protocol_updates,
            "contract_sha256": hashlib.sha256(contract_raw).hexdigest(),
            "normalization_sha256": sha256_file(args.normalization),
            "cache_receipt_sha256": sha256_file(cache_receipt_path),
        }
        for key, value in expected.items():
            if provenance.get(key) != value:
                raise ValueError(f"resume provenance mismatch at {key}")
    start_update = int(saved["update"]) if saved else 0
    sampler = ExactSixTaskDistributedBatchSampler(
        episodes,
        updates=args.protocol_updates,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
        start_update=start_update,
    )
    if saved:
        sampler.validate_cursor(saved["sample_cursor"])

    dataset = TeamTemporalDataset(
        episodes,
        stats,
        args.visual_cache,
        cache_limit=max(16, args.workers * 8),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = B0HPolicy(
        variant=args.variant,
        dino_model=args.dino_model,
    ).to(device)
    router_prefix = (
        "compatibility",
        "role_prototypes",
        "route_state",
        "route_observation",
        "route_mlp",
    )
    router, body = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (router if name.startswith(router_prefix) else body).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": body, "lr": args.lr},
            {"params": router, "lr": args.router_lr},
        ],
        weight_decay=1e-4,
    )

    def multiplier(step: int) -> float:
        warmup = min(1.0, (step + 1) / max(args.warmup, 1))
        progress = min(1.0, (step + 1) / args.protocol_updates)
        return warmup * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
    wrapped = (
        DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        if world_size > 1
        else model
    )
    underlying = wrapped.module if isinstance(wrapped, DistributedDataParallel) else wrapped
    trainable = [parameter for parameter in underlying.parameters() if parameter.requires_grad]
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "format_version": "before-we-act.step2.training_provenance/1",
        "variant": args.variant,
        "stage": args.stage,
        "seed": args.seed,
        "protocol_updates": args.protocol_updates,
        "contract": str(args.contract.resolve()),
        "contract_sha256": hashlib.sha256(contract_raw).hexdigest(),
        "normalization": str(args.normalization.resolve()),
        "normalization_sha256": sha256_file(args.normalization),
        "cache_receipt": str(cache_receipt_path.resolve()),
        "cache_receipt_sha256": sha256_file(cache_receipt_path),
        "dino_model": str(Path(args.dino_model).resolve()),
        "world_size": world_size,
        "effective_batch": EFFECTIVE_BATCH,
        "local_batch": EFFECTIVE_BATCH // world_size,
        "original_640x480_episodes": 720,
        "social_inputs": False,
        "w10_weights_loaded": False,
    }
    config = {
        **vars(args),
        "contract": str(args.contract),
        "normalization": str(args.normalization),
        "visual_cache": str(args.visual_cache),
        "output": str(args.output),
        "policy_variant": f"b0h_{args.variant}",
        "state_dim": 9,
        "action_dim": 8,
        "horizon": 100,
        "d_model": 384,
        "enc_layers": 4,
        "dec_layers": 7,
        "roles": 4,
        "role_rank": 32,
        "history_layers": 2,
        "history_steps": 16,
        "task_text_bytes": 64,
        "vision_backbone": "dual_dinov3_vitb16_frozen_full_current_grid",
        "history_visual": "cached frozen raw-DINO pooled features from original RGB",
        "policy_input": (
            "current original global/local RGB + own qpos + 16-step legal visual/qpos/"
            "executed-action history + canonical task text + masks/reset"
        ),
        "excluded_inputs": (
            "wrist/depth, episode/frame/agent ID, future, success, simulator truth, B/P/T"
        ),
        "tasks": list(SIX_TASKS),
    }
    if rank == 0:
        atomic_json(args.output / "config.json", config)
        atomic_json(args.output / "provenance.json", provenance)
        atomic_json(
            args.output / "status.json",
            {
                "status": "TRAINING",
                "stage": args.stage,
                "variant": args.variant,
                "update": start_update,
                "target_updates": args.updates,
                "world_size": world_size,
                "started_at_utc": utc_now(),
            },
        )
    if world_size > 1:
        dist.barrier()

    started = time.time()
    last: dict[str, float] = saved.get("last_metrics", {}) if saved else {}
    milestones = {20_000, 40_000, 60_000, 80_000, 100_000, 120_000, 5_000}
    if start_update >= args.updates:
        if rank == 0:
            print(json.dumps({"already_complete": start_update}), flush=True)
        if world_size > 1:
            dist.destroy_process_group()
        return

    for update, batch in enumerate(loader, start=start_update + 1):
        if update > args.updates:
            break
        # Per-update/rank seeding makes dropout and the CVAE posterior exactly
        # reproducible across an interrupted/resumed run.
        step_seed = args.seed + 10_000_019 * update + 100_003 * rank
        random.seed(step_seed)
        np.random.seed(step_seed % (2**32))
        torch.manual_seed(step_seed)
        torch.cuda.manual_seed_all(step_seed)
        inputs = model_inputs(batch, device)
        actions = batch["action"].to(device, non_blocking=True)
        mask = batch["action_mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        do_counterfactual = update % args.counterfactual_every == 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            (
                prediction,
                mu,
                logvar,
                routes,
                counterfactual,
                counterfactual_target,
                base_prediction,
                residual,
                _current_visual,
            ) = wrapped(
                **inputs,
                actions=actions,
                return_routing=True,
                counterfactual=do_counterfactual,
            )
            squared = (prediction - actions).square().mean(-1) * mask
            local_action_numerator = squared.sum()
            local_action_denominator = mask.sum().to(squared.dtype)
            global_action_denominator = distributed_sum(
                local_action_denominator, world_size
            ).clamp_min(1)
            action_loss = (
                local_action_numerator * world_size / global_action_denominator
            )
            local_kl_sum = -0.5 * (
                1 + logvar - mu.square() - logvar.exp()
            ).sum(-1).sum()
            kl_loss = local_kl_sum * world_size / EFFECTIVE_BATCH
            coupling = prediction.new_zeros(())
            if do_counterfactual:
                errors = (
                    counterfactual - counterfactual_target.unsqueeze(2)
                ).square().mean(-1)
                temperature = errors.detach().std(-1, keepdim=True).clamp_min(1e-3)
                capability_target = (-errors.detach() / temperature).softmax(-1)
                coupling = F.kl_div(
                    routes[:1].clamp_min(1e-8).log(),
                    capability_target,
                    reduction="none",
                ).sum(-1).mean()
            loss = action_loss + args.beta * kl_loss + args.capability_weight * coupling
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite Step-2 loss at update {update}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite Step-2 gradient at update {update}")
        optimizer.step()
        scheduler.step()

        global_action_numerator = distributed_sum(
            local_action_numerator, world_size
        )
        report_action = global_action_numerator / global_action_denominator
        report_kl = distributed_sum(local_kl_sum, world_size) / EFFECTIVE_BATCH
        report_coupling = distributed_mean(coupling, world_size)
        report_loss = report_action + args.beta * report_kl + args.capability_weight * report_coupling
        route_entropy = -(
            routes.clamp_min(1e-8).log() * routes
        ).sum(-1).mean()
        residual_rms = residual.float().square().mean().sqrt()
        base_rms = base_prediction.float().square().mean().sqrt()
        last = {
            "loss": float(report_loss),
            "action": float(report_action),
            "kl": float(report_kl),
            "coupling": float(report_coupling),
            "route_entropy": float(distributed_mean(route_entropy, world_size)),
            "residual_rms": float(distributed_mean(residual_rms, world_size)),
            "base_action_rms": float(distributed_mean(base_rms, world_size)),
            "grad_norm": float(distributed_mean(grad_norm, world_size)),
        }
        heartbeat = {
            "format_version": "before-we-act.step2.worker_heartbeat/1",
            "rank": rank,
            "world_size": world_size,
            "pid": os.getpid(),
            "variant": args.variant,
            "stage": args.stage,
            "update": update,
            "target_updates": args.updates,
            "updated_at_epoch": time.time(),
        }
        atomic_json(args.output / f"heartbeat_rank_{rank:02d}.json", heartbeat)
        if rank == 0 and (
            update == start_update + 1
            or update % args.log_every == 0
            or update == args.updates
        ):
            elapsed = time.time() - started
            completed = update - start_update
            progress = {
                "update": update,
                "target_updates": args.updates,
                **last,
                "learning_rate": scheduler.get_last_lr()[0],
                "updates_per_hour": completed / max(elapsed, 1e-6) * 3600,
                "eta_hours": (args.updates - update) * elapsed / completed / 3600,
                "gpu_memory_gb": round(
                    torch.cuda.max_memory_allocated(device) / 2**30, 2
                ),
                "updated_at_epoch": time.time(),
            }
            print(json.dumps(progress, sort_keys=True), flush=True)
            with (args.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(progress, sort_keys=True) + "\n")
            atomic_json(
                args.output / "status.json",
                {
                    "status": "TRAINING",
                    "stage": args.stage,
                    "variant": args.variant,
                    "world_size": world_size,
                    **progress,
                },
            )
        should_save = (
            update == args.updates
            or update % args.save_every == 0
            or update in milestones
        )
        if should_save:
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                global_sampler = ExactSixTaskDistributedBatchSampler(
                    episodes,
                    updates=args.protocol_updates,
                    seed=args.seed,
                    rank=0,
                    world_size=1,
                )
                payload = {
                    "model": underlying.state_dict(),
                    "stats": stats,
                    "config": config,
                    "provenance": provenance,
                    "update": update,
                    "last_metrics": last,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "sample_cursor": global_sampler.cursor_receipt(update),
                }
                atomic_torch_save(payload, args.output / "checkpoint_latest.pt")
                if update in milestones or update == args.updates:
                    atomic_torch_save(
                        payload, args.output / f"checkpoint_{update:06d}.pt"
                    )
                print(json.dumps({"saved_update": update}), flush=True)
            if world_size > 1:
                dist.barrier()

    if rank == 0:
        complete = {
            "status": "PASSED",
            "stage": args.stage,
            "variant": args.variant,
            "complete": True,
            "update": args.updates,
            "target_updates": args.updates,
            "world_size": world_size,
            "last_metrics": last,
            "completed_at_utc": utc_now(),
        }
        atomic_json(args.output / "status.json", complete)
        print(json.dumps(complete, sort_keys=True), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
