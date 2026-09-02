"""Train CARE's official DINOv3 temporal B0-H reference on all Duo data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from before_we_act.temporal_history_policy import TemporalHistoryPolicy
from .data import (
    ACTION_DIM,
    ACTION_HORIZON,
    ACTION_LAG_ROWS,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    EFFECTIVE_BATCH,
    HISTORY_STEPS,
    STATE_DIM,
    TASKS,
    DuoBalancedDistributedBatchSampler,
    DuoTemporalDataset,
    load_duo_episodes,
    load_manifest,
)
from .preprocessing import DINO_NORMALIZATION_ID, IMAGE_PREPROCESS_ID


FORMAL_UPDATES = 120_000
DEFAULT_SEED = 20260830


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _distributed_sum(value: torch.Tensor, world: int) -> torch.Tensor:
    result = value.detach().clone()
    if world > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--dino-model", default="/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "pilot", "formal"), required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--protocol-updates", type=int, default=FORMAL_UPDATES)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--router-lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--capability-weight", type=float, default=0.05)
    parser.add_argument("--counterfactual-every", type=int, default=4)
    parser.add_argument(
        "--action-loss-decay", type=float, default=0.0,
        help="Exponential near-term weighting in steps; 0 keeps uniform loss",
    )
    parser.add_argument(
        "--gripper-loss-weight", type=float, default=0.20,
        help="Class-balanced BCE auxiliary weight for binary gripper output",
    )
    parser.add_argument(
        "--gripper-logit-scale", type=float, default=4.0,
        help="Scale from normalized gripper output to BCE logits",
    )
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.protocol_updates != FORMAL_UPDATES:
        raise ValueError("Duo B0-H protocol is fixed at 120000 updates")
    if not 1 <= args.updates <= args.protocol_updates:
        raise ValueError("invalid Duo B0-H update target")
    if args.stage == "formal" and args.updates != FORMAL_UPDATES:
        raise ValueError("formal Duo B0-H training requires 120000 updates")
    if args.stage == "smoke" and args.updates > 10:
        raise ValueError("smoke training is capped at 10 updates")
    if args.stage == "pilot" and args.updates > 10_000:
        raise ValueError("CARE-v2 pilot training is capped at 10000 updates")
    if args.image_height % 16 or args.image_width % 16:
        raise ValueError("DINO dimensions must be divisible by 16")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    if args.action_loss_decay < 0:
        raise ValueError("action-loss-decay cannot be negative")
    if args.gripper_loss_weight < 0 or args.gripper_logit_scale <= 0:
        raise ValueError("invalid gripper auxiliary-loss settings")

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if EFFECTIVE_BATCH % world:
        raise ValueError(f"world size must divide global batch {EFFECTIVE_BATCH}")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(max(1, min(12, (os.cpu_count() or 12) // world)))

    manifest = load_manifest(args.prepared_data)
    episodes = load_duo_episodes(args.prepared_data, require_formal=True)
    receipt_path = args.visual_cache / "cache_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    accepted_cache_status = ("PASSED", "SMOKE") if args.stage == "smoke" else ("PASSED",)
    if (
        receipt.get("schema") != "before-we-act.duobench.dino-cache/1"
        or receipt.get("status") not in accepted_cache_status
        or int(receipt.get("episodes", 0)) != 550
        or int(receipt.get("image_height", 0)) != args.image_height
        or int(receipt.get("image_width", 0)) != args.image_width
        or receipt.get("image_preprocess_id") != IMAGE_PREPROCESS_ID
        or (args.stage == "formal" and receipt.get("dino_normalization_id") != DINO_NORMALIZATION_ID)
        or (args.stage == "formal" and receipt.get("strict_dino_contract") is not True)
    ):
        raise RuntimeError("formal frozen-DINO cache receipt is invalid")

    saved = None
    if args.resume:
        if not args.resume.is_file():
            raise FileNotFoundError(args.resume)
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        expected_resume = {
            "seed": args.seed,
            "protocol_updates": args.protocol_updates,
            "image_height": args.image_height,
            "image_width": args.image_width,
            "action_encoding": "absolute_joint7_binary_gripper1",
        }
        config = saved.get("config", {})
        for key, value in expected_resume.items():
            if config.get(key) != value:
                raise ValueError(f"resume checkpoint differs at {key}")
    start = int(saved["update"]) if saved else 0
    sampler = DuoBalancedDistributedBatchSampler(
        episodes,
        updates=args.protocol_updates,
        seed=args.seed,
        rank=rank,
        world_size=world,
        start_update=start,
    )
    if saved:
        sampler.validate_cursor(saved["sample_cursor"])
    dataset = DuoTemporalDataset(
        args.prepared_data,
        episodes,
        args.visual_cache,
        image_height=args.image_height,
        image_width=args.image_width,
        cache_limit=max(16, args.workers * 4),
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
    model = TemporalHistoryPolicy(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        variant="hidden_residual",
        horizon=ACTION_HORIZON,
        dino_model=args.dino_model,
        image_height=args.image_height,
        image_width=args.image_width,
        strict_dino_contract=True,
    ).to(device)
    router_prefix = (
        "compatibility",
        "role_prototypes",
        "route_state",
        "route_observation",
        "route_mlp",
    )
    body: list[torch.nn.Parameter] = []
    router: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (router if name.startswith(router_prefix) else body).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": body, "lr": args.lr}, {"params": router, "lr": args.router_lr}],
        weight_decay=1e-4,
    )

    def schedule(step: int) -> float:
        warmup = min(1.0, (step + 1) / max(1, args.warmup))
        progress = min(1.0, (step + 1) / args.protocol_updates)
        return warmup * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    if saved:
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
    wrapped = (
        DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        if world > 1
        else model
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    norm = manifest["normalization"]
    config = {
        "format_version": "before-we-act.duobench.dino-b0h-config/1",
        # Keep the formal reference's concrete policy family explicit.  CARE
        # is the downstream method/benchmark family, whereas this checkpoint
        # is the project-owned temporal-history Transformer itself.  A generic
        # ``CARE`` tag made it too easy for an ACT/ConvNeXt artifact to pass a
        # superficial metadata check.
        "policy_family": "TemporalHistoryPolicy",
        "method_family": "CARE",
        "architecture": "TemporalHistoryPolicy_hidden_residual",
        "stage": args.stage,
        "seed": args.seed,
        "protocol_updates": args.protocol_updates,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "horizon": ACTION_HORIZON,
        "history_steps": HISTORY_STEPS,
        "variant": "hidden_residual",
        "d_model": 384,
        "enc_layers": 4,
        "dec_layers": 7,
        "roles": 4,
        "role_rank": 32,
        "history_layers": 2,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "dino_model": args.dino_model,
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "strict_dino_contract": True,
        "vision": "dinov3_vitb16_frozen",
        "vision_backbone": "dinov3_vitb16_frozen",
        "action_encoding": "absolute_joint7_binary_gripper1",
        "action_lag_rows": ACTION_LAG_ROWS,
        "action_loss_decay": float(args.action_loss_decay),
        "gripper_loss_weight": float(args.gripper_loss_weight),
        "gripper_logit_scale": float(args.gripper_logit_scale),
        "policy_contract": "shared_weights_strictly_decentralized_head_rgb_own_wrist_rgb_local_qpos8_to_local_absolute_action8",
        "tasks": list(TASKS),
        "effective_batch": EFFECTIVE_BATCH,
        "sampling": {
            "base_samples_per_task_per_update": 4,
            "rotating_extra_samples_per_update": 4,
            "balance_cycle_updates": 11,
            "matched_compute_effective_batch": 48,
            "four_gpu_local_batch": 12,
        },
        "all_550_demonstrations": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        _atomic_json(args.output / "config.json", config)
        _atomic_json(
            args.output / "status.json",
            {
                "status": "TRAINING",
                "stage": args.stage,
                "update": start,
                "target_updates": args.updates,
                "started_at_utc": _now(),
            },
        )
    started = time.time()
    for update, batch in enumerate(loader, start=start + 1):
        if update > args.updates:
            break
        step_seed = args.seed + 10_000_019 * update + 100_003 * rank
        random.seed(step_seed)
        np.random.seed(step_seed % (2**32))
        torch.manual_seed(step_seed)
        torch.cuda.manual_seed_all(step_seed)
        inputs = {
            key: batch[key].to(device, non_blocking=True)
            for key in DuoTemporalDataset.MODEL_INPUT_FIELDS
        }
        inputs["global_rgb"] = inputs["global_rgb"].float().div_(255)
        inputs["local_rgb"] = inputs["local_rgb"].float().div_(255)
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
                _base,
                _residual,
                _visual,
            ) = wrapped(
                **inputs,
                actions=actions,
                return_routing=True,
                counterfactual=do_counterfactual,
            )
            per_step = (prediction - actions).square().mean(-1)
            if args.action_loss_decay > 0:
                offsets = torch.arange(ACTION_HORIZON, device=device, dtype=per_step.dtype)
                weights = torch.exp(-offsets / float(args.action_loss_decay))[None, :]
            else:
                weights = torch.ones(1, ACTION_HORIZON, device=device, dtype=per_step.dtype)
            weighted_mask = mask.to(per_step.dtype) * weights
            numerator = (per_step * weighted_mask).sum()
            denominator = _distributed_sum(weighted_mask.sum(), world).clamp_min(1)
            action_loss = numerator * world / denominator
            # Gripper targets are binary but live in normalized action8.  A
            # plain MSE learns the majority-class mean and can decode to an
            # always-open command early in training.  This auxiliary term is
            # class-balanced per batch and leaves the action head/contract
            # unchanged.
            gripper_threshold = (0.5 - float(norm["action_mean"][7])) / float(norm["action_std"][7])
            gripper_target = (
                actions[..., 7] >= gripper_threshold
            ).to(per_step.dtype)
            gripper_logits = (prediction[..., 7] - gripper_threshold) * args.gripper_logit_scale
            valid_gripper = weighted_mask
            positive = _distributed_sum((gripper_target * valid_gripper).sum(), world)
            negative = _distributed_sum(((1.0 - gripper_target) * valid_gripper).sum(), world)
            total = (positive + negative).clamp_min(1.0)
            # Inverse-frequency weights, normalized to unit mean over valid
            # elements.  The distributed reductions make this deterministic
            # under the existing four-rank protocol.
            pos_weight = 0.5 * total / positive.clamp_min(1.0)
            neg_weight = 0.5 * total / negative.clamp_min(1.0)
            sample_weight = torch.where(gripper_target > 0.5, pos_weight, neg_weight)
            gripper_bce_sum = F.binary_cross_entropy_with_logits(
                gripper_logits, gripper_target, reduction="none"
            )
            gripper_loss = (
                (gripper_bce_sum * sample_weight * valid_gripper).sum() * world
                / _distributed_sum(valid_gripper.sum(), world).clamp_min(1.0)
            )
            local_kl = -0.5 * (
                1 + logvar - mu.square() - logvar.exp()
            ).sum(-1).sum()
            kl_loss = local_kl * world / EFFECTIVE_BATCH
            coupling = prediction.new_zeros(())
            if do_counterfactual:
                errors = (counterfactual - counterfactual_target.unsqueeze(2)).square().mean(-1)
                target = (
                    -errors.detach()
                    / errors.detach().std(-1, keepdim=True).clamp_min(1e-3)
                ).softmax(-1)
                coupling = F.kl_div(
                    routes[:1].clamp_min(1e-8).log(), target, reduction="batchmean"
                )
            loss = (
                action_loss
                + args.gripper_loss_weight * gripper_loss
                + args.beta * kl_loss
                + args.capability_weight * coupling
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite Duo B0-H loss at {update}")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(gradient):
            raise FloatingPointError(f"non-finite Duo B0-H gradient at {update}")
        optimizer.step()
        scheduler.step()
        metrics = {
            "status": "TRAINING",
            "stage": args.stage,
            "update": update,
            "target_updates": args.updates,
            "loss": float(loss.detach()),
            "action_loss": float(action_loss.detach()),
            "gripper_loss": float(gripper_loss.detach()),
            "kl_loss": float(kl_loss.detach()),
            "coupling_loss": float(coupling.detach()),
            "gradient_norm": float(gradient.detach()),
            "body_lr": scheduler.get_last_lr()[0],
            "router_lr": scheduler.get_last_lr()[1],
            "elapsed_seconds": time.time() - started,
            "strictly_decentralized": True,
        }
        if rank == 0 and (
            update == start + 1 or update % args.log_every == 0 or update == args.updates
        ):
            print(json.dumps(metrics), flush=True)
            with (args.output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            _atomic_json(args.output / "status.json", metrics)
        if rank == 0 and (update % args.save_every == 0 or update == args.updates):
            payload = {
                "format": "before-we-act.duobench.dino-b0h/1",
                "update": update,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "sample_cursor": sampler.cursor_receipt(update),
                "stats": {
                    "q_mean": norm["qpos_mean"],
                    "q_std": norm["qpos_std"],
                    "a_mean": norm["action_mean"],
                    "a_std": norm["action_std"],
                },
                "config": config,
                "last_metrics": metrics,
            }
            _atomic_save(payload, args.output / f"checkpoint_{update:06d}.pt")
            _atomic_save(payload, args.output / "checkpoint_latest.pt")
            if update == args.updates:
                _atomic_save(payload, args.output / "final.pt")
            _atomic_json(
                args.output / "checkpoint_receipt.json",
                {
                    "schema": "before-we-act.duobench.dino-b0h-checkpoint/1",
                    "status": "PASSED",
                    "stage": args.stage,
                    "update": update,
                    "policy_family": "TemporalHistoryPolicy",
                    "method_family": "CARE",
                    "architecture": "TemporalHistoryPolicy_hidden_residual",
                    "vision_backbone": "dinov3_vitb16_frozen",
                    "image_preprocess_id": IMAGE_PREPROCESS_ID,
                    "dino_normalization_id": DINO_NORMALIZATION_ID,
                    "strict_dino_contract": True,
                    "action_encoding": "absolute_joint7_binary_gripper1",
                    "state_dim": 8,
                    "action_dim": 8,
                    "history_steps": 16,
                    "action_horizon": 100,
                    "effective_batch": 48,
                    "four_gpu_local_batch": 12,
                    "task_balance": "4/task/update + 4 rotating extras; exact over 11 updates",
                    "strictly_decentralized": True,
                    "checkpoint": str((args.output / "checkpoint_latest.pt").resolve()),
                    "created_at_utc": _now(),
                },
            )
    if rank == 0:
        _atomic_json(
            args.output / "status.json",
            {
                "status": "PASSED",
                "stage": args.stage,
                "update": args.updates,
                "target_updates": args.updates,
                "completed_at_utc": _now(),
            },
        )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
