from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from diffusion_policy.model.diffusion.ema_model import EMAModel

from .common import (
    POLICY_CONTRACT,
    TEMPORAL_CONTRACT,
    atomic_json,
    policy_contract,
    sha256_file,
)
from .dataset import DuoDPDataset, TaskEpisodeBatchSampler, compute_corpus_stats
from .modeling import build_policy


def atomic_torch(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def make_scheduler(optimizer, total: int, warmup: int, start: int = 0):
    def multiplier(step):
        if step < warmup:
            return max((step + 1) / max(warmup, 1), 1e-8)
        progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier, last_epoch=start - 1)


def migrate_policy_state(model, source: dict[str, torch.Tensor]) -> None:
    """Load a baseline DP into a task-conditioned DP without changing its output.

    The only learned tensors that grow are the residual-block conditioning
    projections.  Their first 128 columns are the diffusion-time embedding;
    the remaining columns are three flattened [ResNet512, local-state]
    observations.  Copy every old feature into its exact semantic location
    and leave the 11 new task columns at the target initialization (zeroed
    below), making the task-conditioned model initially baseline-equivalent.
    """

    target = model.state_dict()
    migrated = {}
    for key, value in target.items():
        old = source.get(key)
        if old is not None and old.shape == value.shape:
            migrated[key] = old
            continue
        if (
            old is not None
            and key.endswith("cond_encoder.1.weight")
            and old.ndim == value.ndim == 2
            and value.shape[0] == old.shape[0]
            and value.shape[1] > old.shape[1]
        ):
            time_dim = 128
            old_obs_dim = (old.shape[1] - time_dim) // 3
            new_obs_dim = (value.shape[1] - time_dim) // 3
            if time_dim + 3 * old_obs_dim != old.shape[1] or new_obs_dim <= old_obs_dim:
                raise RuntimeError(f"cannot migrate conditioning tensor {key}: {old.shape}/{value.shape}")
            expanded = torch.zeros_like(value)
            expanded[:, :time_dim] = old[:, :time_dim]
            for step in range(3):
                old_start = time_dim + step * old_obs_dim
                new_start = time_dim + step * new_obs_dim
                expanded[:, new_start : new_start + old_obs_dim] = old[
                    :, old_start : old_start + old_obs_dim
                ]
            migrated[key] = expanded
            continue
        # The task-conditioned agent normalizer is intentionally rebuilt.
        if "normalizer.params_dict.agent_pos" in key:
            migrated[key] = value
            continue
        raise RuntimeError(
            f"unsupported initialization tensor mismatch {key}: "
            f"{None if old is None else tuple(old.shape)} -> {tuple(value.shape)}"
        )
    model.load_state_dict(migrated, strict=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--transition-fraction", type=float, default=0.0)
    parser.add_argument("--gripper-loss-weight", type=float, default=1.0)
    parser.add_argument("--task-conditioning", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=500)
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.save_every <= 0:
        raise ValueError("steps, batch size, and save interval must be positive")
    if not 0.0 <= args.transition_fraction <= 1.0:
        raise ValueError("transition-fraction must be in [0, 1]")
    if args.gripper_loss_weight < 1.0:
        raise ValueError("gripper-loss-weight must be >= 1")
    if args.learning_rate <= 0 or args.warmup < 0:
        raise ValueError("learning-rate must be positive and warmup nonnegative")
    if args.resume and args.init_checkpoint:
        raise ValueError("resume and init-checkpoint are mutually exclusive")
    target = min(args.steps, 10) if args.smoke else args.steps
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    args.output.mkdir(parents=True, exist_ok=True)

    stats = compute_corpus_stats(args.data)
    atomic_json(args.output / "normalization.json", stats)
    dataset = DuoDPDataset(args.data)
    device = torch.device("cuda:0")
    model = build_policy(
        stats,
        device,
        task_conditioning=args.task_conditioning,
        gripper_loss_weight=args.gripper_loss_weight,
    )
    contract = policy_contract(args.task_conditioning)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6
    )
    start = 0
    saved = {}
    initialized = {}
    latest = args.output / "latest.pt"
    if args.resume and latest.is_file() and not args.smoke:
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved.get("policy_contract") != contract:
            raise RuntimeError("resume checkpoint policy contract mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start = int(saved["step"])
    elif args.init_checkpoint:
        initialized = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if initialized.get("policy_contract") != POLICY_CONTRACT:
            raise RuntimeError("init checkpoint must use the unconditioned baseline contract")
        migrate_policy_state(model, initialized["model"])
    # Diffusion Policy's custom normalizer loader replaces its ParameterDict
    # with tensors cloned from the CPU checkpoint.  Move it back explicitly
    # after every resume/init load, matching load_policy().
    model = model.to(device)
    model.normalizer = model.normalizer.to(device)
    if start >= target:
        return
    scheduler = make_scheduler(optimizer, args.steps, args.warmup, start)
    if saved.get("scheduler"):
        scheduler.load_state_dict(saved["scheduler"])
    ema_model = copy.deepcopy(model).to(device)
    ema = EMAModel(
        ema_model,
        update_after_step=0,
        inv_gamma=1.0,
        power=0.75,
        min_value=0.0,
        max_value=0.9999,
    )
    if saved.get("ema_model"):
        ema_model.load_state_dict(saved["ema_model"])
        ema.optimization_step = int(saved.get("ema_step", start))
    elif initialized:
        migrate_policy_state(ema_model, initialized["ema_model"])
        # Reset the averaging schedule so newly introduced task columns can
        # enter EMA quickly during the finite probe.
        ema.optimization_step = 0
    ema_model = ema_model.to(device)
    ema_model.normalizer = ema_model.normalizer.to(device)
    sampler = TaskEpisodeBatchSampler(
        dataset,
        args.batch_size,
        target - start,
        args.seed + start,
        transition_fraction=args.transition_fraction,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers else None,
    )
    model.train()
    started = time.monotonic()
    progress_path = args.output / "progress.jsonl"
    for local_step, batch in enumerate(loader, start=1):
        step = start + local_step
        prepared = {
            "obs": {
                "head_wrist": batch["head_wrist"].to(device, non_blocking=True).float().div_(255.0),
                "agent_pos": batch["agent_pos"].to(device, non_blocking=True),
            },
            "action": batch["action"].to(device, non_blocking=True),
        }
        optimizer.zero_grad(set_to_none=True)
        if args.task_conditioning:
            prepared["obs"]["agent_pos"] = torch.cat(
                (prepared["obs"]["agent_pos"], batch["task_id"].to(device, non_blocking=True)), dim=-1
            )
        loss = model.compute_loss(prepared)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite DP loss at update {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        scheduler.step()
        ema.step(model)
        with torch.no_grad():
            for ema_buffer, online_buffer in zip(ema_model.buffers(), model.buffers(), strict=True):
                ema_buffer.copy_(online_buffer)
        if step == 1 or step % 20 == 0:
            row = {
                "step": step,
                "target_steps": target,
                "loss": float(loss),
                "grad_norm": grad_norm,
                "lr": scheduler.get_last_lr()[0],
                "updates_per_second": local_step / max(time.monotonic() - started, 1e-6),
                "gpu_max_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            }
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % args.save_every == 0 or step == target:
            common = {
                "schema": "duobench.dp.checkpoint.v1",
                "step": step,
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "ema_step": ema.optimization_step,
                "stats": stats,
                "policy_contract": contract,
                "temporal_contract": TEMPORAL_CONTRACT,
                "config": {
                    "obs_steps": 3,
                    "horizon": 8,
                    "executable_steps": 6,
                    "action_lag_rows": 1,
                    "diffusion_train_steps": 100,
                    "batch_size": args.batch_size,
                    "all_550_episodes_no_split": True,
                    "task_sampling": "uniform_task_then_uniform_episode_arm_then_timestep",
                    "image": "uint8_head_plus_local_wrist_horizontal_then_div255_imagenet_norm",
                    "state": "own_absolute_joint7_binary_gripper1",
                    "action": "own_controller_equivalent_absolute_joint7_binary_gripper1",
                    "seed": args.seed,
                    "transition_fraction": args.transition_fraction,
                    "gripper_loss_weight": args.gripper_loss_weight,
                    "task_conditioning": args.task_conditioning,
                    "learning_rate": args.learning_rate,
                    "warmup": args.warmup,
                    "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
                    "init_checkpoint_sha256": (
                        sha256_file(args.init_checkpoint) if args.init_checkpoint else None
                    ),
                },
            }
            resumable = {
                **common,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            atomic_torch(latest, resumable)
            if step == target:
                atomic_torch(args.output / "final.pt", common)
    status = {
        "schema": "duobench.dp.training-status.v1",
        "status": "complete",
        "step": target,
        "formal_target_steps": args.steps,
        "checkpoint": str(args.output / "final.pt"),
        "checkpoint_sha256": sha256_file(args.output / "final.pt"),
        "episodes": stats["episodes"],
        "indexed_local_samples": stats["indexed_local_samples"],
        "samples_drawn": target * args.batch_size,
        "equivalent_indexed_sample_traversals": target * args.batch_size / stats["indexed_local_samples"],
        "all_data": True,
        "transition_fraction": args.transition_fraction,
        "gripper_loss_weight": args.gripper_loss_weight,
        "task_conditioning": args.task_conditioning,
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "init_checkpoint_sha256": sha256_file(args.init_checkpoint) if args.init_checkpoint else None,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.output / "status.json", status)


if __name__ == "__main__":
    main()
