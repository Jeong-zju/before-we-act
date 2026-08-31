from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import FROZEN_CONFIG, POLICY_CONTRACT, atomic_json, load_config, sha256
from .dataset import DuoLatentToMDataset, TaskBalancedBatchSampler
from .policy import LocalLatentToMPolicy


class EMA:
    def __init__(self, model: torch.nn.Module, payload: dict | None = None):
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)
        if payload:
            self.model.load_state_dict(payload)
        self.step_count = 0

    @torch.no_grad()
    def step(self, model: torch.nn.Module) -> None:
        self.step_count += 1
        decay = min(0.9999, 1.0 - (1.0 + max(self.step_count - 1, 0)) ** -0.75)
        for dst, src in zip(self.model.parameters(), model.parameters(), strict=True):
            dst.mul_(decay).add_(src.detach(), alpha=1.0 - decay)
        for dst, src in zip(self.model.buffers(), model.buffers(), strict=True):
            dst.copy_(src)


def _scheduler(optimizer, total: int, warmup: int, start: int = 0):
    def rate(step: int) -> float:
        if warmup and step < warmup:
            return max((step + 1) / warmup, 1e-8)
        progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    return torch.optim.lr_scheduler.LambdaLR(optimizer, rate, last_epoch=start - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=FROZEN_CONFIG)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    opt_cfg = config["optimization"]
    loader_cfg = config["loader"]
    updates = int(opt_cfg["smoke_updates"] if args.smoke else opt_cfg["formal_updates"])
    batch_size = int(opt_cfg["smoke_batch_size"] if args.smoke else opt_cfg["formal_batch_size"])
    workers = int(loader_cfg["smoke_workers"] if args.smoke else loader_cfg["formal_workers"])
    seed = int(opt_cfg["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(opt_cfg["torch_cpu_threads"]))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("DuoBench formal training requires exactly one visible CUDA GPU")
    device = torch.device(config["runtime"]["training_device"])
    dataset = DuoLatentToMDataset(
        args.data, obs_steps=config["model"]["observation_steps"],
        horizon=config["model"]["action_horizon"],
    )
    sampler = TaskBalancedBatchSampler(dataset.task_indices, batch_size, updates, seed)
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=workers,
        pin_memory=bool(loader_cfg["pin_memory"]),
        persistent_workers=bool(loader_cfg["persistent_workers"]) and workers > 0,
        prefetch_factor=int(loader_cfg["prefetch_factor"]) if workers else None,
    )
    model = LocalLatentToMPolicy.from_config(config).to(device)
    model.set_stats(dataset.stats)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(opt_cfg["learning_rate"]),
        betas=tuple(opt_cfg["betas"]), eps=float(opt_cfg["epsilon"]),
        weight_decay=float(opt_cfg["weight_decay"]),
    )
    start = 0
    payload: dict = {}
    latest = args.output / "last.pt"
    if args.resume and latest.is_file() and not args.smoke:
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        if payload.get("policy_contract") != POLICY_CONTRACT:
            raise ValueError("resume checkpoint policy contract drift")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start = int(payload["step"])
    scheduler = _scheduler(optimizer, updates, int(opt_cfg["warmup_updates"]), start)
    if payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    ema = EMA(model, payload.get("ema_model"))
    ema.step_count = int(payload.get("ema_step", start))
    args.output.mkdir(parents=True, exist_ok=True)
    iterator = iter(loader)
    model.train()
    started = time.monotonic()
    for step_zero in range(start, updates):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader); batch = next(iterator)
        obs = {key: value.to(device, non_blocking=True) for key, value in batch.items() if key in ("image", "qpos", "task")}
        action = batch["action"].to(device, non_blocking=True)
        mask = batch["action_mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, pieces = model.loss(obs, action, mask)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at update {step_zero + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt_cfg["gradient_clip_norm"]))
        optimizer.step(); scheduler.step(); ema.step(model)
        update = step_zero + 1
        if update == 1 or update % int(config["checkpointing"]["log_interval_updates"]) == 0:
            elapsed = max(time.monotonic() - started, 1e-6)
            print(json.dumps({"event": "train", "step": update, "target_steps": updates,
                              "loss": float(loss.detach()), "diffusion": float(pieces["diffusion"]),
                              "tom": float(pieces["tom"]), "lr": scheduler.get_last_lr()[0],
                              "steps_per_sec": (update - start) / elapsed}), flush=True)
        if update % int(config["checkpointing"]["save_interval_updates"]) == 0 or update == updates:
            saved = {
                "schema": "duobench.latent-tom.checkpoint.v1", "step": update,
                "model": model.state_dict(), "ema_model": ema.model.state_dict(),
                "ema_step": ema.step_count, "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "stats": dataset.stats,
                "policy_contract": POLICY_CONTRACT, "config": config,
                "config_sha256": sha256(args.config), "all_data_no_split": True,
                "indexed_local_samples": len(dataset),
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            temporary = args.output / f"last.pt.tmp-{os.getpid()}"
            torch.save(saved, temporary); os.replace(temporary, latest)
            step_path = args.output / f"step_{update:06d}.pt"
            step_tmp = args.output / f"{step_path.name}.tmp-{os.getpid()}"
            torch.save(saved, step_tmp); os.replace(step_tmp, step_path)
    final = args.output / "final.pt"
    if not final.is_file() or final.stat().st_size != latest.stat().st_size:
        temporary = final.with_suffix(f".tmp-{os.getpid()}")
        temporary.write_bytes(latest.read_bytes()); os.replace(temporary, final)
    status_name = "smoke_status.json" if args.smoke else "status.json"
    atomic_json(args.output / status_name, {
        "schema": "duobench.latent-tom.training.v1", "status": "complete", "step": updates,
        "target_steps": updates, "checkpoint": str(final), "last_checkpoint": str(latest),
        "checkpoint_sha256": sha256(final), "episodes": 550, "indexed_local_samples": len(dataset),
        "all_episodes": True, "samples_seen": updates * batch_size,
        "effective_dataset_passes": (updates * batch_size) / len(dataset),
        "formal_config": str(args.config), "policy_contract": POLICY_CONTRACT,
    })


if __name__ == "__main__":
    main()
