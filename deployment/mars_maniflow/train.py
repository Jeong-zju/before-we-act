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
from torch.utils.data import DataLoader, Sampler

from .dataset import MarsManiFlowDataset, TaskBalancedBatchSampler
from .modeling import build_policy, model_config
from maniflow.model.diffusion.ema_model import EMAModel


class LocalityBatchSampler(Sampler):
    """Full-coverage shuffled local runs for efficient HDF5 slicing."""
    def __init__(self, size: int, batch_size: int, seed: int, locality: int = 8):
        self.size, self.batch_size, self.seed, self.locality = int(size), int(batch_size), int(seed), int(locality)
        self.epoch = 0

    def __len__(self): return math.ceil(self.size / self.batch_size)

    def __iter__(self):
        runs = [list(range(start, min(start + self.locality, self.size))) for start in range(0, self.size, self.locality)]
        random.Random(self.seed + self.epoch).shuffle(runs); self.epoch += 1
        batch = []
        for run in runs:
            batch.extend(run)
            while len(batch) >= self.batch_size:
                yield batch[:self.batch_size]; batch = batch[self.batch_size:]
        if batch: yield batch


def atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent); os.close(fd)
    try:
        torch.save(payload, tmp); os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def scheduler(optimizer, steps: int, warmup: int, start: int):
    def rate(step):
        if step < warmup: return max((step + 1) / max(warmup, 1), 1e-8)
        progress = min(max((step - warmup) / max(steps - warmup, 1), 0), 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups: group.setdefault("initial_lr", group["lr"])
    return torch.optim.lr_scheduler.LambdaLR(optimizer, rate, last_epoch=start - 1)


def checkpoint_payload(model, ema, optimizer, lr_scheduler, step, args, dataset):
    return {
        "schema": "bwa.maniflow.local_checkpoint.v1", "contract": model_config()["policy_contract"],
        "config": model_config(), "model": model.state_dict(), "ema_model": ema.averaged_model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": lr_scheduler.state_dict(),
        "ema_step": ema.optimization_step, "step": step, "target_steps": args.steps,
        "dataset_size": len(dataset), "all_episodes": True, "episodes": 600,
        "local_streams": 1650, "stats": dataset.stats, "seed": args.seed,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default=os.getenv("MARS_MANIFLOW_DATA_ROOT", "/workspace/datasets/mars_control"))
    p.add_argument("--stats", default=os.getenv("MARS_MANIFLOW_STATS", "/workspace/runs/mars_maniflow/normalization.json"))
    p.add_argument("--output", default=os.getenv("BWA_MANIFLOW_OUTPUT", "/workspace/bwa_maniflow_runs/formal"))
    p.add_argument("--steps", type=int, default=int(os.getenv("MANIFLOW_STEPS", "60000")))
    p.add_argument("--batch-size", type=int, default=int(os.getenv("MANIFLOW_BATCH", "128")))
    p.add_argument("--grad-accum", type=int, default=int(os.getenv("MANIFLOW_GRAD_ACCUM", "1")))
    p.add_argument("--workers", type=int, default=int(os.getenv("MANIFLOW_WORKERS", "16")))
    p.add_argument("--locality", type=int, default=int(os.getenv("MANIFLOW_LOCALITY", "8")))
    p.add_argument("--save-every", type=int, default=int(os.getenv("MANIFLOW_SAVE_EVERY", "5000")))
    p.add_argument("--log-every", type=int, default=20); p.add_argument("--seed", type=int, default=20260822)
    p.add_argument("--resume", action="store_true"); p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.batch_size % 4:
        raise ValueError("ManiFlow 3:1 flow/consistency split requires batch size divisible by four")
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    dataset = MarsManiFlowDataset(args.dataset_root, args.stats)
    sampler = TaskBalancedBatchSampler(dataset.task_indices, args.batch_size, args.steps, args.seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0, prefetch_factor=2 if args.workers else None)
    model = build_policy(device); ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    ema = EMAModel(ema_model, update_after_step=0, inv_gamma=1.0, power=0.75, min_value=0.0, max_value=0.9999)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-3)
    start = 0; latest = output / "last.pt"
    if args.resume and latest.is_file():
        old = torch.load(latest, map_location="cpu", weights_only=False)
        if old.get("config") != model_config(): raise RuntimeError("resume config drift")
        model.load_state_dict(old["model"]); ema.averaged_model.load_state_dict(old["ema_model"])
        optimizer.load_state_dict(old["optimizer"]); start = int(old["step"]); ema.optimization_step = int(old["ema_step"])
    lr = scheduler(optimizer, args.steps, min(500, max(args.steps // 20, 1)), start)
    if start and "scheduler" in old: lr.load_state_dict(old["scheduler"])
    model.train(); optimizer.zero_grad(set_to_none=True)
    log = (output / "train.jsonl").open("a", buffering=1); iterator = iter(loader)
    started = time.monotonic(); interval = started
    for step in range(start + 1, args.steps + 1):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        batch.pop("task", None)
        batch = {"obs": {k: v.to(device, non_blocking=True) for k, v in batch["obs"].items()},
                 "action": batch["action"].to(device, non_blocking=True)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = model.compute_loss(batch, ema.averaged_model)
            scaled = loss / args.grad_accum
        scaled.backward()
        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); lr.step(); ema.step(model)
        if step == 1 or step % args.log_every == 0:
            now = time.monotonic(); elapsed = now - started
            row = {"step": step, "target_steps": args.steps, "loss": float(loss.detach()),
                   "loss_flow": metrics["loss_flow"], "loss_ct": metrics["loss_ct"],
                   "lr": lr.get_last_lr()[0], "steps_per_second": (step - start) / max(elapsed, 1e-6),
                   "interval_seconds": now - interval, "gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "dataset_size": len(dataset), "samples_seen": step * args.batch_size,
                   "dataset_passes": step * args.batch_size / len(dataset), "at": datetime.now(timezone.utc).isoformat()}
            print(json.dumps(row), flush=True); log.write(json.dumps(row) + "\n"); interval = now
        if step % args.save_every == 0 or step == args.steps:
            atomic_torch(latest, checkpoint_payload(model, ema, optimizer, lr, step, args, dataset))
    log.close()
    status = {"schema": "mars-control.maniflow.training.v1", "status": "complete", "step": args.steps,
              "all_episodes": True, "episodes": 600, "local_streams": 1650, "dataset_size": len(dataset),
              "samples_seen": args.steps * args.batch_size, "dataset_passes": args.steps * args.batch_size / len(dataset),
              "contract": model_config()["policy_contract"], "completed_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(output / ("smoke_status.json" if args.smoke else "status.json"), status)


if __name__ == "__main__": main()
