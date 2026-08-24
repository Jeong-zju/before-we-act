from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from local_dataset import LocalWindowDataset
from local_policy import LocalLatentToMPolicy
from diffusion_policy.model.diffusion.ema_model import EMAModel


class LocalityBatchSampler:
    """Mix shuffled short contiguous runs into full-coverage batches.

    Short runs retain HDF5 read coalescing while every optimizer batch contains
    many independently shuffled trajectory regions.  This avoids presenting a
    512-sample batch from only one episode/task to the visual backbone.
    """

    def __init__(self, size: int, batch_size: int, seed: int, locality: int = 16):
        self.size = int(size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.locality = min(int(locality), self.batch_size)
        self.epoch = 0

    def __len__(self):
        return (self.size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        runs = [range(start, min(start + self.locality, self.size))
                for start in range(0, self.size, self.locality)]
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(runs)
        self.epoch += 1
        batch = []
        for run in runs:
            batch.extend(run)
            while len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
        if batch:
            yield batch


def atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def load_stats(path: Path):
    return json.loads(path.read_text())


def make_lr_scheduler(opt, total_steps: int, warmup_steps: int, start_step: int = 0):
    """LatentToM-style linear warmup followed by cosine decay."""
    import math
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)
    def schedule(step):
        if warmup_steps and step < warmup_steps:
            return max(float(step + 1) / warmup_steps, 1e-8)
        progress = min(max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in opt.param_groups:
        group.setdefault("initial_lr", group["lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule, last_epoch=start_step - 1)
    return scheduler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default=os.environ.get("BWA_DATASET_ROOT", "/workspace/datasets/robofactory_multitask"))
    p.add_argument("--output", default=os.environ.get("BWA_LATENT_TOM_OUTPUT", "/workspace/bwa_latent_tom_runs/formal"))
    p.add_argument("--stats", default=None)
    p.add_argument("--seed", type=int, default=int(os.environ.get("LATENT_TOM_SEED", "20260822")))
    p.add_argument("--steps", type=int, default=int(os.environ.get("LATENT_TOM_STEPS", "300000")))
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("LATENT_TOM_BATCH", "16")))
    p.add_argument("--grad-accum", type=int, default=int(os.environ.get("LATENT_TOM_GRAD_ACCUM", "2")))
    p.add_argument("--workers", type=int, default=int(os.environ.get("LATENT_TOM_WORKERS", "8")))
    p.add_argument("--locality", type=int, default=int(os.environ.get("LATENT_TOM_LOCALITY", "16")))
    p.add_argument("--save-every", type=int, default=int(os.environ.get("LATENT_TOM_SAVE_EVERY", "5000")))
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    stats_path = Path(args.stats or out / "normalization.json")
    if not stats_path.is_file(): raise FileNotFoundError(stats_path)
    stats = load_stats(stats_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LocalLatentToMPolicy(
        horizon=int(os.environ.get("LATENT_TOM_ACTION_HORIZON", "40")),
        obs_steps=int(os.environ.get("LATENT_TOM_OBS_STEPS", "2")),
        action_dim=int(os.environ.get("LATENT_TOM_ACTION_DIM", "8")),
    ).to(device)
    model.set_stats(stats)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(os.environ.get("LATENT_TOM_LEARNING_RATE", "1e-4")),
        betas=(float(os.environ.get("LATENT_TOM_BETA1", "0.95")),
               float(os.environ.get("LATENT_TOM_BETA2", "0.999"))),
        weight_decay=float(os.environ.get("LATENT_TOM_WEIGHT_DECAY", "1e-6")),
    )
    start_step = 0
    scheduler = None
    payload = {}
    latest = out / "last.pt"
    if args.resume and latest.is_file():
        payload = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"]); opt.load_state_dict(payload["optimizer"]); start_step = int(payload["step"])
        scheduler = make_lr_scheduler(opt, args.steps, int(os.environ.get("LATENT_TOM_WARMUP", "500")), start_step)
        if payload.get("scheduler"):
            scheduler.load_state_dict(payload["scheduler"])
        print(json.dumps({"event": "resume", "step": start_step}), flush=True)
    if scheduler is None:
        scheduler = make_lr_scheduler(opt, args.steps, int(os.environ.get("LATENT_TOM_WARMUP", "500")), start_step)
    ema_model = copy.deepcopy(model).to(device)
    if payload.get("ema_model"):
        ema_model.load_state_dict(payload["ema_model"])
    ema = EMAModel(ema_model, update_after_step=0, inv_gamma=1.0, power=0.75,
                   min_value=0.0, max_value=0.9999)
    ema.optimization_step = int(payload.get("ema_optimization_step", start_step))
    ds = LocalWindowDataset(args.dataset_root)
    batch_sampler = LocalityBatchSampler(len(ds), args.batch_size, seed=args.seed,
                                         locality=args.locality)
    loader = DataLoader(ds, batch_sampler=batch_sampler, num_workers=args.workers,
                        pin_memory=True,
                        persistent_workers=args.workers > 0, prefetch_factor=2)
    iterator = iter(loader)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    model.train(); t0 = time.monotonic(); opt.zero_grad(set_to_none=True)
    target_steps = min(args.steps, 10 if args.smoke else args.steps)
    for step in range(start_step, target_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        obs = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k in ("image", "qpos", "task")}
        action = batch["action"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = model.loss(obs, action) / args.grad_accum
        loss.backward()
        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); scheduler.step(); opt.zero_grad(set_to_none=True)
            ema.step(model)
            # Upstream EMA updates parameters but not buffers.  Keep local
            # BatchNorm running statistics and normalization buffers current.
            with torch.no_grad():
                for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
                    ema_buffer.copy_(buffer)
        if (step + 1) % args.log_every == 0 or step == start_step:
            elapsed = time.monotonic() - t0
            print(json.dumps({"event": "train", "step": step + 1, "loss": float(loss.item() * args.grad_accum), "steps_per_sec": (step + 1 - start_step) / max(elapsed, 1e-6)}), flush=True)
        if (step + 1) % args.save_every == 0 or step + 1 == target_steps:
            payload = {"schema": "bwa.latent_tom.local_checkpoint.v1", "step": step + 1,
                       "model": model.state_dict(), "optimizer": opt.state_dict(),
                       "scheduler": scheduler.state_dict(),
                       "ema_model": ema_model.state_dict(),
                       "ema_optimization_step": ema.optimization_step,
                       "stats": stats, "contract": "shared_weights_local_rgb_qpos_task_to_local_action8",
                       "official_repo": "StanfordMSL/LatentToM@a51d929027799a53d54e7d7d2ba90e2703642b4a",
                       "seed": args.seed}
            tmp = out / f"last.pt.tmp-{os.getpid()}"; torch.save(payload, tmp); os.replace(tmp, latest)
            torch.save(payload, out / f"step_{step + 1:08d}.pt")
    status = {"status": "complete", "step": target_steps, "checkpoint": str(latest),
              "checkpoint_sha256": sha256(latest), "protocol": "decentralized_local_rgb_qpos_task_action8",
              "all_episodes": True, "indexed_local_timesteps": len(ds),
              "effective_dataset_passes": target_steps * args.batch_size / len(ds),
              "official_latent_tom_commit": "a51d929027799a53d54e7d7d2ba90e2703642b4a",
              "completed_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(out / ("smoke_status.json" if args.smoke else "status.json"), status)


if __name__ == "__main__": main()
