from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from diffusion_policy.model.diffusion.ema_model import EMAModel

from .common import FROZEN_CONFIG, POLICY_CONTRACT, UPSTREAM_COMMIT, atomic_json, load_frozen_config, sha256
from .dataset import MarsLatentToMDataset, TaskBalancedBatchSampler
from .policy import LocalLatentToMPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--config", type=Path, default=FROZEN_CONFIG)
    parser.add_argument("--steps", type=int); parser.add_argument("--batch-size", type=int); parser.add_argument("--workers", type=int)
    parser.add_argument("--grad-accum", type=int); parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); frozen = load_frozen_config(args.config)
    optimization = frozen["optimization"]; loader_config = frozen["loader"]
    def pinned(value, expected, name):
        if value is not None and int(value) != int(expected):
            raise ValueError(f"{name} disagrees with frozen config: {value} != {expected}")
        return int(expected)
    args.steps = pinned(args.steps, optimization["smoke_updates"] if args.smoke else optimization["optimizer_updates"], "steps")
    args.batch_size = pinned(args.batch_size, optimization["global_batch_size"], "batch-size")
    args.grad_accum = pinned(args.grad_accum, optimization["gradient_accumulation_steps"], "grad-accum")
    args.seed = pinned(args.seed, optimization["seed"], "seed")
    args.workers = pinned(args.workers, loader_config["smoke_workers"] if args.smoke else loader_config["workers"], "workers")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(optimization["torch_cpu_threads"])
    args.output.mkdir(parents=True, exist_ok=True); ds = MarsLatentToMDataset(args.data_root, args.stats)
    if args.batch_size % 4: raise ValueError("batch size must be divisible by four tasks")
    sampler = TaskBalancedBatchSampler(ds.task_indices, args.batch_size, args.steps, args.seed)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=args.workers,
                        pin_memory=loader_config["pin_memory"],
                        persistent_workers=loader_config["persistent_workers"],
                        prefetch_factor=loader_config["prefetch_factor"] if args.workers > 0 else None)
    model_config = frozen["model"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LocalLatentToMPolicy.from_frozen_config(frozen).to(device); model.set_stats(ds.stats)
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimization["learning_rate"], betas=tuple(optimization["betas"]), eps=optimization["epsilon"], weight_decay=optimization["weight_decay"], amsgrad=optimization["amsgrad"])
    start = 0; payload = {}; latest = args.output / "last.pt"
    if args.resume and latest.is_file():
        try:
            payload = torch.load(latest, map_location=device, weights_only=False)
            if payload.get("frozen_config") and payload["frozen_config"] != frozen:
                raise ValueError("checkpoint frozen config does not match project config")
            model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"]); start = int(payload["step"])
        except (EOFError, RuntimeError, KeyError, ValueError, OSError) as error:
            candidates = sorted(args.output.glob("step_*.pt"), reverse=True)
            for candidate in candidates:
                try:
                    fallback = torch.load(candidate, map_location=device, weights_only=False)
                    if fallback.get("frozen_config") and fallback["frozen_config"] != frozen:
                        raise ValueError("checkpoint frozen config does not match project config")
                    model.load_state_dict(fallback["model"]); optimizer.load_state_dict(fallback["optimizer"])
                    payload, start = fallback, int(fallback["step"])
                    print(json.dumps({"event": "resume_fallback", "checkpoint": str(candidate), "step": start, "invalid_latest": str(error)}), flush=True)
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError(f"no valid resume checkpoint in {args.output}") from error
    def rate(step):
        import math
        warmup = optimization["warmup_updates"]
        if step < warmup: return max((step + 1) / warmup, 1e-8)
        return 0.5 * (1 + math.cos(math.pi * min(max((step - warmup) / max(args.steps - warmup, 1), 0), 1)))
    for group in optimizer.param_groups: group.setdefault("initial_lr", group["lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate, last_epoch=start - 1)
    if payload.get("scheduler"): scheduler.load_state_dict(payload["scheduler"])
    ema_config = frozen["ema"]
    ema_model = copy.deepcopy(model).to(device); ema_model.load_state_dict(payload.get("ema_model", model.state_dict()))
    ema = EMAModel(ema_model, update_after_step=ema_config["update_after_step"], inv_gamma=ema_config["inv_gamma"], power=ema_config["power"], min_value=ema_config["min_value"], max_value=ema_config["max_value"]); ema.optimization_step = int(payload.get("ema_step", start))
    iterator = iter(loader); target = args.steps; started = time.monotonic(); optimizer.zero_grad(set_to_none=True)
    for step in range(start, target):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        obs = {key: value.to(device, non_blocking=True) for key, value in batch.items() if key in ("image", "qpos")}; action = batch["action"].to(device, non_blocking=True); mask = batch["action_mask"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, pieces = model.loss(obs, action, mask); loss = loss / args.grad_accum
        if not torch.isfinite(loss): raise RuntimeError(f"nonfinite loss at step {step + 1}")
        loss.backward()
        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), optimization["gradient_clip_norm"]); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); ema.step(model)
            if frozen["ema"]["copy_buffers_each_update"]:
                with torch.no_grad():
                    for dst, src in zip(ema_model.buffers(), model.buffers()): dst.copy_(src)
        if step == start or (step + 1) % frozen["checkpointing"]["log_interval_updates"] == 0: print(json.dumps({"step": step + 1, "target_steps": target, "loss": float(loss * args.grad_accum), "diffusion": float(pieces["diffusion"]), "tom": float(pieces["tom"]), "lr": scheduler.get_last_lr()[0], "steps_per_sec": (step + 1 - start) / max(time.monotonic() - started, 1e-6)}), flush=True)
        if (step + 1) % frozen["checkpointing"]["interval_updates"] == 0 or step + 1 == target:
            saved = {"schema": "mars-control.latent-tom.checkpoint.v1", "step": step + 1, "model": model.state_dict(), "ema_model": ema_model.state_dict(), "ema_step": ema.optimization_step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "stats": ds.stats, "contract": POLICY_CONTRACT, "upstream_commit": UPSTREAM_COMMIT, "frozen_config": frozen, "config": {"tasks": [x.name for x in __import__("deployment.mars_latent_tom.common", fromlist=["TASKS"]).TASKS], "horizon": model_config["action_horizon"], "obs_steps": model_config["observation_steps"], "batch_size": args.batch_size, "grad_accum": args.grad_accum, "workers": args.workers, "all_600_episodes_no_split": True, "indexed_local_timesteps": len(ds)}}
            temporary = args.output / f"last.pt.tmp-{os.getpid()}"
            torch.save(saved, temporary); os.replace(temporary, latest)
            step_path = args.output / f"step_{step + 1:06d}.pt"
            step_tmp = args.output / f"{step_path.name}.tmp-{os.getpid()}"
            torch.save(saved, step_tmp); os.replace(step_tmp, step_path)
    atomic_json(args.output / ("smoke_status.json" if args.smoke else "status.json"), {"schema": "mars-control.latent-tom.training.v1", "status": "complete", "step": target, "target_steps": target, "checkpoint": str(latest), "checkpoint_sha256": sha256(latest), "episodes": 600, "local_streams": 1650, "indexed_local_timesteps": len(ds), "all_episodes": True, "effective_batch_size": args.batch_size * args.grad_accum, "samples_seen": target * args.batch_size * args.grad_accum, "frozen_config": str(args.config), "frozen_config_sha256": sha256(args.config), "upstream_commit": UPSTREAM_COMMIT})


if __name__ == "__main__": main()
