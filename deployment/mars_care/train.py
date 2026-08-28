from __future__ import annotations

import argparse, json, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .dataset import MarsLocalDataset, TaskBalancedDistributedSampler
from .model import CAREPolicy, ModelConfig


def atomic_save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(".tmp"); torch.save(value, tmp); os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--raw-root", type=Path, required=True); p.add_argument("--normalization", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=100000); p.add_argument("--batch-size", type=int, default=96); p.add_argument("--workers", type=int, default=12); p.add_argument("--save-every", type=int, default=5000); p.add_argument("--smoke", action="store_true"); p.add_argument("--pretrained", action="store_true")
    p.add_argument("--init-checkpoint", type=Path); p.add_argument("--init-vision-only", action="store_true")
    args = p.parse_args(); distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl"); rank = dist.get_rank(); local_rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local_rank)
    else: rank = local_rank = 0
    random.seed(20260824 + rank); np.random.seed(20260824 + rank); torch.manual_seed(20260824 + rank)
    dataset = MarsLocalDataset(args.raw_root, args.normalization)
    sampler = TaskBalancedDistributedSampler(dataset, rank=rank, replicas=dist.get_world_size() if distributed else 1)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0, drop_last=True)
    device = torch.device("cuda", local_rank); model = CAREPolicy(pretrained=args.pretrained and rank == 0).to(device)
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)["model"]
        current = model.state_dict()
        compatible = {
            key: value for key, value in initial.items()
            if key in current and current[key].shape == value.shape
            and (not args.init_vision_only or key.startswith("vision."))
        }
        model.load_state_dict(compatible, strict=False)
        if rank == 0: print(json.dumps({"event": "initialized", "checkpoint": str(args.init_checkpoint), "vision_only": args.init_vision_only, "tensors": len(compatible)}), flush=True)
    if distributed: dist.barrier()
    # DDP's constructor broadcasts rank 0 parameters, including the pretrained
    # vision weights, without serializing a 100+ MB Python object.
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False) if distributed else model
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=3e-6)
    start = 0; latest = args.output / "latest.pt"
    if latest.is_file() and not args.smoke:
        saved = torch.load(latest, map_location="cpu", weights_only=False); model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"]); start = int(saved["step"])
    iterator = iter(loader); started = time.time(); progress = args.output / "progress.jsonl"
    for step in range(start + 1, args.steps + 1):
        try: batch = next(iterator)
        except StopIteration:
            if distributed and hasattr(sampler, "set_epoch"): sampler.set_epoch(step)
            iterator = iter(loader); batch = next(iterator)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = wrapped(batch["image"], batch["qpos"], batch["task_id"], batch["history"], batch["history_mask"])
            loss, pieces = model.loss_from_output(output, batch)
        if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss at {step}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0); optimizer.step(); scheduler.step()
        if rank == 0 and (step == 1 or step % 20 == 0):
            row = {"step": step, "total_steps": args.steps, "lr": scheduler.get_last_lr()[0], "elapsed_seconds": time.time() - started, **{k: float(v) for k, v in pieces.items()}}
            progress.parent.mkdir(parents=True, exist_ok=True)
            with progress.open("a") as stream: stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if rank == 0 and (step % args.save_every == 0 or step == args.steps):
            payload = {"format": "mars-care-checkpoint-v2-residual-history", "step": step, "model_config": model.config_dict(), "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "normalization": json.loads(args.normalization.read_text()), "strict_local_inputs": ["head_camera_agent{i}/rgb", "panda-{i}/qpos_history", "panda-{i}/previous_local_action", "task_id"], "action_encoding": "joint_residual_gripper_absolute", "task_balanced_sampling": True}
            atomic_save(payload, latest)
            if step == args.steps: atomic_save(payload, args.output / "final.pt")
    if distributed: dist.destroy_process_group()


if __name__ == "__main__": main()
