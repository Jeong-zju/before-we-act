from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.plan_dataset import PlanWindowDataset, compute_plan_normalization, save_normalization
from models.plan_tokenizer import PlanTokenizer, compute_losses, codebook_usage, make_config_from_args


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def get_amp_context(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return torch.amp.autocast(device_type="cpu", enabled=False)


def make_scaler(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        try:
            return torch.amp.GradScaler("cuda")
        except TypeError:
            return torch.cuda.amp.GradScaler()
    return None


def evaluate(model, loader, device, norm_stats, amp_enabled: bool):
    model.eval()
    sums = {}
    n_batches = 0
    all_codes = []

    action_mean = norm_stats["action_mean"].to(device)
    action_std = norm_stats["action_std"].to(device)
    traj_mean = norm_stats["traj_mean"].to(device)
    traj_std = norm_stats["traj_std"].to(device)

    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            with get_amp_context(device, amp_enabled):
                losses = compute_losses(model, batch, action_mean, action_std, traj_mean, traj_std)

            for k, v in losses.items():
                if k.startswith("loss"):
                    sums[k] = sums.get(k, 0.0) + float(v.item())
            all_codes.append(losses["code_indices"].detach().cpu())
            n_batches += 1

    metrics = {k: v / max(1, n_batches) for k, v in sums.items()}
    codes = torch.cat(all_codes, dim=0)
    metrics.update({f"codebook_{k}": v for k, v in codebook_usage(codes, model.cfg.codebook_size).items()})
    return metrics


def save_checkpoint(path: Path, model, optimizer, epoch: int, cfg, norm_stats, best_val: float, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": cfg.__dict__,
            "normalization": {k: v.cpu() for k, v in norm_stats.items()},
            "best_val": best_val,
            "args": vars(args),
        },
        path,
    )
    print("saved checkpoint:", path)


def append_log(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="datasets/stage2/train")
    parser.add_argument("--val_dir", type=str, default="datasets/stage2/val")
    parser.add_argument("--out_dir", type=str, default="checkpoints/plan_tokenizer")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--codebook_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--commitment_weight", type=float, default=0.25)
    parser.add_argument("--phase_weight", type=float, default=0.1)
    parser.add_argument("--residual_weight", type=float, default=0.05)
    parser.add_argument("--residual_dropout", type=float, default=0.0)
    parser.add_argument("--stop_residual_grad_to_encoder", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--max_train_episodes", type=int, default=-1)
    parser.add_argument("--max_val_episodes", type=int, default=-1)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save_every", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))

    cfg = make_config_from_args(args)

    train_ds = PlanWindowDataset(
        args.train_dir,
        horizon=args.horizon,
        stride=args.stride,
        include_failures=True,
        max_episodes=args.max_train_episodes,
    )
    val_ds = PlanWindowDataset(
        args.val_dir,
        horizon=args.horizon,
        stride=args.stride,
        include_failures=True,
        max_episodes=args.max_val_episodes,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    norm_path = out_dir / "normalization.pt"
    if norm_path.exists():
        norm_stats = torch.load(norm_path, map_location="cpu")
        print("loaded normalization:", norm_path)
    else:
        norm_stats = compute_plan_normalization(args.train_dir, horizon=args.horizon)
        save_normalization(norm_stats, str(norm_path))

    model = PlanTokenizer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_scaler(device, bool(args.amp))
    start_epoch = 0
    best_val = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt.get("best_val", best_val))
        print("resumed:", args.resume, "start_epoch:", start_epoch)

    action_mean = norm_stats["action_mean"].to(device)
    action_std = norm_stats["action_std"].to(device)
    traj_mean = norm_stats["traj_mean"].to(device)
    traj_std = norm_stats["traj_std"].to(device)

    log_path = out_dir / "train_log.csv"

    with open(out_dir / "config.json", "w") as f:
        json.dump({"config": cfg.__dict__, "args": vars(args)}, f, indent=2)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_sums = {}
        n_batches = 0
        all_train_codes = []

        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with get_amp_context(device, bool(args.amp)):
                losses = compute_losses(model, batch, action_mean, action_std, traj_mean, traj_std)
                loss = losses["loss"]

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            for k, v in losses.items():
                if k.startswith("loss"):
                    train_sums[k] = train_sums.get(k, 0.0) + float(v.item())
            all_train_codes.append(losses["code_indices"].detach().cpu())
            n_batches += 1

            pbar.set_postfix(loss=float(loss.item()), action=float(losses["loss_action"].item()), traj=float(losses["loss_traj"].item()))

        train_metrics = {f"train_{k}": v / max(1, n_batches) for k, v in train_sums.items()}
        train_codes = torch.cat(all_train_codes, dim=0)
        train_usage = codebook_usage(train_codes, cfg.codebook_size)
        train_metrics.update({f"train_codebook_{k}": v for k, v in train_usage.items()})

        val_metrics = evaluate(model, val_loader, device, norm_stats, bool(args.amp))
        val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}

        row = {"epoch": epoch, **train_metrics, **val_metrics}
        append_log(log_path, row)

        print(json.dumps(row, indent=2))

        val_loss = row["val_loss"]
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, cfg, norm_stats, best_val, args)

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(out_dir / f"epoch_{epoch:04d}.pt", model, optimizer, epoch, cfg, norm_stats, best_val, args)

    save_checkpoint(out_dir / "last.pt", model, optimizer, args.epochs - 1, cfg, norm_stats, best_val, args)


if __name__ == "__main__":
    main()
