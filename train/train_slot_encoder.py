from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.slot_dataset import SlotWindowDataset, compute_slot_normalization
from models.slot_encoder import AgentObjectSlotEncoder, compute_slot_losses, make_config_from_args


def sync_args_with_tokenizer(args):
    if not args.tokenizer_ckpt:
        return
    ckpt_path = Path(args.tokenizer_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"tokenizer checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    tokenizer_cfg = ckpt.get("config", {})
    if "codebook_size" not in tokenizer_cfg:
        raise KeyError(f"tokenizer checkpoint missing config.codebook_size: {ckpt_path}")

    codebook_size = int(tokenizer_cfg["codebook_size"])
    if args.plan_codebook_size != codebook_size:
        print(
            f"Overriding plan_codebook_size: args={args.plan_codebook_size} "
            f"-> tokenizer={codebook_size}"
        )
        args.plan_codebook_size = codebook_size

    if "horizon" in tokenizer_cfg and args.horizon != int(tokenizer_cfg["horizon"]):
        print(f"Overriding horizon: args={args.horizon} -> tokenizer={int(tokenizer_cfg['horizon'])}")
        args.horizon = int(tokenizer_cfg["horizon"])


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


def append_log(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate(model, loader, device, norm_stats, amp_enabled: bool):
    model.eval()
    sums = {}
    n = 0
    phase_correct = 0
    phase_total = 0
    plan_correct = 0
    plan_total = 0
    contact_correct = 0
    contact_total = 0

    norm_stats = {k: v.to(device) for k, v in norm_stats.items()}

    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            with get_amp_context(device, amp_enabled):
                losses = compute_slot_losses(model, batch, norm_stats)

            for k, v in losses.items():
                if k.startswith("loss"):
                    sums[k] = sums.get(k, 0.0) + float(v.item())

            phase_correct += int((losses["pred_phase"] == batch["phase"]).sum().item())
            phase_total += int(batch["phase"].numel())

            valid_plan = batch["plan_token"] >= 0
            if valid_plan.any():
                plan_correct += int((losses["pred_plan"][valid_plan] == batch["plan_token"][valid_plan]).sum().item())
                plan_total += int(valid_plan.sum().item())

            pred_contact = losses["pred_contact_prob"] > 0.5
            contact_correct += int((pred_contact.float() == batch["contact"]).sum().item())
            contact_total += int(batch["contact"].numel())

            n += 1

    metrics = {k: v / max(1, n) for k, v in sums.items()}
    metrics["phase_acc"] = phase_correct / max(1, phase_total)
    metrics["plan_acc"] = plan_correct / max(1, plan_total)
    metrics["contact_acc"] = contact_correct / max(1, contact_total)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="datasets/stage2/train")
    parser.add_argument("--val_dir", type=str, default="datasets/stage2/val")
    parser.add_argument("--out_dir", type=str, default="checkpoints/slot_encoder")
    parser.add_argument("--tokenizer_ckpt", type=str, default="artifacts/plan_tokenizer/plan_tokenizer.pt")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--slot_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_object_slots", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--plan_codebook_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pose_weight", type=float, default=1.0)
    parser.add_argument("--object_weight", type=float, default=1.0)
    parser.add_argument("--contact_weight", type=float, default=0.2)
    parser.add_argument("--force_weight", type=float, default=0.1)
    parser.add_argument("--phase_weight", type=float, default=0.2)
    parser.add_argument("--plan_weight", type=float, default=0.2)
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

    sync_args_with_tokenizer(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    cfg = make_config_from_args(args)

    train_ds = SlotWindowDataset(
        args.train_dir,
        history=args.history,
        horizon=args.horizon,
        stride=args.stride,
        tokenizer_ckpt=args.tokenizer_ckpt,
        max_episodes=args.max_train_episodes,
        include_failures=True,
    )
    val_ds = SlotWindowDataset(
        args.val_dir,
        history=args.history,
        horizon=args.horizon,
        stride=args.stride,
        tokenizer_ckpt=args.tokenizer_ckpt,
        max_episodes=args.max_val_episodes,
        include_failures=True,
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
        norm_stats = compute_slot_normalization(args.train_dir, history=args.history, horizon=args.horizon)
        torch.save(norm_stats, norm_path)
        print("saved normalization:", norm_path)

    model = AgentObjectSlotEncoder(cfg).to(device)
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

    norm_stats_device = {k: v.to(device) for k, v in norm_stats.items()}

    with open(out_dir / "config.json", "w") as f:
        json.dump({"config": cfg.__dict__, "args": vars(args)}, f, indent=2)

    log_path = out_dir / "train_log.csv"

    for epoch in range(start_epoch, args.epochs):
        model.train()
        sums = {}
        n = 0
        skipped_nonfinite = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with get_amp_context(device, bool(args.amp)):
                losses = compute_slot_losses(model, batch, norm_stats_device)
                loss = losses["loss"]

            if not torch.isfinite(loss):
                skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix(loss="nonfinite", skipped=skipped_nonfinite)
                continue

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

            finite_for_log = True
            for k, v in losses.items():
                if k.startswith("loss") and not torch.isfinite(v):
                    finite_for_log = False
                    break

            if finite_for_log:
                for k, v in losses.items():
                    if k.startswith("loss"):
                        sums[k] = sums.get(k, 0.0) + float(v.item())
                n += 1
            else:
                skipped_nonfinite += 1

            pbar.set_postfix(loss=float(loss.item()), obj=float(losses["loss_object_pose"].item()), other=float(losses["loss_other_pose"].item()), skipped=skipped_nonfinite)

        train_metrics = {f"train_{k}": v / max(1, n) for k, v in sums.items()}
        train_metrics["train_skipped_nonfinite_batches"] = skipped_nonfinite
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
