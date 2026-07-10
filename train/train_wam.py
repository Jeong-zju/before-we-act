from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.wam_dataset import WAMWindowDataset
from models.plan_tokenizer import PlanTokenizer, PlanTokenizerConfig
from models.slot_encoder import AgentObjectSlotEncoder, SlotEncoderConfig
from models.wam import LatentWorldActionModel, compute_wam_losses, count_parameters, make_config_from_args


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def amp_context(device: torch.device, dtype: str, enabled: bool):
    if device.type == "cuda" and enabled:
        if dtype == "bf16":
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if dtype == "fp16":
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return torch.amp.autocast(device_type="cpu", enabled=False)


def make_scaler(device: torch.device, dtype: str, enabled: bool):
    if device.type == "cuda" and enabled and dtype == "fp16":
        try:
            return torch.amp.GradScaler("cuda")
        except TypeError:
            return torch.cuda.amp.GradScaler()
    return None


def load_slot_encoder(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    cfg = SlotEncoderConfig(**ckpt["config"])
    model = AgentObjectSlotEncoder(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    norm = {k: v.to(device) for k, v in ckpt["normalization"].items()}
    return model, cfg, norm


def load_plan_tokenizer(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    cfg = PlanTokenizerConfig(**ckpt["config"])
    model = PlanTokenizer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    norm = {k: v.to(device) for k, v in ckpt["normalization"].items()}
    return model, cfg, norm


def sync_args_with_frozen_configs(args, slot_cfg: SlotEncoderConfig, plan_cfg: PlanTokenizerConfig):
    expected_slots = 2 + int(getattr(slot_cfg, "num_object_slots", 2))
    overrides = {
        "history": int(slot_cfg.history),
        "horizon": int(plan_cfg.horizon),
        "slots_per_agent": expected_slots,
        "slot_dim": int(slot_cfg.slot_dim),
        "plan_codebook_size": int(plan_cfg.codebook_size),
        "plan_latent_dim": int(plan_cfg.latent_dim),
    }
    for name, value in overrides.items():
        old = getattr(args, name)
        if old != value:
            source = "slot checkpoint" if name in {"history", "slots_per_agent", "slot_dim"} else "tokenizer checkpoint"
            print(f"Overriding {name}: args={old} -> {source}={value}")
            setattr(args, name, value)


@torch.no_grad()
def build_wam_targets(batch, slot_encoder, slot_norm, plan_tokenizer, plan_norm, device):
    # local_history_seq: [B, H+1, A, L, 17]
    local = batch["local_history_seq"]
    phase_hist = batch["phase_history_seq"]
    agent_ids = batch["agent_ids"]

    B, Hp1, A, L, D = local.shape

    local_flat = local.reshape(B * Hp1 * A, L, D)
    phase_flat = phase_hist.reshape(B * Hp1 * A, L)
    agent_ids_flat = agent_ids.view(B, 1, A).expand(B, Hp1, A).reshape(-1)

    local_flat_norm = (local_flat - slot_norm["local_mean"].view(1, 1, -1)) / slot_norm["local_std"].view(1, 1, -1)

    enc = slot_encoder.encode_slots(local_flat_norm, agent_ids_flat, phase_flat)
    slots = enc["slots"].reshape(B, Hp1, A, -1, enc["slots"].shape[-1])

    current_slots = slots[:, 0]
    future_slots = slots[:, 1:]

    # plan_actions: [B, A, H, 4], plan_trajectory: [B, A, H, 5]
    pa = batch["plan_actions"]
    pt = batch["plan_trajectory"]
    _, _, H, _ = pa.shape

    pa_flat = pa.reshape(B * A, H, 4)
    pt_flat = pt.reshape(B * A, H, 5)

    pa_norm = (pa_flat - plan_norm["action_mean"].view(1, 1, -1)) / plan_norm["action_std"].view(1, 1, -1)
    pt_norm = (pt_flat - plan_norm["traj_mean"].view(1, 1, -1)) / plan_norm["traj_std"].view(1, 1, -1)

    plan_enc = plan_tokenizer.encode_future_segment(pa_norm, pt_norm)
    plan_codes = plan_enc["code_indices"].reshape(B, A)
    plan_residuals = plan_enc["residual"].reshape(B, A, -1)

    return {
        "current_slots": current_slots,
        "future_slots": future_slots,
        "plan_codes": plan_codes,
        "plan_residuals": plan_residuals,
    }


def append_log(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def evaluate(model, loader, slot_encoder, slot_norm, plan_tokenizer, plan_norm, device, amp_enabled: bool, amp_dtype: str, max_batches: int = -1):
    model.eval()
    sums = {}
    n = 0
    contact_correct = 0
    contact_total = 0

    for batch in tqdm(loader, desc="val"):
        batch = to_device(batch, device)
        targets = build_wam_targets(batch, slot_encoder, slot_norm, plan_tokenizer, plan_norm, device)

        with amp_context(device, amp_dtype, amp_enabled):
            losses = compute_wam_losses(model, batch, targets)

        for k, v in losses.items():
            if k.startswith("loss"):
                sums[k] = sums.get(k, 0.0) + float(v.item())

        pred_contact = (losses["pred_contact_prob"] > 0.5).float()
        contact_correct += int((pred_contact == batch["target_contact"]).sum().item())
        contact_total += int(batch["target_contact"].numel())

        n += 1
        if max_batches > 0 and n >= max_batches:
            break

    metrics = {k: v / max(1, n) for k, v in sums.items()}
    metrics["contact_acc"] = contact_correct / max(1, contact_total)
    return metrics


def save_checkpoint(path: Path, model, optimizer, epoch: int, cfg, best_val: float, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": cfg.__dict__,
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
    parser.add_argument("--out_dir", type=str, default="checkpoints/wam")
    parser.add_argument("--slot_ckpt", type=str, default="artifacts/slot_encoder/slot_encoder.pt")
    parser.add_argument("--plan_ckpt", type=str, default="artifacts/plan_tokenizer/plan_tokenizer.pt")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--slots_per_agent", type=int, default=4)
    parser.add_argument("--slot_dim", type=int, default=128)
    parser.add_argument("--plan_codebook_size", type=int, default=32)
    parser.add_argument("--plan_latent_dim", type=int, default=64)
    parser.add_argument("--model_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--ffn_dim", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_checkpoint", type=int, default=1)
    parser.add_argument("--slot_loss_weight", type=float, default=1.0)
    parser.add_argument("--action_loss_weight", type=float, default=1.0)
    parser.add_argument("--contact_loss_weight", type=float, default=0.2)
    parser.add_argument("--force_loss_weight", type=float, default=0.2)
    parser.add_argument("--progress_loss_weight", type=float, default=0.5)
    parser.add_argument("--smooth_action_weight", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "none"])
    parser.add_argument("--max_train_episodes", type=int, default=-1)
    parser.add_argument("--max_val_episodes", type=int, default=-1)
    parser.add_argument("--max_val_batches", type=int, default=-1)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
        print("bf16 supported:", torch.cuda.is_bf16_supported())

    amp_enabled = bool(args.amp) and args.amp_dtype != "none"

    slot_encoder, slot_cfg, slot_norm = load_slot_encoder(args.slot_ckpt, device)
    plan_tokenizer, plan_cfg, plan_norm = load_plan_tokenizer(args.plan_ckpt, device)
    sync_args_with_frozen_configs(args, slot_cfg, plan_cfg)

    cfg = make_config_from_args(args)
    model = LatentWorldActionModel(cfg).to(device)
    n_params = count_parameters(model)
    print("WAM parameters:", n_params, f"({n_params / 1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_scaler(device, args.amp_dtype, amp_enabled)

    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt.get("best_val", best_val))
        print("resumed:", args.resume, "start_epoch:", start_epoch)

    train_ds = WAMWindowDataset(args.train_dir, history=args.history, horizon=args.horizon, stride=args.stride, max_episodes=args.max_train_episodes)
    val_ds = WAMWindowDataset(args.val_dir, history=args.history, horizon=args.horizon, stride=args.stride, max_episodes=args.max_val_episodes)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=False)

    with open(out_dir / "config.json", "w") as f:
        json.dump({"config": cfg.__dict__, "args": vars(args), "num_parameters": n_params}, f, indent=2)

    log_path = out_dir / "train_log.csv"

    for epoch in range(start_epoch, args.epochs):
        model.train()
        sums = {}
        n = 0
        skipped = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for step, batch in enumerate(pbar):
            batch = to_device(batch, device)

            with torch.no_grad():
                targets = build_wam_targets(batch, slot_encoder, slot_norm, plan_tokenizer, plan_norm, device)

            with amp_context(device, args.amp_dtype, amp_enabled):
                losses = compute_wam_losses(model, batch, targets)
                loss = losses["loss"] / args.grad_accum_steps

            if not torch.isfinite(loss):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix(loss="nonfinite", skipped=skipped)
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = ((step + 1) % args.grad_accum_steps == 0)
            if should_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            finite_log = True
            for k, v in losses.items():
                if k.startswith("loss") and not torch.isfinite(v):
                    finite_log = False
                    break

            if finite_log:
                for k, v in losses.items():
                    if k.startswith("loss"):
                        sums[k] = sums.get(k, 0.0) + float(v.item())
                n += 1
            else:
                skipped += 1

            pbar.set_postfix(
                loss=float(losses["loss"].item()),
                slots=float(losses["loss_slots"].item()),
                act=float(losses["loss_actions"].item()),
                skipped=skipped,
            )

        train_metrics = {f"train_{k}": v / max(1, n) for k, v in sums.items()}
        train_metrics["train_skipped"] = skipped

        val_metrics = evaluate(model, val_loader, slot_encoder, slot_norm, plan_tokenizer, plan_norm, device, amp_enabled, args.amp_dtype, max_batches=args.max_val_batches)
        val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}

        row = {"epoch": epoch, **train_metrics, **val_metrics}
        append_log(log_path, row)
        print(json.dumps(row, indent=2))

        val_loss = row["val_loss"]
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, cfg, best_val, args)

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(out_dir / f"epoch_{epoch:04d}.pt", model, optimizer, epoch, cfg, best_val, args)

    save_checkpoint(out_dir / "last.pt", model, optimizer, args.epochs - 1, cfg, best_val, args)


if __name__ == "__main__":
    main()
