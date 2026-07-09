from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.intention_dataset import IntentionWindowDataset
from models.intention import IntentionInferenceModel, compute_intention_losses, count_parameters, make_config_from_args
from models.plan_tokenizer import PlanTokenizer, PlanTokenizerConfig
from models.slot_encoder import AgentObjectSlotEncoder, SlotEncoderConfig
from models.wam import LatentWorldActionModel, WAMConfig


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


def load_wam(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    cfg = WAMConfig(**ckpt["config"])
    model = LatentWorldActionModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


@torch.no_grad()
def encode_plans(batch, plan_tokenizer, plan_norm):
    B = batch["ego_plan_actions"].shape[0]
    H = batch["ego_plan_actions"].shape[1]

    ego_a = batch["ego_plan_actions"]
    ego_x = batch["ego_plan_trajectory"]
    tgt_a = batch["target_plan_actions"]
    tgt_x = batch["target_plan_trajectory"]

    ego_a_norm = (ego_a - plan_norm["action_mean"].view(1, 1, -1)) / plan_norm["action_std"].view(1, 1, -1)
    ego_x_norm = (ego_x - plan_norm["traj_mean"].view(1, 1, -1)) / plan_norm["traj_std"].view(1, 1, -1)
    tgt_a_norm = (tgt_a - plan_norm["action_mean"].view(1, 1, -1)) / plan_norm["action_std"].view(1, 1, -1)
    tgt_x_norm = (tgt_x - plan_norm["traj_mean"].view(1, 1, -1)) / plan_norm["traj_std"].view(1, 1, -1)

    ego_enc = plan_tokenizer.encode_future_segment(ego_a_norm, ego_x_norm)
    tgt_enc = plan_tokenizer.encode_future_segment(tgt_a_norm, tgt_x_norm)

    ego_codes = ego_enc["code_indices"].long()
    tgt_codes = tgt_enc["code_indices"].long()

    codebook_size = int(plan_tokenizer.cfg.codebook_size)
    if ego_codes.numel() > 0:
        ego_min, ego_max = int(ego_codes.min().item()), int(ego_codes.max().item())
        tgt_min, tgt_max = int(tgt_codes.min().item()), int(tgt_codes.max().item())
        if ego_min < 0 or tgt_min < 0 or ego_max >= codebook_size or tgt_max >= codebook_size:
            raise ValueError(
                f"Plan code out of range for tokenizer codebook_size={codebook_size}: "
                f"ego=[{ego_min},{ego_max}], target=[{tgt_min},{tgt_max}]"
            )

    return {
        "ego_plan_codes": ego_codes,
        "ego_plan_residuals": ego_enc["residual"],
        "target_plan_codes": tgt_codes,
        "target_plan_residuals": tgt_enc["residual"],
    }


@torch.no_grad()
def encode_ego_slots(batch, slot_encoder, slot_norm):
    local = batch["local_history"]
    phase_hist = batch["phase_history"]
    local_norm = (local - slot_norm["local_mean"].view(1, 1, -1)) / slot_norm["local_std"].view(1, 1, -1)
    enc = slot_encoder.encode_slots(local_norm, batch["ego_id"], phase_hist)
    return enc["slots"]


def build_consistency_loss(batch, targets, intention_model, wam, amp_enabled: bool = False, device=None):
    # Use predicted teammate plan with ego plan and ego slots repeated into a 2-agent WAM input.
    # This is a lightweight consistency proxy: predicted teammate plan should produce actions close to recorded joint actions.
    with torch.no_grad():
        out_int = intention_model(
            ego_slots=targets["ego_slots"],
            ego_plan_codes=targets["ego_plan_codes"],
            ego_plan_residuals=targets["ego_plan_residuals"],
            ego_id=batch["ego_id"],
            phase_history=batch["phase_history"],
            rel_target_pose=batch["rel_target_pose"],
            object_rel_pose=batch["object_rel_pose"],
        )

    pred_code = out_int["target_code_logits"].argmax(dim=-1)
    pred_residual = out_int["target_residual_mu"]

    B = pred_code.shape[0]
    A = 2

    # WAM expects current_slots [B, 2, slots, dim].
    # We only have ego slots; duplicate as a stable proxy for consistency training.
    current_slots = targets["ego_slots"].unsqueeze(1).expand(B, A, -1, -1).contiguous()

    plan_codes = torch.zeros(B, A, dtype=torch.long, device=pred_code.device)
    plan_residuals = torch.zeros(B, A, targets["ego_plan_residuals"].shape[-1], device=pred_code.device)

    ego_id = batch["ego_id"]
    target_id = 1 - ego_id

    plan_codes[torch.arange(B, device=pred_code.device), ego_id] = targets["ego_plan_codes"]
    plan_codes[torch.arange(B, device=pred_code.device), target_id] = pred_code

    plan_residuals[torch.arange(B, device=pred_code.device), ego_id] = targets["ego_plan_residuals"]
    plan_residuals[torch.arange(B, device=pred_code.device), target_id] = pred_residual

    wam_out = wam(current_slots, plan_codes, plan_residuals)
    loss_action = F.mse_loss(wam_out["pred_actions"], torch.cat([batch["ego_plan_actions"], batch["target_plan_actions"]], dim=-1))
    return {"loss_consistency": loss_action.detach()}


def append_log(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def evaluate(model, loader, slot_encoder, slot_norm, plan_tokenizer, plan_norm, device, amp_enabled, amp_dtype, max_batches=-1):
    model.eval()
    sums = {}
    n = 0
    code_correct = 0
    code_total = 0
    uncertainties = []

    for batch in tqdm(loader, desc="val"):
        batch = to_device(batch, device)

        targets = encode_plans(batch, plan_tokenizer, plan_norm)
        targets["ego_slots"] = encode_ego_slots(batch, slot_encoder, slot_norm)

        with amp_context(device, amp_dtype, amp_enabled):
            losses = compute_intention_losses(model, batch, targets, consistency=None)

        for k, v in losses.items():
            if k.startswith("loss") or k in ["code_acc", "residual_mse", "entropy"]:
                sums[k] = sums.get(k, 0.0) + float(v.item())

        code_correct += int((losses["pred_code"] == targets["target_plan_codes"]).sum().item())
        code_total += int(targets["target_plan_codes"].numel())
        uncertainties.append(losses["uncertainty"].detach().float().cpu())

        n += 1
        if max_batches > 0 and n >= max_batches:
            break

    metrics = {k: v / max(1, n) for k, v in sums.items()}
    metrics["code_acc_direct"] = code_correct / max(1, code_total)
    u = torch.cat(uncertainties) if uncertainties else torch.zeros(1)
    metrics["uncertainty_mean"] = float(u.mean())
    metrics["uncertainty_std"] = float(u.std())
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
    parser.add_argument("--out_dir", type=str, default="checkpoints/intention")
    parser.add_argument("--slot_ckpt", type=str, default="artifacts/slot_encoder/slot_encoder.pt")
    parser.add_argument("--plan_ckpt", type=str, default="artifacts/plan_tokenizer/plan_tokenizer.pt")
    parser.add_argument("--wam_ckpt", type=str, default="artifacts/wam/wam.pt")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--slot_dim", type=int, default=128)
    parser.add_argument("--slots_per_agent", type=int, default=4)
    parser.add_argument("--plan_codebook_size", type=int, default=32)
    parser.add_argument("--plan_latent_dim", type=int, default=64)
    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ffn_dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--residual_weight", type=float, default=1.0)
    parser.add_argument("--kl_weight", type=float, default=0.01)
    parser.add_argument("--consistency_weight", type=float, default=0.0)
    parser.add_argument("--entropy_weight", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=256)
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
    parser.add_argument("--save_every", type=int, default=10)
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
    wam, wam_cfg = load_wam(args.wam_ckpt, device)

    # Keep intention model dimensions consistent with the frozen plan tokenizer.
    if args.plan_codebook_size != plan_cfg.codebook_size:
        print(
            f"Overriding plan_codebook_size: args={args.plan_codebook_size} "
            f"-> tokenizer={plan_cfg.codebook_size}"
        )
        args.plan_codebook_size = int(plan_cfg.codebook_size)

    if args.plan_latent_dim != plan_cfg.latent_dim:
        print(
            f"Overriding plan_latent_dim: args={args.plan_latent_dim} "
            f"-> tokenizer={plan_cfg.latent_dim}"
        )
        args.plan_latent_dim = int(plan_cfg.latent_dim)

    cfg = make_config_from_args(args)
    model = IntentionInferenceModel(cfg).to(device)
    n_params = count_parameters(model)
    print("Intention parameters:", n_params, f"({n_params / 1e6:.2f}M)")

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

    train_ds = IntentionWindowDataset(args.train_dir, history=args.history, horizon=args.horizon, stride=args.stride, max_episodes=args.max_train_episodes)
    val_ds = IntentionWindowDataset(args.val_dir, history=args.history, horizon=args.horizon, stride=args.stride, max_episodes=args.max_val_episodes)

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

        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            batch = to_device(batch, device)

            with torch.no_grad():
                targets = encode_plans(batch, plan_tokenizer, plan_norm)
                targets["ego_slots"] = encode_ego_slots(batch, slot_encoder, slot_norm)

            optimizer.zero_grad(set_to_none=True)

            with amp_context(device, args.amp_dtype, amp_enabled):
                losses = compute_intention_losses(model, batch, targets, consistency=None)
                loss = losses["loss"]

            if not torch.isfinite(loss):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix(loss="nonfinite", skipped=skipped)
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

            for k, v in losses.items():
                if k.startswith("loss") or k in ["code_acc", "residual_mse", "entropy"]:
                    sums[k] = sums.get(k, 0.0) + float(v.item())
            n += 1

            pbar.set_postfix(
                loss=float(loss.item()),
                acc=float(losses["code_acc"].item()),
                res=float(losses["residual_mse"].item()),
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
