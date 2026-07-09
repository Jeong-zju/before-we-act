from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.wam_dataset import WAMWindowDataset
from models.wam import LatentWorldActionModel, WAMConfig, compute_wam_losses
from train.train_wam import load_slot_encoder, load_plan_tokenizer, build_wam_targets, to_device, amp_context


def load_wam(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    cfg = WAMConfig(**ckpt["config"])
    model = LatentWorldActionModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


def scalar_float(x) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu().float().item())
    return float(x)


def plot_seq(gt, pred, out_path: Path, title: str, ylabel: str, max_dims: int = 8):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gt = gt.detach().cpu().float()
    pred = pred.detach().cpu().float()
    H = gt.shape[0]
    plt.figure(figsize=(12, 6))
    if gt.ndim == 1:
        plt.plot(gt, label="gt")
        plt.plot(pred, linestyle="--", label="pred")
    else:
        D = min(gt.shape[-1], max_dims)
        for d in range(D):
            plt.plot(gt[:, d], label=f"gt_{d}")
            plt.plot(pred[:, d], linestyle="--", label=f"pred_{d}")
    plt.xlabel("future step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(ncol=4)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/wam/best.pt")
    parser.add_argument("--data_dir", type=str, default="datasets/stage2/val")
    parser.add_argument("--out_dir", type=str, default="outputs/wam")
    parser.add_argument("--slot_ckpt", type=str, default="artifacts/slot_encoder/slot_encoder.pt")
    parser.add_argument("--plan_ckpt", type=str, default="artifacts/plan_tokenizer/plan_tokenizer.pt")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "none"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, ckpt = load_wam(args.ckpt, device)
    slot_encoder, slot_cfg, slot_norm = load_slot_encoder(args.slot_ckpt, device)
    plan_tokenizer, plan_cfg, plan_norm = load_plan_tokenizer(args.plan_ckpt, device)

    ds = WAMWindowDataset(args.data_dir, history=8, horizon=cfg.horizon, stride=max(1, cfg.horizon // 2))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    amp_enabled = bool(args.amp) and args.amp_dtype != "none"

    sums = {}
    n = 0
    contact_correct = 0
    contact_total = 0
    slot_errors = []
    first = None

    with torch.no_grad():
        for batch in tqdm(dl, desc="evaluate WAM"):
            batch = to_device(batch, device)
            targets = build_wam_targets(batch, slot_encoder, slot_norm, plan_tokenizer, plan_norm, device)

            with amp_context(device, args.amp_dtype, amp_enabled):
                losses = compute_wam_losses(model, batch, targets)

            for k, v in losses.items():
                if k.startswith("loss"):
                    sums[k] = sums.get(k, 0.0) + float(v.item())

            pred_contact = (losses["pred_contact_prob"] > 0.5).float()
            contact_correct += int((pred_contact == batch["target_contact"]).sum().item())
            contact_total += int(batch["target_contact"].numel())

            se = (losses["pred_slots"] - targets["future_slots"]).pow(2).mean(dim=(0, 2, 3, 4)).detach().cpu()
            slot_errors.append(se)

            if first is None:
                first = {
                    "batch": {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in batch.items()},
                    "losses": {k: v.detach().cpu().float() if torch.is_tensor(v) and torch.is_floating_point(v) else (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in losses.items()},
                    "targets": {k: v.detach().cpu().float() if torch.is_tensor(v) and torch.is_floating_point(v) else (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in targets.items()},
                }

            n += 1
            if args.max_batches > 0 and n >= args.max_batches:
                break

    metrics = {k: v / max(1, n) for k, v in sums.items()}
    metrics["contact_acc"] = contact_correct / max(1, contact_total)

    slot_error = torch.stack(slot_errors).mean(dim=0)
    metrics["slot_error_h1"] = float(slot_error[0])
    metrics["slot_error_hmid"] = float(slot_error[len(slot_error) // 2])
    metrics["slot_error_hlast"] = float(slot_error[-1])

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    plt.figure(figsize=(8, 5))
    plt.plot(slot_error.numpy())
    plt.xlabel("future step")
    plt.ylabel("slot MSE")
    plt.title("Slot rollout error over horizon")
    plt.tight_layout()
    plt.savefig(out_dir / "slot_error_over_horizon.png")
    plt.close()

    if first is not None:
        b = first["batch"]
        l = first["losses"]
        idx = 0
        plot_seq(b["future_actions"][idx], l["pred_actions"][idx], out_dir / "action_pred_gt.png", "Action prediction", "action")
        plot_seq(b["target_progress"][idx], l["pred_progress"][idx], out_dir / "progress_pred_gt.png", "Progress prediction", "progress")
        plot_seq(b["target_force"][idx], l["pred_force"][idx], out_dir / "force_pred_gt.png", "Force prediction", "force")
        plot_seq(b["target_contact"][idx], l["pred_contact_prob"][idx], out_dir / "contact_pred_gt.png", "Contact probability", "contact")

        rows = []
        for h in range(cfg.horizon):
            row = {
                "h": h,
                "target_contact": scalar_float(b["target_contact"][idx, h]),
                "pred_contact_prob": scalar_float(l["pred_contact_prob"][idx, h]),
                "target_force": scalar_float(b["target_force"][idx, h]),
                "pred_force": scalar_float(l["pred_force"][idx, h]),
                "target_progress": scalar_float(b["target_progress"][idx, h]),
                "pred_progress": scalar_float(l["pred_progress"][idx, h]),
            }
            for d in range(cfg.action_dim):
                row[f"target_action_{d}"] = scalar_float(b["future_actions"][idx, h, d])
                row[f"pred_action_{d}"] = scalar_float(l["pred_actions"][idx, h, d])
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "sample_rollout.csv", index=False)

    print("saved outputs to:", out_dir)


if __name__ == "__main__":
    main()
