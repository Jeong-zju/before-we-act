from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.plan_dataset import PlanWindowDataset
from models.plan_tokenizer import PlanTokenizer, PlanTokenizerConfig, compute_losses, codebook_usage


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = PlanTokenizerConfig(**ckpt["config"])
    model = PlanTokenizer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    norm = {k: v.to(device) for k, v in ckpt["normalization"].items()}
    return model, cfg, norm, ckpt


def plot_reconstruction(batch, losses, norm, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    action_mean = norm["action_mean"].detach().cpu().view(1, 1, -1)
    action_std = norm["action_std"].detach().cpu().view(1, 1, -1)
    traj_mean = norm["traj_mean"].detach().cpu().view(1, 1, -1)
    traj_std = norm["traj_std"].detach().cpu().view(1, 1, -1)

    actions = batch["actions"].detach().cpu()
    traj = batch["trajectory"].detach().cpu()

    recon_actions = losses["recon_actions"].detach().cpu() * action_std + action_mean
    recon_traj = losses["recon_trajectory"].detach().cpu() * traj_std + traj_mean

    idx = 0

    plt.figure(figsize=(11, 6))
    for d in range(actions.shape[-1]):
        plt.plot(actions[idx, :, d], label=f"gt_a{d}")
        plt.plot(recon_actions[idx, :, d], linestyle="--", label=f"recon_a{d}")
    plt.xlabel("future step")
    plt.ylabel("action")
    plt.title("Future action reconstruction")
    plt.legend(ncol=4)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_action_reconstruction.png")
    plt.close()

    plt.figure(figsize=(11, 6))
    names = ["robot_x", "robot_y", "robot_yaw", "object_x", "object_y"]
    for d in range(traj.shape[-1]):
        plt.plot(traj[idx, :, d], label=f"gt_{names[d]}")
        plt.plot(recon_traj[idx, :, d], linestyle="--", label=f"recon_{names[d]}")
    plt.xlabel("future step")
    plt.ylabel("trajectory feature")
    plt.title("Future trajectory reconstruction")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_trajectory_reconstruction.png")
    plt.close()


def plot_codebook_usage(counts: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(len(counts)), counts)
    plt.xlabel("code index")
    plt.ylabel("count")
    plt.title("Codebook usage")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/plan_tokenizer/best.pt")
    parser.add_argument("--data_dir", type=str, default="datasets/stage2/val")
    parser.add_argument("--out_dir", type=str, default="outputs/plan_tokenizer")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, norm, ckpt = load_model(args.ckpt, device)

    ds = PlanWindowDataset(args.data_dir, horizon=cfg.horizon, stride=max(1, cfg.horizon // 2))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    sums = {}
    all_codes = []
    all_phase = []
    first_batch = None
    first_losses = None
    n_batches = 0

    action_mean = norm["action_mean"]
    action_std = norm["action_std"]
    traj_mean = norm["traj_mean"]
    traj_std = norm["traj_std"]

    with torch.no_grad():
        for batch in tqdm(dl, desc="evaluate"):
            batch_dev = to_device(batch, device)
            losses = compute_losses(model, batch_dev, action_mean, action_std, traj_mean, traj_std)

            for k, v in losses.items():
                if k.startswith("loss"):
                    sums[k] = sums.get(k, 0.0) + float(v.item())

            codes = losses["code_indices"].detach().cpu()
            all_codes.append(codes)
            all_phase.append(batch["phase"][:, 0].detach().cpu())

            if first_batch is None:
                first_batch = batch
                first_losses = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in losses.items()}

            n_batches += 1
            if args.max_batches > 0 and n_batches >= args.max_batches:
                break

    metrics = {k: v / max(1, n_batches) for k, v in sums.items()}

    codes = torch.cat(all_codes, dim=0)
    phases = torch.cat(all_phase, dim=0)
    usage = codebook_usage(codes, cfg.codebook_size)
    metrics.update({f"codebook_{k}": v for k, v in usage.items()})

    counts = torch.bincount(codes.reshape(-1), minlength=cfg.codebook_size).cpu().numpy()
    plot_codebook_usage(counts, out_dir / "codebook_usage.png")

    # token-phase contingency using first phase label of segment
    table = np.zeros((cfg.codebook_size, cfg.num_phases), dtype=np.int64)
    for c, p in zip(codes.numpy().reshape(-1), phases.numpy().reshape(-1)):
        if 0 <= p < cfg.num_phases:
            table[int(c), int(p)] += 1

    pd.DataFrame(table).to_csv(out_dir / "token_phase_table.csv", index_label="code")

    rows = []
    for code in range(cfg.codebook_size):
        total = int(table[code].sum())
        if total == 0:
            dominant_phase = -1
            purity = 0.0
        else:
            dominant_phase = int(table[code].argmax())
            purity = float(table[code].max() / total)
        rows.append(
            {
                "code": code,
                "count": total,
                "dominant_phase": dominant_phase,
                "phase_purity": purity,
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "token_behavior_summary.csv", index=False)

    if first_batch is not None and first_losses is not None:
        norm_cpu = {k: v.detach().cpu() for k, v in norm.items()}
        plot_reconstruction(first_batch, first_losses, norm_cpu, out_dir, prefix="val_example")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print("saved outputs to:", out_dir)


if __name__ == "__main__":
    main()
