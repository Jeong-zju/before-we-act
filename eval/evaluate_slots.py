from __future__ import annotations

import argparse
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
from sklearn.decomposition import PCA

from data.slot_dataset import SlotWindowDataset
from models.slot_encoder import AgentObjectSlotEncoder, SlotEncoderConfig, compute_slot_losses


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = SlotEncoderConfig(**ckpt["config"])
    model = AgentObjectSlotEncoder(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    norm = {k: v.to(device) for k, v in ckpt["normalization"].items()}
    return model, cfg, norm, ckpt


def denorm(pred, mean, std):
    return pred * std.view(1, -1) + mean.view(1, -1)


def plot_pred_gt(gt, pred, names, out_path: Path, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(len(gt), 500)
    plt.figure(figsize=(11, 6))
    for i, name in enumerate(names):
        plt.plot(gt[:n, i], label=f"gt_{name}")
        plt.plot(pred[:n, i], linestyle="--", label=f"pred_{name}")
    plt.title(title)
    plt.xlabel("sample")
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/slot_encoder/best.pt")
    parser.add_argument("--data_dir", type=str, default="datasets/stage2/val")
    parser.add_argument("--out_dir", type=str, default="outputs/slot_encoder")
    parser.add_argument("--tokenizer_ckpt", type=str, default="artifacts/plan_tokenizer/plan_tokenizer.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, norm, ckpt = load_model(args.ckpt, device)

    ds = SlotWindowDataset(
        args.data_dir,
        history=cfg.history,
        horizon=16,
        stride=max(1, cfg.history // 2),
        tokenizer_ckpt=args.tokenizer_ckpt,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    sums = {}
    n_batches = 0

    all_self_gt = []
    all_self_pred = []
    all_other_gt = []
    all_other_pred = []
    all_object_gt = []
    all_object_pred = []
    all_phase = []
    all_phase_pred = []
    all_plan = []
    all_plan_pred = []
    all_contact = []
    all_contact_prob = []
    all_slots = []

    with torch.no_grad():
        for batch in tqdm(dl, desc="evaluate slots"):
            batch_dev = to_device(batch, device)
            losses = compute_slot_losses(model, batch_dev, norm)

            for k, v in losses.items():
                if k.startswith("loss"):
                    sums[k] = sums.get(k, 0.0) + float(v.item())

            pred_self = denorm(losses["pred_self_pose"], norm["self_pose_mean"], norm["self_pose_std"]).cpu()
            pred_other = denorm(losses["pred_other_rel_pose"], norm["other_rel_mean"], norm["other_rel_std"]).cpu()
            pred_object = denorm(losses["pred_object_rel_pose"], norm["object_rel_mean"], norm["object_rel_std"]).cpu()

            all_self_gt.append(batch["self_pose"].cpu())
            all_self_pred.append(pred_self)
            all_other_gt.append(batch["other_rel_pose"].cpu())
            all_other_pred.append(pred_other)
            all_object_gt.append(batch["object_rel_pose"].cpu())
            all_object_pred.append(pred_object)
            all_phase.append(batch["phase"].cpu())
            all_phase_pred.append(losses["pred_phase"].cpu())
            all_plan.append(batch["plan_token"].cpu())
            all_plan_pred.append(losses["pred_plan"].cpu())
            all_contact.append(batch["contact"].cpu())
            all_contact_prob.append(losses["pred_contact_prob"].cpu())
            all_slots.append(losses["slots"].cpu())

            n_batches += 1
            if args.max_batches > 0 and n_batches >= args.max_batches:
                break

    metrics = {k: v / max(1, n_batches) for k, v in sums.items()}

    self_gt = torch.cat(all_self_gt).numpy()
    self_pred = torch.cat(all_self_pred).numpy()
    other_gt = torch.cat(all_other_gt).numpy()
    other_pred = torch.cat(all_other_pred).numpy()
    object_gt = torch.cat(all_object_gt).numpy()
    object_pred = torch.cat(all_object_pred).numpy()
    phase = torch.cat(all_phase).numpy()
    phase_pred = torch.cat(all_phase_pred).numpy()
    plan = torch.cat(all_plan).numpy()
    plan_pred = torch.cat(all_plan_pred).numpy()
    contact = torch.cat(all_contact).numpy()
    contact_prob = torch.cat(all_contact_prob).numpy()
    slots = torch.cat(all_slots).numpy()

    metrics["self_pose_mae"] = float(np.mean(np.abs(self_pred - self_gt)))
    metrics["other_rel_pose_mae"] = float(np.mean(np.abs(other_pred - other_gt)))
    metrics["object_rel_pose_mae"] = float(np.mean(np.abs(object_pred - object_gt)))
    metrics["phase_acc"] = float(np.mean(phase_pred == phase))
    valid_plan = plan >= 0
    metrics["plan_acc"] = float(np.mean(plan_pred[valid_plan] == plan[valid_plan])) if valid_plan.any() else 0.0
    metrics["contact_acc"] = float(np.mean((contact_prob > 0.5).astype(np.float32) == contact))

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    plot_pred_gt(self_gt, self_pred, ["x", "y", "yaw"], out_dir / "self_pose_pred_gt.png", "Self pose prediction")
    plot_pred_gt(other_gt, other_pred, ["dx", "dy", "dyaw"], out_dir / "other_rel_pose_pred_gt.png", "Other relative pose prediction")
    plot_pred_gt(object_gt, object_pred, ["dx", "dy", "dyaw"], out_dir / "object_rel_pose_pred_gt.png", "Object relative pose prediction")

    slot_norm = np.linalg.norm(slots.reshape(-1, slots.shape[-1]), axis=-1)
    plt.figure(figsize=(8, 5))
    plt.hist(slot_norm, bins=40)
    plt.xlabel("slot norm")
    plt.ylabel("count")
    plt.title("Slot norm distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "slot_norm_hist.png")
    plt.close()

    flat_slots = slots.reshape(-1, slots.shape[-1])
    max_points = min(3000, len(flat_slots))
    pca = PCA(n_components=2)
    z = pca.fit_transform(flat_slots[:max_points])
    slot_ids = np.tile(np.arange(slots.shape[1]), slots.shape[0])[:max_points]

    plt.figure(figsize=(7, 6))
    plt.scatter(z[:, 0], z[:, 1], c=slot_ids, s=6, alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Slot embedding PCA")
    plt.tight_layout()
    plt.savefig(out_dir / "slot_pca.png")
    plt.close()

    rows = []
    for i in range(len(phase)):
        rows.append(
            {
                "phase": int(phase[i]),
                "phase_pred": int(phase_pred[i]),
                "plan_token": int(plan[i]),
                "plan_pred": int(plan_pred[i]),
                "contact": float(contact[i]),
                "contact_prob": float(contact_prob[i]),
                "self_mae": float(np.mean(np.abs(self_pred[i] - self_gt[i]))),
                "other_mae": float(np.mean(np.abs(other_pred[i] - other_gt[i]))),
                "object_mae": float(np.mean(np.abs(object_pred[i] - object_gt[i]))),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "slot_eval_samples.csv", index=False)
    print("saved outputs to:", out_dir)


if __name__ == "__main__":
    main()
