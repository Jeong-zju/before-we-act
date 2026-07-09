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
from sklearn.metrics import confusion_matrix, accuracy_score

from data.intention_dataset import IntentionWindowDataset
from models.intention import IntentionConfig, IntentionInferenceModel, compute_intention_losses
from train.train_intention import (
    load_slot_encoder,
    load_plan_tokenizer,
    encode_ego_slots,
    encode_plans,
    to_device,
    amp_context,
)


def load_model(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    cfg = IntentionConfig(**ckpt["config"])
    model = IntentionInferenceModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


def plot_confusion(y_true, y_pred, codebook_size, out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(codebook_size)))
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, aspect="auto")
    plt.colorbar()
    plt.xlabel("pred code")
    plt.ylabel("true code")
    plt.title("Teammate plan token confusion matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_active_confusion(y_true, y_pred, out_path: Path):
    active = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=active)
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(active)), active, rotation=45)
    plt.yticks(range(len(active)), active)
    plt.xlabel("pred code")
    plt.ylabel("true code")
    plt.title("Active-code teammate plan confusion matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/intention/best.pt")
    parser.add_argument("--data_dir", type=str, default="datasets/stage2/val")
    parser.add_argument("--out_dir", type=str, default="outputs/intention")
    parser.add_argument("--slot_ckpt", type=str, default="artifacts/slot_encoder/slot_encoder.pt")
    parser.add_argument("--plan_ckpt", type=str, default="artifacts/plan_tokenizer/plan_tokenizer.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=100)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "none"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, ckpt = load_model(args.ckpt, device)
    slot_encoder, slot_cfg, slot_norm = load_slot_encoder(args.slot_ckpt, device)
    plan_tokenizer, plan_cfg, plan_norm = load_plan_tokenizer(args.plan_ckpt, device)

    if cfg.plan_codebook_size != plan_cfg.codebook_size:
        raise ValueError(
            f"Checkpoint plan_codebook_size={cfg.plan_codebook_size} does not match "
            f"tokenizer codebook_size={plan_cfg.codebook_size}. "
            f"Use the same tokenizer artifact used during intention training."
        )

    ds = IntentionWindowDataset(args.data_dir, history=8, horizon=16, stride=4)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    amp_enabled = bool(args.amp) and args.amp_dtype != "none"

    sums = {}
    n = 0
    all_true = []
    all_pred = []
    all_unc = []
    all_res_mse = []
    rows = []

    with torch.no_grad():
        for batch in tqdm(dl, desc="evaluate intention"):
            batch = to_device(batch, device)
            targets = encode_plans(batch, plan_tokenizer, plan_norm)
            targets["ego_slots"] = encode_ego_slots(batch, slot_encoder, slot_norm)

            with amp_context(device, args.amp_dtype, amp_enabled):
                losses = compute_intention_losses(model, batch, targets, consistency=None)

            for k, v in losses.items():
                if k.startswith("loss") or k in ["code_acc", "residual_mse", "entropy"]:
                    sums[k] = sums.get(k, 0.0) + float(v.item())

            true = targets["target_plan_codes"].detach().cpu().numpy()
            pred = losses["pred_code"].detach().cpu().numpy()
            unc = losses["uncertainty"].detach().cpu().float().numpy()
            res_mse = ((losses["pred_residual"].detach().float() - targets["target_plan_residuals"].detach().float()) ** 2).mean(dim=-1).cpu().numpy()

            all_true.append(true)
            all_pred.append(pred)
            all_unc.append(unc)
            all_res_mse.append(res_mse)

            for i in range(len(true)):
                rows.append(
                    {
                        "true_code": int(true[i]),
                        "pred_code": int(pred[i]),
                        "correct": int(true[i] == pred[i]),
                        "uncertainty": float(unc[i]),
                        "residual_mse": float(res_mse[i]),
                        "ego_id": int(batch["ego_id"][i].detach().cpu()),
                        "phase": int(batch["phase"][i].detach().cpu()),
                    }
                )

            n += 1
            if args.max_batches > 0 and n >= args.max_batches:
                break

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    unc = np.concatenate(all_unc)
    res_mse = np.concatenate(all_res_mse)

    metrics = {k: v / max(1, n) for k, v in sums.items()}
    metrics["code_acc_direct"] = float(accuracy_score(y_true, y_pred))
    metrics["uncertainty_mean"] = float(np.mean(unc))
    metrics["uncertainty_std"] = float(np.std(unc))
    metrics["residual_mse_mean"] = float(np.mean(res_mse))

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    plot_confusion(y_true, y_pred, cfg.plan_codebook_size, out_dir / "confusion_matrix.png")
    plot_active_confusion(y_true, y_pred, out_dir / "confusion_matrix_active.png")

    plt.figure(figsize=(8, 5))
    plt.hist(unc, bins=40)
    plt.xlabel("predicted uncertainty")
    plt.ylabel("count")
    plt.title("Intention uncertainty distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "uncertainty_hist.png")
    plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "intention_eval_samples.csv", index=False)

    by_token = df.groupby("true_code").agg(
        count=("true_code", "count"),
        acc=("correct", "mean"),
        uncertainty=("uncertainty", "mean"),
        residual_mse=("residual_mse", "mean"),
    )
    by_token.to_csv(out_dir / "residual_mse_by_token.csv")

    print("saved outputs to:", out_dir)


if __name__ == "__main__":
    main()
