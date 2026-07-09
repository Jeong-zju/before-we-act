from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.wam_dataset import WAMWindowDataset
from models.free_energy import FreeEnergyEvaluator, make_config_from_args
from models.wam import LatentWorldActionModel, WAMConfig
from train.train_wam import load_slot_encoder, load_plan_tokenizer, build_wam_targets, to_device, amp_context


def load_wam(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    cfg = WAMConfig(**ckpt["config"])
    model = LatentWorldActionModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, ckpt


def parse_active_codes(active_codes: str, codebook_size: int) -> List[int]:
    if active_codes.strip().lower() in ["", "all", "none"]:
        return list(range(codebook_size))
    codes = []
    for x in active_codes.split(","):
        x = x.strip()
        if not x:
            continue
        v = int(x)
        if 0 <= v < codebook_size:
            codes.append(v)
    return sorted(set(codes)) or list(range(codebook_size))


def generate_candidate_plans(
    base_codes: torch.Tensor,
    base_residuals: torch.Tensor,
    ego_id: int,
    num_candidates: int,
    active_codes: List[int],
    residual_noise_std: float,
) -> Dict[str, torch.Tensor]:
    B, A = base_codes.shape
    device = base_codes.device
    D = base_residuals.shape[-1]

    cand_codes = base_codes[:, None, :].expand(B, num_candidates, A).clone()
    cand_residuals = base_residuals[:, None, :, :].expand(B, num_candidates, A, D).clone()

    # Candidate 0 is the ground-truth plan. Other candidates perturb ego plan only.
    if num_candidates > 1:
        active = torch.tensor(active_codes, device=device, dtype=torch.long)
        rand_idx = torch.randint(0, len(active), (B, num_candidates - 1), device=device)
        sampled_codes = active[rand_idx]
        noise = torch.randn(B, num_candidates - 1, D, device=device, dtype=base_residuals.dtype) * residual_noise_std

        cand_codes[:, 1:, ego_id] = sampled_codes
        cand_residuals[:, 1:, ego_id, :] = base_residuals[:, ego_id].unsqueeze(1) + noise

    return {
        "plan_codes": cand_codes,
        "plan_residuals": cand_residuals,
    }


def flatten_candidates(current_slots, cand_codes, cand_residuals):
    B, K, A = cand_codes.shape
    slots = current_slots[:, None].expand(B, K, *current_slots.shape[1:]).reshape(B * K, *current_slots.shape[1:])
    codes = cand_codes.reshape(B * K, A)
    residuals = cand_residuals.reshape(B * K, A, cand_residuals.shape[-1])
    return slots, codes, residuals


def unflatten_scores(score_dict: Dict[str, torch.Tensor], B: int, K: int) -> Dict[str, torch.Tensor]:
    return {k: v.reshape(B, K) for k, v in score_dict.items()}


def plot_score_components(df: pd.DataFrame, out_path: Path):
    means = df.groupby("candidate")[["G", "L_goal", "L_safety", "L_collab", "U_intent", "C_ctrl"]].mean()
    plt.figure(figsize=(10, 6))
    for col in means.columns:
        plt.plot(means.index, means[col], marker="o", label=col)
    plt.xlabel("candidate index")
    plt.ylabel("mean score component")
    plt.title("Free-energy component breakdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_selected_hist(selected, out_path: Path):
    plt.figure(figsize=(8, 5))
    plt.hist(selected, bins=max(1, int(selected.max()) + 1))
    plt.xlabel("selected candidate")
    plt.ylabel("count")
    plt.title("Selected candidate histogram")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="datasets/stage2/val")
    parser.add_argument("--out_dir", type=str, default="outputs/free_energy")
    parser.add_argument("--wam_ckpt", type=str, default="artifacts/wam/wam.pt")
    parser.add_argument("--slot_ckpt", type=str, default="artifacts/slot_encoder/slot_encoder.pt")
    parser.add_argument("--plan_ckpt", type=str, default="artifacts/plan_tokenizer/plan_tokenizer.pt")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--num_candidates", type=int, default=8)
    parser.add_argument("--ego_id", type=int, default=0)
    parser.add_argument("--active_codes", type=str, default="2,3,6,24,32,44,51")
    parser.add_argument("--residual_noise_std", type=float, default=0.5)
    parser.add_argument("--goal_y", type=float, default=3.05)
    parser.add_argument("--force_limit", type=float, default=1.0)
    parser.add_argument("--alpha_goal", type=float, default=1.0)
    parser.add_argument("--alpha_safety", type=float, default=2.0)
    parser.add_argument("--alpha_collab", type=float, default=1.0)
    parser.add_argument("--alpha_unc", type=float, default=0.5)
    parser.add_argument("--alpha_ctrl", type=float, default=0.05)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "none"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    amp_enabled = bool(args.amp) and args.amp_dtype != "none"

    wam, wam_cfg, wam_ckpt = load_wam(args.wam_ckpt, device)
    slot_encoder, slot_cfg, slot_norm = load_slot_encoder(args.slot_ckpt, device)
    plan_tokenizer, plan_cfg, plan_norm = load_plan_tokenizer(args.plan_ckpt, device)

    active_codes = parse_active_codes(args.active_codes, int(plan_cfg.codebook_size))
    print("active_codes:", active_codes)

    ds = WAMWindowDataset(args.data_dir, history=args.history, horizon=args.horizon, stride=args.stride)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    evaluator = FreeEnergyEvaluator(make_config_from_args(args))

    rows = []
    selected_all = []
    gt_selected = 0
    total_samples = 0

    with torch.inference_mode():
        for bi, batch in enumerate(tqdm(dl, desc="free-energy eval")):
            batch = to_device(batch, device)
            targets = build_wam_targets(batch, slot_encoder, slot_norm, plan_tokenizer, plan_norm, device)

            B = batch["future_actions"].shape[0]
            K = args.num_candidates

            cand = generate_candidate_plans(
                targets["plan_codes"],
                targets["plan_residuals"],
                ego_id=args.ego_id,
                num_candidates=K,
                active_codes=active_codes,
                residual_noise_std=args.residual_noise_std,
            )

            slots_flat, codes_flat, residuals_flat = flatten_candidates(
                targets["current_slots"],
                cand["plan_codes"],
                cand["plan_residuals"],
            )

            with amp_context(device, args.amp_dtype, amp_enabled):
                rollout = wam.rollout(slots_flat, codes_flat, residuals_flat)

            # First version uses a simple uncertainty proxy: residual perturbation magnitude.
            unc = (residuals_flat[:, args.ego_id] - targets["plan_residuals"][:, args.ego_id].repeat_interleave(K, dim=0)).pow(2).mean(dim=-1)

            score = evaluator.total_score(rollout, uncertainty=unc)
            score_bk = unflatten_scores(score, B, K)

            selected = score_bk["G"].argmin(dim=1)
            selected_all.append(selected.detach().cpu())
            gt_selected += int((selected == 0).sum().item())
            total_samples += B

            for b in range(B):
                for k in range(K):
                    row = {
                        "batch": bi,
                        "sample": total_samples - B + b,
                        "candidate": k,
                        "selected": int(selected[b].item() == k),
                        "ego_id": args.ego_id,
                        "ego_code": int(cand["plan_codes"][b, k, args.ego_id].detach().cpu().item()),
                    }
                    for name in ["G", "L_goal", "L_safety", "L_collab", "U_intent", "C_ctrl"]:
                        row[name] = float(score_bk[name][b, k].detach().cpu().float().item())
                    rows.append(row)

            if args.max_batches > 0 and bi + 1 >= args.max_batches:
                break

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "candidate_scores.csv", index=False)

    selected_cat = torch.cat(selected_all).numpy() if selected_all else torch.zeros(1).numpy()
    metrics = {
        "num_samples": int(total_samples),
        "num_candidates": int(args.num_candidates),
        "gt_selected_rate": float(gt_selected / max(1, total_samples)),
        "mean_selected_G": float(df[df["selected"] == 1]["G"].mean()),
        "mean_gt_G": float(df[df["candidate"] == 0]["G"].mean()),
        "mean_all_G": float(df["G"].mean()),
        "alpha_goal": args.alpha_goal,
        "alpha_safety": args.alpha_safety,
        "alpha_collab": args.alpha_collab,
        "alpha_unc": args.alpha_unc,
        "alpha_ctrl": args.alpha_ctrl,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    plot_score_components(df, out_dir / "score_components.png")
    plot_selected_hist(selected_cat, out_dir / "selected_candidate_hist.png")

    selected_df = df[df["selected"] == 1].copy()
    selected_df.to_csv(out_dir / "selected_candidates.csv", index=False)

    print("saved outputs to:", out_dir)


if __name__ == "__main__":
    main()
