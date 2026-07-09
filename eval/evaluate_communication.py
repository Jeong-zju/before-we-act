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
from models.communication import CommunicationTrigger, make_config_from_args
from models.free_energy import FreeEnergyEvaluator, FreeEnergyConfig
from models.intention import IntentionInferenceModel, IntentionConfig
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
    return model, cfg


def load_intention(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    cfg = IntentionConfig(**ckpt["config"])
    model = IntentionInferenceModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


def parse_active_codes(active_codes: str, codebook_size: int) -> List[int]:
    if active_codes.strip().lower() in ["", "all", "none"]:
        return list(range(codebook_size))
    out = []
    for x in active_codes.split(","):
        x = x.strip()
        if x:
            v = int(x)
            if 0 <= v < codebook_size:
                out.append(v)
    return sorted(set(out)) or list(range(codebook_size))


def generate_own_candidates(base_codes, base_residuals, ego_id: int, K: int, active_codes: List[int], residual_noise_std: float):
    B, A = base_codes.shape
    D = base_residuals.shape[-1]
    device = base_codes.device

    codes = base_codes[:, None, :].expand(B, K, A).clone()
    residuals = base_residuals[:, None, :, :].expand(B, K, A, D).clone()

    if K > 1:
        active = torch.tensor(active_codes, device=device, dtype=torch.long)
        rand_idx = torch.randint(0, len(active), (B, K - 1), device=device)
        codes[:, 1:, ego_id] = active[rand_idx]
        residuals[:, 1:, ego_id, :] = base_residuals[:, ego_id].unsqueeze(1) + torch.randn(B, K - 1, D, device=device, dtype=base_residuals.dtype) * residual_noise_std

    return codes, residuals


def flatten_for_wam(current_slots, codes, residuals):
    B, K, A = codes.shape
    slots = current_slots[:, None].expand(B, K, *current_slots.shape[1:]).reshape(B * K, *current_slots.shape[1:])
    codes_f = codes.reshape(B * K, A)
    residuals_f = residuals.reshape(B * K, A, residuals.shape[-1])
    return slots, codes_f, residuals_f


def score_candidates(wam, evaluator, current_slots, codes, residuals, uncertainty, amp_enabled, amp_dtype, device):
    B, K, A = codes.shape
    slots_f, codes_f, residuals_f = flatten_for_wam(current_slots, codes, residuals)
    unc_f = uncertainty[:, None].expand(B, K).reshape(B * K)

    with amp_context(device, amp_dtype, amp_enabled):
        rollout = wam.rollout(slots_f, codes_f, residuals_f)

    score = evaluator.total_score(rollout, uncertainty=unc_f)
    score_bk = {k: v.reshape(B, K) for k, v in score.items()}
    best_idx = score_bk["G"].argmin(dim=1)
    best_G = score_bk["G"].gather(1, best_idx[:, None]).squeeze(1)

    return score_bk, best_idx, best_G


def plot_hist(values, out_path: Path, title: str, xlabel: str, bins: int = 40):
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_trigger_scatter(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(7, 6))
    plt.scatter(df["C_comm"], df["delta_G"], c=df["trigger"], s=10, alpha=0.7)
    lim = max(float(df["C_comm"].max()), float(df["delta_G"].max()), 1e-6)
    plt.plot([0, lim], [0, lim], linestyle="--")
    plt.xlabel("communication cost C_comm")
    plt.ylabel("free-energy reduction ΔG")
    plt.title("Selective communication trigger")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="datasets/stage2/val")
    parser.add_argument("--out_dir", type=str, default="outputs/communication")
    parser.add_argument("--wam_ckpt", type=str, default="artifacts/wam/wam.pt")
    parser.add_argument("--slot_ckpt", type=str, default="artifacts/slot_encoder/slot_encoder.pt")
    parser.add_argument("--plan_ckpt", type=str, default="artifacts/plan_tokenizer/plan_tokenizer.pt")
    parser.add_argument("--intention_ckpt", type=str, default="artifacts/intention/intention.pt")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=100)
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

    parser.add_argument("--codebook_size", type=int, default=64)
    parser.add_argument("--residual_dim", type=int, default=64)
    parser.add_argument("--residual_bits", type=int, default=8)
    parser.add_argument("--envelope_bits", type=int, default=32)
    parser.add_argument("--uncertainty_bits", type=int, default=8)
    parser.add_argument("--lambda_bits", type=float, default=1e-4)
    parser.add_argument("--lambda_delay", type=float, default=0.05)
    parser.add_argument("--lambda_redundancy", type=float, default=0.1)
    parser.add_argument("--delay_steps", type=float, default=1.0)
    parser.add_argument("--delta_margin", type=float, default=0.0)
    parser.add_argument("--message_uncertainty_floor", type=float, default=0.10)
    parser.add_argument("--require_physical_gain", type=int, default=0)

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

    wam, wam_cfg = load_wam(args.wam_ckpt, device)
    intention, intention_cfg = load_intention(args.intention_ckpt, device)
    slot_encoder, slot_cfg, slot_norm = load_slot_encoder(args.slot_ckpt, device)
    plan_tokenizer, plan_cfg, plan_norm = load_plan_tokenizer(args.plan_ckpt, device)

    if args.codebook_size != plan_cfg.codebook_size:
        print(f"Overriding codebook_size: args={args.codebook_size} -> tokenizer={plan_cfg.codebook_size}")
        args.codebook_size = int(plan_cfg.codebook_size)
    if args.residual_dim != plan_cfg.latent_dim:
        print(f"Overriding residual_dim: args={args.residual_dim} -> tokenizer={plan_cfg.latent_dim}")
        args.residual_dim = int(plan_cfg.latent_dim)

    active_codes = parse_active_codes(args.active_codes, int(plan_cfg.codebook_size))
    print("active_codes:", active_codes)

    fe_cfg = FreeEnergyConfig(
        goal_y=args.goal_y,
        force_limit=args.force_limit,
        alpha_goal=args.alpha_goal,
        alpha_safety=args.alpha_safety,
        alpha_collab=args.alpha_collab,
        alpha_unc=args.alpha_unc,
        alpha_ctrl=args.alpha_ctrl,
    )
    evaluator = FreeEnergyEvaluator(fe_cfg)
    trigger = CommunicationTrigger(make_config_from_args(args))

    ds = WAMWindowDataset(args.data_dir, history=args.history, horizon=args.horizon, stride=args.stride)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    rows = []
    total = 0

    with torch.inference_mode():
        for bi, batch in enumerate(tqdm(dl, desc="communication eval")):
            batch = to_device(batch, device)
            targets = build_wam_targets(batch, slot_encoder, slot_norm, plan_tokenizer, plan_norm, device)

            B = batch["future_actions"].shape[0]
            K = args.num_candidates
            ego = int(args.ego_id)
            teammate = 1 - ego

            current_slots = targets["current_slots"]
            true_codes = targets["plan_codes"]
            true_residuals = targets["plan_residuals"]

            ego_slots = current_slots[:, ego]
            ego_plan_codes = true_codes[:, ego]
            ego_plan_residuals = true_residuals[:, ego]

            phase_hist = batch["phase_history_seq"][:, 0, ego]
            rel_target_pose = batch["rel_target_pose_agents"][:, ego]
            object_rel_pose = batch["object_rel_pose_agents"][:, ego]

            with amp_context(device, args.amp_dtype, amp_enabled):
                intent_out = intention.infer_teammate_plan(
                    ego_slots=ego_slots,
                    ego_plan_codes=ego_plan_codes,
                    ego_plan_residuals=ego_plan_residuals,
                    ego_id=torch.full((B,), ego, dtype=torch.long, device=device),
                    phase_history=phase_hist,
                    rel_target_pose=rel_target_pose,
                    object_rel_pose=object_rel_pose,
                )

            inferred_code = intent_out["target_code"].long()
            inferred_residual = intent_out["target_residual"]
            inferred_uncertainty = intent_out["uncertainty"].float()

            own_codes, own_residuals = generate_own_candidates(
                true_codes,
                true_residuals,
                ego_id=ego,
                K=K,
                active_codes=active_codes,
                residual_noise_std=args.residual_noise_std,
            )

            no_comm_codes = own_codes.clone()
            no_comm_residuals = own_residuals.clone()
            no_comm_codes[:, :, teammate] = inferred_code[:, None].expand(B, K)
            no_comm_residuals[:, :, teammate, :] = inferred_residual[:, None, :].expand(B, K, -1)

            comm_codes = own_codes.clone()
            comm_residuals = own_residuals.clone()
            comm_codes[:, :, teammate] = true_codes[:, teammate][:, None].expand(B, K)
            comm_residuals[:, :, teammate, :] = true_residuals[:, teammate][:, None, :].expand(B, K, -1)

            # Physical rollout scores exclude epistemic uncertainty.
            zero_unc = torch.zeros_like(inferred_uncertainty)
            no_score, no_idx, G_no_physical = score_candidates(
                wam, evaluator, current_slots, no_comm_codes, no_comm_residuals,
                uncertainty=zero_unc, amp_enabled=amp_enabled, amp_dtype=args.amp_dtype, device=device
            )
            comm_score, comm_idx, G_comm_physical = score_candidates(
                wam, evaluator, current_slots, comm_codes, comm_residuals,
                uncertainty=zero_unc, amp_enabled=amp_enabled, amp_dtype=args.amp_dtype, device=device
            )

            physical_gain = G_no_physical.float() - G_comm_physical.float()

            U_no = inferred_uncertainty.float()
            U_comm = torch.full_like(U_no, float(args.message_uncertainty_floor))
            info_gain = (U_no - U_comm).clamp_min(0.0)

            # Total communication benefit follows the EFE decomposition:
            # pragmatic/physical improvement + epistemic uncertainty reduction.
            total_benefit = physical_gain + args.alpha_unc * info_gain

            decision = trigger.decide(
                G_no_comm=total_benefit.float(),
                G_comm=torch.zeros_like(total_benefit).float(),
                inferred_code=inferred_code,
                message_code=true_codes[:, teammate],
            )

            G_no = G_no_physical + args.alpha_unc * U_no
            G_comm = G_comm_physical + args.alpha_unc * U_comm

            if bool(args.require_physical_gain):
                decision["trigger"] = decision["trigger"] & (physical_gain > 0)

            for b in range(B):
                rows.append({
                    "sample": total + b,
                    "batch": bi,
                    "ego_id": ego,
                    "teammate_id": teammate,
                    "G_no_comm": float(G_no[b].detach().cpu().float()),
                    "G_comm": float(G_comm[b].detach().cpu().float()),
                    "delta_G": float(decision["delta_G"][b].detach().cpu().float()),
                    "physical_gain": float(physical_gain[b].detach().cpu().float()),
                    "info_gain": float(info_gain[b].detach().cpu().float()),
                    "U_no_comm": float(U_no[b].detach().cpu().float()),
                    "U_comm": float(U_comm[b].detach().cpu().float()),
                    "C_comm": float(decision["C_comm"][b].detach().cpu().float()),
                    "trigger": int(decision["trigger"][b].detach().cpu().item()),
                    "redundancy": float(decision["redundancy"][b].detach().cpu().float()),
                    "bits": float(decision["bits"][b].detach().cpu().float()),
                    "inferred_code": int(inferred_code[b].detach().cpu()),
                    "message_code": int(true_codes[b, teammate].detach().cpu()),
                    "code_correct": int((inferred_code[b] == true_codes[b, teammate]).detach().cpu().item()),
                    "uncertainty": float(inferred_uncertainty[b].detach().cpu().float()),
                    "no_comm_selected_candidate": int(no_idx[b].detach().cpu()),
                    "comm_selected_candidate": int(comm_idx[b].detach().cpu()),
                })

            total += B

            if args.max_batches > 0 and bi + 1 >= args.max_batches:
                break

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "communication_decisions.csv", index=False)

    metrics = {
        "num_samples": int(len(df)),
        "trigger_rate": float(df["trigger"].mean()),
        "mean_delta_G": float(df["delta_G"].mean()),
        "mean_C_comm": float(df["C_comm"].mean()),
        "mean_G_no_comm": float(df["G_no_comm"].mean()),
        "mean_G_comm": float(df["G_comm"].mean()),
        "mean_gain_when_triggered": float(df[df["trigger"] == 1]["delta_G"].mean()) if (df["trigger"] == 1).any() else 0.0,
        "mean_physical_gain": float(df["physical_gain"].mean()),
        "mean_info_gain": float(df["info_gain"].mean()),
        "mean_U_no_comm": float(df["U_no_comm"].mean()),
        "mean_U_comm": float(df["U_comm"].mean()),
        "physical_gain_positive_rate": float((df["physical_gain"] > 0).mean()),
        "code_acc": float(df["code_correct"].mean()),
        "redundancy_rate": float(df["redundancy"].mean()),
        "mean_bits": float(df["bits"].mean()),
        "lambda_bits": args.lambda_bits,
        "lambda_delay": args.lambda_delay,
        "lambda_redundancy": args.lambda_redundancy,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    plot_hist(df["delta_G"], out_dir / "delta_G_hist.png", "Free-energy reduction from communication", "ΔG")
    plot_hist(df["C_comm"], out_dir / "communication_cost_hist.png", "Communication cost", "C_comm")
    plot_trigger_scatter(df, out_dir / "trigger_scatter.png")

    summary = df.groupby("trigger").agg(
        count=("trigger", "count"),
        delta_G=("delta_G", "mean"),
        C_comm=("C_comm", "mean"),
        G_no_comm=("G_no_comm", "mean"),
        G_comm=("G_comm", "mean"),
        uncertainty=("uncertainty", "mean"),
        physical_gain=("physical_gain", "mean"),
        info_gain=("info_gain", "mean"),
        U_no_comm=("U_no_comm", "mean"),
        U_comm=("U_comm", "mean"),
        code_correct=("code_correct", "mean"),
        redundancy=("redundancy", "mean"),
    )
    summary.to_csv(out_dir / "trigger_group_summary.csv")

    print("saved outputs to:", out_dir)


if __name__ == "__main__":
    main()
