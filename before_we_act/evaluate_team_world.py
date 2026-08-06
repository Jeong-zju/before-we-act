from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

import torch

from before_we_act.data.world_windows import CachedWorldWindows, legal_model_inputs
from before_we_act.world_model.base import CandidateConditionedWorldModel, load_r13_config


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def subset(data, indices, device):
    return {key: value.index_select(0, indices).to(device) for key, value in data.items()}


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    weights = mask.to(values.dtype)
    return float((values * weights).sum() / weights.sum().clamp_min(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    config = load_r13_config(args.config)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("round") != "R13" or saved.get("candidate_id") != config.candidate_id:
        raise ValueError("R13 checkpoint identity differs")
    device = torch.device(args.device)
    model = CandidateConditionedWorldModel(config).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    dataset = CachedWorldWindows(args.cache, "validation")
    data = dict(dataset.data)
    latent_sse = qpos_sse = progress_sse = 0.0
    latent_base_sse = qpos_base_sse = progress_base_sse = 0.0
    action_shuffle_delta = 0.0
    count = 0.0
    per_task = {str(index): {"samples": 0, "latent_sse": 0.0, "mask": 0.0} for index in range(5)}
    train_progress_mean = float(saved["train_progress_mean"])
    with torch.no_grad():
        for start in range(0, len(dataset), args.batch_size):
            indices = torch.arange(start, min(start + args.batch_size, len(dataset)))
            batch = subset(data, indices, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(**legal_model_inputs(batch))
                if len(indices) > 1:
                    shuffled = dict(legal_model_inputs(batch))
                    shuffled["candidate_actions"] = shuffled["candidate_actions"].roll(1, 0)
                    shuffled_prediction = model(**shuffled)
                else:
                    shuffled_prediction = prediction
            latent = prediction.latent_by_horizon[:, 0].float()
            target = batch["future_latent"].float()
            latent_error = (latent - target).square().mean(dim=(-1, -2))
            latent_base = (
                batch["current_latent"].float()[:, None, None] - target
            ).square().mean(dim=(-1, -2))
            qpos = prediction.qpos_delta_by_horizon[:, 0].float()
            qpos_error = (qpos - batch["future_qpos_delta"].float()).square().mean(dim=(-1, -2))
            qpos_base = batch["future_qpos_delta"].float().square().mean(dim=(-1, -2))
            progress = prediction.progress_by_horizon[:, 0].float()
            progress_error = (progress - batch["future_progress"].float()).square()
            progress_base = (train_progress_mean - batch["future_progress"].float()).square()
            horizon_mask = batch["horizon_mask"].float()
            latent_sse += float((latent_error * horizon_mask).sum())
            latent_base_sse += float((latent_base * horizon_mask).sum())
            qpos_sse += float((qpos_error * horizon_mask).sum())
            qpos_base_sse += float((qpos_base * horizon_mask).sum())
            progress_sse += float((progress_error * horizon_mask).sum())
            progress_base_sse += float((progress_base * horizon_mask).sum())
            count += float(horizon_mask.sum())
            shuffled_latent = shuffled_prediction.latent_by_horizon[:, 0].float()
            action_shuffle_delta += float(
                (((shuffled_latent - target).square().mean(dim=(-1, -2)) - latent_error) * horizon_mask).sum()
            )
            for row, task_index in enumerate(batch["task_index"].tolist()):
                target_row = per_task[str(int(task_index))]
                target_row["samples"] += 1
                target_row["latent_sse"] += float((latent_error[row] * horizon_mask[row]).sum())
                target_row["mask"] += float(horizon_mask[row].sum())
    timing_size = min(args.batch_size, len(dataset))
    timing_batch = subset(data, torch.arange(timing_size), device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(20):
            model(**legal_model_inputs(timing_batch))
        torch.cuda.synchronize(device)
        repeats = 200
        started = time.perf_counter()
        for _ in range(repeats):
            model(**legal_model_inputs(timing_batch))
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    throughput = repeats * timing_size / elapsed
    latent_mse = latent_sse / count
    latent_baseline = latent_base_sse / count
    qpos_mse = qpos_sse / count
    qpos_baseline = qpos_base_sse / count
    progress_mse = progress_sse / count
    progress_baseline = progress_base_sse / count
    latent_gain = max(-1.0, min(1.0, 1 - latent_mse / max(latent_baseline, 1e-12)))
    qpos_gain = max(-1.0, min(1.0, 1 - qpos_mse / max(qpos_baseline, 1e-12)))
    progress_r2 = max(-1.0, min(1.0, 1 - progress_mse / max(progress_baseline, 1e-12)))
    saturation = float(config.raw["selection_rule"]["throughput_saturation_windows_per_second"])
    throughput_score = max(0.0, min(1.0, math.log1p(throughput) / math.log1p(saturation)))
    rule = config.raw["selection_rule"]
    screen = (
        float(rule["latent_gain"]) * latent_gain
        + float(rule["qpos_gain"]) * qpos_gain
        + float(rule["progress_r2"]) * progress_r2
        + float(rule["throughput"]) * throughput_score
    )
    for values in per_task.values():
        values["latent_mse"] = values.pop("latent_sse") / max(values.pop("mask"), 1)
    aggregate = {
        "latent_mse": latent_mse,
        "latent_persistence_mse": latent_baseline,
        "latent_gain": latent_gain,
        "qpos_delta_mse": qpos_mse,
        "qpos_zero_delta_mse": qpos_baseline,
        "qpos_gain": qpos_gain,
        "progress_mse": progress_mse,
        "progress_mean_baseline_mse": progress_baseline,
        "progress_r2": progress_r2,
        "action_shuffle_latent_mse_delta": action_shuffle_delta / count,
        "windows_per_second": throughput,
        "throughput_score": throughput_score,
        "world_screen_score": screen,
    }
    result = {
        "schema_version": 1,
        "round": "R13",
        "candidate_id": config.candidate_id,
        "created_at": now(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_update": int(saved["update"]),
        "aggregate": aggregate,
        "per_task_by_index": per_task,
        "selection_rule": dict(rule),
        "quality_threshold": None,
        "optional_diagnostics": {
            "pair_accuracy": None,
            "spearman": None,
            "auroc": None,
            "ece": None,
            "oracle_retention": None,
            "reason": "shared success-demonstration cache has no counterfactual branch outcomes or failure class",
        },
        "gate20": "N/A (action hash equal)",
        "failure_head_claim": "trained on success demonstrations only; no AUROC claim",
    }
    atomic_json(Path(args.output), result)
    print(json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
