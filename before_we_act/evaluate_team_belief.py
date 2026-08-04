from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

import torch

from before_we_act.data.raw_team_windows import CachedTeamWindows, TASKS
from before_we_act.team_belief.base import PredictiveBeliefModel, load_r11_config


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def subset(data, indices, device):
    return {key: value.index_select(0, indices).to(device) for key, value in data.items()}


def masked_sum(error, mask):
    weights = mask.to(error.dtype)
    return float((error * weights).sum()), float(weights.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    config = load_r11_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["candidate_id"] != config.candidate_id:
        raise ValueError("checkpoint candidate identity differs")
    device = torch.device(args.device)
    model = PredictiveBeliefModel(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset = CachedTeamWindows(args.cache, "validation")
    data = dict(dataset.data)
    accumulators = {
        task: {"n": 0, "future_sse": 0.0, "future_base_sse": 0.0, "action_sse": 0.0, "action_base_sse": 0.0, "action_count": 0.0, "progress_sse": 0.0, "progress_base_sse": 0.0}
        for task in TASKS
    }
    with torch.no_grad():
        for start in range(0, len(dataset), args.batch_size):
            indices = torch.arange(start, min(start + args.batch_size, len(dataset)))
            batch = subset(data, indices, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(batch)
            future_error = (prediction["future_visual"].float() - batch["future_visual"].float()).square().mean(dim=(1, 2))
            future_base = (batch["visual"][:, -1].float() - batch["future_visual"].float()).square().mean(dim=(1, 2))
            action_error = (prediction["partner_action"].float() - batch["partner_action"].float()).square().mean(dim=-1)
            action_base = (batch["actions"][:, -1].float() - batch["partner_action"].float()).square().mean(dim=-1)
            progress_error = (prediction["shared_progress"].float() - batch["shared_progress"].float()).square()
            progress_base = (float(checkpoint["train_progress_mean"]) - batch["shared_progress"].float()).square()
            for row, task_index in enumerate(batch["task_index"].tolist()):
                target = accumulators[TASKS[int(task_index)]]
                target["n"] += 1
                target["future_sse"] += float(future_error[row])
                target["future_base_sse"] += float(future_base[row])
                masked, count = masked_sum(action_error[row], batch["agent_mask"][row])
                masked_base, _ = masked_sum(action_base[row], batch["agent_mask"][row])
                target["action_sse"] += masked
                target["action_base_sse"] += masked_base
                target["action_count"] += count
                target["progress_sse"] += float(progress_error[row])
                target["progress_base_sse"] += float(progress_base[row])
    metrics = {}
    totals = {key: 0.0 for key in ("n", "future_sse", "future_base_sse", "action_sse", "action_base_sse", "action_count", "progress_sse", "progress_base_sse")}
    for task, values in accumulators.items():
        for key in totals:
            totals[key] += values[key]
        metrics[task] = {
            "samples": values["n"],
            "future_feature_mse": values["future_sse"] / values["n"],
            "future_persistence_mse": values["future_base_sse"] / values["n"],
            "partner_action_mse": values["action_sse"] / values["action_count"],
            "partner_last_action_mse": values["action_base_sse"] / values["action_count"],
            "shared_progress_mse": values["progress_sse"] / values["n"],
            "shared_progress_mean_baseline_mse": values["progress_base_sse"] / values["n"],
        }
    batch_indices = torch.arange(min(args.batch_size, len(dataset)))
    timing_batch = subset(data, batch_indices, device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(20):
            model(timing_batch)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        timing_repeats = 200
        for _ in range(timing_repeats):
            model(timing_batch)
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    windows_per_second = timing_repeats * len(batch_indices) / elapsed
    future_mse = totals["future_sse"] / totals["n"]
    future_baseline = totals["future_base_sse"] / totals["n"]
    action_mse = totals["action_sse"] / totals["action_count"]
    action_baseline = totals["action_base_sse"] / totals["action_count"]
    progress_mse = totals["progress_sse"] / totals["n"]
    progress_baseline = totals["progress_base_sse"] / totals["n"]
    future_gain = max(-1.0, min(1.0, 1.0 - future_mse / max(future_baseline, 1e-12)))
    action_gain = max(-1.0, min(1.0, 1.0 - action_mse / max(action_baseline, 1e-12)))
    progress_r2 = max(-1.0, min(1.0, 1.0 - progress_mse / max(progress_baseline, 1e-12)))
    saturation = float(config.raw["selection_rule"]["throughput_saturation_windows_per_second"])
    throughput_score = max(0.0, min(1.0, math.log1p(windows_per_second) / math.log1p(saturation)))
    weights = config.raw["selection_rule"]
    screen_score = (
        float(weights["future_feature_gain"]) * future_gain
        + float(weights["partner_action_gain"]) * action_gain
        + float(weights["shared_progress_r2"]) * progress_r2
        + float(weights["throughput"]) * throughput_score
    )
    result = {
        "schema_version": 1,
        "round": "R11",
        "candidate_id": config.candidate_id,
        "created_at": now(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_update": checkpoint["update"],
        "per_task": metrics,
        "aggregate": {
            "future_feature_mse": future_mse,
            "future_persistence_mse": future_baseline,
            "future_feature_gain": future_gain,
            "partner_action_mse": action_mse,
            "partner_last_action_mse": action_baseline,
            "partner_action_gain": action_gain,
            "shared_progress_mse": progress_mse,
            "shared_progress_mean_baseline_mse": progress_baseline,
            "shared_progress_r2": progress_r2,
            "windows_per_second": windows_per_second,
            "throughput_score": throughput_score,
            "representation_screen_score": screen_score,
        },
        "selection_rule": dict(weights),
        "gate20": "N/A (action hash equal)",
        "quality_threshold": None,
    }
    atomic_json(Path(args.output), result)
    print(json.dumps(result["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
