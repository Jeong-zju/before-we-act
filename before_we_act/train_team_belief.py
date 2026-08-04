from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from before_we_act.data.raw_team_windows import CachedTeamWindows
from before_we_act.team_belief.base import (
    PredictiveBeliefModel,
    load_r11_config,
    masked_action_mse,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def batch_from(data: dict[str, torch.Tensor], indices: torch.Tensor, device: torch.device):
    return {
        key: value.index_select(0, indices).to(device, non_blocking=True)
        for key, value in data.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = load_r11_config(args.config)
    training = config.training
    target_updates = args.updates or int(training["updates"])
    if target_updates not in (2, int(training["updates"])):
        raise ValueError("R11 supports only two-update preflight or the frozen full budget")
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal R11 training requires the assigned CUDA GPU")
    dataset = CachedTeamWindows(args.cache, "train")
    data = dict(dataset.data)
    output = Path(args.output)
    progress_path = output / "progress.jsonl"
    model = PredictiveBeliefModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    start_update = 0
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["candidate_id"] != config.candidate_id:
            raise ValueError("resume candidate identity differs")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_update = int(checkpoint["update"])
        generator.set_state(checkpoint["sample_generator_state"])
    weights = config.raw["loss_weights"]
    started = time.monotonic()
    model.train()
    last = {}
    interrupted = False
    try:
        for update in range(start_update + 1, target_updates + 1):
            indices = torch.randint(
                len(dataset),
                (int(training["batch_size"]),),
                generator=generator,
            )
            batch = batch_from(data, indices, device)
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast("cuda", dtype=torch.bfloat16)
            with context:
                prediction = model(batch)
                future_loss = (prediction["future_visual"].float() - batch["future_visual"].float()).square().mean()
                partner_loss = masked_action_mse(
                    prediction["partner_action"].float(),
                    batch["partner_action"].float(),
                    batch["agent_mask"],
                )
                progress_loss = (
                    prediction["shared_progress"].float() - batch["shared_progress"].float()
                ).square().mean()
                loss = (
                    float(weights["future_feature"]) * future_loss
                    + float(weights["partner_action"]) * partner_loss
                    + float(weights["shared_progress"]) * progress_loss
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("R11 loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = (update - start_update) / elapsed
            eta = (target_updates - update) / max(rate, 1e-9)
            last = {
                "updated_at": now(),
                "candidate_id": config.candidate_id,
                "update": update,
                "target_updates": target_updates,
                "loss": float(loss.detach()),
                "future_feature_loss": float(future_loss.detach()),
                "partner_action_loss": float(partner_loss.detach()),
                "shared_progress_loss": float(progress_loss.detach()),
                "updates_per_second": rate,
                "eta_hours": eta / 3600,
            }
            if update % int(training["progress_every"]) == 0 or update == target_updates:
                append_jsonl(progress_path, last)
            if update % int(training["checkpoint_every"]) == 0 or update == target_updates:
                payload = {
                    "schema_version": 1,
                    "round": "R11",
                    "candidate_id": config.candidate_id,
                    "config": dict(config.raw),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "update": update,
                    "sample_generator_state": generator.get_state(),
                    "train_progress_mean": float(data["shared_progress"].float().mean()),
                    "last_metrics": last,
                }
                checkpoint_dir = output / "checkpoints"
                save_checkpoint(checkpoint_dir / f"checkpoint_{update:06d}.pt", payload)
                save_checkpoint(checkpoint_dir / "checkpoint_latest.pt", payload)
    except KeyboardInterrupt:
        interrupted = True
    if interrupted:
        payload = {
            "schema_version": 1,
            "round": "R11",
            "candidate_id": config.candidate_id,
            "config": dict(config.raw),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": int(last.get("update", start_update)),
            "sample_generator_state": generator.get_state(),
            "train_progress_mean": float(data["shared_progress"].float().mean()),
            "last_metrics": last,
            "operator_stop": True,
        }
        save_checkpoint(output / "checkpoints/checkpoint_interrupted.pt", payload)
        raise SystemExit(130)
    print(json.dumps(last, sort_keys=True))


if __name__ == "__main__":
    main()
