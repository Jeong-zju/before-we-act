"""Train one independent raw team-signal representation seed to a platform."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from before_we_act.raw_team_signal_data import (
    CAPACITY_CANDIDATES,
    BalancedTeamBatchSampler,
    RawTeamSignalDataset,
    validation_requests,
)
from before_we_act.raw_team_signal_model import RawTeamSignalEncoder, representation_losses


MAX_UPDATES = 120_000
MIN_UPDATES = 25_000
EARLIEST_PLATFORM = 35_000
EVAL_EVERY = 5_000
LR_DROP = 20_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, default=20260815)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    return parser.parse_args()


def device_batch(batch: Mapping[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items() if torch.is_tensor(value)}


def shuffle_index(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    task = batch["task_index"].detach().cpu()
    phase = batch["phase_bin"].detach().cpu()
    result = torch.arange(len(task))
    for task_index in range(6):
        task_rows = torch.where(task == task_index)[0]
        for phase_index in range(4):
            rows = torch.where((task == task_index) & (phase == phase_index))[0]
            source = rows if len(rows) > 1 else task_rows
            if len(source) > 1:
                result[rows] = torch.roll(source, 1)[: len(rows)]
    return result.to(batch["task_index"].device)


def shuffled_targets(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    permutation = shuffle_index(batch)
    result = dict(batch)
    for key in ("future_visual", "future_mask", "teammate_qpos", "teammate_delta"):
        result[key] = batch[key][permutation]
    return result


def validation_loader(dataset: RawTeamSignalDataset) -> DataLoader:
    requests = validation_requests(dataset.root)
    batches = [requests[index : index + 192] for index in range(0, len(requests), 192)]
    return DataLoader(dataset, batch_sampler=batches, num_workers=0, pin_memory=True)


def sample_components(item, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    mask = batch["future_mask"].to(item.future_visual.dtype)
    visual = (item.future_visual - batch["future_visual"]).square().mean((-1, -2))
    qpos = (item.teammate_qpos - batch["teammate_qpos"]).square().mean(-1)
    delta = (item.teammate_delta - batch["teammate_delta"]).square().mean(-1)
    per_anchor = (visual + delta) / 2
    future = (per_anchor * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    return (future + qpos) / 2, per_anchor


@torch.no_grad()
def evaluate(real, shuffled, loader, device: torch.device) -> dict:
    real.eval(); shuffled.eval()
    values: dict[str, dict[int, list[float]]] = {
        condition: {capacity: [] for capacity in CAPACITY_CANDIDATES}
        for condition in ("real", "persistence", "zero", "shuffle_model")
    }
    task_values = {condition: {capacity: {task: [] for task in range(6)} for capacity in CAPACITY_CANDIDATES} for condition in values}
    anchor_values = {condition: {capacity: [[] for _ in range(4)] for capacity in CAPACITY_CANDIDATES} for condition in values}
    collapse = {capacity: [] for capacity in CAPACITY_CANDIDATES}
    for raw in loader:
        batch = device_batch(raw, device)
        inputs = {key: batch[key] for key in RawTeamSignalDataset.RUNTIME_FIELDS}
        real_output = real(**inputs)
        shuffle_output = shuffled(**inputs)
        for capacity in CAPACITY_CANDIDATES:
            score, anchors = sample_components(real_output.capacities[capacity], batch)
            shuffle_score, _ = sample_components(shuffle_output.capacities[capacity], batch)
            persistence_visual = batch["current_target_visual"][:, None].expand_as(batch["future_visual"])
            persistence_item = type(real_output.capacities[capacity])(
                tokens=real_output.capacities[capacity].tokens,
                future_visual=persistence_visual,
                teammate_qpos=batch["previous_teammate_qpos"],
                teammate_delta=torch.zeros_like(batch["teammate_delta"]),
            )
            persistence_score, persistence_anchors = sample_components(persistence_item, batch)
            zero_item = type(real_output.capacities[capacity])(
                tokens=real_output.capacities[capacity].tokens,
                future_visual=torch.zeros_like(batch["future_visual"]),
                teammate_qpos=torch.zeros_like(batch["teammate_qpos"]),
                teammate_delta=torch.zeros_like(batch["teammate_delta"]),
            )
            zero_score, zero_anchors = sample_components(zero_item, batch)
            _, shuffle_anchors = sample_components(shuffle_output.capacities[capacity], batch)
            for condition, tensor, condition_anchors in (
                ("real", score, anchors), ("persistence", persistence_score, persistence_anchors),
                ("zero", zero_score, zero_anchors), ("shuffle_model", shuffle_score, shuffle_anchors),
            ):
                values[condition][capacity].extend(tensor.float().cpu().tolist())
                for task in range(6):
                    rows = batch["task_index"] == task
                    task_values[condition][capacity][task].extend(tensor[rows].float().cpu().tolist())
                for anchor in range(4):
                    valid = batch["future_mask"][:, anchor]
                    anchor_values[condition][capacity][anchor].extend(condition_anchors[valid, anchor].float().cpu().tolist())
            flat = real_output.capacities[capacity].tokens.reshape(-1, real.d_model).float()
            collapse[capacity].append(float(flat.std(0, unbiased=False).mean()))
    return {
        "macro": {condition: {str(capacity): float(np.mean(rows)) for capacity, rows in by_capacity.items()} for condition, by_capacity in values.items()},
        "per_task": {condition: {str(capacity): {str(task): float(np.mean(rows)) for task, rows in by_task.items()} for capacity, by_task in by_capacity.items()} for condition, by_capacity in task_values.items()},
        "per_anchor": {condition: {str(capacity): [float(np.mean(rows)) for rows in anchors] for capacity, anchors in by_capacity.items()} for condition, by_capacity in anchor_values.items()},
        "token_feature_std": {str(capacity): float(np.mean(rows)) for capacity, rows in collapse.items()},
        "rows": len(next(iter(values["real"].values()))),
    }


def plateau(metrics: list[dict]) -> bool:
    if len(metrics) < 4 or metrics[-1]["update"] < EARLIEST_PLATFORM:
        return False
    recent = metrics[-4:]
    for capacity in CAPACITY_CANDIDATES:
        scores = [row["validation"]["macro"]["real"][str(capacity)] for row in recent]
        improvements = [(a - b) / max(abs(a), 1e-12) for a, b in zip(scores, scores[1:])]
        if any(value >= 0.01 for value in improvements):
            return False
        for anchor in range(4):
            anchor_scores = [row["validation"]["per_anchor"]["real"][str(capacity)][anchor] for row in recent]
            anchor_improvements = [(a - b) / max(abs(a), 1e-12) for a, b in zip(anchor_scores, anchor_scores[1:])]
            if any(value >= 0.01 for value in anchor_improvements):
                return False
    return True


def overfit(metrics: list[dict]) -> bool:
    if len(metrics) < 4 or metrics[-1]["update"] < MIN_UPDATES:
        return False
    scores = [row["validation"]["macro"]["real"]["16"] for row in metrics[-4:]]
    return all(later > earlier for earlier, later in zip(scores, scores[1:]))


def main() -> None:
    args = parse_args()
    contract = json.loads(args.contract.read_text())
    if contract.get("status") != "FROZEN_BEFORE_F0_F1" or args.seed not in contract["seeds"]:
        raise RuntimeError("invalid or unfrozen 3-N1 contract")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(max(1, min(12, os.cpu_count() or 12)))
    random.seed(args.seed); np.random.seed(args.seed % (2**32)); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    dataset = RawTeamSignalDataset(args.cache)
    latest = args.output / "checkpoint_latest.pt"
    saved = torch.load(latest, map_location="cpu", weights_only=False) if latest.is_file() else None
    real = RawTeamSignalEncoder().to(device)
    shuffled = RawTeamSignalEncoder().to(device)
    optimizer = torch.optim.AdamW(list(real.parameters()) + list(shuffled.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 if step < LR_DROP else 0.25)
    start = 0
    metrics: list[dict] = []
    if saved:
        real.load_state_dict(saved["real_model"]); shuffled.load_state_dict(saved["shuffle_model"])
        optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
        start = int(saved["update"]); metrics = list(saved["evaluations"])
    sampler = BalancedTeamBatchSampler(dataset.episodes, updates=MAX_UPDATES, data_seed=args.data_seed, start_update=start)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0, prefetch_factor=2 if args.workers > 0 else None)
    validation = validation_loader(dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "status.json", {"status": "TRAINING", "seed": args.seed, "update": start, "started_at_utc": utc_now()})
    started = time.time()
    completion = None
    for update, raw in enumerate(loader, start=start + 1):
        step_seed = args.seed + 10_000_019 * update
        random.seed(step_seed); np.random.seed(step_seed % (2**32)); torch.manual_seed(step_seed); torch.cuda.manual_seed_all(step_seed)
        batch = device_batch(raw, device)
        inputs = {key: batch[key] for key in RawTeamSignalDataset.RUNTIME_FIELDS}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            real_output = real(**inputs)
            shuffle_output = shuffled(**inputs)
            real_losses = representation_losses(real_output, batch)
            shuffle_losses = representation_losses(shuffle_output, shuffled_targets(batch))
            loss = real_losses["total"] + shuffle_losses["total"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite N1 representation loss at {update}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(list(real.parameters()) + list(shuffled.parameters()), 1.0)
        optimizer.step(); scheduler.step()
        if update % 100 == 0:
            atomic_json(args.output / "heartbeat.json", {"status": "TRAINING", "seed": args.seed, "update": update, "loss": float(loss), "updated_at_epoch": time.time()})
        if update % EVAL_EVERY == 0:
            validation_metrics = evaluate(real, shuffled, validation, device)
            row = {"update": update, "train_real": float(real_losses["total"]), "train_shuffle": float(shuffle_losses["total"]), "validation": validation_metrics, "learning_rate": scheduler.get_last_lr()[0]}
            metrics.append(row)
            with (args.output / "evaluations.jsonl").open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            payload = {"real_model": real.state_dict(), "shuffle_model": shuffled.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "update": update, "evaluations": metrics, "seed": args.seed, "data_seed": args.data_seed, "model_config": real.config_dict(), "sample_cursor": sampler.cursor_receipt(update)}
            atomic_save(payload, latest)
            atomic_save(payload, args.output / f"checkpoint_{update:06d}.pt")
            print(json.dumps(row, sort_keys=True), flush=True)
            if overfit(metrics):
                completion = "SATURATED_BY_OVERFIT"; break
            if plateau(metrics):
                completion = "PLATFORM_REACHED"; break
            if update >= MAX_UPDATES:
                completion = "INCONCLUSIVE_TRAINING_NOT_CONVERGED"; break
    if completion is None:
        completion = "INCONCLUSIVE_TRAINING_NOT_CONVERGED"
    best = min(metrics, key=lambda row: row["validation"]["macro"]["real"]["16"])
    atomic_json(args.output / "status.json", {"status": completion, "seed": args.seed, "update": metrics[-1]["update"], "selected_update": best["update"], "selected_validation": best["validation"], "elapsed_hours": (time.time() - started) / 3600, "completed_at_utc": utc_now()})


if __name__ == "__main__":
    main()
