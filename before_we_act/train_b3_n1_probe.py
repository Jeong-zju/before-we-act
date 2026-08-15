"""Train matched 3-N1 action probes on a frozen raw-signal representation."""
from __future__ import annotations

import argparse
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

from before_we_act.b3_n1_data import (
    CAPACITY_CANDIDATES,
    N1BalancedBatchSampler,
    N1RawSignalDataset,
    validation_requests,
)
from before_we_act.b3_n1_model import N1ActionProbeSet, N1RawSignalModel, masked_mse
from before_we_act.train_b3_n1_representation import (
    EARLIEST_PLATFORM,
    EVAL_EVERY,
    LR_DROP,
    MAX_UPDATES,
    MIN_UPDATES,
    atomic_json,
    atomic_save,
    device_batch,
    shuffle_index,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--representation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, default=20260815)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    return parser.parse_args()


def validation_loader(dataset: N1RawSignalDataset) -> DataLoader:
    requests = validation_requests(dataset.root)
    batches = [requests[index : index + 192] for index in range(0, len(requests), 192)]
    return DataLoader(dataset, batch_sampler=batches, num_workers=0, pin_memory=True)


def action_sample_mse(prediction: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    squared = (prediction - batch["action"]).square().mean(-1)
    mask = batch["action_mask"].to(squared.dtype)
    return (squared * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def features(representation, probes, batch):
    inputs = {key: batch[key] for key in N1RawSignalDataset.RUNTIME_FIELDS}
    with torch.no_grad():
        output = representation(**inputs)
    permutation = shuffle_index(batch)
    time = probes.time_feature(batch["task_index"], batch["phase"])
    result = {}
    for capacity in CAPACITY_CANDIDATES:
        belief = output.capacities[capacity].tokens.mean(1)
        result[capacity] = {
            "belief": belief,
            "hidden": output.history_summary,
            "time": time,
            "row_shuffle": belief[permutation],
            "phase_shuffle": belief[permutation],
        }
    return result


@torch.no_grad()
def evaluate(representation, probes, loader, device: torch.device) -> dict:
    representation.eval(); probes.eval()
    conditions = (*N1ActionProbeSet.CONDITIONS, "phase_shuffle")
    values = {condition: {capacity: [] for capacity in CAPACITY_CANDIDATES} for condition in conditions}
    tasks = {condition: {capacity: {task: [] for task in range(6)} for capacity in CAPACITY_CANDIDATES} for condition in conditions}
    episode_features: dict[int, list[tuple[int, np.ndarray]]] = {capacity: [] for capacity in CAPACITY_CANDIDATES}
    for raw in loader:
        batch = device_batch(raw, device)
        encoded = features(representation, probes, batch)
        for capacity in CAPACITY_CANDIDATES:
            for condition in conditions:
                head_condition = "belief" if condition == "phase_shuffle" else condition
                prediction = probes.forward_cell(head_condition, capacity, encoded[capacity][condition])
                score = action_sample_mse(prediction, batch)
                values[condition][capacity].extend(score.float().cpu().tolist())
                for task in range(6):
                    rows = batch["task_index"] == task
                    tasks[condition][capacity][task].extend(score[rows].float().cpu().tolist())
            for label, vector in zip(batch["episode_label"].cpu().tolist(), encoded[capacity]["belief"].float().cpu().numpy()):
                episode_features[capacity].append((label, vector))
    identity = {}
    for capacity, rows in episode_features.items():
        grouped: dict[int, list[np.ndarray]] = {}
        for label, vector in rows:
            grouped.setdefault(label, []).append(vector)
        labels = sorted(grouped)
        centroids = np.stack([np.mean(grouped[label][::2], axis=0) for label in labels])
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True).clip(1e-8)
        correct = total = 0
        for label_index, label in enumerate(labels):
            for vector in grouped[label][1::2]:
                normalized = vector / max(np.linalg.norm(vector), 1e-8)
                correct += int(np.argmax(centroids @ normalized) == label_index); total += 1
        identity[str(capacity)] = {"nearest_centroid_accuracy": correct / max(total, 1), "chance": 1 / max(len(labels), 1), "episodes": len(labels)}
    return {
        "macro": {condition: {str(capacity): float(np.mean(rows)) for capacity, rows in by_capacity.items()} for condition, by_capacity in values.items()},
        "per_task": {condition: {str(capacity): {str(task): float(np.mean(rows)) for task, rows in by_task.items()} for capacity, by_task in by_capacity.items()} for condition, by_capacity in tasks.items()},
        "episode_identity_probe": identity,
        "rows": len(next(iter(values["belief"].values()))),
    }


def platform(metrics: list[dict]) -> bool:
    if len(metrics) < 4 or metrics[-1]["update"] < EARLIEST_PLATFORM:
        return False
    recent = metrics[-4:]
    for condition in N1ActionProbeSet.CONDITIONS:
        for capacity in CAPACITY_CANDIDATES:
            scores = [row["validation"]["macro"][condition][str(capacity)] for row in recent]
            improvements = [(a - b) / max(abs(a), 1e-12) for a, b in zip(scores, scores[1:])]
            if any(value >= 0.01 for value in improvements):
                return False
    return True


def overfit(metrics: list[dict]) -> bool:
    if len(metrics) < 4 or metrics[-1]["update"] < MIN_UPDATES:
        return False
    scores = [row["validation"]["macro"]["belief"]["16"] for row in metrics[-4:]]
    return all(later > earlier for earlier, later in zip(scores, scores[1:]))


def main() -> None:
    args = parse_args()
    contract = json.loads(args.contract.read_text())
    if contract.get("status") != "FROZEN_BEFORE_F0_F1" or args.seed not in contract["seeds"]:
        raise RuntimeError("invalid or unfrozen 3-N1 contract")
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    random.seed(args.seed); np.random.seed(args.seed % (2**32)); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    dataset = N1RawSignalDataset(args.cache)
    representation_payload = torch.load(args.representation, map_location="cpu", weights_only=False)
    representation = N1RawSignalModel(**{key: value for key, value in representation_payload["model_config"].items() if key != "capacities"}).to(device)
    representation.load_state_dict(representation_payload["real_model"])
    representation.eval().requires_grad_(False)
    latest = args.output / "checkpoint_latest.pt"
    saved = torch.load(latest, map_location="cpu", weights_only=False) if latest.is_file() else None
    probes = N1ActionProbeSet().to(device)
    optimizer = torch.optim.AdamW(probes.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 if step < LR_DROP else 0.25)
    start = 0; metrics = []
    if saved:
        probes.load_state_dict(saved["probes"]); optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
        start = int(saved["update"]); metrics = list(saved["evaluations"])
    sampler = N1BalancedBatchSampler(dataset.episodes, updates=MAX_UPDATES, data_seed=args.data_seed, start_update=start)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0, prefetch_factor=2 if args.workers > 0 else None)
    validation = validation_loader(dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "status.json", {"status": "TRAINING", "seed": args.seed, "update": start, "started_at_utc": utc_now()})
    started = time.time(); completion = None
    for update, raw in enumerate(loader, start=start + 1):
        step_seed = args.seed + 10_000_019 * update
        random.seed(step_seed); np.random.seed(step_seed % (2**32)); torch.manual_seed(step_seed); torch.cuda.manual_seed_all(step_seed)
        batch = device_batch(raw, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = features(representation, probes, batch)
            losses = []
            for capacity in CAPACITY_CANDIDATES:
                for condition in N1ActionProbeSet.CONDITIONS:
                    prediction = probes.forward_cell(condition, capacity, encoded[capacity][condition])
                    losses.append(masked_mse(prediction, batch["action"], batch["action_mask"]))
            loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite N1 probe loss at {update}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(probes.parameters(), 1.0); optimizer.step(); scheduler.step()
        if update % 100 == 0:
            atomic_json(args.output / "heartbeat.json", {"status": "TRAINING", "seed": args.seed, "update": update, "loss": float(loss), "updated_at_epoch": time.time()})
        if update % EVAL_EVERY == 0:
            validation_metrics = evaluate(representation, probes, validation, device)
            row = {"update": update, "train": float(loss), "validation": validation_metrics, "learning_rate": scheduler.get_last_lr()[0]}
            metrics.append(row)
            with (args.output / "evaluations.jsonl").open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            payload = {"probes": probes.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "update": update, "evaluations": metrics, "seed": args.seed, "representation": str(args.representation), "sample_cursor": sampler.cursor_receipt(update)}
            atomic_save(payload, latest); atomic_save(payload, args.output / f"checkpoint_{update:06d}.pt")
            print(json.dumps(row, sort_keys=True), flush=True)
            if overfit(metrics): completion = "SATURATED_BY_OVERFIT"; break
            if platform(metrics): completion = "PLATFORM_REACHED"; break
            if update >= MAX_UPDATES: completion = "INCONCLUSIVE_TRAINING_NOT_CONVERGED"; break
    if completion is None: completion = "INCONCLUSIVE_TRAINING_NOT_CONVERGED"
    best = min(metrics, key=lambda row: row["validation"]["macro"]["belief"]["16"])
    atomic_json(args.output / "status.json", {"status": completion, "seed": args.seed, "update": metrics[-1]["update"], "selected_update": best["update"], "selected_validation": best["validation"], "elapsed_hours": (time.time() - started) / 3600, "completed_at_utc": utc_now()})


if __name__ == "__main__":
    main()
