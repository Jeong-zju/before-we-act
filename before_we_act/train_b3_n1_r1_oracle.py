"""Train one seed of the conditional R1-2 privileged teammate oracle."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from before_we_act.b3_n1_r1 import (
    FrozenR1Backbones,
    R1BalancedBatchSampler,
    R1OracleDataset,
    R1OracleProbeSet,
    R1_DATA_SEED,
    R1_EVAL_EVERY,
    R1_LR_DROP,
    R1_MAX_UPDATES,
    R1_MIN_UPDATES,
    R1_ORACLE_CONDITIONS,
    R1_SEEDS,
    action_sample_mse,
    all_oracle_conditions_platform,
    fixed_requests,
    load_split,
    oracle_predictions,
    split_by_episode_key,
)
from before_we_act.step2_temporal_data import sha256_file
from before_we_act.train_b3_n1_r1_fair_probe import atomic_json, atomic_save, device_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--parent-contract", type=Path, required=True)
    parser.add_argument("--oracle-contract", type=Path, required=True)
    parser.add_argument("--scenario-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def fixed_loader(dataset, split, name):
    requests = fixed_requests(dataset.episodes, split, name)
    batches = [requests[index : index + 192] for index in range(0, len(requests), 192)]
    return DataLoader(dataset, batch_sampler=batches, num_workers=0, pin_memory=True)


@torch.no_grad()
def evaluate_oracle(backbones, probes, loader, device, *, include_samples=False):
    backbones.eval(); probes.eval()
    scores = {condition: [] for condition in R1_ORACLE_CONDITIONS}
    tasks = {condition: {task: [] for task in range(6)} for condition in R1_ORACLE_CONDITIONS}
    samples = []
    for raw in loader:
        batch = device_batch(raw, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            frozen = backbones(batch)
            output = oracle_predictions(probes, frozen, batch)
        values = {
            condition: action_sample_mse(prediction.float(), batch).detach().cpu().tolist()
            for condition, prediction in output.items()
        }
        task_indices = batch["task_index"].detach().cpu().tolist()
        for condition in R1_ORACLE_CONDITIONS:
            for index, score in enumerate(values[condition]):
                scores[condition].append(float(score))
                tasks[condition][int(task_indices[index])].append(float(score))
        if include_samples:
            for index, sample_key in enumerate(raw["sample_key"]):
                samples.append(
                    {
                        "sample_key": sample_key,
                        "episode_label": int(batch["episode_label"][index]),
                        "task_index": int(task_indices[index]),
                        "scores": {condition: values[condition][index] for condition in values},
                    }
                )
    result = {
        "macro": {condition: float(np.mean(value)) for condition, value in scores.items()},
        "per_task": {
            condition: {str(task): float(np.mean(value)) for task, value in by_task.items()}
            for condition, by_task in tasks.items()
        },
        "rows": len(next(iter(scores.values()))),
    }
    if include_samples:
        result["samples"] = samples
    return result


def main() -> None:
    args = parse_args()
    parent_raw = args.parent_contract.read_bytes()
    parent = json.loads(parent_raw)
    oracle_raw = args.oracle_contract.read_bytes()
    oracle_contract = json.loads(oracle_raw)
    if oracle_contract.get("status") != "FROZEN_BEFORE_F0_F1" or args.seed not in R1_SEEDS:
        raise RuntimeError("invalid R1-2 oracle contract")
    if sha256_file(args.parent_contract) != oracle_contract["parent_contract_sha256"]:
        raise RuntimeError("R1-2 parent contract hash differs")
    split_payload = load_split(args.scenario_split)
    split = split_by_episode_key(split_payload)
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    random.seed(args.seed); np.random.seed(args.seed % 2**32); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    dataset = R1OracleDataset(args.cache)
    n1 = parent["old_n1_read_only"]["representation_checkpoints"][str(args.seed)]
    backbones = FrozenR1Backbones(
        b0h_checkpoint=Path(parent["b0h"]["checkpoint"]),
        n1_checkpoint=Path(n1["path"]),
        visual_mean=dataset.visual_mean,
        visual_std=dataset.visual_std,
    ).to(device)
    probes = R1OracleProbeSet().to(device)
    optimizer = torch.optim.AdamW(probes.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 if step < R1_LR_DROP else 0.1)
    latest = args.output / "checkpoint_latest.pt"
    saved = torch.load(latest, map_location="cpu", weights_only=False) if latest.is_file() else None
    start = 0; metrics = []
    provenance = {
        "seed": args.seed,
        "parent_contract_sha256": hashlib.sha256(parent_raw).hexdigest(),
        "oracle_contract_sha256": hashlib.sha256(oracle_raw).hexdigest(),
        "scenario_split_sha256": sha256_file(args.scenario_split),
    }
    if saved:
        if saved.get("provenance") != provenance:
            raise RuntimeError("R1-2 resume provenance differs")
        probes.load_state_dict(saved["probes"]); optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
        start = int(saved["update"]); metrics = list(saved["evaluations"])
    sampler = R1BalancedBatchSampler(dataset.episodes, split, updates=R1_MAX_UPDATES, data_seed=R1_DATA_SEED, start_update=start)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0, prefetch_factor=2 if args.workers > 0 else None)
    validation = fixed_loader(dataset, split, "validation")
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "status.json", {"status": "TRAINING", "seed": args.seed, "update": start, "started_at_utc": utc_now(), "parameter_counts": probes.parameter_counts()})
    started = time.time(); completion = None
    for update, raw in enumerate(loader, start=start + 1):
        step_seed = args.seed + 10_000_019 * update
        random.seed(step_seed); np.random.seed(step_seed % 2**32); torch.manual_seed(step_seed); torch.cuda.manual_seed_all(step_seed)
        batch = device_batch(raw, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            frozen = backbones(batch)
            output = oracle_predictions(probes, frozen, batch)
            losses = {condition: action_sample_mse(prediction.float(), batch).mean() for condition, prediction in output.items()}
            loss = torch.stack(list(losses.values())).mean()
        if not torch.isfinite(loss): raise FloatingPointError(f"non-finite R1-2 loss at {update}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(probes.parameters(), 1.0); optimizer.step(); scheduler.step()
        if update % 100 == 0:
            atomic_json(args.output / "heartbeat.json", {"status": "TRAINING", "seed": args.seed, "update": update, "loss": float(loss), "updated_at_epoch": time.time()})
        if update % R1_EVAL_EVERY: continue
        validation_metrics = evaluate_oracle(backbones, probes, validation, device)
        row = {"update": update, "train": {key: float(value.detach()) for key, value in losses.items()}, "validation": validation_metrics, "learning_rate": scheduler.get_last_lr()[0]}
        metrics.append(row)
        with (args.output / "evaluations.jsonl").open("a", encoding="utf-8") as stream: stream.write(json.dumps(row, sort_keys=True) + "\n")
        payload = {"format_version": "before-we-act.b3-n1-r1-oracle-checkpoint/1", "probes": probes.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "update": update, "evaluations": metrics, "sample_cursor": sampler.cursor_receipt(update), "provenance": provenance}
        atomic_save(payload, latest); atomic_save(payload, args.output / f"checkpoint_{update:06d}.pt")
        print(json.dumps(row, sort_keys=True), flush=True)
        if all_oracle_conditions_platform(metrics): completion = "PLATFORM_REACHED"; break
        if update >= R1_MIN_UPDATES and len(metrics) >= 4:
            scores = [float(item["validation"]["macro"]["h_oracle"]) for item in metrics[-4:]]
            if all(later > earlier for earlier, later in zip(scores, scores[1:])): completion = "SATURATED_BY_OVERFIT"; break
        if update >= R1_MAX_UPDATES: completion = "INCONCLUSIVE_TRAINING_NOT_CONVERGED"; break
    completion = completion or "INCONCLUSIVE_TRAINING_NOT_CONVERGED"
    selected = min(metrics, key=lambda item: item["validation"]["macro"]["h_oracle"])
    atomic_json(args.output / "status.json", {"status": completion, "seed": args.seed, "update": int(metrics[-1]["update"]), "selected_update": int(selected["update"]), "selected_validation": selected["validation"], "parameter_counts": probes.parameter_counts(), "elapsed_hours": (time.time()-started)/3600, "completed_at_utc": utc_now()})


if __name__ == "__main__":
    main()
