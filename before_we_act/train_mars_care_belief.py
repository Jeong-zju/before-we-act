"""Train the unchanged CARE scorer on every MARS branch family."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from before_we_act.care_belief import CAREBeliefConfig, CAREBeliefHead, care_training_loss
from before_we_act.care_training_data import CARETrainingDataset, atomic_json, load_prepared_care, sha256_file


FORMAT_VERSION = "before-we-act.care-mars-training-checkpoint/1"
PROTOCOL_UPDATES = 4000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def deterministic_batch(dataset: CARETrainingDataset, update: int, seed: int, size: int):
    generator = torch.Generator().manual_seed(seed + 1_000_003 * update)
    indices = torch.randint(len(dataset), (size,), generator=generator).tolist()
    return default_collate([dataset[index] for index in indices])


@torch.no_grad()
def evaluate(model: CAREBeliefHead, loader: DataLoader, device: torch.device, variant: str) -> dict[str, Any]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    count = selection_count = pair_correct = pair_count = 0
    task_rows: dict[int, list[float]] = defaultdict(list)
    median = model.config.quantiles.index(0.5)
    component = 0 if variant == "replay_only" else 2
    for raw in loader:
        batch = to_device(raw, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(batch["memory"], batch["memory_mask"], batch["candidate_chunks"], batch["horizon_index"])
            loss, pieces = care_training_loss(output, batch["target"], batch["hard_safety"], variant)
        scores = output.quantiles[:, :, component, median].float()
        target = batch["target"][:, :, component].float()
        selected = scores.argmax(1)
        best = target.argmax(1)
        regret = target.max(1).values - target.gather(1, selected[:, None]).squeeze(1)
        selection_count += int((selected == best).sum())
        for value, task_id in zip(regret.cpu().tolist(), batch["task_id"].cpu().tolist()):
            task_rows[int(task_id)].append(float(value))
        prediction_delta = scores[:, :, None] - scores[:, None, :]
        target_delta = target[:, :, None] - target[:, None, :]
        mask = target_delta.abs() > 1e-6
        pair_correct += int(((prediction_delta.sign() == target_delta.sign()) & mask).sum())
        pair_count += int(mask.sum())
        size = int(target.shape[0])
        count += size
        totals["loss"] += float(loss) * size
        totals["regret"] += float(regret.sum())
        totals["median_mae"] += float((scores - target).abs().mean(1).sum())
        for key, value in pieces.items():
            totals[key] += float(value) * size
    if not count:
        raise RuntimeError("empty MARS CARE diagnostic loader")
    return {
        **{key: value / count for key, value in totals.items()},
        "rows": count,
        "top1_accuracy": selection_count / count,
        "pairwise_accuracy": pair_correct / pair_count if pair_count else 0.0,
        "mean_regret_by_task_id": {
            str(key): float(np.mean(values)) for key, values in sorted(task_rows.items())
        },
    }


def atomic_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def checkpoint_payload(
    model: CAREBeliefHead,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    seed: int,
    variant: str,
    update: int,
    evaluations: list[dict[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "created_at_utc": utc_now(),
        "seed": seed,
        "variant": variant,
        "update": update,
        "config": model.config.to_dict(),
        "model": {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "evaluations": evaluations,
        "provenance": dict(provenance),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=("care", "reactive_only", "replay_only", "capacity"), required=True)
    parser.add_argument("--stage", choices=("smoke", "formal"), required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--protocol-updates", type=int, default=PROTOCOL_UPDATES)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.protocol_updates != PROTOCOL_UPDATES:
        raise ValueError("official CARE scorer protocol is fixed at 4000 updates")
    if not 1 <= args.updates <= args.protocol_updates:
        raise ValueError("invalid MARS CARE update target")
    if args.stage == "formal" and args.updates != PROTOCOL_UPDATES:
        raise ValueError("formal MARS CARE scorer requires 4000 updates")
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        expected = "COMPLETED" if args.stage == "formal" else "PASSED_SMOKE"
        if status.get("status") == expected and int(status.get("update", -1)) >= args.updates:
            print(json.dumps({"status": "PRESERVED", "output": str(args.output)}))
            return
    seed_everything(args.seed)
    device = torch.device(args.device)
    prepared = load_prepared_care(args.prepared_data)
    action_std = tuple(float(value) for value in prepared.manifest.get("action_std", (1.0,) * 8))
    if action_std == (1.0,) * 8:
        raw = torch.load(args.prepared_data, map_location="cpu", weights_only=False)
        action_std = tuple(float(value) for value in raw["action_std"])
    model = CAREBeliefHead(CAREBeliefConfig(variant=args.variant, action_std=action_std)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.5 * (1.0 + math.cos(math.pi * min(step + 1, PROTOCOL_UPDATES) / PROTOCOL_UPDATES)),
    )
    train = CARETrainingDataset(prepared, "all")
    diagnostic = CARETrainingDataset(prepared, "all", primary_horizon_only=True, primary_horizon=16)
    loader = DataLoader(diagnostic, batch_size=args.batch_size, shuffle=False, num_workers=0)
    provenance = {
        "prepared_data_sha256": sha256_file(args.prepared_data),
        "all_family_training": True,
        "family_count": len(prepared.snapshot_ids),
        "seed": args.seed,
        "variant": args.variant,
        "protocol_updates": PROTOCOL_UPDATES,
    }
    latest = args.output / "checkpoint_latest.pt"
    evaluations: list[dict[str, Any]] = []
    start = 0
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved.get("provenance") != provenance:
            raise RuntimeError("MARS CARE resume provenance drift")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        evaluations = list(saved["evaluations"])
        start = int(saved["update"])
    atomic_json(status_path, {"status": "TRAINING", "stage": args.stage, "seed": args.seed, "variant": args.variant, "update": start, "target_updates": args.updates})
    for update in range(start + 1, args.updates + 1):
        step_seed = args.seed + 10_000_019 * update
        seed_everything(step_seed)
        batch = to_device(deterministic_batch(train, update, args.seed, args.batch_size), device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(batch["memory"], batch["memory_mask"], batch["candidate_chunks"], batch["horizon_index"])
            loss, _pieces = care_training_loss(output, batch["target"], batch["hard_safety"], args.variant)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite CARE loss at {update}")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient):
            raise FloatingPointError(f"non-finite CARE gradient at {update}")
        optimizer.step()
        scheduler.step()
        should_eval = update == args.updates or (args.stage == "formal" and update % args.eval_every == 0)
        if should_eval:
            metrics = evaluate(model, loader, device, args.variant)
            evaluations.append({"update": update, "validation": metrics})
            print(json.dumps({"variant": args.variant, "seed": args.seed, "update": update, **metrics}, sort_keys=True), flush=True)
        if update == args.updates or update % args.save_every == 0:
            payload = checkpoint_payload(model, optimizer, scheduler, seed=args.seed, variant=args.variant, update=update, evaluations=evaluations, provenance=provenance)
            atomic_save(payload, latest)
            atomic_save(payload, args.output / f"checkpoint_{update:06d}.pt")
    if args.stage == "smoke":
        atomic_json(status_path, {"status": "PASSED_SMOKE", "stage": args.stage, "seed": args.seed, "variant": args.variant, "update": args.updates, "resume_start_update": start, "all_family_training": True})
        return
    selected = min(
        evaluations,
        key=lambda row: (float(row["validation"]["regret"]), float(row["validation"]["loss"]), int(row["update"])),
    )
    selected_source = args.output / f"checkpoint_{int(selected['update']):06d}.pt"
    selected_path = args.output / "selected_checkpoint.pt"
    selected_payload = torch.load(selected_source, map_location="cpu", weights_only=False)
    selected_payload["selected_validation"] = selected["validation"]
    atomic_save(selected_payload, selected_path)
    result = {
        "format_version": "before-we-act.care-mars-training-status/1",
        "status": "COMPLETED",
        "completed_at_utc": utc_now(),
        "stage": args.stage,
        "variant": args.variant,
        "seed": args.seed,
        "update": args.updates,
        "updates": args.updates,
        "train_rows": len(train),
        "diagnostic_rows": len(diagnostic),
        "all_family_training": True,
        "selected_update": int(selected["update"]),
        "selected_validation": selected["validation"],
        "selected_checkpoint": str(selected_path.resolve()),
        "selected_checkpoint_sha256": sha256_file(selected_path),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    atomic_json(status_path, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
