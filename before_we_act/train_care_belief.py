"""Three-seed/matched-scorer trainer for CARE/RoboFactory reproduction."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader

from before_we_act.care_belief import (
    CAREBeliefConfig,
    CAREBeliefHead,
    care_training_loss,
)
from before_we_act.care_training_data import (
    CARETrainingDataset,
    atomic_json,
    load_prepared_care,
    sha256_file,
)


FORMAT_VERSION = "before-we-act.care-robofactory-training-checkpoint/1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infinite(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        yield from loader


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate(
    model: CAREBeliefHead,
    loader: DataLoader,
    device: torch.device,
    variant: str,
) -> dict[str, Any]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    count = 0
    selection_count = 0
    pair_correct = pair_count = 0
    task_rows: dict[int, list[float]] = defaultdict(list)
    median_index = model.config.quantiles.index(0.5)
    component = 0 if variant == "replay_only" else 2
    for raw in loader:
        batch = to_device(raw, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(
                batch["memory"],
                batch["memory_mask"],
                batch["candidate_chunks"],
                batch["horizon_index"],
            )
            loss, pieces = care_training_loss(
                output, batch["target"], batch["hard_safety"], variant
            )
        scores = output.quantiles[:, :, component, median_index].float()
        target = batch["target"][:, :, component].float()
        selected = scores.argmax(1)
        best = target.argmax(1)
        regret = target.max(1).values - target.gather(1, selected[:, None]).squeeze(1)
        selection_count += int((selected == best).sum().item())
        for row_regret, task_id in zip(regret.cpu().tolist(), batch["task_id"].cpu().tolist()):
            task_rows[int(task_id)].append(float(row_regret))
        delta_prediction = scores[:, :, None] - scores[:, None, :]
        delta_target = target[:, :, None] - target[:, None, :]
        mask = delta_target.abs() > 1e-6
        pair_correct += int(((delta_prediction.sign() == delta_target.sign()) & mask).sum().item())
        pair_count += int(mask.sum().item())
        size = int(target.shape[0])
        count += size
        totals["loss"] += float(loss) * size
        totals["regret"] += float(regret.sum())
        totals["median_mae"] += float((scores - target).abs().mean(1).sum())
        if variant == "care":
            response = output.response[:, :, median_index].float()
            totals["response_mae"] += float(
                (response - batch["target"][:, :, 1]).abs().mean(1).sum()
            )
        for key, value in pieces.items():
            totals[key] += float(value) * size
    if not count:
        raise RuntimeError("CARE evaluation loader is empty")
    result = {key: value / count for key, value in totals.items()}
    result.update(
        {
            "rows": count,
            "top1_accuracy": selection_count / count,
            "pairwise_accuracy": pair_correct / pair_count if pair_count else 0.0,
            "mean_regret_by_task_id": {
                str(key): float(np.mean(values)) for key, values in sorted(task_rows.items())
            },
        }
    )
    return result


def save_checkpoint(
    path: Path,
    model: CAREBeliefHead,
    *,
    seed: int,
    update: int,
    metrics: dict[str, Any],
    prepared_path: Path,
) -> None:
    value = {
        "format_version": FORMAT_VERSION,
        "created_at_utc": utc_now(),
        "seed": seed,
        "update": update,
        "config": model.config.to_dict(),
        "model": {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
        "validation": metrics,
        "prepared_data": str(prepared_path.resolve()),
        "prepared_data_sha256": sha256_file(prepared_path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--variant",
        choices=("care", "reactive_only", "replay_only", "capacity"),
        required=True,
    )
    parser.add_argument("--updates", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.updates < args.eval_every or args.updates % args.eval_every:
        raise ValueError("CARE updates must be a positive multiple of eval-every")
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "COMPLETED":
            print(json.dumps({"status": "PRESERVED", "output": str(args.output)}))
            return

    seed_everything(args.seed)
    device = torch.device(args.device)
    prepared = load_prepared_care(args.prepared_data)
    action_std = tuple(float(value) for value in prepared.manifest.get("action_std", (1.0,) * 8))
    # Older prepared manifests keep action scale in the training contract.
    if action_std == (1.0,) * 8:
        raw = torch.load(args.prepared_data, map_location="cpu", weights_only=False)
        action_std = tuple(float(value) for value in raw.get("action_std", action_std))
    config = CAREBeliefConfig(variant=args.variant, action_std=action_std)
    model = CAREBeliefHead(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train = CARETrainingDataset(prepared, "train")
    validation = CARETrainingDataset(
        prepared, "validation", primary_horizon_only=True, primary_horizon=16
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    batches = infinite(train_loader)
    curve: list[dict[str, Any]] = []
    best_key = (float("inf"), float("inf"))
    best_update = 0
    best_path = args.output / "selected_checkpoint.pt"
    for update in range(1, args.updates + 1):
        model.train()
        batch = to_device(next(batches), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(
                batch["memory"],
                batch["memory_mask"],
                batch["candidate_chunks"],
                batch["horizon_index"],
            )
            loss, _pieces = care_training_loss(
                output, batch["target"], batch["hard_safety"], args.variant
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite CARE loss at update {update}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % args.eval_every:
            continue
        metrics = evaluate(model, validation_loader, device, args.variant)
        row = {"update": update, **metrics}
        curve.append(row)
        key = (float(metrics["regret"]), float(metrics["loss"]))
        if key < best_key:
            best_key = key
            best_update = update
            save_checkpoint(
                best_path,
                model,
                seed=args.seed,
                update=update,
                metrics=metrics,
                prepared_path=args.prepared_data,
            )
        print(json.dumps({"variant": args.variant, "seed": args.seed, **row}, sort_keys=True), flush=True)

    selected = torch.load(best_path, map_location="cpu", weights_only=False)
    result = {
        "format_version": "before-we-act.care-robofactory-training-status/1",
        "status": "COMPLETED",
        "completed_at_utc": utc_now(),
        "variant": args.variant,
        "seed": args.seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "selected_update": best_update,
        "selected_validation": selected["validation"],
        "selected_checkpoint": str(best_path.resolve()),
        "selected_checkpoint_sha256": sha256_file(best_path),
        "curve": curve,
    }
    atomic_json(status_path, result)
    print(json.dumps({"status": "COMPLETED", "variant": args.variant, "seed": args.seed, "best_update": best_update}))


if __name__ == "__main__":
    main()
