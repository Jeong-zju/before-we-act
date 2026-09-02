"""Train the protocol-isolated CARE scorer-v2 on MARS branch families.

This entry point never resumes or writes a legacy CARE scorer directory.  It
keeps the 4,000-update budget and all-family training recipe while exposing the
executed action prefix and fixed robust utility scaling in checkpoint
provenance.  Its same-corpus metrics are diagnostics only; deployment
selection/calibration must use an independent or out-of-fold gate.
"""
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

from before_we_act.care_belief_v2 import (
    CAREBeliefV2Config,
    CAREBeliefV2Head,
    CARELossV2Config,
    care_v2_training_loss,
    robust_task_component_scales,
)
from before_we_act.care_training_data import (
    CARETrainingDataset,
    atomic_json,
    load_prepared_care,
    sha256_file,
)


FORMAT_VERSION = "before-we-act.care-mars-training-checkpoint-v2/1"
STATUS_FORMAT_VERSION = "before-we-act.care-mars-training-status-v2/1"
PROTOCOL_UPDATES = 4000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def deterministic_batch(
    dataset: CARETrainingDataset, update: int, seed: int, size: int
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + 1_000_003 * update)
    indices = torch.randint(len(dataset), (size,), generator=generator).tolist()
    return default_collate([dataset[index] for index in indices])


def hard_safety_nonzero_count(dataset: CARETrainingDataset) -> int:
    """Count non-reference hard-safety labels in a prepared row dataset."""

    return sum(
        int(torch.count_nonzero(dataset[index]["hard_safety"][1:]))
        for index in range(len(dataset))
    )


def prepared_intervention_steps(manifest: Mapping[str, Any]) -> int:
    """Resolve the branch execution window, defaulting legacy MARS data to 1."""

    candidates = (
        manifest.get("intervention_steps"),
        manifest.get("executed_intervention_steps"),
        manifest.get("branch_intervention_steps"),
    )
    present = [int(value) for value in candidates if value is not None]
    if len(set(present)) > 1:
        raise ValueError("CARE v2 prepared intervention metadata disagrees")
    value = present[0] if present else 1
    if value not in {1, 4, 8, 16}:
        raise ValueError(f"unsupported CARE v2 intervention steps: {value}")
    return value


@torch.no_grad()
def evaluate(
    model: CAREBeliefV2Head,
    loader: DataLoader,
    device: torch.device,
    variant: str,
    task_component_scales: torch.Tensor,
    loss_config: CARELossV2Config,
    safety_supervision_nonzero_count: int,
) -> dict[str, Any]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    count = top1 = pair_correct = pair_count = 0
    overrides = harmful = beneficial = sign_correct = sign_count = 0
    q05_positive = q05_count = 0
    task_regret: dict[int, list[float]] = defaultdict(list)
    task_override: dict[int, list[int]] = defaultdict(list)
    median = model.config.quantiles.index(0.5)
    component = 0 if variant == "replay_only" else 2
    for raw in loader:
        batch = to_device(raw, device)
        scale = task_component_scales.index_select(0, batch["task_id"])
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model(
                batch["memory"],
                batch["memory_mask"],
                batch["candidate_chunks"],
                batch["horizon_index"],
                utility_scale=scale,
            )
            loss, pieces = care_v2_training_loss(
                output,
                batch["target"],
                batch["hard_safety"],
                variant,
                target_scale=scale,
                loss_config=loss_config,
                quantiles=model.config.quantiles,
            )
        scores = output.quantiles[:, :, component, median].float()
        lower = output.quantiles[:, :, component, 0].float()
        target = batch["target"][:, :, component].float()
        selected = scores.argmax(1)
        best = target.argmax(1)
        regret = target.max(1).values - target.gather(1, selected[:, None]).squeeze(1)
        top1 += int((selected == best).sum())
        prediction_delta = scores[:, :, None] - scores[:, None, :]
        target_delta = target[:, :, None] - target[:, None, :]
        upper = torch.triu(
            torch.ones(
                scores.shape[1], scores.shape[1], dtype=torch.bool, device=device
            ),
            diagonal=1,
        )
        mask = (target_delta.abs() > 1e-6) & upper.unsqueeze(0)
        pair_correct += int(((prediction_delta.sign() == target_delta.sign()) & mask).sum())
        pair_count += int(mask.sum())

        proposal = lower.clone()
        proposal[:, 0] = 0.0
        lower_selected = proposal.argmax(1)
        override = lower_selected != 0
        selected_target = target.gather(1, lower_selected[:, None]).squeeze(1)
        overrides += int(override.sum())
        harmful += int((override & (selected_target < 0)).sum())
        beneficial += int((override & (selected_target > 0)).sum())
        q05_positive += int((lower[:, 1:] > 0).sum())
        q05_count += int(lower[:, 1:].numel())
        relative_target = target[:, 1:] - target[:, :1]
        relative_score = scores[:, 1:] - scores[:, :1]
        sign_mask = relative_target.abs() > 1e-6
        sign_correct += int(
            ((relative_score.sign() == relative_target.sign()) & sign_mask).sum()
        )
        sign_count += int(sign_mask.sum())

        size = int(target.shape[0])
        count += size
        totals["loss"] += float(loss) * size
        totals["regret"] += float(regret.sum())
        totals["median_mae"] += float((scores - target).abs().mean(1).sum())
        totals["q05_mean"] += float(lower[:, 1:].mean(1).sum())
        for key, value in pieces.items():
            totals[key] += float(value) * size
        for value, task_id, was_override in zip(
            regret.cpu().tolist(),
            batch["task_id"].cpu().tolist(),
            override.cpu().tolist(),
        ):
            task_regret[int(task_id)].append(float(value))
            task_override[int(task_id)].append(int(was_override))
    if not count:
        raise RuntimeError("empty MARS CARE v2 diagnostic loader")
    return {
        **{key: value / count for key, value in totals.items()},
        "rows": count,
        "top1_accuracy": top1 / count,
        "pairwise_accuracy_including_reference": (
            pair_correct / pair_count if pair_count else 0.0
        ),
        "candidate_vs_reference_sign_accuracy": (
            sign_correct / sign_count if sign_count else 0.0
        ),
        "uncalibrated_q05_positive_rate": q05_positive / max(q05_count, 1),
        "uncalibrated_override_rate": overrides / count,
        "uncalibrated_harmful_override_rate": harmful / max(overrides, 1),
        "uncalibrated_beneficial_override_rate": beneficial / max(overrides, 1),
        "mean_regret_by_task_id": {
            str(key): float(np.mean(values))
            for key, values in sorted(task_regret.items())
        },
        "uncalibrated_override_rate_by_task_id": {
            str(key): float(np.mean(values))
            for key, values in sorted(task_override.items())
        },
        "safety_supervision_nonzero_count": int(
            safety_supervision_nonzero_count
        ),
        "safety_supervision_degenerate": safety_supervision_nonzero_count == 0,
        "safety_gate_mode": (
            "legality_only"
            if safety_supervision_nonzero_count == 0
            else "not_applied_uncalibrated_diagnostic"
        ),
        "learned_safety_mask_applied": False,
        "same_corpus_diagnostic_only": True,
    }


def atomic_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def checkpoint_payload(
    model: CAREBeliefV2Head,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    seed: int,
    variant: str,
    update: int,
    evaluations: list[dict[str, Any]],
    provenance: Mapping[str, Any],
    task_component_scales: torch.Tensor,
    requested_loss_config: CARELossV2Config,
    effective_loss_config: CARELossV2Config,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "created_at_utc": utc_now(),
        "seed": seed,
        "variant": variant,
        "update": update,
        "config": model.config.to_dict(),
        "loss_config": requested_loss_config.to_dict(),
        "requested_loss_config": requested_loss_config.to_dict(),
        "effective_loss_config": effective_loss_config.to_dict(),
        "task_component_scales": task_component_scales.detach().cpu(),
        "model": {
            key: tensor.detach().cpu() for key, tensor in model.state_dict().items()
        },
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
    parser.add_argument(
        "--variant",
        choices=("care", "reactive_only", "replay_only", "capacity"),
        required=True,
    )
    parser.add_argument("--stage", choices=("smoke", "formal"), required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--protocol-updates", type=int, default=PROTOCOL_UPDATES)
    parser.add_argument(
        "--action-prefix-steps", type=int, choices=(1, 4, 8, 16), required=True
    )
    parser.add_argument("--scale-quantile", type=float, default=0.90)
    parser.add_argument("--scale-floor", type=float, default=1e-4)
    parser.add_argument("--ranking-min-gap", type=float, default=1e-3)
    parser.add_argument("--candidate-ranking-weight", type=float, default=0.10)
    parser.add_argument("--reference-ranking-weight", type=float, default=0.10)
    parser.add_argument("--consistency-weight", type=float, default=0.20)
    parser.add_argument("--safety-weight", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.protocol_updates != PROTOCOL_UPDATES:
        raise ValueError("CARE v2 scorer protocol is fixed at 4000 updates")
    if not 1 <= args.updates <= args.protocol_updates:
        raise ValueError("invalid MARS CARE v2 update target")
    if args.stage == "formal" and args.updates != PROTOCOL_UPDATES:
        raise ValueError("formal MARS CARE v2 scorer requires 4000 updates")
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        expected = "COMPLETED" if args.stage == "formal" else "PASSED_SMOKE"
        if status.get("format_version") != STATUS_FORMAT_VERSION:
            raise RuntimeError("refusing to reuse a non-v2 CARE output directory")
        if (
            status.get("status") == expected
            and int(status.get("update", -1)) >= args.updates
        ):
            print(json.dumps({"status": "PRESERVED", "output": str(args.output)}))
            return

    seed_everything(args.seed)
    device = torch.device(args.device)
    prepared = load_prepared_care(args.prepared_data)
    intervention_steps = prepared_intervention_steps(prepared.manifest)
    if intervention_steps != args.action_prefix_steps:
        raise ValueError(
            "CARE v2 action prefix must equal the prepared branch intervention: "
            f"prefix={args.action_prefix_steps}, branch={intervention_steps}"
        )
    action_std = tuple(
        float(value) for value in prepared.manifest.get("action_std", (1.0,) * 8)
    )
    if action_std == (1.0,) * 8:
        raw = torch.load(args.prepared_data, map_location="cpu", weights_only=False)
        action_std = tuple(float(value) for value in raw["action_std"])
    config = CAREBeliefV2Config(
        variant=args.variant,
        action_std=action_std,
        action_prefix_steps=args.action_prefix_steps,
    )
    model = CAREBeliefV2Head(config).to(device)
    loss_config = CARELossV2Config(
        consistency_weight=args.consistency_weight,
        candidate_ranking_weight=args.candidate_ranking_weight,
        reference_ranking_weight=args.reference_ranking_weight,
        safety_weight=args.safety_weight,
        ranking_min_gap=args.ranking_min_gap,
    )
    task_component_scales = robust_task_component_scales(
        prepared.targets,
        prepared.usable,
        prepared.task_id,
        quantile=args.scale_quantile,
        floor=args.scale_floor,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.5
        * (
            1.0
            + math.cos(
                math.pi * min(step + 1, PROTOCOL_UPDATES) / PROTOCOL_UPDATES
            )
        ),
    )
    train = CARETrainingDataset(prepared, "all")
    diagnostic = CARETrainingDataset(
        prepared, "all", primary_horizon_only=True, primary_horizon=16
    )
    safety_supervision_nonzero_count = hard_safety_nonzero_count(train)
    effective_loss_config = CARELossV2Config(
        consistency_weight=loss_config.consistency_weight,
        candidate_ranking_weight=loss_config.candidate_ranking_weight,
        reference_ranking_weight=loss_config.reference_ranking_weight,
        safety_weight=(
            loss_config.safety_weight
            if safety_supervision_nonzero_count > 0
            else 0.0
        ),
        ranking_min_gap=loss_config.ranking_min_gap,
    )
    loader = DataLoader(
        diagnostic, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    provenance = {
        "prepared_data_sha256": sha256_file(args.prepared_data),
        "all_family_training": True,
        "family_count": len(prepared.snapshot_ids),
        "seed": args.seed,
        "variant": args.variant,
        "protocol_updates": PROTOCOL_UPDATES,
        "action_prefix_steps": args.action_prefix_steps,
        "prepared_intervention_steps": intervention_steps,
        "scale_quantile": float(args.scale_quantile),
        "scale_floor": float(args.scale_floor),
        "task_component_scales": task_component_scales.detach().cpu().tolist(),
        "utility_output_scale_applied": True,
        "utility_scale_source": "fixed_train_task_component_scales",
        "safety_supervision_nonzero_count": safety_supervision_nonzero_count,
        "safety_supervision_degenerate": safety_supervision_nonzero_count == 0,
        "requested_safety_weight": float(loss_config.safety_weight),
        "effective_safety_weight": float(effective_loss_config.safety_weight),
        "legacy_formal_run_unchanged": True,
        "deployment_requires_out_of_fold_gate": True,
    }
    latest = args.output / "checkpoint_latest.pt"
    evaluations: list[dict[str, Any]] = []
    start = 0
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved.get("format_version") != FORMAT_VERSION:
            raise RuntimeError("refusing to resume a non-v2 CARE checkpoint")
        if saved.get("provenance") != provenance:
            raise RuntimeError("MARS CARE v2 resume provenance drift")
        if not torch.equal(saved["task_component_scales"], task_component_scales.cpu()):
            raise RuntimeError("MARS CARE v2 robust target scale drift")
        if saved.get("loss_config") != loss_config.to_dict():
            raise RuntimeError("MARS CARE v2 loss contract drift")
        if saved.get("effective_loss_config") != effective_loss_config.to_dict():
            raise RuntimeError("MARS CARE v2 effective loss contract drift")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        evaluations = list(saved["evaluations"])
        start = int(saved["update"])

    atomic_json(
        status_path,
        {
            "format_version": STATUS_FORMAT_VERSION,
            "status": "TRAINING",
            "stage": args.stage,
            "seed": args.seed,
            "variant": args.variant,
            "update": start,
            "target_updates": args.updates,
            "action_prefix_steps": args.action_prefix_steps,
            "safety_supervision_nonzero_count": safety_supervision_nonzero_count,
            "safety_supervision_degenerate": safety_supervision_nonzero_count == 0,
            "requested_safety_weight": float(loss_config.safety_weight),
            "effective_safety_weight": float(effective_loss_config.safety_weight),
            "legacy_formal_run_unchanged": True,
        },
    )
    for update in range(start + 1, args.updates + 1):
        seed_everything(args.seed + 10_000_019 * update)
        batch = to_device(
            deterministic_batch(train, update, args.seed, args.batch_size), device
        )
        scale = task_component_scales.index_select(0, batch["task_id"])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model(
                batch["memory"],
                batch["memory_mask"],
                batch["candidate_chunks"],
                batch["horizon_index"],
                utility_scale=scale,
            )
            loss, _pieces = care_v2_training_loss(
                output,
                batch["target"],
                batch["hard_safety"],
                args.variant,
                target_scale=scale,
                loss_config=effective_loss_config,
                quantiles=model.config.quantiles,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite CARE v2 loss at {update}")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient):
            raise FloatingPointError(f"non-finite CARE v2 gradient at {update}")
        optimizer.step()
        scheduler.step()
        should_eval = update == args.updates or (
            args.stage == "formal" and update % args.eval_every == 0
        )
        if should_eval:
            metrics = evaluate(
                model,
                loader,
                device,
                args.variant,
                task_component_scales,
                effective_loss_config,
                safety_supervision_nonzero_count,
            )
            evaluations.append({"update": update, "diagnostic": metrics})
            print(
                json.dumps(
                    {
                        "protocol": "care-scorer-v2",
                        "variant": args.variant,
                        "seed": args.seed,
                        "update": update,
                        **metrics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if update == args.updates or update % args.save_every == 0:
            payload = checkpoint_payload(
                model,
                optimizer,
                scheduler,
                seed=args.seed,
                variant=args.variant,
                update=update,
                evaluations=evaluations,
                provenance=provenance,
                task_component_scales=task_component_scales,
                requested_loss_config=loss_config,
                effective_loss_config=effective_loss_config,
            )
            atomic_save(payload, latest)
            atomic_save(payload, args.output / f"checkpoint_{update:06d}.pt")

    final_status = "PASSED_SMOKE" if args.stage == "smoke" else "COMPLETED"
    result = {
        "format_version": STATUS_FORMAT_VERSION,
        "status": final_status,
        "completed_at_utc": utc_now(),
        "stage": args.stage,
        "variant": args.variant,
        "seed": args.seed,
        "update": args.updates,
        "train_rows": len(train),
        "diagnostic_rows": len(diagnostic),
        "all_family_training": True,
        "action_prefix_steps": args.action_prefix_steps,
        "prepared_intervention_steps": intervention_steps,
        "safety_supervision_nonzero_count": safety_supervision_nonzero_count,
        "safety_supervision_degenerate": safety_supervision_nonzero_count == 0,
        "requested_safety_weight": float(loss_config.safety_weight),
        "effective_safety_weight": float(effective_loss_config.safety_weight),
        "safety_gate_mode": (
            "legality_only"
            if safety_supervision_nonzero_count == 0
            else "diagnostic_not_applied"
        ),
        "task_component_scales": task_component_scales.cpu().tolist(),
        "loss_config": loss_config.to_dict(),
        "requested_loss_config": loss_config.to_dict(),
        "effective_loss_config": effective_loss_config.to_dict(),
        "latest_diagnostic": evaluations[-1]["diagnostic"],
        "checkpoint": str(latest.resolve()),
        "checkpoint_sha256": sha256_file(latest),
        "same_corpus_diagnostic_only": True,
        "deployment_requires_out_of_fold_gate": True,
        "legacy_formal_run_unchanged": True,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    atomic_json(status_path, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
