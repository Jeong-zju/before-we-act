#!/usr/bin/env python3
"""Family-disjoint diagnostic for the protocol-isolated CARE scorer-v2.

This script is deliberately separate from the frozen MARS CARE pipeline.  It
does not alter ``care_prepared.pt`` or any legacy checkpoint.  The only purpose
is to answer a narrow, pre-registered question before a new formal run:

* does an action-query prefix matched to the executed intervention (1, 4, 8,
  or 16 steps) improve ranking?
* does fixed robust task/component scaling improve the small-utility numerical
  signal?
* does explicitly ranking candidates against the structural reference (zero)
  improve the actual abstain/override decision?

Families are split deterministically per task (18 train / 6 validation / 6
calibration out of the 30 formal families).  No sibling branch from a family
can cross a split.  The resulting metrics are development diagnostics only;
the formal protocol still trains on all families and must use a fresh run and
fresh Validation20 seeds.

The script accepts the v2 modules added by the isolated scorer patch, while the
``legacy`` condition uses the original CARE head/loss for an apples-to-apples
reference.  All labels and model outputs stay in the original physical utility
units.  Robust scaling, when enabled, is applied only inside the v2 loss.  The
condition syntax is ``name:prefix:robust_scaling:reference_ranking``; ``100``
denotes the full 100-step action query used by the legacy encoder.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, default_collate

from before_we_act.care_belief import (
    CAREBeliefConfig,
    CAREBeliefHead,
    CAREBeliefOutput,
    CARECalibration,
    care_training_loss,
    select_care_candidate,
)
from before_we_act.care_belief_v2 import (
    CAREBeliefV2Config,
    CAREBeliefV2Head,
    CARELossV2Config,
    care_v2_training_loss,
    robust_task_component_scales,
)
from before_we_act.care_training_data import (
    CARETrainingDataset,
    PreparedCAREData,
    atomic_json,
    load_prepared_care,
    sha256_file,
)


SCRIPT_FORMAT = "before-we-act.care-mars-scorer-v2-diagnostic/2"
DEFAULT_SEED = 20260901
DEFAULT_UPDATES = 1000
DEFAULT_BATCH = 48
DEFAULT_CONDITIONS = (
    "legacy:100:0:0,"
    "full_robust:100:1:0,"
    "prefix1_robust_no_ref:1:1:0,"
    "prefix1_robust_ref:1:1:1"
)


@dataclass(frozen=True)
class Condition:
    """One pre-registered diagnostic condition."""

    name: str
    prefix: int
    robust_scaling: bool
    include_reference_ranking: bool
    legacy: bool = False

    def __post_init__(self) -> None:
        if self.prefix not in (1, 4, 8, 16, 100):
            raise ValueError(f"unsupported CARE prefix: {self.prefix}")
        if not isinstance(self.robust_scaling, bool):
            raise ValueError("robust_scaling must be boolean")
        if self.legacy and (
            self.prefix != 100
            or self.robust_scaling
            or self.include_reference_ranking
        ):
            raise ValueError("legacy condition must be the exact frozen recipe")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deterministic_split(prepared: PreparedCAREData) -> dict[str, tuple[int, ...]]:
    """Create a task-stratified family split without using branch outcomes."""

    by_task: dict[int, list[int]] = defaultdict(list)
    for family, task in enumerate(prepared.task_id.tolist()):
        by_task[int(task)].append(family)
    result: dict[str, list[int]] = {"train": [], "validation": [], "calibration": []}
    for task, families in sorted(by_task.items()):
        # Sorting by the immutable snapshot id makes this independent of the
        # order in which a collector happened to write files.
        families = sorted(families, key=lambda index: str(prepared.snapshot_ids[index]))
        if len(families) < 5:
            raise ValueError(f"task {task} has too few families for a split")
        for ordinal, family in enumerate(families):
            bucket = ordinal % 5
            result["train" if bucket < 3 else ("validation" if bucket == 3 else "calibration")].append(family)
    return {key: tuple(sorted(value)) for key, value in result.items()}


class FamilyDataset(Dataset[dict[str, torch.Tensor]]):
    """A CARE row dataset restricted to an explicit family set."""

    def __init__(
        self,
        prepared: PreparedCAREData,
        families: Iterable[int],
        *,
        primary_horizon_only: bool = False,
        primary_horizon: int = 16,
    ) -> None:
        self.base = CARETrainingDataset(
            prepared, "all", primary_horizon_only=primary_horizon_only,
            primary_horizon=primary_horizon,
        )
        allowed = set(int(value) for value in families)
        self.indices = tuple(
            index for index, (family, _horizon, _repeat) in enumerate(self.base.rows)
            if family in allowed
        )
        if not self.indices:
            raise ValueError("family split has no usable CARE rows")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.base[self.indices[index]]


def deterministic_batch(dataset: Dataset, update: int, seed: int, size: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + 1_000_003 * update)
    indices = torch.randint(len(dataset), (size,), generator=generator).tolist()
    return default_collate([dataset[index] for index in indices])


def to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def hard_safety_nonzero_count(
    source: Dataset | Iterable[Mapping[str, Any]],
) -> int:
    """Count non-reference hard-safety labels in a row source.

    Candidate zero is a structural reference and is always safe by contract;
    it is excluded from the count.  This count is deliberately based on
    labels, never on the randomly initialized safety head.
    """

    if isinstance(source, Dataset):
        rows = (source[index] for index in range(len(source)))
    else:
        rows = iter(source)
    total = 0
    for row in rows:
        labels = torch.as_tensor(row["hard_safety"])
        if labels.ndim != 1 or labels.shape[0] < 2:
            raise ValueError("CARE hard_safety rows must be [candidate] vectors")
        total += int(torch.count_nonzero(labels[1:]))
    return total


def safety_supervision_status(
    training_nonzero_count: int,
    calibration_nonzero_count: int,
) -> dict[str, Any]:
    """Resolve the diagnostic safety mode from observed label support.

    A missing positive class in either the training or calibration split means
    that a learned hard-safety probability is not evidence: the head may be
    random (training) or the threshold cannot be validated (calibration).
    Such diagnostics therefore use legality-only masking.  Formal deployment
    safety remains owned by the frozen selector and is not changed here.
    """

    training_nonzero_count = int(training_nonzero_count)
    calibration_nonzero_count = int(calibration_nonzero_count)
    if training_nonzero_count < 0 or calibration_nonzero_count < 0:
        raise ValueError("hard-safety support counts must be non-negative")
    training_degenerate = training_nonzero_count == 0
    calibration_degenerate = calibration_nonzero_count == 0
    degenerate = training_degenerate or calibration_degenerate
    return {
        "safety_supervision_degenerate": degenerate,
        "safety_training_supervision_degenerate": training_degenerate,
        "safety_calibration_degenerate": calibration_degenerate,
        "safety_gate_mode": (
            "legality_only" if degenerate else "learned_probability_uncalibrated"
        ),
        "learned_safety_mask_applied": not degenerate,
        "safety_probability_calibrated": False,
        "training_hard_safety_nonzero_count": training_nonzero_count,
        "calibration_hard_safety_nonzero_count": calibration_nonzero_count,
    }


def subset_scales(
    prepared: PreparedCAREData,
    families: Sequence[int],
    *,
    quantile: float,
    floor: float,
) -> torch.Tensor:
    """Estimate robust units from train families only (no calibration leakage)."""

    selected = torch.zeros(len(prepared.snapshot_ids), dtype=torch.bool)
    selected[list(families)] = True
    targets = prepared.targets[selected]
    usable = prepared.usable[selected]
    task_id = prepared.task_id[selected]
    return robust_task_component_scales(
        targets, usable, task_id, quantile=quantile, floor=floor
    )


def _physical_quantiles(output: CAREBeliefOutput) -> torch.Tensor:
    """Return selector-facing quantiles in the unchanged physical units."""

    return output.quantiles.float()


def _pair_metrics(scores: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    delta = scores[:, :, None] - scores[:, None, :]
    target_delta = targets[:, :, None] - targets[:, None, :]
    upper = torch.triu(
        torch.ones(scores.shape[1], scores.shape[1], dtype=torch.bool, device=scores.device),
        diagonal=1,
    )
    mask = (target_delta.abs() > 1e-6) & upper.unsqueeze(0)
    return int(((delta.sign() == target_delta.sign()) & mask).sum()), int(mask.sum())


@torch.no_grad()
def prediction_rows(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    condition: Condition,
    task_scales: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    for raw in loader:
        batch = to_device(raw, device)
        utility_scale = (
            task_scales.index_select(0, batch["task_id"])
            if condition.robust_scaling
            else None
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if condition.legacy:
                output = model(
                    batch["memory"],
                    batch["memory_mask"],
                    batch["candidate_chunks"],
                    batch["horizon_index"],
                )
            else:
                output = model(
                    batch["memory"],
                    batch["memory_mask"],
                    batch["candidate_chunks"],
                    batch["horizon_index"],
                    utility_scale=utility_scale,
                )
        quantiles = _physical_quantiles(output).cpu()
        safety = output.hard_safety_logit.float().cpu()
        target = batch["target"].float().cpu()
        for index in range(target.shape[0]):
            rows.append(
                {
                    "family_index": int(batch["family_index"][index]),
                    "task_id": int(batch["task_id"][index]),
                    "quantiles": quantiles[index],
                    "hard_safety_logit": safety[index],
                    "target": target[index],
                    "hard_safety": batch["hard_safety"][index].float().cpu(),
                }
            )
    return rows


def conformal_correction(rows: Sequence[Mapping[str, Any]], nominal: float = 0.9) -> float:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        lower = row["quantiles"][:, 2, 0]
        target = row["target"][:, 2]
        grouped[int(row["family_index"])].extend((lower[1:] - target[1:]).tolist())
    scores = np.asarray([max(values) for values in grouped.values() if values], dtype=np.float64)
    if not len(scores):
        raise ValueError("calibration split has no finite rows")
    adjusted = min(1.0, math.ceil((len(scores) + 1) * nominal) / len(scores))
    try:
        value = float(np.quantile(scores, adjusted, method="higher"))
    except TypeError:  # pragma: no cover - old numpy compatibility
        value = float(np.quantile(scores, adjusted, interpolation="higher"))
    return max(0.0, value)


def metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    correction: float | None = None,
    safety_probability_max: float = 0.25,
    safety_supervision_degenerate: bool | None = None,
    safety_training_nonzero_count: int | None = None,
    safety_calibration_nonzero_count: int | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot score an empty row set")
    scores = torch.stack([row["quantiles"][:, 2, 2] for row in rows])
    lower = torch.stack([row["quantiles"][:, 2, 0] for row in rows])
    targets = torch.stack([row["target"][:, 2] for row in rows])
    selected = scores.argmax(1)
    best = targets.argmax(1)
    ranking_regret = (
        targets.max(1).values
        - targets.gather(1, selected[:, None]).squeeze(1)
    )
    pair_correct, pair_count = _pair_metrics(scores, targets)
    relative_target = targets[:, 1:] - targets[:, :1]
    relative_score = scores[:, 1:] - scores[:, :1]
    sign_mask = relative_target.abs() > 1e-6
    sign_correct = int(((relative_score.sign() == relative_target.sign()) & sign_mask).sum())
    observed_safety_nonzero_count = hard_safety_nonzero_count(rows)
    # A zero-positive split must never let an untrained/random safety head
    # screen every candidate.  If a caller supplies corpus-level context, it
    # can mark the condition degenerate even when this particular report split
    # happens to contain a positive label.
    if safety_training_nonzero_count is not None and int(safety_training_nonzero_count) < 0:
        raise ValueError("safety_training_nonzero_count must be non-negative")
    if safety_calibration_nonzero_count is not None and int(safety_calibration_nonzero_count) < 0:
        raise ValueError("safety_calibration_nonzero_count must be non-negative")
    has_corpus_context = (
        safety_training_nonzero_count is not None
        or safety_calibration_nonzero_count is not None
    )
    safety_supervision_degenerate = bool(
        safety_supervision_degenerate is True
        or safety_training_nonzero_count == 0
        or safety_calibration_nonzero_count == 0
        or (
            observed_safety_nonzero_count == 0
            and (
                not has_corpus_context
                or safety_training_nonzero_count is None
                or safety_calibration_nonzero_count is None
            )
        )
    )
    proposed = lower.clone()
    if correction is not None:
        proposed = proposed - float(correction)
    raw_unsafe = torch.stack(
        [row["hard_safety_logit"].sigmoid() > safety_probability_max for row in rows]
    )
    if safety_supervision_degenerate:
        # All prepared branches are legality-checked before entering this
        # diagnostic.  With no positive safety labels, legality-only is the
        # only auditable gate; learned logits are intentionally ignored.
        unsafe = torch.zeros_like(
            torch.stack([row["hard_safety_logit"] for row in rows]), dtype=torch.bool
        )
    else:
        unsafe = raw_unsafe
    proposed = proposed.masked_fill(unsafe, -torch.inf)
    proposed[:, 0] = 0.0
    selected_lower = proposed.argmax(1)
    override = selected_lower != 0
    selected_target = targets.gather(1, selected_lower[:, None]).squeeze(1)
    selector_regret = targets.max(1).values - selected_target
    ranking_task_rows: dict[int, list[float]] = defaultdict(list)
    selector_task_rows: dict[int, list[float]] = defaultdict(list)
    task_override: dict[int, list[bool]] = defaultdict(list)
    for ranking_value, selector_value, row, over in zip(
        ranking_regret.tolist(), selector_regret.tolist(), rows, override.tolist()
    ):
        task = int(row["task_id"])
        ranking_task_rows[task].append(float(ranking_value))
        selector_task_rows[task].append(float(selector_value))
        task_override[task].append(bool(over))
    return {
        "rows": len(rows),
        "families": len({int(row["family_index"]) for row in rows}),
        "median_ranking_top1_accuracy": float((selected == best).float().mean()),
        "pairwise_accuracy_including_reference": pair_correct / pair_count if pair_count else 0.0,
        "candidate_vs_reference_sign_accuracy": sign_correct / int(sign_mask.sum()) if int(sign_mask.sum()) else 0.0,
        "median_ranking_mean_regret": float(ranking_regret.mean()),
        "median_ranking_median_regret": float(ranking_regret.median()),
        "median_ranking_mean_regret_by_task_id": {
            str(key): float(np.mean(value))
            for key, value in sorted(ranking_task_rows.items())
        },
        "correction": None if correction is None else float(correction),
        "selector_top1_accuracy": float((selected_lower == best).float().mean()),
        "selector_mean_regret": float(selector_regret.mean()),
        "selector_median_regret": float(selector_regret.median()),
        "selector_mean_regret_by_task_id": {
            str(key): float(np.mean(value))
            for key, value in sorted(selector_task_rows.items())
        },
        "override_rate": float(override.float().mean()),
        "harmful_override_rate": float((override & (selected_target < 0)).float().sum() / max(int(override.sum()), 1)),
        "beneficial_override_rate": float((override & (selected_target > 0)).float().sum() / max(int(override.sum()), 1)),
        "unsafe_candidate_rate": float(unsafe[:, 1:].float().mean()),
        "unsafe_candidate_count": int(unsafe[:, 1:].sum()),
        "raw_predicted_unsafe_candidate_rate": float(
            raw_unsafe[:, 1:].float().mean()
        ),
        "raw_predicted_unsafe_candidate_count": int(raw_unsafe[:, 1:].sum()),
        "requested_safety_probability_max": float(safety_probability_max),
        "effective_safety_probability_max": (
            1.0 if safety_supervision_degenerate else float(safety_probability_max)
        ),
        "safety_positive_label_count_in_rows": int(observed_safety_nonzero_count),
        "safety_training_nonzero_count": (
            None
            if safety_training_nonzero_count is None
            else int(safety_training_nonzero_count)
        ),
        "safety_calibration_nonzero_count": (
            None
            if safety_calibration_nonzero_count is None
            else int(safety_calibration_nonzero_count)
        ),
        "safety_supervision_degenerate": safety_supervision_degenerate,
        "safety_gate_mode": (
            "legality_only"
            if safety_supervision_degenerate
            else "learned_probability_uncalibrated"
        ),
        "learned_safety_mask_applied": not safety_supervision_degenerate,
        "safety_probability_calibrated": False,
        "override_rate_by_task_id": {str(key): float(np.mean(value)) for key, value in sorted(task_override.items())},
    }


def build_condition(
    name: str,
    prefix: int,
    robust_scaling: bool,
    include_reference: bool,
    legacy: bool = False,
) -> Condition:
    return Condition(
        name=name,
        prefix=int(prefix),
        robust_scaling=bool(robust_scaling),
        include_reference_ranking=bool(include_reference),
        legacy=bool(legacy),
    )


def parse_conditions(spec: str) -> list[Condition]:
    """Parse ``name:prefix:robust_scaling:reference_ranking`` conditions."""
    result: list[Condition] = []
    for raw in (part.strip() for part in spec.split(",")):
        if not raw:
            continue
        fields = raw.split(":")
        if len(fields) != 4:
            raise ValueError(
                "condition must be name:prefix:robust_scaling:reference_ranking, "
                f"got {raw!r}"
            )
        name, prefix, robust, ref = fields
        if robust not in {"0", "1"} or ref not in {"0", "1"}:
            raise ValueError("condition booleans must be 0 or 1")
        result.append(
            build_condition(
                name,
                int(prefix),
                bool(int(robust)),
                bool(int(ref)),
                name == "legacy",
            )
        )
    if not result:
        raise ValueError("no conditions specified")
    return result


def train_condition(
    prepared: PreparedCAREData,
    split: Mapping[str, tuple[int, ...]],
    condition: Condition,
    *,
    seed: int,
    updates: int,
    batch_size: int,
    eval_every: int,
    device: torch.device,
    scale_quantile: float,
    scale_floor: float,
    loss_config: CARELossV2Config,
    action_std: Sequence[float],
) -> dict[str, Any]:
    seed_everything(seed)
    train = FamilyDataset(prepared, split["train"])
    validation = FamilyDataset(prepared, split["validation"], primary_horizon_only=True)
    calibration = FamilyDataset(prepared, split["calibration"], primary_horizon_only=True)
    training_safety_nonzero_count = hard_safety_nonzero_count(train)
    calibration_safety_nonzero_count = hard_safety_nonzero_count(calibration)
    safety_status = safety_supervision_status(
        training_safety_nonzero_count, calibration_safety_nonzero_count
    )
    requested_safety_weight = float(loss_config.safety_weight)
    effective_safety_weight = (
        requested_safety_weight if training_safety_nonzero_count > 0 else 0.0
    )
    condition_loss_config = CARELossV2Config(
        consistency_weight=loss_config.consistency_weight,
        candidate_ranking_weight=loss_config.candidate_ranking_weight,
        reference_ranking_weight=(
            loss_config.reference_ranking_weight
            if condition.include_reference_ranking
            else 0.0
        ),
        safety_weight=effective_safety_weight,
        ranking_min_gap=loss_config.ranking_min_gap,
    )
    robust_scales = subset_scales(
        prepared, split["train"], quantile=scale_quantile, floor=scale_floor
    )
    # The scale tensor is indexed by task id.  Keeping it on the selected
    # device avoids accidentally moving data through the CPU in every update.
    scales = robust_scales.to(device)
    action_std = tuple(float(value) for value in action_std)
    if len(action_std) != 8 or any(not math.isfinite(value) or value <= 0 for value in action_std):
        raise ValueError("prepared action_std must contain eight positive finite values")
    if condition.legacy:
        config = CAREBeliefConfig(variant="care", action_std=action_std)
        model: torch.nn.Module = CAREBeliefHead(config).to(device)
    else:
        config = CAREBeliefV2Config(
            variant="care",
            action_std=action_std,
            action_prefix_steps=condition.prefix,
        )
        model = CAREBeliefV2Head(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(updates, 1), eta_min=3e-6)
    train_history: list[dict[str, Any]] = []
    train_scale_metadata = robust_scales.tolist()
    for update in range(1, updates + 1):
        seed_everything(seed + 10_000_019 * update)
        batch = to_device(deterministic_batch(train, update, seed, batch_size), device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if condition.legacy:
                output = model(
                    batch["memory"],
                    batch["memory_mask"],
                    batch["candidate_chunks"],
                    batch["horizon_index"],
                )
                # Legacy CARE deliberately excludes candidate 0 from ranking.
                loss, pieces = care_training_loss(
                    output,
                    batch["target"],
                    batch["hard_safety"],
                    "care",
                )
                if effective_safety_weight == 0.0:
                    # The frozen legacy loss has a fixed 0.10 BCE term.  In
                    # this isolated diagnostic, subtract the same live term
                    # when the corpus has no positive safety supervision; the
                    # formal legacy selector and loss remain untouched.
                    live_safety = F.binary_cross_entropy_with_logits(
                        output.hard_safety_logit[:, 1:],
                        batch["hard_safety"][:, 1:].float(),
                    )
                    loss = loss - 0.10 * live_safety
            else:
                batch_scales = scales.index_select(0, batch["task_id"])
                output_scale = batch_scales if condition.robust_scaling else None
                loss_scale = (
                    batch_scales
                    if condition.robust_scaling
                    else torch.ones_like(batch_scales)
                )
                output = model(
                    batch["memory"],
                    batch["memory_mask"],
                    batch["candidate_chunks"],
                    batch["horizon_index"],
                    utility_scale=output_scale,
                )
                loss, pieces = care_v2_training_loss(
                    output,
                    batch["target"],
                    batch["hard_safety"],
                    "care",
                    target_scale=loss_scale,
                    loss_config=condition_loss_config,
                    quantiles=config.quantiles,
                )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at update {update}")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient):
            raise FloatingPointError(f"non-finite gradient at update {update}")
        optimizer.step()
        scheduler.step()
        if update == updates or update % max(eval_every, 1) == 0:
            val_rows = prediction_rows(
                model,
                DataLoader(validation, batch_size=batch_size, shuffle=False),
                device,
                condition,
                scales,
            )
            cal_rows = prediction_rows(
                model,
                DataLoader(calibration, batch_size=batch_size, shuffle=False),
                device,
                condition,
                scales,
            )
            correction = conformal_correction(cal_rows)
            safety_metric_args = {
                "safety_supervision_degenerate": bool(
                    safety_status["safety_supervision_degenerate"]
                ),
                "safety_training_nonzero_count": training_safety_nonzero_count,
                "safety_calibration_nonzero_count": calibration_safety_nonzero_count,
            }
            record = {
                "update": update,
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient),
                "safety_supervision": dict(safety_status),
                "validation": metrics(val_rows, **safety_metric_args),
                "validation_calibrated": metrics(
                    val_rows, correction=correction, **safety_metric_args
                ),
                "calibration": metrics(cal_rows, **safety_metric_args),
            }
            train_history.append(record)
    final = train_history[-1]
    return {
        "condition": asdict(condition),
        "seed": int(seed),
        "updates": int(updates),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "calibration_rows": len(calibration),
        "train_families": list(split["train"]),
        "validation_families": list(split["validation"]),
        "calibration_families": list(split["calibration"]),
        "train_task_component_scales": (
            train_scale_metadata if condition.robust_scaling else None
        ),
        "utility_units": "physical",
        "utility_output_scale_applied": condition.robust_scaling,
        "safety_supervision": dict(safety_status),
        "requested_safety_weight": requested_safety_weight,
        "effective_safety_weight": effective_safety_weight,
        "requested_loss_config": loss_config.to_dict(),
        "effective_loss_config": condition_loss_config.to_dict(),
        "safety_supervision_degenerate": bool(
            safety_status["safety_supervision_degenerate"]
        ),
        "history": train_history,
        "final": final,
        "same_corpus_diagnostic_only": True,
        "formal_promotion_requires_fresh_run": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        default=DEFAULT_CONDITIONS,
        help=(
            "comma-separated name:prefix:robust_scaling:reference_ranking "
            "conditions (boolean fields are 0/1)"
        ),
    )
    parser.add_argument("--seeds", default=str(DEFAULT_SEED), help="comma-separated integer seeds")
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--scale-quantile", type=float, default=0.90)
    parser.add_argument("--scale-floor", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepared-sha256", default="", help="optional expected prepared-data hash")
    args = parser.parse_args()
    if args.updates < 1:
        raise ValueError("updates must be positive")
    prepared_hash = sha256_file(args.prepared_data)
    if args.prepared_sha256 and prepared_hash != args.prepared_sha256:
        raise ValueError("prepared-data SHA256 mismatch")
    prepared = load_prepared_care(args.prepared_data)
    raw_prepared = torch.load(args.prepared_data, map_location="cpu", weights_only=False)
    action_std = tuple(float(value) for value in raw_prepared.get("action_std", prepared.manifest.get("action_std", (1.0,) * 8)))
    split = deterministic_split(prepared)
    conditions = parse_conditions(args.conditions)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    device = torch.device(args.device)
    loss_config = CARELossV2Config()
    results: list[dict[str, Any]] = []
    for condition in conditions:
        for seed in seeds:
            print(json.dumps({"event": "condition_start", "condition": asdict(condition), "seed": seed}, sort_keys=True), flush=True)
            results.append(
                train_condition(
                    prepared,
                    split,
                    condition,
                    seed=seed,
                    updates=args.updates,
                    batch_size=args.batch_size,
                    eval_every=args.eval_every,
                    device=device,
                    scale_quantile=args.scale_quantile,
                    scale_floor=args.scale_floor,
                    loss_config=loss_config,
                    action_std=action_std,
                )
            )
            print(json.dumps({"event": "condition_complete", "condition": condition.name, "seed": seed, "final": results[-1]["final"]}, sort_keys=True), flush=True)
    payload = {
        "format_version": SCRIPT_FORMAT,
        "created_at_utc": utc_now(),
        "prepared_data": str(args.prepared_data.resolve()),
        "prepared_data_sha256": prepared_hash,
        "family_count": len(prepared.snapshot_ids),
        "tasks": list(prepared.tasks),
        "split_protocol": "per-task sorted snapshot ids; ordinal mod 5: 0-2 train, 3 validation, 4 calibration",
        "split": {key: list(value) for key, value in split.items()},
        "conditions": [asdict(value) for value in conditions],
        "seeds": list(seeds),
        "updates": args.updates,
        "batch_size": args.batch_size,
        "scale_quantile": args.scale_quantile,
        "scale_floor": args.scale_floor,
        "utility_units": "physical",
        "target_or_output_multiplier_used": False,
        "ablation_axes": ["robust_scaling", "executed_prefix", "reference_ranking"],
        "legacy_formal_run_unchanged": True,
        "same_corpus_diagnostic_only": True,
        "results": results,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"status": "complete", "output": str(args.output.resolve()), "conditions": len(conditions), "seeds": len(seeds)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
