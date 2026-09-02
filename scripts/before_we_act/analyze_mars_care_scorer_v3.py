#!/usr/bin/env python3
"""Family-disjoint ablation for protocol-isolated CARE scorer-v3.

The experiment changes no CARE target, candidate, reference policy, quantile,
selector, or calibration guarantee.  It tests whether making candidate-slot
identity, task identity, and task/horizon numerical units explicit improves
the same scorer objective.  Families remain disjoint across train, validation,
and calibration.  The report also names extreme q05 conformity families but
never removes them from the correction or relaxes the admission thresholds.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from before_we_act.care_belief_v2 import (
    CARELossV2Config,
    care_v2_training_loss,
)
from before_we_act.care_belief_v3 import (
    CAREBeliefV3Config,
    CAREBeliefV3Head,
    robust_task_horizon_component_scales,
)
from before_we_act.care_training_data import (
    PreparedCAREData,
    atomic_json,
    load_prepared_care,
    sha256_file,
)
from scripts.before_we_act.analyze_mars_care_scorer_v2 import (
    FamilyDataset,
    conformal_correction,
    deterministic_batch,
    deterministic_split,
    hard_safety_nonzero_count,
    metrics,
    safety_supervision_status,
    seed_everything,
    subset_scales,
    to_device,
)


SCRIPT_FORMAT = "before-we-act.care-mars-scorer-v3-diagnostic/1"
DEFAULT_CONDITIONS = (
    "v2_control:0:0:0,"
    "slot_only:1:0:0,"
    "task_only:0:1:0,"
    "slot_task_horizon:1:1:1"
)


@dataclass(frozen=True)
class ConditionV3:
    name: str
    candidate_slot_embedding: bool
    task_embedding: bool
    horizon_aware_scaling: bool

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CARE v3 condition name cannot be empty")
        for value in (
            self.candidate_slot_embedding,
            self.task_embedding,
            self.horizon_aware_scaling,
        ):
            if not isinstance(value, bool):
                raise ValueError("CARE v3 condition flags must be boolean")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_conditions(spec: str) -> list[ConditionV3]:
    result: list[ConditionV3] = []
    names: set[str] = set()
    for raw in (value.strip() for value in spec.split(",")):
        if not raw:
            continue
        fields = raw.split(":")
        if len(fields) != 4 or any(value not in {"0", "1"} for value in fields[1:]):
            raise ValueError(
                "CARE v3 condition must be name:candidate_slot:task:horizon_scale"
            )
        if fields[0] in names:
            raise ValueError(f"duplicate CARE v3 condition: {fields[0]}")
        names.add(fields[0])
        result.append(
            ConditionV3(
                name=fields[0],
                candidate_slot_embedding=bool(int(fields[1])),
                task_embedding=bool(int(fields[2])),
                horizon_aware_scaling=bool(int(fields[3])),
            )
        )
    if not result:
        raise ValueError("no CARE v3 conditions specified")
    return result


def subset_horizon_scales(
    prepared: PreparedCAREData,
    families: Sequence[int],
    *,
    quantile: float,
    floor: float,
) -> torch.Tensor:
    selected = torch.zeros(len(prepared.snapshot_ids), dtype=torch.bool)
    selected[list(families)] = True
    return robust_task_horizon_component_scales(
        prepared.targets[selected],
        prepared.usable[selected],
        prepared.task_id[selected],
        quantile=quantile,
        floor=floor,
    )


def _batch_scale(scales: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    task_id = batch["task_id"].long()
    if scales.ndim == 2:
        return scales.index_select(0, task_id)
    if scales.ndim == 3:
        return scales[task_id, batch["horizon_index"].long()]
    raise ValueError("CARE v3 scale must be [task,component] or [task,horizon,component]")


@torch.no_grad()
def prediction_rows(
    model: CAREBeliefV3Head,
    loader: DataLoader,
    device: torch.device,
    scales: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    for raw in loader:
        batch = to_device(raw, device)
        scale = _batch_scale(scales, batch)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model(
                batch["memory"],
                batch["memory_mask"],
                batch["candidate_chunks"],
                batch["horizon_index"],
                batch["task_id"],
                utility_scale=scale,
            )
        for index in range(batch["target"].shape[0]):
            rows.append(
                {
                    "family_index": int(batch["family_index"][index]),
                    "task_id": int(batch["task_id"][index]),
                    "horizon_index": int(batch["horizon_index"][index]),
                    "quantiles": output.quantiles[index].float().cpu(),
                    "hard_safety_logit": output.hard_safety_logit[index].float().cpu(),
                    "target": batch["target"][index].float().cpu(),
                    "hard_safety": batch["hard_safety"][index].float().cpu(),
                }
            )
    return rows


def _higher_quantile(values: Sequence[float], nominal: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("CARE v3 conformal set is empty")
    probability = min(1.0, math.ceil((len(array) + 1) * nominal) / len(array))
    try:
        return float(np.quantile(array, probability, method="higher"))
    except TypeError:  # pragma: no cover - NumPy < 1.22
        return float(np.quantile(array, probability, interpolation="higher"))


def _family_q05_scores(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for row in rows:
        family = int(row["family_index"])
        task = int(row["task_id"])
        lower = torch.as_tensor(row["quantiles"])[:, 2, 0].float()
        target = torch.as_tensor(row["target"])[:, 2].float()
        errors = (lower[1:] - target[1:]).cpu().numpy()
        state = grouped.setdefault(
            family,
            {"task_id": task, "errors": [], "rows": 0},
        )
        if int(state["task_id"]) != task:
            raise ValueError("CARE v3 family spans multiple tasks")
        state["errors"].extend(float(value) for value in errors)
        state["rows"] = int(state["rows"]) + 1
    result: dict[int, dict[str, Any]] = {}
    for family, state in grouped.items():
        errors = np.asarray(state["errors"], dtype=np.float64)
        if not np.isfinite(errors).all() or errors.size == 0:
            raise ValueError(f"CARE v3 family {family} has invalid q05 errors")
        result[family] = {
            "task_id": int(state["task_id"]),
            "q05_overestimate": max(0.0, float(np.max(errors))),
            "rows": int(state["rows"]),
        }
    if not result:
        raise ValueError("CARE v3 q05 diagnostic is empty")
    return result


def taskwise_conformal_report(
    calibration_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    nominal: float = 0.90,
) -> dict[str, Any]:
    """Report Mondrian task corrections with unchanged family-wise coverage."""

    if not 0.0 < float(nominal) <= 1.0:
        raise ValueError("CARE v3 nominal coverage must lie in (0,1]")
    calibration = _family_q05_scores(calibration_rows)
    validation = _family_q05_scores(validation_rows)
    by_task: dict[int, list[float]] = defaultdict(list)
    for value in calibration.values():
        by_task[int(value["task_id"])].append(float(value["q05_overestimate"]))
    corrections = {
        task: max(0.0, _higher_quantile(values, float(nominal)))
        for task, values in sorted(by_task.items())
    }
    coverage: dict[int, list[bool]] = defaultdict(list)
    for value in validation.values():
        task = int(value["task_id"])
        if task not in corrections:
            raise ValueError(f"CARE v3 task {task} lacks calibration families")
        coverage[task].append(
            float(value["q05_overestimate"]) <= float(corrections[task])
        )
    all_covered = [item for values in coverage.values() for item in values]
    return {
        "nominal_simultaneous_coverage": float(nominal),
        "familywise_within_task": True,
        "correction_by_task_id": {
            str(key): float(value) for key, value in corrections.items()
        },
        "calibration_family_count_by_task_id": {
            str(key): len(value) for key, value in sorted(by_task.items())
        },
        "validation_family_coverage": float(np.mean(all_covered)),
        "validation_family_coverage_by_task_id": {
            str(key): float(np.mean(value))
            for key, value in sorted(coverage.items())
        },
        "correction_fitted_without_validation": True,
        "admission_thresholds_unchanged": True,
    }


def _task_horizon_scales(
    value: torch.Tensor | Sequence[Any],
    horizons: Sequence[int],
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Validate selector-facing task/horizon total-utility units."""

    scales = torch.as_tensor(value, dtype=torch.float64).cpu()
    requested = tuple(int(item) for item in horizons)
    if scales.ndim != 3 or scales.shape[0] < 1 or scales.shape[2] < 3:
        raise ValueError(
            "CARE v3 conformal scales must be [task,horizon,component>=3]"
        )
    if len(requested) != scales.shape[1] or len(set(requested)) != len(requested):
        raise ValueError("CARE v3 horizon labels differ from the scale tensor")
    if any(item <= 0 for item in requested):
        raise ValueError("CARE v3 horizon labels must be positive")
    if not torch.isfinite(scales).all() or bool((scales <= 0).any()):
        raise ValueError("CARE v3 conformal scales must be finite and positive")
    return scales, requested


def q05_family_horizon_residuals(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_horizon_component_scales: torch.Tensor | Sequence[Any],
    snapshot_ids: Sequence[str] | None = None,
    horizons: Sequence[int],
) -> list[dict[str, Any]]:
    """Return untrimmed selector-q05 residuals for each family and horizon.

    CARE selects with the total-utility component (component 2), so the
    conformity residual is the largest lower-quantile overestimate among the
    five learned candidates.  Candidate zero is the exact reference and is
    excluded.  Repeated rows from the same family/horizon remain in the max;
    no outlier is clipped, winsorized, or discarded.
    """

    scales, requested = _task_horizon_scales(
        task_horizon_component_scales, horizons
    )
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    family_tasks: dict[int, int] = {}
    for row in rows:
        family = int(row["family_index"])
        task = int(row["task_id"])
        horizon_index = int(row["horizon_index"])
        if family < 0:
            raise ValueError("CARE v3 family index must be non-negative")
        if not 0 <= task < scales.shape[0]:
            raise ValueError("CARE v3 task id is out of scale range")
        if not 0 <= horizon_index < scales.shape[1]:
            raise ValueError("CARE v3 horizon index is out of scale range")
        previous_task = family_tasks.setdefault(family, task)
        if previous_task != task:
            raise ValueError("CARE v3 family spans multiple tasks")

        quantiles = torch.as_tensor(row["quantiles"], dtype=torch.float64).cpu()
        target = torch.as_tensor(row["target"], dtype=torch.float64).cpu()
        if (
            quantiles.ndim != 3
            or target.ndim != 2
            or quantiles.shape[0] != target.shape[0]
            or quantiles.shape[0] < 2
            or quantiles.shape[1] < 3
            or target.shape[1] < 3
            or quantiles.shape[2] < 1
        ):
            raise ValueError("CARE v3 q05 row has an invalid prediction/target shape")
        errors = quantiles[1:, 2, 0] - target[1:, 2]
        if errors.numel() == 0 or not torch.isfinite(errors).all():
            raise ValueError("CARE v3 q05 row has no finite learned-candidate errors")
        worst, relative_candidate = torch.max(errors, dim=0)
        physical = max(0.0, float(worst))
        candidate = int(relative_candidate) + 1
        key = (family, horizon_index)
        state = grouped.setdefault(
            key,
            {
                "family_index": family,
                "task_id": task,
                "horizon_index": horizon_index,
                "q05_overestimate_physical": physical,
                "worst_candidate_id": candidate,
                "rows": 0,
            },
        )
        if int(state["task_id"]) != task:
            raise ValueError("CARE v3 family/horizon spans multiple tasks")
        state["rows"] = int(state["rows"]) + 1
        if physical > float(state["q05_overestimate_physical"]):
            state["q05_overestimate_physical"] = physical
            state["worst_candidate_id"] = candidate

    if not grouped:
        raise ValueError("CARE v3 family/horizon q05 diagnostic is empty")
    result: list[dict[str, Any]] = []
    for (family, horizon_index), state in sorted(grouped.items()):
        task = int(state["task_id"])
        physical = float(state["q05_overestimate_physical"])
        total_scale = float(scales[task, horizon_index, 2])
        value = {
            **state,
            "horizon_steps": int(requested[horizon_index]),
            "total_utility_scale": total_scale,
            "q05_overestimate_normalized": physical / total_scale,
            "candidate_zero_excluded": True,
            "residuals_trimmed_or_clipped": False,
        }
        if snapshot_ids is not None:
            if family >= len(snapshot_ids):
                raise ValueError("CARE v3 snapshot id index is out of range")
            value["snapshot_id"] = str(snapshot_ids[family])
        result.append(value)
    return result


def _complete_family_maxima(
    residuals: Sequence[Mapping[str, Any]],
    *,
    horizons: Sequence[int],
    field: str,
) -> dict[int, dict[str, Any]]:
    expected = set(int(value) for value in horizons)
    grouped: dict[int, dict[str, Any]] = {}
    for row in residuals:
        family = int(row["family_index"])
        task = int(row["task_id"])
        horizon = int(row["horizon_steps"])
        state = grouped.setdefault(
            family, {"task_id": task, "horizons": set(), "values": []}
        )
        if int(state["task_id"]) != task:
            raise ValueError("CARE v3 family spans multiple tasks")
        if horizon in state["horizons"]:
            raise ValueError("CARE v3 family has duplicate collapsed horizon residuals")
        state["horizons"].add(horizon)
        state["values"].append(float(row[field]))
    if not grouped:
        raise ValueError("CARE v3 conformal family set is empty")
    result: dict[int, dict[str, Any]] = {}
    for family, state in grouped.items():
        if state["horizons"] != expected:
            missing = sorted(expected.difference(state["horizons"]))
            extra = sorted(state["horizons"].difference(expected))
            raise ValueError(
                f"CARE v3 family {family} lacks the complete horizon set "
                f"(missing={missing}, extra={extra})"
            )
        result[family] = {
            "task_id": int(state["task_id"]),
            "maximum": max(float(value) for value in state["values"]),
        }
    return result


def normalized_task_horizon_conformal_report(
    calibration_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    task_horizon_component_scales: torch.Tensor | Sequence[Any],
    horizons: Sequence[int],
    nominal: float = 0.90,
) -> dict[str, Any]:
    """Diagnose scale-normalized simultaneous q05 calibration.

    Scales are an explicit *training-family-only* input.  Calibration takes a
    max over every requested horizon within each held-out family before the
    finite-sample higher quantile is computed.  Validation is used only for
    reporting coverage.  This helper is deliberately diagnostic-only until a
    fully cross-fitted OOF run and paired closed-loop gate admit it.
    """

    if not 0.0 < float(nominal) <= 1.0:
        raise ValueError("CARE v3 nominal coverage must lie in (0,1]")
    scales, requested = _task_horizon_scales(
        task_horizon_component_scales, horizons
    )
    calibration = q05_family_horizon_residuals(
        calibration_rows,
        task_horizon_component_scales=scales,
        horizons=requested,
    )
    validation = q05_family_horizon_residuals(
        validation_rows,
        task_horizon_component_scales=scales,
        horizons=requested,
    )
    calibration_normalized = _complete_family_maxima(
        calibration,
        horizons=requested,
        field="q05_overestimate_normalized",
    )
    calibration_physical = _complete_family_maxima(
        calibration,
        horizons=requested,
        field="q05_overestimate_physical",
    )
    validation_normalized = _complete_family_maxima(
        validation,
        horizons=requested,
        field="q05_overestimate_normalized",
    )
    overlap = set(calibration_normalized).intersection(validation_normalized)
    if overlap:
        raise ValueError(
            f"CARE v3 calibration/validation families overlap: {sorted(overlap)}"
        )

    normalized_values = [
        float(value["maximum"]) for value in calibration_normalized.values()
    ]
    physical_values = [
        float(value["maximum"]) for value in calibration_physical.values()
    ]
    normalized_correction = max(
        0.0, _higher_quantile(normalized_values, float(nominal))
    )
    raw_physical_correction = max(
        0.0, _higher_quantile(physical_values, float(nominal))
    )
    validation_covered = [
        float(value["maximum"]) <= normalized_correction
        for value in validation_normalized.values()
    ]
    coverage_by_task: dict[int, list[bool]] = defaultdict(list)
    for value in validation_normalized.values():
        coverage_by_task[int(value["task_id"])].append(
            float(value["maximum"]) <= normalized_correction
        )

    physical_correction: dict[str, dict[str, float]] = {}
    efficiency: dict[str, dict[str, float | None]] = {}
    for task in range(scales.shape[0]):
        physical_correction[str(task)] = {}
        efficiency[str(task)] = {}
        for horizon_index, horizon in enumerate(requested):
            decoded = normalized_correction * float(scales[task, horizon_index, 2])
            physical_correction[str(task)][str(horizon)] = decoded
            efficiency[str(task)][str(horizon)] = (
                decoded / raw_physical_correction
                if raw_physical_correction > 0.0
                else (1.0 if decoded == 0.0 else None)
            )

    return {
        "nominal_simultaneous_coverage": float(nominal),
        "normalized_familywise_correction": normalized_correction,
        "raw_familywise_physical_correction": raw_physical_correction,
        "physical_correction_by_task_horizon": physical_correction,
        "physical_efficiency_ratio_vs_raw": efficiency,
        "calibration_family_count": len(calibration_normalized),
        "validation_family_count": len(validation_normalized),
        "validation_family_coverage": float(np.mean(validation_covered)),
        "validation_family_coverage_by_task_id": {
            str(task): float(np.mean(values))
            for task, values in sorted(coverage_by_task.items())
        },
        "family_max_includes_all_requested_horizons": True,
        "requested_horizons": list(requested),
        "candidate_zero_excluded": True,
        "correction_used_without_trimming": True,
        "nominal_coverage_unchanged": True,
        "scale_fitted_on_training_families_only": True,
        "calibration_fitted_without_validation": True,
        "calibration_validation_family_disjoint": True,
        "admission_thresholds_unchanged": True,
        "eligible_for_admission_or_deployment": False,
    }


def q05_extreme_family_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    snapshot_ids: Sequence[str] | None = None,
    nominal: float = 0.90,
    top_k: int = 8,
) -> dict[str, Any]:
    """Name correction-dominating families without deleting or clipping them."""

    if int(top_k) < 1:
        raise ValueError("CARE v3 q05 top_k must be positive")
    scores = _family_q05_scores(rows)
    ordered = sorted(
        scores.items(),
        key=lambda item: (-float(item[1]["q05_overestimate"]), item[0]),
    )
    values = [float(value["q05_overestimate"]) for value in scores.values()]
    full = max(0.0, _higher_quantile(values, float(nominal)))
    top: list[dict[str, Any]] = []
    for family, value in ordered[: int(top_k)]:
        row = {"family_index": family, **value}
        if snapshot_ids is not None:
            if family < 0 or family >= len(snapshot_ids):
                raise ValueError("CARE v3 snapshot id index is out of range")
            row["snapshot_id"] = str(snapshot_ids[family])
        top.append(row)
    remaining = [
        float(value["q05_overestimate"]) for _family, value in ordered[1:]
    ]
    return {
        "nominal_simultaneous_coverage": float(nominal),
        "family_count": len(scores),
        "full_familywise_correction": float(full),
        "correction_used_without_trimming": True,
        "top_families": top,
        "leave_one_extreme_out_diagnostic_only": {
            "excluded_family_index": int(ordered[0][0]),
            "correction": (
                None
                if not remaining
                else max(0.0, _higher_quantile(remaining, float(nominal)))
            ),
            "eligible_for_selection_or_deployment": False,
        },
    }


def _taskwise_selector_metrics(
    rows: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    safety_args: Mapping[str, Any],
) -> dict[str, Any]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["task_id"])].append(row)
    result = {}
    for task, task_rows in sorted(grouped.items()):
        correction = float(report["correction_by_task_id"][str(task)])
        result[str(task)] = metrics(
            task_rows, correction=correction, **dict(safety_args)
        )
    return result


def train_condition(
    prepared: PreparedCAREData,
    split: Mapping[str, tuple[int, ...]],
    condition: ConditionV3,
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
    validation = FamilyDataset(
        prepared, split["validation"], primary_horizon_only=True
    )
    calibration = FamilyDataset(
        prepared, split["calibration"], primary_horizon_only=True
    )
    train_safety = hard_safety_nonzero_count(train)
    calibration_safety = hard_safety_nonzero_count(calibration)
    safety_status = safety_supervision_status(train_safety, calibration_safety)
    effective_loss = CARELossV2Config(
        consistency_weight=loss_config.consistency_weight,
        candidate_ranking_weight=loss_config.candidate_ranking_weight,
        reference_ranking_weight=loss_config.reference_ranking_weight,
        safety_weight=loss_config.safety_weight if train_safety > 0 else 0.0,
        ranking_min_gap=loss_config.ranking_min_gap,
    )
    if condition.horizon_aware_scaling:
        scale_cpu = subset_horizon_scales(
            prepared,
            split["train"],
            quantile=scale_quantile,
            floor=scale_floor,
        )
        scale_contract = "fixed_train_task_horizon_component"
    else:
        scale_cpu = subset_scales(
            prepared,
            split["train"],
            quantile=scale_quantile,
            floor=scale_floor,
        )
        scale_contract = "fixed_train_task_component"
    scales = scale_cpu.to(device)
    action_std = tuple(float(value) for value in action_std)
    if len(action_std) != 8 or any(
        not math.isfinite(value) or value <= 0 for value in action_std
    ):
        raise ValueError("prepared action_std must contain eight positive finite values")
    config = CAREBeliefV3Config(
        variant="care",
        action_std=action_std,
        action_prefix_steps=1,
        task_count=len(prepared.tasks),
        use_candidate_slot_embedding=condition.candidate_slot_embedding,
        use_task_embedding=condition.task_embedding,
    )
    model = CAREBeliefV3Head(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(updates, 1), eta_min=3e-6
    )
    history: list[dict[str, Any]] = []
    for update in range(1, updates + 1):
        seed_everything(seed + 10_000_019 * update)
        batch = to_device(
            deterministic_batch(train, update, seed, batch_size), device
        )
        scale = _batch_scale(scales, batch)
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
                batch["task_id"],
                utility_scale=scale,
            )
            loss, _pieces = care_v2_training_loss(
                output,
                batch["target"],
                batch["hard_safety"],
                "care",
                target_scale=scale,
                loss_config=effective_loss,
                quantiles=config.quantiles,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite CARE v3 loss at {update}")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient):
            raise FloatingPointError(f"non-finite CARE v3 gradient at {update}")
        optimizer.step()
        scheduler.step()
        if update == updates or update % max(eval_every, 1) == 0:
            val_rows = prediction_rows(
                model,
                DataLoader(validation, batch_size=batch_size, shuffle=False),
                device,
                scales,
            )
            cal_rows = prediction_rows(
                model,
                DataLoader(calibration, batch_size=batch_size, shuffle=False),
                device,
                scales,
            )
            scalar_correction = conformal_correction(cal_rows)
            taskwise = taskwise_conformal_report(cal_rows, val_rows, nominal=0.90)
            safety_args = {
                "safety_supervision_degenerate": bool(
                    safety_status["safety_supervision_degenerate"]
                ),
                "safety_training_nonzero_count": train_safety,
                "safety_calibration_nonzero_count": calibration_safety,
            }
            history.append(
                {
                    "update": update,
                    "loss": float(loss.detach()),
                    "gradient_norm": float(gradient),
                    "validation": metrics(val_rows, **safety_args),
                    "validation_global_calibrated": metrics(
                        val_rows,
                        correction=scalar_correction,
                        **safety_args,
                    ),
                    "validation_taskwise_calibrated": _taskwise_selector_metrics(
                        val_rows, taskwise, safety_args
                    ),
                    "taskwise_conformal": taskwise,
                    "q05_extreme_family_diagnostics": q05_extreme_family_diagnostics(
                        cal_rows,
                        snapshot_ids=prepared.snapshot_ids,
                        nominal=0.90,
                    ),
                    "calibration": metrics(cal_rows, **safety_args),
                }
            )
    return {
        "condition": asdict(condition),
        "seed": int(seed),
        "updates": int(updates),
        "train_families": list(split["train"]),
        "validation_families": list(split["validation"]),
        "calibration_families": list(split["calibration"]),
        "scale_contract": scale_contract,
        "train_scales": scale_cpu.tolist(),
        "requested_loss_config": loss_config.to_dict(),
        "effective_loss_config": effective_loss.to_dict(),
        "safety_supervision": safety_status,
        "history": history,
        "final": history[-1],
        "candidate0_bit_exact": True,
        "reference_policy_unchanged": True,
        "paired_target_definition_unchanged": True,
        "quantile_and_calibration_definition_unchanged": True,
        "admission_thresholds_unchanged": True,
        "same_corpus_diagnostic_only": True,
        "formal_promotion_requires_fresh_run": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    parser.add_argument("--seeds", default="20260904,20260905,20260906")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--scale-quantile", type=float, default=0.90)
    parser.add_argument("--scale-floor", type=float, default=1e-4)
    parser.add_argument("--prepared-sha256", default="")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.updates < 1:
        raise ValueError("CARE v3 diagnostic updates must be positive")
    prepared_hash = sha256_file(args.prepared_data)
    if args.prepared_sha256 and args.prepared_sha256 != prepared_hash:
        raise ValueError("prepared-data SHA256 mismatch")
    prepared = load_prepared_care(args.prepared_data)
    raw = torch.load(args.prepared_data, map_location="cpu", weights_only=False)
    action_std = tuple(
        float(value)
        for value in raw.get(
            "action_std", prepared.manifest.get("action_std", (1.0,) * 8)
        )
    )
    split = deterministic_split(prepared)
    conditions = parse_conditions(args.conditions)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise ValueError("CARE v3 diagnostic requires at least one seed")
    device = torch.device(args.device)
    loss_config = CARELossV2Config()
    results: list[dict[str, Any]] = []
    for condition in conditions:
        for seed in seeds:
            print(
                json.dumps(
                    {"event": "condition_start", "condition": asdict(condition), "seed": seed},
                    sort_keys=True,
                ),
                flush=True,
            )
            result = train_condition(
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
            results.append(result)
            print(
                json.dumps(
                    {"event": "condition_complete", "condition": condition.name, "seed": seed},
                    sort_keys=True,
                ),
                flush=True,
            )
    payload = {
        "format_version": SCRIPT_FORMAT,
        "created_at_utc": utc_now(),
        "prepared_data": str(args.prepared_data.resolve()),
        "prepared_data_sha256": prepared_hash,
        "family_count": len(prepared.snapshot_ids),
        "tasks": list(prepared.tasks),
        "split": {key: list(value) for key, value in split.items()},
        "conditions": [asdict(value) for value in conditions],
        "seeds": list(seeds),
        "updates": int(args.updates),
        "batch_size": int(args.batch_size),
        "ablation_axes": [
            "candidate_slot_embedding",
            "task_embedding",
            "task_horizon_component_scaling",
        ],
        "care_theory_contract_unchanged": True,
        "legacy_and_v2_formal_runs_unchanged": True,
        "validation20_used_for_tuning": False,
        "results": results,
    }
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {"status": "complete", "output": str(args.output.resolve())},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
