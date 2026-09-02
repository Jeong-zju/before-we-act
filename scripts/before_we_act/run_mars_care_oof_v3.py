#!/usr/bin/env python3
"""Train and aggregate strict family-disjoint CARE-v3 OOF scorers.

This runner is the promotion bridge between the small three-way scorer
diagnostic and a deployable CARE checkpoint.  Every held-out family is scored
by a model that was fit on the other four folds, and the emitted JSON carries
the explicit ``fit_families`` evidence required by :mod:`oof_care_gate`.

The runner deliberately does not read Validation20 results, alter CARE's
direct/response/total targets, remove difficult families, or relax any gate.
It supports independent fold/seed jobs so a single-shot supervisor can spread
the 5 x N fits across all GPUs.  ``aggregate`` averages the pre-registered
seed ensemble only after checking that targets and provenance agree exactly.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from before_we_act.care_belief import CARE_HORIZONS
from before_we_act.care_belief_v2 import CARELossV2Config, care_v2_training_loss
from before_we_act.care_belief_v3 import CAREBeliefV3Config, CAREBeliefV3Head
from before_we_act.care_training_data import (
    PreparedCAREData,
    atomic_json,
    load_prepared_care,
    sha256_file,
)
from scripts.before_we_act.analyze_mars_care_scorer_v2 import (
    FamilyDataset,
    deterministic_batch,
    hard_safety_nonzero_count,
    seed_everything,
    to_device,
)
from scripts.before_we_act.analyze_mars_care_scorer_v3 import (
    _batch_scale,
    prediction_rows,
    subset_horizon_scales,
)
from scripts.before_we_act.oof_care_gate import (
    fit_oof_calibration,
    fold_manifest,
    make_family_folds,
    oof_metrics,
    validate_oof_rows,
)


FORMAT_VERSION = "before-we-act.care-mars-oof-training-v3/2-all-horizon"
PREDICTION_FORMAT_VERSION = "before-we-act.care-mars-oof-predictions-v3/2-all-horizon"
DEFAULT_FOLD_SEED = 20260901
DEFAULT_SEEDS = (20260904, 20260905, 20260906)
DEFAULT_FOLDS = 5
PRIMARY_HORIZON = 16
PRIMARY_HORIZON_INDEX = CARE_HORIZONS.index(PRIMARY_HORIZON)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _json_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def family_folds(
    prepared: PreparedCAREData,
    *,
    n_splits: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_FOLD_SEED,
) -> dict[int, int]:
    """Return the one canonical task-stratified OOF assignment."""

    return make_family_folds(
        prepared.task_id.tolist(),
        prepared.snapshot_ids,
        n_splits=n_splits,
        seed=seed,
    )


def _fit_families(
    fold_by_family: Mapping[int, int], heldout_fold: int
) -> tuple[int, ...]:
    heldout_fold = int(heldout_fold)
    folds = set(int(value) for value in fold_by_family.values())
    if heldout_fold not in folds:
        raise ValueError(f"unknown held-out fold {heldout_fold}")
    return tuple(
        family
        for family, fold in sorted(
            (int(key), int(value)) for key, value in fold_by_family.items()
        )
        if fold != heldout_fold
    )


def _heldout_families(
    fold_by_family: Mapping[int, int], heldout_fold: int
) -> tuple[int, ...]:
    return tuple(
        family
        for family, fold in sorted(
            (int(key), int(value)) for key, value in fold_by_family.items()
        )
        if fold == int(heldout_fold)
    )


def aggregate_repeat_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    fold: int,
    fit_families: Sequence[int],
    snapshot_ids: Sequence[str],
    task_horizon_component_scales: torch.Tensor | Sequence[Any] | None = None,
    horizons: Sequence[int] = CARE_HORIZONS,
) -> list[dict[str, Any]]:
    """Collapse the two stochastic repeats into one auditable family row.

    The scorer input is identical for both repeats, so quantiles and logits
    should agree up to batching precision.  Averaging them is harmless and
    makes the contract robust to a future stochastic scorer.  Utility labels
    are averaged, while a safety violation in either repeat is retained.

    Rows are grouped by ``(family, horizon)`` rather than just family.  This
    is important for the simultaneous CARE guarantee: H8/H16/H32/H64 are
    distinct outcomes and may not be silently collapsed to the primary H16
    row.  The optional scale argument is recorded for auditability and is
    required by the full-horizon OOF path; it is optional only for backwards
    compatibility with the small unit tests that exercise repeat averaging.
    """

    requested_horizons = tuple(int(value) for value in horizons)
    if not requested_horizons or len(set(requested_horizons)) != len(requested_horizons):
        raise ValueError("OOF horizons must be a non-empty unique sequence")
    scales: torch.Tensor | None = None
    if task_horizon_component_scales is not None:
        scales = torch.as_tensor(task_horizon_component_scales).float().cpu()
        if (
            scales.ndim != 3
            or scales.shape[1] != len(requested_horizons)
            or scales.shape[2] < 3
            or not torch.isfinite(scales).all()
            or bool((scales <= 0).any())
        ):
            raise ValueError(
                "OOF task/horizon scales must be positive [task,horizon,component]"
            )
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        family = int(row["family_index"])
        horizon_index = int(row.get("horizon_index", PRIMARY_HORIZON_INDEX))
        grouped[(family, horizon_index)].append(row)
    if not grouped:
        raise ValueError("OOF fold produced no prediction rows")
    fit = tuple(sorted({int(value) for value in fit_families}))
    result: list[dict[str, Any]] = []
    for (family, horizon_index), family_rows in sorted(grouped.items()):
        if len(family_rows) != 2:
            raise ValueError(
                f"OOF family {family}/horizon {horizon_index} must contain "
                "exactly two repeat rows"
            )
        tasks = {int(row["task_id"]) for row in family_rows}
        if len(tasks) != 1:
            raise ValueError(f"OOF family {family} spans multiple tasks")
        if not 0 <= horizon_index < len(requested_horizons):
            raise ValueError("OOF horizon index is out of range")
        task = next(iter(tasks))
        if scales is not None and not 0 <= task < scales.shape[0]:
            raise ValueError("OOF task id is outside the scale tensor")
        quantiles = torch.stack(
            [torch.as_tensor(row["quantiles"]).float() for row in family_rows]
        ).mean(0)
        target = torch.stack(
            [torch.as_tensor(row["target"]).float() for row in family_rows]
        ).mean(0)
        safety_logit = torch.stack(
            [torch.as_tensor(row["hard_safety_logit"]).float() for row in family_rows]
        ).mean(0)
        hard_safety = torch.stack(
            [torch.as_tensor(row["hard_safety"]).float() for row in family_rows]
        ).amax(0)
        if quantiles.shape != (6, 3, 5) or target.shape != (6, 3):
            raise ValueError(f"OOF family {family} tensor contract drifted")
        if not torch.equal(quantiles[0], torch.zeros_like(quantiles[0])):
            raise ValueError(f"OOF family {family} lost candidate-zero identity")
        if family < 0 or family >= len(snapshot_ids):
            raise ValueError("OOF family index is outside snapshot ids")
        result.append(
            {
                "family_index": family,
                "snapshot_id": str(snapshot_ids[family]),
                "fold": int(fold),
                "fit_families": list(fit),
                "task_id": task,
                "horizon_index": horizon_index,
                "horizon_steps": requested_horizons[horizon_index],
                "quantiles": quantiles,
                "target": target,
                "hard_safety_logit": safety_logit,
                "hard_safety": hard_safety,
                # The prepared formal corpus was admitted only after every
                # fixed candidate passed the physical legality audit.
                "candidate_legality": torch.ones(6, dtype=torch.bool),
            }
        )
        if scales is not None:
            result[-1]["total_utility_scale"] = float(
                scales[task, horizon_index, 2]
            )
    return result


def _horizon_key(row: Mapping[str, Any]) -> tuple[int, int]:
    """Return the immutable family/horizon key used by strict OOF rows."""

    if "horizon_index" not in row:
        raise ValueError("strict all-horizon OOF row is missing horizon_index")
    return int(row["family_index"]), int(row["horizon_index"])


def validate_oof_horizon_rows(
    rows: Iterable[Mapping[str, Any]],
    fold_by_family: Mapping[int, int],
    *,
    horizons: Sequence[int] = CARE_HORIZONS,
    require_complete_family_coverage: bool = True,
) -> list[dict[str, Any]]:
    """Validate one row per ``family × horizon`` without losing provenance.

    ``oof_care_gate.validate_oof_rows`` intentionally rejects duplicate family
    rows because its legacy contract is one row per family.  The scorer-v3
    protocol needs four rows per family so that calibration can take a
    simultaneous maximum over H8/H16/H32/H64.  This wrapper applies the same
    leakage/shape checks independently to each horizon and then enforces the
    complete horizon set.
    """

    requested = tuple(int(value) for value in horizons)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("OOF horizons must be a non-empty unique sequence")
    source = list(rows)
    if not source:
        raise ValueError("all-horizon OOF prediction set is empty")
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for row in source:
        family, horizon_index = _horizon_key(row)
        if horizon_index < 0 or horizon_index >= len(requested):
            raise ValueError(
                f"OOF horizon index {horizon_index} is outside 0..{len(requested)-1}"
            )
        declared_steps = row.get("horizon_steps")
        if declared_steps is not None and int(declared_steps) != requested[horizon_index]:
            raise ValueError(
                f"OOF family {family} horizon label disagrees with horizon index"
            )
        key = (family, horizon_index)
        if key in by_key:
            raise ValueError(
                f"duplicate OOF prediction for family {family}/horizon {horizon_index}"
            )
        # Validate the original one-family contract while retaining the extra
        # horizon metadata in the normalized row below.
        checked = validate_oof_rows(
            [row],
            fold_by_family,
            require_complete_family_coverage=False,
        )[0]
        checked["horizon_index"] = horizon_index
        checked["horizon_steps"] = requested[horizon_index]
        scale = row.get("total_utility_scale")
        if scale is None:
            raise ValueError(
                f"OOF family {family}/horizon {horizon_index} lacks total_utility_scale"
            )
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("OOF total_utility_scale must be finite and positive")
        checked["total_utility_scale"] = scale
        by_key[key] = checked

    assignments = {int(key): int(value) for key, value in fold_by_family.items()}
    expected_horizons = set(range(len(requested)))
    by_family: dict[int, set[int]] = defaultdict(set)
    for family, horizon_index in by_key:
        by_family[family].add(horizon_index)
    if require_complete_family_coverage:
        missing_families = set(assignments).difference(by_family)
        if missing_families:
            raise ValueError(
                "all-horizon OOF predictions omit families "
                f"{sorted(missing_families)}"
            )
        incomplete = {
            family: sorted(expected_horizons.difference(indices))
            for family, indices in by_family.items()
            if indices != expected_horizons
        }
        if incomplete:
            raise ValueError(
                "all-horizon OOF predictions omit horizons by family: "
                f"{incomplete}"
            )
    return [by_key[key] for key in sorted(by_key)]


def _higher_quantile(values: Sequence[float], probability: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot compute an OOF quantile from an empty set")
    try:
        return float(np.quantile(array, probability, method="higher"))
    except TypeError:  # pragma: no cover - NumPy < 1.22 compatibility
        return float(np.quantile(array, probability, interpolation="higher"))


def _horizon_family_maxima(
    rows: Sequence[Mapping[str, Any]],
    *,
    horizons: Sequence[int] = CARE_HORIZONS,
) -> dict[int, dict[str, Any]]:
    """Compute untrimmed normalized q05 maxima for each complete family."""

    requested = tuple(int(value) for value in horizons)
    # Rows reaching this helper have already passed
    # ``validate_oof_horizon_rows`` against the complete assignment.  Do not
    # reconstruct a partial fold manifest here: a calibration subset may omit
    # fold zero, and treating its sparse fold labels as a new 0..K manifest
    # would reject a valid subset for the wrong reason.
    normalized = list(rows)
    if not normalized:
        raise ValueError("OOF horizon family maxima are empty")
    grouped: dict[int, dict[str, Any]] = {}
    for row in normalized:
        family = int(row["family_index"])
        horizon_index = int(row["horizon_index"])
        task = int(row["task_id"])
        quantiles = np.asarray(row["quantiles"], dtype=np.float64)
        target = np.asarray(row["target"], dtype=np.float64)
        if quantiles.shape != (6, 3, 5) or target.shape != (6, 3):
            raise ValueError("OOF horizon prediction/target shape differs")
        legality = np.asarray(
            row.get("candidate_legality", np.ones(6, dtype=bool)), dtype=bool
        )
        if legality.shape != (6,) or not bool(legality[0]):
            raise ValueError("OOF horizon row has an invalid candidate-legality mask")
        # Illegal candidates must never influence the conformal correction.
        # They are not executable evidence and can otherwise inflate the
        # calibration bound enough to hide a selector regression.
        errors = (quantiles[1:, 2, 0] - target[1:, 2])[legality[1:]]
        if errors.size and not np.isfinite(errors).all():
            raise ValueError("OOF horizon row has non-finite learned-candidate errors")
        physical = max(0.0, float(np.max(errors))) if errors.size else 0.0
        scale = float(row["total_utility_scale"])
        state = grouped.setdefault(
            family,
            {"task_id": task, "horizons": {}, "maximum": 0.0, "physical_maximum": 0.0},
        )
        if int(state["task_id"]) != task:
            raise ValueError("OOF family spans multiple tasks")
        if horizon_index in state["horizons"]:
            raise ValueError("OOF family has duplicate horizon rows")
        normalized_error = physical / scale
        state["horizons"][horizon_index] = normalized_error
        state["maximum"] = max(float(state["maximum"]), normalized_error)
        state["physical_maximum"] = max(float(state["physical_maximum"]), physical)
    expected = set(range(len(requested)))
    for family, state in grouped.items():
        seen = set(state["horizons"])
        if seen != expected:
            raise ValueError(
                f"OOF family {family} lacks complete horizon set "
                f"(missing={sorted(expected - seen)}, extra={sorted(seen - expected)})"
            )
        state["horizons"] = {
            str(index): float(value)
            for index, value in sorted(state["horizons"].items())
        }
    if not grouped:
        raise ValueError("OOF horizon family maxima are empty")
    return grouped


def fit_oof_horizon_calibration(
    rows: Sequence[Mapping[str, Any]],
    fold_by_family: Mapping[int, int],
    *,
    horizons: Sequence[int] = CARE_HORIZONS,
    nominal: float = 0.90,
) -> dict[str, Any]:
    """Fit a strict family-wise correction over every requested horizon.

    Each family is the statistical unit.  For every held-out fold, the
    correction is fitted only on complete family maxima from the other folds;
    validation coverage then checks the held-out complete maxima.  No family,
    horizon, or extreme residual is trimmed.
    """

    if not 0.0 < float(nominal) <= 1.0:
        raise ValueError("OOF nominal coverage must lie in (0,1]")
    requested = tuple(int(value) for value in horizons)
    normalized = validate_oof_horizon_rows(
        rows, fold_by_family, horizons=requested
    )
    assignments = {int(key): int(value) for key, value in fold_by_family.items()}
    full_maxima = _horizon_family_maxima(normalized, horizons=requested)
    values = [float(state["maximum"]) for state in full_maxima.values()]
    adjusted = min(1.0, math.ceil((len(values) + 1) * float(nominal)) / len(values))
    global_correction = max(0.0, _higher_quantile(values, adjusted))
    folds = sorted(set(assignments.values()))
    fold_corrections: dict[str, float] = {}
    correction_by_family: dict[str, float] = {}
    coverage_by_fold: dict[str, float] = {}
    coverage_by_task: dict[str, list[bool]] = defaultdict(list)
    covered: list[bool] = []
    for fold in folds:
        fit_rows = [row for row in normalized if int(row["fold"]) != fold]
        eval_rows = [row for row in normalized if int(row["fold"]) == fold]
        fit_maxima = _horizon_family_maxima(fit_rows, horizons=requested)
        eval_maxima = _horizon_family_maxima(eval_rows, horizons=requested)
        fit_values = [float(state["maximum"]) for state in fit_maxima.values()]
        fit_adjusted = min(
            1.0,
            math.ceil((len(fit_values) + 1) * float(nominal)) / len(fit_values),
        )
        correction = max(0.0, _higher_quantile(fit_values, fit_adjusted))
        fold_corrections[str(fold)] = correction
        for family, state in eval_maxima.items():
            ok = float(state["maximum"]) <= correction
            correction_by_family[str(family)] = correction
            covered.append(ok)
            coverage_by_task[str(int(state["task_id"]))].append(ok)
        coverage_by_fold[str(fold)] = float(
            np.mean(
                [
                    float(state["maximum"]) <= correction
                    for state in eval_maxima.values()
                ]
            )
        )
    return {
        "normalized_familywise_correction": float(global_correction),
        "nominal_simultaneous_coverage": float(nominal),
        "family_count": len(full_maxima),
        "requested_horizons": list(requested),
        "family_max_includes_all_requested_horizons": True,
        "residuals_trimmed_or_clipped": False,
        "crossfit_family_coverage": float(np.mean(covered)) if covered else 0.0,
        "crossfit_coverage_by_fold": coverage_by_fold,
        "crossfit_coverage_by_task_id": {
            key: float(np.mean(value)) for key, value in sorted(coverage_by_task.items())
        },
        "crossfit_correction_by_family": correction_by_family,
        "crossfit_fold_corrections": fold_corrections,
        "calibration_family_disjoint": True,
        "calibration_fitted_without_validation": True,
        "eligible_for_deployment": False,
    }


@dataclass(frozen=True)
class FoldTrainingConfig:
    updates: int = 4000
    batch_size: int = 48
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    scale_quantile: float = 0.90
    scale_floor: float = 1e-4
    action_prefix_steps: int = 1

    def __post_init__(self) -> None:
        if int(self.updates) < 1 or int(self.batch_size) < 1:
            raise ValueError("OOF updates and batch size must be positive")
        if not math.isfinite(float(self.learning_rate)) or self.learning_rate <= 0:
            raise ValueError("OOF learning rate must be positive")
        if not math.isfinite(float(self.weight_decay)) or self.weight_decay < 0:
            raise ValueError("OOF weight decay must be non-negative")
        if not 0.0 < float(self.scale_quantile) <= 1.0:
            raise ValueError("OOF scale quantile must lie in (0,1]")
        if not math.isfinite(float(self.scale_floor)) or self.scale_floor <= 0:
            raise ValueError("OOF scale floor must be positive")
        if int(self.action_prefix_steps) not in {1, 4, 8, 16}:
            raise ValueError(
                "OOF action prefix must use a registered intervention window"
            )


def train_fold(
    prepared: PreparedCAREData,
    fold_by_family: Mapping[int, int],
    *,
    heldout_fold: int,
    seed: int,
    device: torch.device,
    config: FoldTrainingConfig,
    prepared_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit one fold/seed scorer and return checkpoint plus family rows."""

    fit = _fit_families(fold_by_family, heldout_fold)
    heldout = _heldout_families(fold_by_family, heldout_fold)
    if set(fit) & set(heldout):
        raise AssertionError("OOF fit and held-out families overlap")
    seed_everything(int(seed))
    prepared_intervention = int(
        prepared.manifest.get(
            "intervention_steps",
            prepared.manifest.get("branch_intervention_steps", 1),
        )
    )
    if prepared_intervention != int(config.action_prefix_steps):
        raise ValueError(
            "OOF action prefix does not match prepared branch intervention: "
            f"prefix={config.action_prefix_steps}, branch={prepared_intervention}"
        )
    train = FamilyDataset(prepared, fit)
    # Every held-out family must contribute all usable outcome horizons.  The
    # previous implementation silently restricted this dataset to H16, which
    # made the advertised simultaneous H8/H16/H32/H64 OOF claim unverifiable.
    evaluation = FamilyDataset(prepared, heldout)
    scales_cpu = subset_horizon_scales(
        prepared,
        fit,
        quantile=config.scale_quantile,
        floor=config.scale_floor,
    )
    scales = scales_cpu.to(device)
    raw_std = prepared.manifest.get("action_std", (1.0,) * 8)
    action_std = tuple(float(value) for value in raw_std)
    if action_std == (1.0,) * 8:
        raise ValueError("OOF prepared manifest must carry fitted action_std")
    model_config = CAREBeliefV3Config(
        variant="care",
        action_std=action_std,
        action_prefix_steps=config.action_prefix_steps,
        task_count=len(prepared.tasks),
        use_candidate_slot_embedding=True,
        use_task_embedding=True,
    )
    model = CAREBeliefV3Head(model_config).to(device)
    requested_loss = CARELossV2Config()
    safety_positive = hard_safety_nonzero_count(train)
    effective_loss = CARELossV2Config(
        consistency_weight=requested_loss.consistency_weight,
        candidate_ranking_weight=requested_loss.candidate_ranking_weight,
        reference_ranking_weight=requested_loss.reference_ranking_weight,
        safety_weight=requested_loss.safety_weight if safety_positive > 0 else 0.0,
        ranking_min_gap=requested_loss.ranking_min_gap,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.updates,
        eta_min=3e-6,
    )
    last_loss = last_gradient = math.nan
    for update in range(1, config.updates + 1):
        # Make a fold/seed job bit-reproducible even if jobs are scheduled in
        # a different GPU order by the supervisor.
        seed_everything(int(seed) + 10_000_019 * update)
        batch = to_device(
            deterministic_batch(train, update, int(seed), config.batch_size),
            device,
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
                quantiles=model_config.quantiles,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite OOF loss fold={heldout_fold} seed={seed} update={update}"
            )
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient):
            raise FloatingPointError(
                f"non-finite OOF gradient fold={heldout_fold} seed={seed} update={update}"
            )
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.detach())
        last_gradient = float(gradient)

    repeat_rows = prediction_rows(
        model,
        DataLoader(evaluation, batch_size=config.batch_size, shuffle=False),
        device,
        scales,
    )
    family_rows = aggregate_repeat_rows(
        repeat_rows,
        fold=heldout_fold,
        fit_families=fit,
        snapshot_ids=prepared.snapshot_ids,
        task_horizon_component_scales=scales_cpu,
        horizons=CARE_HORIZONS,
    )
    checkpoint = {
        "format_version": FORMAT_VERSION,
        "created_at_utc": utc_now(),
        "prepared_data_sha256": str(prepared_sha256),
        "fold": int(heldout_fold),
        "seed": int(seed),
        "fit_families": list(fit),
        "heldout_families": list(heldout),
        "updates": int(config.updates),
        "training_config": asdict(config),
        "model_config": model_config.to_dict(),
        "model": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "task_horizon_component_scales": scales_cpu,
        "evaluated_horizons": list(CARE_HORIZONS),
        "requested_loss_config": requested_loss.to_dict(),
        "effective_loss_config": effective_loss.to_dict(),
        "safety_positive_label_count": int(safety_positive),
        "last_loss": last_loss,
        "last_gradient_norm": last_gradient,
        "family_disjoint": True,
        "validation20_used_for_tuning": False,
        "candidate0_bit_exact": True,
    }
    return checkpoint, family_rows


def train_final(
    prepared: PreparedCAREData,
    *,
    seed: int,
    device: torch.device,
    config: FoldTrainingConfig,
    prepared_sha256: str,
) -> dict[str, Any]:
    """Fit the promoted scorer once on all families after strict OOF.

    OOF remains the only source of calibration evidence.  This final fit is
    deliberately a separate, all-family model: it never evaluates or tunes
    on Validation20 and carries the exact H8 prefix used by the corpus.
    """

    prepared_intervention = int(
        prepared.manifest.get(
            "intervention_steps",
            prepared.manifest.get("branch_intervention_steps", 1),
        )
    )
    if prepared_intervention != int(config.action_prefix_steps):
        raise ValueError(
            "final action prefix does not match prepared branch intervention: "
            f"prefix={config.action_prefix_steps}, branch={prepared_intervention}"
        )
    seed_everything(int(seed))
    train = FamilyDataset(prepared, tuple(range(len(prepared.snapshot_ids))))
    scales_cpu = subset_horizon_scales(
        prepared,
        tuple(range(len(prepared.snapshot_ids))),
        quantile=config.scale_quantile,
        floor=config.scale_floor,
    )
    scales = scales_cpu.to(device)
    raw_std = prepared.manifest.get("action_std", (1.0,) * 8)
    action_std = tuple(float(value) for value in raw_std)
    if action_std == (1.0,) * 8:
        raise ValueError("final prepared manifest must carry fitted action_std")
    model_config = CAREBeliefV3Config(
        variant="care",
        action_std=action_std,
        action_prefix_steps=config.action_prefix_steps,
        task_count=len(prepared.tasks),
        use_candidate_slot_embedding=True,
        use_task_embedding=True,
    )
    model = CAREBeliefV3Head(model_config).to(device)
    requested_loss = CARELossV2Config()
    safety_positive = hard_safety_nonzero_count(train)
    effective_loss = CARELossV2Config(
        consistency_weight=requested_loss.consistency_weight,
        candidate_ranking_weight=requested_loss.candidate_ranking_weight,
        reference_ranking_weight=requested_loss.reference_ranking_weight,
        safety_weight=requested_loss.safety_weight if safety_positive > 0 else 0.0,
        ranking_min_gap=requested_loss.ranking_min_gap,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.updates, eta_min=3e-6
    )
    last_loss = last_gradient = math.nan
    for update in range(1, config.updates + 1):
        seed_everything(int(seed) + 10_000_019 * update)
        batch = to_device(
            deterministic_batch(train, update, int(seed), config.batch_size), device
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
                quantiles=model_config.quantiles,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite final CARE-v3 loss at update {update}")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient):
            raise FloatingPointError(
                f"non-finite final CARE-v3 gradient at update {update}"
            )
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.detach())
        last_gradient = float(gradient)
    return {
        "format_version": "before-we-act.care-mars-final-training-v3/1",
        "created_at_utc": utc_now(),
        "prepared_data_sha256": str(prepared_sha256),
        "seed": int(seed),
        "updates": int(config.updates),
        "training_config": asdict(config),
        "model_config": model_config.to_dict(),
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "task_horizon_component_scales": scales_cpu,
        "evaluated_horizons": list(CARE_HORIZONS),
        "requested_loss_config": requested_loss.to_dict(),
        "effective_loss_config": effective_loss.to_dict(),
        "safety_positive_label_count": int(safety_positive),
        "last_loss": last_loss,
        "last_gradient_norm": last_gradient,
        "all_family_training": True,
        "family_count": len(prepared.snapshot_ids),
        "action_prefix_steps": int(config.action_prefix_steps),
        "family_disjoint_oof_before_final_fit": True,
        "validation20_used_for_tuning": False,
        "candidate0_bit_exact": True,
        "care_theory_contract_unchanged": True,
    }


def _row_tensor(row: Mapping[str, Any], key: str) -> torch.Tensor:
    return torch.as_tensor(row[key]).float()


def ensemble_seed_rows(
    seed_rows: Sequence[Sequence[Mapping[str, Any]]],
    *,
    expected_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """Average OOF predictions across fixed seeds after strict parity checks."""

    if len(seed_rows) != len(tuple(expected_seeds)) or not seed_rows:
        raise ValueError("OOF ensemble is missing a pre-registered seed")
    def key(row: Mapping[str, Any]) -> tuple[int, int]:
        if "horizon_index" not in row:
            raise ValueError("OOF seed row is missing horizon_index")
        return int(row["family_index"]), int(row["horizon_index"])

    indexed: list[dict[tuple[int, int], Mapping[str, Any]]] = []
    for rows in seed_rows:
        current: dict[tuple[int, int], Mapping[str, Any]] = {}
        for row in rows:
            row_key = key(row)
            if row_key in current:
                raise ValueError(
                    "OOF seed contains duplicate prediction for family/horizon "
                    f"{row_key}"
                )
            current[row_key] = row
        indexed.append(current)
    families = set(indexed[0])
    if not families or any(set(value) != families for value in indexed[1:]):
        raise ValueError("OOF seed prediction family/horizon coverage differs")
    result: list[dict[str, Any]] = []
    for family_horizon in sorted(families):
        rows = [value[family_horizon] for value in indexed]
        first = rows[0]
        stable_keys = (
            "fold",
            "fit_families",
            "task_id",
            "snapshot_id",
            "horizon_index",
            "horizon_steps",
            "total_utility_scale",
        )
        for row in rows[1:]:
            if any(row.get(key) != first.get(key) for key in stable_keys):
                raise ValueError(
                    "OOF seed provenance differs for family/horizon "
                    f"{family_horizon}"
                )
            for key in ("target", "hard_safety", "candidate_legality"):
                if not torch.equal(torch.as_tensor(row[key]), torch.as_tensor(first[key])):
                    raise ValueError(
                        f"OOF seed {key} differs for family/horizon {family_horizon}"
                    )
        merged = dict(first)
        merged["quantiles"] = torch.stack(
            [_row_tensor(row, "quantiles") for row in rows]
        ).mean(0)
        merged["hard_safety_logit"] = torch.stack(
            [_row_tensor(row, "hard_safety_logit") for row in rows]
        ).mean(0)
        merged["ensemble_seeds"] = [int(value) for value in expected_seeds]
        result.append(merged)
    return result


def _read_prediction_rows(
    path: Path,
    *,
    expected_prepared_data_sha256: str,
    expected_fold: int,
    expected_seed: int,
) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format_version") != PREDICTION_FORMAT_VERSION:
        raise ValueError(f"wrong OOF prediction format: {path}")
    if value.get("prepared_data_sha256") != str(expected_prepared_data_sha256):
        raise ValueError(f"OOF prepared-data hash mismatch: {path}")
    if int(value.get("fold", -1)) != int(expected_fold):
        raise ValueError(f"OOF top-level fold mismatch: {path}")
    if int(value.get("seed", -1)) != int(expected_seed):
        raise ValueError(f"OOF top-level seed mismatch: {path}")
    checkpoint_path = path.parent / "checkpoint.pt"
    declared_checkpoint = value.get("checkpoint")
    if declared_checkpoint is None:
        raise ValueError(f"OOF checkpoint path is missing: {path}")
    try:
        declared_checkpoint_path = Path(str(declared_checkpoint)).resolve()
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"OOF checkpoint path is invalid: {path}") from error
    if declared_checkpoint_path != checkpoint_path.resolve():
        raise ValueError(f"OOF checkpoint path mismatch: {path}")
    declared_checkpoint_sha256 = value.get("checkpoint_sha256")
    if not isinstance(declared_checkpoint_sha256, str) or not declared_checkpoint_sha256:
        raise ValueError(f"OOF checkpoint hash is missing: {path}")
    if not checkpoint_path.is_file():
        raise ValueError(f"OOF checkpoint is missing: {checkpoint_path}")
    if sha256_file(checkpoint_path) != declared_checkpoint_sha256:
        raise ValueError(f"OOF checkpoint hash mismatch: {path}")
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"OOF prediction rows missing: {path}")
    return rows


def aggregate_jobs(
    prepared: PreparedCAREData,
    root: Path,
    *,
    seeds: Sequence[int],
    n_splits: int,
    fold_seed: int,
    nominal: float,
    prepared_data_sha256: str,
) -> dict[str, Any]:
    if not isinstance(prepared_data_sha256, str) or not prepared_data_sha256:
        raise ValueError("aggregate requires the prepared-data file SHA256")
    assignments = family_folds(
        prepared, n_splits=n_splits, seed=fold_seed
    )
    rows_by_seed: list[list[dict[str, Any]]] = []
    for seed in seeds:
        current: list[dict[str, Any]] = []
        for fold in range(n_splits):
            path = root / f"fold_{fold}" / f"seed_{int(seed)}" / "predictions.json"
            current.extend(
                _read_prediction_rows(
                    path,
                    expected_prepared_data_sha256=prepared_data_sha256,
                    expected_fold=fold,
                    expected_seed=int(seed),
                )
            )
        # Validate every seed independently before it can enter the ensemble.
        # The full-horizon contract requires four rows per family; selecting
        # H16 happens only after this check for legacy ranking diagnostics.
        validate_oof_horizon_rows(current, assignments, horizons=CARE_HORIZONS)
        rows_by_seed.append(current)
    ensemble = ensemble_seed_rows(rows_by_seed, expected_seeds=seeds)
    normalized_horizon = validate_oof_horizon_rows(
        ensemble, assignments, horizons=CARE_HORIZONS
    )
    primary_rows = [
        row
        for row in normalized_horizon
        if int(row["horizon_index"]) == PRIMARY_HORIZON_INDEX
    ]
    if len(primary_rows) != len(assignments):
        raise ValueError("OOF primary horizon rows do not cover every family")
    normalized = validate_oof_rows(primary_rows, assignments)
    calibration = fit_oof_calibration(normalized, assignments, nominal=nominal)
    horizon_calibration = fit_oof_horizon_calibration(
        normalized_horizon,
        assignments,
        horizons=CARE_HORIZONS,
        nominal=nominal,
    )
    legacy_primary_metrics = oof_metrics(
        normalized,
        family_corrections=calibration.crossfit_correction_by_family,
    )
    primary_scale_by_family = {
        str(int(row["family_index"])): float(row["total_utility_scale"])
        for row in normalized_horizon
        if int(row["horizon_index"]) == PRIMARY_HORIZON_INDEX
    }
    horizon_primary_corrections = {
        family: float(normalized_correction) * primary_scale_by_family[family]
        for family, normalized_correction in horizon_calibration[
            "crossfit_correction_by_family"
        ].items()
    }
    metrics = oof_metrics(
        normalized,
        family_corrections=horizon_primary_corrections,
    )
    payload = {
        "format_version": PREDICTION_FORMAT_VERSION,
        "created_at_utc": utc_now(),
        "status": "COMPLETE",
        "seeds": [int(value) for value in seeds],
        "fold_seed": int(fold_seed),
        "fold_count": int(n_splits),
        "family_count": len(normalized),
        # ``rows`` remains the one-row-per-family primary-horizon view used by
        # the existing selector metrics.  ``horizon_rows`` is the canonical
        # all-horizon evidence and is what must be used for deployment
        # calibration admission.
        "rows": normalized,
        "primary_horizon": PRIMARY_HORIZON,
        "primary_horizon_index": PRIMARY_HORIZON_INDEX,
        "horizon_rows": normalized_horizon,
        "horizon_count": len(CARE_HORIZONS),
        "evaluated_horizons": list(CARE_HORIZONS),
        "horizon_oof_complete": True,
        # The unqualified calibration entry is deliberately the canonical
        # simultaneous all-horizon contract.  Keeping the old H16-only value
        # here would let a generic downstream consumer silently bypass the
        # H8/H32/H64 correction even though the report advertises complete
        # horizon OOF evidence.
        "calibration": horizon_calibration,
        "horizon_calibration": horizon_calibration,
        "legacy_primary_horizon_calibration": calibration.to_dict(),
        "legacy_primary_horizon_metrics": legacy_primary_metrics,
        "crossfit_family_coverage": horizon_calibration[
            "crossfit_family_coverage"
        ],
        "horizon_crossfit_family_coverage": horizon_calibration[
            "crossfit_family_coverage"
        ],
        "metrics": metrics,
        "family_disjoint": True,
        "calibration_independent": True,
        "validation20_used_for_tuning": False,
        "candidate0_bit_exact": True,
        "care_theory_contract_unchanged": True,
    }
    return _json_value(payload)


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise ValueError("integer list must be non-empty and unique")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    folds = sub.add_parser("folds")
    folds.add_argument("--prepared-data", type=Path, required=True)
    folds.add_argument("--output", type=Path, required=True)
    folds.add_argument("--n-splits", type=int, default=DEFAULT_FOLDS)
    folds.add_argument("--fold-seed", type=int, default=DEFAULT_FOLD_SEED)

    train = sub.add_parser("train-fold")
    train.add_argument("--prepared-data", type=Path, required=True)
    train.add_argument("--output-root", type=Path, required=True)
    train.add_argument("--fold", type=int, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--n-splits", type=int, default=DEFAULT_FOLDS)
    train.add_argument("--fold-seed", type=int, default=DEFAULT_FOLD_SEED)
    train.add_argument("--updates", type=int, default=4000)
    train.add_argument("--batch-size", type=int, default=48)
    train.add_argument(
        "--action-prefix-steps", type=int, choices=(1, 4, 8, 16), default=1
    )
    train.add_argument("--device", default="cuda:0")

    final = sub.add_parser("fit-final")
    final.add_argument("--prepared-data", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument("--seed", type=int, default=20260907)
    final.add_argument("--updates", type=int, default=4000)
    final.add_argument("--batch-size", type=int, default=48)
    # ``main`` constructs the canonical family-fold assignment before
    # dispatching every command (including fit-final).  Keep these options
    # available for the full-data path even though fit-final does not use the
    # assignment for optimization; this avoids a namespace failure in the
    # pre-formal training smoke.
    final.add_argument("--n-splits", type=int, default=DEFAULT_FOLDS)
    final.add_argument("--fold-seed", type=int, default=DEFAULT_FOLD_SEED)
    final.add_argument(
        "--action-prefix-steps", type=int, choices=(1, 4, 8, 16), default=1
    )
    final.add_argument("--device", default="cuda:0")

    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--prepared-data", type=Path, required=True)
    aggregate.add_argument("--output-root", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument(
        "--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS)
    )
    aggregate.add_argument("--n-splits", type=int, default=DEFAULT_FOLDS)
    aggregate.add_argument("--fold-seed", type=int, default=DEFAULT_FOLD_SEED)
    aggregate.add_argument("--nominal", type=float, default=0.90)
    args = parser.parse_args()

    prepared = load_prepared_care(args.prepared_data)
    prepared_hash = sha256_file(args.prepared_data)
    assignments = family_folds(
        prepared,
        n_splits=int(args.n_splits),
        seed=int(args.fold_seed),
    )
    if args.command == "folds":
        value = fold_manifest(assignments, prepared.snapshot_ids)
        value["fold_seed"] = int(args.fold_seed)
        value["prepared_data_sha256"] = prepared_hash
        atomic_json(args.output, value)
        print(json.dumps({"status": "complete", "output": str(args.output)}))
        return

    if args.command == "train-fold":
        if not 0 <= int(args.fold) < int(args.n_splits):
            raise ValueError("OOF fold index is out of range")
        output = (
            args.output_root
            / f"fold_{int(args.fold)}"
            / f"seed_{int(args.seed)}"
        )
        prediction_path = output / "predictions.json"
        checkpoint_path = output / "checkpoint.pt"
        if prediction_path.exists() or checkpoint_path.exists():
            raise RuntimeError(f"refusing to overwrite OOF job {output}")
        training_config = FoldTrainingConfig(
            updates=int(args.updates),
            batch_size=int(args.batch_size),
            action_prefix_steps=int(args.action_prefix_steps),
        )
        checkpoint, rows = train_fold(
            prepared,
            assignments,
            heldout_fold=int(args.fold),
            seed=int(args.seed),
            device=torch.device(args.device),
            config=training_config,
            prepared_sha256=prepared_hash,
        )
        atomic_torch_save(checkpoint, checkpoint_path)
        atomic_json(
            prediction_path,
            _json_value(
                {
                    "format_version": PREDICTION_FORMAT_VERSION,
                    "created_at_utc": utc_now(),
                    "prepared_data_sha256": prepared_hash,
                    "fold": int(args.fold),
                    "seed": int(args.seed),
                    "checkpoint": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "rows": rows,
                }
            ),
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "fold": int(args.fold),
                    "seed": int(args.seed),
                    "families": len(checkpoint["heldout_families"]),
                    "horizon_rows": len(rows),
                }
            )
        )
        return

    if args.command == "fit-final":
        if args.output.exists():
            raise RuntimeError(f"refusing to overwrite final CARE-v3 checkpoint {args.output}")
        config = FoldTrainingConfig(
            updates=int(args.updates),
            batch_size=int(args.batch_size),
            action_prefix_steps=int(args.action_prefix_steps),
        )
        payload = train_final(
            prepared,
            seed=int(args.seed),
            device=torch.device(args.device),
            config=config,
            prepared_sha256=prepared_hash,
        )
        atomic_torch_save(payload, args.output)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output": str(args.output),
                    "updates": int(args.updates),
                    "action_prefix_steps": int(args.action_prefix_steps),
                }
            )
        )
        return

    seeds = _parse_ints(args.seeds)
    payload = aggregate_jobs(
        prepared,
        args.output_root,
        seeds=seeds,
        n_splits=int(args.n_splits),
        fold_seed=int(args.fold_seed),
        nominal=float(args.nominal),
        prepared_data_sha256=prepared_hash,
    )
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output),
                "metrics": payload["metrics"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
