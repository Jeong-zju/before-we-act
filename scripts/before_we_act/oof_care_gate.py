#!/usr/bin/env python3
"""Leakage-resistant out-of-fold calibration and CARE admission gate.

This module is intentionally independent from the frozen MARS CARE pipeline.
It provides the small piece that was missing from the development diagnostics:
every family is scored by a model whose fit-family set excludes that family,
and calibration/coverage is checked with a leave-one-fold-out correction.

The module does *not* change CARE's candidate set, utility definition, or
selector.  It only validates provenance and computes diagnostics.  A caller
must provide one already-aggregated row per family (horizon/repeat aggregation
belongs to the caller) with the following fields::

    family_index, fold, fit_families, quantiles [6,3,5], target [6,3],
    hard_safety_logit [6], hard_safety [6], candidate_legality [6]

``fit_families`` is deliberately mandatory.  A fold number alone is not
evidence that a training job did not accidentally include the held-out family.
The command line interface consumes JSON rows and writes an auditable report;
it is suitable for a preflight/admission supervisor, not for Validation20.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FORMAT_VERSION = "before-we-act.care-mars-oof-gate/1"
DEFAULT_NOMINAL_COVERAGE = 0.90
DEFAULT_CANDIDATES = 6
DEFAULT_COMPONENT = 2  # total/direct+response utility used by the selector
DEFAULT_LOWER_INDEX = 0
DEFAULT_MEDIAN_INDEX = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stable_key(snapshot_id: object, seed: int) -> bytes:
    return hashlib.sha256(
        f"before-we-act/care-oof-fold-v1|{int(seed)}|{snapshot_id}".encode(
            "utf-8"
        )
    ).digest()


def make_family_folds(
    task_ids: Sequence[int],
    snapshot_ids: Sequence[str],
    *,
    n_splits: int = 5,
    seed: int = 0,
) -> dict[int, int]:
    """Return deterministic task-stratified ``family_index -> fold`` labels.

    Families are sorted by an immutable snapshot id hash, never by branch
    outcomes.  Each task is distributed round-robin, so every fold receives a
    near-equal number of families from every task.  At least ``n_splits``
    families per task are required; silently creating an empty task/fold would
    make an OOF claim unverifiable.
    """

    if len(task_ids) != len(snapshot_ids):
        raise ValueError("task_ids and snapshot_ids must have equal length")
    if len(task_ids) == 0:
        raise ValueError("cannot split an empty family set")
    if int(n_splits) < 2:
        raise ValueError("OOF requires at least two folds")
    n_splits = int(n_splits)
    if len(set(str(value) for value in snapshot_ids)) != len(snapshot_ids):
        raise ValueError("snapshot_ids must be unique")
    by_task: dict[int, list[int]] = defaultdict(list)
    for index, task in enumerate(task_ids):
        task_value = int(task)
        if task_value < 0:
            raise ValueError("task ids must be non-negative")
        by_task[task_value].append(index)
    too_small = {task: len(values) for task, values in by_task.items() if len(values) < n_splits}
    if too_small:
        raise ValueError(f"each task needs at least n_splits families: {too_small}")

    result: dict[int, int] = {}
    for task, families in sorted(by_task.items()):
        ordered = sorted(
            families,
            key=lambda index: (_stable_key(snapshot_ids[index], seed), str(snapshot_ids[index])),
        )
        for ordinal, family in enumerate(ordered):
            result[int(family)] = int(ordinal % n_splits)
    if set(result) != set(range(len(task_ids))):
        raise AssertionError("OOF fold assignment omitted a family")
    return result


def fold_manifest(
    fold_by_family: Mapping[int, int],
    snapshot_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Serialize fold provenance without embedding model outputs."""

    rows = []
    for family, fold in sorted((int(k), int(v)) for k, v in fold_by_family.items()):
        row: dict[str, Any] = {"family_index": family, "fold": fold}
        if snapshot_ids is not None:
            if family < 0 or family >= len(snapshot_ids):
                raise ValueError("fold manifest family index is out of range")
            row["snapshot_id"] = str(snapshot_ids[family])
        rows.append(row)
    return {
        "format_version": f"{FORMAT_VERSION}-folds",
        "created_at_utc": utc_now(),
        "family_count": len(rows),
        "fold_count": (max((row["fold"] for row in rows), default=-1) + 1),
        "families": rows,
    }


def _array(value: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return result


def _row_value(row: Mapping[str, Any], key: str, alias: str | None = None) -> Any:
    if key in row:
        return row[key]
    if alias is not None and alias in row:
        return row[alias]
    raise ValueError(f"OOF row is missing required field {key!r}")


def validate_oof_rows(
    rows: Iterable[Mapping[str, Any]],
    fold_by_family: Mapping[int, int],
    *,
    n_splits: int | None = None,
    require_complete_family_coverage: bool = True,
) -> list[dict[str, Any]]:
    """Validate row shapes and training provenance, returning normalized rows.

    A row is rejected when its held-out family appears in ``fit_families``.
    Duplicate family rows are rejected by default because calibration must be
    family-level (sibling horizons/repeats should be aggregated before this
    function).  ``fit_families`` is kept in the normalized result so the
    provenance can be written into a report and audited later.
    """

    assignments = {int(key): int(value) for key, value in fold_by_family.items()}
    if not assignments:
        raise ValueError("fold_by_family cannot be empty")
    if n_splits is None:
        n_splits = max(assignments.values()) + 1
    n_splits = int(n_splits)
    if n_splits < 2:
        raise ValueError("OOF requires at least two folds")
    if set(assignments.values()) != set(range(n_splits)):
        raise ValueError("fold assignments must contain every fold 0..n_splits-1")

    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    all_families = set(assignments)
    for source in rows:
        if not isinstance(source, Mapping):
            raise ValueError("each OOF row must be an object")
        family = int(_row_value(source, "family_index"))
        if family not in assignments:
            raise ValueError(f"OOF row references unknown family {family}")
        if family in seen:
            raise ValueError(f"duplicate OOF prediction for family {family}")
        seen.add(family)
        fold = int(_row_value(source, "fold"))
        expected_fold = assignments[family]
        if fold != expected_fold:
            raise ValueError(
                f"family {family} declares fold {fold}, expected {expected_fold}"
            )
        if fold < 0 or fold >= n_splits:
            raise ValueError(f"OOF fold out of range: {fold}")
        fit_raw = _row_value(source, "fit_families", "fit_family_indices")
        if isinstance(fit_raw, (str, bytes)):
            raise ValueError("fit_families must be a sequence of integer ids")
        fit = tuple(sorted({int(value) for value in fit_raw}))
        unknown_fit = set(fit) - all_families
        if unknown_fit:
            raise ValueError(f"fit_families contains unknown ids: {sorted(unknown_fit)}")
        held_out_fold = {
            other_family
            for other_family, other_fold in assignments.items()
            if other_fold == fold
        }
        leaked = set(fit) & held_out_fold
        if leaked:
            raise ValueError(
                f"OOF leakage: fold {fold} fit_families contains held-out ids "
                f"{sorted(leaked)}"
            )
        quantiles = _array(_row_value(source, "quantiles"), name="quantiles", shape=(6, 3, 5))
        target = _array(_row_value(source, "target"), name="target", shape=(6, 3))
        safety_logit = _array(
            _row_value(source, "hard_safety_logit"),
            name="hard_safety_logit",
            shape=(6,),
        )
        hard_safety = _array(
            _row_value(source, "hard_safety"), name="hard_safety", shape=(6,)
        )
        legality_raw = source.get("candidate_legality", np.ones(6, dtype=bool))
        legality_numeric = _array(
            legality_raw, name="candidate_legality", shape=(6,)
        )
        if not np.isin(legality_numeric, (0.0, 1.0)).all():
            raise ValueError("candidate_legality must contain only booleans")
        candidate_legality = legality_numeric.astype(bool)
        if not bool(candidate_legality[0]):
            raise ValueError(
                f"family {family} marks reference candidate zero illegal"
            )
        if not np.array_equal(quantiles[0], np.zeros_like(quantiles[0])):
            raise ValueError(f"family {family} violates candidate-0 exact-zero contract")
        normalized.append(
            {
                "family_index": family,
                "fold": fold,
                "fit_families": list(fit),
                "quantiles": quantiles,
                "target": target,
                "hard_safety_logit": safety_logit,
                "hard_safety": hard_safety,
                "candidate_legality": candidate_legality,
                "task_id": None if source.get("task_id") is None else int(source["task_id"]),
            }
        )
    if require_complete_family_coverage and seen != all_families:
        missing = sorted(all_families - seen)
        raise ValueError(f"OOF predictions do not cover every family; missing {missing}")
    if not normalized:
        raise ValueError("OOF prediction set is empty")
    return normalized


def _higher_quantile(values: Sequence[float], probability: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot compute a quantile of an empty set")
    try:
        return float(np.quantile(values, probability, method="higher"))
    except TypeError:  # pragma: no cover - NumPy < 1.22 compatibility
        return float(np.quantile(values, probability, interpolation="higher"))


def _scores(
    rows: Sequence[Mapping[str, Any]], *, component: int = DEFAULT_COMPONENT
) -> dict[int, float]:
    result: dict[int, float] = {}
    for row in rows:
        family = int(row["family_index"])
        lower = np.asarray(row["quantiles"], dtype=np.float64)[:, component, DEFAULT_LOWER_INDEX]
        target = np.asarray(row["target"], dtype=np.float64)[:, component]
        legality = np.asarray(
            row.get("candidate_legality", np.ones(DEFAULT_CANDIDATES)), dtype=bool
        )
        values = (lower[1:] - target[1:])[legality[1:]]
        if values.size == 0:
            # No alternative can be executed, so this family places no
            # constraint on a non-reference lower-bound calibration.
            result[family] = 0.0
            continue
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite conformal score for family {family}")
        result[family] = float(np.max(values))
    return result


def conformal_correction(
    rows: Sequence[Mapping[str, Any]], nominal: float = DEFAULT_NOMINAL_COVERAGE
) -> float:
    """Fit a family-wise finite-sample lower-bound correction."""

    nominal = float(nominal)
    if not 0.0 < nominal <= 1.0:
        raise ValueError("nominal coverage must lie in (0,1]")
    values = list(_scores(rows).values())
    adjusted = min(1.0, math.ceil((len(values) + 1) * nominal) / len(values))
    return max(0.0, _higher_quantile(values, adjusted))


@dataclass(frozen=True)
class OOFCalibration:
    lower_correction: float
    task_lower_corrections: dict[str, float]
    nominal_coverage: float
    family_count: int
    fold_corrections: dict[str, float]
    fold_task_corrections: dict[str, dict[str, float]]
    crossfit_correction_by_family: dict[str, float]
    crossfit_family_coverage: float
    coverage_by_fold: dict[str, float]
    coverage_by_task_id: dict[str, float]
    score_by_family: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_oof_calibration(
    rows: Sequence[Mapping[str, Any]],
    fold_by_family: Mapping[int, int],
    *,
    nominal: float = DEFAULT_NOMINAL_COVERAGE,
) -> OOFCalibration:
    """Fit global correction and measure strict leave-one-fold-out coverage.

    The global correction is the value one would carry to a final model after
    retraining on all data.  It is *not* used to claim coverage on the same
    rows.  The reported coverage uses, for each fold, a correction fitted only
    on the other folds; this prevents calibration-family reuse in the gate.
    """

    normalized = validate_oof_rows(rows, fold_by_family)
    assignments = {int(key): int(value) for key, value in fold_by_family.items()}
    score_by_family = _scores(normalized)
    global_correction = conformal_correction(normalized, nominal)
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in normalized:
        by_task[str(row.get("task_id", "unknown"))].append(row)
    task_corrections = {
        task: conformal_correction(task_rows, nominal)
        for task, task_rows in sorted(by_task.items())
    }
    folds = sorted(set(assignments.values()))
    fold_corrections: dict[str, float] = {}
    fold_task_corrections: dict[str, dict[str, float]] = {}
    correction_by_family: dict[str, float] = {}
    coverage_by_fold: dict[str, float] = {}
    covered_by_task: dict[str, list[bool]] = defaultdict(list)
    covered: list[bool] = []
    for fold in folds:
        fit_rows = [row for row in normalized if int(row["fold"]) != fold]
        eval_rows = [row for row in normalized if int(row["fold"]) == fold]
        correction = conformal_correction(fit_rows, nominal)
        fold_corrections[str(fold)] = float(correction)
        current_task_corrections: dict[str, float] = {}
        for task in sorted(by_task):
            task_fit = [
                row
                for row in fit_rows
                if str(row.get("task_id", "unknown")) == task
            ]
            # A missing task in the calibration side cannot silently produce
            # an optimistic zero.  Fall back to the fold-global correction.
            current_task_corrections[task] = (
                conformal_correction(task_fit, nominal)
                if task_fit
                else float(correction)
            )
        fold_task_corrections[str(fold)] = current_task_corrections
        fold_covered: list[bool] = []
        for row in eval_rows:
            task = str(row.get("task_id", "unknown"))
            row_correction = float(current_task_corrections[task])
            correction_by_family[str(int(row["family_index"]))] = row_correction
            lower = np.asarray(row["quantiles"], dtype=np.float64)[:, DEFAULT_COMPONENT, DEFAULT_LOWER_INDEX]
            target = np.asarray(row["target"], dtype=np.float64)[:, DEFAULT_COMPONENT]
            legality = np.asarray(row["candidate_legality"], dtype=bool)
            ok = bool(
                np.all(target[1:][legality[1:]] >= lower[1:][legality[1:]] - row_correction)
            )
            fold_covered.append(ok)
            covered.append(ok)
            covered_by_task[task].append(ok)
        coverage_by_fold[str(fold)] = float(np.mean(fold_covered)) if fold_covered else 0.0
    return OOFCalibration(
        lower_correction=float(global_correction),
        task_lower_corrections={
            key: float(value) for key, value in task_corrections.items()
        },
        nominal_coverage=float(nominal),
        family_count=len(normalized),
        fold_corrections=fold_corrections,
        fold_task_corrections=fold_task_corrections,
        crossfit_correction_by_family=correction_by_family,
        crossfit_family_coverage=float(np.mean(covered)) if covered else 0.0,
        coverage_by_fold=coverage_by_fold,
        coverage_by_task_id={
            key: float(np.mean(values))
            for key, values in sorted(covered_by_task.items())
        },
        score_by_family={str(key): float(value) for key, value in sorted(score_by_family.items())},
    )


def oof_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    correction: float | Mapping[str, float] | None = None,
    task_corrections: Mapping[str, float] | None = None,
    family_corrections: Mapping[str, float] | None = None,
    safety_probability_max: float = 0.25,
    selector_delta: float = 0.0,
) -> dict[str, Any]:
    """Compute ranking and selector-facing metrics in physical utility units.

    The unconstrained median-ranking metrics retain the original names.  The
    ``selector_*`` fields model the deployed lower-bound decision after
    conformal correction, legality masking and (when supported) safety
    masking.  Keeping these quantities separate prevents an illegal top-ranked
    candidate from being counted as a selector success.
    """

    normalized = validate_oof_rows(
        rows,
        {int(row["family_index"]): int(row["fold"]) for row in rows},
        require_complete_family_coverage=False,
    )
    if not normalized:
        raise ValueError("cannot score an empty OOF set")
    safety_positive = sum(
        int(np.count_nonzero(np.asarray(row["hard_safety"])[1:])) for row in normalized
    )
    safety_degenerate = safety_positive == 0
    regrets: list[float] = []
    selector_regrets: list[float] = []
    predicted_values: list[int] = []
    best_values: list[int] = []
    selector_values: list[int] = []
    selector_best_values: list[int] = []
    pair_correct = pair_count = 0
    legal_pair_correct = legal_pair_count = 0
    ref_correct = ref_count = 0
    overrides: list[bool] = []
    harmful: list[bool] = []
    beneficial: list[bool] = []
    task_regret: dict[str, list[float]] = defaultdict(list)
    illegal_candidates = total_candidates = 0
    for row in normalized:
        quantiles = np.asarray(row["quantiles"], dtype=np.float64)
        target = np.asarray(row["target"], dtype=np.float64)[:, DEFAULT_COMPONENT]
        median = quantiles[:, DEFAULT_COMPONENT, DEFAULT_MEDIAN_INDEX]
        legality = np.asarray(row["candidate_legality"], dtype=bool)
        if legality.shape != (DEFAULT_CANDIDATES,):
            raise ValueError("candidate_legality must have six entries")
        if not bool(legality[0]):
            raise ValueError("candidate zero must be legal")
        illegal_candidates += int(np.count_nonzero(~legality[1:]))
        total_candidates += DEFAULT_CANDIDATES - 1
        best = int(np.argmax(target))
        predicted = int(np.argmax(median))
        selected_scores = quantiles[:, DEFAULT_COMPONENT, DEFAULT_LOWER_INDEX].copy()
        task_key = str(row.get("task_id", "unknown"))
        row_correction: float | None = None
        family_key = str(int(row["family_index"]))
        if family_corrections is not None and family_key in family_corrections:
            row_correction = float(family_corrections[family_key])
        elif task_corrections is not None and task_key in task_corrections:
            row_correction = float(task_corrections[task_key])
        elif isinstance(correction, Mapping):
            if task_key in correction:
                row_correction = float(correction[task_key])
        elif correction is not None:
            row_correction = float(correction)
        if row_correction is not None:
            if not math.isfinite(row_correction) or row_correction < 0.0:
                raise ValueError(
                    "OOF selector correction must be finite and non-negative"
                )
            selected_scores -= row_correction
        # Legality is a physical certificate and is applied before argmax.
        selected_scores[~legality] = -np.inf
        if not safety_degenerate:
            unsafe = 1.0 / (1.0 + np.exp(-np.asarray(row["hard_safety_logit"]))) > float(
                safety_probability_max
            )
            selected_scores[unsafe] = -np.inf
        else:
            unsafe = np.zeros(DEFAULT_CANDIDATES, dtype=bool)
        selected_scores[0] = 0.0
        selected = int(np.argmax(selected_scores))
        if selected_scores[selected] <= float(selector_delta):
            selected = 0
        predicted_values.append(predicted)
        best_values.append(best)
        regret = float(target[best] - target[predicted])
        regrets.append(regret)
        legal_best = int(np.argmax(np.where(legality, target, -np.inf)))
        selector_regret = float(target[legal_best] - target[selected])
        selector_regrets.append(selector_regret)
        selector_values.append(selected)
        selector_best_values.append(legal_best)
        task = task_key
        task_regret[task].append(regret)
        override = selected != 0
        overrides.append(override)
        harmful.append(override and float(target[selected]) < 0.0)
        beneficial.append(override and float(target[selected]) > 0.0)
        score_delta = median[:, None] - median[None, :]
        target_delta = target[:, None] - target[None, :]
        upper = np.triu(np.ones_like(score_delta, dtype=bool), k=1)
        informative = upper & (np.abs(target_delta) > 1e-6)
        pair_correct += int(np.count_nonzero((np.sign(score_delta) == np.sign(target_delta)) & informative))
        pair_count += int(np.count_nonzero(informative))
        legal_informative = informative & legality[:, None] & legality[None, :]
        legal_pair_correct += int(
            np.count_nonzero(
                (np.sign(score_delta) == np.sign(target_delta)) & legal_informative
            )
        )
        legal_pair_count += int(np.count_nonzero(legal_informative))
        ref_delta = target[1:] - target[0]
        ref_score_delta = median[1:] - median[0]
        ref_mask = np.abs(ref_delta) > 1e-6
        ref_correct += int(np.count_nonzero((np.sign(ref_score_delta) == np.sign(ref_delta)) & ref_mask))
        ref_count += int(np.count_nonzero(ref_mask))
    return {
        "rows": len(normalized),
        "families": len({int(row["family_index"]) for row in normalized}),
        "top1_accuracy": float(np.mean(np.asarray(predicted_values) == np.asarray(best_values))),
        "pairwise_accuracy_including_reference": pair_correct / pair_count if pair_count else 0.0,
        "legal_pairwise_accuracy_including_reference": (
            legal_pair_correct / legal_pair_count if legal_pair_count else 0.0
        ),
        "candidate_vs_reference_sign_accuracy": ref_correct / ref_count if ref_count else 0.0,
        "mean_regret": float(np.mean(regrets)),
        "median_regret": float(np.median(regrets)),
        "selector_top1_accuracy": float(
            np.mean(
                np.asarray(selector_values) == np.asarray(selector_best_values)
            )
        ),
        "selector_mean_regret": float(np.mean(selector_regrets)),
        "selector_median_regret": float(np.median(selector_regrets)),
        "mean_regret_by_task_id": {
            key: float(np.mean(values)) for key, values in sorted(task_regret.items())
        },
        "correction": (
            None
            if correction is None or isinstance(correction, Mapping)
            else float(correction)
        ),
        "task_corrections": (
            None
            if task_corrections is None
            else {str(key): float(value) for key, value in task_corrections.items()}
        ),
        "family_crossfit_corrections_applied": family_corrections is not None,
        "override_rate": float(np.mean(overrides)),
        "harmful_override_rate": float(np.sum(harmful) / max(np.sum(overrides), 1)),
        "beneficial_override_rate": float(np.sum(beneficial) / max(np.sum(overrides), 1)),
        "safety_positive_label_count": int(safety_positive),
        "safety_supervision_degenerate": bool(safety_degenerate),
        "safety_gate_mode": "legality_only" if safety_degenerate else "learned_probability",
        "learned_safety_mask_applied": bool(not safety_degenerate),
        "illegal_candidate_rate": illegal_candidates / max(total_candidates, 1),
        "selector_legality_mask_applied": True,
        "selector_selected_candidate_counts": {
            str(candidate): int(selector_values.count(candidate))
            for candidate in sorted(set(selector_values))
        },
    }


@dataclass(frozen=True)
class AdmissionThresholds:
    """Pre-registered thresholds for promotion to a fresh Validation20 run."""

    min_pairwise_accuracy: float = 0.65
    min_top1_accuracy: float = 0.40
    min_candidate_reference_accuracy: float = 0.65
    min_crossfit_coverage: float = 0.90
    max_harmful_override_rate: float = 0.05
    min_dev_gain: float = 0.05
    min_dev_bootstrap_lower_95: float = 0.0
    min_act_gain: float = 0.05
    min_act_bootstrap_lower_95: float = 0.0


def _check(
    checks: list[dict[str, Any]],
    name: str,
    observed: Any,
    predicate: bool,
    threshold: Any,
    reason: str,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(predicate),
            "observed": observed,
            "threshold": threshold,
            "reason": reason if not predicate else "",
        }
    )


def admission_gate(
    report: Mapping[str, Any],
    *,
    thresholds: AdmissionThresholds | None = None,
    require_closed_loop: bool = True,
    require_proposal: bool = False,
) -> dict[str, Any]:
    """Evaluate a fail-closed, auditable promotion gate.

    Missing fields are failures.  This is intentional: an omitted contract
    receipt must never be interpreted as a passing experiment.  Set
    ``require_closed_loop=False`` for a scorer-only development gate; such a
    result cannot authorize Validation20 by itself.
    """

    threshold = thresholds or AdmissionThresholds()
    checks: list[dict[str, Any]] = []
    contract = report.get("contract", {})
    if not isinstance(contract, Mapping):
        contract = {}
    for key, expected in (
        ("candidate0_bit_exact", True),
        ("strict_local", True),
        ("normalization_action_decode_parity", True),
        ("snapshot_restore_parity", True),
        ("branch_horizon_complete", True),
    ):
        observed = contract.get(key)
        _check(checks, f"contract.{key}", observed, observed is expected, expected, "contract receipt missing or failed")
    legality = contract.get("candidate_legality_rate")
    _check(checks, "contract.candidate_legality_rate", legality, isinstance(legality, (int, float)) and float(legality) >= 1.0, 1.0, "every candidate must be legal")
    provenance = report.get("provenance", {})
    if not isinstance(provenance, Mapping):
        provenance = {}
    for key in ("family_disjoint", "calibration_independent", "no_validation20_tuning"):
        observed = provenance.get(key)
        _check(checks, f"provenance.{key}", observed, observed is True, True, "family/protocol leakage or tuning provenance is not proven")

    metrics = report.get("metrics", report)
    if not isinstance(metrics, Mapping):
        metrics = {}
    for key, minimum in (
        ("pairwise_accuracy_including_reference", threshold.min_pairwise_accuracy),
        ("top1_accuracy", threshold.min_top1_accuracy),
        ("candidate_vs_reference_sign_accuracy", threshold.min_candidate_reference_accuracy),
    ):
        observed = metrics.get(key)
        _check(checks, f"metrics.{key}", observed, isinstance(observed, (int, float)) and float(observed) >= minimum, minimum, "scorer signal below the pre-registered floor")
    coverage = report.get("crossfit_family_coverage", metrics.get("crossfit_family_coverage"))
    _check(checks, "crossfit_family_coverage", coverage, isinstance(coverage, (int, float)) and float(coverage) >= threshold.min_crossfit_coverage, threshold.min_crossfit_coverage, "OOF family-wise coverage below nominal")
    beneficial = metrics.get("beneficial_override_rate")
    _check(checks, "beneficial_override_rate", beneficial, isinstance(beneficial, (int, float)) and float(beneficial) > 0.0, ">0", "selector produced no demonstrably positive override")
    harmful = metrics.get("harmful_override_rate")
    _check(checks, "harmful_override_rate", harmful, isinstance(harmful, (int, float)) and float(harmful) <= threshold.max_harmful_override_rate, threshold.max_harmful_override_rate, "harmful override rate too high")
    safety_mode = metrics.get("safety_gate_mode")
    safety_support = metrics.get("safety_positive_label_count")
    safety_auditable = isinstance(safety_support, (int, float)) and (
        (safety_mode == "legality_only" and int(safety_support) == 0)
        or (safety_mode == "learned_probability" and int(safety_support) > 0)
    )
    _check(checks, "safety_support_or_legality_mode", safety_mode, safety_auditable, "learned_probability with positives | legality_only with zero positives", "safety supervision mode is not auditable")

    if require_proposal:
        oracle_gain = report.get("oracle_headroom_gain")
        pair_ratio = report.get("effective_pair_ratio")
        _check(checks, "proposal.oracle_headroom_gain", oracle_gain, isinstance(oracle_gain, (int, float)) and float(oracle_gain) > 0.0, ">0", "learned proposal has no fresh oracle headroom")
        _check(checks, "proposal.effective_pair_ratio", pair_ratio, isinstance(pair_ratio, (int, float)) and float(pair_ratio) >= 1.20, 1.20, "proposal does not increase effective pairs")

    if require_closed_loop:
        for key, minimum in (
            ("care_minus_selector_off", threshold.min_dev_gain),
            ("care_minus_act", threshold.min_act_gain),
        ):
            observed = report.get(key)
            _check(checks, key, observed, isinstance(observed, (int, float)) and float(observed) >= minimum, minimum, "closed-loop gain is below the claim floor")
        for key, minimum in (
            ("paired_bootstrap_lower_95", threshold.min_dev_bootstrap_lower_95),
            ("care_vs_act_bootstrap_lower_95", threshold.min_act_bootstrap_lower_95),
        ):
            observed = report.get(key)
            _check(checks, key, observed, isinstance(observed, (int, float)) and float(observed) > minimum, f">{minimum}", "paired uncertainty interval does not exclude zero")
    failed = [row for row in checks if not row["passed"]]
    return {
        "format_version": FORMAT_VERSION,
        "status": "ADMITTED" if not failed else "REJECTED",
        "passed": not failed,
        "checks": checks,
        "failed_checks": [row["name"] for row in failed],
        "require_closed_loop": bool(require_closed_loop),
        "require_proposal": bool(require_proposal),
        "thresholds": asdict(threshold),
    }


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="JSON list or object containing rows")
    parser.add_argument("--folds", type=Path, required=True, help="fold manifest JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nominal", type=float, default=DEFAULT_NOMINAL_COVERAGE)
    parser.add_argument("--scorer-report", type=Path, help="optional JSON report to gate")
    parser.add_argument("--scorer-only", action="store_true", help="do not require closed-loop fields")
    args = parser.parse_args()
    prediction_payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    rows = prediction_payload.get("rows", prediction_payload) if isinstance(prediction_payload, Mapping) else prediction_payload
    if not isinstance(rows, list):
        raise ValueError("predictions JSON must be a list or contain a rows list")
    fold_payload = json.loads(args.folds.read_text(encoding="utf-8"))
    family_rows = fold_payload.get("families", fold_payload) if isinstance(fold_payload, Mapping) else fold_payload
    if not isinstance(family_rows, list):
        raise ValueError("folds JSON must be a list or contain a families list")
    fold_by_family = {int(row["family_index"]): int(row["fold"]) for row in family_rows}
    normalized = validate_oof_rows(rows, fold_by_family)
    calibration = fit_oof_calibration(normalized, fold_by_family, nominal=args.nominal)
    metrics = oof_metrics(
        normalized,
        correction=calibration.lower_correction,
        family_corrections=calibration.crossfit_correction_by_family,
    )
    report: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "created_at_utc": utc_now(),
        "fold_manifest": str(args.folds.resolve()),
        "prediction_source": str(args.predictions.resolve()),
        "calibration": calibration.to_dict(),
        "crossfit_family_coverage": calibration.crossfit_family_coverage,
        "metrics": metrics,
        "promotion_scope": "diagnostic_only_until_fresh_formal_run",
    }
    if args.scorer_report:
        scorer_report = json.loads(args.scorer_report.read_text(encoding="utf-8"))
        report["admission"] = admission_gate(
            scorer_report, require_closed_loop=not args.scorer_only
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_jsonify(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output": str(args.output.resolve()), "families": len(normalized)}))


if __name__ == "__main__":
    main()
