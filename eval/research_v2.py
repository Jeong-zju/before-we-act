"""Held-out component metrics and preregistered gates for Research-v2."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def return_quantile_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    levels: Sequence[float] = (0.1, 0.5, 0.9),
) -> dict[str, object]:
    values = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    levels_array = np.asarray(levels, dtype=np.float64)
    if values.shape != (target.shape[0], levels_array.shape[0]):
        raise ValueError("quantile predictions/targets shape mismatch")
    error = target[:, None] - values
    pinball = np.maximum(levels_array * error, (levels_array - 1.0) * error)
    coverage = (target[:, None] <= values).mean(axis=0)
    return {
        "pinball": float(pinball.mean()),
        "coverage": {str(level): float(value) for level, value in zip(levels_array, coverage)},
        "coverage_mae": float(np.abs(coverage - levels_array).mean()),
        "quantile_crossing_rate": float((np.diff(values, axis=-1) < 0).mean()),
    }


def binary_calibration_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, float]:
    probability = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    if probability.shape != target.shape or np.any((probability < 0) | (probability > 1)):
        raise ValueError("binary probabilities/targets are invalid")
    brier = np.square(probability - target).mean()
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        include_right = index == bins - 1
        mask = (probability >= edges[index]) & (
            probability <= edges[index + 1] if include_right else probability < edges[index + 1]
        )
        if mask.any():
            ece += mask.mean() * abs(probability[mask].mean() - target[mask].mean())
    return {"brier": float(brier), "ece": float(ece)}


def grouped_branch_regret(
    predicted_score: np.ndarray,
    realized_return: np.ndarray,
    group_ids: np.ndarray,
) -> dict[str, float]:
    predicted = np.asarray(predicted_score, dtype=np.float64).reshape(-1)
    realized = np.asarray(realized_return, dtype=np.float64).reshape(-1)
    groups = np.asarray(group_ids).reshape(-1)
    if predicted.shape != realized.shape or predicted.shape != groups.shape:
        raise ValueError("branch arrays must align")
    regrets = []
    for group in np.unique(groups):
        rows = groups == group
        chosen = np.argmax(predicted[rows])
        regrets.append(np.max(realized[rows]) - realized[rows][chosen])
    return {"branch_regret": float(np.mean(regrets)), "groups": float(len(regrets))}


def proposal_oracle_coverage(topk_codes: np.ndarray, oracle_codes: np.ndarray) -> float:
    topk = np.asarray(topk_codes)
    oracle = np.asarray(oracle_codes).reshape(-1)
    if topk.ndim != 2 or topk.shape[0] != oracle.shape[0]:
        raise ValueError("proposal top-K/oracle shapes mismatch")
    return float((topk == oracle[:, None]).any(axis=-1).mean())


def vpi_calibration(predicted_vpi: np.ndarray, realized_value: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(predicted_vpi, dtype=np.float64).reshape(-1)
    realized = np.asarray(realized_value, dtype=np.float64).reshape(-1)
    if predicted.shape != realized.shape:
        raise ValueError("predicted and realized VPI shapes differ")
    if predicted.size < 2 or np.std(predicted) < 1e-12:
        slope, intercept, correlation = 0.0, float(realized.mean()), 0.0
    else:
        slope, intercept = np.polyfit(predicted, realized, 1)
        correlation = float(np.corrcoef(predicted, realized)[0, 1])
    calibrated = slope * predicted + intercept
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "correlation": correlation,
        "mae": float(np.abs(calibrated - realized).mean()),
    }


def data_scaling_gate(
    *,
    d1_branch_regret: float,
    d2_branch_regret: float,
    d1_success_rate: float,
    d2_success_rate: float,
    d1_constraint_ece: float,
    d2_constraint_ece: float,
) -> dict[str, bool | float]:
    relative_regret_improvement = (d1_branch_regret - d2_branch_regret) / max(
        abs(d1_branch_regret), 1e-8
    )
    success_improvement = d2_success_rate - d1_success_rate
    ece_degradation = d2_constraint_ece - d1_constraint_ece
    passed = (
        relative_regret_improvement >= 0.03 or success_improvement >= 0.01
    ) and ece_degradation <= 0.02
    return {
        "passed": bool(passed),
        "relative_branch_regret_improvement": float(relative_regret_improvement),
        "success_rate_improvement": float(success_improvement),
        "constraint_ece_degradation": float(ece_degradation),
    }
