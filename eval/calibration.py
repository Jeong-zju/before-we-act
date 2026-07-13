"""Validation-only utility calibration and communication Pareto selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ParetoPoint:
    bit_price: float
    delay_price: float
    success_rate: float
    bits_per_episode: float
    return_mean: float | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ParetoPoint":
        point = cls(
            bit_price=float(value["bit_price"]),
            delay_price=float(value["delay_price"]),
            success_rate=float(value["success_rate"]),
            bits_per_episode=float(value["bits_per_episode"]),
            return_mean=(
                None if value.get("return_mean") is None else float(value["return_mean"])
            ),
        )
        if (
            point.bit_price < 0
            or point.delay_price < 0
            or not 0 <= point.success_rate <= 1
            or point.bits_per_episode < 0
        ):
            raise ValueError("invalid communication sweep point")
        return point


def pareto_frontier(points: Sequence[ParetoPoint]) -> list[ParetoPoint]:
    """Keep points not dominated in success (higher) and bits (lower)."""

    frontier: list[ParetoPoint] = []
    for candidate in points:
        dominated = any(
            other.success_rate >= candidate.success_rate
            and other.bits_per_episode <= candidate.bits_per_episode
            and (
                other.success_rate > candidate.success_rate
                or other.bits_per_episode < candidate.bits_per_episode
            )
            for other in points
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda item: (item.bits_per_episode, -item.success_rate))


def select_pareto_utopia(points: Sequence[ParetoPoint]) -> dict[str, Any]:
    """Select the validation point closest to zero success loss/communication."""

    frontier = pareto_frontier(points)
    if not frontier:
        raise ValueError("communication sweep contains no points")
    success = np.asarray([point.success_rate for point in frontier], dtype=np.float64)
    bits = np.asarray([point.bits_per_episode for point in frontier], dtype=np.float64)
    success_loss = success.max() - success
    success_scale = max(float(success_loss.max()), 1e-12)
    bits_span = max(float(bits.max() - bits.min()), 1e-12)
    normalized_loss = success_loss / success_scale
    normalized_bits = (bits - bits.min()) / bits_span
    distance = np.sqrt(normalized_loss**2 + normalized_bits**2)
    selected_index = int(np.argmin(distance))
    selected = frontier[selected_index]
    return {
        "selection_method": "normalized_distance_to_success_bits_utopia",
        "selected": selected.__dict__,
        "selected_distance": float(distance[selected_index]),
        "frontier": [point.__dict__ for point in frontier],
        "test_set_used": False,
    }


def fit_affine_cost_calibration(
    predicted_cost: Sequence[float], realized_cost: Sequence[float]
) -> dict[str, float]:
    """Fit validation-only affine calibration from model cost to realized cost."""

    x = np.asarray(predicted_cost, dtype=np.float64)
    y = np.asarray(realized_cost, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError("calibration arrays must be equal one-dimensional arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("calibration arrays must be finite")
    design = np.stack([x, np.ones_like(x)], axis=1)
    scale, bias = np.linalg.lstsq(design, y, rcond=None)[0]
    calibrated = scale * x + bias
    return {
        "scale": float(scale),
        "bias": float(bias),
        "rmse": float(np.sqrt(np.mean((calibrated - y) ** 2))),
        "samples": int(len(x)),
    }
