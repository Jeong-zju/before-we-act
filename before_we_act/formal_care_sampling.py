"""正式 CARE 采集的冻结计数、分组和协调临界状态选择。"""
from __future__ import annotations

from bisect import bisect_right
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence


TASKS = tuple(
    sorted(
        (
            "lift_barrier",
            "camera_alignment",
            "long_pipeline_delivery",
            "take_photo",
            "pass_shoe",
            "place_food",
        )
    )
)
SPLIT_RANGES = {
    "train": (0, 69),
    "validation": (70, 79),
    "calibration": (80, 89),
    "test": (90, 99),
}
CRITICAL_WEIGHTS = {
    "residual_norm": 0.25,
    "belief_entropy": 0.20,
    "one_minus_reliability": 0.15,
    "partner_occlusion": 0.15,
    "contact_or_phase_change": 0.15,
    "paired_inactivity": 0.10,
}


def hash_integer(namespace: str, modulo: int) -> int:
    if modulo <= 0:
        raise ValueError("modulo must be positive")
    return int.from_bytes(hashlib.sha256(namespace.encode("utf-8")).digest()[:8], "big") % modulo


def split_bucket(task: str, scenario_group_id: str) -> int:
    return hash_integer(
        f"A4-CARE-CONTRACT|split-v1|{task}|{scenario_group_id}", 100
    )


def split_name(task: str, scenario_group_id: str) -> str:
    bucket = split_bucket(task, scenario_group_id)
    for name, (lower, upper) in SPLIT_RANGES.items():
        if lower <= bucket <= upper:
            return name
    raise AssertionError(bucket)


def allocate_even(total: int, tasks: Sequence[str] = TASKS) -> dict[str, int]:
    quotient, remainder = divmod(int(total), len(tasks))
    return {
        task: quotient + int(index < remainder)
        for index, task in enumerate(tasks)
    }


def formal_targets() -> dict[tuple[str, str, str], int]:
    """Return exact (split, stratum, task) family counts."""

    targets: dict[tuple[str, str, str], int] = {}
    train = allocate_even(10_000)
    extra_uniform = True
    for task in TASKS:
        total = train[task]
        uniform = total // 2
        critical = total // 2
        if total % 2:
            if extra_uniform:
                uniform += 1
            else:
                critical += 1
            extra_uniform = not extra_uniform
        targets[("train", "uniform", task)] = uniform
        targets[("train", "critical", task)] = critical
    for split in ("validation", "calibration"):
        allocated = allocate_even(1_200)
        for task in TASKS:
            targets[(split, "uniform", task)] = allocated[task] // 2
            targets[(split, "critical", task)] = allocated[task] // 2
    for task, count in allocate_even(1_200).items():
        targets[("test", "uniform", task)] = count
        targets[("test", "critical", task)] = count
    assert sum(targets.values()) == 14_800
    assert sum(value for (split, stratum, _), value in targets.items() if split == "train" and stratum == "uniform") == 5_000
    assert sum(value for (split, stratum, _), value in targets.items() if split == "train" and stratum == "critical") == 5_000
    return targets


def gate_first_targets() -> dict[tuple[str, str, str], int]:
    """Return the frozen 150-state-per-task Gate A/B allocation."""

    targets: dict[tuple[str, str, str], int] = {}
    for task in TASKS:
        targets[("test", "critical", task)] = 100
        targets[("test", "uniform", task)] = 50
    assert sum(targets.values()) == 900
    assert all(
        sum(
            value
            for (_split, _stratum, row_task), value in targets.items()
            if row_task == task
        )
        == 150
        for task in TASKS
    )
    return targets


def compact_gate_targets() -> dict[tuple[str, str, str], int]:
    """Return the frozen 30-state-per-task compact Gate A/B allocation."""

    targets: dict[tuple[str, str, str], int] = {}
    for task in TASKS:
        targets[("test", "critical", task)] = 20
        targets[("test", "uniform", task)] = 10
    assert sum(targets.values()) == 180
    return targets


def pool_size(target: int, stratum: str) -> int:
    if stratum == "uniform":
        return math.ceil(target / 0.95)
    if stratum == "critical":
        return math.ceil(target / 0.28)
    raise ValueError(stratum)


def empirical_percentile(sorted_reference: Sequence[float], value: float) -> float:
    if not sorted_reference:
        raise ValueError("percentile reference cannot be empty")
    return bisect_right(sorted_reference, float(value)) / len(sorted_reference)


def fitted_references(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[float]]:
    values = {
        "residual_norm": [],
        "belief_entropy": [],
        "one_minus_reliability": [],
    }
    for row in rows:
        features = row["features"]
        values["residual_norm"].append(float(features["residual_norm"]))
        values["belief_entropy"].append(float(features["belief_entropy"]))
        values["one_minus_reliability"].append(
            1.0 - float(features["reliability"])
        )
    for items in values.values():
        items.sort()
    if any(not items for items in values.values()):
        raise ValueError("every continuous critical feature needs a train reference")
    return values


def score_critical(
    row: Mapping[str, Any], references: Mapping[str, Sequence[float]]
) -> tuple[float, dict[str, float]]:
    features = row["features"]
    percentiles = {
        "residual_norm": empirical_percentile(
            references["residual_norm"], float(features["residual_norm"])
        ),
        "belief_entropy": empirical_percentile(
            references["belief_entropy"], float(features["belief_entropy"])
        ),
        "one_minus_reliability": empirical_percentile(
            references["one_minus_reliability"],
            1.0 - float(features["reliability"]),
        ),
        "partner_occlusion": float(bool(features["partner_occlusion"])),
        "contact_or_phase_change": float(
            bool(features["contact_or_phase_change"])
        ),
        "paired_inactivity": float(bool(features["paired_inactivity"])),
    }
    score = sum(CRITICAL_WEIGHTS[key] * percentiles[key] for key in CRITICAL_WEIGHTS)
    return float(score), percentiles


__all__ = [
    "CRITICAL_WEIGHTS",
    "SPLIT_RANGES",
    "TASKS",
    "allocate_even",
    "compact_gate_targets",
    "empirical_percentile",
    "fitted_references",
    "formal_targets",
    "gate_first_targets",
    "hash_integer",
    "pool_size",
    "score_critical",
    "split_bucket",
    "split_name",
]
