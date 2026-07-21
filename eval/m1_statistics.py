"""Deterministic, dependency-light statistics for Phase M1 evaluation.

The helpers in this module operate only on in-memory records and numeric
sequences.  They deliberately do not import the simulator, model code, or
SciPy, so the same implementation can be used by unit tests and the formal
acceptance CLI.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FORMAT_VERSION = "wam.multimodal.m1.statistics/1"


@dataclass(frozen=True)
class M1EpisodeRecord:
    """One closed-loop result identified by every pairing dimension."""

    task_id: str
    evaluation_seed: int
    cue_id: int
    model_variant: str
    train_seed: int
    intervention: str
    success: bool
    steps: int
    total_reward: float
    action_source: str
    presented_observation_paths: tuple[str, ...] = ()
    consumed_observation_paths: tuple[str, ...] = ()
    privileged_observation_seen: bool = False
    fallback_used: bool = False
    actions_finite_and_bounded: bool = True
    replan_events: int = 0
    cold_replan_events: int = 0
    warm_replan_events: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def identity(self) -> tuple[str, str, int, str, int, int]:
        return (
            self.model_variant,
            self.intervention,
            self.train_seed,
            self.task_id,
            self.evaluation_seed,
            self.cue_id,
        )


EpisodeLike = M1EpisodeRecord | Mapping[str, Any]


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Return a two-sided Wilson score interval for a Bernoulli rate."""

    successes = int(successes)
    total = int(total)
    _validate_confidence(confidence)
    if total <= 0:
        raise ValueError("total must be positive")
    if successes < 0 or successes > total:
        raise ValueError("successes must be in [0,total]")
    z = NormalDist().inv_cdf(0.5 + 0.5 * confidence)
    rate = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (rate + z2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z2 / (4.0 * total * total))
        / denominator
    )
    return {
        "successes": successes,
        "total": total,
        "rate": rate,
        "confidence": confidence,
        "lower": 0.0 if successes == 0 else max(0.0, center - radius),
        "upper": 1.0 if successes == total else min(1.0, center + radius),
    }


def paired_bootstrap_difference(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Bootstrap the paired mean difference ``first - second``.

    A local NumPy generator and fixed chunking make repeated calls with the same
    arguments bit-for-bit deterministic without mutating global RNG state.
    """

    left, right = _paired_finite(first, second)
    samples = _bootstrap_settings(bootstrap_samples, confidence)
    differences = left - right
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 512):
        stop = min(start + 512, samples)
        indices = rng.integers(0, differences.size, size=(stop - start, differences.size))
        bootstrap[start:stop] = differences[indices].mean(axis=1)
    lower, upper = _quantile_interval(bootstrap, confidence)
    return {
        "pairs": int(differences.size),
        "mean_difference": float(differences.mean()),
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": confidence,
        "bootstrap_samples": samples,
        "bootstrap_seed": int(seed),
        "wins": int((differences > 0.0).sum()),
        "ties": int((differences == 0.0).sum()),
        "losses": int((differences < 0.0).sum()),
    }


def exact_binomial_two_sided(successes: int, trials: int) -> float:
    """Exact two-sided binomial p-value for the symmetric ``p=0.5`` null.

    Integer arithmetic keeps the result exact until the final float conversion.
    This is the binomial test used by exact McNemar and paired sign tests.
    """

    successes = int(successes)
    trials = int(trials)
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes/trials are inconsistent")
    if trials == 0:
        return 1.0
    tail = min(successes, trials - successes)
    numerator = sum(math.comb(trials, index) for index in range(tail + 1))
    probability = numerator / (1 << trials)
    return min(1.0, 2.0 * float(probability))


def exact_mcnemar(
    first: Sequence[bool] | np.ndarray,
    second: Sequence[bool] | np.ndarray,
) -> dict[str, float | int]:
    """Return exact McNemar counts and the two-sided binomial p-value."""

    left = np.asarray(first)
    right = np.asarray(second)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape or not left.size:
        raise ValueError("paired McNemar inputs must be non-empty equal 1-D arrays")
    if not _boolean_values(left) or not _boolean_values(right):
        raise ValueError("McNemar inputs must contain only boolean/0/1 values")
    left = left.astype(np.bool_)
    right = right.astype(np.bool_)
    first_only = int(np.logical_and(left, np.logical_not(right)).sum())
    second_only = int(np.logical_and(np.logical_not(left), right).sum())
    discordant = first_only + second_only
    return {
        "pairs": int(left.size),
        "both_success": int(np.logical_and(left, right).sum()),
        "both_failure": int(np.logical_and(~left, ~right).sum()),
        "first_only_success": first_only,
        "second_only_success": second_only,
        "discordant": discordant,
        "p_value_two_sided": exact_binomial_two_sided(first_only, discordant),
    }


def aggregate_episode_records(
    records: Sequence[EpisodeLike],
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Validate identities and aggregate success by variant/condition/seed/task."""

    _validate_confidence(confidence)
    normalized = tuple(coerce_episode_record(record) for record in records)
    identities = [record.identity for record in normalized]
    identity_counts = Counter(identities)
    duplicate_identities = sorted(
        identity for identity, count in identity_counts.items() if count > 1
    )
    invalid = [record.as_dict() for record in normalized if not _valid_record(record)]
    groups: dict[tuple[str, str, int, str], list[M1EpisodeRecord]] = defaultdict(list)
    for record in normalized:
        groups[
            (
                record.model_variant,
                record.intervention,
                record.train_seed,
                record.task_id,
            )
        ].append(record)
    summaries: dict[str, Any] = {}
    for key, items in sorted(groups.items()):
        variant, intervention, train_seed, task_id = key
        successes = sum(item.success for item in items)
        summary = wilson_interval(successes, len(items), confidence=confidence)
        summary.update(
            {
                "model_variant": variant,
                "intervention": intervention,
                "train_seed": train_seed,
                "task_id": task_id,
                "evaluation_seeds": sorted({item.evaluation_seed for item in items}),
                "cue_ids": sorted({item.cue_id for item in items}),
                "mean_return": float(np.mean([item.total_reward for item in items])),
            }
        )
        summaries[_group_key(key)] = summary
    return {
        "format_version": FORMAT_VERSION,
        "passed": bool(normalized) and not duplicate_identities and not invalid,
        "records": len(normalized),
        "unique_identities": len(set(identities)),
        "duplicate_identities": [list(value) for value in duplicate_identities],
        "invalid_record_count": len(invalid),
        "invalid_records": invalid[:100],
        "groups": summaries,
    }


def paired_episode_success(
    records: Sequence[EpisodeLike],
    *,
    first_variant: str,
    first_intervention: str,
    second_variant: str,
    second_intervention: str,
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare two episode conditions on exactly matched rollout identities.

    Confidence intervals resample physical ``evaluation_seed`` blocks.  Every
    train seed, task, and cue attached to a physical seed therefore remains in
    the same bootstrap draw instead of being treated as an independent trial.
    """

    normalized = tuple(coerce_episode_record(record) for record in records)
    first, first_duplicates = _selected_successes(
        normalized, first_variant, first_intervention
    )
    second, second_duplicates = _selected_successes(
        normalized, second_variant, second_intervention
    )
    first_keys, second_keys = set(first), set(second)
    common = sorted(first_keys & second_keys)
    exact_pairs = bool(common) and first_keys == second_keys
    result: dict[str, Any] = {
        "first": {
            "model_variant": first_variant,
            "intervention": first_intervention,
            "records": len(first),
            "success_rate": _optional_rate(first.values()),
        },
        "second": {
            "model_variant": second_variant,
            "intervention": second_intervention,
            "records": len(second),
            "success_rate": _optional_rate(second.values()),
        },
        "exact_pairs": exact_pairs,
        "paired_records": len(common),
        "missing_from_first": [list(value) for value in sorted(second_keys - first_keys)],
        "missing_from_second": [list(value) for value in sorted(first_keys - second_keys)],
        "duplicate_first_keys": [list(value) for value in first_duplicates],
        "duplicate_second_keys": [list(value) for value in second_duplicates],
        "pairing_key": ["train_seed", "task_id", "evaluation_seed", "cue_id"],
    }
    if not common:
        result.update({"difference": None, "mcnemar": None, "per_train_seed": {}})
        return result
    result.update(
        _episode_success_statistics(
            common,
            first,
            second,
            confidence=confidence,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    )
    train_seeds = sorted({key[0] for key in first_keys | second_keys})
    per_train_seed: dict[str, Any] = {}
    for offset, train_seed in enumerate(train_seeds, start=1):
        seed_first_keys = {key for key in first_keys if key[0] == train_seed}
        seed_second_keys = {key for key in second_keys if key[0] == train_seed}
        seed_common = sorted(seed_first_keys & seed_second_keys)
        seed_result: dict[str, Any] = {
            "train_seed": train_seed,
            "exact_pairs": bool(seed_common)
            and seed_first_keys == seed_second_keys,
            "paired_records": len(seed_common),
            "first_success_rate": _optional_rate(first[key] for key in seed_first_keys),
            "second_success_rate": _optional_rate(
                second[key] for key in seed_second_keys
            ),
            "missing_from_first": [
                list(value) for value in sorted(seed_second_keys - seed_first_keys)
            ],
            "missing_from_second": [
                list(value) for value in sorted(seed_first_keys - seed_second_keys)
            ],
            "duplicate_first_keys": [
                list(value) for value in first_duplicates if value[0] == train_seed
            ],
            "duplicate_second_keys": [
                list(value) for value in second_duplicates if value[0] == train_seed
            ],
        }
        if seed_common:
            seed_result.update(
                _episode_success_statistics(
                    seed_common,
                    first,
                    second,
                    confidence=confidence,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + 1_000 + offset,
                )
            )
        else:
            seed_result.update({"difference": None, "mcnemar": None})
        per_train_seed[str(train_seed)] = seed_result
    result["per_train_seed"] = per_train_seed
    return result


def _episode_success_statistics(
    keys: Sequence[tuple[int, str, int, int]],
    first: Mapping[tuple[int, str, int, int], bool],
    second: Mapping[tuple[int, str, int, int], bool],
    *,
    confidence: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    first_values = np.asarray([first[key] for key in keys], dtype=np.float64)
    second_values = np.asarray([second[key] for key in keys], dtype=np.float64)
    cluster_ids = np.asarray([key[2] for key in keys], dtype=np.int64)
    return {
        "difference": _clustered_paired_bootstrap_difference(
            first_values,
            second_values,
            cluster_ids,
            confidence=confidence,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        # Descriptive only: acceptance uses the physical-seed cluster interval.
        "mcnemar": exact_mcnemar(first_values, second_values),
        "mcnemar_resampling_unit": "paired_episode_record_descriptive_only",
    }


def _clustered_paired_bootstrap_difference(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
    cluster_ids: Sequence[int] | np.ndarray,
    *,
    confidence: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap paired row differences by resampling whole physical seeds."""

    left, right = _paired_finite(first, second)
    clusters_array = np.asarray(cluster_ids, dtype=np.int64)
    if clusters_array.ndim != 1 or clusters_array.shape != left.shape:
        raise ValueError("cluster_ids must match the paired 1-D inputs")
    samples = _bootstrap_settings(bootstrap_samples, confidence)
    differences = left - right
    unique_clusters = np.unique(clusters_array)
    if not unique_clusters.size:
        raise ValueError("at least one bootstrap cluster is required")
    cluster_indices = [
        np.flatnonzero(clusters_array == cluster_id)
        for cluster_id in unique_clusters
    ]
    cluster_sums = np.asarray(
        [differences[indices].sum() for indices in cluster_indices],
        dtype=np.float64,
    )
    cluster_counts = np.asarray(
        [indices.size for indices in cluster_indices], dtype=np.int64
    )
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(samples, dtype=np.float64)
    cluster_count = int(unique_clusters.size)
    for start in range(0, samples, 512):
        stop = min(start + 512, samples)
        selected = rng.integers(
            0, cluster_count, size=(stop - start, cluster_count)
        )
        bootstrap[start:stop] = cluster_sums[selected].sum(axis=1) / (
            cluster_counts[selected].sum(axis=1)
        )
    lower, upper = _quantile_interval(bootstrap, confidence)
    records_per_cluster = {
        str(int(cluster_id)): int(count)
        for cluster_id, count in zip(
            unique_clusters.tolist(), cluster_counts.tolist(), strict=True
        )
    }
    return {
        "pairs": int(differences.size),
        "mean_difference": float(differences.mean()),
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": confidence,
        "bootstrap_samples": samples,
        "bootstrap_seed": int(seed),
        "wins": int((differences > 0.0).sum()),
        "ties": int((differences == 0.0).sum()),
        "losses": int((differences < 0.0).sum()),
        "cluster_bootstrap": True,
        "cluster_key": "evaluation_seed",
        "clusters": cluster_count,
        "cluster_ids": [int(value) for value in unique_clusters.tolist()],
        "records_per_cluster": records_per_cluster,
        "minimum_records_per_cluster": int(cluster_counts.min()),
        "maximum_records_per_cluster": int(cluster_counts.max()),
        "balanced_cluster_sizes": bool(np.all(cluster_counts == cluster_counts[0])),
    }


def paired_rmse_comparison(
    model_errors: Sequence[float] | np.ndarray,
    baseline_errors: Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Compare paired object errors; positive means the model has lower RMSE."""

    model, baseline = _paired_finite(model_errors, baseline_errors)
    samples = _bootstrap_settings(bootstrap_samples, confidence)
    model_rmse = float(np.sqrt(np.mean(np.square(model))))
    baseline_rmse = float(np.sqrt(np.mean(np.square(baseline))))
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 512):
        stop = min(start + 512, samples)
        indices = rng.integers(0, model.size, size=(stop - start, model.size))
        model_values = np.sqrt(np.mean(np.square(model[indices]), axis=1))
        baseline_values = np.sqrt(np.mean(np.square(baseline[indices]), axis=1))
        bootstrap[start:stop] = baseline_values - model_values
    lower, upper = _quantile_interval(bootstrap, confidence)
    return {
        "pairs": int(model.size),
        "model_rmse": model_rmse,
        "baseline_rmse": baseline_rmse,
        "baseline_minus_model_rmse": baseline_rmse - model_rmse,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": confidence,
        "bootstrap_samples": samples,
        "bootstrap_seed": int(seed),
    }


def paired_balanced_accuracy_comparison(
    model_predictions: Sequence[int] | np.ndarray,
    baseline_predictions: Sequence[int] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Stratified paired bootstrap for binary balanced accuracy."""

    model = np.asarray(model_predictions)
    baseline = np.asarray(baseline_predictions)
    truth = np.asarray(labels)
    if (
        model.ndim != 1
        or baseline.ndim != 1
        or truth.ndim != 1
        or model.shape != baseline.shape
        or model.shape != truth.shape
        or not model.size
    ):
        raise ValueError("balanced-accuracy inputs must be non-empty paired 1-D arrays")
    if not all(_boolean_values(value) for value in (model, baseline, truth)):
        raise ValueError("balanced-accuracy inputs must contain only 0/1 values")
    model = model.astype(np.int8)
    baseline = baseline.astype(np.int8)
    truth = truth.astype(np.int8)
    class_indices = [np.flatnonzero(truth == label) for label in (0, 1)]
    if any(not indices.size for indices in class_indices):
        raise ValueError("balanced accuracy requires both label classes")
    model_score = _balanced_accuracy(model, truth)
    baseline_score = _balanced_accuracy(baseline, truth)
    samples = _bootstrap_settings(bootstrap_samples, confidence)
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        selected = np.concatenate(
            [
                indices[rng.integers(0, indices.size, size=indices.size)]
                for indices in class_indices
            ]
        )
        bootstrap[sample] = _balanced_accuracy(
            model[selected], truth[selected]
        ) - _balanced_accuracy(baseline[selected], truth[selected])
    lower, upper = _quantile_interval(bootstrap, confidence)
    model_correct = model == truth
    baseline_correct = baseline == truth
    return {
        "pairs": int(model.size),
        "model_balanced_accuracy": model_score,
        "baseline_balanced_accuracy": baseline_score,
        "model_minus_baseline_balanced_accuracy": model_score - baseline_score,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": confidence,
        "bootstrap_samples": samples,
        "bootstrap_seed": int(seed),
        "mcnemar": exact_mcnemar(model_correct, baseline_correct),
    }


def coerce_episode_record(record: EpisodeLike) -> M1EpisodeRecord:
    """Normalize a dataclass or JSON-compatible mapping without truthy coercion."""

    if isinstance(record, M1EpisodeRecord):
        return record
    if not isinstance(record, Mapping):
        raise TypeError("M1 episode records must be mappings or M1EpisodeRecord")
    required = {
        "task_id",
        "evaluation_seed",
        "cue_id",
        "model_variant",
        "train_seed",
        "intervention",
        "success",
        "steps",
        "total_reward",
        "action_source",
    }
    missing = sorted(required - set(record))
    if missing:
        raise KeyError(f"M1 episode record is missing {missing}")
    boolean_fields = (
        "success",
        "privileged_observation_seen",
        "fallback_used",
        "actions_finite_and_bounded",
    )
    for name in boolean_fields:
        if name in record and not isinstance(record[name], (bool, np.bool_)):
            raise TypeError(f"M1 episode field {name} must be boolean")
    for name in ("replan_events", "cold_replan_events", "warm_replan_events"):
        value = record.get(name, 0)
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"M1 episode field {name} must be an integer")
        if int(value) < 0:
            raise ValueError(f"M1 episode field {name} cannot be negative")
    return M1EpisodeRecord(
        task_id=str(record["task_id"]),
        evaluation_seed=int(record["evaluation_seed"]),
        cue_id=int(record["cue_id"]),
        model_variant=str(record["model_variant"]),
        train_seed=int(record["train_seed"]),
        intervention=str(record["intervention"]),
        success=bool(record["success"]),
        steps=int(record["steps"]),
        total_reward=float(record["total_reward"]),
        action_source=str(record["action_source"]),
        presented_observation_paths=tuple(
            str(value) for value in record.get("presented_observation_paths", ())
        ),
        consumed_observation_paths=tuple(
            str(value) for value in record.get("consumed_observation_paths", ())
        ),
        privileged_observation_seen=bool(
            record.get("privileged_observation_seen", False)
        ),
        fallback_used=bool(record.get("fallback_used", False)),
        actions_finite_and_bounded=bool(
            record.get("actions_finite_and_bounded", True)
        ),
        replan_events=int(record.get("replan_events", 0)),
        cold_replan_events=int(record.get("cold_replan_events", 0)),
        warm_replan_events=int(record.get("warm_replan_events", 0)),
    )


def _valid_record(record: M1EpisodeRecord) -> bool:
    return bool(
        record.task_id
        and record.model_variant
        and record.intervention
        and record.action_source
        and record.steps > 0
        and math.isfinite(record.total_reward)
        and record.evaluation_seed >= 0
        and record.replan_events
        == record.cold_replan_events + record.warm_replan_events
    )


def _selected_successes(
    records: Iterable[M1EpisodeRecord], variant: str, intervention: str
) -> tuple[dict[tuple[int, str, int, int], bool], list[tuple[int, str, int, int]]]:
    values: dict[tuple[int, str, int, int], bool] = {}
    duplicates: list[tuple[int, str, int, int]] = []
    for record in records:
        if record.model_variant != variant or record.intervention != intervention:
            continue
        key = (
            record.train_seed,
            record.task_id,
            record.evaluation_seed,
            record.cue_id,
        )
        if key in values:
            duplicates.append(key)
        values[key] = record.success
    return values, sorted(set(duplicates))


def _paired_finite(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape or not left.size:
        raise ValueError("paired inputs must be non-empty equal 1-D arrays")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("paired inputs must be finite")
    return left, right


def _bootstrap_settings(bootstrap_samples: int, confidence: float) -> int:
    _validate_confidence(confidence)
    samples = int(bootstrap_samples)
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    return samples


def _validate_confidence(confidence: float) -> None:
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0,1)")


def _quantile_interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = 0.5 * (1.0 - confidence)
    return (
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1.0 - alpha)),
    )


def _boolean_values(values: np.ndarray) -> bool:
    return bool(np.isin(values, (0, 1, False, True)).all())


def _balanced_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    recalls = [
        float((predictions[labels == label] == label).mean()) for label in (0, 1)
    ]
    return 0.5 * sum(recalls)


def _optional_rate(values: Iterable[bool]) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


def _group_key(key: tuple[str, str, int, str]) -> str:
    variant, intervention, train_seed, task_id = key
    return f"{variant}/{intervention}/train-{train_seed}/{task_id}"


__all__ = [
    "FORMAT_VERSION",
    "M1EpisodeRecord",
    "aggregate_episode_records",
    "coerce_episode_record",
    "exact_binomial_two_sided",
    "exact_mcnemar",
    "paired_balanced_accuracy_comparison",
    "paired_bootstrap_difference",
    "paired_episode_success",
    "paired_rmse_comparison",
    "wilson_interval",
]
