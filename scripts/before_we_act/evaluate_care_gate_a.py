#!/usr/bin/env python3
"""Evaluate the frozen CARE Gate A on immutable A5R7 quality-filtered data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.before_we_act.label_care_simulator_anomalies import (
    sha256_file,
    verify_source_artifacts,
)


GATE_STAGE = "A4-CARE-CONTRACT"
COLLECTION_CONTRACT_STAGE = "A4R7-CARE-COMMON-SUPPORT-COLLECTION"
COLLECTION_STAGE = "A5R7-CARE-COMMON-SUPPORT-BRANCHES"
QUALITY_STAGE = "A5R7Q1-CARE-SIMULATOR-QUALITY-LABELS"
ANALYSIS_STAGE = "A5R7Q2-CARE-GATE-A-ANALYSIS"
HORIZONS = (8, 16, 32, 64)
REGIME = "reactive"
EXPECTED_TASKS = (
    "camera_alignment",
    "lift_barrier",
    "long_pipeline_delivery",
    "pass_shoe",
    "place_food",
    "take_photo",
)
COMPONENT_NAMES = (
    "task_progress_gain",
    "success",
    "collision_or_drop_rate",
    "robot_conflict_rate",
    "duplicate_work_rate",
    "deadlock_rate",
    "idle_imbalance",
    "normalized_completion_steps",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [math.nan, math.nan]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [center - radius, center + radius]


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def group_bootstrap_mean(
    values: Sequence[float], groups: Sequence[str], *, draws: int, seed: int
) -> dict[str, Any]:
    if len(values) != len(groups) or not values:
        raise ValueError("bootstrap values/groups must be non-empty and aligned")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups, strict=True):
        grouped[str(group)].append(float(value))
    keys = sorted(grouped)
    sums = np.asarray([sum(grouped[key]) for key in keys], dtype=np.float64)
    counts = np.asarray([len(grouped[key]) for key in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 2048):
        stop = min(start + 2048, draws)
        selected = rng.integers(0, len(keys), size=(stop - start, len(keys)))
        bootstrap[start:stop] = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    return {
        "mean": float(np.mean(values)),
        "ci95": percentile_interval(bootstrap),
        "draws": draws,
        "seed": seed,
        "group_count": len(keys),
    }


def task_equal_bootstrap_mean(
    values_by_task: Mapping[str, Sequence[float]], *, draws: int, seed: int
) -> dict[str, Any]:
    if not values_by_task:
        raise ValueError("task-equal bootstrap requires at least one task")
    rng = np.random.default_rng(seed)
    bootstrap = np.zeros(draws, dtype=np.float64)
    task_means = []
    for task in sorted(values_by_task):
        values = np.asarray(values_by_task[task], dtype=np.float64)
        if values.size == 0:
            raise ValueError(f"task {task} has no values")
        task_means.append(float(values.mean()))
        sampled = np.empty(draws, dtype=np.float64)
        for start in range(0, draws, 2048):
            stop = min(start + 2048, draws)
            selected = rng.integers(0, values.size, size=(stop - start, values.size))
            sampled[start:stop] = values[selected].mean(axis=1)
        bootstrap += sampled / len(values_by_task)
    return {
        "mean": float(np.mean(task_means)),
        "ci95": percentile_interval(bootstrap),
        "draws": draws,
        "seed": seed,
        "represented_task_count": len(values_by_task),
    }


def maximum_distinct_task_matching(eligible: Mapping[int, Sequence[str]]) -> dict[int, str]:
    best: dict[int, str] = {}

    def visit(candidate_ids: list[int], index: int, used: set[str], chosen: dict[int, str]) -> None:
        nonlocal best
        if len(chosen) > len(best):
            best = dict(chosen)
        if index >= len(candidate_ids):
            return
        candidate_id = candidate_ids[index]
        visit(candidate_ids, index + 1, used, chosen)
        for task in sorted(eligible.get(candidate_id, [])):
            if task in used:
                continue
            used.add(task)
            chosen[candidate_id] = task
            visit(candidate_ids, index + 1, used, chosen)
            del chosen[candidate_id]
            used.remove(task)

    visit(sorted(eligible), 0, set(), {})
    return best


def branch_by_key(
    row: Mapping[str, Any], candidate_id: int, repeat_id: int
) -> Mapping[str, Any]:
    selected = [
        branch
        for branch in row["branches"]
        if int(branch["candidate_id"]) == candidate_id
        and branch["regime"] == REGIME
        and int(branch["repeat_id"]) == repeat_id
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"{row['snapshot_id']} missing unique branch {(candidate_id, REGIME, repeat_id)}"
        )
    return selected[0]


def utility(outcome: Mapping[str, Any], weights: Sequence[float]) -> float:
    vector = np.asarray(outcome["bounded_utility_vector"], dtype=np.float64)
    if vector.shape != (8,):
        raise RuntimeError("bounded utility vector must have eight components")
    return -3.0 * float(bool(outcome["hard_safety_violation"])) + float(
        np.dot(np.asarray(weights, dtype=np.float64), vector)
    )


def candidate_summary(
    row: Mapping[str, Any], horizon: int, profiles: Mapping[str, Sequence[float]]
) -> dict[int, dict[str, Any]]:
    result = {}
    for candidate_id in range(6):
        outcomes = [
            branch_by_key(row, candidate_id, repeat_id)["outcomes"][str(horizon)]
            for repeat_id in (0, 1)
        ]
        vector = np.mean(
            np.asarray([outcome["bounded_utility_vector"] for outcome in outcomes], dtype=np.float64),
            axis=0,
        )
        hard_safety = float(
            np.mean([bool(outcome["hard_safety_violation"]) for outcome in outcomes])
        )
        utilities = {
            name: float(np.mean([utility(outcome, weights) for outcome in outcomes]))
            for name, weights in profiles.items()
        }
        stored_main = float(np.mean([outcome["utility_main"] for outcome in outcomes]))
        if not math.isclose(stored_main, utilities["main"], abs_tol=1e-9):
            raise RuntimeError(f"stored utility mismatch in {row['snapshot_id']}")
        result[candidate_id] = {
            "utility": utilities,
            "hard_safety_rate": hard_safety,
            "bounded_utility_vector": vector.tolist(),
        }
    return result


def per_task_mean(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        values[str(row["task"])].append(float(row[field]))
    return {task: float(np.mean(items)) for task, items in sorted(values.items())}


def analyze_horizon(
    rows: Sequence[Mapping[str, Any]],
    *,
    horizon: int,
    thresholds: Mapping[str, Any],
    candidate_names: Mapping[str, str],
    profiles: Mapping[str, Sequence[float]],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["sampling_stratum"] == "critical"
        and row["quality_horizons"][str(horizon)]["use_for_gate_analysis"]
    ]
    if not selected:
        raise RuntimeError(f"no usable critical families at horizon {horizon}")
    tolerance = float(thresholds["tie_tolerance"])
    analyzed = []
    for row in selected:
        candidates = candidate_summary(row, horizon, profiles)
        main_values = [candidates[candidate_id]["utility"]["main"] for candidate_id in range(6)]
        best_id = int(np.argmax(main_values))
        strict_nonreference = max(main_values[1:]) > main_values[0] + tolerance
        strict_winners = [
            candidate_id
            for candidate_id in range(1, 6)
            if main_values[candidate_id]
            > max(main_values[other] for other in range(6) if other != candidate_id)
            + tolerance
        ]
        best = candidates[best_id]
        reference = candidates[0]
        vector_delta = np.asarray(best["bounded_utility_vector"]) - np.asarray(
            reference["bounded_utility_vector"]
        )
        bad_rate_delta = {
            COMPONENT_NAMES[index]: float(-vector_delta[index]) for index in range(2, 8)
        }
        profile_gains = {}
        for profile in profiles:
            values = [candidates[candidate_id]["utility"][profile] for candidate_id in range(6)]
            profile_gains[profile] = float(max(values) - values[0])
        analyzed.append(
            {
                "snapshot_id": row["snapshot_id"],
                "scenario_group_id": row["scenario_group_id"],
                "task": row["task"],
                "best_candidate_id": best_id,
                "strict_nonreference_best": strict_nonreference,
                "strict_winner_ids": strict_winners,
                "gain": float(main_values[best_id] - main_values[0]),
                "profile_gains": profile_gains,
                "hard_safety_rate_delta": float(
                    best["hard_safety_rate"] - reference["hard_safety_rate"]
                ),
                "task_progress_delta": float(vector_delta[0]),
                "success_rate_delta": float(vector_delta[1]),
                "bad_rate_delta": bad_rate_delta,
            }
        )

    task_counts = Counter(row["task"] for row in analyzed)
    gains = [float(row["gain"]) for row in analyzed]
    groups = [str(row["scenario_group_id"]) for row in analyzed]
    strict_count = sum(bool(row["strict_nonreference_best"]) for row in analyzed)
    task_gains = per_task_mean(analyzed, "gain")
    task_equal = task_equal_bootstrap_mean(
        {
            task: [row["gain"] for row in analyzed if row["task"] == task]
            for task in sorted(task_counts)
        },
        draws=draws,
        seed=seed + 1000,
    )
    bootstrap = group_bootstrap_mean(gains, groups, draws=draws, seed=seed)

    candidate_task_counts: dict[int, Counter[str]] = {
        candidate_id: Counter() for candidate_id in range(1, 6)
    }
    for row in analyzed:
        for candidate_id in row["strict_winner_ids"]:
            candidate_task_counts[candidate_id][row["task"]] += 1
    candidate_task_fractions = {
        str(candidate_id): {
            task: candidate_task_counts[candidate_id][task] / task_counts[task]
            for task in sorted(task_counts)
        }
        for candidate_id in range(1, 6)
    }
    required_ids = [int(value) for value in thresholds["coordination_candidate_ids"]]
    required_fraction = float(
        thresholds["each_required_candidate_strict_win_fraction_in_one_distinct_task_min"]
    )
    eligible_tasks = {
        candidate_id: [
            task
            for task, fraction in candidate_task_fractions[str(candidate_id)].items()
            if fraction >= required_fraction
        ]
        for candidate_id in required_ids
    }
    distinct_matching = maximum_distinct_task_matching(eligible_tasks)

    hard_delta = float(np.mean([row["hard_safety_rate_delta"] for row in analyzed]))
    hard_delta_by_task = per_task_mean(analyzed, "hard_safety_rate_delta")
    raw_delta_fields = (
        "collision_or_drop_rate",
        "robot_conflict_rate",
        "duplicate_work_rate",
        "deadlock_rate",
        "idle_imbalance",
        "normalized_completion_steps",
    )
    raw_deltas = {
        field: float(np.mean([row["bad_rate_delta"][field] for row in analyzed]))
        for field in raw_delta_fields
    }
    raw_deltas_by_task = {
        task: {
            field: float(
                np.mean(
                    [
                        row["bad_rate_delta"][field]
                        for row in analyzed
                        if row["task"] == task
                    ]
                )
            )
            for field in raw_delta_fields
        }
        for task in sorted(task_counts)
    }
    profile_gain = {
        profile: float(np.mean([row["profile_gains"][profile] for row in analyzed]))
        for profile in profiles
    }
    strict_fraction = strict_count / len(analyzed)
    numeric_conditions = {
        "strict_nonreference_best_fraction": strict_fraction
        >= float(thresholds["strict_nonreference_best_fraction_min"]),
        "oracle_family_weighted_mean_gain": bootstrap["mean"]
        >= float(thresholds["oracle_mean_utility_gain_min"]),
        "oracle_group_bootstrap_95_lower": bootstrap["ci95"][0]
        > float(thresholds["oracle_group_bootstrap_95_lower_min"]),
        "tasks_with_positive_point_gain": sum(value > 0.0 for value in task_gains.values())
        >= int(thresholds["tasks_with_positive_point_gain_min"]),
        "coordination_candidate_diversity": len(distinct_matching)
        >= int(thresholds["coordination_candidates_required"]),
        "hard_safety_rate_delta_aggregate": hard_delta
        <= float(thresholds["hard_safety_rate_delta_aggregate_max"]),
        "hard_safety_rate_delta_per_task": max(hard_delta_by_task.values())
        <= float(thresholds["hard_safety_rate_delta_per_task_max"]),
        "alternate_weight_profiles_positive": all(
            profile_gain[profile] > 0.0 for profile in profiles if profile != "main"
        ),
    }
    coverage_all_tasks = set(task_counts) == set(EXPECTED_TASKS) and all(
        task_counts[task] > 0 for task in EXPECTED_TASKS
    )
    protected_nonworsening = all(
        raw_deltas_by_task[task][field] <= 1e-12
        for task in raw_deltas_by_task
        for field in ("collision_or_drop_rate", "normalized_completion_steps")
    )
    top = max(analyzed, key=lambda row: row["gain"])
    without_top = [row["gain"] for row in analyzed if row is not top]
    strict_winner_counts = Counter(
        candidate_id for row in analyzed for candidate_id in row["strict_winner_ids"]
    )
    return {
        "horizon_steps": horizon,
        "usable_critical_family_count": len(analyzed),
        "planned_critical_family_count": 120,
        "usable_critical_coverage_fraction": len(analyzed) / 120.0,
        "usable_critical_count_by_task": {
            task: int(task_counts.get(task, 0)) for task in EXPECTED_TASKS
        },
        "all_six_tasks_represented": coverage_all_tasks,
        "strict_nonreference_best": {
            "count": strict_count,
            "fraction": strict_fraction,
            "wilson_ci95": wilson_interval(strict_count, len(analyzed)),
        },
        "oracle_gain_family_weighted": bootstrap,
        "oracle_gain_task_equal_sensitivity_only": task_equal,
        "oracle_gain_by_task": task_gains,
        "tasks_with_positive_point_gain": sum(value > 0.0 for value in task_gains.values()),
        "strict_winner_count_by_candidate": {
            candidate_names[str(candidate_id)]: int(strict_winner_counts[candidate_id])
            for candidate_id in range(1, 6)
        },
        "coordination_candidate_diversity": {
            "required_candidate_task_fractions": {
                candidate_names[str(candidate_id)]: candidate_task_fractions[str(candidate_id)]
                for candidate_id in required_ids
            },
            "eligible_tasks": {
                candidate_names[str(candidate_id)]: eligible_tasks[candidate_id]
                for candidate_id in required_ids
            },
            "distinct_task_matching": {
                candidate_names[str(candidate_id)]: task
                for candidate_id, task in sorted(distinct_matching.items())
            },
            "matched_candidate_count": len(distinct_matching),
        },
        "alternate_weight_profile_oracle_gain": {
            profile: value for profile, value in profile_gain.items() if profile != "main"
        },
        "main_oracle_outcome_deltas": {
            "hard_safety_rate": hard_delta,
            **raw_deltas,
            "by_task": {
                task: {
                    "hard_safety_rate": hard_delta_by_task[task],
                    **raw_deltas_by_task[task],
                }
                for task in sorted(task_counts)
            },
            "collision_and_completion_nonworsening_in_every_represented_task": protected_nonworsening,
        },
        "outlier_sensitivity": {
            "largest_gain_snapshot_id": top["snapshot_id"],
            "largest_gain_task": top["task"],
            "largest_gain": top["gain"],
            "family_weighted_mean_without_largest_gain": float(np.mean(without_top)),
        },
        "numeric_conditions": numeric_conditions,
        "numeric_gate_pass": all(numeric_conditions.values()) and protected_nonworsening,
        "claimable_gate_pass": all(numeric_conditions.values())
        and protected_nonworsening
        and coverage_all_tasks,
    }


def verify_quality_artifacts(
    quality_summary: Mapping[str, Any], quality_root: Path
) -> dict[str, Any]:
    verified = []
    root = quality_root.resolve()
    for item in quality_summary.get("artifacts", []):
        path = Path(str(item["path"])).resolve()
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(f"quality artifact is outside quality root: {path}") from error
        if not path.is_file():
            raise RuntimeError(f"missing quality sidecar: {path}")
        digest = sha256_file(path)
        size = path.stat().st_size
        if digest != item["sha256"] or size != int(item["size_bytes"]):
            raise RuntimeError(f"quality sidecar drifted: {path}")
        verified.append((str(relative), size, digest))
    if len(verified) != 180:
        raise RuntimeError(f"expected 180 quality sidecars, found {len(verified)}")
    aggregate = __import__("hashlib").sha256()
    for relative, size, digest in sorted(verified):
        aggregate.update(f"{relative}\0{size}\0{digest}\n".encode("utf-8"))
    return {"artifact_count": len(verified), "aggregate_sha256": aggregate.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-contract", type=Path, required=True)
    parser.add_argument("--collection-contract", type=Path, required=True)
    parser.add_argument("--quality-contract", type=Path, required=True)
    parser.add_argument("--analysis-protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--quality-summary", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sha_output = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sha_output.exists():
        raise RuntimeError("refusing to overwrite an existing Gate A report")

    gate_contract = json.loads(args.gate_contract.read_text(encoding="utf-8"))
    collection_contract = json.loads(args.collection_contract.read_text(encoding="utf-8"))
    quality_contract = json.loads(args.quality_contract.read_text(encoding="utf-8"))
    protocol = json.loads(args.analysis_protocol.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_receipt = json.loads(args.source_receipt.read_text(encoding="utf-8"))
    quality_summary = json.loads(args.quality_summary.read_text(encoding="utf-8"))
    if gate_contract.get("stage_id") != GATE_STAGE:
        raise RuntimeError("wrong Gate A source contract")
    if collection_contract.get("stage_id") != COLLECTION_CONTRACT_STAGE:
        raise RuntimeError("wrong A5R7 collection contract")
    if quality_contract.get("stage_id") != QUALITY_STAGE:
        raise RuntimeError("wrong A5R7Q1 quality contract")
    if protocol.get("stage_id") != ANALYSIS_STAGE:
        raise RuntimeError("wrong Gate A analysis protocol")
    if manifest.get("stage_id") != COLLECTION_STAGE:
        raise RuntimeError("wrong A5R7 manifest")
    if quality_summary.get("stage_id") != QUALITY_STAGE:
        raise RuntimeError("wrong A5R7Q1 quality summary")

    source_paths = {
        "gate_contract_sha256": args.gate_contract,
        "collection_contract_sha256": args.collection_contract,
        "quality_contract_sha256": args.quality_contract,
    }
    for key, path in source_paths.items():
        if protocol["source_contracts"][key] != sha256_file(path):
            raise RuntimeError(f"analysis source hash drifted: {key}")
    if manifest.get("contract_sha256") != sha256_file(args.collection_contract):
        raise RuntimeError("manifest does not match collection contract")
    if quality_summary.get("label_contract_sha256") != sha256_file(args.quality_contract):
        raise RuntimeError("quality summary does not match quality contract")
    thresholds = gate_contract["preregistered_gates"]["gate_a_candidate_headroom"]
    if protocol["unchanged_gate_thresholds"] != {
        key: thresholds[key]
        for key in protocol["unchanged_gate_thresholds"]
    }:
        raise RuntimeError("Gate A thresholds changed in the analysis protocol")

    utility_contract = gate_contract["team_outcome_contract"]["utility"]
    profiles = {"main": utility_contract["main_weights"], **utility_contract["weight_profiles"]}
    settings = protocol["analysis_operationalization"]["bootstrap"]
    draws = int(settings["draws"])
    seed = int(settings["seed"])
    raw_before = verify_source_artifacts(source_receipt, args.family_root)
    quality_before = verify_quality_artifacts(quality_summary, args.quality_root)

    rows = []
    for family in manifest["families"]:
        family_path = args.family_root / family["task"] / f"{family['snapshot_id']}.json"
        quality_path = args.quality_root / family["task"] / f"{family['snapshot_id']}.quality.json"
        row = json.loads(family_path.read_text(encoding="utf-8"))
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        if row.get("stage_id") != COLLECTION_STAGE or row.get("branch_count") != 24:
            raise RuntimeError(f"invalid A5R7 family: {family_path}")
        if quality.get("stage_id") != QUALITY_STAGE:
            raise RuntimeError(f"invalid A5R7Q1 sidecar: {quality_path}")
        if quality.get("source_family_sha256") != sha256_file(family_path):
            raise RuntimeError(f"quality sidecar source drifted: {quality_path}")
        row["quality_horizons"] = quality["horizons"]
        rows.append(row)
    if len(rows) != 180:
        raise RuntimeError(f"expected 180 A5R7 families, found {len(rows)}")

    horizon_results = {
        str(horizon): analyze_horizon(
            rows,
            horizon=horizon,
            thresholds=thresholds,
            candidate_names=protocol["candidate_names"],
            profiles=profiles,
            draws=draws,
            seed=seed,
        )
        for horizon in HORIZONS
    }
    primary = horizon_results[str(protocol["analysis_operationalization"]["primary_horizon_steps"])]
    raw_after = verify_source_artifacts(source_receipt, args.family_root)
    quality_after = verify_quality_artifacts(quality_summary, args.quality_root)
    if raw_after != raw_before or quality_after != quality_before:
        raise RuntimeError("A5R7 raw data or quality sidecars changed during Gate A analysis")

    failed_numeric = [
        key for key, passed in primary["numeric_conditions"].items() if not passed
    ]
    decision_reasons = []
    if failed_numeric:
        decision_reasons.append("PRIMARY_HORIZON_NUMERIC_CONDITION_FAILED")
    if not primary["main_oracle_outcome_deltas"][
        "collision_and_completion_nonworsening_in_every_represented_task"
    ]:
        decision_reasons.append("PROTECTED_OUTCOME_SACRIFICE")
    if not primary["all_six_tasks_represented"]:
        decision_reasons.append("PRIMARY_HORIZON_INCOMPLETE_TASK_COVERAGE")
    report = {
        "format_version": "before-we-act.a5r7q2-care-gate-a-report/1",
        "stage_id": ANALYSIS_STAGE,
        "created_at_utc": utc_now(),
        "verdict": "PASS" if primary["claimable_gate_pass"] else "NOT_PASSED",
        "required_action": (
            "PROCEED_TO_GATE_B_ONLY_IF_SEPARATELY_AUTHORIZED"
            if primary["claimable_gate_pass"]
            else protocol["decision_rule"]["failure_action"]
        ),
        "decision_reasons": decision_reasons,
        "failed_primary_numeric_conditions": failed_numeric,
        "plain_language_conclusion_zh": (
            "Gate A 通过。"
            if primary["claimable_gate_pass"]
            else "Gate A 没有通过：当前六种候选虽经常有替代动作略好，但平均收益没有达到冻结门槛，且第64步缺少完整六任务覆盖。"
        ),
        "source_hashes": {
            "gate_contract": sha256_file(args.gate_contract),
            "collection_contract": sha256_file(args.collection_contract),
            "quality_contract": sha256_file(args.quality_contract),
            "analysis_protocol": sha256_file(args.analysis_protocol),
            "manifest": sha256_file(args.manifest),
            "source_receipt": sha256_file(args.source_receipt),
            "quality_summary": sha256_file(args.quality_summary),
        },
        "source_integrity": {
            "raw_before": raw_before,
            "raw_after": raw_after,
            "quality_before": quality_before,
            "quality_after": quality_after,
            "raw_modified": False,
            "quality_sidecars_modified": False,
        },
        "analysis_protocol": protocol["analysis_operationalization"],
        "posthoc_disclosure": protocol["posthoc_disclosure"],
        "thresholds": protocol["unchanged_gate_thresholds"],
        "primary_horizon_steps": protocol["analysis_operationalization"][
            "primary_horizon_steps"
        ],
        "horizons": horizon_results,
        "claim_limits": [
            "A5R7 and this report are Gate-only and forbidden for training.",
            "NOT_PASSED means the frozen Gate A burden of proof was not met; it does not prove every alternative action is useless.",
            "The task-equal sensitivity analysis is not allowed to override the frozen family-weighted main result.",
            "No Gate B decision is made by this report."
        ],
    }
    atomic_json(args.output, report)
    digest = sha256_file(args.output)
    sha_output.write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "required_action": report["required_action"],
                "decision_reasons": decision_reasons,
                "failed_primary_numeric_conditions": failed_numeric,
                "primary": primary,
                "report": str(args.output.resolve()),
                "report_sha256": digest,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
