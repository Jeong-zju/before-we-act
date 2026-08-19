#!/usr/bin/env python3
"""Retest A5R7 Gate A with hard safety separated from ordinary benefit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.before_we_act.evaluate_care_gate_a import (
    COLLECTION_CONTRACT_STAGE,
    COLLECTION_STAGE,
    EXPECTED_TASKS,
    GATE_STAGE,
    HORIZONS,
    QUALITY_STAGE,
    atomic_json,
    branch_by_key,
    group_bootstrap_mean,
    maximum_distinct_task_matching,
    sha256_file,
    task_equal_bootstrap_mean,
    utc_now,
    verify_quality_artifacts,
    verify_source_artifacts,
    wilson_interval,
)


SEPARATED_CONTRACT_STAGE = "A4R8-CARE-SEPARATED-GATE-A"
REPORT_STAGE = "A5R7Q3-CARE-SEPARATED-GATE-A-RETEST"
REPEAT_STABILITY_TOLERANCE = 0.02
COLLISION_TOLERANCE = 1e-12


def separated_weight_profiles(gate_contract: Mapping[str, Any]) -> dict[str, list[float]]:
    utility = gate_contract["team_outcome_contract"]["utility"]
    source = {"main": utility["main_weights"], **utility["weight_profiles"]}
    result = {}
    for name, profile in source.items():
        weights = np.asarray(profile, dtype=np.float64)
        if weights.shape != (8,):
            raise RuntimeError(f"weight profile {name} must have eight entries")
        weights[2] = 0.0
        total = float(weights.sum())
        if total <= 0.0:
            raise RuntimeError(f"weight profile {name} has no ordinary-benefit mass")
        result[name] = (weights / total).tolist()
    return result


def ordinary_benefit(outcome: Mapping[str, Any], weights: Sequence[float]) -> float:
    vector = np.asarray(outcome["bounded_utility_vector"], dtype=np.float64)
    if vector.shape != (8,):
        raise RuntimeError("bounded utility vector must have eight entries")
    return float(np.dot(np.asarray(weights, dtype=np.float64), vector))


def candidate_records(
    row: Mapping[str, Any],
    horizon: int,
    profiles: Mapping[str, Sequence[float]],
) -> dict[int, dict[str, Any]]:
    raw: dict[int, dict[str, Any]] = {}
    for candidate_id in range(6):
        outcomes = [
            branch_by_key(row, candidate_id, repeat_id)["outcomes"][str(horizon)]
            for repeat_id in (0, 1)
        ]
        raw[candidate_id] = {
            "hard_safety": [bool(outcome["hard_safety_violation"]) for outcome in outcomes],
            "collision_rate": [
                -float(outcome["bounded_utility_vector"][2]) for outcome in outcomes
            ],
            "completion_rate": [
                -float(outcome["bounded_utility_vector"][7]) for outcome in outcomes
            ],
            "benefit": {
                profile: [ordinary_benefit(outcome, weights) for outcome in outcomes]
                for profile, weights in profiles.items()
            },
        }
    reference = raw[0]
    for candidate_id, record in raw.items():
        hard_stable = record["hard_safety"][0] == record["hard_safety"][1]
        benefit_difference = abs(record["benefit"]["main"][0] - record["benefit"]["main"][1])
        benefit_stable = benefit_difference <= REPEAT_STABILITY_TOLERANCE
        hard_nonworsening = all(
            int(record["hard_safety"][repeat_id])
            <= int(reference["hard_safety"][repeat_id])
            for repeat_id in (0, 1)
        )
        collision_nonworsening = all(
            record["collision_rate"][repeat_id]
            <= reference["collision_rate"][repeat_id] + COLLISION_TOLERANCE
            for repeat_id in (0, 1)
        )
        reasons = []
        if not hard_stable:
            reasons.append("HARD_SAFETY_REPEAT_UNSTABLE")
        if not benefit_stable:
            reasons.append("ORDINARY_BENEFIT_REPEAT_UNSTABLE")
        if not hard_nonworsening:
            reasons.append("PAIRED_HARD_SAFETY_WORSE")
        if not collision_nonworsening:
            reasons.append("PAIRED_COLLISION_RATE_WORSE")
        record.update(
            {
                "eligible": not reasons,
                "ineligibility_reasons": reasons,
                "ordinary_benefit_repeat_abs_difference": benefit_difference,
                "mean_benefit": {
                    profile: float(np.mean(values))
                    for profile, values in record["benefit"].items()
                },
                "mean_hard_safety": float(np.mean(record["hard_safety"])),
                "mean_collision_rate": float(np.mean(record["collision_rate"])),
                "mean_completion_rate": float(np.mean(record["completion_rate"])),
            }
        )
    if not raw[0]["eligible"]:
        raise RuntimeError(f"reference candidate is ineligible in {row['snapshot_id']}")
    return raw


def mean_by_task(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task"])].append(float(row[field]))
    return {task: float(np.mean(values)) for task, values in sorted(grouped.items())}


def analyze_horizon(
    rows: Sequence[Mapping[str, Any]],
    *,
    horizon: int,
    separated_contract: Mapping[str, Any],
    candidate_names: Mapping[int, str],
    profiles: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    scope = separated_contract["unchanged_scope"]
    gate = separated_contract["separated_gate_a"]
    selected = [
        row
        for row in rows
        if row["sampling_stratum"] == "critical"
        and row["quality_horizons"][str(horizon)]["use_for_gate_analysis"]
    ]
    if not selected:
        raise RuntimeError(f"no usable critical families at horizon {horizon}")
    tolerance = float(scope["tie_tolerance"])
    analyzed = []
    exclusion_counts: Counter[tuple[int, str]] = Counter()
    eligibility_counts: Counter[int] = Counter()
    eligibility_counts_by_task: Counter[tuple[str, int]] = Counter()
    stable_safety_improvement_count = 0
    stable_collision_improvement_count = 0
    for row in selected:
        records = candidate_records(row, horizon, profiles)
        eligible = [candidate_id for candidate_id, record in records.items() if record["eligible"]]
        for candidate_id, record in records.items():
            if record["eligible"]:
                eligibility_counts[candidate_id] += 1
                eligibility_counts_by_task[(str(row["task"]), candidate_id)] += 1
            else:
                for reason in record["ineligibility_reasons"]:
                    exclusion_counts[(candidate_id, reason)] += 1
        reference = records[0]
        if any(
            all(
                int(reference["hard_safety"][repeat_id])
                > int(records[candidate_id]["hard_safety"][repeat_id])
                for repeat_id in (0, 1)
            )
            for candidate_id in eligible
            if candidate_id != 0
        ):
            stable_safety_improvement_count += 1
        if any(
            all(
                records[candidate_id]["collision_rate"][repeat_id]
                <= reference["collision_rate"][repeat_id] + COLLISION_TOLERANCE
                for repeat_id in (0, 1)
            )
            and any(
                records[candidate_id]["collision_rate"][repeat_id]
                < reference["collision_rate"][repeat_id] - COLLISION_TOLERANCE
                for repeat_id in (0, 1)
            )
            for candidate_id in eligible
            if candidate_id != 0
        ):
            stable_collision_improvement_count += 1

        values = {candidate_id: records[candidate_id]["mean_benefit"]["main"] for candidate_id in eligible}
        best_id = max(eligible, key=lambda candidate_id: (values[candidate_id], -candidate_id))
        strict_nonreference = max(
            (values[candidate_id] for candidate_id in eligible if candidate_id != 0),
            default=-math.inf,
        ) > values[0] + tolerance
        strict_winners = [
            candidate_id
            for candidate_id in eligible
            if candidate_id != 0
            and values[candidate_id]
            > max(values[other] for other in eligible if other != candidate_id) + tolerance
        ]
        profile_gains = {}
        for profile in profiles:
            profile_values = {
                candidate_id: records[candidate_id]["mean_benefit"][profile]
                for candidate_id in eligible
            }
            profile_gains[profile] = float(max(profile_values.values()) - profile_values[0])
        best = records[best_id]
        analyzed.append(
            {
                "snapshot_id": row["snapshot_id"],
                "scenario_group_id": row["scenario_group_id"],
                "task": row["task"],
                "best_candidate_id": best_id,
                "strict_nonreference_best": strict_nonreference,
                "strict_winner_ids": strict_winners,
                "gain": float(values[best_id] - values[0]),
                "profile_gains": profile_gains,
                "hard_safety_rate_delta": float(
                    best["mean_hard_safety"] - reference["mean_hard_safety"]
                ),
                "collision_rate_delta": float(
                    best["mean_collision_rate"] - reference["mean_collision_rate"]
                ),
                "completion_rate_delta": float(
                    best["mean_completion_rate"] - reference["mean_completion_rate"]
                ),
            }
        )

    task_counts = Counter(row["task"] for row in analyzed)
    gains = [float(row["gain"]) for row in analyzed]
    groups = [str(row["scenario_group_id"]) for row in analyzed]
    draws = int(scope["bootstrap_draws"])
    seed = int(scope["bootstrap_seed"])
    bootstrap = group_bootstrap_mean(gains, groups, draws=draws, seed=seed)
    task_equal = task_equal_bootstrap_mean(
        {
            task: [row["gain"] for row in analyzed if row["task"] == task]
            for task in sorted(task_counts)
        },
        draws=draws,
        seed=seed + 1000,
    )
    strict_count = sum(bool(row["strict_nonreference_best"]) for row in analyzed)
    task_gains = mean_by_task(analyzed, "gain")
    hard_delta = float(np.mean([row["hard_safety_rate_delta"] for row in analyzed]))
    collision_delta = float(np.mean([row["collision_rate_delta"] for row in analyzed]))
    completion_delta = float(np.mean([row["completion_rate_delta"] for row in analyzed]))
    hard_by_task = mean_by_task(analyzed, "hard_safety_rate_delta")
    collision_by_task = mean_by_task(analyzed, "collision_rate_delta")
    completion_by_task = mean_by_task(analyzed, "completion_rate_delta")
    profile_gain = {
        profile: float(np.mean([row["profile_gains"][profile] for row in analyzed]))
        for profile in profiles
    }

    winner_counts = Counter(
        candidate_id for row in analyzed for candidate_id in row["strict_winner_ids"]
    )
    required_ids = [int(value) for value in gate["coordination_candidate_ids"]]
    task_fractions = {
        candidate_id: {
            task: sum(
                candidate_id in row["strict_winner_ids"]
                for row in analyzed
                if row["task"] == task
            )
            / task_counts[task]
            for task in sorted(task_counts)
        }
        for candidate_id in required_ids
    }
    required_fraction = float(
        gate["each_required_candidate_strict_win_fraction_in_one_distinct_task_min"]
    )
    eligible_tasks = {
        candidate_id: [
            task for task, fraction in task_fractions[candidate_id].items() if fraction >= required_fraction
        ]
        for candidate_id in required_ids
    }
    matching = maximum_distinct_task_matching(eligible_tasks)
    strict_fraction = strict_count / len(analyzed)
    conditions = {
        "strict_nonreference_best_fraction": strict_fraction
        >= float(gate["strict_nonreference_best_fraction_min"]),
        "mean_ordinary_benefit_gain": bootstrap["mean"]
        >= float(gate["oracle_mean_ordinary_benefit_gain_min"]),
        "group_bootstrap_95_lower": bootstrap["ci95"][0]
        > float(gate["oracle_group_bootstrap_95_lower_min"]),
        "tasks_with_positive_point_gain": sum(value > 0.0 for value in task_gains.values())
        >= int(gate["tasks_with_positive_point_gain_min"]),
        "coordination_candidate_diversity": len(matching)
        >= int(gate["coordination_candidates_required"]),
        "alternate_weight_profiles_positive": all(
            profile_gain[profile] > 0.0 for profile in profiles if profile != "main"
        ),
        "selected_hard_safety_nonworsening": hard_delta <= COLLISION_TOLERANCE
        and max(hard_by_task.values()) <= COLLISION_TOLERANCE,
        "selected_collision_rate_nonworsening": collision_delta <= COLLISION_TOLERANCE
        and max(collision_by_task.values()) <= COLLISION_TOLERANCE,
        "selected_completion_time_nonworsening": max(completion_by_task.values())
        <= COLLISION_TOLERANCE,
    }
    all_tasks = set(task_counts) == set(EXPECTED_TASKS) and all(
        task_counts[task] > 0 for task in EXPECTED_TASKS
    )
    top = max(analyzed, key=lambda row: row["gain"])
    return {
        "horizon_steps": horizon,
        "usable_critical_family_count": len(analyzed),
        "planned_critical_family_count": 120,
        "usable_critical_count_by_task": {
            task: int(task_counts.get(task, 0)) for task in EXPECTED_TASKS
        },
        "all_six_tasks_represented": all_tasks,
        "candidate_eligibility": {
            "eligible_family_count_by_candidate": {
                candidate_names[candidate_id]: int(eligibility_counts[candidate_id])
                for candidate_id in range(6)
            },
            "eligible_fraction_by_task_and_candidate": {
                task: {
                    candidate_names[candidate_id]: eligibility_counts_by_task[(task, candidate_id)]
                    / task_counts[task]
                    for candidate_id in range(6)
                }
                for task in sorted(task_counts)
            },
            "exclusion_reason_counts": {
                f"{candidate_names[candidate_id]}|{reason}": count
                for (candidate_id, reason), count in sorted(exclusion_counts.items())
            },
        },
        "separate_safety_channel": {
            "families_with_stable_hard_safety_improvement": stable_safety_improvement_count,
            "families_with_stable_collision_rate_improvement": stable_collision_improvement_count,
            "selected_oracle_hard_safety_rate_delta": hard_delta,
            "selected_oracle_collision_rate_delta": collision_delta,
            "selected_oracle_hard_safety_rate_delta_by_task": hard_by_task,
            "selected_oracle_collision_rate_delta_by_task": collision_by_task,
        },
        "ordinary_benefit": {
            "strict_nonreference_best_count": strict_count,
            "strict_nonreference_best_fraction": strict_fraction,
            "strict_fraction_wilson_ci95": wilson_interval(strict_count, len(analyzed)),
            "family_weighted_oracle_gain": bootstrap,
            "task_equal_sensitivity_only": task_equal,
            "gain_by_task": task_gains,
            "alternate_profile_oracle_gain": {
                profile: value for profile, value in profile_gain.items() if profile != "main"
            },
            "largest_gain_snapshot_id": top["snapshot_id"],
            "largest_gain_task": top["task"],
            "largest_gain": top["gain"],
        },
        "coordination_candidate_diversity": {
            "strict_winner_count": {
                candidate_names[candidate_id]: int(winner_counts[candidate_id])
                for candidate_id in range(1, 6)
            },
            "required_candidate_task_fractions": {
                candidate_names[candidate_id]: task_fractions[candidate_id]
                for candidate_id in required_ids
            },
            "distinct_task_matching": {
                candidate_names[candidate_id]: task
                for candidate_id, task in sorted(matching.items())
            },
            "matched_candidate_count": len(matching),
        },
        "completion_rate_delta": completion_delta,
        "completion_rate_delta_by_task": completion_by_task,
        "conditions": conditions,
        "diagnostic_numeric_gate_pass": all(conditions.values()),
        "complete_diagnostic_gate_pass": all(conditions.values()) and all_tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-contract", type=Path, required=True)
    parser.add_argument("--collection-contract", type=Path, required=True)
    parser.add_argument("--quality-contract", type=Path, required=True)
    parser.add_argument("--separated-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--quality-summary", type=Path, required=True)
    parser.add_argument("--old-gate-report", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checksum_path = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or checksum_path.exists():
        raise RuntimeError("refusing to overwrite separated Gate A report")

    gate_contract = json.loads(args.gate_contract.read_text(encoding="utf-8"))
    collection_contract = json.loads(args.collection_contract.read_text(encoding="utf-8"))
    quality_contract = json.loads(args.quality_contract.read_text(encoding="utf-8"))
    separated_contract = json.loads(args.separated_contract.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_receipt = json.loads(args.source_receipt.read_text(encoding="utf-8"))
    quality_summary = json.loads(args.quality_summary.read_text(encoding="utf-8"))
    if gate_contract.get("stage_id") != GATE_STAGE:
        raise RuntimeError("wrong A4 source contract")
    if collection_contract.get("stage_id") != COLLECTION_CONTRACT_STAGE:
        raise RuntimeError("wrong A4R7 source contract")
    if quality_contract.get("stage_id") != QUALITY_STAGE:
        raise RuntimeError("wrong A5R7Q1 quality contract")
    if separated_contract.get("stage_id") != SEPARATED_CONTRACT_STAGE:
        raise RuntimeError("wrong A4R8 separated Gate A contract")
    if manifest.get("stage_id") != COLLECTION_STAGE:
        raise RuntimeError("wrong A5R7 manifest")
    if quality_summary.get("stage_id") != QUALITY_STAGE:
        raise RuntimeError("wrong A5R7Q1 summary")
    parents = separated_contract["parent_artifacts"]
    expected = {
        "a4_gate_contract": args.gate_contract,
        "a4r7_collection_contract": args.collection_contract,
        "a5r7q1_quality_contract": args.quality_contract,
        "a5r7q2_original_gate_a_report": args.old_gate_report,
    }
    for key, path in expected.items():
        if parents[key]["sha256"] != sha256_file(path):
            raise RuntimeError(f"A4R8 parent artifact drifted: {key}")
    old_report_hash_before = sha256_file(args.old_gate_report)
    raw_before = verify_source_artifacts(source_receipt, args.family_root)
    quality_before = verify_quality_artifacts(quality_summary, args.quality_root)
    profiles = separated_weight_profiles(gate_contract)
    frozen_main = np.asarray(
        separated_contract["separated_outcomes"]["ordinary_benefit_channel"][
            "main_weights_after_renormalization"
        ],
        dtype=np.float64,
    )
    if not np.allclose(frozen_main, profiles["main"], rtol=0.0, atol=1e-15):
        raise RuntimeError("A4R8 frozen separated main weights drifted")

    candidate_names = {
        int(row["id"]): str(row["name"])
        for row in gate_contract["candidate_family"]["candidates"]
    }
    rows = []
    for family in manifest["families"]:
        family_path = args.family_root / family["task"] / f"{family['snapshot_id']}.json"
        quality_path = args.quality_root / family["task"] / f"{family['snapshot_id']}.quality.json"
        row = json.loads(family_path.read_text(encoding="utf-8"))
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        if row.get("stage_id") != COLLECTION_STAGE or int(row.get("branch_count", -1)) != 24:
            raise RuntimeError(f"invalid A5R7 family: {family_path}")
        if quality.get("stage_id") != QUALITY_STAGE:
            raise RuntimeError(f"invalid A5R7Q1 sidecar: {quality_path}")
        if quality.get("source_family_sha256") != sha256_file(family_path):
            raise RuntimeError(f"quality sidecar source drifted: {quality_path}")
        row["quality_horizons"] = quality["horizons"]
        rows.append(row)
    if len(rows) != 180:
        raise RuntimeError(f"expected 180 A5R7 families, found {len(rows)}")

    results = {
        str(horizon): analyze_horizon(
            rows,
            horizon=horizon,
            separated_contract=separated_contract,
            candidate_names=candidate_names,
            profiles=profiles,
        )
        for horizon in HORIZONS
    }
    primary_horizon = int(separated_contract["unchanged_scope"]["primary_horizon_steps"])
    primary = results[str(primary_horizon)]
    raw_after = verify_source_artifacts(source_receipt, args.family_root)
    quality_after = verify_quality_artifacts(quality_summary, args.quality_root)
    old_report_hash_after = sha256_file(args.old_gate_report)
    if raw_after != raw_before or quality_after != quality_before:
        raise RuntimeError("source data changed during separated Gate A retest")
    if old_report_hash_after != old_report_hash_before:
        raise RuntimeError("old Gate A report changed during separated retest")

    failed = [key for key, passed in primary["conditions"].items() if not passed]
    decision_reasons = []
    if failed:
        decision_reasons.append("SEPARATED_PRIMARY_NUMERIC_CONDITION_FAILED")
    if not primary["all_six_tasks_represented"]:
        decision_reasons.append("PRIMARY_HORIZON_INCOMPLETE_TASK_COVERAGE")
    verdict = (
        "POSTHOC_DIAGNOSTIC_PASS_REQUIRES_UNTOUCHED_CONFIRMATION"
        if primary["complete_diagnostic_gate_pass"]
        else "NOT_PASSED"
    )
    report = {
        "format_version": "before-we-act.a5r7q3-care-separated-gate-a-report/1",
        "stage_id": REPORT_STAGE,
        "created_at_utc": utc_now(),
        "verdict": verdict,
        "required_action": (
            "FREEZE_AND_RUN_UNTOUCHED_CONFIRMATION_BEFORE_TRAINING"
            if primary["complete_diagnostic_gate_pass"]
            else separated_contract["coverage_and_decision"]["failure_action"]
        ),
        "decision_reasons": decision_reasons,
        "failed_primary_conditions": failed,
        "plain_language_conclusion_zh": (
            "安全与普通收益分开、并排除重复不稳定候选后，Gate A 仍未通过。"
            if verdict == "NOT_PASSED"
            else "事后复核的数值条件通过，但不能代替未看结果的正式确认。"
        ),
        "primary_horizon_steps": primary_horizon,
        "source_hashes": {
            "gate_contract": sha256_file(args.gate_contract),
            "collection_contract": sha256_file(args.collection_contract),
            "quality_contract": sha256_file(args.quality_contract),
            "separated_contract": sha256_file(args.separated_contract),
            "manifest": sha256_file(args.manifest),
            "source_receipt": sha256_file(args.source_receipt),
            "quality_summary": sha256_file(args.quality_summary),
            "old_gate_report": old_report_hash_before,
        },
        "source_integrity": {
            "raw_before": raw_before,
            "raw_after": raw_after,
            "quality_before": quality_before,
            "quality_after": quality_after,
            "old_gate_report_before": old_report_hash_before,
            "old_gate_report_after": old_report_hash_after,
            "raw_modified": False,
            "quality_sidecars_modified": False,
            "old_gate_report_modified": False,
        },
        "separated_weight_profiles": profiles,
        "posthoc_disclosure": separated_contract["posthoc_disclosure"],
        "horizons": results,
        "claim_limits": [
            "This retest does not relabel or delete any A5R7 branch.",
            "Candidate-level repeat instability changes Gate eligibility only and is not called a simulator anomaly.",
            "A5R7 remains forbidden for training.",
            "The unchanged 0.05 threshold cannot be lowered after inspecting A5R7 to manufacture a pass."
        ],
    }
    atomic_json(args.output, report)
    digest = sha256_file(args.output)
    checksum_path.write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": verdict,
                "required_action": report["required_action"],
                "decision_reasons": decision_reasons,
                "failed_primary_conditions": failed,
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
