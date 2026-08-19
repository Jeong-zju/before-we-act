#!/usr/bin/env python3
"""Evaluate the preregistered A4R10 common-snapshot option pilot."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.before_we_act.evaluate_care_gate_a import (
    atomic_json,
    group_bootstrap_mean,
    task_equal_bootstrap_mean,
    utc_now,
)
from scripts.before_we_act.evaluate_care_gate_a_separated import ordinary_benefit
from before_we_act.care_branch_collector import sha256_file


CONTRACT_STAGE = "A4R10-CARE-COMMON-SNAPSHOT-OPTION-PILOT"
MANIFEST_STAGE = "A5R9-CARE-COMMON-SNAPSHOT-OPTION-PILOT"
REPORT_STAGE = "A5R9Q1-CARE-COMMON-SNAPSHOT-OPTION-PILOT-EVALUATION"
EXPECTED_TASKS = (
    "camera_alignment",
    "lift_barrier",
    "long_pipeline_delivery",
    "pass_shoe",
    "place_food",
    "take_photo",
)
VALID_STATUSES = {"VALID", "SUCCESS_TERMINATION"}
COLLISION_TOLERANCE = 1e-12


def branch_by_duration(
    row: Mapping[str, Any], candidate_id: int, repeat_id: int, duration: int
) -> Mapping[str, Any]:
    selected = [
        branch
        for branch in row["branches"]
        if int(branch["candidate_id"]) == int(candidate_id)
        and int(branch["repeat_id"]) == int(repeat_id)
        and branch["regime"] == "reactive"
        and int(branch.get("intervention_steps_requested", 1)) == int(duration)
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"{row['snapshot_id']} missing unique branch "
            f"{(candidate_id, repeat_id, duration)}"
        )
    return selected[0]


def collision_rate(outcome: Mapping[str, Any]) -> float:
    return -float(outcome["bounded_utility_vector"][2])


def completion_rate(outcome: Mapping[str, Any]) -> float:
    return -float(outcome["bounded_utility_vector"][7])


def candidate_records(
    branches: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    horizon: int,
    weights: Sequence[float],
    repeat_tolerance: float,
) -> dict[int, dict[str, Any]]:
    raw: dict[int, dict[str, Any]] = {}
    for candidate_id in range(6):
        pair = list(branches[candidate_id])
        if len(pair) != 2:
            raise RuntimeError("candidate record must contain two repeats")
        outcomes = []
        branch_failures = []
        for branch in pair:
            reasons = []
            if branch.get("status") not in VALID_STATUSES:
                reasons.append(f"BRANCH_STATUS:{branch.get('status')}")
            if not bool(branch.get("candidate_valid", False)):
                reasons.append("INVALID_CANDIDATE")
            if str(horizon) not in branch.get("outcomes", {}):
                reasons.append("MISSING_HORIZON")
            branch_failures.append(reasons)
            outcomes.append(branch.get("outcomes", {}).get(str(horizon)))
        complete = all(outcome is not None and not failures for outcome, failures in zip(outcomes, branch_failures))
        if complete:
            hard_safety = [bool(outcome["hard_safety_violation"]) for outcome in outcomes]
            collisions = [collision_rate(outcome) for outcome in outcomes]
            completion = [completion_rate(outcome) for outcome in outcomes]
            benefits = [ordinary_benefit(outcome, weights) for outcome in outcomes]
        else:
            hard_safety, collisions, completion, benefits = [], [], [], []
        raw[candidate_id] = {
            "complete": complete,
            "branch_failures": branch_failures,
            "hard_safety": hard_safety,
            "collision_rate": collisions,
            "completion_rate": completion,
            "benefit": benefits,
        }

    reference = raw[0]
    if not reference["complete"]:
        reference["eligible"] = False
        reference["ineligibility_reasons"] = ["REFERENCE_BRANCH_INCOMPLETE"]
        return raw
    for candidate_id, record in raw.items():
        reasons = []
        if not record["complete"]:
            reasons.append("BRANCH_INCOMPLETE")
        else:
            if record["hard_safety"][0] != record["hard_safety"][1]:
                reasons.append("HARD_SAFETY_REPEAT_UNSTABLE")
            benefit_difference = abs(record["benefit"][0] - record["benefit"][1])
            if benefit_difference > repeat_tolerance:
                reasons.append("ORDINARY_BENEFIT_REPEAT_UNSTABLE")
            if any(
                int(record["hard_safety"][repeat])
                > int(reference["hard_safety"][repeat])
                for repeat in (0, 1)
            ):
                reasons.append("PAIRED_HARD_SAFETY_WORSE")
            if any(
                record["collision_rate"][repeat]
                > reference["collision_rate"][repeat] + COLLISION_TOLERANCE
                for repeat in (0, 1)
            ):
                reasons.append("PAIRED_COLLISION_RATE_WORSE")
            record["ordinary_benefit_repeat_abs_difference"] = benefit_difference
            record["mean_benefit"] = float(np.mean(record["benefit"]))
            record["mean_hard_safety"] = float(np.mean(record["hard_safety"]))
            record["mean_collision_rate"] = float(np.mean(record["collision_rate"]))
            record["mean_completion_rate"] = float(np.mean(record["completion_rate"]))
        record["eligible"] = not reasons
        record["ineligibility_reasons"] = reasons
    return raw


def summarize_duration(
    families: Sequence[Mapping[str, Any]],
    *,
    duration: int,
    horizon: int,
    weights: Sequence[float],
    repeat_tolerance: float,
    tie_tolerance: float,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    analyzed = []
    exclusions: Counter[tuple[int, str]] = Counter()
    for family in families:
        pilot = family["pilot"]
        branch_map = {
            0: [branch_by_duration(pilot, 0, repeat, 1) for repeat in (0, 1)]
        }
        for candidate_id in range(1, 6):
            branch_map[candidate_id] = [
                branch_by_duration(pilot, candidate_id, repeat, duration)
                for repeat in (0, 1)
            ]
        records = candidate_records(
            branch_map,
            horizon=horizon,
            weights=weights,
            repeat_tolerance=repeat_tolerance,
        )
        if not records[0].get("eligible", False):
            continue
        eligible = [candidate_id for candidate_id, record in records.items() if record.get("eligible", False)]
        for candidate_id, record in records.items():
            for reason in record.get("ineligibility_reasons", []):
                exclusions[(candidate_id, reason)] += 1
        values = {candidate_id: records[candidate_id]["mean_benefit"] for candidate_id in eligible}
        best_id = max(eligible, key=lambda candidate_id: (values[candidate_id], -candidate_id))
        reference = records[0]
        best = records[best_id]
        analyzed.append(
            {
                "snapshot_id": pilot["snapshot_id"],
                "source_a5r7_snapshot_id": pilot["source_a5r7_snapshot_id"],
                "scenario_group_id": pilot["scenario_group_id"],
                "task": pilot["task"],
                "best_candidate_id": best_id,
                "strict_nonreference_best": max(
                    (values[candidate_id] for candidate_id in eligible if candidate_id != 0),
                    default=-math.inf,
                )
                > values[0] + tie_tolerance,
                "gain": float(values[best_id] - values[0]),
                "hard_safety_rate_delta": float(best["mean_hard_safety"] - reference["mean_hard_safety"]),
                "collision_rate_delta": float(best["mean_collision_rate"] - reference["mean_collision_rate"]),
                "completion_rate_delta": float(best["mean_completion_rate"] - reference["mean_completion_rate"]),
                "eligible_candidate_ids": eligible,
            }
        )
    if not analyzed:
        raise RuntimeError(f"duration {duration} has no analyzable families")
    gains = [row["gain"] for row in analyzed]
    groups = [row["scenario_group_id"] for row in analyzed]
    bootstrap = group_bootstrap_mean(gains, groups, draws=draws, seed=seed + duration)
    by_task = defaultdict(list)
    for row in analyzed:
        by_task[row["task"]].append(row)
    task_gain = {
        task: float(np.mean([row["gain"] for row in rows]))
        for task, rows in sorted(by_task.items())
    }
    task_equal = task_equal_bootstrap_mean(
        {task: [row["gain"] for row in rows] for task, rows in by_task.items()},
        draws=draws,
        seed=seed + 1000 + duration,
    )
    hard_by_task = {
        task: float(np.mean([row["hard_safety_rate_delta"] for row in rows]))
        for task, rows in by_task.items()
    }
    collision_by_task = {
        task: float(np.mean([row["collision_rate_delta"] for row in rows]))
        for task, rows in by_task.items()
    }
    completion_by_task = {
        task: float(np.mean([row["completion_rate_delta"] for row in rows]))
        for task, rows in by_task.items()
    }
    return {
        "duration_steps": duration,
        "horizon_steps": horizon,
        "analyzed_family_count": len(analyzed),
        "task_counts": dict(sorted(Counter(row["task"] for row in analyzed).items())),
        "all_six_tasks_represented": set(by_task) == set(EXPECTED_TASKS),
        "strict_nonreference_best_count": sum(row["strict_nonreference_best"] for row in analyzed),
        "strict_nonreference_best_fraction": float(np.mean([row["strict_nonreference_best"] for row in analyzed])),
        "ordinary_benefit_gain": bootstrap,
        "task_equal_ordinary_benefit_gain": task_equal,
        "ordinary_benefit_gain_by_task": task_gain,
        "tasks_with_positive_point_gain": sum(value > 0 for value in task_gain.values()),
        "selected_hard_safety_rate_delta_mean": float(np.mean([row["hard_safety_rate_delta"] for row in analyzed])),
        "selected_collision_rate_delta_mean": float(np.mean([row["collision_rate_delta"] for row in analyzed])),
        "selected_completion_rate_delta_mean": float(np.mean([row["completion_rate_delta"] for row in analyzed])),
        "selected_hard_safety_rate_delta_by_task": hard_by_task,
        "selected_collision_rate_delta_by_task": collision_by_task,
        "selected_completion_rate_delta_by_task": completion_by_task,
        "candidate_ineligibility_counts": [
            {"candidate_id": candidate_id, "reason": reason, "count": count}
            for (candidate_id, reason), count in sorted(exclusions.items())
        ],
        "best_candidate_counts": dict(sorted(Counter(row["best_candidate_id"] for row in analyzed).items())),
        "families": analyzed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--separated-contract", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite pilot evaluation: {args.output}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    separated = json.loads(args.separated_contract.read_text(encoding="utf-8"))
    if manifest.get("stage_id") != MANIFEST_STAGE or manifest.get("status") != "FROZEN_BEFORE_OPTION_OUTCOMES":
        raise RuntimeError("A5R9 pilot manifest is not frozen")
    if contract.get("stage_id") != CONTRACT_STAGE or contract.get("status") != "FROZEN_BEFORE_OPTION_OUTCOMES":
        raise RuntimeError("A4R10 pilot contract is not frozen")
    if sha256_file(args.contract) != manifest["contract_sha256"]:
        raise RuntimeError("A4R10 contract drifted")
    expected_separated_sha = contract["parents"]["a4r8_separated_gate_a_contract"]["sha256"]
    if sha256_file(args.separated_contract) != expected_separated_sha:
        raise RuntimeError("A4R8 separated Gate A contract drifted")
    weights = separated["separated_outcomes"]["ordinary_benefit_channel"]["main_weights_after_renormalization"]
    repeat_tolerance = 0.02
    tie_tolerance = float(contract["ordinary_benefit_and_safety"]["tie_tolerance"])
    decision = contract["pilot_decision_rule"]
    draws = int(decision["bootstrap_draws"])
    seed = int(decision["bootstrap_seed"])

    loaded = []
    for family in manifest["families"]:
        pilot_path = args.family_root / family["task"] / f"{family['snapshot_id']}.json"
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        source_path = Path(family["source_a5r7_family"])
        quality_path = Path(family["source_a5r7_quality"])
        if sha256_file(source_path) != family["source_a5r7_family_sha256"]:
            raise RuntimeError(f"A5R7 source family drifted: {source_path}")
        if sha256_file(quality_path) != family["source_a5r7_quality_sha256"]:
            raise RuntimeError(f"A5R7Q1 quality sidecar drifted: {quality_path}")
        if pilot.get("contract_sha256") != manifest["contract_sha256"] or pilot.get("branch_count") != 32:
            raise RuntimeError(f"invalid A5R9 pilot family: {pilot_path}")
        loaded.append({"manifest": family, "pilot": pilot})

    horizon_reports = {}
    for horizon in (8, 16, 32):
        eligible = loaded
        horizon_reports[str(horizon)] = {
            str(duration): summarize_duration(
                eligible,
                duration=duration,
                horizon=horizon,
                weights=weights,
                repeat_tolerance=repeat_tolerance,
                tie_tolerance=tie_tolerance,
                draws=draws,
                seed=seed + horizon * 100,
            )
            for duration in (1, 4, 8)
        }

    primary_horizon = str(int(decision["primary_horizon_steps"]))
    one = horizon_reports[primary_horizon]["1"]
    four = horizon_reports[primary_horizon]["4"]
    one_by_id = {row["snapshot_id"]: row for row in one["families"]}
    paired = [
        (row["gain"] - one_by_id[row["snapshot_id"]]["gain"], row["scenario_group_id"])
        for row in four["families"]
        if row["snapshot_id"] in one_by_id
    ]
    paired_improvement = group_bootstrap_mean(
        [row[0] for row in paired],
        [row[1] for row in paired],
        draws=draws,
        seed=seed + 4000,
    )
    thresholds = decision["worth_formal_confirmation_if_all"]
    protected = (
        four["selected_hard_safety_rate_delta_mean"] <= COLLISION_TOLERANCE
        and four["selected_collision_rate_delta_mean"] <= COLLISION_TOLERANCE
        and max(four["selected_hard_safety_rate_delta_by_task"].values()) <= COLLISION_TOLERANCE
        and max(four["selected_collision_rate_delta_by_task"].values()) <= COLLISION_TOLERANCE
        and max(four["selected_completion_rate_delta_by_task"].values()) <= COLLISION_TOLERANCE
    )
    conditions = {
        "mean_ordinary_benefit_gain": four["ordinary_benefit_gain"]["mean"]
        >= float(thresholds["mean_ordinary_benefit_gain_min"]),
        "ordinary_benefit_gain_95_lower": four["ordinary_benefit_gain"]["ci95"][0]
        > float(thresholds["ordinary_benefit_gain_95_lower_min_exclusive"]),
        "paired_gain_improvement_over_one_step_95_lower": paired_improvement["ci95"][0]
        > float(thresholds["paired_gain_improvement_over_one_step_95_lower_min_exclusive"]),
        "strict_nonreference_best_fraction": four["strict_nonreference_best_fraction"]
        >= float(thresholds["strict_nonreference_best_fraction_min"]),
        "tasks_with_positive_point_gain": four["tasks_with_positive_point_gain"]
        >= int(thresholds["tasks_with_positive_point_gain_min"]),
        "protected_outcomes_nonworsening": protected,
        "all_six_tasks_represented": bool(four["all_six_tasks_represented"]),
    }
    worth_formal = all(conditions.values())
    strong = worth_formal and four["ordinary_benefit_gain"]["mean"] >= float(
        decision["strong_signal_if_mean_gain_reaches_formal_threshold"]
    )
    verdict = (
        "STRONG_SIGNAL_FORMAL_CONFIRMATION_JUSTIFIED"
        if strong
        else "PROMISING_SMALL_FORMAL_CONFIRMATION_JUSTIFIED"
        if worth_formal
        else "FORMAL_RECOLLECTION_NOT_JUSTIFIED"
    )
    report = {
        "format_version": "before-we-act.a5r9q1-care-common-snapshot-option-pilot-evaluation/1",
        "stage_id": REPORT_STAGE,
        "created_at_utc": utc_now(),
        "status": "COMPLETED",
        "verdict": verdict,
        "contract": str(args.contract.resolve()),
        "contract_sha256": sha256_file(args.contract),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "family_count": len(loaded),
        "new_branch_count": sum(row["pilot"]["branch_count"] for row in loaded),
        "source_data_modified": False,
        "formal_gate_a_claim": False,
        "gate_b_claim": False,
        "ordinary_benefit_weights": weights,
        "common_snapshot_state_match_a5r7_count": sum(
            bool(row["pilot"]["snapshot_state_matches_a5r7"]) for row in loaded
        ),
        "horizons": horizon_reports,
        "primary_decision": {
            "duration_steps": 4,
            "horizon_steps": int(primary_horizon),
            "one_step_mean_gain": one["ordinary_benefit_gain"]["mean"],
            "four_step_mean_gain": four["ordinary_benefit_gain"]["mean"],
            "paired_four_minus_one_gain": paired_improvement,
            "conditions": conditions,
            "worth_formal_confirmation": worth_formal,
            "strong_signal": strong,
            "verdict": verdict,
        },
        "sensitivity_disclosure": "Eight-step results are exploratory and did not determine the preregistered verdict.",
    }
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "verdict": verdict,
                "one_step_mean_gain": report["primary_decision"]["one_step_mean_gain"],
                "four_step_mean_gain": report["primary_decision"]["four_step_mean_gain"],
                "paired_improvement_ci95": paired_improvement["ci95"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
