#!/usr/bin/env python3
"""Issue the evidence-bounded Step-4 offline or closed-loop verdict."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean

import torch

from before_we_act.temporal_history_data import SIX_TASKS, sha256_file


SEEDS = (20260815, 20260816, 20260817)
VARIANTS = ("a4_full", "a4_no_bottleneck")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def training_rows(run_root: Path, bcore_root: Path) -> tuple[dict, dict]:
    rows = {}
    old = {}
    for seed in SEEDS:
        old_status_path = bcore_root / f"seed_{seed}" / "status.json"
        old_status = read_json(old_status_path)
        old[seed] = {
            "status_path": str(old_status_path.resolve()),
            "status_sha256": sha256_file(old_status_path),
            "action_mse": float(
                old_status["selected_validation"]["macro"]["b_core"]
            ),
        }
    for variant in VARIANTS:
        rows[variant] = {}
        for seed in SEEDS:
            root = run_root / "training" / variant / f"seed_{seed}"
            status_path = root / "status.json"
            status = read_json(status_path)
            validation = status["selected_validation"]
            evaluations_path = root / "evaluations.jsonl"
            evaluations = read_jsonl(evaluations_path)
            last_four_action = [
                float(item["validation"]["macro"]["b_core"])
                for item in evaluations[-4:]
            ]
            last_four_relative_range = (
                (max(last_four_action) - min(last_four_action))
                / max(mean(last_four_action), 1e-12)
                if len(last_four_action) == 4
                else None
            )
            deployment_path = root / "deployment_checkpoint.pt"
            deployment = torch.load(
                deployment_path, map_location="cpu", weights_only=False
            )
            deployment_keys = tuple(deployment["model"])
            deployment_safety = deployment["config"].get(
                "residual_safety", {}
            )
            training_only_absent = not any(
                "base_conditioned_prior" in key
                or "teacher_branch" in key
                for key in deployment_keys
            )
            correct = float(validation["macro"]["b_core"])
            shuffled = float(validation["macro"]["b_shuffle"])
            feature_std = float(validation["belief"]["feature_std_mean"])
            shuffle_relative = abs(shuffled - correct) / max(correct, 1e-12)
            rows[variant][seed] = {
                "root": str(root.resolve()),
                "status_path": str(status_path.resolve()),
                "status_sha256": sha256_file(status_path),
                "evaluations_path": str(evaluations_path.resolve()),
                "evaluations_sha256": sha256_file(evaluations_path),
                "deployment_checkpoint": str(deployment_path.resolve()),
                "deployment_checkpoint_sha256": sha256_file(deployment_path),
                "training_only_prior_and_teacher_absent": training_only_absent,
                "deployment_residual_safety_enabled": bool(
                    deployment_safety.get("enabled", False)
                ),
                "deployment_residual_safety": deployment_safety,
                "status": status["status"],
                "update": int(status["update"]),
                "selected_update": int(status["selected_update"]),
                "action_mse": correct,
                "b0h_action_mse": float(validation["macro"]["b0h"]),
                "shuffled_action_mse": shuffled,
                "shuffle_minus_matched": shuffled - correct,
                "shuffle_relative_improvement": (shuffled - correct)
                / max(shuffled, 1e-12),
                "belief_off_max_abs": float(validation["belief_off_max_abs"]),
                "belief_feature_std_mean": feature_std,
                "collapsed": feature_std <= 1e-4 and shuffle_relative <= 1e-3,
                "conditional_kl_nats": float(
                    validation["base_relative"]["conditional_kl_nats"]
                ),
                "nuisance_belief_relative_mse": float(
                    validation["base_relative"]["nuisance_proxy"][
                        "belief_relative_mse"
                    ]
                ),
                "nuisance_residual_relative_mse": float(
                    validation["base_relative"]["nuisance_proxy"][
                        "residual_relative_mse"
                    ]
                ),
                "last_four_action_mse": last_four_action,
                "last_four_action_mse_relative_range": last_four_relative_range,
            }
    return rows, old


def closed_loop_scores(root: Path, labels) -> dict:
    result = {}
    for label in labels:
        task_rows = {}
        for task in SIX_TASKS:
            path = root / label / f"{task}.json"
            payload = read_json(path)
            task_rows[task] = {
                "successes": int(payload["successes"]),
                "episodes": int(payload["episodes"]),
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
        result[label] = {
            "tasks": task_rows,
            "successes": sum(row["successes"] for row in task_rows.values()),
            "episodes": sum(row["episodes"] for row in task_rows.values()),
        }
    return result


def validation20_gates(row: dict) -> dict:
    tasks = row["tasks"]
    four = ("lift_barrier", "long_pipeline_delivery", "take_photo", "pass_shoe")
    return {
        "a4_upgrade_total_ge_105": row["successes"] >= 105,
        "all_episodes_120": row["episodes"] == 120,
        "four_easy_total_ge_72": sum(tasks[name]["successes"] for name in four)
        >= 72,
        "each_easy_ge_16": all(tasks[name]["successes"] >= 16 for name in four),
        "camera_ge_6": tasks["camera_alignment"]["successes"] >= 6,
        "camera_plus_food_ge_8": tasks["camera_alignment"]["successes"]
        + tasks["place_food"]["successes"]
        >= 8,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bcore-training-root", type=Path, required=True)
    parser.add_argument("--sufficiency", type=Path)
    parser.add_argument("--validation5-root", type=Path)
    parser.add_argument("--validation20-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = read_json(args.contract)
    thresholds = contract["acceptance"]
    rows, old = training_rows(args.run_root, args.bcore_training_root)
    full = list(rows["a4_full"].values())
    control = list(rows["a4_no_bottleneck"].values())

    completed = all(row["update"] == 120_000 for row in full + control)
    no_overfit = all(row["status"] != "SATURATED_BY_OVERFIT" for row in full)
    no_collapse = all(not row["collapsed"] for row in full)
    belief_off = all(row["belief_off_max_abs"] == 0.0 for row in full)
    deployment_clean = all(
        row["training_only_prior_and_teacher_absent"] for row in full
    )
    deployment_safe = all(
        row["deployment_residual_safety_enabled"] for row in full
    )
    ratios = {
        seed: rows["a4_full"][seed]["action_mse"] / old[seed]["action_mse"]
        for seed in SEEDS
    }
    offline_vs_old = all(
        value <= thresholds["per_seed_action_mse_vs_b_core_max_ratio"]
        for value in ratios.values()
    )
    mean_b0h = mean(row["b0h_action_mse"] for row in full)
    mean_full = mean(row["action_mse"] for row in full)
    mean_control = mean(row["action_mse"] for row in control)
    better_than_b0h = mean_full < mean_b0h
    relevance_positive = sum(row["shuffle_minus_matched"] > 0 for row in full)
    relevance_mean = mean(row["shuffle_relative_improvement"] for row in full)
    action_relevance = (
        relevance_positive
        >= thresholds["action_relevance_seeds_positive_min"]
        and relevance_mean
        >= thresholds["action_relevance_mean_relative_improvement_min"]
    )
    hard_offline = {
        "all_six_runs_completed_120k": completed,
        "no_full_seed_continuous_overfit": no_overfit,
        "no_full_seed_joint_collapse": no_collapse,
        "belief_off_exact_base_all_full_seeds": belief_off,
        "deployment_has_no_prior_or_privileged_teacher": deployment_clean,
        "deployment_has_residual_safety_and_base_fallback": deployment_safe,
        "each_full_seed_action_mse_le_old_bcore_1p05": offline_vs_old,
        "full_mean_action_mse_better_than_b0h": better_than_b0h,
    }
    hard_offline_passed = all(hard_offline.values())

    full_k = mean(row["conditional_kl_nats"] for row in full)
    control_k = mean(row["conditional_kl_nats"] for row in control)
    full_nuisance = mean(row["nuisance_residual_relative_mse"] for row in full)
    control_nuisance = mean(
        row["nuisance_residual_relative_mse"] for row in control
    )
    conditional_minimality = {
        "full_mean_k_cond_nats": full_k,
        "no_bottleneck_mean_k_cond_nats": control_k,
        "full_k_cond_lower": full_k < control_k,
        "full_mean_nuisance_residual_relative_mse": full_nuisance,
        "no_bottleneck_mean_nuisance_residual_relative_mse": control_nuisance,
        "full_nuisance_lower": full_nuisance < control_nuisance,
        "full_action_mse_not_over_five_percent_worse": mean_full
        <= 1.05 * mean_control,
    }
    conditional_minimality["soft_direction_passed"] = all(
        value
        for key, value in conditional_minimality.items()
        if key.endswith("lower") or key.endswith("worse")
    )

    stability_limit = float(
        thresholds["curve_stability_relative_range_max"]
    )
    curve_stability = {
        variant: {
            str(seed): {
                "last_four_action_mse": rows[variant][seed][
                    "last_four_action_mse"
                ],
                "relative_range": rows[variant][seed][
                    "last_four_action_mse_relative_range"
                ],
                "stable_below_one_percent": rows[variant][seed][
                    "last_four_action_mse_relative_range"
                ]
                is not None
                and rows[variant][seed][
                    "last_four_action_mse_relative_range"
                ]
                < stability_limit,
            }
            for seed in SEEDS
        }
        for variant in VARIANTS
    }
    curve_stability["all_full_seeds_soft_direction_passed"] = all(
        row["stable_below_one_percent"]
        for row in curve_stability["a4_full"].values()
    )

    sufficiency = None
    if args.sufficiency is not None:
        sufficiency = read_json(args.sufficiency)

    validation5 = None
    validation5_gate = None
    if args.validation5_root is not None:
        validation5 = closed_loop_scores(
            args.validation5_root,
            [f"seed_{seed}" for seed in SEEDS],
        )
        scores = [value["successes"] for value in validation5.values()]
        validation5_gate = {
            "each_seed_has_30_episodes": all(
                value["episodes"] == 30 for value in validation5.values()
            ),
            "mean_successes_ge_24": mean(scores)
            >= thresholds["validation5_mean_min_successes"],
            "each_seed_successes_ge_23": min(scores)
            >= thresholds["validation5_single_seed_min_successes"],
        }
        validation5_gate["passed"] = all(validation5_gate.values())

    validation20 = None
    validation20_gate = None
    if args.validation20_root is not None:
        validation20 = closed_loop_scores(
            args.validation20_root, ["selected"]
        )["selected"]
        validation20_gate = validation20_gates(validation20)
        validation20_gate["passed"] = all(validation20_gate.values())

    if not hard_offline_passed:
        status = "FAILED_HARD_OFFLINE_GATES"
    elif validation5_gate is not None and not validation5_gate["passed"]:
        status = "FAILED_VALIDATION5_GATE"
    elif validation20_gate is not None and not validation20_gate["passed"]:
        status = "FAILED_VALIDATION20_HARD_GATE"
    elif validation20_gate is None:
        status = "PASSED_OFFLINE_AWAITING_CLOSED_LOOP"
    elif action_relevance:
        status = "PASSED_STEP4_ACCEPT"
    else:
        status = "PASSED_STEP4_CONDITIONAL_ACTION_ATTRIBUTION_OPEN"

    payload = {
        "format_version": "before-we-act.a4-acceptance/1",
        "status": status,
        "contract_sha256": sha256_file(args.contract),
        "training": rows,
        "old_bcore": old,
        "offline": {
            "hard_checks": hard_offline,
            "hard_passed": hard_offline_passed,
            "full_to_old_bcore_ratios": ratios,
            "full_mean_action_mse": mean_full,
            "no_bottleneck_mean_action_mse": mean_control,
            "b0h_mean_action_mse": mean_b0h,
        },
        "action_relevance": {
            "positive_seeds": relevance_positive,
            "mean_relative_improvement": relevance_mean,
            "passed_moderate_gate": action_relevance,
        },
        "conditional_minimality": conditional_minimality,
        "curve_stability": curve_stability,
        "control_sufficiency": sufficiency,
        "validation5": validation5,
        "validation5_gate": validation5_gate,
        "validation20": validation20,
        "validation20_gate": validation20_gate,
        "claim_boundary": {
            "control_sufficiency": "validated" if sufficiency and sufficiency["status"] == "PASSED_SOFT_TARGET" else "approximate-learning-objective-only",
            "conditional_minimality": "direction-supported" if conditional_minimality["soft_direction_passed"] else "objective-only",
            "belief_action_use": "supported" if action_relevance else "conditional-attribution-only",
            "confirmation50_completed": False,
            "independent_belief_contribution_fully_identified": False,
        },
    }
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
