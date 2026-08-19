#!/usr/bin/env python3
"""Summarize the independent paired Confirmation50 closed-loop experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


TASKS = (
    "lift_barrier",
    "camera_alignment",
    "long_pipeline_delivery",
    "take_photo",
    "pass_shoe",
    "place_food",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_mcnemar_two_sided(bcore_only: int, direct_only: int) -> float:
    discordant = bcore_only + direct_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(bcore_only, direct_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_bootstrap_ci(
    differences: np.ndarray, *, samples: int, seed: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    count = len(differences)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        size = min(1000, samples - start)
        indices = rng.integers(0, count, size=(size, count))
        means[start : start + size] = differences[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def load_result(path: Path, mode: str, expected_seeds: set[int]) -> dict[int, dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("rows", [])
    mapped = {int(row["seed"]): row for row in rows}
    if value.get("mode") != mode or len(rows) != 50 or len(mapped) != 50:
        raise RuntimeError(f"incomplete or wrong-mode result: {path}")
    if set(mapped) != expected_seeds:
        raise RuntimeError(f"result seeds differ from frozen manifest: {path}")
    return mapped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--bcore-root", type=Path, required=True)
    parser.add_argument("--direct-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    expected = contract["immutable_inputs"]
    authorization_receipts_valid = True
    for item in contract["authorization_evidence"].values():
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            authorization_receipts_valid = False
            continue
        receipt = json.loads(path.read_text(encoding="utf-8"))
        authorization_receipts_valid &= receipt.get("status") == item["required_status"]
    if not authorization_receipts_valid:
        raise RuntimeError("Confirmation50 authorization evidence drifted")
    for name in ("bcore_checkpoint", "training_checkpoint", "b0h_checkpoint"):
        item = expected[name]
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"frozen checkpoint drifted: {name}")

    overall_differences: list[int] = []
    task_summaries = []
    all_direct_actions_finite = True
    all_seed_receipts_match = True
    all_confirmation_seeds_disjoint = True
    result_hashes = {"bcore": {}, "direct": {}}
    for task in TASKS:
        seed_path = args.seed_root / f"{task}.json"
        expected_seed_hash = contract["seed_protocol"]["confirmation50_sha256"][task]
        if sha256_file(seed_path) != expected_seed_hash:
            raise RuntimeError(f"Confirmation50 seed manifest drifted: {task}")
        manifest = json.loads(seed_path.read_text(encoding="utf-8"))
        expected_seeds = {int(value) for value in manifest["seeds"]}
        validation_path = (
            Path(contract["seed_protocol"]["excluded_validation20_root"])
            / f"{task}.json"
        )
        if sha256_file(validation_path) != contract["seed_protocol"][
            "excluded_validation20_sha256"
        ][task]:
            raise RuntimeError(f"Validation20 seed manifest drifted: {task}")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        all_confirmation_seeds_disjoint &= not expected_seeds.intersection(
            int(value) for value in validation["seeds"]
        )
        bcore_path = args.bcore_root / f"{task}.json"
        direct_path = args.direct_root / f"{task}.json"
        bcore_value = json.loads(bcore_path.read_text(encoding="utf-8"))
        direct_value = json.loads(direct_path.read_text(encoding="utf-8"))
        bcore = load_result(bcore_path, "n2", expected_seeds)
        direct = load_result(direct_path, "direct_reactive", expected_seeds)
        all_seed_receipts_match &= (
            bcore_value.get("seed_protocol", {}).get("sha256") == expected_seed_hash
            and direct_value.get("seed_protocol", {}).get("sha256") == expected_seed_hash
        )

        bcore_only = direct_only = both_success = both_failure = 0
        differences = []
        for seed in manifest["seeds"]:
            b_success = bool(bcore[int(seed)]["success"])
            d_success = bool(direct[int(seed)]["success"])
            bcore_only += int(b_success and not d_success)
            direct_only += int(d_success and not b_success)
            both_success += int(b_success and d_success)
            both_failure += int(not b_success and not d_success)
            differences.append(int(b_success) - int(d_success))
            all_direct_actions_finite &= bool(
                direct[int(seed)].get("finite_actions", False)
            )
        difference = np.asarray(differences, dtype=np.int8)
        overall_differences.extend(differences)
        task_summaries.append(
            {
                "task": task,
                "episodes": 50,
                "bcore_successes": int(sum(bool(row["success"]) for row in bcore.values())),
                "direct_successes": int(sum(bool(row["success"]) for row in direct.values())),
                "bcore_only_successes": bcore_only,
                "direct_only_successes": direct_only,
                "both_success": both_success,
                "both_failure": both_failure,
                "net_bcore_successes": int(difference.sum()),
                "mcnemar_exact_two_sided_p": exact_mcnemar_two_sided(
                    bcore_only, direct_only
                ),
            }
        )
        result_hashes["bcore"][task] = sha256_file(bcore_path)
        result_hashes["direct"][task] = sha256_file(direct_path)

    difference = np.asarray(overall_differences, dtype=np.int8)
    bcore_only = int(sum(row["bcore_only_successes"] for row in task_summaries))
    direct_only = int(sum(row["direct_only_successes"] for row in task_summaries))
    bcore_successes = int(sum(row["bcore_successes"] for row in task_summaries))
    direct_successes = int(sum(row["direct_successes"] for row in task_summaries))
    p_value = exact_mcnemar_two_sided(bcore_only, direct_only)
    positive_tasks = sum(row["net_bcore_successes"] > 0 for row in task_summaries)
    negative_tasks = sum(row["net_bcore_successes"] < 0 for row in task_summaries)
    rules = contract["decision_rules"]
    overall_confirmed = (
        bcore_successes > direct_successes
        and p_value < float(rules["paired_significance_alpha"])
    )
    broad = (
        overall_confirmed
        and positive_tasks >= int(rules["broad_support_min_positive_tasks"])
        and negative_tasks <= int(rules["broad_support_max_negative_tasks"])
    )
    if broad:
        status = "CONFIRMED_BROAD_CLOSED_LOOP_BENEFIT"
    elif overall_confirmed:
        status = "CONFIRMED_TASK_CONCENTRATED_BENEFIT"
    else:
        status = "CLOSED_LOOP_BENEFIT_NOT_CONFIRMED"

    integrity = {
        "complete_300_paired_episodes": len(difference) == 300,
        "authorization_receipts_valid": authorization_receipts_valid,
        "all_direct_actions_finite": all_direct_actions_finite,
        "all_seed_receipts_match": all_seed_receipts_match,
        "confirmation50_disjoint_from_validation20": all_confirmation_seeds_disjoint,
    }
    if not all(integrity.values()):
        status = "BLOCKED_BY_INTEGRITY_FAILURE"

    result = {
        "contract": str(args.contract.resolve()),
        "contract_sha256": sha256_file(args.contract),
        "decision_rule_frozen_before_results": rules,
        "format_version": "before-we-act.b3-n3-confirmation50-summary/1",
        "integrity": integrity,
        "interpretation_boundary_zh": [
            "该实验确认完整B-core相对原训练中直接历史修正分支的闭环差异。",
            "两者训练数据、训练种子和更新次数相同，但总参数量并不严格相等。",
            "总体显著但跨任务广度不足时，只能报告收益集中于特定协作任务。",
            "该实验不使用CARE分支数据，也不把内部位置强行命名为人工语义因子。",
        ],
        "overall": {
            "episodes": 300,
            "bcore_successes": bcore_successes,
            "direct_successes": direct_successes,
            "bcore_success_rate": bcore_successes / 300,
            "direct_success_rate": direct_successes / 300,
            "net_bcore_successes": bcore_successes - direct_successes,
            "percentage_point_difference": 100.0 * (bcore_successes - direct_successes) / 300,
            "bcore_only_successes": bcore_only,
            "direct_only_successes": direct_only,
            "both_success": int(sum(row["both_success"] for row in task_summaries)),
            "both_failure": int(sum(row["both_failure"] for row in task_summaries)),
            "mcnemar_exact_two_sided_p": p_value,
            "paired_difference_bootstrap_ci95": paired_bootstrap_ci(
                difference,
                samples=int(rules["bootstrap_samples"]),
                seed=int(rules["bootstrap_seed"]),
            ),
            "positive_tasks": positive_tasks,
            "negative_tasks": negative_tasks,
        },
        "result_sha256": result_hashes,
        "stage": contract["stage"],
        "status": status,
        "tasks": task_summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
