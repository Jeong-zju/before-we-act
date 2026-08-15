#!/usr/bin/env python3
"""Classify N2 offline evidence, then optionally its paired Validation5."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import median

from before_we_act.step2_temporal_data import SIX_TASKS, sha256_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_training(contract: dict, root: Path) -> tuple[dict, bool]:
    result = {}
    all_sufficient = True
    for seed in contract["training"]["seeds"]:
        seed_root = root / "training" / f"seed_{seed}"
        status_path = seed_root / "status.json"
        sufficiency_path = seed_root / "training_sufficiency.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        sufficiency = json.loads(sufficiency_path.read_text(encoding="utf-8"))
        if status["training_sufficiency_sha256"] != sha256_file(sufficiency_path):
            raise RuntimeError(f"N2 seed {seed} sufficiency receipt hash differs")
        selected_update = int(status["selected_update"])
        selected = next(
            json.loads(line)
            for line in (seed_root / "evaluations.jsonl").read_text().splitlines()
            if int(json.loads(line)["update"]) == selected_update
        )
        deployment = seed_root / "deployment_checkpoint.pt"
        if status["deployment_checkpoint_sha256"] != sha256_file(deployment):
            raise RuntimeError(f"N2 seed {seed} deployment hash differs")
        result[str(seed)] = {
            "status": status["status"],
            "selected_update": selected_update,
            "selected_validation": selected["validation"],
            "training_sufficiency": sufficiency,
            "training_sufficiency_sha256": sha256_file(sufficiency_path),
            "deployment_checkpoint": str(deployment.resolve()),
            "deployment_checkpoint_sha256": sha256_file(deployment),
        }
        all_sufficient &= status["status"] in {"PLATFORM_REACHED", "SATURATED_BY_OVERFIT"}
    return result, all_sufficient


def offline_gate(training: dict) -> dict:
    rows = list(training.values())
    action_better = all(
        row["selected_validation"]["macro"]["b_core"]
        < row["selected_validation"]["macro"]["b0h"]
        for row in rows
    )
    shuffle_clean = all(
        row["selected_validation"]["macro"]["b_core"]
        < row["selected_validation"]["macro"]["b_shuffle"]
        for row in rows
    )
    task_medians = {}
    for task_index, task in enumerate(SIX_TASKS):
        improvements = [
            (
                row["selected_validation"]["per_task"]["b0h"][str(task_index)]
                - row["selected_validation"]["per_task"]["b_core"][str(task_index)]
            )
            / max(
                abs(row["selected_validation"]["per_task"]["b0h"][str(task_index)]),
                1e-12,
            )
            for row in rows
        ]
        task_medians[task] = float(median(improvements))
    positive_tasks = sum(value > 0 for value in task_medians.values())
    future_1p6 = all(
        row["selected_validation"]["future_mse"]["model"]["1.6s"]
        < row["selected_validation"]["future_mse"]["persistence"]["1.6s"]
        and row["selected_validation"]["future_mse"]["model"]["1.6s"]
        < row["selected_validation"]["future_mse"]["shuffle"]["1.6s"]
        for row in rows
    )
    noncollapsed = all(
        row["selected_validation"]["belief"]["feature_std_mean"] > 0.01
        and row["selected_validation"]["belief"]["effective_rank"] > 4
        for row in rows
    )
    off_exact = all(
        row["selected_validation"]["belief_off_max_abs"] == 0.0 for row in rows
    )
    uncertainty = all(
        row["selected_validation"]["uncertainty_occlusion"]["sigma_occluded"]
        > row["selected_validation"]["uncertainty_occlusion"]["sigma_clean"]
        and row["selected_validation"]["uncertainty_occlusion"]["reliability_occluded"]
        < row["selected_validation"]["uncertainty_occlusion"]["reliability_clean"]
        for row in rows
    )
    passed = all(
        (action_better, shuffle_clean, positive_tasks >= 4, future_1p6, noncollapsed, off_exact, uncertainty)
    )
    return {
        "b_core_beats_b0h_every_seed": action_better,
        "b_core_beats_shuffle_every_seed": shuffle_clean,
        "task_relative_improvement_medians": task_medians,
        "positive_task_medians": positive_tasks,
        "future_1p6_beats_persistence_and_shuffle_every_seed": future_1p6,
        "belief_noncollapsed_every_seed": noncollapsed,
        "belief_off_exact_every_seed": off_exact,
        "occlusion_uncertainty_direction_every_seed": uncertainty,
        "passed": passed,
    }


def load_validation(root: Path, contract: dict) -> dict:
    modes = {"b0h": root / "b0h"}
    modes.update(
        {
            f"seed_{seed}": root / f"seed_{seed}"
            for seed in contract["training"]["seeds"]
        }
    )
    output = {}
    for name, mode_root in modes.items():
        task_rows = {
            task: json.loads((mode_root / f"{task}.json").read_text(encoding="utf-8"))
            for task in SIX_TASKS
        }
        if any(row.get("episodes") != 5 for row in task_rows.values()):
            raise RuntimeError(f"incomplete N2 Validation5: {name}")
        output[name] = {
            "successes": sum(int(row["successes"]) for row in task_rows.values()),
            "episodes": 30,
            "paired_inactivity_steps": sum(int(row["paired_inactivity_steps"]) for row in task_rows.values()),
            "steps": sum(int(row["steps"]) for row in task_rows.values()),
            "tasks": {task: int(row["successes"]) for task, row in task_rows.items()},
            "receipts": {task: sha256_file(mode_root / f"{task}.json") for task in SIX_TASKS},
        }
        output[name]["paired_inactivity_rate"] = (
            output[name]["paired_inactivity_steps"] / max(output[name]["steps"], 1)
        )
    return output


def validation_gate(validation: dict, contract: dict) -> dict:
    baseline = validation["b0h"]
    candidates = [validation[f"seed_{seed}"] for seed in contract["training"]["seeds"]]
    totals = [row["successes"] for row in candidates]
    aggregate_positive = median(totals) > baseline["successes"]
    nonnegative_seeds = sum(value >= baseline["successes"] for value in totals)
    protected = ("lift_barrier", "long_pipeline_delivery", "take_photo", "pass_shoe")
    protected_not_damaged = all(
        median([row["tasks"][task] for row in candidates]) >= baseline["tasks"][task]
        for task in protected
    )
    lower_wait = sum(
        row["paired_inactivity_rate"] < baseline["paired_inactivity_rate"]
        for row in candidates
    )
    return {
        "baseline_successes": baseline["successes"],
        "candidate_successes": totals,
        "candidate_median_successes": float(median(totals)),
        "aggregate_positive": aggregate_positive,
        "nonnegative_seed_count": nonnegative_seeds,
        "protected_tasks_not_damaged_in_median": protected_not_damaged,
        "lower_paired_inactivity_seed_count": lower_wait,
        "cooperation_proxy_positive": lower_wait >= 2,
        "passed": aggregate_positive and nonnegative_seeds >= 2 and protected_not_damaged and lower_wait >= 2,
    }


def main() -> None:
    args = parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    training, all_sufficient = load_training(contract, args.run_root)
    gate = offline_gate(training) if all_sufficient else None
    validation_authorized = bool(all_sufficient and gate and gate["passed"])
    validation = None
    validation_result = None
    if args.validation_root is not None:
        if not validation_authorized:
            raise RuntimeError("Validation5 was supplied without a training-sufficient positive offline gate")
        validation = load_validation(args.validation_root, contract)
        validation_result = validation_gate(validation, contract)

    if not all_sufficient:
        status = "INCONCLUSIVE_TRAINING_NOT_CONVERGED"
        summary = "三颗 N2 都跑满了，但至少一颗的验证曲线还在明显变化；现在既不能说架构有效，也不能说无效，更不能打开 Validation5。"
    elif not gate["passed"]:
        status = "NO_SIGNAL"
        summary = "训练已经够久，但完整 B-core 没同时守住动作、1.6 秒后果、打乱对照、非坍缩和不确定性方向；当前版本不值得直接做闭环。"
    elif validation_result is None:
        status = "OFFLINE_POSITIVE_VALIDATION5_AUTHORIZED"
        summary = "完整 B-core 在三颗种子上通过了离线方向门禁，且 1.6 秒后果不是简单延续或打乱目标能解释；现在允许跑固定 Validation5，但还不能声称闭环有效。"
    elif validation_result["passed"]:
        status = "POSITIVE_SIGNAL"
        summary = "完整 B-core 的离线动作和 1.6 秒后果信号都稳定，固定 Validation5 的总体成功和等待代理也朝更好方向走；值得进入 N3 做结构归因，但这还不是正式 B-core 通过。"
    elif validation_result["aggregate_positive"]:
        status = "WEAK_SIGNAL"
        summary = "离线收益稳定，Validation5 总成功略有正向，但强任务或等待代理没有同时站住；可以保留候选分析，不能直接进入正式训练。"
    else:
        status = "NO_SIGNAL"
        summary = "离线动作看起来更准，但固定 Validation5 没有给出总体正向响应；当前完整架构不能继续按正信号解释。"
    payload = {
        "format_version": "before-we-act.b3-n2-conclusion/1",
        "stage": "B3-N2-ARCHITECTURE",
        "status": status,
        "completed_at_utc": utc_now(),
        "contract": str(args.contract.resolve()),
        "contract_sha256": sha256_file(args.contract),
        "training": training,
        "all_seeds_training_sufficient": all_sufficient,
        "offline_gate": gate,
        "validation5_authorized": validation_authorized,
        "validation5": validation,
        "validation5_gate": validation_result,
        "n3_authorized": status == "POSITIVE_SIGNAL",
        "formal_pass": False,
        "human_summary": summary,
        "claim_limits": [
            "N2 is exploratory and cannot issue PASSED_FORMAL",
            "direct/reactive attribution remains for N3",
            "the invalid all-zero R1-3 target was excluded, so causal teammate-correction claims remain deferred",
            "Validation5 has 5 episodes/task and is directional, not a formal success-rate estimate",
        ],
    }
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
