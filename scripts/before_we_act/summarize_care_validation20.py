#!/usr/bin/env python3
"""Summarize paired six-task CARE Validation20 in plain, auditable terms."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from before_we_act.care_training_data import atomic_json, sha256_file


TASKS = (
    "lift_barrier",
    "camera_alignment",
    "long_pipeline_delivery",
    "take_photo",
    "pass_shoe",
    "place_food",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def wilson(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def paired_bootstrap(values: Sequence[float], draws: int = 100_000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(20260818)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 2000):
        count = min(2000, draws - start)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        means[start : start + count] = array[indices].mean(1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def load_task(root: Path, mode: str, task: str) -> dict[str, Any]:
    path = root / mode / f"{task}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("mode") != mode or value.get("task") != task or value.get("episodes") != 20:
        raise RuntimeError(f"incomplete CARE Validation20 artifact: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--offline-report", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--seed-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        print(json.dumps({"status": "PRESERVED", "output": str(args.output)}))
        return
    offline = json.loads(args.offline_report.read_text(encoding="utf-8"))
    task_results = {}
    paired_differences: list[float] = []
    total_care = total_off = total_overrides = total_steps = 0
    paired_wins = paired_losses = paired_ties = 0
    exact_no_override_pairs = exact_no_override_matches = 0
    care_checkpoint_sha = None
    for task in TASKS:
        care = load_task(args.validation_root, "care", task)
        off = load_task(args.validation_root, "selector_off", task)
        care_by_seed = {int(row["seed"]): row for row in care["rows"]}
        off_by_seed = {int(row["seed"]): row for row in off["rows"]}
        if set(care_by_seed) != set(off_by_seed) or len(care_by_seed) != 20:
            raise RuntimeError(f"CARE paired seeds differ: {task}")
        if care_checkpoint_sha is None:
            care_checkpoint_sha = care["care_checkpoint_sha256"]
        elif care_checkpoint_sha != care["care_checkpoint_sha256"]:
            raise RuntimeError("CARE checkpoint changed across tasks")
        task_differences = []
        task_overrides = 0
        task_steps = 0
        for seed in sorted(care_by_seed):
            care_row = care_by_seed[seed]
            off_row = off_by_seed[seed]
            difference = int(bool(care_row["success"])) - int(bool(off_row["success"]))
            paired_differences.append(difference)
            task_differences.append(difference)
            paired_wins += int(difference > 0)
            paired_losses += int(difference < 0)
            paired_ties += int(difference == 0)
            overrides = int(care_row["override_steps"])
            task_overrides += overrides
            task_steps += int(care_row["steps"])
            if overrides == 0:
                exact_no_override_pairs += 1
                exact_no_override_matches += int(
                    bool(care_row["success"]) == bool(off_row["success"])
                    and int(care_row["steps"]) == int(off_row["steps"])
                    and care_row["reference_action_trace_sha256"]
                    == off_row["reference_action_trace_sha256"]
                )
        care_successes = int(care["successes"])
        off_successes = int(off["successes"])
        total_care += care_successes
        total_off += off_successes
        total_overrides += task_overrides
        total_steps += task_steps
        task_results[task] = {
            "care_successes": care_successes,
            "selector_off_successes": off_successes,
            "paired_success_delta": int(sum(task_differences)),
            "care_wilson_95": wilson(care_successes, 20),
            "selector_off_wilson_95": wilson(off_successes, 20),
            "override_steps": task_overrides,
            "override_rate": task_overrides / max(task_steps, 1),
            "care_sha256": sha256_file(args.validation_root / "care" / f"{task}.json"),
            "selector_off_sha256": sha256_file(
                args.validation_root / "selector_off" / f"{task}.json"
            ),
        }
    paired_ci = paired_bootstrap(paired_differences)
    care_rate = total_care / 120
    off_rate = total_off / 120
    conditions = {
        "overall_no_material_regression": paired_ci[0] >= -0.02,
        "closed_loop_positive_point_delta": total_care > total_off,
        "selector_made_nonzero_interventions": total_overrides > 0,
        "offline_gate_c_passed": bool(offline["gate_c_passed"]),
        "stress_suite_completed": False,
        "unseen_teammate_beats_reactive_only": False,
    }
    gate_d_passed = all(conditions.values())
    if total_overrides == 0:
        conclusion = (
            "CARE 的三 seed 训练和每任务 20 局闭环已经跑完，但 90% 校准下界没有放行任何替代动作；"
            "因此它在本轮等价于冻结 B-core，不能证明 CARE 带来控制增益。"
        )
    elif total_care > total_off and paired_ci[0] > 0:
        conclusion = (
            "CARE 确实在新种子上把一部分 B-core 动作替换成了候选动作，且成对成功率区间支持正增益；"
            "但 Gate A 的历史失败和尚未完成的压力/未见队友实验仍限制 team-belief 强主张。"
        )
    elif total_care >= total_off:
        conclusion = (
            "CARE 会做少量保守替换，整体成功数没有低于 B-core，但 20 局/任务的证据还不能排除零收益；"
            "当前最多算可行性信号，不是论文 winner。"
        )
    else:
        conclusion = (
            "CARE 的替代动作在新种子闭环中净伤害了成功率；训练本身完成，但当前候选族和监督信号不值得继续包装。"
        )
    report = {
        "format_version": "before-we-act.a7r1-care-validation20-summary/1",
        "stage_id": "A7R1-CARE-OWNER-AUTHORIZED-VALIDATION20",
        "completed_at_utc": utc_now(),
        "episodes_per_task": 20,
        "episodes_total_per_mode": 120,
        "task_results": task_results,
        "aggregate": {
            "care_successes": total_care,
            "selector_off_successes": total_off,
            "care_success_rate": care_rate,
            "selector_off_success_rate": off_rate,
            "paired_success_rate_delta": care_rate - off_rate,
            "paired_bootstrap_95": paired_ci,
            "paired_wins": paired_wins,
            "paired_losses": paired_losses,
            "paired_ties": paired_ties,
            "override_steps": total_overrides,
            "override_rate": total_overrides / max(total_steps, 1),
            "no_override_exact_fallback_pairs": exact_no_override_pairs,
            "no_override_exact_fallback_matches": exact_no_override_matches,
        },
        "gate_d_conditions": conditions,
        "gate_d_passed": gate_d_passed,
        "gate_a_preserved_as_not_passed": True,
        "gate_b_complete": False,
        "response_decomposition_claim_allowed": False,
        "care_checkpoint_sha256": care_checkpoint_sha,
        "contract_sha256": sha256_file(args.contract),
        "offline_report_sha256": sha256_file(args.offline_report),
        "seed_receipt_sha256": sha256_file(args.seed_receipt),
        "human_conclusion_zh": conclusion,
    }
    atomic_json(args.output, report)
    print(json.dumps({"status": "A7R1_VALIDATION20_COMPLETED", **report["aggregate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
