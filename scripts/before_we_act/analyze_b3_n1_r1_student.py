#!/usr/bin/env python3
"""Open sealed splits and classify the offline portion of the R1-5 gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import torch

from before_we_act.b3_n1_r1 import (
    FrozenR1Backbones,
    R1OracleDataset,
    R1_SEEDS,
    load_split,
    split_by_episode_key,
)
from before_we_act.b3_n1_r1_teacher_student import (
    DirectReactiveControl,
    LegalBeliefStudent,
)
from before_we_act.step2_temporal_data import SIX_TASKS, sha256_file
from before_we_act.train_b3_n1_r1_student import evaluate, load_teacher
from before_we_act.train_b3_n1_r1_teacher import fixed_loader, load_base_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--parent-contract", type=Path, required=True)
    parser.add_argument("--student-contract", type=Path, required=True)
    parser.add_argument("--scenario-split", type=Path, required=True)
    parser.add_argument("--fair-run-root", type=Path, required=True)
    parser.add_argument("--teacher-run-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def summarize(metrics: dict) -> dict:
    per_task: dict[str, dict] = {}
    for task_index, task in enumerate(SIX_TASKS):
        row = {
            name: float(metrics["per_task"][name][str(task_index)])
            for name in metrics["per_task"]
        }
        row["student_relative_improvement_vs_h"] = (
            row["h"] - row["h_student"]
        ) / max(abs(row["h"]), 1e-12)
        per_task[task] = row
    return {
        "macro": metrics["macro"],
        "student_relative_improvement_vs_h": (
            float(metrics["macro"]["h"])
            - float(metrics["macro"]["h_student"])
        )
        / max(abs(float(metrics["macro"]["h"])), 1e-12),
        "per_task": per_task,
        "belief_off_max_abs": float(metrics["belief_off_max_abs"]),
        "teacher_token_gaussian_nll": float(
            metrics["teacher_token_gaussian_nll"]
        ),
        "teammate_action_gaussian_nll": float(
            metrics["teammate_action_gaussian_nll"]
        ),
        "teammate_delta_mse": float(metrics["teammate_delta_mse"]),
        "future_visual_mse": float(metrics["future_visual_mse"]),
        "rows": int(metrics["rows"]),
    }


def main() -> None:
    args = parse_args()
    parent = json.loads(args.parent_contract.read_text(encoding="utf-8"))
    contract = json.loads(args.student_contract.read_text(encoding="utf-8"))
    if contract.get("format_version") != "before-we-act.b3-n1-r1-student-contract/2":
        raise RuntimeError("student analysis requires the owner-revised contract")
    sealed_test_allowed = bool(
        contract.get("evidence_scope", {}).get("sealed_test_allowed", False)
    )
    split_names = ("validation", "test") if sealed_test_allowed else ("validation",)
    split_payload = load_split(args.scenario_split)
    split = split_by_episode_key(split_payload)
    statuses = {
        str(seed): json.loads(
            (
                args.run_root
                / "r1_5_student"
                / f"seed_{seed}"
                / "status.json"
            ).read_text(encoding="utf-8")
        )
        for seed in R1_SEEDS
    }
    sufficient = all(row.get("status") == "PLATFORM_REACHED" for row in statuses.values())
    receipts_valid = all(
        sha256_file(
            args.run_root
            / "r1_5_student"
            / f"seed_{seed}"
            / "training_sufficiency.json"
        )
        == statuses[str(seed)].get("training_sufficiency_sha256")
        for seed in R1_SEEDS
    )
    if not sufficient or not receipts_valid:
        result = {
            "format_version": "before-we-act.b3-n1-r1-student-conclusion/2",
            "stage": "R1-5-DEPLOYMENT-LEGAL-STUDENT-OWNER-REVISION",
            "status": "INCONCLUSIVE_TRAINING_NOT_CONVERGED",
            "completed_at_utc": utc_now(),
            "training_status": statuses,
            "training_sufficiency_receipts_valid": receipts_valid,
            "test_opened": False,
            "n2_authorized": False,
            "human_summary": "合法学生或直接对照至少一组没有跑完四阶段并达到冻结平台，证据不足，不能判通过或失败。",
        }
        atomic_json(args.output, result)
        print(json.dumps({"status": result["status"]}, sort_keys=True))
        return

    dataset = R1OracleDataset(args.cache)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    evaluation: dict[str, dict] = {}
    for seed in R1_SEEDS:
        seed_root = args.run_root / "r1_5_student" / f"seed_{seed}"
        selected = int(statuses[str(seed)]["selected_update"])
        checkpoint = seed_root / f"checkpoint_{selected:06d}.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        student = LegalBeliefStudent().to(device)
        direct = DirectReactiveControl().to(device)
        student.load_state_dict(payload["student"], strict=True)
        direct.load_state_dict(payload["direct"], strict=True)
        teacher, _, teacher_sha = load_teacher(args.teacher_run_root, seed, device)
        base_probe, _, base_sha = load_base_probe(args.fair_run_root, seed, device)
        n1 = Path(
            parent["old_n1_read_only"]["representation_checkpoints"][str(seed)][
                "path"
            ]
        )
        backbones = FrozenR1Backbones(
            b0h_checkpoint=Path(parent["b0h"]["checkpoint"]),
            n1_checkpoint=n1,
            visual_mean=dataset.visual_mean,
            visual_std=dataset.visual_std,
        ).to(device)
        evaluation[str(seed)] = {
            "selected_update": selected,
            "checkpoint_sha256": sha256_file(checkpoint),
            "teacher_checkpoint_sha256": teacher_sha,
            "base_h_checkpoint_sha256": base_sha,
        }
        for split_name in split_names:
            evaluation[str(seed)][split_name] = summarize(
                evaluate(
                    backbones,
                    base_probe,
                    teacher,
                    student,
                    direct,
                    fixed_loader(dataset, split, split_name),
                    device,
                )
            )
        del student, direct, teacher, base_probe, backbones
        torch.cuda.empty_cache()

    every_seed_student = all(
        evaluation[str(seed)][name]["macro"]["h_student"]
        < evaluation[str(seed)][name]["macro"]["h"]
        for seed in R1_SEEDS
        for name in split_names
    )
    every_seed_teacher = all(
        evaluation[str(seed)][name]["macro"]["h_teacher"]
        < evaluation[str(seed)][name]["macro"]["h"]
        for seed in R1_SEEDS
        for name in split_names
    )
    shuffle_clean = all(
        evaluation[str(seed)][name]["macro"]["h_student"]
        < evaluation[str(seed)][name]["macro"]["h_student_shuffle"]
        for seed in R1_SEEDS
        for name in split_names
    )
    direct_clean = all(
        evaluation[str(seed)][name]["macro"]["h_student"]
        < evaluation[str(seed)][name]["macro"]["direct_reactive"]
        for seed in R1_SEEDS
        for name in split_names
    )
    belief_off_exact = all(
        evaluation[str(seed)][name]["belief_off_max_abs"] == 0.0
        and evaluation[str(seed)][name]["macro"]["belief_off"]
        == evaluation[str(seed)][name]["macro"]["h"]
        for seed in R1_SEEDS
        for name in split_names
    )
    task_medians: dict[str, dict] = {}
    positive_tasks = 0
    for task in SIX_TASKS:
        validation = float(
            np.median(
                [
                    evaluation[str(seed)]["validation"]["per_task"][task][
                        "student_relative_improvement_vs_h"
                    ]
                    for seed in R1_SEEDS
                ]
            )
        )
        test = (
            float(
                np.median(
                    [
                        evaluation[str(seed)]["test"]["per_task"][task][
                            "student_relative_improvement_vs_h"
                        ]
                        for seed in R1_SEEDS
                    ]
                )
            )
            if sealed_test_allowed
            else None
        )
        positive = validation > 0 and (test is None or test > 0)
        positive_tasks += int(positive)
        task_medians[task] = {
            "validation_relative_median": validation,
            "test_relative_median": test,
            "positive_on_all_opened_splits": positive,
        }
    passed = (
        every_seed_student
        and every_seed_teacher
        and shuffle_clean
        and direct_clean
        and belief_off_exact
        and positive_tasks >= 4
    )
    if passed:
        status = (
            "EXPLORATORY_DEPLOYMENT_LEGAL_STUDENT_OFFLINE_VALUE_CONFIRMED"
            if sealed_test_allowed
            else "EXPLORATORY_DEPLOYMENT_LEGAL_STUDENT_VALIDATION_SIGNAL_CONFIRMED"
        )
        human = (
            "只看合法 16 步历史的学生，在冻结验证集上稳定优于 H，且打乱 belief 和同动作容量的直接模型都复现不了。"
            "教师没有训练到平台，所以密封测试仍未打开；这只是明确的探索信号，不是正式通过，也不证明闭环因果效果。"
        )
    elif every_seed_student and not direct_clean:
        status = "EXPLORATORY_DIRECT_REACTIVE_CONTROL_EXPLAINS_GAIN"
        human = "学生的离线动作误差有改善，但同样动作路径的直接 reactive 模型能做到一样好或更好，不能把收益归因给显式 belief。"
    else:
        status = "EXPLORATORY_LEGAL_HISTORY_CANNOT_RECOVER_ACTION_VALUE"
        human = "全知信息有用，但只看合法 16 步历史的学生没能跨 seed、跨场景稳定恢复这份动作价值。"
    result = {
        "format_version": "before-we-act.b3-n1-r1-student-conclusion/2",
        "stage": "R1-5-DEPLOYMENT-LEGAL-STUDENT-OWNER-REVISION",
        "status": status,
        "completed_at_utc": utc_now(),
        "contract_sha256": sha256_file(args.student_contract),
        "training_status": statuses,
        "test_opened": sealed_test_allowed,
        "evaluation": evaluation,
        "task_medians": task_medians,
        "gate": {
            "student_beats_h_every_seed_all_opened_splits": every_seed_student,
            "teacher_beats_h_every_seed_all_opened_splits": every_seed_teacher,
            "positive_tasks": positive_tasks,
            "student_shuffle_control_clean": shuffle_clean,
            "student_beats_direct_reactive_every_seed": direct_clean,
            "belief_off_exact_h": belief_off_exact,
            "offline_passed": passed,
            "causal_online_deferred_to_paper_final": True,
        },
        "evidence_scope": (
            "offline exploratory student validation only; sealed test unopened; "
            "R1-3 deferred and not used"
        ),
        "r1_3_used": False,
        "n2_authorized": False,
        "human_summary": human,
    }
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "status": status,
                "positive_tasks": positive_tasks,
                "direct_clean": direct_clean,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
