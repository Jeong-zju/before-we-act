#!/usr/bin/env python3
"""Freeze the owner's explicit continuation from teacher validation to R1-5."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from before_we_act.action_grounded_belief import BELIEF_SEEDS
from before_we_act.temporal_history_data import SIX_TASKS, sha256_file


AUTHORIZED = "AUTHORIZED_R1_5_EXPLORATORY_VALIDATION_ONLY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-revision", type=Path, required=True)
    parser.add_argument("--teacher-conclusion", type=Path, required=True)
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


def validation_gate(teacher: dict) -> dict:
    rows: dict[str, dict] = {}
    task_improvements: dict[str, list[float]] = {task: [] for task in SIX_TASKS}
    for seed in BELIEF_SEEDS:
        status = teacher["training_status"][str(seed)]
        metrics = status["selected_validation"]
        macro = metrics["macro"]
        h = float(macro["h"])
        value = float(macro["h_teacher"])
        rows[str(seed)] = {
            "selected_update": int(status["selected_update"]),
            "training_status": status["status"],
            "h": h,
            "h_teacher": value,
            "h_teacher_shuffle": float(macro["h_teacher_shuffle"]),
            "h_matched_capacity": float(macro["h_matched_capacity"]),
            "relative_improvement_vs_h": (h - value) / max(abs(h), 1e-12),
            "beats_h_shuffle_and_matched": (
                value < h
                and value < float(macro["h_teacher_shuffle"])
                and value < float(macro["h_matched_capacity"])
            ),
        }
        for index, task in enumerate(SIX_TASKS):
            base = float(metrics["per_task"]["h"][str(index)])
            task_value = float(metrics["per_task"]["h_teacher"][str(index)])
            task_improvements[task].append(
                (base - task_value) / max(abs(base), 1e-12)
            )
    task_medians = {
        task: float(np.median(values)) for task, values in task_improvements.items()
    }
    positive_tasks = sum(value > 0 for value in task_medians.values())
    all_controls_clean = all(
        row["beats_h_shuffle_and_matched"] for row in rows.values()
    )
    all_terminal = all(
        row["training_status"]
        in {
            "PLATFORM_REACHED",
            "SATURATED_BY_OVERFIT",
            "INCONCLUSIVE_TRAINING_NOT_CONVERGED",
        }
        for row in rows.values()
    )
    return {
        "per_seed": rows,
        "task_relative_improvement_medians": task_medians,
        "positive_tasks": positive_tasks,
        "all_seeds_beat_h_shuffle_and_matched": all_controls_clean,
        "all_teacher_runs_terminal": all_terminal,
        "passed_for_exploratory_student_validation": (
            all_terminal and all_controls_clean and positive_tasks >= 4
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("student continuation receipt is already frozen")
    owner = json.loads(args.owner_revision.read_text(encoding="utf-8"))
    teacher = json.loads(args.teacher_conclusion.read_text(encoding="utf-8"))
    if owner.get("status") != "AUTHORIZED_R1_4_R1_5_EXPLORATORY_TEST":
        raise RuntimeError("student continuation lacks the owner's R1 revision")
    teacher_status = teacher.get("status")
    if teacher_status not in {
        "EXPLORATORY_OMNISCIENT_TEACHER_ACTION_VALUE_CONFIRMED",
        "INCONCLUSIVE_TRAINING_NOT_CONVERGED",
    }:
        raise RuntimeError("teacher result does not support a student continuation")
    gate = validation_gate(teacher)
    authorized = bool(gate["passed_for_exploratory_student_validation"])
    result = {
        "format_version": "before-we-act.b3-n1-r1-student-continuation/1",
        "stage": "B3-N1-R1-OWNER-STUDENT-CONTINUATION",
        "status": (
            AUTHORIZED
            if authorized
            else "NOT_AUTHORIZED_TEACHER_VALIDATION_SIGNAL_WEAK"
        ),
        "created_at_utc": utc_now(),
        "decision_source": "project owner instruction given before R1-4 execution",
        "owner_revision": str(args.owner_revision.resolve()),
        "owner_revision_sha256": sha256_file(args.owner_revision),
        "teacher_conclusion": str(args.teacher_conclusion.resolve()),
        "teacher_conclusion_sha256": sha256_file(args.teacher_conclusion),
        "teacher_formal_status_preserved": teacher_status,
        "teacher_test_opened": bool(teacher.get("test_opened", False)),
        "validation_gate": gate,
        "student_scope": "exploratory validation only",
        "student_sealed_test_allowed": False,
        "r1_3_valid_causal_measurement": "deferred to paper-final experiments",
        "n2_authorized": False,
        "forbidden_claims": [
            "the teacher formally converged or passed",
            "validation-only student results are sealed-test results",
            "offline action prediction proves closed-loop causal correction",
            "this continuation authorizes 3-N2",
        ],
        "human_summary": (
            "负责人在教师实验开始前已经明确允许继续测试学生。教师三颗种子虽然没有达到训练平台，"
            "但冻结验证集上的方向足够一致，因此允许学生继续做探索性验证；教师和学生的密封测试"
            "仍不打开，所有未收敛标签继续保留。"
            if authorized
            else "教师冻结验证集信号没有达到预先写明的跨种子和跨任务要求，学生不启动。"
        ),
    }
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
