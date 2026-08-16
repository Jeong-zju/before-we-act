#!/usr/bin/env python3
"""Summarize the unopened R1-5 validation trend without changing its formal status."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from before_we_act.action_grounded_belief import BELIEF_SEEDS
from before_we_act.temporal_history_data import SIX_TASKS, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def summarize(statuses: dict[str, dict]) -> dict:
    rows: dict[str, dict] = {}
    task_values: dict[str, list[float]] = {task: [] for task in SIX_TASKS}
    for seed in BELIEF_SEEDS:
        status = statuses[str(seed)]
        validation = status["selected_validation"]
        macro = validation["macro"]
        h = float(macro["h"])
        student = float(macro["h_student"])
        row = {
            "training_status": status["status"],
            "selected_update": int(status["selected_update"]),
            "h": h,
            "h_student": student,
            "h_student_shuffle": float(macro["h_student_shuffle"]),
            "direct_reactive": float(macro["direct_reactive"]),
            "relative_improvement_vs_h": (h - student) / max(abs(h), 1e-12),
            "student_beats_h": student < h,
            "student_beats_shuffle": student < float(macro["h_student_shuffle"]),
            "student_beats_direct": student < float(macro["direct_reactive"]),
            "belief_off_exact_h": (
                float(validation["belief_off_max_abs"]) == 0.0
                and float(macro["belief_off"]) == h
            ),
        }
        rows[str(seed)] = row
        for index, task in enumerate(SIX_TASKS):
            task_h = float(validation["per_task"]["h"][str(index)])
            task_student = float(
                validation["per_task"]["h_student"][str(index)]
            )
            task_values[task].append(
                (task_h - task_student) / max(abs(task_h), 1e-12)
            )
    medians = {
        task: float(np.median(values)) for task, values in task_values.items()
    }
    return {
        "per_seed": rows,
        "task_relative_improvement_medians": medians,
        "positive_tasks": sum(value > 0 for value in medians.values()),
        "student_beats_h_all_seeds": all(row["student_beats_h"] for row in rows.values()),
        "student_beats_shuffle_all_seeds": all(
            row["student_beats_shuffle"] for row in rows.values()
        ),
        "student_beats_direct_seed_count": sum(
            row["student_beats_direct"] for row in rows.values()
        ),
        "student_beats_direct_all_seeds": all(
            row["student_beats_direct"] for row in rows.values()
        ),
        "belief_off_exact_h_all_seeds": all(
            row["belief_off_exact_h"] for row in rows.values()
        ),
        "all_training_platform_reached": all(
            row["training_status"] == "PLATFORM_REACHED" for row in rows.values()
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("student validation diagnostic is already frozen")
    conclusion_path = args.run_root / "r1_5_student" / "conclusion.json"
    conclusion = json.loads(conclusion_path.read_text(encoding="utf-8"))
    if conclusion.get("status") != "INCONCLUSIVE_TRAINING_NOT_CONVERGED":
        raise RuntimeError("this diagnostic is only for an inconclusive R1-5 run")
    if bool(conclusion.get("test_opened", False)):
        raise RuntimeError("validation-only diagnostic requires the sealed test to remain closed")
    statuses = {
        str(seed): json.loads(
            (
                args.run_root
                / "r1_5_student"
                / f"seed_{seed}"
                / "status.json"
            ).read_text(encoding="utf-8")
        )
        for seed in BELIEF_SEEDS
    }
    trend = summarize(statuses)
    strong = (
        trend["student_beats_h_all_seeds"]
        and trend["student_beats_shuffle_all_seeds"]
        and trend["belief_off_exact_h_all_seeds"]
        and trend["positive_tasks"] >= 4
    )
    result = {
        "format_version": "before-we-act.b3-n1-r1-student-validation-diagnostic/1",
        "stage": "R1-5-DEPLOYMENT-LEGAL-STUDENT-VALIDATION-DIAGNOSTIC",
        "status": (
            "STRONG_POSITIVE_VALIDATION_TREND_BUT_NOT_CONVERGED_AND_DIRECT_CONTROL_UNRESOLVED"
            if strong and not trend["student_beats_direct_all_seeds"]
            else "INCONCLUSIVE_VALIDATION_TREND"
        ),
        "completed_at_utc": utc_now(),
        "formal_conclusion": str(conclusion_path.resolve()),
        "formal_conclusion_sha256": sha256_file(conclusion_path),
        "formal_status_preserved": conclusion["status"],
        "test_opened": False,
        "trend": trend,
        "n2_authorized": False,
        "human_summary": (
            "合法学生在三颗种子、六个任务的冻结验证集上都优于 H，打乱 belief 后收益消失，"
            "关闭 belief 可精确退回 H；但训练没有达到平台，而且同容量直接网络在一颗种子上更好。"
            "因此只能说学生学到了明确的队友相关动作信号，不能说显式 belief 的独立必要性已经证明。"
        ),
    }
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
