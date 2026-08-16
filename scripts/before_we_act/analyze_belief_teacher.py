#!/usr/bin/env python3
"""Evaluate and classify the owner-revised R1-4 offline teacher test."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import torch

from before_we_act.action_grounded_belief import (
    FrozenBeliefBackbones,
    ActionGroundedDataset,
    BELIEF_SEEDS,
    load_split,
    split_by_episode_key,
)
from before_we_act.belief_distillation import PrivilegedBeliefTeacher
from before_we_act.temporal_history_data import SIX_TASKS, sha256_file
from before_we_act.train_belief_teacher import (
    evaluate_teacher,
    fixed_loader,
    load_base_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--parent-contract", type=Path, required=True)
    parser.add_argument("--teacher-contract", type=Path, required=True)
    parser.add_argument("--scenario-split", type=Path, required=True)
    parser.add_argument("--fair-run-root", type=Path, required=True)
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
    h = float(metrics["macro"]["h"])
    teacher = float(metrics["macro"]["h_teacher"])
    per_task: dict[str, dict] = {}
    for index, task in enumerate(SIX_TASKS):
        base = float(metrics["per_task"]["h"][str(index)])
        value = float(metrics["per_task"]["h_teacher"][str(index)])
        per_task[task] = {
            "h": base,
            "h_teacher": value,
            "relative_improvement": (base - value) / max(abs(base), 1e-12),
            "absolute_h_minus_teacher": base - value,
        }
    return {
        "macro": metrics["macro"],
        "relative_improvement": (h - teacher) / max(abs(h), 1e-12),
        "absolute_h_minus_teacher": h - teacher,
        "per_task": per_task,
        "teammate_action_nll": metrics["teammate_action_nll"],
        "teammate_delta_mse": metrics["teammate_delta_mse"],
        "rows": metrics["rows"],
    }


def main() -> None:
    args = parse_args()
    parent = json.loads(args.parent_contract.read_text(encoding="utf-8"))
    contract = json.loads(args.teacher_contract.read_text(encoding="utf-8"))
    if contract.get("format_version") != "before-we-act.b3-n1-r1-teacher-contract/2":
        raise RuntimeError("teacher analysis requires the owner-revised contract")
    split_payload = load_split(args.scenario_split)
    split = split_by_episode_key(split_payload)
    statuses = {
        str(seed): json.loads(
            (args.run_root / "r1_4_teacher" / f"seed_{seed}" / "status.json").read_text(
                encoding="utf-8"
            )
        )
        for seed in BELIEF_SEEDS
    }
    sufficient = all(
        row["status"] in {"PLATFORM_REACHED", "SATURATED_BY_OVERFIT"}
        for row in statuses.values()
    )
    if not sufficient:
        result = {
            "format_version": "before-we-act.b3-n1-r1-teacher-conclusion/2",
            "stage": "R1-4-OMNISCIENT-TEACHER-OWNER-REVISION",
            "status": "INCONCLUSIVE_TRAINING_NOT_CONVERGED",
            "training_status": statuses,
            "completed_at_utc": utc_now(),
            "test_opened": False,
            "r1_3_used": False,
            "n2_authorized": False,
            "human_summary": "全知教师至少一个 seed 还没训练到平台，暂时不能判断特权协作信息是否有离线动作价值。",
        }
        atomic_json(args.output, result)
        print(json.dumps({"status": result["status"]}, sort_keys=True))
        return

    dataset = ActionGroundedDataset(args.cache)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    evaluation: dict[str, dict] = {}
    for seed in BELIEF_SEEDS:
        selected = int(statuses[str(seed)]["selected_update"])
        checkpoint = (
            args.run_root
            / "r1_4_teacher"
            / f"seed_{seed}"
            / f"checkpoint_{selected:06d}.pt"
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        teacher = PrivilegedBeliefTeacher().to(device)
        matched = PrivilegedBeliefTeacher().to(device)
        teacher.load_state_dict(payload["teacher"], strict=True)
        matched.load_state_dict(payload["matched"], strict=True)
        base_probe, _, _ = load_base_probe(args.fair_run_root, seed, device)
        signal_checkpoint = Path(parent["old_n1_read_only"]["representation_checkpoints"][str(seed)]["path"])
        backbones = FrozenBeliefBackbones(
            temporal_checkpoint=Path(parent["b0h"]["checkpoint"]),
            signal_checkpoint=signal_checkpoint,
            visual_mean=dataset.visual_mean,
            visual_std=dataset.visual_std,
        ).to(device)
        evaluation[str(seed)] = {
            "selected_update": selected,
            "checkpoint_sha256": sha256_file(checkpoint),
        }
        for name in ("validation", "test"):
            evaluation[str(seed)][name] = summarize(
                evaluate_teacher(
                    backbones,
                    base_probe,
                    teacher,
                    matched,
                    fixed_loader(dataset, split, name),
                    device,
                )
            )
        del teacher, matched, base_probe, backbones
        torch.cuda.empty_cache()

    every_seed = all(
        evaluation[str(seed)][name]["absolute_h_minus_teacher"] > 0
        for seed in BELIEF_SEEDS
        for name in ("validation", "test")
    )
    controls = all(
        evaluation[str(seed)][name]["macro"]["h_teacher"]
        < evaluation[str(seed)][name]["macro"][control]
        for seed in BELIEF_SEEDS
        for name in ("validation", "test")
        for control in ("h_teacher_shuffle", "h_matched_capacity")
    )
    task_medians: dict[str, dict] = {}
    positive_tasks = 0
    for task in SIX_TASKS:
        validation = float(
            np.median(
                [
                    evaluation[str(seed)]["validation"]["per_task"][task]["relative_improvement"]
                    for seed in BELIEF_SEEDS
                ]
            )
        )
        test = float(
            np.median(
                [
                    evaluation[str(seed)]["test"]["per_task"][task]["relative_improvement"]
                    for seed in BELIEF_SEEDS
                ]
            )
        )
        positive = validation > 0 and test > 0
        positive_tasks += int(positive)
        task_medians[task] = {
            "validation_relative_median": validation,
            "test_relative_median": test,
            "positive_both": positive,
        }
    passed = every_seed and controls and positive_tasks >= 4
    status = (
        "EXPLORATORY_OMNISCIENT_TEACHER_ACTION_VALUE_CONFIRMED"
        if passed
        else "EXPLORATORY_PRIVILEGED_TEACHER_HAS_NO_ACTION_VALUE"
    )
    result = {
        "format_version": "before-we-act.b3-n1-r1-teacher-conclusion/2",
        "stage": "R1-4-OMNISCIENT-TEACHER-OWNER-REVISION",
        "status": status,
        "completed_at_utc": utc_now(),
        "contract_sha256": sha256_file(args.teacher_contract),
        "training_status": statuses,
        "test_opened": True,
        "evaluation": evaluation,
        "task_medians": task_medians,
        "gate": {
            "teacher_beats_h_every_seed_both_splits": every_seed,
            "positive_tasks": positive_tasks,
            "shuffle_and_matched_controls_clean": controls,
            "passed": passed,
        },
        "evidence_scope": "offline exploratory teacher test; R1-3 deferred and not used",
        "r1_3_used": False,
        "n2_authorized": False,
        "human_summary": (
            "训练期全知教师在未见场景上稳定改善了 ego 动作预测，说明真实队友状态和动作包含额外离线动作信息；下一步测试只看合法历史的学生能否恢复这份信息。"
            if passed
            else "即使训练时提供真实队友状态和动作，教师也没有得到跨 seed、跨场景的稳定离线动作增量。"
        ),
    }
    atomic_json(args.output, result)
    print(
        json.dumps(
            {"status": status, "positive_tasks": positive_tasks, "controls_clean": controls},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
