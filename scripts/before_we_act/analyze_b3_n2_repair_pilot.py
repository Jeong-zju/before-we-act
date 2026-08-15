#!/usr/bin/env python3
"""Issue the fail-closed decision for the short discrete-belief repair pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    status = json.loads((args.pilot / "status.json").read_text(encoding="utf-8"))
    progress = [
        json.loads(line)
        for line in (args.pilot / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if status.get("status") != "PASSED_SMOKE" or "selected_validation" not in status:
        raise RuntimeError("repair pilot did not finish with its validation diagnostic")
    validation = status["selected_validation"]
    alignment = [float(row["teacher_alignment"]) for row in progress]
    kl_upper_bound = 8.867290
    kl_stable = bool(alignment) and all(
        math.isfinite(value) and value <= kl_upper_bound for value in alignment
    )

    future = [float(row["future_latent"]) for row in progress]
    window = min(5, max(1, len(future) // 4))
    initial_future = mean(future[:window])
    final_future = mean(future[-window:])
    future_improvement = (initial_future - final_future) / max(
        abs(initial_future), 1e-12
    )
    belief = validation["belief"]
    mutual_information = float(belief["categorical_mutual_information"])
    estimable = (
        math.isfinite(mutual_information)
        and mutual_information > 1e-4
        and future_improvement > 0.01
    )
    effective_rank = float(belief["effective_rank"])
    active_factors = int(belief["active_categorical_factors_mi_gt_0_01"])
    multidimensional = effective_rank > 4.0 and active_factors >= 4

    if not kl_stable:
        decision = "FAILED_KL_STABILITY"
        summary = "离散化后 KL 仍未被理论上界约束，不能继续训练。"
    elif not estimable:
        decision = "FAILED_BELIEF_ESTIMABILITY"
        summary = "KL 已稳定，但 belief 还没有从预测任务中学到可测信息，不能继续训练。"
    elif not multidimensional:
        decision = "NEEDS_ONE_STRUCTURAL_MIGRATION"
        summary = "KL 已稳定且 belief 已可估计，但团队状态仍未形成足够多的独立维度；下一步只能再引入一个结构思想。"
    else:
        decision = "PASSED_REPAIR_GATES_FORMAL_TRAINING_REQUIRES_OWNER_DECISION"
        summary = "KL、可估计性和多维性三关均通过；仍不自动启动长训练。"

    payload = {
        "format_version": "before-we-act.b3-n2-repair-pilot/1",
        "stage": contract["stage_id"],
        "status": decision,
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "pilot": {
            "seed": status["seed"],
            "updates": status["update"],
            "logged_points": len(progress),
        },
        "gates": {
            "kl_stable": {
                "passed": kl_stable,
                "maximum_observed": max(alignment, default=float("nan")),
                "theoretical_upper_bound": kl_upper_bound,
                "final": alignment[-1] if alignment else None,
            },
            "belief_estimable": {
                "passed": estimable,
                "future_loss_initial_window": initial_future,
                "future_loss_final_window": final_future,
                "future_loss_relative_improvement": future_improvement,
                "categorical_mutual_information": mutual_information,
            },
            "multidimensional_team_state": {
                "passed": multidimensional,
                "effective_rank": effective_rank,
                "active_categorical_factors_mi_gt_0_01": active_factors,
                "required_effective_rank_strictly_greater_than": 4.0,
                "required_active_factors": 4,
            },
        },
        "action_diagnostics": {
            "b_core_mse": validation["macro"]["b_core"],
            "b_shuffle_mse": validation["macro"]["b_shuffle"],
            "direct_reactive_mse": validation["macro"]["direct_reactive"],
            "b0h_mse": validation["macro"]["b0h"],
        },
        "human_summary": summary,
        "formal_training_started": False,
    }
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
