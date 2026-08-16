#!/usr/bin/env python3
"""Consolidate the three pre-registered 3-N1 seeds into one immutable receipt."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from before_we_act.raw_team_signal_data import CAPACITY_CANDIDATES
from before_we_act.temporal_history_data import SIX_TASKS
from before_we_act.train_raw_team_signal import atomic_json


SUFFICIENT = {"PLATFORM_REACHED", "SATURATED_BY_OVERFIT"}
CONTROLS = ("persistence", "zero", "shuffle_model")
ACTION_CONTROLS = ("hidden", "time", "row_shuffle", "phase_shuffle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def median(values) -> float:
    return float(statistics.median(values))


def relative_improvement(candidate: float, baseline: float) -> float:
    return (baseline - candidate) / max(abs(baseline), 1e-12) * 100


def main() -> None:
    args = parse_args()
    contract = load(args.run_root / "contract" / "n1_contract.json")
    seeds = contract["seeds"]
    representation_status = [load(args.run_root / "representation" / f"seed_{seed}" / "status.json") for seed in seeds]
    probe_status = [load(args.run_root / "probe" / f"seed_{seed}" / "status.json") for seed in seeds]
    sufficient = all(row["status"] in SUFFICIENT for row in (*representation_status, *probe_status))
    raw = [row.get("selected_validation", {}) for row in representation_status]
    action = [row.get("selected_validation", {}) for row in probe_status]
    if not sufficient:
        result = {
            "format_version": "before-we-act.b3-n1-conclusion/1",
            "status": "INCONCLUSIVE_TRAINING_NOT_CONVERGED",
            "representation_status": representation_status,
            "probe_status": probe_status,
            "human_summary": "至少一个 seed 的表示模型或动作探针还没有训练到平台，因此现在不能判断有信号还是没信号。",
            "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        atomic_json(args.output, result); return

    raw_median = {capacity: median([row["macro"]["real"][str(capacity)] for row in raw]) for capacity in CAPACITY_CANDIDATES}
    action_median = {capacity: median([row["macro"]["belief"][str(capacity)] for row in action]) for capacity in CAPACITY_CANDIDATES}
    best_raw = min(raw_median.values()); best_action = min(action_median.values())
    eligible = [capacity for capacity in CAPACITY_CANDIDATES if raw_median[capacity] <= 1.01 * best_raw and action_median[capacity] <= 1.01 * best_action]
    selected = min(eligible) if eligible else None
    if selected is None:
        status = "INCONCLUSIVE_CAPACITY_NOT_SATURATED"
        raw_positive = action_positive = False
        selected_key = None
    else:
        selected_key = str(selected)
        non_collapse = all(row["token_feature_std"][selected_key] > 0.05 for row in raw)
        raw_seed = all(
            row["macro"]["real"][selected_key] < row["macro"][control][selected_key]
            for row in raw for control in CONTROLS
        )
        raw_task_count = sum(
            all(
                median([row["per_task"]["real"][selected_key][str(task)] for row in raw])
                < median([row["per_task"][control][selected_key][str(task)] for row in raw])
                for control in CONTROLS
            )
            for task in range(6)
        )
        raw_anchor = all(
            all(
                median([row["per_anchor"]["real"][selected_key][anchor] for row in raw])
                < median([row["per_anchor"][control][selected_key][anchor] for row in raw])
                for control in CONTROLS
            )
            for anchor in range(4)
        )
        raw_positive = non_collapse and raw_seed and raw_task_count >= 4 and raw_anchor
        action_seed = all(
            row["macro"]["belief"][selected_key] < row["macro"][control][selected_key]
            for row in action for control in ACTION_CONTROLS
        )
        action_task_count = sum(
            median([row["per_task"]["belief"][selected_key][str(task)] for row in action])
            < median([row["per_task"]["hidden"][selected_key][str(task)] for row in action])
            for task in range(6)
        )
        action_positive = action_seed and action_task_count >= 4
        if raw_positive and action_positive:
            status = "POSITIVE_SIGNAL"
        elif raw_positive:
            status = "MODELABLE_NO_ACTION_VALUE"
        else:
            status = "NO_STABLE_RAW_SIGNAL"

    details = {"capacity_raw_median": raw_median, "capacity_action_median": action_median, "eligible_capacities": eligible, "selected_capacity": selected}
    if selected is not None:
        details.update(
            {
                "raw_macro_by_seed": [row["macro"]["real"][selected_key] for row in raw],
                "raw_control_improvement_percent_median": {control: relative_improvement(median([row["macro"]["real"][selected_key] for row in raw]), median([row["macro"][control][selected_key] for row in raw])) for control in CONTROLS},
                "action_macro_by_seed": [row["macro"]["belief"][selected_key] for row in action],
                "action_vs_control_improvement_percent_median": {control: relative_improvement(median([row["macro"]["belief"][selected_key] for row in action]), median([row["macro"][control][selected_key] for row in action])) for control in ACTION_CONTROLS},
                "action_vs_hidden_task_direction": {SIX_TASKS[task]: relative_improvement(median([row["per_task"]["belief"][selected_key][str(task)] for row in action]), median([row["per_task"]["hidden"][selected_key][str(task)] for row in action])) for task in range(6)},
                "token_feature_std_by_seed": [row["token_feature_std"][selected_key] for row in raw],
                "episode_identity_probe_by_seed": [row["episode_identity_probe"][selected_key] for row in action],
            }
        )
    if status == "POSITIVE_SIGNAL":
        human = "原始轨迹里的无人工标签团队信号不仅能预测队友和未来观测，也在三个 seed 的轻量动作探针中稳定胜过同容量普通历史 hidden；打乱和时间捷径不能复现，允许按冻结容量进入 3-N2。"
    elif status == "MODELABLE_NO_ACTION_VALUE":
        human = "模型确实能从原始轨迹预测队友和未来观测，但这些表示没有稳定胜过普通历史 hidden 的动作探针；说明信息可学，不等于对动作有新增价值，按路线应停在 3-N1。"
    elif status == "NO_STABLE_RAW_SIGNAL":
        human = "训练已经充分，但真实原始目标没有稳定胜过持久值、零值和打乱目标；当前这组无标签目标不能作为 3-N2 的入口。"
    else:
        human = "4/8/16 个 token 的原始预测和动作结果没有共同饱和点，不能在不追加搜索的情况下冻结容量，因此暂不进入 3-N2。"
    result = {
        "format_version": "before-we-act.b3-n1-conclusion/1",
        "status": status,
        "raw_positive": raw_positive,
        "action_positive": action_positive,
        "training_sufficient": sufficient,
        "details": details,
        "representation_status": representation_status,
        "probe_status": probe_status,
        "n2_activation": contract["n2_capacity_mapping_if_positive"].get(str(selected)) if status == "POSITIVE_SIGNAL" else None,
        "human_summary": human,
        "claim_boundary": "3-N1 is an offline raw-signal/action-probe result, not closed-loop B-core acceptance.",
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    atomic_json(args.output, result)
    print(json.dumps({"status": status, "selected_capacity": selected}, sort_keys=True))


if __name__ == "__main__":
    main()
