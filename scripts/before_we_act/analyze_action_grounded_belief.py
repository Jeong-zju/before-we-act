#!/usr/bin/env python3
"""Issue the terminal R1 receipt without crossing a failed prerequisite gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from before_we_act.temporal_history_data import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-grounded-contract", type=Path, required=True)
    parser.add_argument("--fair-conclusion", type=Path, required=True)
    parser.add_argument("--training-sufficiency-index", type=Path, required=True)
    parser.add_argument("--pilot-contract", type=Path, required=True)
    parser.add_argument("--pilot-conclusion", type=Path, required=True)
    parser.add_argument("--pilot-rollouts", type=Path, required=True)
    parser.add_argument("--pilot-diagnostic", type=Path, required=True)
    parser.add_argument("--oracle-conclusion", type=Path)
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


def main() -> None:
    args = parse_args()
    contract = json.loads(args.action_grounded_contract.read_text(encoding="utf-8"))
    fair = json.loads(args.fair_conclusion.read_text(encoding="utf-8"))
    sufficiency = json.loads(
        args.training_sufficiency_index.read_text(encoding="utf-8")
    )
    pilot_contract = json.loads(args.pilot_contract.read_text(encoding="utf-8"))
    pilot = json.loads(args.pilot_conclusion.read_text(encoding="utf-8"))
    diagnostic = json.loads(args.pilot_diagnostic.read_text(encoding="utf-8"))
    if contract.get("stage_id") != "B3-N1-R1-ACTION-GROUNDED-BELIEF":
        raise RuntimeError("wrong R1 parent contract")
    if sufficiency.get("contract_sha256") != sha256_file(args.action_grounded_contract):
        raise RuntimeError("R1-1 training sufficiency contract hash differs")
    if set(sufficiency.get("seeds", {})) != {"20260815", "20260816", "20260817"}:
        raise RuntimeError("R1-1 training sufficiency seed set differs")
    if pilot.get("contract_sha256") != sha256_file(args.pilot_contract):
        raise RuntimeError("R1-3 pilot contract hash differs")
    if pilot.get("rollouts_sha256") != sha256_file(args.pilot_rollouts):
        raise RuntimeError("R1-3 rollout hash differs")
    if int(pilot.get("rollouts", -1)) != int(pilot_contract["design"]["rollouts"]):
        raise RuntimeError("R1-3 rollout count differs")
    if diagnostic.get("rollouts_sha256") != sha256_file(args.pilot_rollouts):
        raise RuntimeError("R1-3 diagnostic rollout hash differs")
    if diagnostic.get("conclusion_sha256") != sha256_file(args.pilot_conclusion):
        raise RuntimeError("R1-3 diagnostic conclusion hash differs")
    oracle = None
    if fair.get("status") == "FAILED_R1_1_FAIR_PROBE":
        if args.oracle_conclusion is None:
            raise RuntimeError("failed fair probe requires the conditional R1-2 receipt")
        oracle = json.loads(args.oracle_conclusion.read_text(encoding="utf-8"))

    fair_sufficient = fair.get("status") in {
        "PASSED_R1_1_FAIR_PROBE",
        "FAILED_R1_1_FAIR_PROBE",
    }
    pilot_passed = pilot.get("status") == "PASSED_R1_3_COUNTERFACTUAL_PILOT"
    if not fair_sufficient:
        status = "INCONCLUSIVE_TRAINING_NOT_CONVERGED"
        human = (
            "公平探针至少一个 seed 到冻结上限仍未达到平台，所以离线表示增量不能定性；"
            "同时 720 条同状态分叉也没有给出可复现的任务回报差异。当前只能停在证据不足，不能训练教师、学生或进入 N2。"
        )
    elif not pilot_passed:
        status = "NO_EXPLICIT_TEAMMATE_AWARE_CORRECTION_NEEDED"
        human = (
            "离线 belief 是否更会拟合专家动作，不能替代因果判题。720 条同状态实验里，四种队友模式的 32 步累计回报全部是 0，"
            "6/6 任务的配对区间都没有非零信号，而且恢复重放也未做到逐组精确重复。按预注册门禁，当前任务和控制窗口没有证明需要显式的队友感知补救；"
            "R1-4 全知教师、R1-5 合法学生和 3-N2 均不得启动。"
        )
    else:
        raise RuntimeError("passed pilot requires R1-4/R1-5 receipts before final R1")

    result = {
        "format_version": "before-we-act.b3-n1-r1-final-conclusion/1",
        "stage": "B3-N1-R1-ACTION-GROUNDED-BELIEF",
        "status": status,
        "completed_at_utc": utc_now(),
        "evidence": {
            "r1_contract": str(args.action_grounded_contract.resolve()),
            "r1_contract_sha256": sha256_file(args.action_grounded_contract),
            "fair_conclusion": str(args.fair_conclusion.resolve()),
            "fair_conclusion_sha256": sha256_file(args.fair_conclusion),
            "training_sufficiency_index": str(
                args.training_sufficiency_index.resolve()
            ),
            "training_sufficiency_index_sha256": sha256_file(
                args.training_sufficiency_index
            ),
            "pilot_contract": str(args.pilot_contract.resolve()),
            "pilot_contract_sha256": sha256_file(args.pilot_contract),
            "pilot_conclusion": str(args.pilot_conclusion.resolve()),
            "pilot_conclusion_sha256": sha256_file(args.pilot_conclusion),
            "pilot_rollouts": str(args.pilot_rollouts.resolve()),
            "pilot_rollouts_sha256": sha256_file(args.pilot_rollouts),
            "pilot_diagnostic": str(args.pilot_diagnostic.resolve()),
            "pilot_diagnostic_sha256": sha256_file(args.pilot_diagnostic),
            "oracle_conclusion": (
                str(args.oracle_conclusion.resolve()) if args.oracle_conclusion else None
            ),
            "oracle_conclusion_sha256": (
                sha256_file(args.oracle_conclusion) if args.oracle_conclusion else None
            ),
        },
        "substage_status": {
            "r1_0": "PASSED",
            "r1_1": fair.get("status"),
            "r1_2": oracle.get("status") if oracle else "SKIPPED_BY_FAIR_RESULT",
            "r1_3": pilot.get("status"),
            "r1_4": "NOT_STARTED_FAILED_R1_3_PREREQUISITE",
            "r1_5": "NOT_STARTED_FAILED_R1_3_PREREQUISITE",
            "r1_6": status,
        },
        "counterfactual_gate": {
            "rollouts": int(pilot["rollouts"]),
            "positive_tasks": int(pilot["positive_tasks"]),
            "required_positive_tasks": int(pilot["required_positive_tasks"]),
            "same_mode_restore_repeats_exact": bool(
                pilot["same_mode_restore_repeats_exact"]
            ),
            "task_results": pilot["per_task"],
            "all_recorded_dense_rewards_zero": all(
                row["paired_reward_delta_normal_minus_intervention"] == 0.0
                and row["ci95"] == [0.0, 0.0]
                for row in pilot["per_task"].values()
            ),
            "repeat_groups": int(diagnostic["groups"]),
            "exact_repeat_groups": int(diagnostic["exact_repeat_groups"]),
            "nonexact_repeat_groups": int(diagnostic["nonexact_repeat_groups"]),
            "maximum_nonexact_displacement_range": float(
                diagnostic["maximum_nonexact_displacement_range"]
            ),
        },
        "measurement_limitations": [
            "没有可从任意 snapshot 恢复的可信补救动作专家，因此 pilot 只判团队结果/价值，没有伪造 ego 动作标签。",
            "32 步窗口内累计 dense reward 对所有 720 条分叉退化为 0；共享物体位移虽有差异，但它是次级诊断且恢复重复不稳定，不能事后替换预冻结主指标追正。",
            f"240 个状态×模式组中有 {int(diagnostic['nonexact_repeat_groups'])} 组没有达到冻结的精确重复条件，因此本轮不能把零差异解释成物理上的严格不变。",
        ],
        "r1_passed": False,
        "positive_action_relevant_belief_signal": False,
        "teacher_started": False,
        "student_started": False,
        "n2_authorized": False,
        "human_summary": human,
    }
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "status": status,
                "fair": fair.get("status"),
                "pilot": pilot.get("status"),
                "n2_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
