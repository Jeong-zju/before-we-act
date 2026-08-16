#!/usr/bin/env python3
"""Freeze the owner's R1 ordering revision without rewriting prior evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from before_we_act.temporal_history_data import sha256_file
from prepare_belief_teacher import OWNER_STATUS, r1_1_strong_validation_trend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-contract", type=Path, required=True)
    parser.add_argument("--fair-conclusion", type=Path, required=True)
    parser.add_argument("--pilot-conclusion", type=Path, required=True)
    parser.add_argument("--pilot-diagnostic", type=Path, required=True)
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
    if args.output.exists():
        raise FileExistsError("owner revision is already frozen")
    parent = json.loads(args.parent_contract.read_text(encoding="utf-8"))
    fair = json.loads(args.fair_conclusion.read_text(encoding="utf-8"))
    pilot = json.loads(args.pilot_conclusion.read_text(encoding="utf-8"))
    diagnostic = json.loads(args.pilot_diagnostic.read_text(encoding="utf-8"))
    if parent.get("stage_id") != "B3-N1-R1-ACTION-GROUNDED-BELIEF":
        raise RuntimeError("wrong parent R1 contract")
    trend = r1_1_strong_validation_trend(fair)
    if not bool(diagnostic.get("rewards", {}).get("all_zero")):
        raise RuntimeError("owner revision expects the preserved all-zero R1-3 diagnosis")
    if pilot.get("status") != "FAILED_R1_3_COUNTERFACTUAL_PILOT":
        raise RuntimeError("owner revision expects the preserved failed R1-3 receipt")
    result = {
        "format_version": "before-we-act.b3-n1-r1-owner-revision/1",
        "stage": "B3-N1-R1-OWNER-ORDERING-REVISION",
        "status": OWNER_STATUS,
        "created_at_utc": utc_now(),
        "decision_source": "project owner",
        "parent_contract": str(args.parent_contract.resolve()),
        "parent_contract_sha256": sha256_file(args.parent_contract),
        "fair_conclusion": str(args.fair_conclusion.resolve()),
        "fair_conclusion_sha256": sha256_file(args.fair_conclusion),
        "pilot_conclusion": str(args.pilot_conclusion.resolve()),
        "pilot_conclusion_sha256": sha256_file(args.pilot_conclusion),
        "pilot_diagnostic": str(args.pilot_diagnostic.resolve()),
        "pilot_diagnostic_sha256": sha256_file(args.pilot_diagnostic),
        "r1_1_evidence": trend,
        "preserved_facts": {
            "r1_1_formal_status": fair["status"],
            "r1_1_test_opened": bool(fair.get("test_opened", False)),
            "r1_3_original_status": pilot["status"],
            "r1_3_dense_reward_all_zero": True,
            "r1_3_repeat_groups_exact": int(diagnostic["exact_repeat_groups"]),
            "r1_3_repeat_groups_total": int(diagnostic["groups"]),
        },
        "owner_assumption": (
            "In the six cooperative tasks, materially changing a teammate's future actions "
            "while keeping ego open-loop actions fixed is assumed to materially affect task "
            "success; a valid causal measurement is deferred to paper-final experiments."
        ),
        "ordering_change": {
            "r1_3": "DEFERRED_TO_PAPER_FINAL_VALID_MEASUREMENT",
            "r1_4": "AUTHORIZED_EXPLORATORY_OFFLINE_TEST",
            "r1_5": "AUTHORIZED_ONLY_IF_R1_4_EXPLORATORY_GATE_PASSES",
        },
        "forbidden_claims": [
            "R1-1 formally converged or passed",
            "R1-3 established a causal teammate effect",
            "teacher or student offline success alone proves closed-loop causal correction",
            "this revision alone authorizes 3-N2",
        ],
        "n2_authorized": False,
        "human_summary": (
            "负责人接受 R1-1 三个 seed 上约 40% 的一致验证改善，允许先测试全知教师和合法学生。"
            "旧 R1-1 未收敛和旧 R1-3 判题无效的事实保持不变；有效反事实闭环实验延后到论文定稿阶段。"
        ),
    }
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
