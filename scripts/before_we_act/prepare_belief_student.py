#!/usr/bin/env python3
"""Freeze the four-phase R1-5 legal-student experiment contract."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from before_we_act.action_grounded_belief import BELIEF_DATA_SEED, BELIEF_EVAL_EVERY, BELIEF_SEEDS
from before_we_act.temporal_history_data import sha256_file


PHASES = (
    ("posterior_alignment", 1, 20_000),
    ("teammate_action_and_state_change", 20_001, 40_000),
    ("frozen_belief_action_residual", 40_001, 60_000),
    ("low_lr_last_layer_action_correction", 60_001, 80_000),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-contract", type=Path, required=True)
    parser.add_argument("--teacher-contract", type=Path, required=True)
    parser.add_argument("--teacher-conclusion", type=Path, required=True)
    parser.add_argument("--fair-conclusion", type=Path, required=True)
    parser.add_argument("--owner-revision", type=Path, required=True)
    parser.add_argument("--student-continuation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("R1-5 student contract is already frozen")
    teacher = json.loads(args.teacher_conclusion.read_text(encoding="utf-8"))
    revision = json.loads(args.owner_revision.read_text(encoding="utf-8"))
    continuation = json.loads(args.student_continuation.read_text(encoding="utf-8"))
    if continuation.get("status") != "AUTHORIZED_R1_5_EXPLORATORY_VALIDATION_ONLY":
        raise RuntimeError("R1-5 lacks an explicit exploratory continuation")
    if continuation.get("teacher_conclusion_sha256") != sha256_file(
        args.teacher_conclusion
    ):
        raise RuntimeError("R1-5 continuation is not bound to this teacher conclusion")
    if teacher.get("status") not in {
        "EXPLORATORY_OMNISCIENT_TEACHER_ACTION_VALUE_CONFIRMED",
        "INCONCLUSIVE_TRAINING_NOT_CONVERGED",
    }:
        raise RuntimeError("R1-5 teacher state is not eligible for exploration")
    if revision.get("status") != "AUTHORIZED_R1_4_R1_5_EXPLORATORY_TEST":
        raise RuntimeError("R1-5 lacks the explicit owner revision")
    payload = {
        "format_version": "before-we-act.b3-n1-r1-student-contract/2",
        "stage": "R1-5-DEPLOYMENT-LEGAL-STUDENT-OWNER-REVISION",
        "status": "FROZEN_BEFORE_F0_F1",
        "created_at_utc": utc_now(),
        "parent_contract": str(args.parent_contract.resolve()),
        "parent_contract_sha256": sha256_file(args.parent_contract),
        "teacher_contract": str(args.teacher_contract.resolve()),
        "teacher_contract_sha256": sha256_file(args.teacher_contract),
        "teacher_conclusion": str(args.teacher_conclusion.resolve()),
        "teacher_conclusion_sha256": sha256_file(args.teacher_conclusion),
        "fair_conclusion": str(args.fair_conclusion.resolve()),
        "fair_conclusion_sha256": sha256_file(args.fair_conclusion),
        "owner_revision": str(args.owner_revision.resolve()),
        "owner_revision_sha256": sha256_file(args.owner_revision),
        "student_continuation": str(args.student_continuation.resolve()),
        "student_continuation_sha256": sha256_file(args.student_continuation),
        "seeds": list(BELIEF_SEEDS),
        "data_seed": BELIEF_DATA_SEED,
        "student_runtime_whitelist": [
            "frozen B0-H encoded 16-step legal history",
            "frozen B0-H summary H",
            "frozen H base action",
            "history validity mask",
        ],
        "student_runtime_forbidden": [
            "teammate current or future qpos",
            "teammate future action",
            "future ego action",
            "simulator state",
            "branch outcome/value",
            "teacher tokens",
        ],
        "teacher_use": "no-grad posterior target during phases 1-2 only; absent from deployment checkpoint input signature",
        "architecture": {
            "belief_tokens": 16,
            "d_model": 384,
            "history_reader": "16 shared learned queries cross-attend all frozen B0-H history tokens",
            "action_path": "H cross-attends all 16 student tokens through a zero-init residual",
            "uncertainty": "per-token diagonal log variance and teammate-action Gaussian",
            "robot_slot_parameters": "shared; no slot-specific student weights",
        },
        "phases": [
            {"name": name, "first_update": first, "last_update": last}
            for name, first, last in PHASES
        ],
        "objectives": {
            "phase1_teacher_token_gaussian_nll": 1.0,
            "phase2_teacher_token_gaussian_nll": 0.5,
            "phase2_teammate_action_gaussian_nll": 0.25,
            "phase2_teammate_delta": 0.25,
            "phase2_future_dino_low_weight_auxiliary": 0.01,
            "phase3_4_ego_action": 1.0,
        },
        "direct_control": "same history-query and residual action path, random at phase 3, trained only on ego action; auxiliary heads excluded from action-path capacity count",
        "comparisons": [
            "h",
            "h_student",
            "h_student_shuffle",
            "h_teacher",
            "direct_reactive",
            "belief_off",
        ],
        "updates": 80_000,
        "minimum_updates": 80_000,
        "validation_every": BELIEF_EVAL_EVERY,
        "learning_rate": 2e-4,
        "phase4_learning_rate": 2e-5,
        "platform": "the mandatory four-phase schedule completes and student/direct validation action losses each improve <1% over updates 65k, 70k, 75k, 80k",
        "selection": "lowest validation h_student action MSE among phase-4 checkpoints only",
        "gate": {
            "student": "beats H every seed on validation; cross-seed task median positive at least 4/6",
            "teacher": "remains better than H every seed on validation",
            "controls": "student beats shuffled tokens and direct reactive every seed on validation; belief-off is bitwise identical to H",
            "causal": "deferred by owner to paper-final experiments; not used in this offline exploratory classification",
        },
        "evidence_scope": {
            "kind": "exploratory_offline_student_validation_only",
            "r1_3_used_as_training_target_or_gate": False,
            "offline_success_does_not_prove_closed_loop_causal_correction": True,
            "teacher_formal_status_preserved": teacher.get("status"),
            "sealed_test_allowed": False,
        },
        "n2_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps({"status": payload["status"], "sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
