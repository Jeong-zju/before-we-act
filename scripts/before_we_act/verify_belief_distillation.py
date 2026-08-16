#!/usr/bin/env python3
"""Run fail-closed F0 checks for the owner-revised R1-4/R1-5 experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import torch

from before_we_act.raw_team_signal_data import ACTION_PROBE_HORIZON, FUTURE_OFFSETS
from before_we_act.belief_distillation import (
    DirectReactiveControl,
    LegalBeliefStudent,
    PrivilegedBeliefTeacher,
)
from before_we_act.temporal_history_data import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-revision", type=Path, required=True)
    parser.add_argument("--teacher-contract", type=Path, required=True)
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


def parameters(module: torch.nn.Module) -> int:
    return sum(value.numel() for value in module.parameters())


def action_path_parameters(student: LegalBeliefStudent) -> int:
    return (
        student.queries.numel()
        + parameters(student.reader)
        + parameters(student.token_norm)
        + parameters(student.action_residual)
    )


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("owner revision F0 receipt already exists")
    revision = json.loads(args.owner_revision.read_text(encoding="utf-8"))
    contract = json.loads(args.teacher_contract.read_text(encoding="utf-8"))
    checks = {
        "owner_revision_authorized": revision.get("status")
        == "AUTHORIZED_R1_4_R1_5_EXPLORATORY_TEST",
        "teacher_contract_v2": contract.get("format_version")
        == "before-we-act.b3-n1-r1-teacher-contract/2",
        "owner_revision_hash_bound": contract.get("owner_revision_sha256")
        == sha256_file(args.owner_revision),
        "r1_1_formal_pass_not_claimed": contract.get("evidence_scope", {}).get(
            "r1_1_formal_pass_claimed"
        )
        is False,
        "r1_3_not_used": contract.get("evidence_scope", {}).get(
            "r1_3_used_as_training_target_or_gate"
        )
        is False,
        "pilot_objectives_absent": not {
            "counterfactual_value_and_ranking",
            "shared_state_change",
        }.intersection(contract.get("objectives", {})),
        "n2_not_authorized": contract.get("n2_authorized") is False,
    }

    torch.manual_seed(20260815)
    batch = 3
    d_model = 32
    teacher = PrivilegedBeliefTeacher(d_model=d_model)
    matched = PrivilegedBeliefTeacher(d_model=d_model)
    student = LegalBeliefStudent(d_model=d_model)
    direct = DirectReactiveControl(d_model=d_model)
    base = torch.randn(batch, ACTION_PROBE_HORIZON, 8)
    h = torch.randn(batch, d_model)
    oracle = {
        "teammate_qpos": torch.randn(batch, 9),
        "previous_teammate_qpos": torch.randn(batch, 9),
        "teammate_delta": torch.randn(batch, len(FUTURE_OFFSETS), 9),
        "future_mask": torch.ones(batch, len(FUTURE_OFFSETS), dtype=torch.bool),
        "oracle_teammate_action": torch.randn(batch, ACTION_PROBE_HORIZON, 8),
        "oracle_teammate_action_mask": torch.ones(
            batch, ACTION_PROBE_HORIZON, dtype=torch.bool
        ),
    }
    history = torch.randn(batch, 12, d_model)
    history_mask = torch.ones(batch, 12, dtype=torch.bool)
    teacher_output = teacher(base, h, oracle)
    student_output = student(base, h, history, history_mask)
    direct_output = direct(base, h, history, history_mask)
    checks.update(
        {
            "teacher_action_shape": tuple(teacher_output.action.shape)
            == (batch, ACTION_PROBE_HORIZON, 8),
            "teacher_tokens_shape": tuple(teacher_output.tokens.shape)
            == (batch, 16, d_model),
            "teacher_has_no_pilot_outputs": not hasattr(teacher_output, "branch_value")
            and not hasattr(teacher_output, "shared_change"),
            "student_action_shape": tuple(student_output.action.shape)
            == (batch, ACTION_PROBE_HORIZON, 8),
            "direct_action_shape": tuple(direct_output.shape)
            == (batch, ACTION_PROBE_HORIZON, 8),
            "all_outputs_finite": all(
                bool(torch.isfinite(value).all())
                for value in (
                    teacher_output.action,
                    teacher_output.tokens,
                    student_output.action,
                    student_output.tokens,
                    direct_output,
                )
            ),
            "teacher_zero_init_falls_back_to_base": bool(
                torch.equal(teacher_output.action, base)
            ),
            "student_zero_init_falls_back_to_base": bool(
                torch.equal(student_output.action, base)
            ),
            "teacher_matched_parameter_count": parameters(teacher)
            == parameters(matched),
            "student_direct_action_path_parameter_count": action_path_parameters(student)
            == parameters(direct),
        }
    )
    status = "PASSED" if all(checks.values()) else "FAILED"
    result = {
        "format_version": "before-we-act.b3-n1-r1-owner-revision-f0/1",
        "stage": "R1-OWNER-REVISION-F0",
        "status": status,
        "completed_at_utc": utc_now(),
        "owner_revision_sha256": sha256_file(args.owner_revision),
        "teacher_contract_sha256": sha256_file(args.teacher_contract),
        "checks": checks,
        "parameter_counts": {
            "teacher": parameters(teacher),
            "teacher_matched": parameters(matched),
            "student_total": parameters(student),
            "student_action_path": action_path_parameters(student),
            "direct_action_path": parameters(direct),
        },
    }
    atomic_json(args.output, result)
    print(json.dumps({"status": status, "checks": checks}, sort_keys=True))
    if status != "PASSED":
        raise RuntimeError("owner revision F0 failed")


if __name__ == "__main__":
    main()
