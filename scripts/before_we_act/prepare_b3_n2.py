#!/usr/bin/env python3
"""Freeze the exploratory 3-N2 architecture/training contract before F0."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from before_we_act.step2_temporal_data import sha256_file


SEEDS = (20260815, 20260816, 20260817)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roadmap", type=Path, required=True)
    parser.add_argument("--r1-contract", type=Path, required=True)
    parser.add_argument("--student-contract", type=Path, required=True)
    parser.add_argument("--student-conclusion", type=Path, required=True)
    parser.add_argument("--student-diagnostic", type=Path, required=True)
    parser.add_argument("--step2-contract", type=Path, required=True)
    parser.add_argument("--b0h-checkpoint", type=Path, required=True)
    parser.add_argument("--scenario-split", type=Path, required=True)
    parser.add_argument("--n1-cache", type=Path, required=True)
    parser.add_argument("--action-context-cache", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("3-N2 refuses to overwrite a frozen contract")
    roadmap = args.roadmap.read_text(encoding="utf-8")
    authorization = "PASSED_OWNER_RELATIVE_IMPROVEMENT_GATE_N2_EXPLORATORY_AUTHORIZED"
    if authorization not in roadmap:
        raise RuntimeError("roadmap does not contain the owner's exploratory N2 authorization")
    student = json.loads(args.student_conclusion.read_text(encoding="utf-8"))
    diagnostic = json.loads(args.student_diagnostic.read_text(encoding="utf-8"))
    if student.get("status") != "INCONCLUSIVE_TRAINING_NOT_CONVERGED":
        raise RuntimeError("the immutable student machine status changed")
    if diagnostic.get("status") != "STRONG_POSITIVE_VALIDATION_TREND_BUT_NOT_CONVERGED_AND_DIRECT_CONTROL_UNRESOLVED":
        raise RuntimeError("the owner-authorized student diagnostic changed")
    action_cache_receipt = args.action_context_cache / "cache_receipt.json"
    cache = json.loads(action_cache_receipt.read_text(encoding="utf-8"))
    if cache.get("status") != "PASSED":
        raise RuntimeError("N2 frozen action-context cache is not complete")
    payload = {
        "format_version": "before-we-act.b3-n2-contract/1",
        "stage_id": "B3-N2-ARCHITECTURE",
        "status": "FROZEN_BEFORE_F0_F1",
        "created_at_utc": utc_now(),
        "source_commit": args.source_commit,
        "question": "Can the complete predictive team-belief architecture turn the owner-authorized N1 signal into repeatable offline action gains and a positive Validation5 direction?",
        "authorization": {
            "status": authorization,
            "roadmap": str(args.roadmap.resolve()),
            "roadmap_sha256": sha256_file(args.roadmap),
            "machine_student_status_preserved": student["status"],
            "direct_attribution_deferred_to_n3": True,
            "causal_claim_deferred": True,
        },
        "inputs": {
            "r1_contract": str(args.r1_contract.resolve()),
            "r1_contract_sha256": sha256_file(args.r1_contract),
            "student_contract": str(args.student_contract.resolve()),
            "student_contract_sha256": sha256_file(args.student_contract),
            "student_conclusion": str(args.student_conclusion.resolve()),
            "student_conclusion_sha256": sha256_file(args.student_conclusion),
            "student_diagnostic": str(args.student_diagnostic.resolve()),
            "student_diagnostic_sha256": sha256_file(args.student_diagnostic),
            "step2_contract": str(args.step2_contract.resolve()),
            "step2_contract_sha256": sha256_file(args.step2_contract),
            "b0h_checkpoint": str(args.b0h_checkpoint.resolve()),
            "b0h_checkpoint_sha256": sha256_file(args.b0h_checkpoint),
            "scenario_split": str(args.scenario_split.resolve()),
            "scenario_split_sha256": sha256_file(args.scenario_split),
            "n1_metadata": str((args.n1_cache / "metadata.json").resolve()),
            "n1_metadata_sha256": sha256_file(args.n1_cache / "metadata.json"),
            "action_context_cache": str(args.action_context_cache.resolve()),
            "action_context_cache_receipt_sha256": sha256_file(action_cache_receipt),
        },
        "architecture": {
            "d_model": 384,
            "belief_tokens": 16,
            "agent_anchors": 2,
            "free_interaction_tokens": 14,
            "evidence_queries": 4,
            "event_capacity": 4,
            "temporal_layers": 2,
            "heads": 8,
            "dropout": 0.1,
            "capacity_rationale": "R1 only established the action signal with all 16 student tokens; retain that measured capacity, use four evidence queries/four bounded events, and match the two-layer legal-history depth. These values are frozen before N2 metrics and will not be searched.",
            "action_interface": "all 100 B0-H decoded action queries cross-attend all 16 belief tokens through a zero-init reliability-gated residual",
            "base": "formal B0-H hidden-residual checkpoint; belief-off is bitwise the same base action",
            "teacher": "training-only synchronized three-view/future/joint-state posterior; physically absent from deployment export",
            "runtime": "legal 16-step global+ego-local pooled DINO, ego qpos, executed ego action, task text, validity/reset masks",
        },
        "future_contract": {
            "source_frequency_hz": 20,
            "offset_steps": [4, 8, 16, 32],
            "offset_seconds": [0.2, 0.4, 0.8, 1.6],
            "tail_policy": "mask_missing_anchor",
            "teacher_target_space": "frozen_DINO_latent",
        },
        "training": {
            "seeds": list(SEEDS),
            "data_seed": 20260815,
            "scenario_group_train_validation_test_episodes_per_task": [96, 12, 12],
            "paired_arms_per_situation": True,
            "effective_batch": 48,
            "samples_per_task": 8,
            "minimum_updates": 120000,
            "maximum_updates": 120000,
            "u_b0h": 120000,
            "validation_every": 5000,
            "learning_rate": 0.0002,
            "learning_rate_drop_update": 80000,
            "post_drop_learning_rate": 0.00002,
            "selection_window_updates": [100000, 105000, 110000, 115000, 120000],
            "selection": "lowest validation B-core action MSE in the frozen selection window, only after the 120k sufficiency decision",
            "smoothing": "three-point trailing arithmetic mean",
            "platform": "after the LR drop, the last four smoothed primary scores each improve by less than 1%; no key auxiliary is still improving by >=1%; no three-point validation overfit streak",
        },
        "objectives": {
            "action": 1.0,
            "action_posterior_kl": 0.0,
            "teacher_alignment": 0.1,
            "future_latent": 0.01,
            "teacher_reconstruction": 0.01,
            "teammate_delta": 0.1,
            "teammate_action": 0.1,
            "exchange_consistency": 0.05,
            "anti_collapse": 0.01,
            "direct_reactive_action": 1.0,
        },
        "controls": ["formal_b0h", "direct_reactive", "belief_shuffle", "belief_off"],
        "cooperation_diagnostic": {
            "name": "paired_inactivity_steps",
            "definition": "before success, count steps where both arms' commanded seven-joint change from current qpos has L2 norm <0.02",
            "interpretation": "waiting proxy only; report direction beside success and steps, never call it causal teamwork proof",
        },
        "invalid_targets_excluded": {
            "r1_3_branch_value": True,
            "shared_change_reward": True,
            "reason": "the preserved R1-3 pilot produced all-zero reward/success and is not a valid training label; this limits causal claims but does not revoke the owner's exploratory N2 authorization",
        },
        "classification": {
            "platform_missing": "INCONCLUSIVE_TRAINING_NOT_CONVERGED",
            "offline_positive_validation5_positive": "POSITIVE_SIGNAL",
            "offline_positive_validation5_flat": "WEAK_SIGNAL",
            "offline_or_validation5_negative": "NO_SIGNAL",
            "formal_pass_forbidden": True,
        },
        "validation5_gate": "run only if every seed is training-sufficient; use the existing frozen per-task seed files and compare aggregate/task direction with the formal B0-H results",
    }
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
