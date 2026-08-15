#!/usr/bin/env python3
"""Freeze the action-conditioned predictive-pairing 3-N2 contract before F0."""
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
    parser.add_argument("--failed-n2-run", type=Path, required=True)
    parser.add_argument("--stabilized-n2-run", type=Path, required=True)
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
    failed_status_path = args.failed_n2_run / "pipeline_status.json"
    failed_contract_path = args.failed_n2_run / "contract/n2_contract.json"
    failed_status = json.loads(failed_status_path.read_text(encoding="utf-8"))
    if failed_status.get("status") != "STOPPED":
        raise RuntimeError("the superseded Gaussian N2 run is not stopped")
    stabilized_conclusion_path = args.stabilized_n2_run / "repair_pilot_conclusion.json"
    stabilized_pilot_path = (
        args.stabilized_n2_run
        / "repair_pilot"
        / "seed_20260815"
        / "status.json"
    )
    stabilized = json.loads(
        stabilized_conclusion_path.read_text(encoding="utf-8")
    )
    stabilized_pilot = json.loads(stabilized_pilot_path.read_text(encoding="utf-8"))
    if stabilized.get("status") != (
        "PASSED_REPAIR_GATES_FORMAL_TRAINING_REQUIRES_OWNER_DECISION"
    ):
        raise RuntimeError("the discrete-belief stabilization pilot did not pass")
    stabilized_validation = stabilized_pilot["selected_validation"]
    failed_seed_evidence = {}
    for seed in SEEDS:
        evaluation_path = args.failed_n2_run / f"training/seed_{seed}/evaluations.jsonl"
        rows = [
            json.loads(line)
            for line in evaluation_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        selected = next((row for row in rows if int(row["update"]) == 30_000), None)
        if selected is None:
            raise RuntimeError(f"failed N2 seed {seed} lacks the common 30000 evaluation")
        validation = selected["validation"]
        failed_seed_evidence[str(seed)] = {
            "b_core_action_mse": validation["macro"]["b_core"],
            "b_shuffle_action_mse": validation["macro"]["b_shuffle"],
            "direct_action_mse": validation["macro"]["direct_reactive"],
            "future_1.6s_mse": validation["future_mse"]["model"]["1.6s"],
            "persistence_1.6s_mse": validation["future_mse"]["persistence"]["1.6s"],
            "belief_effective_rank": validation["belief"]["effective_rank"],
            "teacher_alignment": validation["auxiliary"]["teacher_alignment"],
        }
    action_cache_receipt = args.action_context_cache / "cache_receipt.json"
    cache = json.loads(action_cache_receipt.read_text(encoding="utf-8"))
    if cache.get("status") != "PASSED":
        raise RuntimeError("N2 frozen action-context cache is not complete")
    payload = {
        "format_version": "before-we-act.b3-n2-contract/2",
        "stage_id": "B3-N2-R2-ACTION-CONDITIONED-PREDICTIVE-PAIRING",
        "status": "FROZEN_BEFORE_F0_F1",
        "created_at_utc": utc_now(),
        "source_commit": args.source_commit,
        "question": "Can one action-conditioned predictive-pairing migration make the action residual causally depend on the correctly paired belief and beat a legal-view persistence baseline before any new full-budget training?",
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
            "superseded_gaussian_n2_run": str(args.failed_n2_run.resolve()),
            "superseded_gaussian_n2_contract_sha256": sha256_file(failed_contract_path),
            "superseded_gaussian_n2_pipeline_status_sha256": sha256_file(failed_status_path),
            "stabilized_discrete_n2_conclusion": str(
                stabilized_conclusion_path.resolve()
            ),
            "stabilized_discrete_n2_conclusion_sha256": sha256_file(
                stabilized_conclusion_path
            ),
            "stabilized_discrete_n2_pilot_status": str(
                stabilized_pilot_path.resolve()
            ),
            "stabilized_discrete_n2_pilot_status_sha256": sha256_file(
                stabilized_pilot_path
            ),
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
            "belief_distribution": "12 independent 32-class categorical factors per belief token",
            "belief_factors": 12,
            "belief_classes": 32,
            "belief_unimix": 0.01,
            "belief_free_nats": 1.0,
            "belief_representation_scale": 0.1,
            "belief_feature_interface": "centered categorical probabilities projected to d_model; teacher reconstruction and runtime prediction must pass through this bottleneck",
            "capacity_rationale": "R1 only established the action signal with all 16 student tokens; retain that measured capacity, use four evidence queries/four bounded events, and match the two-layer legal-history depth. These values are frozen before N2 metrics and will not be searched.",
            "action_interface": "all 100 B0-H decoded action queries cross-attend all 16 projected categorical belief tokens through a zero-init entropy-reliability-gated residual; raw decoded-action hidden is forbidden from bypassing the belief readout",
            "base": "formal B0-H hidden-residual checkpoint; belief-off is bitwise the same base action",
            "teacher": "training-only synchronized three-view/future/joint-state posterior; physically absent from deployment export",
            "runtime": "legal 16-step global+ego-local pooled DINO, ego qpos, executed ego action, task text, validity/reset masks",
        },
        "superseded_failure_evidence": {
            "status": "STOPPED_BY_OWNER_BEFORE_MORE_WASTED_TRAINING",
            "common_evaluation_update": 30000,
            "seeds": failed_seed_evidence,
            "interpretation": "All seeds beat B0-H slightly but lose to the equally sized direct-history control; belief shuffle is effectively unchanged, future prediction loses badly to persistence, effective rank is below two, and Gaussian teacher KL exploded. The old run is retained as negative evidence, not resumed.",
        },
        "retained_discrete_belief_repair": {
            "status": stabilized["status"],
            "kl_stable": stabilized["gates"]["kl_stable"],
            "belief_estimable": stabilized["gates"]["belief_estimable"],
            "multidimensional_team_state": stabilized["gates"][
                "multidimensional_team_state"
            ],
            "action": {
                "b_core_mse": stabilized_validation["macro"]["b_core"],
                "b_shuffle_mse": stabilized_validation["macro"]["b_shuffle"],
                "direct_reactive_mse": stabilized_validation["macro"][
                    "direct_reactive"
                ],
                "interpretation": "The bounded categorical belief is estimable and multidimensional, but the action output is unchanged by a within-task/phase belief shuffle.",
            },
            "future": {
                "model_mse": stabilized_validation["future_mse"]["model"],
                "persistence_mse": stabilized_validation["future_mse"][
                    "persistence"
                ],
                "interpretation": "The old absolute future head sees no future action and loses to persistence at every horizon; its persistence comparator also included a training-only teammate view.",
            },
        },
        "single_idea_migration": {
            "name": "action-conditioned predictive pairing",
            "primary_model": {
                "paper": "V-JEPA 2",
                "paper_url": "https://arxiv.org/abs/2506.09985",
                "official_repository": "https://github.com/facebookresearch/vjepa2",
                "official_source": "src/models/ac_predictor.py and app/vjepa_droid/train.py",
                "license": "MIT",
            },
            "dino_latent_precedent": {
                "paper": "DINO-WM",
                "paper_url": "https://arxiv.org/abs/2411.04983",
                "official_repository": "https://github.com/gaoyuezhou/dino_wm",
                "license": "MIT",
            },
            "pairing_objective": {
                "paper": "TACO",
                "paper_url": "https://proceedings.neurips.cc/paper_files/paper/2023/file/96d00450ed65531ffe2996daed487536-Paper-Conference.pdf",
                "official_repository": "https://github.com/FrankZheng2022/TACO",
                "official_source": "agents/taco.py",
                "license": "MIT",
            },
            "implementation": "PyTorch adaptation of the mechanisms; no upstream source copied",
            "mechanism": "A shared recurrent predictor rolls legal current DINO features and belief forward under the future ego-action sequence, predicts only a zero-initialized delta over persistence, and trains the real policy output so a correct belief/action pairing outranks a within-task/phase shuffled pairing whenever their residual targets are measurably different.",
            "one_idea_rationale": "Both repairs enforce the same causal statement: the latent must contain information that is useful only when paired with the correct action-conditioned transition. They are evaluated together by correct-vs-shuffled interventions, not treated as independent score boosters.",
        },
        "future_contract": {
            "source_frequency_hz": 20,
            "offset_steps": [4, 8, 16, 32],
            "offset_seconds": [0.2, 0.4, 0.8, 1.6],
            "tail_policy": "mask_missing_anchor",
            "teacher_target_space": "frozen_DINO_latent",
            "runtime_prediction_views": ["global", "ego_local"],
            "teacher_reconstruction_views": [
                "global",
                "ego_local",
                "teammate_local"
            ],
            "baseline": "persistence from the same legal current runtime views only",
            "conditioning": "ground-truth ego action during world-model training; policy action at deployment; report both plus shuffled-action and shuffled-belief interventions",
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
            "repair_pilot": {
                "seed": 20260815,
                "updates": 4000,
                "evaluate_at_end": True,
                "purpose": "test action/belief binding and action-conditioned future prediction against fair interventions; this cannot authorize formal training or Validation5",
            },
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
            "action_pairing": 1.0,
            "action_pairing_margin_fraction": 0.1,
            "action_pairing_margin_cap": 0.01,
            "direct_reactive_action": 1.0,
        },
        "controls": [
            "formal_b0h",
            "direct_reactive",
            "belief_shuffle",
            "belief_off",
            "future_legal_persistence",
            "future_shuffled_action",
            "future_shuffled_belief"
        ],
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
        "repair_gates": {
            "retained_kl_stability": "every logged teacher_alignment is finite and <= 8.867290 (8.061172 dynamics + 0.1 representation)",
            "retained_multidimensional_state": "effective rank > 4 and at least 4/12 categorical factors have normalized mutual information > 0.01",
            "action_binding": "validation b_shuffle MSE must be strictly worse than b_core MSE and shuffled-belief residual-output MSE divided by residual energy must be >= 0.01",
            "future_prediction": "oracle-action future MSE must beat same-view persistence at at least 3/4 horizons and shuffled-action MSE must be worse than oracle-action MSE at at least 3/4 horizons",
            "action_quality_guard": "b_core action MSE must not exceed direct-reactive MSE by more than 1%",
            "resume_rule": "do not start a new 120k three-seed run until every retained and new causal gate passes; owner authorization remains separately required",
        },
        "validation5_gate": "run only if every seed is training-sufficient; use the existing frozen per-task seed files and compare aggregate/task direction with the formal B0-H results",
    }
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
