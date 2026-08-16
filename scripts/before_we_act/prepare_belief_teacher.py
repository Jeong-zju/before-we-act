#!/usr/bin/env python3
"""Freeze the owner-revised R1-4 privileged-teacher experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping

from before_we_act.action_grounded_belief import (
    BELIEF_DATA_SEED,
    BELIEF_EARLIEST_PLATFORM,
    BELIEF_EVAL_EVERY,
    BELIEF_LR_DROP,
    BELIEF_MAX_UPDATES,
    BELIEF_MIN_UPDATES,
    BELIEF_SEEDS,
)
from before_we_act.temporal_history_data import sha256_file


OWNER_STATUS = "AUTHORIZED_R1_4_R1_5_EXPLORATORY_TEST"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-contract", type=Path, required=True)
    parser.add_argument("--fair-conclusion", type=Path, required=True)
    parser.add_argument("--owner-revision", type=Path, required=True)
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


def r1_1_strong_validation_trend(fair: Mapping[str, Any]) -> dict[str, Any]:
    if fair.get("status") != "INCONCLUSIVE_TRAINING_NOT_CONVERGED":
        raise RuntimeError("owner revision expects the preserved inconclusive R1-1 receipt")
    training = fair.get("training_status", {})
    if set(training) != {str(seed) for seed in BELIEF_SEEDS}:
        raise RuntimeError("R1-1 seed set differs from the owner revision")
    per_seed: dict[str, dict[str, float | bool]] = {}
    for seed in BELIEF_SEEDS:
        metrics = training[str(seed)]["selected_validation"]["macro"]
        h = float(metrics["h"])
        h_b = float(metrics["h_b"])
        shuffle = float(metrics["h_b_shuffle"])
        matched = float(metrics["h_matched_capacity"])
        row = {
            "h": h,
            "h_b": h_b,
            "h_b_shuffle": shuffle,
            "h_matched_capacity": matched,
            "relative_improvement_vs_h": (h - h_b) / max(abs(h), 1e-12),
            "beats_h": h_b < h,
            "beats_shuffle": h_b < shuffle,
            "beats_matched_capacity": h_b < matched,
        }
        if not all(
            bool(row[key])
            for key in ("beats_h", "beats_shuffle", "beats_matched_capacity")
        ):
            raise RuntimeError(f"R1-1 seed {seed} lacks the owner-required strong trend")
        per_seed[str(seed)] = row
    return {
        "formal_status_preserved": fair["status"],
        "test_opened": bool(fair.get("test_opened", False)),
        "all_seeds_beat_h_shuffle_and_matched_capacity": True,
        "per_seed": per_seed,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("R1-4 revision contract is already frozen")
    parent = json.loads(args.parent_contract.read_text(encoding="utf-8"))
    fair = json.loads(args.fair_conclusion.read_text(encoding="utf-8"))
    revision = json.loads(args.owner_revision.read_text(encoding="utf-8"))
    if parent.get("stage_id") != "B3-N1-R1-ACTION-GROUNDED-BELIEF":
        raise RuntimeError("wrong parent R1 contract")
    if revision.get("status") != OWNER_STATUS:
        raise RuntimeError("R1-4 lacks the explicit owner revision")
    if revision.get("fair_conclusion_sha256") != sha256_file(args.fair_conclusion):
        raise RuntimeError("owner revision and R1-1 conclusion differ")
    trend = r1_1_strong_validation_trend(fair)
    payload = {
        "format_version": "before-we-act.b3-n1-r1-teacher-contract/2",
        "stage": "R1-4-OMNISCIENT-TEACHER-OWNER-REVISION",
        "status": "FROZEN_BEFORE_F0_F1",
        "created_at_utc": utc_now(),
        "parent_contract": str(args.parent_contract.resolve()),
        "parent_contract_sha256": sha256_file(args.parent_contract),
        "fair_conclusion": str(args.fair_conclusion.resolve()),
        "fair_conclusion_sha256": sha256_file(args.fair_conclusion),
        "owner_revision": str(args.owner_revision.resolve()),
        "owner_revision_sha256": sha256_file(args.owner_revision),
        "prerequisite_path": "owner-authorized-strong-r1-1-validation-trend",
        "r1_1_unlock_evidence": trend,
        "evidence_scope": {
            "kind": "exploratory_offline_teacher_test",
            "r1_1_formal_pass_claimed": False,
            "r1_3_used_as_training_target_or_gate": False,
            "sealed_test": "open only after all teacher conditions reach the frozen platform",
            "n2_authorized": False,
        },
        "seeds": list(BELIEF_SEEDS),
        "data_seed": BELIEF_DATA_SEED,
        "belief_tokens": 16,
        "d_model": 384,
        "frozen_base": "selected R1-1 H action head plus frozen Step-2 B0-H history hidden",
        "teacher_inputs_privileged_only": [
            "current and previous teammate qpos",
            "teammate qpos delta at t+4/8/16/32",
            "actual teammate action t:t+16",
        ],
        "future_ego_action_input_forbidden": True,
        "objectives": {
            "ego_action": 1.0,
            "teammate_action_gaussian_nll": 0.25,
            "teammate_delta": 0.25,
        },
        "conditions": ["h", "h_teacher", "h_teacher_shuffle", "h_matched_capacity"],
        "matched_capacity": (
            "independent identical teacher architecture receives zero privileged values and "
            "identical masks; trained on the same targets and cursor"
        ),
        "minimum_updates": BELIEF_MIN_UPDATES,
        "earliest_platform": BELIEF_EARLIEST_PLATFORM,
        "maximum_updates": BELIEF_MAX_UPDATES,
        "validation_every": BELIEF_EVAL_EVERY,
        "learning_rate_drop_update": BELIEF_LR_DROP,
        "learning_rate": 2e-4,
        "post_drop_multiplier": 0.1,
        "platform": (
            "teacher action, matched action, teammate distribution and teammate-delta losses "
            "each improve <1% over the final three evaluation intervals"
        ),
        "positive": (
            "h_teacher beats h every seed on validation/test; task median positive >=4/6; "
            "real teacher beats shuffled and zero-privileged matched controls"
        ),
        "classification": {
            "positive": "EXPLORATORY_OMNISCIENT_TEACHER_ACTION_VALUE_CONFIRMED",
            "negative": "EXPLORATORY_PRIVILEGED_TEACHER_HAS_NO_ACTION_VALUE",
            "not_converged": "INCONCLUSIVE_TRAINING_NOT_CONVERGED",
        },
        "n2_authorized": False,
    }
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "prerequisite_path": payload["prerequisite_path"],
                "sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
