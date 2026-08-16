#!/usr/bin/env python3
"""Freeze the conditional R1-2 teammate-oracle audit contract."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from before_we_act.action_grounded_belief import (
    BELIEF_DATA_SEED,
    BELIEF_EARLIEST_PLATFORM,
    BELIEF_EVAL_EVERY,
    BELIEF_LR_DROP,
    BELIEF_MAX_UPDATES,
    BELIEF_MIN_UPDATES,
    PRIVILEGED_ORACLE_CONDITIONS,
    BELIEF_SEEDS,
)
from before_we_act.temporal_history_data import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-contract", type=Path, required=True)
    parser.add_argument("--fair-conclusion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("R1-2 oracle contract is already frozen")
    parent = json.loads(args.parent_contract.read_text(encoding="utf-8"))
    fair = json.loads(args.fair_conclusion.read_text(encoding="utf-8"))
    if parent.get("stage_id") != "B3-N1-R1-ACTION-GROUNDED-BELIEF":
        raise RuntimeError("wrong parent R1 contract")
    if fair.get("status") != "FAILED_R1_1_FAIR_PROBE" or not fair.get(
        "r1_2_required"
    ):
        raise RuntimeError("R1-2 is allowed only after a conclusive failed R1-1")
    payload = {
        "format_version": "before-we-act.b3-n1-r1-oracle-contract/1",
        "stage": "R1-2-OLD-DATA-IDENTIFIABILITY",
        "status": "FROZEN_BEFORE_F0_F1",
        "created_at_utc": utc_now(),
        "parent_contract": str(args.parent_contract.resolve()),
        "parent_contract_sha256": sha256_file(args.parent_contract),
        "fair_conclusion": str(args.fair_conclusion.resolve()),
        "fair_conclusion_sha256": sha256_file(args.fair_conclusion),
        "seeds": list(BELIEF_SEEDS),
        "data_seed": BELIEF_DATA_SEED,
        "conditions": list(PRIVILEGED_ORACLE_CONDITIONS),
        "main_comparison": "h_oracle vs h",
        "oracle_training_or_audit_only": [
            "current teammate qpos",
            "previous teammate qpos",
            "teammate qpos deltas at t+4/8/16/32",
            "actual teammate commanded actions at t:t+16",
        ],
        "deployment_use_forbidden": True,
        "ego_future_action_as_input_forbidden": True,
        "architecture": "H query cross-attends 22 projected privileged tokens; matched zero-token branch has identical trainable parameter count",
        "effective_batch": 48,
        "minimum_updates": BELIEF_MIN_UPDATES,
        "earliest_platform": BELIEF_EARLIEST_PLATFORM,
        "maximum_updates": BELIEF_MAX_UPDATES,
        "validation_every": BELIEF_EVAL_EVERY,
        "learning_rate_drop_update": BELIEF_LR_DROP,
        "learning_rate": 3e-4,
        "post_drop_multiplier": 0.1,
        "platform": "all four conditions improve <1% in each of the last three validation intervals",
        "positive": (
            "h_oracle beats h in every seed on validation and sealed test; cross-seed "
            "task median positive in >=4/6; shuffled oracle and matched capacity are worse "
            "than real oracle on both splits"
        ),
        "same_situation_pair_audit": {
            "split": "sealed scenario-group test",
            "pairing": "within task, ego slot, and phase quartile; nearest cosine B0-H hidden from a different episode",
            "statistic": "Pearson correlation between paired teammate-action MSE and paired ego-action MSE",
            "null": "10,000 within-task permutations of teammate-action distance",
            "bootstrap": "10,000 paired-row resamples",
            "positive": "correlation > permutation 97.5th percentile and bootstrap 95% CI lower bound > 0",
            "gate": ">=4/6 tasks positive",
        },
        "classification": {
            "positive": "OLD_DATA_TEAMMATE_ORACLE_VALUE_IDENTIFIED",
            "negative": "DATA_OR_TASK_HAS_NO_IDENTIFIABLE_TEAMMATE_ACTION_VALUE",
            "not_converged": "INCONCLUSIVE_TRAINING_NOT_CONVERGED",
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
    print(json.dumps({"status": payload["status"], "sha256": sha256_file(args.output)}))


if __name__ == "__main__":
    main()
