#!/usr/bin/env python3
"""Issue per-seed/per-condition R1-1 training-sufficiency receipts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from before_we_act.action_grounded_belief import (
    ACTION_GROUNDED_CONDITIONS,
    BELIEF_EARLIEST_PLATFORM,
    BELIEF_EVAL_EVERY,
    BELIEF_LR_DROP,
    BELIEF_MAX_UPDATES,
    BELIEF_MIN_UPDATES,
    BELIEF_SEEDS,
)
from before_we_act.temporal_history_data import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
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
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract.get("stage_id") != "B3-N1-R1-ACTION-GROUNDED-BELIEF":
        raise RuntimeError("wrong R1 contract")
    index: dict[str, dict] = {}
    expected_updates = list(range(BELIEF_EVAL_EVERY, BELIEF_MAX_UPDATES + 1, BELIEF_EVAL_EVERY))
    for seed in BELIEF_SEEDS:
        root = args.run_root / "r1_1_fair_probe" / f"seed_{seed}"
        status_path = root / "status.json"
        evaluations_path = root / "evaluations.jsonl"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        evaluations = [
            json.loads(line)
            for line in evaluations_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        updates = [int(row["update"]) for row in evaluations]
        if updates != expected_updates:
            raise RuntimeError(f"R1-1 seed {seed} evaluation schedule differs")
        conditions: dict[str, dict] = {}
        for condition in ACTION_GROUNDED_CONDITIONS:
            points = [
                {
                    "update": int(row["update"]),
                    "validation_macro_mse": float(
                        row["validation"]["macro"][condition]
                    ),
                    "learning_rate": float(row["learning_rate"]),
                }
                for row in evaluations
            ]
            scores = [row["validation_macro_mse"] for row in points[-4:]]
            improvements = [
                (previous - current) / max(abs(previous), 1e-12)
                for previous, current in zip(scores, scores[1:])
            ]
            platform = (
                updates[-1] >= BELIEF_EARLIEST_PLATFORM
                and all(value < 0.01 for value in improvements)
            )
            if platform != bool(status["condition_platform"][condition]):
                raise RuntimeError(f"R1-1 seed {seed} {condition} platform differs")
            conditions[condition] = {
                "all_evaluation_points": points,
                "smoothing": "none; frozen raw macro validation MSE",
                "recent_three_relative_improvements": improvements,
                "recent_three_each_below_one_percent": platform,
                "learning_rate_drop_update": BELIEF_LR_DROP,
                "learning_rate_before_drop": 3e-4,
                "learning_rate_after_drop": 3e-5,
                "platform_checkpoint": updates[-1] if platform else None,
                "training_sufficient": platform,
            }
        receipt = {
            "format_version": "before-we-act.training-sufficiency/1",
            "stage": "R1-1-FAIR-PROBE",
            "seed": seed,
            "completed_at_utc": utc_now(),
            "contract_sha256": sha256_file(args.contract),
            "status_sha256": sha256_file(status_path),
            "evaluations_sha256": sha256_file(evaluations_path),
            "minimum_updates": BELIEF_MIN_UPDATES,
            "minimum_exposure_met": int(status["update"]) >= BELIEF_MIN_UPDATES,
            "earliest_platform": BELIEF_EARLIEST_PLATFORM,
            "maximum_updates": BELIEF_MAX_UPDATES,
            "maximum_updates_reached": int(status["update"]) == BELIEF_MAX_UPDATES,
            "validation_every": BELIEF_EVAL_EVERY,
            "u_b0h_matched_compute_reference": 120_000,
            "selected_update": int(status["selected_update"]),
            "conditions": conditions,
            "all_trainable_conditions_platform": all(
                conditions[name]["training_sufficient"] for name in ACTION_GROUNDED_CONDITIONS
            ),
            "status": status["status"],
        }
        receipt_path = root / "training_sufficiency.json"
        atomic_json(receipt_path, receipt)
        index[str(seed)] = {
            "path": str(receipt_path.resolve()),
            "sha256": sha256_file(receipt_path),
            "status": receipt["status"],
            "all_trainable_conditions_platform": receipt[
                "all_trainable_conditions_platform"
            ],
        }
    payload = {
        "format_version": "before-we-act.b3-n1-r1-training-sufficiency-index/1",
        "stage": "R1-1-FAIR-PROBE",
        "completed_at_utc": utc_now(),
        "contract_sha256": sha256_file(args.contract),
        "seeds": index,
        "all_seeds_training_sufficient": all(
            row["all_trainable_conditions_platform"] for row in index.values()
        ),
    }
    atomic_json(args.output_index, payload)
    print(
        json.dumps(
            {
                "all_seeds_training_sufficient": payload[
                    "all_seeds_training_sufficient"
                ],
                "sha256": sha256_file(args.output_index),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
