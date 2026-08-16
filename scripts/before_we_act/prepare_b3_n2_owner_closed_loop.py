#!/usr/bin/env python3
"""Freeze the owner's post-training authorization for the N2 closed loop."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Mapping

from before_we_act.step2_temporal_data import sha256_file
from scripts.before_we_act.analyze_b3_n2 import load_training, offline_gate
from scripts.before_we_act.summarize_b3_n2_validation20 import (
    EXPECTED_BASELINES,
    select_validation20_candidate,
)


OWNER_TOKEN = "AUTHORIZED_OWNER_N2_CLOSED_LOOP_AFTER_PRIMARY_PLATEAU_20260816"
OWNER_STATUS = "OWNER_AUTHORIZED_CLOSED_LOOP_AFTER_PRIMARY_PLATEAU"
PRIMARY_SERIES = (
    "b_core_action",
    "future_1.6s",
    "teacher_alignment",
    "teammate_action",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json_new(path: Path, value: object) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite owner authorization: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if path.exists():
        temporary.unlink()
        raise RuntimeError(f"owner authorization appeared concurrently: {path}")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roadmap", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--old-conclusion", type=Path, required=True)
    parser.add_argument("--b0h-checkpoint", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def primary_plateau_override(training: Mapping) -> dict:
    """Verify that only the frozen teammate-delta auxiliary blocks sufficiency."""
    per_seed = {}
    for seed, row in training.items():
        sufficiency = row["training_sufficiency"]
        receipt = sufficiency["receipt"]
        series = receipt["series"]
        blockers = sorted(
            name
            for name, value in series.items()
            if value.get("all_three_below_one_percent") is not True
        )
        primary_passes = {
            name: series[name].get("all_three_below_one_percent") is True
            for name in PRIMARY_SERIES
        }
        checks = {
            "old_status_is_inconclusive": (
                row["status"] == "INCONCLUSIVE_TRAINING_NOT_CONVERGED"
            ),
            "ran_full_budget": (
                sufficiency.get("minimum_exposure_met") is True
                and int(receipt.get("maximum_updates", -1)) == 120000
                and 120000 in receipt.get("points", [])
            ),
            "learning_rate_drop_completed": (
                sufficiency.get("learning_rate_drop_completed") is True
            ),
            "no_overfit_trigger": (
                receipt.get("overfit_last_three_intervals") is False
            ),
            "all_primary_series_plateaued": all(primary_passes.values()),
            "only_teammate_delta_blocks_old_rule": blockers == ["teammate_delta"],
        }
        per_seed[str(seed)] = {
            "checks": checks,
            "primary_series": primary_passes,
            "old_rule_blocking_series": blockers,
            "teammate_delta_last_three_relative_improvements": series[
                "teammate_delta"
            ]["relative_improvements"],
        }
    return {
        "per_seed": per_seed,
        "passed": all(
            all(seed_row["checks"].values()) for seed_row in per_seed.values()
        ),
    }


def prepare(
    contract: Mapping,
    training: Mapping,
    old_conclusion: Mapping,
    *,
    contract_path: Path,
    old_conclusion_path: Path,
    b0h_checkpoint: Path,
    source_commit: str,
) -> dict:
    if old_conclusion.get("status") != "INCONCLUSIVE_TRAINING_NOT_CONVERGED":
        raise RuntimeError("the frozen old conclusion is not the expected inconclusive result")
    if old_conclusion.get("validation5_authorized") is not False:
        raise RuntimeError("the frozen old conclusion unexpectedly authorized Validation5")
    plateau = primary_plateau_override(training)
    if not plateau["passed"]:
        raise RuntimeError("owner override preconditions are not satisfied")
    gate = offline_gate(dict(training))
    if not gate["passed"]:
        raise RuntimeError("the three frozen checkpoints do not pass the offline quality gate")
    if sha256_file(b0h_checkpoint) != EXPECTED_BASELINES["b0h"][
        "checkpoint_sha256"
    ]:
        raise RuntimeError("B0-H checkpoint hash drifted")
    selected = select_validation20_candidate(training)
    return {
        "format_version": "before-we-act.b3-n2-owner-closed-loop-revision/1",
        "stage": "B3-N2-ARCHITECTURE",
        "status": OWNER_STATUS,
        "created_at_utc": utc_now(),
        "authorization_token": OWNER_TOKEN,
        "owner_directive": "继续完成闭环",
        "source_commit": source_commit,
        "frozen_contract": str(contract_path.resolve()),
        "frozen_contract_sha256": sha256_file(contract_path),
        "old_conclusion": str(old_conclusion_path.resolve()),
        "old_conclusion_sha256": sha256_file(old_conclusion_path),
        "old_conclusion_status_preserved": old_conclusion["status"],
        "old_training_sufficiency_rule_overwritten": False,
        "owner_override_scope": (
            "Run the already-trained three-seed Validation5 and the pre-selected "
            "seed Validation20 diagnostic despite the auxiliary teammate_delta "
            "plateau failure."
        ),
        "primary_plateau_evidence": plateau,
        "offline_gate_recomputed_without_rewriting_old_conclusion": gate,
        "training_receipts": {
            str(seed): {
                "status": row["status"],
                "selected_update": row["selected_update"],
                "training_sufficiency_sha256": row[
                    "training_sufficiency_sha256"
                ],
                "deployment_checkpoint": row["deployment_checkpoint"],
                "deployment_checkpoint_sha256": row[
                    "deployment_checkpoint_sha256"
                ],
            }
            for seed, row in training.items()
        },
        "b0h_checkpoint": str(b0h_checkpoint.resolve()),
        "b0h_checkpoint_sha256": sha256_file(b0h_checkpoint),
        "validation5_authorized_by_owner": True,
        "validation20_authorized_by_owner": True,
        "validation20_requires_positive_validation5": False,
        "validation20_candidate": selected,
        "n3_authorized": False,
        "formal_pass": False,
        "claim_limits": [
            "The immutable old training-sufficiency conclusion remains inconclusive.",
            "Validation5 is diagnostic and is not used to select the Validation20 seed.",
            "The owner explicitly requires Validation20 to complete even if Validation5 is not positive.",
            "This authorization cannot issue an N3, N4, Confirmation50, or formal B-core pass.",
        ],
    }


def main() -> None:
    args = parse_args()
    if OWNER_TOKEN not in args.roadmap.read_text(encoding="utf-8"):
        raise RuntimeError("owner closed-loop authorization is not frozen in the roadmap")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    old_conclusion = json.loads(args.old_conclusion.read_text(encoding="utf-8"))
    training, all_sufficient = load_training(contract, args.run_root)
    if all_sufficient:
        raise RuntimeError("owner override is unnecessary because the old gate already passed")
    payload = prepare(
        contract,
        training,
        old_conclusion,
        contract_path=args.contract,
        old_conclusion_path=args.old_conclusion,
        b0h_checkpoint=args.b0h_checkpoint,
        source_commit=args.source_commit,
    )
    atomic_json_new(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
