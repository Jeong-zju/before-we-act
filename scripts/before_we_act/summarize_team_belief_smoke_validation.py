#!/usr/bin/env python3
"""Summarize the owner-authorized N2 Validation5 without changing old gates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Mapping

from before_we_act.temporal_history_data import SIX_TASKS, sha256_file
from scripts.before_we_act.analyze_predictive_team_belief import load_validation, validation_gate
from scripts.before_we_act.prepare_team_belief_closed_loop import (
    OWNER_STATUS,
    OWNER_TOKEN,
)


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
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def verify_validation_receipts(
    validation_root: Path,
    seed_root: Path,
    contract: Mapping,
    authorization: Mapping,
) -> None:
    expected = {
        "b0h": (
            "b0h",
            authorization["b0h_checkpoint_sha256"],
        )
    }
    for seed in contract["training"]["seeds"]:
        expected[f"seed_{seed}"] = (
            "n2",
            authorization["training_receipts"][str(seed)][
                "deployment_checkpoint_sha256"
            ],
        )
    for label, (mode, checkpoint_sha) in expected.items():
        for task in SIX_TASKS:
            path = validation_root / label / f"{task}.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("mode") != mode:
                raise RuntimeError(f"Validation5 mode drifted for {label}/{task}")
            if int(value.get("episodes", -1)) != 5 or len(value.get("rows", [])) != 5:
                raise RuntimeError(f"Validation5 is incomplete for {label}/{task}")
            if value.get("checkpoint_sha256") != checkpoint_sha:
                raise RuntimeError(f"Validation5 checkpoint drifted for {label}/{task}")
            seed_file = seed_root / f"{task}.json"
            if value.get("seed_protocol", {}).get("sha256") != sha256_file(seed_file):
                raise RuntimeError(f"Validation5 seed receipt drifted for {label}/{task}")
            frozen_seeds = json.loads(seed_file.read_text(encoding="utf-8"))["seeds"][:5]
            actual_seeds = [int(row["seed"]) for row in value["rows"]]
            if actual_seeds != [int(seed) for seed in frozen_seeds]:
                raise RuntimeError(f"Validation5 seed order drifted for {label}/{task}")


def summarize(
    contract: Mapping,
    authorization: Mapping,
    validation_root: Path,
    seed_root: Path,
) -> dict:
    if authorization.get("status") != OWNER_STATUS:
        raise RuntimeError("Validation5 lacks the frozen owner authorization")
    if authorization.get("authorization_token") != OWNER_TOKEN:
        raise RuntimeError("Validation5 owner token drifted")
    if authorization.get("validation5_authorized_by_owner") is not True:
        raise RuntimeError("owner did not authorize Validation5")
    verify_validation_receipts(
        validation_root, seed_root, contract, authorization
    )
    validation = load_validation(validation_root, dict(contract))
    gate = validation_gate(validation, dict(contract))
    return {
        "format_version": "before-we-act.b3-n2-owner-validation5/1",
        "stage": "B3-N2-ARCHITECTURE",
        "status": "COMPLETED_OWNER_AUTHORIZED_VALIDATION5_DIAGNOSTIC",
        "completed_at_utc": utc_now(),
        "authorization_token": OWNER_TOKEN,
        "authorization_sha256": None,
        "validation5": validation,
        "frozen_validation5_gate": gate,
        "validation20_candidate": authorization["validation20_candidate"],
        "validation20_required_by_owner_regardless_of_validation5_gate": True,
        "n3_authorized": False,
        "formal_pass": False,
        "human_summary": (
            "Validation5 通过原冻结方向门。"
            if gate["passed"]
            else "Validation5 未通过原冻结方向门；按负责人新增授权仍继续固定候选的 Validation20，以完成同协议闭环比较。"
        ),
        "claim_limits": [
            "This result does not modify the old inconclusive training-sufficiency conclusion.",
            "Validation5 has only five episodes per task and cannot select the Validation20 seed.",
            "Validation20 completion is an explicit owner diagnostic, not an N3 or formal authorization.",
        ],
    }


def main() -> None:
    args = parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
    payload = summarize(
        contract, authorization, args.validation_root, args.seed_root
    )
    payload["authorization_sha256"] = sha256_file(args.authorization)
    atomic_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
