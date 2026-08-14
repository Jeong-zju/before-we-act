#!/usr/bin/env python3
"""Freeze the owner-authorized signal-first R4-B supplement before metrics exist."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[2]
STAGE_ID = "SSC-V7-M3-R4-B-SUPPLEMENT"
RUN_ROOT = Path(
    "/workspace/bwa_runs/ssc-v7-social-state-cooperation-v2/measurement/"
    "m3_r4_b_supplement_v1"
)
SOURCE_ROOT = Path(
    "/workspace/bwa_runs/ssc-v7-social-state-cooperation-v2/measurement/"
    "m3_r4_b_observability_v1"
)
SOURCE_GATE = SOURCE_ROOT / "frozen_gate/m3_r4_b_gate.json"
SOURCE_RECEIPT = SOURCE_ROOT / "formal/r4_b_observability_receipt.json"
OUTPUT = RUN_ROOT / "frozen_gate/m3_r4_b_supplement_gate.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    if RUN_ROOT.exists():
        raise FileExistsError(f"fresh supplement root required: {RUN_ROOT}")
    source_receipt = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    if not source_receipt.get("r4_b_completed"):
        raise RuntimeError("source R4-B is not complete")
    if int(source_receipt.get("test_paths_opened", -1)) != 0:
        raise RuntimeError("source R4-B test boundary is not clean")

    script = Path("scripts/before_we_act/run_ssc_v7_m3_r4_b_supplement.py")
    test = Path("tests/before_we_act/test_ssc_v7_m3_r4_b_supplement.py")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPOSITORY, text=True
    ).strip():
        raise RuntimeError("freeze requires a clean implementation worktree")
    implementation_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    gate: dict[str, Any] = {
        "schema_version": "ssc-v7.m3_r4_b.supplement_gate/1",
        "stage_id": STAGE_ID,
        "status": "FROZEN_BEFORE_SUPPLEMENT_CONTROL_METRICS",
        "created_at_utc": utc_now(),
        "owner_amendment": {
            "source": "Owner instruction on 2026-08-14",
            "data_or_label_changes_authorized": False,
            "supplement_controls_only": True,
            "four_gpu_parallel_execution_requested": True,
            "diagnostic_control_failures_are_non_blocking": True,
            "r4_c_authorization": (
                "Authorize R4-C when the already frozen deployable ARB_hat stack "
                "remains CI-positive over HC, positive in at least two tasks and "
                "two seeds, has no stable 3% task/seed harm, retains at least half "
                "the oracle-direct gain, preserves calibration/exact fallback, and "
                "all source and supplement action branches converge. Hidden-only "
                "and phase-matched shuffle ordering is reported but non-blocking."
            ),
        },
        "source": {
            "r4_b_root": str(SOURCE_ROOT),
            "r4_b_gate": str(SOURCE_GATE),
            "r4_b_gate_sha256": sha256_file(SOURCE_GATE),
            "r4_b_receipt": str(SOURCE_RECEIPT),
            "r4_b_receipt_sha256": sha256_file(SOURCE_RECEIPT),
            "test_paths_opened": 0,
        },
        "conditions": [
            "hc_hidden_only_direct",
            "phase_matched_row_shuffle_direct",
        ],
        "condition_contract": {
            "hc_hidden_only_direct": (
                "Same frozen HC, direct residual architecture, parameter count, "
                "initialization, sampler and budget as ARB_hat; normalized side "
                "features are zero and reliability is one, so the trainable branch "
                "can use HC hidden state but receives no ARB content."
            ),
            "phase_matched_row_shuffle_direct": (
                "Derange predicted ARB_hat and reliability rows only within the "
                "same task and one of the four frozen coarse phase bins; "
                "a singleton bin is deterministically merged into the nearest "
                "non-sparse phase bin. Preserve phase distribution while destroying "
                "current relation detail."
            ),
        },
        "seeds": {
            "phase_matched_row_shuffle": 1519703927,
            "statistics": 1768484041,
            "residual_seed_policy": "reuse all three frozen source R4-B init/sampler pairs",
        },
        "implementation": {
            "branch": "feat/ssc-v7-m3-r4-b-signal-first-supplement",
            "commit": implementation_commit,
            "script": str(script),
            "script_sha256": sha256_file(REPOSITORY / script),
            "test": str(test),
            "test_sha256": sha256_file(REPOSITORY / test),
        },
        "interpretation_policy": {
            "primary": (
                "The deployable ARB_hat residual stack is useful and safe to carry "
                "to a sealed R4-C test under the owner's early-exploration policy."
            ),
            "attribution": (
                "Hidden-only and phase-matched controls quantify residual capacity "
                "and relation-detail identifiability. They may narrow the ARB semantic "
                "claim but do not erase a positive deployable stack or block R4-C."
            ),
            "honesty": (
                "If ARB_hat does not beat hidden-only, state that ARB semantic "
                "increment is not isolated; never relabel generic residual gain as ARB gain."
            ),
        },
        "terminal_locks": {
            "sealed_test_generated": False,
            "r4_c_started": False,
            "test_paths_opened": 0,
            "m4_authorized": False,
            "b_core_authorized": False,
        },
    }
    gate["integrity"] = {
        "algorithm": "sha256-canonical-json-without-integrity",
        "payload_sha256": hashlib.sha256(canonical_bytes(gate)).hexdigest(),
    }
    write_json(OUTPUT, gate)
    write_json(
        OUTPUT.parent / "gate_receipt.json",
        {
            "decision_code": "SSC_V7_M3_R4_B_SUPPLEMENT_GATE_FROZEN",
            "gate": str(OUTPUT),
            "gate_sha256": sha256_file(OUTPUT),
            "test_paths_opened": 0,
            "sealed_test_generated": False,
        },
    )
    print("SSC_V7_M3_R4_B_SUPPLEMENT_GATE_FROZEN")


if __name__ == "__main__":
    main()
