#!/usr/bin/env python3
"""Freeze R4-C code, checkpoints, seeds and policy before test generation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts.before_we_act import audit_ssc_v7_m2 as m2  # noqa: E402

STAGE_ID = "SSC-V7-M3-R4-C-SEALED-TEST"
RUN_ROOT = Path(
    "/workspace/bwa_runs/ssc-v7-social-state-cooperation-v2/measurement/"
    "m3_r4_c_sealed_test_v1"
)
R4B_ROOT = Path(
    "/workspace/bwa_runs/ssc-v7-social-state-cooperation-v2/measurement/"
    "m3_r4_b_observability_v1"
)
SUPPLEMENT_ROOT = Path(
    "/workspace/bwa_runs/ssc-v7-social-state-cooperation-v2/measurement/"
    "m3_r4_b_supplement_v1"
)
SUCCESSOR_ROOT = Path(
    "/workspace/bwa_runs/ssc-v7-social-state-cooperation-v2/measurement/"
    "m3_r4_successor_a1_v1"
)
R4B_GATE = R4B_ROOT / "frozen_gate/m3_r4_b_gate.json"
R4B_RECEIPT = R4B_ROOT / "formal/r4_b_observability_receipt.json"
SUPPLEMENT_GATE = SUPPLEMENT_ROOT / "frozen_gate/m3_r4_b_supplement_gate.json"
SUPPLEMENT_RECEIPT = SUPPLEMENT_ROOT / "formal/r4_b_supplement_receipt.json"
SUCCESSOR_GATE = SUCCESSOR_ROOT / "frozen_gate/m3_r4_successor_a1_a2_gate.json"
SEED_CONTRACT = Path(
    "/workspace/bwa_runs/ssc-v7-social-state-cooperation-v2/"
    "pre_registration/contracts/seed_contract.json"
)
W10_SEED_ROOT = Path("/workspace/bwa_runs/w10-six-task-v1/seeds/validation")
OUTPUT = RUN_ROOT / "frozen_gate/m3_r4_c_gate.json"
TASKS = (
    "lift_barrier",
    "camera_alignment",
    "long_pipeline_delivery",
    "take_photo",
    "pass_shoe",
    "place_food",
)
SOURCE_CONDITIONS = (
    "arb_hat_direct",
    "row_shuffled_direct",
    "time_only_direct",
    "episode_shuffled_direct",
    "stale_8_direct",
    "stale_16_direct",
)
SUPPLEMENT_CONDITIONS = (
    "hc_hidden_only_direct",
    "phase_matched_row_shuffle_direct",
)


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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def branch_checkpoint(root: Path, condition: str, seed_index: int) -> dict[str, Any]:
    receipt_path = (
        root / "formal/branches" / condition / f"seed_{seed_index}/branch_receipt.json"
    )
    receipt = read_json(receipt_path)
    checkpoint = Path(str(receipt["checkpoint"]))
    if sha256_file(checkpoint) != str(receipt["checkpoint_sha256"]):
        raise RuntimeError(f"checkpoint mismatch: {checkpoint}")
    return {
        "condition": condition,
        "seed_index": seed_index,
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "source_receipt": str(receipt_path),
        "source_receipt_sha256": sha256_file(receipt_path),
    }


def next_candidate_indices() -> dict[str, int]:
    result: dict[str, int] = {}
    for task in TASKS:
        path = SUCCESSOR_ROOT / f"confirmation_collections/{task}/task_collection_receipt.json"
        receipt = read_json(path)
        result[task] = 1 + max(int(item["candidate_index"]) for item in receipt["attempts"])
    return result


def main() -> None:
    if RUN_ROOT.exists():
        raise FileExistsError(f"fresh R4-C root required: {RUN_ROOT}")
    supplement_receipt = read_json(SUPPLEMENT_RECEIPT)
    if not supplement_receipt.get("r4_c_authorized"):
        raise RuntimeError("R4-B supplement did not authorize R4-C")
    if int(supplement_receipt.get("test_paths_opened", -1)) != 0:
        raise RuntimeError("test boundary was already opened")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPOSITORY, text=True
    ).strip():
        raise RuntimeError("freeze requires a clean worktree")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()

    successor_gate = read_json(SUCCESSOR_GATE)
    cache_receipt_path = SUCCESSOR_ROOT / "formal_cache/cache_receipt.json"
    cache_receipt = read_json(cache_receipt_path)
    normalization = Path(str(cache_receipt["normalization"]))
    training_manifest = Path(str(successor_gate["data"]["training_manifest"]))
    confirmation_manifest = Path(str(successor_gate["data"]["confirmation_manifest"]))
    r4b_gate = read_json(R4B_GATE)
    predictor_receipt_path = R4B_ROOT / "predictors/predictor_receipt.json"
    predictor_receipt = read_json(predictor_receipt_path)
    predictor_checkpoints: list[dict[str, Any]] = []
    for kind, key in (("legal", "candidate"), ("time_only", "time_only_control")):
        details = predictor_receipt[key]
        checkpoint = Path(str(details["checkpoint"]))
        if sha256_file(checkpoint) != str(details["checkpoint_sha256"]):
            raise RuntimeError(f"predictor checkpoint mismatch: {checkpoint}")
        predictor_checkpoints.append(
            {"kind": kind, "path": str(checkpoint), "sha256": sha256_file(checkpoint)}
        )

    action_checkpoints: list[dict[str, Any]] = []
    for condition in SOURCE_CONDITIONS:
        for seed_index in range(3):
            action_checkpoints.append(branch_checkpoint(R4B_ROOT, condition, seed_index))
    for condition in SUPPLEMENT_CONDITIONS:
        for seed_index in range(3):
            action_checkpoints.append(
                branch_checkpoint(SUPPLEMENT_ROOT, condition, seed_index)
            )
    for seed_index in range(3):
        item = branch_checkpoint(SUCCESSOR_ROOT, "arb_direct", seed_index)
        item["condition"] = "oracle_direct"
        action_checkpoints.append(item)

    hc_receipt_path = Path(str(r4b_gate["source"]["hc_receipt"]))
    hc_receipt = read_json(hc_receipt_path)
    hc_checkpoint = Path(str(hc_receipt["checkpoint"]))
    frozen_paths = [
        R4B_GATE,
        R4B_RECEIPT,
        SUPPLEMENT_GATE,
        SUPPLEMENT_RECEIPT,
        SUCCESSOR_GATE,
        cache_receipt_path,
        normalization,
        training_manifest,
        confirmation_manifest,
        SEED_CONTRACT,
        predictor_receipt_path,
        hc_receipt_path,
        hc_checkpoint,
    ]
    frozen_paths.extend(Path(str(item["path"])) for item in predictor_checkpoints)
    frozen_paths.extend(Path(str(item["path"])) for item in action_checkpoints)
    frozen_paths.extend(Path(str(item["source_receipt"])) for item in action_checkpoints)
    unique_paths = list(dict.fromkeys(frozen_paths))
    first_candidates = next_candidate_indices()
    expanded_seeds = m2.expanded_seed_manifest(
        read_json(SEED_CONTRACT), W10_SEED_ROOT
    )
    candidate_seeds = {
        task: [
            int(value)
            for value in expanded_seeds["per_task"][task]["expert_candidate_pool"][
                first_candidates[task] :
            ]
        ]
        for task in TASKS
    }

    script = Path("scripts/before_we_act/run_ssc_v7_m3_r4_c.py")
    test = Path("tests/before_we_act/test_ssc_v7_m3_r4_c.py")
    gate: dict[str, Any] = {
        "schema_version": "ssc-v7.m3_r4_c.gate/1",
        "stage_id": STAGE_ID,
        "status": "FROZEN_R4_C_BEFORE_TEST_GENERATION",
        "created_at_utc": utc_now(),
        "implementation": {
            "branch": "feat/ssc-v7-m3-r4-c-signal-first-sealed-test",
            "commit": commit,
            "script": str(script),
            "script_sha256": sha256_file(REPOSITORY / script),
            "test": str(test),
            "test_sha256": sha256_file(REPOSITORY / test),
        },
        "source": {
            "r4_b_root": str(R4B_ROOT),
            "r4_b_gate": str(R4B_GATE),
            "r4_b_receipt": str(R4B_RECEIPT),
            "supplement_root": str(SUPPLEMENT_ROOT),
            "supplement_gate": str(SUPPLEMENT_GATE),
            "supplement_receipt": str(SUPPLEMENT_RECEIPT),
            "successor_root": str(SUCCESSOR_ROOT),
            "hc_noise_seed": int(r4b_gate["source"]["hc_noise_seed"]),
            "test_paths_opened": 0,
        },
        "data": {
            "training_manifest": str(training_manifest),
            "training_manifest_sha256": sha256_file(training_manifest),
            "confirmation_manifest": str(confirmation_manifest),
            "confirmation_manifest_sha256": sha256_file(confirmation_manifest),
            "normalization": str(normalization),
            "normalization_sha256": sha256_file(normalization),
            "seed_contract": str(SEED_CONTRACT),
            "seed_contract_sha256": sha256_file(SEED_CONTRACT),
        },
        "test_collection": {
            "purpose": "fresh_sealed_r4_c_test",
            "successful_episodes_per_task": 12,
            "first_unused_candidate_index_by_task": first_candidates,
            "candidate_seeds_by_task": candidate_seeds,
            "candidate_rule": (
                "Use the frozen per-task expert pool sequentially from the first "
                "index after every R4-A1 confirmation attempt, stopping at twelve "
                "successes. No train/confirmation seed or HDF5 identity may overlap."
            ),
        },
        "test_manifest_path": str(RUN_ROOT / "sealed_test/test_manifest.json"),
        "predictor_checkpoints": predictor_checkpoints,
        "action_checkpoints": action_checkpoints,
        "frozen_artifacts": [artifact(path) for path in unique_paths],
        "statistics_seed": 1807054843,
        "acceptance": {
            "positive_tasks_min": 2,
            "stable_task_harm_threshold_abs": 0.03,
            "oracle_gain_retention_min": 0.50,
        },
        "owner_amendment": {
            "source": "Owner instruction on 2026-08-14",
            "data_or_label_changes_authorized": False,
            "checkpoint_retraining_authorized": False,
            "diagnostic_control_failures_are_non_blocking": True,
            "one_time_fresh_sealed_test": True,
        },
        "interpretation_policy": {
            "primary": (
                "R4-C asks whether the frozen deployable ARB_hat + direct residual "
                "stack retains a positive, safe and calibrated trend over HC on a "
                "fresh one-time test. Passing authorizes M4, not B-core."
            ),
            "attribution": (
                "Hidden-only, phase-matched, row/time/episode shuffle and stale "
                "comparisons are reported but non-blocking. If ARB_hat does not beat "
                "them, state that ARB semantic increment remains unisolated."
            ),
            "calibration": (
                "Require mean active-head Brier to beat the frozen train-rate constant "
                "and reliability bins to remain directional; report per-head counts "
                "without using every individual head as an early-exploration veto."
            ),
        },
        "terminal_locks": {
            "test_generated": False,
            "test_open_events": 0,
            "evaluation_retry_authorized": False,
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
            "decision_code": "SSC_V7_M3_R4_C_GATE_FROZEN_BEFORE_TEST_GENERATION",
            "gate": str(OUTPUT),
            "gate_sha256": sha256_file(OUTPUT),
            "test_generated": False,
            "test_open_events": 0,
        },
    )
    print("SSC_V7_M3_R4_C_GATE_FROZEN_BEFORE_TEST_GENERATION")


if __name__ == "__main__":
    main()
