#!/usr/bin/env python3
"""Freeze the owner-authorized SSC-V7 M3-R4-B executable gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


REPOSITORY = Path(__file__).resolve().parents[2]
STAGE_ID = "SSC-V7-M3-R4-B-OBSERVABILITY"
STATUS = "FROZEN_R4_B_BEFORE_PREDICTOR_OR_ACTION_METRICS"
SUCCESSOR_ROOT = Path(
    "/workspace/bwa_runs/ssc-v7-social-state-cooperation-v2/measurement/"
    "m3_r4_successor_a1_v1"
)
ACTIVE_HEADS = list(range(24)) + list(range(27, 32)) + list(range(32, 39))
CONSTANT_HEAD_VALUES = {
    24: 0.0,
    25: 0.0,
    26: 0.0,
    39: 1.0,
    40: 1.0,
    41: 1.0,
    42: 1.0,
    43: 1.0,
    44: 1.0,
    45: 1.0,
    46: 1.0,
    47: 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


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


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def seed(role: str) -> int:
    payload = f"SSC-V7-M3-R4-B-V1|2026-08-13|{role}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def main() -> None:
    args = parse_args()
    if subprocess.check_output(
        ("git", "-C", str(REPOSITORY), "status", "--porcelain"), text=True
    ).strip():
        raise RuntimeError("freeze requires a clean repository")
    branch = subprocess.check_output(
        ("git", "-C", str(REPOSITORY), "branch", "--show-current"), text=True
    ).strip()
    if branch != "feat/ssc-v7-m3-r4-b-observability":
        raise RuntimeError(f"wrong R4-B branch: {branch}")
    commit = subprocess.check_output(
        ("git", "-C", str(REPOSITORY), "rev-parse", "HEAD"), text=True
    ).strip()
    gate_path = args.output_root / "frozen_gate" / "m3_r4_b_gate.json"
    if gate_path.exists():
        raise FileExistsError(gate_path)
    forbidden_existing = (
        args.output_root / "predictors",
        args.output_root / "formal",
        args.output_root / "prediction_cache",
    )
    if any(path.exists() for path in forbidden_existing):
        raise RuntimeError("R4-B result artifacts exist before gate freeze")

    successor_gate_path = SUCCESSOR_ROOT / "frozen_gate" / "m3_r4_successor_a1_a2_gate.json"
    successor_receipt_path = SUCCESSOR_ROOT / "formal" / "successor_a1_a2_receipt.json"
    cache_receipt_path = SUCCESSOR_ROOT / "formal_cache" / "cache_receipt.json"
    hc_receipt_path = SUCCESSOR_ROOT / "formal" / "hc" / "hc_receipt.json"
    successor_gate = read_json(successor_gate_path)
    successor_receipt = read_json(successor_receipt_path)
    if successor_receipt["a1"]["formal_decision_code"] != (
        "PASSED_M3_R4_A1_CONFIRMED_ORACLE_UTILITY"
    ):
        raise RuntimeError("R4-B cannot freeze before successor A1 passes")
    if successor_receipt.get("test_paths_opened") != 0:
        raise RuntimeError("source successor opened a test path")

    script = Path("scripts/before_we_act/run_ssc_v7_m3_r4_b.py")
    test = Path("tests/before_we_act/test_ssc_v7_m3_r4_b.py")
    gate: dict[str, Any] = {
        "schema_version": 1,
        "stage_id": STAGE_ID,
        "status": STATUS,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "authorization": {
            "source": "Owner instruction on 2026-08-13",
            "scope": "Execute roadmap M3-R4-B after A1/A2 showed ARB content plus direct residual is effective.",
            "signal_first_addendum": "ARB_hat may lose oracle utility; early research prioritizes a real positive trend over HC while strict numbers only constrain formal pass and R4-C authorization.",
        },
        "scope": {
            "authorized": [
                "episode-out-of-fold ARB_hat prediction from legal current observation plus 16-step history",
                "held-out confirmation calibration and action evaluation",
                "direct residual branches and frozen destructive controls",
            ],
            "forbidden": [
                "read or generate sealed R4-C test",
                "query-attention fusion",
                "recurrent or cross-episode memory",
                "M4",
                "M5",
                "B-core",
                "change ARB schema after seeing predictor metrics",
            ],
        },
        "source": {
            "successor_root": str(SUCCESSOR_ROOT),
            "successor_gate": str(successor_gate_path),
            "successor_gate_sha256": sha256_file(successor_gate_path),
            "successor_receipt": str(successor_receipt_path),
            "successor_receipt_sha256": sha256_file(successor_receipt_path),
            "cache_receipt": str(cache_receipt_path),
            "cache_receipt_sha256": sha256_file(cache_receipt_path),
            "hc_receipt": str(hc_receipt_path),
            "hc_receipt_sha256": sha256_file(hc_receipt_path),
            "hc_noise_seed": int(successor_gate["seeds"]["hc_noise_seed"]),
            "training_split": "train",
            "evaluation_split": "fresh successor confirmation",
            "episode_overlap_count": 0,
            "existing_tune_used": False,
            "read_only_test_used": False,
        },
        "schema": {
            "token_names": [
                "contact_grasp_custody",
                "handoff_event",
                "teammate_motion_state",
                "blocking_collision",
                "visibility_staleness",
                "uncertainty_missingness",
            ],
            "token_count": 6,
            "token_width": 8,
            "active_head_indices": ACTIVE_HEADS,
            "constant_head_values": CONSTANT_HEAD_VALUES,
            "active_head_rule": "Predict the 36 train-variable relation/event fields; deterministic schema validity/currentness fields are filled with their frozen constants and are not credited as predicted heads.",
        },
        "architecture": {
            "candidate_predictor": "LayerNorm legal[1704] -> MLP 384 -> 192 -> 48 Bernoulli logits",
            "candidate_inputs": "current global/local compact RGB, prior 15 RGB frames, 16 qpos/action history and task text already frozen as legal input",
            "forbidden_predictor_inputs": [
                "oracle sidecar",
                "frame index",
                "episode ID",
                "agent slot metadata",
                "future action",
                "old social vector",
            ],
            "probability_calibration": "train-only per-head shrinkage toward train incidence selected on episode-held-out predictions",
            "reliability": "one minus mean Bernoulli uncertainty over active heads; unavailable input forces reliability zero",
            "action_fusion": "A1 frozen HC plus zero-init direct residual MLP times reliability and learned gate",
        },
        "conditions": [
            "arb_hat_direct",
            "row_shuffled_direct",
            "time_only_direct",
            "episode_shuffled_direct",
            "stale_8_direct",
            "stale_16_direct",
        ],
        "predictor_training": {
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 512,
            "hidden_width": 384,
            "max_epochs": 260,
            "patience": 25,
            "selection_metric": "episode-held-out mean Brier over 36 active heads",
            "outer_folds": 3,
        },
        "action_training": {
            "optimizer": "AdamW",
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "batch_size": 512,
            "max_epochs": 1200,
            "patience": 40,
            "selection_metric": "confirmation six-task macro future-16 NRMSE",
            "residual_seeds": 3,
        },
        "acceptance": {
            "oracle_gain_retention_min": 0.5,
            "paired_95pct_ci_lower_gt": 0.0,
            "positive_tasks_min": 2,
            "stable_task_harm_threshold_abs": 0.03,
            "controls_that_candidate_must_beat": [
                "row_shuffled_direct",
                "time_only_direct",
                "episode_shuffled_direct",
                "stale_8_direct",
                "stale_16_direct",
            ],
            "calibration": "every active head Brier below train-incidence constant on confirmation and error rises as reliability falls",
            "signal_first": "Always report HC gain, task/seed directions, control ordering and calibrated-head count even if a strict threshold misses.",
        },
        "interpretation_policy": {
            "primary": "Does legal-input ARB_hat plus direct residual show useful progress over frozen HC?",
            "secondary": "How much oracle-direct action gain survives and which controls localize remaining shortcuts?",
            "anti_overreaction": "A strict calibration/control miss blocks R4-C but does not erase a positive deployable ARB_hat trend.",
        },
        "seeds": {
            "legal_predictor": {
                "initialization": seed("legal-predictor-init"),
                "sampler": seed("legal-predictor-sampler"),
            },
            "time_only_predictor": {
                "initialization": seed("time-predictor-init"),
                "sampler": seed("time-predictor-sampler"),
            },
            "residual_initialization": [
                seed(f"residual-init-{index}") for index in range(3)
            ],
            "residual_sampler": [
                seed(f"residual-sampler-{index}") for index in range(3)
            ],
            "row_shuffle": seed("row-shuffle"),
            "episode_shuffle": seed("episode-shuffle"),
            "statistics": seed("statistics"),
        },
        "implementation": {
            "branch": branch,
            "implementation_commit": commit,
            "script": str(script),
            "script_sha256": sha256_file(REPOSITORY / script),
            "test": str(test),
            "test_sha256": sha256_file(REPOSITORY / test),
        },
        "terminal_locks": {
            "test_paths_opened": 0,
            "sealed_test_generated": False,
            "r4_c_started": False,
            "m4_authorized": False,
            "b_core_authorized": False,
        },
    }
    gate["integrity"] = {
        "payload_sha256": hashlib.sha256(canonical_bytes(gate)).hexdigest()
    }
    write_json(gate_path, gate)
    write_json(
        gate_path.with_name("gate_receipt.json"),
        {
            "decision_code": "SSC_V7_M3_R4_B_GATE_FROZEN",
            "gate": str(gate_path),
            "gate_sha256": sha256_file(gate_path),
            "implementation_commit": commit,
            "test_paths_opened": 0,
            "sealed_test_generated": False,
        },
    )
    print("SSC_V7_M3_R4_B_GATE_FROZEN")


if __name__ == "__main__":
    main()
