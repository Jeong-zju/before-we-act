#!/usr/bin/env python3
"""Freeze the formal successor gate after fresh confirmation data are complete."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess


REPOSITORY = Path(__file__).resolve().parents[2]
PREPARATION_GATE = (
    REPOSITORY / "docs/experiments/ssc_v7/m3_r4_successor_preparation_gate.json"
)
RUN_ROOT = Path(
    "/workspace/bwa_runs/ssc-v7-social-state-cooperation-v2/"
    "measurement/m3_r4_successor_a1_v1"
)
CONFIRMATION_MANIFEST = RUN_ROOT / "confirmation_data/confirmation_manifest.json"
OUTPUT = RUN_ROOT / "frozen_gate/m3_r4_successor_a1_a2_gate.json"


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"formal gate already exists: {OUTPUT}")
    preparation = json.loads(PREPARATION_GATE.read_text(encoding="utf-8"))
    confirmation = json.loads(CONFIRMATION_MANIFEST.read_text(encoding="utf-8"))
    if confirmation.get("read_only_test_used") is not False:
        raise RuntimeError("confirmation manifest did not preserve the test lock")
    if confirmation.get("existing_tune_used") is not False:
        raise RuntimeError("confirmation manifest used an inspected tune split")
    if int(confirmation.get("test_paths_opened", -1)) != 0:
        raise RuntimeError("confirmation manifest opened a test path")
    per_task = {
        str(item["task"]): sum(
            str(other["task"]) == str(item["task"])
            for other in confirmation["episodes"]
        )
        for item in confirmation["episodes"]
    }
    expected = int(
        preparation["confirmation_collection"]["successful_episodes_per_task"]
    )
    if set(per_task) != set(preparation["confirmation_collection"]["first_unused_candidate_index_by_task"]):
        raise RuntimeError("confirmation task set mismatch")
    if any(count != expected for count in per_task.values()):
        raise RuntimeError(f"confirmation per-task count mismatch: {per_task}")

    gate = deepcopy(preparation)
    gate["schema_version"] = 2
    gate["status"] = "FROZEN_SUCCESSOR_EXECUTABLE_BEFORE_CONFIRMATION_METRICS"
    gate["created_at"] = subprocess.check_output(
        ("date", "--iso-8601=seconds"), text=True
    ).strip()
    gate["data"]["confirmation_manifest"] = str(CONFIRMATION_MANIFEST)
    gate["data"]["confirmation_manifest_sha256"] = sha256_file(
        CONFIRMATION_MANIFEST
    )
    gate["data"]["confirmation_episode_count"] = len(confirmation["episodes"])
    gate["data"]["confirmation_episode_count_by_task"] = dict(sorted(per_task.items()))
    gate["data"]["confirmation_manifest_preparation_gate_sha256"] = str(
        confirmation["preparation_gate_sha256"]
    )
    gate["implementation"]["implementation_commit"] = subprocess.check_output(
        ("git", "-C", str(REPOSITORY), "rev-parse", "HEAD"), text=True
    ).strip()
    gate["run_root"] = str(RUN_ROOT)
    gate["terminal_locks"].update(
        {
            "r4_b_started": False,
            "test_paths_opened": 0,
            "m4_authorized": False,
            "b_core_authorized": False,
        }
    )
    unsigned = {key: value for key, value in gate.items() if key != "integrity"}
    gate["integrity"] = {
        "payload_sha256": hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, OUTPUT)
    receipt = {
        "decision_code": "SSC_V7_M3_R4_SUCCESSOR_FORMAL_GATE_FROZEN",
        "gate": str(OUTPUT),
        "gate_sha256": sha256_file(OUTPUT),
        "payload_sha256": gate["integrity"]["payload_sha256"],
        "confirmation_manifest_sha256": gate["data"][
            "confirmation_manifest_sha256"
        ],
        "implementation_commit": gate["implementation"]["implementation_commit"],
        "r4_b_started": False,
        "test_paths_opened": 0,
    }
    receipt_path = OUTPUT.parent / "formal_gate_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(receipt["decision_code"])


if __name__ == "__main__":
    main()
