#!/usr/bin/env python3
"""Validate and combine P0/P2 on-policy recovery shards for R12-R3."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import torch

from before_we_act.data.raw_team_windows import TASKS


OUTPUT_PROTOCOL = "r12r2_student_on_policy_w10_teacher_recovery_v1"
SHARD_PROTOCOL = "r12r2_student_on_policy_w10_teacher_recovery_shard_v1"
EXPECTED_TEACHER_SHA = "061b7a4acea8fa10f146779e7a1206822179920dfe573db536d237df81eb541d"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0", type=Path, required=True)
    parser.add_argument("--p2", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and args.receipt.exists():
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        if receipt.get("passed") and receipt.get("protocol_variant") == OUTPUT_PROTOCOL:
            print(json.dumps({"reused": str(args.output), "sha256": sha256(args.output)}))
            return
        raise ValueError("existing R12-R3 recovery receipt differs")
    seeds = json.loads(args.seed_manifest.read_text(encoding="utf-8"))
    if seeds.get("protocol") != "training_only_recovery_seeds_v1":
        raise ValueError("recovery seed manifest identity differs")
    if any(row.get("overlap") for row in seeds["tasks"].values()):
        raise ValueError("recovery seeds overlap Gate20")
    shards = {}
    for candidate, path in (("p0", args.p0), ("p2", args.p2)):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("protocol_variant") != SHARD_PROTOCOL or payload.get("candidate") != candidate:
            raise ValueError(f"invalid recovery shard {candidate}")
        metadata = payload["metadata"]
        if metadata.get("teacher_checkpoint_sha256") != EXPECTED_TEACHER_SHA:
            raise ValueError("recovery shard teacher checkpoint differs")
        if metadata.get("seed_manifest_sha256") != sha256(args.seed_manifest):
            raise ValueError("recovery shard seed manifest differs")
        if {episode["task"] for episode in metadata["episodes"]} != set(TASKS):
            raise ValueError("recovery shard does not cover all five tasks")
        shards[candidate] = payload
    keys = set(shards["p0"]["train"])
    if keys != set(shards["p2"]["train"]):
        raise ValueError("recovery shard tensor schema differs")
    combined = {
        key: torch.cat([shards["p0"]["train"][key], shards["p2"]["train"][key]], dim=0)
        for key in sorted(keys)
    }
    task_counts = {
        task: int(combined["task_index"].eq(index).sum())
        for index, task in enumerate(TASKS)
    }
    checks = {
        "two_independent_failed_student_sources": set(combined["source_policy"].tolist()) == {0, 2},
        "all_five_tasks_have_rows": all(value > 0 for value in task_counts.values()),
        "training_seeds_disjoint_from_gate20": all(not row["overlap"] for row in seeds["tasks"].values()),
        "frozen_w10_teacher_identity": all(
            payload["metadata"]["teacher_checkpoint_sha256"] == EXPECTED_TEACHER_SHA
            for payload in shards.values()
        ),
        "spatial_features_finite": bool(torch.isfinite(combined["spatial_tokens"]).all()),
        "teacher_actions_finite": bool(torch.isfinite(combined["joint_actions"]).all()),
        "history_is_lagged_student_execution": all(
            payload["metadata"]["legal_student_state"].endswith("lagged executed student action")
            for payload in shards.values()
        ),
        "teacher_absent_from_deployment": all(
            payload["metadata"]["teacher_usage"].startswith("offline training label only")
            for payload in shards.values()
        ),
    }
    payload = {
        "schema_version": 1,
        "round": "R12-R3",
        "metadata": {
            "created_at": now(),
            "protocol_variant": OUTPUT_PROTOCOL,
            "seed_manifest": str(args.seed_manifest.resolve()),
            "seed_manifest_sha256": sha256(args.seed_manifest),
            "student_sources": {
                candidate: {
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                    "checkpoint": shards[candidate]["metadata"]["student_checkpoint"],
                    "checkpoint_sha256": shards[candidate]["metadata"]["student_checkpoint_sha256"],
                }
                for candidate, path in (("p0", args.p0), ("p2", args.p2))
            },
            "teacher_checkpoint_sha256": EXPECTED_TEACHER_SHA,
            "rows": len(combined["task_index"]),
            "task_counts": task_counts,
            "recommended_sampling_probability": 0.35,
            "checks": checks,
        },
        "train": combined,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    receipt = {
        "schema_version": 1,
        "stage": "R12-R3-on-policy-recovery",
        "protocol_variant": OUTPUT_PROTOCOL,
        "created_at": now(),
        "output": str(args.output),
        "sha256": sha256(args.output),
        "rows": len(combined["task_index"]),
        "task_counts": task_counts,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    raise SystemExit(0 if receipt["passed"] else 10)


if __name__ == "__main__":
    main()
