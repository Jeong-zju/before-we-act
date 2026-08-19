#!/usr/bin/env python3
"""Freeze A6 partitions, derive legal B-core memory, and prepare CARE labels."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from two_three_task_manifest import get_task

from before_we_act.care_belief import CARE_HORIZONS
from before_we_act.care_training_data import (
    SPLIT_IDS,
    atomic_json,
    family_targets,
    sha256_file,
)
from before_we_act.evaluate_predictive_team_belief import load_team_belief
from before_we_act.temporal_history_data import (
    TASK_TEXT,
    TASK_TEXT_BYTES,
    task_text_tensor,
)


CONTRACT_STAGE = "A6R1-CARE-OWNER-AUTHORIZED-DIAGNOSTIC"
TASKS = (
    "lift_barrier",
    "camera_alignment",
    "long_pipeline_delivery",
    "take_photo",
    "pass_shoe",
    "place_food",
)
ALLOCATIONS = {
    "critical": {"train": 12, "validation": 3, "calibration": 2, "test": 3},
    "uniform": {"train": 6, "validation": 1, "calibration": 2, "test": 1},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def rank_key(task: str, stratum: str, snapshot_id: str) -> str:
    return hashlib.sha256(
        f"A6R1|frozen-secondary-split|{task}|{stratum}|{snapshot_id}".encode()
    ).hexdigest()


def freeze_partitions(rows: list[dict[str, Any]]) -> dict[str, str]:
    assignment: dict[str, str] = {}
    for task in TASKS:
        for stratum, allocation in ALLOCATIONS.items():
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["task"] == task and row["sampling_stratum"] == stratum
                ),
                key=lambda row: rank_key(task, stratum, row["snapshot_id"]),
            )
            if len(selected) != sum(allocation.values()):
                raise RuntimeError(
                    f"expected {sum(allocation.values())} {task}/{stratum} families, got {len(selected)}"
                )
            cursor = 0
            for split, count in allocation.items():
                for row in selected[cursor : cursor + count]:
                    assignment[row["snapshot_id"]] = split
                cursor += count
    if len(assignment) != len(rows):
        raise RuntimeError("CARE partition did not assign every family exactly once")
    return assignment


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("stage_id") != CONTRACT_STAGE:
        raise RuntimeError("wrong CARE A6 authorization contract")
    authorization = value.get("authorization", {})
    if not authorization.get("training_authorized", False):
        raise RuntimeError("CARE training is not authorized")
    if not authorization.get("a5r7_use_override_authorized", False):
        raise RuntimeError("A5R7 remains forbidden for CARE training")
    if authorization.get("gate_a_reclassified_as_passed", True):
        raise RuntimeError("A6 contract must preserve the Gate A failure")
    return value


def discover_rows(family_root: Path, quality_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        for family_path in sorted((family_root / task).glob("*.json")):
            family = json.loads(family_path.read_text(encoding="utf-8"))
            snapshot_id = str(family["snapshot_id"])
            quality_path = quality_root / task / f"{snapshot_id}.quality.json"
            if not quality_path.is_file():
                raise RuntimeError(f"quality sidecar missing: {quality_path}")
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            if quality["source_family_sha256"] != sha256_file(family_path):
                raise RuntimeError(f"quality/source hash mismatch: {snapshot_id}")
            if family.get("stage_id") != "A5R7-CARE-COMMON-SUPPORT-BRANCHES":
                raise RuntimeError(f"wrong A5R7 family stage: {snapshot_id}")
            if int(family.get("branch_count", -1)) != 24:
                raise RuntimeError(f"incomplete CARE family: {snapshot_id}")
            if not all(bool(row["valid"]) for row in family["candidate_legality"]):
                raise RuntimeError(f"illegal candidate in source family: {snapshot_id}")
            rows.append(
                {
                    "task": task,
                    "snapshot_id": snapshot_id,
                    "sampling_stratum": str(family["sampling_stratum"]),
                    "family_path": family_path,
                    "family": family,
                    "quality_path": quality_path,
                    "quality": quality,
                    "npz_path": family_path.with_suffix(".npz"),
                }
            )
    if len(rows) != 180:
        raise RuntimeError(f"A6 expects the immutable 180-family A5R7 corpus, got {len(rows)}")
    return rows


@torch.no_grad()
def frozen_memory(
    model: Any,
    row: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = np.load(row["npz_path"], allow_pickle=False)
    task = row["task"]
    arms = tuple(int(value) for value in get_task(task)["agents"])
    focal = int(values["focal_agent"][0])
    if focal not in arms:
        raise RuntimeError(f"focal arm is absent: {row['snapshot_id']}")
    visual = torch.as_tensor(values["history_visual_raw"], device=device)
    qpos = torch.as_tensor(values["history_qpos_normalized"], device=device)
    action = torch.as_tensor(values["history_action_normalized"], device=device)
    history_mask = torch.as_tensor(values["history_mask"], device=device).bool()
    action_mask = torch.as_tensor(values["action_history_mask"], device=device).bool()
    count = len(arms)
    text, text_mask = task_text_tensor(TASK_TEXT[task])
    text = text.unsqueeze(0).expand(count, TASK_TEXT_BYTES).to(device)
    text_mask = text_mask.unsqueeze(0).expand(count, TASK_TEXT_BYTES).to(device)
    task_token = model._task_token(text, text_mask)
    reset = model._window_reset_mask(history_mask)
    runtime_visual = visual.unsqueeze(-2)
    runtime_mask = history_mask[:, :, None, None].expand(
        -1, -1, runtime_visual.shape[2], 1
    )
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        belief = model.belief_core(
            runtime_visual,
            runtime_mask,
            qpos,
            action,
            history_mask,
            action_mask,
            task_token,
            reset,
        )
    focal_index = arms.index(focal)
    memory = torch.cat((belief.mu, belief.event_memory), dim=1)[focal_index]
    memory_mask = torch.cat(
        (
            torch.ones(
                belief.mu.shape[:2], dtype=torch.bool, device=belief.mu.device
            ),
            belief.event_mask,
        ),
        dim=1,
    )[focal_index]
    candidates = torch.as_tensor(values["candidate_chunks"]).float()
    return memory.float().cpu(), memory_mask.cpu(), candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise RuntimeError("refusing to overwrite frozen prepared CARE artifacts")
    contract = load_contract(args.contract)
    expected_reference = contract["reference_policy"]
    reference_sha = sha256_file(args.reference_checkpoint)
    if reference_sha != expected_reference["sha256"]:
        raise RuntimeError("frozen B-core checkpoint hash drifted")
    source = contract["source_data"]
    if sha256_file(source["collection_receipt"]["path"]) != source["collection_receipt"]["sha256"]:
        raise RuntimeError("A5R7 collection receipt drifted")
    if sha256_file(source["quality_summary"]["path"]) != source["quality_summary"]["sha256"]:
        raise RuntimeError("A5R7 quality summary drifted")

    rows = discover_rows(args.family_root, args.quality_root)
    assignment = freeze_partitions(rows)
    device = torch.device(args.device)
    model, stats, _config = load_team_belief(str(args.reference_checkpoint), device)
    model.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("B-core was not frozen before CARE feature extraction")

    memories: list[torch.Tensor] = []
    memory_masks: list[torch.Tensor] = []
    candidates: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    hard_safety: list[torch.Tensor] = []
    usable: list[torch.Tensor] = []
    task_ids: list[int] = []
    snapshot_ids: list[str] = []
    manifest_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        memory, mask, chunks = frozen_memory(model, row, device)
        target, unsafe, use = family_targets(row["family"], row["quality"])
        memories.append(memory.to(torch.float16))
        memory_masks.append(mask)
        candidates.append(chunks)
        targets.append(torch.from_numpy(target))
        hard_safety.append(torch.from_numpy(unsafe))
        usable.append(torch.from_numpy(use))
        task_ids.append(TASKS.index(row["task"]))
        snapshot_ids.append(row["snapshot_id"])
        manifest_rows.append(
            {
                "index": index,
                "snapshot_id": row["snapshot_id"],
                "task": row["task"],
                "sampling_stratum": row["sampling_stratum"],
                "secondary_split": assignment[row["snapshot_id"]],
                "family_path": str(row["family_path"].resolve()),
                "family_sha256": sha256_file(row["family_path"]),
                "npz_path": str(row["npz_path"].resolve()),
                "npz_sha256": sha256_file(row["npz_path"]),
                "quality_path": str(row["quality_path"].resolve()),
                "quality_sha256": sha256_file(row["quality_path"]),
                "usable_horizons": [
                    horizon for horizon, keep in zip(CARE_HORIZONS, use.tolist()) if keep
                ],
            }
        )
        if (index + 1) % 20 == 0:
            print(json.dumps({"prepared_families": index + 1}), flush=True)

    manifest = {
        "format_version": "before-we-act.a6r1-care-prepared-manifest/1",
        "stage_id": CONTRACT_STAGE,
        "created_at_utc": utc_now(),
        "contract": str(args.contract.resolve()),
        "contract_sha256": sha256_file(args.contract),
        "reference_checkpoint": str(args.reference_checkpoint.resolve()),
        "reference_checkpoint_sha256": reference_sha,
        "source_family_count": len(rows),
        "split_rule": "exact task/stratum quotas after frozen SHA256 rank",
        "split_counts": dict(Counter(assignment.values())),
        "task_split_counts": {
            task: dict(
                Counter(
                    assignment[row["snapshot_id"]]
                    for row in rows
                    if row["task"] == task
                )
            )
            for task in TASKS
        },
        "gate_a_preserved_as_not_passed": True,
        "a5r7_original_use_for_training_flags_modified": False,
        "authorization_scope": "owner-authorized diagnostic training only",
        "rows": manifest_rows,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.manifest, manifest)
    prepared = {
        "format_version": "before-we-act.a6r1-care-prepared-data/1",
        "memory": torch.stack(memories),
        "memory_mask": torch.stack(memory_masks),
        "candidate_chunks": torch.stack(candidates),
        "targets": torch.stack(targets),
        "hard_safety": torch.stack(hard_safety),
        "usable": torch.stack(usable),
        "split_id": torch.tensor(
            [SPLIT_IDS[assignment[value]] for value in snapshot_ids], dtype=torch.long
        ),
        "task_id": torch.tensor(task_ids, dtype=torch.long),
        "snapshot_ids": snapshot_ids,
        "tasks": list(TASKS),
        "action_std": stats["a_std"].detach().float().cpu(),
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256_file(args.manifest),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    torch.save(prepared, temporary)
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "status": "A6R1_PREPARED",
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "split_counts": manifest["split_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
