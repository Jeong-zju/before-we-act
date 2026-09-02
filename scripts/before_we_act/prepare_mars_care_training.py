#!/usr/bin/env python3
"""Freeze MARS CARE families, quality sidecars, and all-family scorer data."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from before_we_act.care_training_data import SPLIT_IDS, atomic_json, family_targets, sha256_file
from before_we_act.mars_action_contract import (
    ACTION_CONTRACT_VERSION,
    action_contract_hash,
    normalization_stats_hash,
    validate_checkpoint_action_contract,
)
from before_we_act.mars_temporal_data import MARS_TASKS, load_mars_episodes, validate_mars_normalization
from deployment.mars_care.common import TASK_BY_NAME


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_rank(task: str, cache_key: str) -> str:
    return hashlib.sha256(f"care-mars-family-v1|{task}|{cache_key}".encode()).hexdigest()


def episode_seed(episode: Any) -> int:
    sidecar = Path(episode.path).with_suffix(".json")
    value = json.loads(sidecar.read_text(encoding="utf-8"))
    trajectory_index = int(str(episode.trajectory).rsplit("_", 1)[-1])
    rows = {int(row["episode_id"]): row for row in value["episodes"]}
    if trajectory_index not in rows:
        raise RuntimeError(f"missing episode metadata: {sidecar}:{episode.trajectory}")
    row = rows[trajectory_index]
    return int(row.get("episode_seed", row.get("reset_kwargs", {}).get("seed")))


def prepare_manifest(raw_root: Path, output: Path, families_per_task: int) -> dict[str, Any]:
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("format_version") != "before-we-act.care-mars-family-manifest/1":
            raise RuntimeError(f"refusing to reuse {output}")
        return existing
    if families_per_task != 30:
        raise ValueError("formal CARE migration keeps the official compact 30-family/task protocol")
    episodes = load_mars_episodes(raw_root)
    families: list[dict[str, Any]] = []
    for task in MARS_TASKS:
        selected = sorted(
            (episode for episode in episodes if episode.task == task),
            key=lambda row: stable_rank(task, row.cache_key),
        )[:families_per_task]
        if len(selected) != families_per_task:
            raise RuntimeError(f"not enough MARS episodes for {task}")
        spec = TASK_BY_NAME[task]
        for ordinal, episode in enumerate(selected):
            stratum = "critical" if ordinal < 20 else "uniform"
            # Critical snapshots emphasize transition-rich mid/late execution;
            # uniform snapshots cover the full non-terminal support.
            if stratum == "critical":
                phase = 0.35 + 0.45 * ((ordinal + 0.5) / 20.0)
            else:
                phase = 0.10 + 0.75 * ((ordinal - 20 + 0.5) / 10.0)
            maximum = max(1, min(int(episode.length) - 65, int(spec.max_steps) - 65))
            anchor = min(maximum, max(1, int(round(phase * maximum))))
            seed = episode_seed(episode)
            focal = ordinal % len(episode.arms)
            identity = hashlib.sha256(
                f"{task}|{seed}|{anchor}|{focal}|{stratum}".encode()
            ).hexdigest()
            families.append(
                {
                    "snapshot_id": identity,
                    "task": task,
                    "episode_seed": seed,
                    "anchor_step": anchor,
                    "focal_agent": focal,
                    "sampling_stratum": stratum,
                    "scenario_group_id": episode.cache_key,
                    "source_episode_path": episode.path,
                    "source_trajectory": episode.trajectory,
                    "source_episode_length": int(episode.length),
                }
            )
    result = {
        "format_version": "before-we-act.care-mars-family-manifest/1",
        "created_at_utc": utc_now(),
        "status": "FROZEN",
        "families_per_task": families_per_task,
        "family_count": len(families),
        "branches_per_family": 24,
        "sampling": "official compact 20 critical + 10 uniform per task",
        "families": families,
    }
    atomic_json(output, result)
    return result


def build_quality(family_root: Path, quality_root: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for task in MARS_TASKS:
        paths = sorted((family_root / task).glob("*.json"))
        for path in paths:
            family = json.loads(path.read_text(encoding="utf-8"))
            snapshot_id = str(family["snapshot_id"])
            if int(family.get("branch_count", -1)) != 24:
                raise RuntimeError(f"incomplete MARS CARE family: {snapshot_id}")
            if not all(bool(row["valid"]) for row in family["candidate_legality"]):
                raise RuntimeError(f"illegal MARS CARE family: {snapshot_id}")
            horizon_rows: dict[str, Any] = {}
            for horizon in (8, 16, 32, 64):
                usable = all(
                    str(horizon) in branch.get("outcomes", {})
                    and branch.get("candidate_valid") is True
                    and not str(branch.get("status", "")).startswith("SIMULATOR_FATAL")
                    for branch in family["branches"]
                )
                horizon_rows[str(horizon)] = {
                    "label": "USE" if usable else "DROP",
                    "reason": "complete paired reactive/replay support" if usable else "incomplete branch support",
                }
                counts[f"horizon_{horizon}_{'use' if usable else 'drop'}"] += 1
            quality = {
                "format_version": "before-we-act.care-mars-quality/1",
                "snapshot_id": snapshot_id,
                "source_family": str(path.resolve()),
                "source_family_sha256": sha256_file(path),
                "horizons": horizon_rows,
            }
            target = quality_root / task / f"{snapshot_id}.quality.json"
            if target.exists():
                existing = json.loads(target.read_text())
                if existing != quality:
                    raise RuntimeError(f"quality drift: {target}")
            else:
                atomic_json(target, quality)
            counts[task] += 1
    receipt = {
        "format_version": "before-we-act.care-mars-quality-receipt/1",
        "created_at_utc": utc_now(),
        "status": "PASSED",
        "counts": dict(counts),
    }
    atomic_json(quality_root / "receipt.json", receipt)
    return receipt


def discover_rows(
    family_root: Path,
    quality_root: Path,
    expected_families: int = 120,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in MARS_TASKS:
        for family_path in sorted((family_root / task).glob("*.json")):
            family = json.loads(family_path.read_text(encoding="utf-8"))
            snapshot_id = str(family["snapshot_id"])
            quality_path = quality_root / task / f"{snapshot_id}.quality.json"
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            if quality["source_family_sha256"] != sha256_file(family_path):
                raise RuntimeError(f"family/quality mismatch: {snapshot_id}")
            rows.append(
                {
                    "task": task,
                    "snapshot_id": snapshot_id,
                    "family_path": family_path,
                    "family": family,
                    "quality_path": quality_path,
                    "quality": quality,
                    "npz_path": family_path.with_suffix(".npz"),
                }
            )
    if len(rows) != expected_families:
        raise RuntimeError(
            f"MARS CARE expects {expected_families} families, got {len(rows)}"
        )
    return rows


def intervention_steps_for_rows(rows: list[Mapping[str, Any]]) -> int:
    """Return the one frozen branch execution window used by a corpus."""

    values = {
        int(row["family"].get("intervention_steps", -1)) for row in rows
    }
    if values != {1} and values != {4} and values != {8} and values != {16}:
        raise RuntimeError(
            "MARS CARE prepared corpus must contain one registered intervention "
            f"window, got {sorted(values)}"
        )
    return next(iter(values))


def prepare_data(
    family_root: Path,
    quality_root: Path,
    reference_checkpoint: Path,
    output: Path,
    manifest_output: Path,
    expected_families: int = 120,
) -> dict[str, Any]:
    if output.exists() or manifest_output.exists():
        raise RuntimeError("refusing to overwrite frozen MARS CARE prepared data")
    rows = discover_rows(family_root, quality_root, expected_families)
    intervention_steps = intervention_steps_for_rows(rows)
    memories: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    chunks: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    hard_safety: list[torch.Tensor] = []
    usable: list[torch.Tensor] = []
    task_ids: list[int] = []
    snapshot_ids: list[str] = []
    manifest_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        with np.load(row["npz_path"], allow_pickle=False) as values:
            memories.append(torch.from_numpy(np.asarray(values["memory"])).float())
            masks.append(torch.from_numpy(np.asarray(values["memory_mask"])).bool())
            chunks.append(torch.from_numpy(np.asarray(values["candidate_chunks"])).float())
        target, unsafe, use = family_targets(row["family"], row["quality"])
        if not bool(use.any()):
            raise RuntimeError(f"no usable horizon: {row['snapshot_id']}")
        targets.append(torch.from_numpy(target))
        hard_safety.append(torch.from_numpy(unsafe))
        usable.append(torch.from_numpy(use))
        task_ids.append(MARS_TASKS.index(row["task"]))
        snapshot_ids.append(row["snapshot_id"])
        manifest_rows.append(
            {
                "index": index,
                "snapshot_id": row["snapshot_id"],
                "task": row["task"],
                "family_sha256": sha256_file(row["family_path"]),
                "npz_sha256": sha256_file(row["npz_path"]),
                "quality_sha256": sha256_file(row["quality_path"]),
                "training_membership": "all",
                "diagnostic_membership": "all",
                "calibration_membership": "all",
            }
        )
    reference = torch.load(reference_checkpoint, map_location="cpu", weights_only=False)
    reference_contract = validate_checkpoint_action_contract(reference)
    validate_mars_normalization(reference.get("stats"))
    normalization_sha256 = normalization_stats_hash(reference["stats"])
    if reference_contract.get("annotations", {}).get("normalization_sha256") != normalization_sha256:
        raise ValueError("reference checkpoint action statistics hash differs")
    action_std = [float(value) for value in reference["stats"]["a_std"]]
    manifest = {
        "format_version": "before-we-act.care-mars-prepared-manifest/1",
        "created_at_utc": utc_now(),
        "source_family_count": len(rows),
        "tasks": list(MARS_TASKS),
        "policy_training_split": "all 600 demonstrations",
        "scorer_training_split": "all 120 branch families",
        "intervention_steps": intervention_steps,
        "branch_intervention_steps": intervention_steps,
        "diagnostic_and_calibration_note": "same branch corpus; Validation20 uses independent seeds",
        "reference_checkpoint": str(reference_checkpoint.resolve()),
        "reference_checkpoint_sha256": sha256_file(reference_checkpoint),
        "action_contract_version": ACTION_CONTRACT_VERSION,
        "action_contract_sha256": action_contract_hash(),
        "normalization_sha256": normalization_sha256,
        "action_std": action_std,
        "rows": manifest_rows,
    }
    atomic_json(manifest_output, manifest)
    prepared = {
        "format_version": "before-we-act.care-mars-prepared-data/1",
        "memory": torch.stack(memories).to(torch.float16),
        "memory_mask": torch.stack(masks),
        "candidate_chunks": torch.stack(chunks),
        "targets": torch.stack(targets),
        "hard_safety": torch.stack(hard_safety),
        "usable": torch.stack(usable),
        "split_id": torch.full((len(rows),), SPLIT_IDS["train"], dtype=torch.long),
        "task_id": torch.tensor(task_ids, dtype=torch.long),
        "snapshot_ids": snapshot_ids,
        "tasks": list(MARS_TASKS),
        "action_std": torch.tensor(action_std),
        "intervention_steps": intervention_steps,
        "manifest": {
            "path": str(manifest_output.resolve()),
            "sha256": sha256_file(manifest_output),
            "action_std": action_std,
            "intervention_steps": intervention_steps,
            "branch_intervention_steps": intervention_steps,
            "action_contract_version": ACTION_CONTRACT_VERSION,
            "action_contract_sha256": action_contract_hash(),
            "normalization_sha256": normalization_sha256,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(prepared, temporary)
    os.replace(temporary, output)
    receipt = {
        "status": "PASSED",
        "families": len(rows),
        "all_family_training": True,
        "intervention_steps": intervention_steps,
        "prepared_data": str(output.resolve()),
        "prepared_data_sha256": sha256_file(output),
    }
    atomic_json(output.with_suffix(".receipt.json"), receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--raw-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--families-per-task", type=int, default=30)
    quality = sub.add_parser("quality")
    quality.add_argument("--family-root", type=Path, required=True)
    quality.add_argument("--quality-root", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--family-root", type=Path, required=True)
    prepare.add_argument("--quality-root", type=Path, required=True)
    prepare.add_argument("--reference-checkpoint", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--manifest-output", type=Path, required=True)
    prepare.add_argument(
        "--expected-families",
        type=int,
        default=120,
        help="formal default is 120; smoke corpora may use a smaller exact count",
    )
    args = parser.parse_args()
    if args.command == "manifest":
        result = prepare_manifest(args.raw_root, args.output, args.families_per_task)
    elif args.command == "quality":
        result = build_quality(args.family_root, args.quality_root)
    else:
        result = prepare_data(
            args.family_root,
            args.quality_root,
            args.reference_checkpoint,
            args.output,
            args.manifest_output,
            args.expected_families,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
