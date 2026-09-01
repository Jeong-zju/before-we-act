"""Pack branch-family shards into the CARE training tensor contract."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .config import TASKS
from .bcore_data import BICOORD_CARE_MEMORY_TOKENS, BICOORD_CARE_MEMORY_WIDTH
from .branch_fidelity import (
    physical_branch_family_rows_valid,
    seed_replay_probe_valid,
    strict_fidelity_receipts_valid,
)
from .data import load_normalization_receipt
from .stage_common import artifact, assert_common_paths, atomic_json, common_parser, publish_result, require_stage_result, sha256_file


def _resolve_regular(path: Path, *, root: Path | None = None) -> Path:
    """Resolve an evidence path while rejecting symlinked parents."""

    path = path.expanduser()
    if root is not None:
        root = root.expanduser().resolve(strict=True)
        try:
            lexical_path = path if path.is_absolute() else Path.cwd() / path
            lexical = lexical_path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"branch evidence escapes namespace: {path}") from error
        cursor = root
        for part in lexical.parts:
            cursor /= part
            if cursor.is_symlink():
                raise ValueError(f"branch evidence has symbolic component: {cursor}")
    if path.is_symlink():
        raise ValueError(f"branch evidence is symbolic: {path}")
    resolved = path.resolve(strict=True)
    if root is not None:
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(f"branch evidence resolves outside namespace: {path}") from error
    return resolved


def _branch_manifests(run: Path, *, smoke: bool = False) -> list[Path]:
    # Never mix the 18 smoke fixtures into the 540-family formal tensor.
    roots = [run / "artifacts" / ("branch_smoke" if smoke else "branches")]
    result: list[Path] = []
    for root in roots:
        result.extend(sorted(root.glob("rank_*/manifest.json")))
        if (root / "manifest.json").is_file(): result.append(root / "manifest.json")
    return list(dict.fromkeys(result))


def _records(
    paths: list[Path], *, smoke: bool | None = None, root: Path | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    canonical_root = root.expanduser().resolve() if root is not None else None
    for path in paths:
        path = _resolve_regular(path, root=canonical_root)
        if canonical_root is not None and canonical_root not in path.parents:
            raise ValueError(f"branch shard manifest escapes its namespace: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "PASSED" or value.get("provider_policy") != "B-core/TUNE":
            raise ValueError(f"invalid branch shard manifest: {path}")
        if smoke is not None and value.get("smoke") is not bool(smoke):
            raise ValueError(
                f"branch shard smoke/formal provenance differs at {path}: "
                f"{value.get('smoke')!r} != {bool(smoke)!r}"
            )
        if value.get("physical_simulator_outcomes") is not True or value.get("offline_demonstration_error_used") is not False:
            raise ValueError(f"branch shard lacks physical-only provenance: {path}")
        for row in value.get("records", []):
            if not isinstance(row, Mapping): raise ValueError("branch manifest record is not an object")
            npz = Path(str(row.get("npz", ""))).expanduser()
            manifest = Path(str(row.get("manifest", ""))).expanduser()
            if not npz.is_absolute():
                npz = path.parent / npz
            if not manifest.is_absolute():
                manifest = path.parent / manifest
            npz = _resolve_regular(npz, root=canonical_root)
            manifest = _resolve_regular(manifest, root=canonical_root)
            try:
                npz = npz.resolve(strict=True)
                manifest = manifest.resolve(strict=True)
            except FileNotFoundError as error:
                raise ValueError("branch tensor/metadata is missing") from error
            if canonical_root is not None and (
                canonical_root not in npz.parents or canonical_root not in manifest.parents
            ):
                raise ValueError("branch tensor or metadata escapes its namespace")
            if not npz.is_file() or sha256_file(npz) != row.get("npz_sha256"): raise ValueError(f"branch tensor hash differs: {npz}")
            if not manifest.is_file() or sha256_file(manifest) != row.get("manifest_sha256"): raise ValueError(f"branch metadata hash differs: {manifest}")
            family = json.loads(manifest.read_text(encoding="utf-8"))
            if family.get("status") != "PASSED" or int(family.get("branches_per_family", -1)) != 24: raise ValueError(f"invalid branch family: {manifest}")
            if smoke is not None and family.get("smoke") is not bool(smoke):
                raise ValueError(
                    f"branch family smoke/formal provenance differs at {manifest}: "
                    f"{family.get('smoke')!r} != {bool(smoke)!r}"
                )
            if family.get("physical_simulator_outcomes") is not True:
                raise ValueError(f"branch family is not backed by physical simulator outcomes: {manifest}")
            if (
                family.get("schema") != "before-we-act.bicoord.care-physical-branch-family/2"
                or family.get("simulator_restore_mode")
                != "official_seed_plus_reference_prefix_replay"
            ):
                raise ValueError(f"branch family does not use exact seeded prefix reconstruction: {manifest}")
            if family.get("offline_demonstration_error_used") is not False or family.get("pseudo_labels_used") is not False:
                raise ValueError(f"branch family contains a non-physical label path: {manifest}")
            if family.get("care_memory_semantics") != "PredictiveTeamBeliefPolicy.belief.mu+belief.event_memory":
                raise ValueError(f"branch family memory semantics differ: {manifest}")
            probe = family.get("restore_probe")
            if not seed_replay_probe_valid(probe):
                raise ValueError(f"branch family restore probe failed: {manifest}")
            fidelity = family.get("reference_reactive_replay_fidelity")
            if not strict_fidelity_receipts_valid(fidelity):
                raise ValueError(f"branch family strict fidelity failed: {manifest}")
            if not physical_branch_family_rows_valid(family.get("branches")):
                raise ValueError(f"branch family physical row contract failed: {manifest}")
            keys = {
                (int(branch.get("candidate_id", -1)), str(branch.get("regime", "")), int(branch.get("repeat_id", -1)))
                for branch in family.get("branches", [])
                if isinstance(branch, Mapping) and branch.get("physical_simulator_outcome") is True
            }
            expected = {(candidate, regime, repeat) for candidate in range(6) for regime in ("reactive", "replay") for repeat in (0, 1)}
            if keys != expected or len(family.get("branches", [])) != 24:
                raise ValueError(f"branch family does not contain 24 unique physical branches: {manifest}")
            rows.append({"npz": npz, "family": family})
    rows.sort(key=lambda row: int(row["family"]["family_id"]))
    ids = [int(row["family"]["family_id"]) for row in rows]
    if len(ids) != len(set(ids)): raise ValueError("duplicate branch family IDs")
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    assert_common_paths(args, need_dataset=True)
    smoke = args.operation == "prepare-smoke"
    dependency_stage = "branch_smoke" if smoke else "branch_collection"
    require_stage_result(args.run, dependency_stage, config_sha256=args.config_sha256)
    branch_root = (args.run / "artifacts" / ("branch_smoke" if smoke else "branches")).resolve()
    rows = _records(
        _branch_manifests(args.run, smoke=smoke), smoke=smoke, root=branch_root
    )
    if not rows: raise RuntimeError("branch preparation found no family records")
    expected_families = len(TASKS) if smoke else len(TASKS) * 30
    if len(rows) != expected_families:
        raise RuntimeError(
            f"branch preparation requires {expected_families} {'smoke' if smoke else 'formal'} "
            f"families, found {len(rows)}"
        )
    normalization_path = args.run / "artifacts" / "dataset_audit" / "normalization.json"
    normalization = load_normalization_receipt(normalization_path, require_formal=True)
    memories: list[np.ndarray] = []; masks: list[np.ndarray] = []; candidates: list[np.ndarray] = []; targets: list[np.ndarray] = []; safety: list[np.ndarray] = []; usable: list[np.ndarray] = []; task_ids: list[int] = []; snapshots: list[str] = []
    for row in rows:
        with np.load(row["npz"], allow_pickle=False) as source:
            memory = np.asarray(source["memory"], np.float32); memory_mask = np.asarray(source["memory_mask"], bool); candidate = np.asarray(source["candidates"], np.float32); target = np.asarray(source["targets"], np.float32); hard = np.asarray(source["hard_safety"], np.float32); use = np.asarray(source["usable"], bool)
            task_id = int(np.asarray(source["task_id"]).item()); snapshot = str(np.asarray(source["snapshot_id"]).item())
        if memory.shape != (BICOORD_CARE_MEMORY_TOKENS, BICOORD_CARE_MEMORY_WIDTH) or memory_mask.shape != (BICOORD_CARE_MEMORY_TOKENS,) or candidate.shape != (6, 100, 7) or target.shape != (4, 6, 2, 3) or hard.shape != (4, 6, 2) or use.shape != (4,):
            raise ValueError(f"branch family tensor shapes differ: {row['npz']}")
        if not np.isfinite(memory).all() or not np.isfinite(candidate).all() or not np.isfinite(target).all() or not np.isfinite(hard).all(): raise ValueError(f"non-finite branch tensor: {row['npz']}")
        if not 0 <= task_id < len(TASKS): raise ValueError("branch task id out of range")
        if not memory_mask.any(): raise ValueError(f"branch family memory mask is empty: {row['npz']}")
        memories.append(memory); masks.append(memory_mask); candidates.append(candidate); targets.append(target); safety.append(hard); usable.append(use); task_ids.append(task_id); snapshots.append(snapshot)
    checkpoint_hashes = {str(row["family"].get("checkpoint_sha256", "")) for row in rows}
    checkpoint_paths = {str(row["family"].get("checkpoint", "")) for row in rows}
    normalization_hashes = {str(row["family"].get("normalization_receipt_sha256", "")) for row in rows}
    if len(checkpoint_hashes) != 1 or "" in checkpoint_hashes or len(checkpoint_paths) != 1 or "" in checkpoint_paths:
        raise ValueError("branch families do not share one B-core reference checkpoint")
    if normalization_hashes != {sha256_file(normalization_path)}:
        raise ValueError("branch families do not share the prepared normalization receipt")
    payload = {
        "format_version": "before-we-act.care-bicoord-prepared-data/1",
        "source_format_version": "before-we-act.care-bicoord-physical-prepared-data/1",
        "memory": torch.from_numpy(np.stack(memories)),
        "memory_mask": torch.from_numpy(np.stack(masks)),
        "candidate_chunks": torch.from_numpy(np.stack(candidates)),
        "targets": torch.from_numpy(np.stack(targets)),
        "hard_safety": torch.from_numpy(np.stack(safety)),
        "usable": torch.from_numpy(np.stack(usable)),
        # All families are training rows.  No validation/test split is used;
        # model selection is based on fixed offline diagnostics only.
        "split_id": torch.zeros(len(rows), dtype=torch.long),
        "task_id": torch.tensor(task_ids, dtype=torch.long),
        "snapshot_ids": tuple(snapshots),
        "tasks": tuple(TASKS),
        "action_std": torch.tensor(normalization["action_std"], dtype=torch.float32),
        "normalization_receipt": str(normalization_path.resolve()),
        "normalization_receipt_sha256": sha256_file(normalization_path),
        "reference_checkpoint": next(iter(checkpoint_paths)),
        "reference_checkpoint_sha256": next(iter(checkpoint_hashes)),
        "manifest": {"schema": "before-we-act.bicoord.prepared/1", "families": len(rows), "all_families_for_training": True, "held_out_families": 0, "provider_policy": "B-core/TUNE", "source_frequency_hz": 15, "physical_simulator_outcomes": True, "smoke": smoke, "care_memory_tokens": BICOORD_CARE_MEMORY_TOKENS, "care_memory_width": BICOORD_CARE_MEMORY_WIDTH, "care_memory_semantics": "PredictiveTeamBeliefPolicy.belief.mu+belief.event_memory"},
    }
    output = args.run / "artifacts" / ("prepared_branches_smoke.pt" if smoke else "prepared_branches.pt"); output.parent.mkdir(parents=True, exist_ok=True); temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp"); torch.save(payload, temporary); os.replace(temporary, output)
    report = output.with_suffix(".json"); atomic_json(report, {"schema": "before-we-act.bicoord.prepare-branches/1", "status": "PASSED", "families": len(rows), "tasks": {task: sum(task_id == i for task_id in task_ids) for i, task in enumerate(TASKS)}, "held_out_families": 0, "output": str(output.resolve()), "output_sha256": sha256_file(output)})
    stage = "branch_prepare_smoke" if smoke else "branch_prepare"
    return publish_result(args, stage=stage, include_model_contract=True, artifacts=[artifact(output, kind="prepared_branches"), artifact(report, kind="prepared_manifest")], families=len(rows), held_out_families=0, provider_policy="B-core/TUNE", all_families_for_training=True, physical_simulator_outcomes=True, reference_checkpoint_sha256=next(iter(checkpoint_hashes)))


def main(argv: list[str] | None = None) -> int:
    parser = common_parser(__doc__, ("prepare-smoke", "prepare")); parser.add_argument("--use-all-families", action="store_true"); parser.add_argument("--held-out-families", type=int, default=0); args = parser.parse_args(argv)
    if args.held_out_families != 0: raise ValueError("BiCoord CARE uses all branch families")
    run(args); return 0


if __name__ == "__main__": raise SystemExit(main())
