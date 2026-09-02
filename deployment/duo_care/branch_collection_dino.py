"""Collect diagnostic same-snapshot branches with the frozen DINO B0-H policy.

This legacy entry point intentionally does not accept ACT checkpoints, but it
is not the formal CARE branch path: it has only B0-H and therefore cannot
stand in for the PredictiveTeamBeliefPolicy B-core.  The formal supervisor uses
``duo_dino_branch_launcher``.  This collector remains useful for diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import numpy as np

from deployment.duo_dino_reference.data import TASKS
from deployment.duo_dino_reference.preprocessing import (
    DINO_NORMALIZATION_ID,
    IMAGE_PREPROCESS_ID,
)
from deployment.duo_care.branch_collection_v2 import (
    KernelConfig,
    advance_to_anchor,
    collect_from_anchor,
)
from deployment.duo_care.branch_signal import HORIZONS, stratified_anchor_steps
from deployment.duo_care.care_signal_audit import audit_family_json
from deployment.duo_care.duobench_adapter import (
    DuoBenchEnvironment,
    DuoDinoProposalProvider,
)


FORMAT_VERSION = "before-we-act.care-duobench-dino-branch-collection/2"
REFERENCE_POLICY_FAMILY = "TemporalHistoryPolicy"
VISION = "dinov3_vitb16_frozen"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def _family_signal(family: Mapping[str, Any], tolerance: float = 1e-7) -> dict[str, Any]:
    by = {
        (int(row["candidate_id"]), str(row["regime"]), int(row["repeat_id"])): row
        for row in family["branches"]
    }
    report: dict[str, Any] = {}
    for horizon in HORIZONS:
        deltas: list[float] = []
        utility_values: list[float] = []
        for regime in ("reactive", "replay"):
            for repeat in (0, 1):
                reference = float(by[(0, regime, repeat)]["outcomes"][str(horizon)]["utility_main"])
                for candidate in range(1, 6):
                    value = float(by[(candidate, regime, repeat)]["outcomes"][str(horizon)]["utility_main"])
                    utility_values.append(value)
                    deltas.append(value - reference)
        values = np.asarray(deltas, dtype=np.float64)
        report[str(horizon)] = {
            "nonzero_candidate_advantages": int(np.count_nonzero(np.abs(values) > tolerance)),
            "families_with_signal": int(bool(np.any(np.abs(values) > tolerance))),
            "advantage_linf": float(np.max(np.abs(values), initial=0.0)),
            "utility_range": float(np.ptp(np.asarray(utility_values, dtype=np.float64))) if utility_values else 0.0,
        }
    return report


def _aggregate_signal(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    result: dict[str, Any] = {}
    for horizon in HORIZONS:
        key = str(horizon)
        result[key] = {
            "nonzero_candidate_advantages": int(sum(int(row[key]["nonzero_candidate_advantages"]) for row in values)),
            "families_with_signal": int(sum(int(row[key]["families_with_signal"]) for row in values)),
            "advantage_linf": float(max((float(row[key]["advantage_linf"]) for row in values), default=0.0)),
        }
    return result


def _episode_length(data_manifest: Mapping[str, Any], task: str) -> int:
    row = data_manifest.get("tasks", {}).get(task, {})
    for key in ("validation_max_steps", "max_demo_steps", "mean_steps"):
        if key in row:
            return int(round(float(row[key])))
    raise ValueError(f"data manifest has no episode length for {task}")


def collect_task(
    *,
    provider: DuoDinoProposalProvider,
    task: str,
    output: Path,
    data_manifest: Mapping[str, Any],
    families_per_task: int,
    seed_start: int,
    image_size: int,
    smoke: bool,
) -> dict[str, Any]:
    maximum = _episode_length(data_manifest, task)
    anchors = stratified_anchor_steps(
        maximum,
        max_steps=maximum,
        count=families_per_task,
        horizon=max(HORIZONS),
        critical_count=min(20, families_per_task),
    )
    task_root = output / task
    task_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    started = time.perf_counter()
    env = DuoBenchEnvironment(task, image_size=image_size)
    try:
        for row in anchors:
            ordinal = int(row["ordinal"])
            seed = int(seed_start + TASKS.index(task) * 100_000 + ordinal)
            focal = ordinal % 2
            anchor_step = int(row["anchor_step"])
            snapshot_id = hashlib.sha256(
                f"duobench-care-dino-v2|{task}|{seed}|{anchor_step}|arm={focal}".encode()
            ).hexdigest()
            json_path = task_root / f"{snapshot_id}.json"
            npz_path = task_root / f"{snapshot_id}.npz"
            if json_path.is_file() and npz_path.is_file():
                family = json.loads(json_path.read_text(encoding="utf-8"))
            else:
                anchor = advance_to_anchor(
                    env,
                    provider,
                    task=task,
                    episode_seed=seed,
                    anchor_step=anchor_step,
                    focal_agent=focal,
                    sampling_stratum=str(row["sampling_stratum"]),
                    snapshot_id=snapshot_id,
                    config=KernelConfig(),
                )
                family, arrays = collect_from_anchor(env, provider, anchor, config=KernelConfig())
                family.update(
                    {
                        "reference_policy_family": REFERENCE_POLICY_FAMILY,
                        "policy_family": REFERENCE_POLICY_FAMILY,
                        "method_family": "CARE",
                        "architecture": "TemporalHistoryPolicy_hidden_residual",
                        "vision": VISION,
                        "vision_backbone": VISION,
                        "image_preprocess_id": IMAGE_PREPROCESS_ID,
                        "dino_normalization_id": DINO_NORMALIZATION_ID,
                        "strict_dino_contract": True,
                        "diagnostic_only": True,
                        "provider_checkpoint": str(provider.checkpoint),
                        "provider_checkpoint_sha256": provider.checkpoint_sha256,
                        "formal_provider": False,
                        "act_provider_allowed": False,
                    }
                )
                _atomic_npz(npz_path, arrays)
                _atomic_json(json_path, family)
            audit = audit_family_json(json_path, strict=True)
            if audit["status"] != "PASSED":
                raise RuntimeError(f"branch audit failed for {snapshot_id}: {audit['errors']}")
            signal = _family_signal(family)
            signals.append(signal)
            record = {
                "ordinal": ordinal,
                "snapshot_id": snapshot_id,
                "task": task,
                "episode_seed": seed,
                "anchor_step": anchor_step,
                "focal_agent": focal,
                "sampling_stratum": row["sampling_stratum"],
                "json": str(json_path.resolve()),
                "npz": str(npz_path.resolve()),
                "json_sha256": _sha256(json_path),
                "npz_sha256": _sha256(npz_path),
                "branch_gate": family.get("branch_gate"),
                "signal": signal,
            }
            records.append(record)
            print(
                json.dumps(
                    {
                        "event": "family_complete",
                        "task": task,
                        "ordinal": ordinal,
                        "anchor_step": anchor_step,
                        "focal_agent": focal,
                        "signal_h16": signal["16"]["nonzero_candidate_advantages"],
                    }
                ),
                flush=True,
            )
    finally:
        env.close()
    aggregate = _aggregate_signal(signals)
    errors: list[str] = []
    if not smoke:
        for horizon in (8, 16, 32):
            if aggregate[str(horizon)]["nonzero_candidate_advantages"] == 0:
                errors.append(f"horizon_{horizon}_all_candidate_advantages_zero")
    receipt = {
        "format_version": FORMAT_VERSION,
        "status": "PASSED" if not errors else "FAILED",
        "task": task,
        "smoke": bool(smoke),
        "families": len(records),
        "branches": 24 * len(records),
        "reference_policy_family": REFERENCE_POLICY_FAMILY,
        "policy_family": REFERENCE_POLICY_FAMILY,
        "method_family": "CARE",
        "architecture": "TemporalHistoryPolicy_hidden_residual",
        "vision": VISION,
        "vision_backbone": VISION,
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "strict_dino_contract": True,
        "diagnostic_only": True,
        "strictly_decentralized": True,
        "action_encoding": "joint_residual_gripper_absolute",
        "provider_checkpoint": str(provider.checkpoint),
        "provider_checkpoint_sha256": provider.checkpoint_sha256,
        "act_provider_allowed": False,
        "signal": aggregate,
        "errors": errors,
        "wall_seconds": time.perf_counter() - started,
        "records": records,
    }
    _atomic_json(output / f"{task}.receipt.json", receipt)
    if errors:
        raise RuntimeError(f"formal CARE branch signal gate failed: {errors}")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", action="append", choices=TASKS)
    parser.add_argument("--families-per-task", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=20261001)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dino-model")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.families_per_task < 1:
        parser.error("--families-per-task must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    data_manifest = json.loads(args.data_manifest.read_text(encoding="utf-8"))
    provider = DuoDinoProposalProvider(
        args.checkpoint,
        device=args.device,
        dino_model=args.dino_model,
        image_height=args.image_size,
        image_width=args.image_size,
    )
    tasks = tuple(args.task or TASKS)
    receipts = [
        collect_task(
            provider=provider,
            task=task,
            output=args.output,
            data_manifest=data_manifest,
            families_per_task=args.families_per_task,
            seed_start=args.seed_start,
            image_size=args.image_size,
            smoke=args.smoke,
        )
        for task in tasks
    ]
    aggregate = {
        "format_version": FORMAT_VERSION,
        "status": "PASSED" if all(row["status"] == "PASSED" for row in receipts) else "FAILED",
        "reference_policy_family": REFERENCE_POLICY_FAMILY,
        "policy_family": REFERENCE_POLICY_FAMILY,
        "method_family": "CARE",
        "vision": VISION,
        "vision_backbone": VISION,
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "strict_dino_contract": True,
        "diagnostic_only": True,
        "strictly_decentralized": True,
        "act_provider_allowed": False,
        "tasks": list(tasks),
        "family_count": int(sum(row["families"] for row in receipts)),
        "branch_count": int(sum(row["branches"] for row in receipts)),
        "task_receipts": {row["task"]: str((args.output / f"{row['task']}.receipt.json").resolve()) for row in receipts},
    }
    _atomic_json(args.output / "collection_receipt.json", aggregate)
    print(json.dumps(aggregate, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()


__all__ = ["REFERENCE_POLICY_FAMILY", "VISION", "collect_task"]
