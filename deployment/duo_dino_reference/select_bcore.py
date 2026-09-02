"""Offline, pre-closed-loop selection of the three formal Duo B-core seeds."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import torch

from .bcore_data import (
    BCORE_DEPLOYMENT_FORMAT,
    BCORE_SEEDS,
    BCORE_TRAINING_FORMAT,
    sha256_file,
    validate_b0h_payload,
    DUO_CARE_MEMORY_SEMANTICS,
    DUO_CARE_MEMORY_TOKENS,
    DUO_CARE_MEMORY_WIDTH,
)
from deployment.duo_act.action_target import ACTION_TARGET_CONTRACT_ID, ACTION_TARGET_CONTRACT_SHA256
from .preprocessing import DINO_NORMALIZATION_ID, IMAGE_PREPROCESS_ID


SELECTION_SCHEMA = "before-we-act.duobench.bcore-selection/1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _offline_score(payload: Mapping[str, Any]) -> float:
    """Read a deterministic score from training diagnostics only."""

    evaluations = payload.get("evaluations", [])
    rows: list[tuple[int, float]] = []
    if isinstance(evaluations, list):
        for item in evaluations:
            if not isinstance(item, Mapping):
                continue
            try:
                update = int(item["update"])
                validation = item["validation"]
                macro = validation["macro"]
                score = float(macro["b_core"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if score == score and score < float("inf"):
                rows.append((update, score))
    if rows:
        # Use the best checkpoint in the final 20k-update sufficiency window;
        # this is fixed before any closed-loop result exists.
        final = [row for row in rows if row[0] >= max(0, 120_000 - 20_000)]
        return min(final or rows, key=lambda row: row[1])[1]
    metrics = payload.get("last_metrics", {})
    if isinstance(metrics, Mapping):
        for key in ("b_core_action_mse", "action", "combined", "loss"):
            value = metrics.get(key)
            if isinstance(value, (int, float)) and float(value) == float(value):
                return float(value)
    # A missing diagnostic is intentionally worst, never a reason to choose a
    # seed because it happened to omit validation metadata.
    return float("inf")


def _validate_training_payload(
    path: Path,
    *,
    seed: int,
    b0h_sha: str,
) -> tuple[Mapping[str, Any], Path, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"B-core seed {seed} checkpoint is not a mapping")
    format_value = payload.get("format") or payload.get("format_version")
    if format_value != BCORE_TRAINING_FORMAT:
        raise ValueError(f"B-core seed {seed} has wrong format {format_value!r}")
    config = payload.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError(f"B-core seed {seed} has no config")
    for key, expected in (
        ("policy_family", "PredictiveTeamBeliefPolicy"),
        # CARE labels the method/protocol; it is not the policy family.
        # Missing or mismatched method metadata must fail closed.
        ("method_family", "CARE"),
        ("benchmark_adapter", "DuoBench"),
        ("vision_backbone", "dinov3_vitb16_frozen"),
        ("image_preprocess_id", IMAGE_PREPROCESS_ID),
        ("dino_normalization_id", DINO_NORMALIZATION_ID),
        ("action_encoding", "absolute_joint7_binary_gripper1"),
    ):
        if config.get(key) != expected:
            raise ValueError(f"B-core seed {seed} config differs at {key}")
    if config.get("strictly_decentralized") is not True or config.get("strict_local") is not True:
        raise ValueError(f"B-core seed {seed} is not strict-local")
    if config.get("act_provider_allowed") is not False:
        raise ValueError(f"B-core seed {seed} allows an ACT provider")
    source_hash = config.get("source_b0h_checkpoint_sha256")
    if source_hash != b0h_sha:
        raise ValueError(f"B-core seed {seed} was trained from a different B0-H")
    if int(payload.get("update", -1)) != 120_000:
        raise ValueError(f"B-core seed {seed} did not complete 120000 updates")
    state = payload.get("model")
    if not isinstance(state, Mapping) or not any(
        str(key).startswith("belief_core.") for key in state
    ):
        raise ValueError(f"B-core seed {seed} is not a PredictiveTeamBeliefPolicy state")
    deployment = path.parent / "deployment_checkpoint.pt"
    if not deployment.is_file():
        raise FileNotFoundError(deployment)
    deployed = torch.load(deployment, map_location="cpu", weights_only=False)
    if not isinstance(deployed, Mapping) or (
        deployed.get("format") or deployed.get("format_version")
    ) != BCORE_DEPLOYMENT_FORMAT:
        raise ValueError(f"B-core seed {seed} deployment format differs")
    if deployed.get("source_b0h_checkpoint_sha256") != b0h_sha:
        raise ValueError(f"B-core seed {seed} deployment B0-H provenance differs")
    if deployed.get("policy_family") != "PredictiveTeamBeliefPolicy":
        raise ValueError(f"B-core seed {seed} deployment family differs")
    if deployed.get("method_family") != "CARE":
        raise ValueError(f"B-core seed {seed} deployment method family differs")
    if deployed.get("benchmark_adapter") != "DuoBench":
        raise ValueError(f"B-core seed {seed} deployment benchmark differs")
    if deployed.get("image_preprocess_id") != IMAGE_PREPROCESS_ID:
        raise ValueError(f"B-core seed {seed} deployment image preprocessing differs")
    if deployed.get("dino_normalization_id") != DINO_NORMALIZATION_ID:
        raise ValueError(f"B-core seed {seed} deployment DINO normalization differs")
    if deployed.get("strictly_decentralized") is not True or deployed.get("strict_local") is not True:
        raise ValueError(f"B-core seed {seed} deployment is not strict-local")
    return payload, deployment, _offline_score(payload)


def select_bcore(
    training_root: Path,
    b0h_checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    """Select and copy one seed using frozen offline diagnostics only."""

    b0h = torch.load(b0h_checkpoint, map_location="cpu", weights_only=False)
    validate_b0h_payload(b0h)
    b0h_sha = sha256_file(b0h_checkpoint)
    rows: list[dict[str, Any]] = []
    candidates: list[tuple[float, int, Path]] = []
    for seed in BCORE_SEEDS:
        root = training_root / f"seed_{seed}"
        status_path = root / "status.json"
        if not status_path.is_file():
            raise FileNotFoundError(status_path)
        status = json.loads(status_path.read_text())
        if status.get("status") not in (
            "COMPLETED",
            "PASSED",
            "PLATFORM_REACHED",
            "INCONCLUSIVE_TRAINING_NOT_CONVERGED",
            "SATURATED_BY_OVERFIT",
        ):
            raise ValueError(f"B-core seed {seed} has non-terminal status")
        checkpoint = root / "checkpoint_latest.pt"
        payload, deployment, score = _validate_training_payload(
            checkpoint, seed=seed, b0h_sha=b0h_sha
        )
        rows.append(
            {
                "seed": seed,
                "offline_score_b_core_mse": score,
                "training_checkpoint": str(checkpoint.resolve()),
                "training_checkpoint_sha256": sha256_file(checkpoint),
                "deployment_checkpoint": str(deployment.resolve()),
                "deployment_checkpoint_sha256": sha256_file(deployment),
                "update": int(payload.get("update", -1)),
                "closed_loop_results_used": False,
            }
        )
        candidates.append((score, seed, deployment))
    # Stable tie-break by seed keeps resume/rebuild bit-identical.
    score, selected_seed, selected_path = min(candidates, key=lambda row: (row[0], row[1]))
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "deployment_checkpoint.pt"
    temporary = output / f".deployment_checkpoint.{os.getpid()}.tmp"
    shutil.copyfile(selected_path, temporary)
    os.replace(temporary, destination)
    receipt = {
        "schema": SELECTION_SCHEMA,
        "status": "PASSED",
        "selected_seed": int(selected_seed),
        "selected_offline_score_b_core_mse": float(score),
        "candidates": rows,
        "seeds": list(BCORE_SEEDS),
        "updates_per_seed": 120_000,
        "closed_loop_results_used_for_selection": False,
        "selection_stage": "pre_closed_loop_offline_only",
        "policy_family": "PredictiveTeamBeliefPolicy",
        "reference_policy_family": "PredictiveTeamBeliefPolicy",
        "method_family": "CARE",
        "architecture": "PredictiveTeamBeliefPolicy_direct_belief_residual",
        "benchmark_adapter": "DuoBench",
        "vision": "dinov3_vitb16_frozen",
        "vision_backbone": "dinov3_vitb16_frozen",
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "strict_dino_contract": True,
        "action_encoding": "absolute_joint7_binary_gripper1",
        "strictly_decentralized": True,
        "strict_local": True,
        "act_provider_allowed": False,
        "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
        "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
        "memory_semantics": DUO_CARE_MEMORY_SEMANTICS,
        "care_memory_tokens": DUO_CARE_MEMORY_TOKENS,
        "care_memory_width": DUO_CARE_MEMORY_WIDTH,
        "all_550_demonstrations": True,
        "source_b0h_checkpoint": str(b0h_checkpoint.resolve()),
        "source_b0h_checkpoint_sha256": b0h_sha,
        "source_checkpoint": str(selected_path.resolve()),
        "source_checkpoint_sha256": sha256_file(selected_path),
        "deployment_checkpoint": str(destination.resolve()),
        "deployment_checkpoint_sha256": sha256_file(destination),
        "created_at_utc": _now(),
    }
    _atomic_json(output / "selection_receipt.json", receipt)
    _atomic_json(
        output / "status.json",
        {
            "status": "PASSED",
            "selected_seed": int(selected_seed),
            "deployment_checkpoint_sha256": sha256_file(destination),
            "closed_loop_results_used_for_selection": False,
            "completed_at_utc": _now(),
        },
    )
    print(json.dumps(receipt), flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--b0h-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    select_bcore(args.training_root, args.b0h_checkpoint, args.output)


if __name__ == "__main__":
    main()


__all__ = ["select_bcore"]
