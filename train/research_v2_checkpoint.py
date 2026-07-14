"""Checkpoint and runtime-bundle contract for FE-PC-WAM Research-v2."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from data.research_v2 import RESEARCH_V2_SCHEMA_VERSION


RESEARCH_V2_CHECKPOINT_CONTRACT = "fe_pc_wam/research_v2_local"
RESEARCH_V2_CHECKPOINT_FORMAT = 1
RESEARCH_V2_STAGES = (
    "plan",
    "belief",
    "world_direct",
    "world_block",
    "proposal",
    "intention",
    "calibration",
)
CALIBRATION_V2_REQUIRED_EXTRA = (
    "quantile_scale",
    "constraint_temperature",
    "posterior_temperature",
    "posterior_variance_scale",
    "communication_price_frozen",
    "world_ensemble_size",
    "world_ensemble_sha256",
)


class IncompatibleResearchV2Checkpoint(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def make_research_v2_checkpoint(
    *,
    stage: str,
    model_class: str,
    model_config: Any,
    model_state_dict: Mapping[str, torch.Tensor],
    training_config: Any,
    dataset_manifest_sha256: str,
    forward_inputs: Sequence[str],
    metrics: Mapping[str, Any],
    upstream: Mapping[str, Mapping[str, Any]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if stage not in RESEARCH_V2_STAGES:
        raise ValueError(f"unknown Research-v2 stage {stage!r}")
    if any("privileged" in name or "truth" in name for name in forward_inputs):
        raise ValueError("runtime forward signature cannot contain privileged/truth inputs")
    checkpoint = {
        "checkpoint_contract": RESEARCH_V2_CHECKPOINT_CONTRACT,
        "checkpoint_format": RESEARCH_V2_CHECKPOINT_FORMAT,
        "schema_version": RESEARCH_V2_SCHEMA_VERSION,
        "stage": stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_class": str(model_class),
        "model_config": _to_dict(model_config),
        "model_state_dict": dict(model_state_dict),
        "training_config": _to_dict(training_config),
        "dataset_manifest_sha256": str(dataset_manifest_sha256),
        "forward_inputs": list(forward_inputs),
        "metrics": dict(metrics),
        "upstream": {name: dict(value) for name, value in (upstream or {}).items()},
        "extra": dict(extra or {}),
    }
    validate_research_v2_checkpoint(checkpoint)
    return checkpoint


def save_research_v2_checkpoint(path: str | Path, checkpoint: Mapping[str, Any]) -> Path:
    validate_research_v2_checkpoint(checkpoint)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(dict(checkpoint), temporary)
    temporary.replace(destination)
    return destination


def load_research_v2_checkpoint(
    path: str | Path,
    *,
    expected_stage: str | Sequence[str] | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise IncompatibleResearchV2Checkpoint("checkpoint payload is not a mapping")
    result = dict(checkpoint)
    validate_research_v2_checkpoint(result, expected_stage=expected_stage)
    return result


def validate_research_v2_checkpoint(
    checkpoint: Mapping[str, Any], expected_stage: str | Sequence[str] | None = None
) -> None:
    if checkpoint.get("checkpoint_contract") != RESEARCH_V2_CHECKPOINT_CONTRACT:
        raise IncompatibleResearchV2Checkpoint(
            "V1/legacy checkpoints cannot be used by Research-v2"
        )
    if checkpoint.get("checkpoint_format") != RESEARCH_V2_CHECKPOINT_FORMAT:
        raise IncompatibleResearchV2Checkpoint("unsupported Research-v2 checkpoint format")
    if checkpoint.get("schema_version") != RESEARCH_V2_SCHEMA_VERSION:
        raise IncompatibleResearchV2Checkpoint("checkpoint data schema mismatch")
    stage = checkpoint.get("stage")
    if stage not in RESEARCH_V2_STAGES:
        raise IncompatibleResearchV2Checkpoint(f"unknown stage {stage!r}")
    if expected_stage is not None:
        allowed = {expected_stage} if isinstance(expected_stage, str) else set(expected_stage)
        if stage not in allowed:
            raise IncompatibleResearchV2Checkpoint(f"expected stage {sorted(allowed)}, got {stage}")
    forward_inputs = checkpoint.get("forward_inputs")
    if not isinstance(forward_inputs, list) or any(
        "privileged" in str(name) or "truth" in str(name) for name in forward_inputs
    ):
        raise IncompatibleResearchV2Checkpoint("invalid deployable forward signature")
    for field in (
        "model_class",
        "model_config",
        "model_state_dict",
        "training_config",
        "dataset_manifest_sha256",
    ):
        if field not in checkpoint:
            raise IncompatibleResearchV2Checkpoint(f"checkpoint misses {field}")
    if not isinstance(checkpoint.get("model_state_dict"), Mapping):
        raise IncompatibleResearchV2Checkpoint("model_state_dict must be a mapping")
    if stage == "calibration":
        _validate_calibration_extra(checkpoint.get("extra"))


def checkpoint_reference(path: str | Path, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "stage": checkpoint["stage"],
        "contract": checkpoint["checkpoint_contract"],
    }


def write_runtime_bundle_manifest(
    output_dir: str | Path,
    checkpoints: Mapping[str, str | Path],
    *,
    ensemble_members: Sequence[str | Path],
    parameter_counts: Mapping[str, int],
) -> Path:
    """Freeze hashes and deployable signatures without copying training probes."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    required = {"plan", "belief", "proposal", "intention", "world_block", "calibration"}
    if set(checkpoints) != required:
        raise ValueError(f"runtime bundle requires exactly {sorted(required)}")
    artifacts: dict[str, Any] = {}
    artifact_states: dict[str, dict[str, Any]] = {}
    dataset_hash: str | None = None
    expected_stage = {
        "plan": "plan",
        "belief": "belief",
        "proposal": "proposal",
        "intention": "intention",
        "world_block": "world_block",
        "calibration": "calibration",
    }
    for name, path in checkpoints.items():
        state = load_research_v2_checkpoint(path, expected_stage=expected_stage[name])
        artifact_states[name] = state
        if dataset_hash is None:
            dataset_hash = state["dataset_manifest_sha256"]
        elif state["dataset_manifest_sha256"] != dataset_hash:
            raise ValueError("runtime checkpoints were trained from different dataset manifests")
        artifacts[name] = {
            "path": os.path.relpath(Path(path).resolve(), root.resolve()),
            "sha256": sha256_file(path),
            "stage": state["stage"],
            "model_class": state["model_class"],
            "forward_inputs": state["forward_inputs"],
        }
    if not ensemble_members:
        raise ValueError("runtime bundle requires at least one world-model member")
    ensemble = []
    ensemble_states: list[dict[str, Any]] = []
    for path in ensemble_members:
        state = load_research_v2_checkpoint(path, expected_stage="world_block")
        ensemble_states.append(state)
        if state["dataset_manifest_sha256"] != dataset_hash:
            raise ValueError("world ensemble members use a different dataset manifest")
        ensemble.append(
            {
                "path": os.path.relpath(Path(path).resolve(), root.resolve()),
                "sha256": sha256_file(path),
            }
        )
    unique_world_hashes = {member["sha256"] for member in ensemble}
    calibration_state = load_research_v2_checkpoint(
        checkpoints["calibration"], expected_stage="calibration"
    )
    calibrated_world_hashes = calibration_state["extra"]["world_ensemble_sha256"]
    actual_world_hashes = [member["sha256"] for member in ensemble]
    if calibrated_world_hashes != actual_world_hashes:
        raise ValueError(
            "runtime world ensemble differs from the ensemble used for calibration"
        )
    if actual_world_hashes[0] != artifacts["world_block"]["sha256"]:
        raise ValueError("primary world artifact must be ensemble member zero")

    def require_upstream(state, artifact_name, upstream_name, expected_hash):
        reference = state.get("upstream", {}).get(upstream_name)
        if reference is None or reference.get("sha256") != expected_hash:
            raise ValueError(
                f"{artifact_name} checkpoint has stale/missing {upstream_name} lineage"
            )

    plan_hash = artifacts["plan"]["sha256"]
    belief_hash = artifacts["belief"]["sha256"]
    primary_world_hash = artifacts["world_block"]["sha256"]
    intention_hash = artifacts["intention"]["sha256"]
    require_upstream(artifact_states["belief"], "belief", "plan", plan_hash)
    for index, state in enumerate(ensemble_states):
        require_upstream(state, f"world member {index}", "plan", plan_hash)
        require_upstream(state, f"world member {index}", "belief", belief_hash)
    for name in ("proposal", "intention"):
        require_upstream(artifact_states[name], name, "plan", plan_hash)
        require_upstream(artifact_states[name], name, "belief", belief_hash)
        require_upstream(
            artifact_states[name], name, "world_block", primary_world_hash
        )
    require_upstream(calibration_state, "calibration", "plan", plan_hash)
    require_upstream(calibration_state, "calibration", "belief", belief_hash)
    require_upstream(
        calibration_state, "calibration", "world_block", primary_world_hash
    )
    require_upstream(
        calibration_state, "calibration", "intention", intention_hash
    )
    for index, expected_hash in enumerate(actual_world_hashes):
        require_upstream(
            calibration_state,
            "calibration",
            f"world_block_member_{index:02d}",
            expected_hash,
        )
    payload = {
        "bundle_contract": "fe_pc_wam/research_v2_runtime_bundle",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": RESEARCH_V2_SCHEMA_VERSION,
        "dataset_manifest_sha256": dataset_hash,
        "artifacts": artifacts,
        "world_ensemble": ensemble,
        "world_ensemble_size": len(ensemble),
        "unique_world_model_count": len(unique_world_hashes),
        "epistemic_uncertainty_available": len(unique_world_hashes) >= 2,
        "parameter_counts": dict(parameter_counts),
        "privileged_runtime_inputs": [],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["bundle_sha256"] = hashlib.sha256(canonical).hexdigest()
    destination = root / "runtime_bundle.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return destination


def _to_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("configuration must be a dataclass or mapping")


def _validate_calibration_extra(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise IncompatibleResearchV2Checkpoint("calibration checkpoint misses extra mapping")
    missing = [name for name in CALIBRATION_V2_REQUIRED_EXTRA if name not in value]
    if missing:
        raise IncompatibleResearchV2Checkpoint(
            f"calibration checkpoint misses extra fields {missing}"
        )
    positive = (
        "quantile_scale",
        "constraint_temperature",
        "posterior_temperature",
        "posterior_variance_scale",
    )
    for name in positive:
        try:
            parsed = float(value[name])
        except (TypeError, ValueError) as error:
            raise IncompatibleResearchV2Checkpoint(
                f"calibration field {name} is not numeric"
            ) from error
        if not math.isfinite(parsed) or parsed <= 0:
            raise IncompatibleResearchV2Checkpoint(
                f"calibration field {name} must be finite and positive"
            )
    for name in ("quantile_bias", "constraint_logit_bias", "constraint_bias"):
        if name in value and not math.isfinite(float(value[name])):
            raise IncompatibleResearchV2Checkpoint(
                f"calibration field {name} must be finite"
            )
    frozen = value["communication_price_frozen"]
    if not isinstance(frozen, bool):
        raise IncompatibleResearchV2Checkpoint(
            "communication_price_frozen must be a boolean"
        )
    price = value.get("communication_price", value.get("communication_cost"))
    if frozen and price is None:
        raise IncompatibleResearchV2Checkpoint(
            "frozen calibration requires communication_price"
        )
    if price is not None and (not math.isfinite(float(price)) or float(price) < 0):
        raise IncompatibleResearchV2Checkpoint(
            "communication_price must be finite and non-negative"
        )
    ensemble_size = value["world_ensemble_size"]
    ensemble_hashes = value["world_ensemble_sha256"]
    if not isinstance(ensemble_size, int) or ensemble_size <= 0:
        raise IncompatibleResearchV2Checkpoint(
            "calibration world_ensemble_size must be a positive integer"
        )
    if (
        not isinstance(ensemble_hashes, list)
        or len(ensemble_hashes) != ensemble_size
        or any(not isinstance(item, str) or len(item) != 64 for item in ensemble_hashes)
    ):
        raise IncompatibleResearchV2Checkpoint(
            "calibration world_ensemble_sha256 must match world_ensemble_size"
        )
