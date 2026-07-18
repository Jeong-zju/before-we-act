"""Self-contained checkpoint I/O for the Joint World-Action Model."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
import uuid

from safetensors.torch import load_file, save_file
import torch
import yaml

from models.wam import (
    NormalizationStats,
    RWMARConfig,
    RWMARWorldModel,
    StatefulActionFlow,
    StatefulActionFlowConfig,
)


CHECKPOINT_FORMAT_VERSION = "wam.joint_wam/1"
GENERATED_ACTION_WORLD_TARGET_SOURCE = "frozen_world_model_same_generated_actions"
WORLD_MODEL_WEIGHTS = "world_model.safetensors"
ACTION_FLOW_WEIGHTS = "action_flow.safetensors"
NORMALIZATION_FILE = "normalization.npz"
_HASHED_ARTIFACTS = (
    WORLD_MODEL_WEIGHTS,
    ACTION_FLOW_WEIGHTS,
    NORMALIZATION_FILE,
    "config.yaml",
    "dataset_manifest.json",
    "metrics.json",
    "provenance.json",
)


def save_joint_wam_checkpoint(
    directory: str | Path,
    world_model: RWMARWorldModel,
    flow: StatefulActionFlow,
    normalization: NormalizationStats,
    *,
    experiment_config: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    provenance: Mapping[str, Any],
    schema_version: str,
    source_fingerprints: Mapping[str, str] | None = None,
) -> Path:
    """Atomically save a portable model with no intermediate-checkpoint dependency."""

    _validate_model_contract(world_model, flow, normalization)
    flow.freeze_anchor()
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    _atomic_save_safetensors(
        _cpu_state_dict(world_model),
        target / WORLD_MODEL_WEIGHTS,
        metadata={
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_family": "joint_wam_world_model",
        },
    )
    _atomic_save_safetensors(
        _cpu_state_dict(flow),
        target / ACTION_FLOW_WEIGHTS,
        metadata={
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_family": "joint_wam_stateful_action_flow",
        },
    )
    _atomic_save_normalization(normalization, target / NORMALIZATION_FILE)
    config = dict(experiment_config)
    config["world_model_config"] = asdict(world_model.config)
    config["action_flow_config"] = asdict(flow.config)
    _atomic_write_text(
        target / "config.yaml", yaml.safe_dump(_plain(config), sort_keys=False)
    )
    _atomic_write_json(target / "dataset_manifest.json", dataset_manifest)
    _atomic_write_json(target / "metrics.json", metrics)
    _atomic_write_json(target / "provenance.json", provenance)
    schema = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "schema_version": schema_version,
        "model": "joint_wam",
        "runtime_inputs": ["states", "past_actions", "valid_mask"],
        "forbidden_runtime_inputs": [
            "privileged_state",
            "braking_agent",
            "braking_time",
        ],
        "normalization_sha256": normalization.sha256(),
        "action_prior_fallback_required": False,
        "embedded_frozen_action_prior_anchor": True,
        "generated_action_world_target_source": (GENERATED_ACTION_WORLD_TARGET_SOURCE),
        "generated_action_demo_state_is_ground_truth": False,
        "weight_files": {
            "world_model": WORLD_MODEL_WEIGHTS,
            "action_flow": ACTION_FLOW_WEIGHTS,
        },
        "source_fingerprints": dict(source_fingerprints or {}),
        "artifact_sha256": {name: _sha256(target / name) for name in _HASHED_ARTIFACTS},
    }
    _atomic_write_json(target / "schema.json", schema)
    return target


def load_joint_wam_checkpoint(
    directory: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_schema_version: str | None = None,
) -> tuple[RWMARWorldModel, StatefulActionFlow, dict[str, Any]]:
    """Strictly reconstruct a Joint WAM using only its own directory."""

    source = Path(directory)
    required = (*_HASHED_ARTIFACTS, "schema.json")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Joint WAM checkpoint is missing {missing}")
    schema = _read_json(source / "schema.json")
    _validate_schema(schema, expected_schema_version=expected_schema_version)
    _validate_artifact_hashes(source, schema)
    normalization = NormalizationStats.load(source / NORMALIZATION_FILE)
    if schema["normalization_sha256"] != normalization.sha256():
        raise ValueError("Joint WAM normalization fingerprint mismatch")
    payload = yaml.safe_load((source / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Joint WAM config root must be a mapping")
    raw_world = payload.get("world_model_config")
    raw_flow = payload.get("action_flow_config")
    if not isinstance(raw_world, Mapping) or not isinstance(raw_flow, Mapping):
        raise ValueError("Joint WAM checkpoint has no model configuration")
    world_model = RWMARWorldModel(RWMARConfig(**dict(raw_world)), normalization).to(
        device
    )
    flow = StatefulActionFlow(
        StatefulActionFlowConfig(**dict(raw_flow)), normalization
    ).to(device)
    _strict_load(
        world_model,
        source / WORLD_MODEL_WEIGHTS,
        device=device,
        component="world model",
    )
    _strict_load(
        flow,
        source / ACTION_FLOW_WEIGHTS,
        device=device,
        component="action flow",
    )
    _validate_model_contract(world_model, flow, normalization)
    world_model.eval()
    flow.eval()
    flow.freeze_anchor()
    return (
        world_model,
        flow,
        {
            "experiment_config": dict(payload),
            "schema": schema,
            "dataset_manifest": _read_json(source / "dataset_manifest.json"),
            "metrics": _read_json(source / "metrics.json"),
            "provenance": _read_json(source / "provenance.json"),
            "normalization": normalization,
        },
    )


def _validate_model_contract(
    world_model: RWMARWorldModel,
    flow: StatefulActionFlow,
    normalization: NormalizationStats,
) -> None:
    if flow.config.feature_dim != world_model.planning_feature_dim:
        raise ValueError("action-flow features do not match the world model")
    if flow.config.action_dim != world_model.config.action_dim:
        raise ValueError("action-flow action dimension does not match the world model")
    if normalization.state_mean.shape != (world_model.config.state_dim,):
        raise ValueError("normalization state dimension does not match the model")
    if normalization.action_mean.shape != (flow.config.action_dim,):
        raise ValueError("normalization action dimension does not match the model")


def _validate_schema(
    schema: Mapping[str, Any], *, expected_schema_version: str | None
) -> None:
    if schema.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported Joint WAM checkpoint")
    if schema.get("model") != "joint_wam":
        raise ValueError("checkpoint does not contain a Joint WAM")
    if (
        expected_schema_version is not None
        and schema.get("schema_version") != expected_schema_version
    ):
        raise ValueError("Joint WAM data schema mismatch")
    if schema.get("action_prior_fallback_required") is not False:
        raise ValueError("Joint WAM checkpoint requires a fallback")
    if schema.get("embedded_frozen_action_prior_anchor") is not True:
        raise ValueError("Joint WAM checkpoint has no frozen action anchor")
    if (
        schema.get("generated_action_world_target_source")
        != GENERATED_ACTION_WORLD_TARGET_SOURCE
    ):
        raise ValueError("generated-action target source is unsupported")
    if schema.get("generated_action_demo_state_is_ground_truth") is not False:
        raise ValueError("demo states may not supervise different generated actions")
    if schema.get("weight_files") != {
        "world_model": WORLD_MODEL_WEIGHTS,
        "action_flow": ACTION_FLOW_WEIGHTS,
    }:
        raise ValueError("Joint WAM weight-file manifest is invalid")
    artifact_hashes = schema.get("artifact_sha256")
    if not isinstance(artifact_hashes, Mapping) or set(artifact_hashes) != set(
        _HASHED_ARTIFACTS
    ):
        raise ValueError("Joint WAM artifact fingerprint manifest is invalid")
    if any(not _is_sha256(value) for value in artifact_hashes.values()):
        raise ValueError("Joint WAM artifact fingerprint is invalid")


def _validate_artifact_hashes(source: Path, schema: Mapping[str, Any]) -> None:
    for name in _HASHED_ARTIFACTS:
        if schema["artifact_sha256"][name] != _sha256(source / name):
            raise ValueError(f"Joint WAM artifact fingerprint mismatch: {name}")


def _strict_load(
    module: torch.nn.Module,
    path: Path,
    *,
    device: str | torch.device,
    component: str,
) -> None:
    incompatible = module.load_state_dict(
        load_file(path, device=str(device)), strict=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict Joint WAM {component} load failed: {incompatible}")


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous()
        for name, value in module.state_dict().items()
    }


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_save_safetensors(
    payload: Mapping[str, torch.Tensor],
    path: Path,
    *,
    metadata: Mapping[str, str],
) -> None:
    temporary = _temporary_path(path)
    try:
        save_file(dict(payload), temporary, metadata=dict(metadata))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_normalization(stats: NormalizationStats, path: Path) -> None:
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.npz")
    try:
        stats.save(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(_plain(payload), indent=2, sort_keys=True))


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = _temporary_path(path)
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain an object")
    return payload


__all__ = [
    "ACTION_FLOW_WEIGHTS",
    "CHECKPOINT_FORMAT_VERSION",
    "GENERATED_ACTION_WORLD_TARGET_SOURCE",
    "NORMALIZATION_FILE",
    "WORLD_MODEL_WEIGHTS",
    "load_joint_wam_checkpoint",
    "save_joint_wam_checkpoint",
]
