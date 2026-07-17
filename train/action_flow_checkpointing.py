"""Temporary checkpoint I/O for Joint WAM action-flow warm-up."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

from safetensors.torch import load_file, save_file
import torch
import yaml

from models.wam import StatefulActionFlow, StatefulActionFlowConfig
from train.action_prior import world_model_member_fingerprint
from train.rwm_u_checkpointing import load_rwm_u_member_checkpoint

CHECKPOINT_FORMAT_VERSION = "wam.action_flow_warmup/1"


def save_action_flow_checkpoint(
    directory: str | Path,
    flow: StatefulActionFlow,
    *,
    world_model_checkpoint: str | Path,
    experiment_config: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    provenance: Mapping[str, Any],
    schema_version: str,
    normalization_sha256: str,
) -> Path:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in flow.state_dict().items()
        },
        target / "action_flow.safetensors",
        metadata={
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_family": "wam_conditioned_stateful_action_flow",
            "stage": "action_flow_warmup",
        },
    )
    config = dict(experiment_config)
    config["stateful_action_flow_config"] = asdict(flow.config)
    (target / "config.yaml").write_text(
        yaml.safe_dump(_plain(config), sort_keys=False), encoding="utf-8"
    )
    _write_json(
        target / "schema.json",
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "schema_version": schema_version,
            "stage": "action_flow_warmup",
            "member_index": 0,
            "runtime_inputs": ["states", "past_actions", "valid_mask"],
            "forbidden_runtime_inputs": [
                "privileged_state",
                "braking_agent",
                "braking_time",
            ],
            "normalization_sha256": normalization_sha256,
            "world_model_member_fingerprint": world_model_member_fingerprint(
                world_model_checkpoint, 0
            ),
            "online_ensemble_required": False,
            "action_prior_fallback_required": False,
            "embedded_frozen_action_prior_anchor": True,
        },
    )
    _write_json(target / "dataset_manifest.json", dataset_manifest)
    _write_json(target / "metrics.json", metrics)
    _write_json(target / "provenance.json", provenance)
    return target


def load_action_flow_checkpoint(
    directory: str | Path,
    *,
    world_model_checkpoint: str | Path,
    device: str | torch.device = "cpu",
    expected_schema_version: str | None = None,
) -> tuple[StatefulActionFlow, dict[str, Any]]:
    source = Path(directory)
    required = (
        "action_flow.safetensors",
        "config.yaml",
        "schema.json",
        "dataset_manifest.json",
        "metrics.json",
        "provenance.json",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"action-flow checkpoint is missing {missing}")
    schema = _read_json(source / "schema.json")
    if schema.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported action-flow checkpoint")
    if schema.get("member_index") != 0:
        raise ValueError("action-flow warm-up must be bound to member 0")
    if schema.get("online_ensemble_required") is not False:
        raise ValueError("action-flow checkpoint requests an online ensemble")
    if schema.get("embedded_frozen_action_prior_anchor") is not True:
        raise ValueError("action-flow checkpoint has no frozen action anchor")
    if (
        expected_schema_version is not None
        and schema.get("schema_version") != expected_schema_version
    ):
        raise ValueError("action-flow data schema mismatch")
    current = world_model_member_fingerprint(world_model_checkpoint, 0)
    if schema.get("world_model_member_fingerprint") != current:
        raise ValueError("world-model member fingerprint does not match action flow")
    member, metadata = load_rwm_u_member_checkpoint(
        world_model_checkpoint,
        0,
        device=device,
        expected_schema_version=expected_schema_version,
    )
    if schema.get("normalization_sha256") != metadata["normalization"].sha256():
        raise ValueError("action-flow normalization hash mismatch")
    payload = yaml.safe_load((source / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("action-flow config root must be a mapping")
    raw_model = payload.get("stateful_action_flow_config")
    if not isinstance(raw_model, Mapping):
        raise ValueError("checkpoint has no stateful_action_flow_config")
    flow = StatefulActionFlow(
        StatefulActionFlowConfig(**dict(raw_model)), metadata["normalization"]
    )
    if flow.config.feature_dim != member.planning_feature_dim:
        raise ValueError("action-flow feature dimension does not match member 0")
    incompatible = flow.load_state_dict(
        load_file(source / "action_flow.safetensors", device=str(device)),
        strict=True,
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict action-flow load failed: {incompatible}")
    flow.to(device).eval()
    return flow, {
        "experiment_config": dict(payload),
        "schema": schema,
        "dataset_manifest": _read_json(source / "dataset_manifest.json"),
        "metrics": _read_json(source / "metrics.json"),
        "provenance": _read_json(source / "provenance.json"),
        "world_model_metadata": metadata,
    }


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain an object")
    return payload


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "load_action_flow_checkpoint",
    "save_action_flow_checkpoint",
]
