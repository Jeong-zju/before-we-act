"""Strict Phase 3 planning-head checkpoints referencing immutable Phase 2 weights."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from safetensors.torch import load_file, save_file
import torch
import yaml

from models.wam import WAMPlanningHeadConfig, WAMPlanningHeads

CHECKPOINT_FORMAT_VERSION = "wam.mppi_heads/1"


def phase2_checkpoint_fingerprint(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    schema = root / "schema.json"
    members = sorted((root / "members").glob("member_*.safetensors"))
    if not schema.is_file() or not members:
        raise FileNotFoundError("Phase 2 checkpoint is incomplete")
    return {
        "schema_sha256": _sha256(schema),
        "member_sha256": {path.name: _sha256(path) for path in members},
    }


def save_wam_mppi_heads_checkpoint(
    directory: str | Path,
    heads: WAMPlanningHeads,
    *,
    phase2_checkpoint: str | Path,
    experiment_config: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    provenance: Mapping[str, Any],
    schema_version: str,
    normalization_sha256: str,
) -> Path:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    weights = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in heads.state_dict().items()
    }
    save_file(
        weights,
        target / "planning_heads.safetensors",
        metadata={
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_family": "wam_mppi_planning_heads",
        },
    )
    config = dict(experiment_config)
    config["planning_head_config"] = asdict(heads.config)
    (target / "config.yaml").write_text(
        yaml.safe_dump(_plain(config), sort_keys=False), encoding="utf-8"
    )
    _write_json(
        target / "schema.json",
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "schema_version": schema_version,
            "runtime_inputs": ["states", "past_actions", "valid_mask"],
            "forbidden_runtime_inputs": [
                "privileged_state",
                "braking_agent",
                "braking_time",
            ],
            "normalization_sha256": normalization_sha256,
            "phase2_checkpoint": str(Path(phase2_checkpoint).resolve()),
            "phase2_fingerprint": phase2_checkpoint_fingerprint(phase2_checkpoint),
        },
    )
    _write_json(target / "dataset_manifest.json", dataset_manifest)
    _write_json(target / "metrics.json", metrics)
    _write_json(target / "provenance.json", provenance)
    return target


def load_wam_mppi_heads_checkpoint(
    directory: str | Path,
    *,
    phase2_checkpoint: str | Path,
    device: str | torch.device = "cpu",
    expected_schema_version: str | None = None,
    expected_normalization_sha256: str | None = None,
) -> tuple[WAMPlanningHeads, dict[str, Any]]:
    source = Path(directory)
    required = (
        "planning_heads.safetensors",
        "config.yaml",
        "schema.json",
        "dataset_manifest.json",
        "metrics.json",
        "provenance.json",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Phase 3 checkpoint is missing {missing}")
    schema = _read_json(source / "schema.json")
    if schema.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported Phase 3 planning-head checkpoint")
    if (
        expected_schema_version is not None
        and schema.get("schema_version") != expected_schema_version
    ):
        raise ValueError("Phase 3 data schema mismatch")
    if (
        expected_normalization_sha256 is not None
        and schema.get("normalization_sha256") != expected_normalization_sha256
    ):
        raise ValueError("Phase 3 normalization hash mismatch")
    actual_phase2 = phase2_checkpoint_fingerprint(phase2_checkpoint)
    if schema.get("phase2_fingerprint") != actual_phase2:
        raise ValueError("Phase 2 checkpoint fingerprint does not match Phase 3 heads")
    payload = yaml.safe_load((source / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or "planning_head_config" not in payload:
        raise ValueError("Phase 3 config has no planning_head_config")
    heads = WAMPlanningHeads(
        WAMPlanningHeadConfig(**dict(payload["planning_head_config"]))
    )
    incompatible = heads.load_state_dict(
        load_file(source / "planning_heads.safetensors", device=str(device)),
        strict=True,
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict planning-head load failed: {incompatible}")
    return heads.to(device).eval(), {
        "experiment_config": dict(payload),
        "schema": schema,
        "dataset_manifest": _read_json(source / "dataset_manifest.json"),
        "metrics": _read_json(source / "metrics.json"),
        "provenance": _read_json(source / "provenance.json"),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_plain(value), indent=2, sort_keys=True), encoding="utf-8"
    )


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "load_wam_mppi_heads_checkpoint",
    "phase2_checkpoint_fingerprint",
    "save_wam_mppi_heads_checkpoint",
]
