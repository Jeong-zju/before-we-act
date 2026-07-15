"""Safe, versioned Phase 1 RWM-AR checkpoint save/load helpers."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from safetensors.torch import load_file, save_file
import torch
import yaml

from models.wam import NormalizationStats, RWMARConfig, RWMARWorldModel

CHECKPOINT_FORMAT_VERSION = "wam.rwm_ar/1"


def save_wam_checkpoint(
    directory: str | Path,
    model: RWMARWorldModel,
    stats: NormalizationStats,
    *,
    experiment_config: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    provenance: Mapping[str, Any],
    schema_version: str,
) -> Path:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    weights = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    save_file(
        weights,
        target / "model.safetensors",
        metadata={
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_family": "rwm_ar",
        },
    )
    # Phase 1 has no EMA training path yet.  Preserve the planned inference
    # surface with byte-identical safe weights rather than inventing EMA state.
    shutil.copyfile(target / "model.safetensors", target / "ema_model.safetensors")
    stats.save(target / "normalization.npz")
    config_payload = dict(experiment_config)
    config_payload["model_config"] = _plain(asdict(model.config))
    (target / "config.yaml").write_text(
        yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8"
    )
    schema = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "schema_version": schema_version,
        "state_dim": model.config.state_dim,
        "action_dim": model.config.action_dim,
        "history_horizon": model.config.history_horizon,
        "normalization_sha256": stats.sha256(),
        "runtime_inputs": ["states", "past_actions", "valid_mask", "candidate_actions"],
        "forbidden_runtime_inputs": [
            "privileged_state",
            "braking_agent",
            "braking_time",
        ],
    }
    _write_json(target / "schema.json", schema)
    _write_json(target / "dataset_manifest.json", dataset_manifest)
    _write_json(target / "metrics.json", metrics)
    _write_json(target / "provenance.json", provenance)
    return target


def load_wam_checkpoint(
    directory: str | Path,
    *,
    device: str | torch.device = "cpu",
    use_ema: bool = False,
    expected_schema_version: str | None = None,
) -> tuple[RWMARWorldModel, dict[str, Any]]:
    source = Path(directory)
    required = (
        "model.safetensors",
        "ema_model.safetensors",
        "config.yaml",
        "normalization.npz",
        "schema.json",
        "dataset_manifest.json",
        "metrics.json",
        "provenance.json",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint is missing {missing}")
    experiment_config = yaml.safe_load((source / "config.yaml").read_text())
    if (
        not isinstance(experiment_config, dict)
        or "model_config" not in experiment_config
    ):
        raise ValueError("checkpoint config.yaml has no model_config")
    model_payload = dict(experiment_config["model_config"])
    for name in ("yaw_indices", "gripper_closed_indices"):
        if name in model_payload:
            model_payload[name] = tuple(model_payload[name])
    config = RWMARConfig(**model_payload)
    stats = NormalizationStats.load(source / "normalization.npz")
    schema = json.loads((source / "schema.json").read_text())
    if schema.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format {schema.get('format_version')!r}"
        )
    if (
        expected_schema_version is not None
        and schema.get("schema_version") != expected_schema_version
    ):
        raise ValueError(
            f"schema mismatch: {schema.get('schema_version')!r} != "
            f"{expected_schema_version!r}"
        )
    if schema.get("normalization_sha256") != stats.sha256():
        raise ValueError("normalization hash does not match schema.json")
    if (
        schema.get("state_dim") != config.state_dim
        or schema.get("action_dim") != config.action_dim
    ):
        raise ValueError("schema dimensions do not match model config")
    model = RWMARWorldModel(config, stats)
    weights_name = "ema_model.safetensors" if use_ema else "model.safetensors"
    state_dict = load_file(source / weights_name, device=str(device))
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict checkpoint load failed: {incompatible}")
    model.to(device)
    model.eval()
    metadata = {
        "experiment_config": experiment_config,
        "schema": schema,
        "dataset_manifest": json.loads((source / "dataset_manifest.json").read_text()),
        "metrics": json.loads((source / "metrics.json").read_text()),
        "provenance": json.loads((source / "provenance.json").read_text()),
        "normalization": stats,
    }
    return model, metadata


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
    "load_wam_checkpoint",
    "save_wam_checkpoint",
]
