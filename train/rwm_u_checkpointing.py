"""Safe, versioned checkpoint helpers for Phase 2 RWM-U ensembles."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

from safetensors.torch import load_file, save_file
import torch
import yaml

from models.wam import (
    NormalizationStats,
    RWMARConfig,
    RWMARWorldModel,
    RWMUEnsemble,
    RWMUEnsembleConfig,
    RWMURiskConfig,
)

CHECKPOINT_FORMAT_VERSION = "wam.rwm_u/1"


def save_rwm_u_member_weights(
    directory: str | Path, member_index: int, model: RWMARWorldModel
) -> Path:
    target = Path(directory) / "members" / f"member_{member_index:02d}.safetensors"
    target.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        _cpu_state_dict(model),
        target,
        metadata={
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_family": "rwm_u_member",
            "member_index": str(member_index),
        },
    )
    return target


def save_teacher_forcing_weights(
    directory: str | Path, model: RWMARWorldModel
) -> Path:
    target = Path(directory) / "teacher_forcing.safetensors"
    target.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        _cpu_state_dict(model),
        target,
        metadata={
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_family": "rwm_ar_teacher_forcing_ablation",
        },
    )
    return target


def load_rwm_u_member_weights(
    directory: str | Path,
    member_index: int,
    model: RWMARWorldModel,
    *,
    device: str | torch.device,
) -> RWMARWorldModel:
    path = Path(directory) / "members" / f"member_{member_index:02d}.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"missing member checkpoint {path}")
    incompatible = model.load_state_dict(load_file(path, device=str(device)), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict member {member_index} load failed: {incompatible}")
    return model.to(device)


def load_teacher_forcing_weights(
    directory: str | Path,
    model: RWMARWorldModel,
    *,
    device: str | torch.device,
) -> RWMARWorldModel:
    path = Path(directory) / "teacher_forcing.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"missing teacher-forcing checkpoint {path}")
    incompatible = model.load_state_dict(load_file(path, device=str(device)), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict teacher-forcing load failed: {incompatible}")
    return model.to(device)


def save_rwm_u_checkpoint(
    directory: str | Path,
    ensemble: RWMUEnsemble,
    stats: NormalizationStats,
    *,
    teacher_forcing_model: RWMARWorldModel | None,
    experiment_config: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    bootstrap_manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    provenance: Mapping[str, Any],
    schema_version: str,
) -> Path:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    for index, member in enumerate(ensemble.members):
        save_rwm_u_member_weights(target, index, member)
    if teacher_forcing_model is not None:
        save_teacher_forcing_weights(target, teacher_forcing_model)
    stats.save(target / "normalization.npz")
    config_payload = dict(experiment_config)
    config_payload["member_model_config"] = _plain(asdict(ensemble.member_config))
    config_payload["ensemble_config"] = _plain(asdict(ensemble.config))
    config_payload["risk_config"] = _plain(asdict(ensemble.risk_config))
    (target / "config.yaml").write_text(
        yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8"
    )
    schema = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "schema_version": schema_version,
        "state_dim": ensemble.member_config.state_dim,
        "action_dim": ensemble.member_config.action_dim,
        "history_horizon": ensemble.member_config.history_horizon,
        "ensemble_size": ensemble.config.ensemble_size,
        "normalization_sha256": stats.sha256(),
        "teacher_forcing_ablation": teacher_forcing_model is not None,
        "runtime_inputs": ["states", "past_actions", "valid_mask", "candidate_actions"],
        "forbidden_runtime_inputs": [
            "privileged_state",
            "braking_agent",
            "braking_time",
        ],
        "particle_member_semantics": "fixed_member_for_complete_trajectory",
    }
    _write_json(target / "schema.json", schema)
    _write_json(target / "dataset_manifest.json", dataset_manifest)
    _write_json(target / "bootstrap_manifest.json", bootstrap_manifest)
    _write_json(target / "metrics.json", metrics)
    _write_json(target / "provenance.json", provenance)
    return target


def load_rwm_u_checkpoint(
    directory: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_schema_version: str | None = None,
) -> tuple[RWMUEnsemble, dict[str, Any]]:
    source = Path(directory)
    required = (
        "config.yaml",
        "normalization.npz",
        "schema.json",
        "dataset_manifest.json",
        "bootstrap_manifest.json",
        "metrics.json",
        "provenance.json",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"RWM-U checkpoint is missing {missing}")
    payload = yaml.safe_load((source / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("checkpoint config.yaml root must be a mapping")
    for name in ("member_model_config", "ensemble_config", "risk_config"):
        if name not in payload:
            raise ValueError(f"checkpoint config.yaml has no {name}")
    member_payload = dict(payload["member_model_config"])
    for name in ("yaw_indices", "gripper_closed_indices"):
        if name in member_payload:
            member_payload[name] = tuple(member_payload[name])
    member_config = RWMARConfig(**member_payload)
    ensemble_config = RWMUEnsembleConfig(**dict(payload["ensemble_config"]))
    risk_config = RWMURiskConfig(**dict(payload["risk_config"]))
    stats = NormalizationStats.load(source / "normalization.npz")
    schema = json.loads((source / "schema.json").read_text(encoding="utf-8"))
    _validate_schema(
        schema,
        stats,
        member_config,
        ensemble_config,
        expected_schema_version=expected_schema_version,
    )
    ensemble = RWMUEnsemble.create(
        member_config, ensemble_config, stats, risk_config=risk_config
    )
    for index, member in enumerate(ensemble.members):
        path = source / "members" / f"member_{index:02d}.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"RWM-U checkpoint is missing {path.name}")
        incompatible = member.load_state_dict(
            load_file(path, device=str(device)), strict=True
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"strict member {index} load failed: {incompatible}")
    ensemble.to(device).eval()
    metadata = {
        "experiment_config": payload,
        "schema": schema,
        "dataset_manifest": _read_json(source / "dataset_manifest.json"),
        "bootstrap_manifest": _read_json(source / "bootstrap_manifest.json"),
        "metrics": _read_json(source / "metrics.json"),
        "provenance": _read_json(source / "provenance.json"),
        "normalization": stats,
    }
    return ensemble, metadata


def load_teacher_forcing_ablation(
    directory: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> RWMARWorldModel:
    source = Path(directory)
    payload = yaml.safe_load((source / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "member_model_config" not in payload:
        raise ValueError("checkpoint config has no member_model_config")
    model_payload = dict(payload["member_model_config"])
    for name in ("yaw_indices", "gripper_closed_indices"):
        if name in model_payload:
            model_payload[name] = tuple(model_payload[name])
    model = RWMARWorldModel(
        RWMARConfig(**model_payload),
        NormalizationStats.load(source / "normalization.npz"),
    )
    weights = source / "teacher_forcing.safetensors"
    if not weights.is_file():
        raise FileNotFoundError("checkpoint has no teacher_forcing.safetensors")
    incompatible = model.load_state_dict(
        load_file(weights, device=str(device)), strict=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict teacher-forcing load failed: {incompatible}")
    return model.to(device).eval()


def _validate_schema(
    schema: Mapping[str, Any],
    stats: NormalizationStats,
    member_config: RWMARConfig,
    ensemble_config: RWMUEnsembleConfig,
    *,
    expected_schema_version: str | None,
) -> None:
    if schema.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format {schema.get('format_version')!r}")
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
    if schema.get("state_dim") != member_config.state_dim or schema.get(
        "action_dim"
    ) != member_config.action_dim:
        raise ValueError("schema dimensions do not match member config")
    if schema.get("ensemble_size") != ensemble_config.ensemble_size:
        raise ValueError("schema ensemble size does not match ensemble config")


def _cpu_state_dict(model: RWMARWorldModel) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_plain(value), indent=2, sort_keys=True), encoding="utf-8")


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
    "load_rwm_u_checkpoint",
    "load_rwm_u_member_weights",
    "load_teacher_forcing_ablation",
    "load_teacher_forcing_weights",
    "save_rwm_u_checkpoint",
    "save_rwm_u_member_weights",
    "save_teacher_forcing_weights",
]
