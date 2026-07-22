"""Fail-closed checkpoint I/O for scratch M1 without legacy artifacts."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np
from safetensors.torch import load_file, save_file
import torch
from torch import Tensor, nn

from models.wam import AffineActionCodecConfig, NormalizationStats
from train.m1_scratch_builder import (
    ScratchM1BuildConfig,
    ScratchM1Bundle,
    build_scratch_m1,
    vision_encoder_identity,
)


SCRATCH_CHECKPOINT_FORMAT = "wam.multimodal.m1.scratch_checkpoint/1"
LATENT_WAM_WEIGHTS = "latent_wam.safetensors"
ACTION_FLOW_WEIGHTS = "action_flow.safetensors"
NORMALIZATION_FILE = "normalization.npz"
ACTION_CODEC_FILE = "action_codec.json"
BUILD_CONFIG_FILE = "build_config.json"
DATASET_LINEAGE_FILE = "dataset_lineage.json"
STAGE_STATE_FILE = "stage_state.json"
METRICS_FILE = "metrics.json"
PROVENANCE_FILE = "provenance.json"


def save_scratch_m1_checkpoint(
    directory: str | Path,
    bundle: ScratchM1Bundle,
    *,
    dataset_lineage: Mapping[str, Any],
    stage_state: Mapping[str, Any],
    metrics: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Save task-side weights and immutable codec/data/vision identities."""

    _validate_bundle(bundle)
    target = Path(directory)
    if target.exists():
        if not target.is_dir():
            raise FileExistsError(f"checkpoint target is not a directory: {target}")
        if any(target.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite non-empty checkpoint {target}"
            )
    target.mkdir(parents=True, exist_ok=True)
    _atomic_safetensors(
        bundle.model,
        target / LATENT_WAM_WEIGHTS,
        excluded_prefixes=("vision_encoder.",),
    )
    _atomic_safetensors(bundle.action_flow, target / ACTION_FLOW_WEIGHTS)
    _atomic_normalization(bundle.normalization, target / NORMALIZATION_FILE)
    _atomic_json(target / ACTION_CODEC_FILE, bundle.action_codec.config.to_dict())
    build_payload = {
        "build_config": bundle.build_config.to_dict(),
        "initialization_hashes": dict(bundle.initialization_hashes),
        "vision_identity": bundle.vision_identity,
    }
    _atomic_json(target / BUILD_CONFIG_FILE, build_payload)
    _atomic_json(target / DATASET_LINEAGE_FILE, dataset_lineage)
    _atomic_json(target / STAGE_STATE_FILE, stage_state)
    _atomic_json(target / METRICS_FILE, metrics or {})
    _atomic_json(target / PROVENANCE_FILE, provenance or {})
    artifact_names = (
        LATENT_WAM_WEIGHTS,
        ACTION_FLOW_WEIGHTS,
        NORMALIZATION_FILE,
        ACTION_CODEC_FILE,
        BUILD_CONFIG_FILE,
        DATASET_LINEAGE_FILE,
        STAGE_STATE_FILE,
        METRICS_FILE,
        PROVENANCE_FILE,
    )
    schema = {
        "format_version": SCRATCH_CHECKPOINT_FORMAT,
        "model": "multimodal_m1_scratch",
        "initialization_mode": "scratch",
        "train_seed": int(bundle.build_config.seed),
        "state_dim": int(bundle.model.world_model.config.state_dim),
        "action_dim": int(bundle.action_flow.config.action_dim),
        "action_domain": bundle.action_codec.config.encoded_domain,
        "action_codec_sha256": bundle.action_codec.semantic_sha256,
        "normalization_sha256": bundle.normalization.sha256(),
        "frozen_visual_backbone": bundle.model.vision_encoder is not None,
        "vision_bundled": False,
        "vision_identity": bundle.vision_identity,
        "action_anchor_mode": bundle.action_flow.config.anchor_mode,
        "legacy_weight_files": [],
        "weight_files": {
            "latent_wam": LATENT_WAM_WEIGHTS,
            "action_flow": ACTION_FLOW_WEIGHTS,
        },
        "artifact_sha256": {
            name: _sha256(target / name) for name in artifact_names
        },
    }
    _atomic_json(target / "schema.json", schema)
    return target


def load_scratch_m1_checkpoint(
    directory: str | Path,
    *,
    vision_encoder: nn.Module | None,
    device: str | torch.device = "cpu",
) -> tuple[ScratchM1Bundle, dict[str, Any]]:
    """Strictly rebuild a scratch checkpoint using a verified frozen teacher."""

    source = Path(directory)
    schema_path = source / "schema.json"
    if not schema_path.is_file():
        raise FileNotFoundError("scratch M1 checkpoint is missing schema.json")
    schema = _read_json(schema_path)
    if schema.get("format_version") != SCRATCH_CHECKPOINT_FORMAT:
        raise ValueError("unsupported scratch M1 checkpoint format")
    if schema.get("initialization_mode") != "scratch":
        raise ValueError("checkpoint is not a scratch initialization")
    hashes = schema.get("artifact_sha256")
    if not isinstance(hashes, Mapping) or not hashes:
        raise ValueError("scratch checkpoint artifact hashes are invalid")
    for raw_name, expected in hashes.items():
        name = str(raw_name)
        if Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError("scratch checkpoint contains an unsafe artifact path")
        path = source / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"scratch checkpoint artifact is missing: {name}")
        if _sha256(path) != str(expected):
            raise ValueError(f"scratch checkpoint artifact hash mismatch: {name}")

    normalization = NormalizationStats.load(source / NORMALIZATION_FILE)
    if normalization.sha256() != str(schema.get("normalization_sha256", "")):
        raise ValueError("scratch checkpoint normalization hash mismatch")
    codec_config = AffineActionCodecConfig.load(source / ACTION_CODEC_FILE)
    if codec_config.sha256() != str(schema.get("action_codec_sha256", "")):
        raise ValueError("scratch checkpoint action codec hash mismatch")
    build_payload = _read_json(source / BUILD_CONFIG_FILE)
    build_config = ScratchM1BuildConfig.from_dict(
        _mapping(build_payload, "build_config")
    )
    if int(schema.get("train_seed", -1)) != build_config.seed:
        raise ValueError("scratch checkpoint seed differs between schema/config")
    expected_vision = build_payload.get("vision_identity")
    observed_vision = vision_encoder_identity(vision_encoder)
    if expected_vision != observed_vision or schema.get("vision_identity") != observed_vision:
        raise ValueError("scratch checkpoint vision identity mismatch")
    bundle = build_scratch_m1(
        build_config,
        normalization,
        codec_config,
        vision_encoder=vision_encoder,
    )
    expected_initial = build_payload.get("initialization_hashes")
    if expected_initial != bundle.initialization_hashes:
        raise ValueError("scratch initialization hashes are not reproducible")

    model_state = load_file(source / LATENT_WAM_WEIGHTS, device=str(device))
    incompatible = bundle.model.load_state_dict(model_state, strict=False)
    expected_missing = {
        name
        for name in bundle.model.state_dict()
        if name.startswith("vision_encoder.")
    }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "strict scratch latent reload failed: "
            f"missing={sorted(incompatible.missing_keys)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    incompatible = bundle.action_flow.load_state_dict(
        load_file(source / ACTION_FLOW_WEIGHTS, device=str(device)), strict=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict scratch action-flow reload failed: {incompatible}")
    bundle.to(device)
    bundle.model.eval()
    bundle.action_flow.eval()
    _validate_bundle(bundle)
    metadata = {
        "schema": schema,
        "dataset_lineage": _read_json(source / DATASET_LINEAGE_FILE),
        "stage_state": _read_json(source / STAGE_STATE_FILE),
        "metrics": _read_json(source / METRICS_FILE),
        "provenance": _read_json(source / PROVENANCE_FILE),
    }
    return bundle, metadata


def scratch_checkpoint_tree_sha256(directory: str | Path) -> str:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(root)
    tree = {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }
    if not tree:
        raise ValueError("scratch checkpoint tree is empty")
    payload = json.dumps(tree, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_bundle(bundle: ScratchM1Bundle) -> None:
    if bundle.action_flow.has_anchor or bundle.action_flow.config.anchor_mode != "none":
        raise ValueError("scratch checkpoint forbids an embedded action anchor")
    if bundle.action_codec.action_dim != bundle.action_flow.config.action_dim:
        raise ValueError("scratch action codec/flow dimensions differ")
    if bundle.model.world_model.config.action_dim != bundle.action_flow.config.action_dim:
        raise ValueError("scratch world/flow action dimensions differ")
    if bundle.model.vision_encoder is not None:
        if any(
            parameter.requires_grad
            for parameter in bundle.model.vision_encoder.parameters()
        ):
            raise ValueError("scratch checkpoint vision encoder is not frozen")
        if vision_encoder_identity(bundle.model.vision_encoder) != bundle.vision_identity:
            raise ValueError("scratch checkpoint vision identity drifted")


def _atomic_safetensors(
    module: nn.Module,
    path: Path,
    *,
    excluded_prefixes: tuple[str, ...] = (),
) -> None:
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in module.state_dict().items()
        if not name.startswith(excluded_prefixes)
    }
    temporary = _temporary(path)
    try:
        save_file(state, temporary, metadata={"format": SCRATCH_CHECKPOINT_FORMAT})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_normalization(stats: NormalizationStats, path: Path) -> None:
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.npz")
    try:
        stats.save(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = _temporary(path)
    try:
        temporary.write_text(
            json.dumps(_plain(payload), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ValueError(f"checkpoint field {key!r} must be a mapping")
    return item


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    return value


__all__ = [
    "SCRATCH_CHECKPOINT_FORMAT",
    "load_scratch_m1_checkpoint",
    "save_scratch_m1_checkpoint",
    "scratch_checkpoint_tree_sha256",
]
