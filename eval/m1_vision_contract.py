"""Shared fail-closed visual identity checks for Phase M1 evidence readers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from train.m1_checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    LEGACY_CHECKPOINT_FORMAT_VERSION,
)


RESNET18_ENCODER = "resnet18_imagenet1k_v1"


def is_dinov3_config(config: Mapping[str, Any]) -> bool:
    return str(_mapping(config, "initialization")["vision_backbone"]).startswith(
        "dinov3_"
    )


def allowed_checkpoint_formats(config: Mapping[str, Any]) -> frozenset[str]:
    """Return formats allowed for this config without conflating old and new M1."""

    encoder_name = str(_mapping(config, "initialization")["vision_backbone"])
    if encoder_name.startswith("dinov3_"):
        return frozenset({CHECKPOINT_FORMAT_VERSION})
    if encoder_name == RESNET18_ENCODER:
        # Existing accepted ResNet evidence is v1. A diagnostic re-save by the
        # current writer is v2, so both remain readable only under this explicit
        # historical config.
        return frozenset({LEGACY_CHECKPOINT_FORMAT_VERSION, CHECKPOINT_FORMAT_VERSION})
    raise ValueError(f"unsupported M1 vision_backbone {encoder_name!r}")


def vision_artifact_paths(
    config: Mapping[str, Any], *, project_root: Path
) -> tuple[Path, Path | None]:
    initialization = _mapping(config, "initialization")
    weights = _resolve(project_root, initialization["vision_weights"])
    config_path = None
    if is_dinov3_config(config):
        config_path = _resolve(project_root, initialization["vision_config"])
    return weights, config_path


def validate_source_artifacts(
    config: Mapping[str, Any], *, project_root: Path
) -> dict[str, Any]:
    """Verify the exact local source artifacts pinned by an M1 config."""

    initialization = _mapping(config, "initialization")
    weights, config_path = vision_artifact_paths(config, project_root=project_root)
    weights_sha256 = _file_sha256(weights)
    expected_weights = str(initialization["expected_vision_weights_sha256"])
    checks: dict[str, Any] = {
        "encoder_name": str(initialization["vision_backbone"]),
        "weights": str(weights),
        "weights_sha256": weights_sha256,
        "expected_weights_sha256": expected_weights,
        "weights_match": weights_sha256 == expected_weights,
        "config": None,
        "config_sha256": None,
        "expected_config_sha256": None,
        "config_match": True,
    }
    if config_path is not None:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("DINOv3 config.json root must be a mapping")
        config_sha256 = _canonical_json_sha256(raw)
        expected_config = str(initialization["expected_vision_config_sha256"])
        checks.update(
            {
                "config": str(config_path),
                "config_sha256": config_sha256,
                "expected_config_sha256": expected_config,
                "config_match": config_sha256 == expected_config,
            }
        )
    checks["passed"] = bool(checks["weights_match"] and checks["config_match"])
    return checks


def validate_training_summary_vision(
    summary: Mapping[str, Any], config: Mapping[str, Any], *, project_root: Path
) -> None:
    """Bind training evidence to the configured encoder and both DINO artifacts."""

    initialization = _mapping(config, "initialization")
    expected_weights = str(initialization["expected_vision_weights_sha256"])
    if summary.get("visual_backbone_sha256") != expected_weights:
        raise ValueError("training summary uses a different visual weights artifact")

    structured = summary.get("visual_backbone")
    if is_dinov3_config(config):
        if not isinstance(structured, Mapping):
            raise ValueError("DINOv3 training summary lacks structured visual identity")
        expected_config = str(initialization["expected_vision_config_sha256"])
        if (
            structured.get("encoder_name") != str(initialization["vision_backbone"])
            or structured.get("weights_sha256") != expected_weights
            or structured.get("config_sha256") != expected_config
        ):
            raise ValueError(
                "DINOv3 training summary visual identity differs from config"
            )
        weights, config_path = vision_artifact_paths(config, project_root=project_root)
        if _resolve(project_root, structured.get("weights")) != weights:
            raise ValueError("DINOv3 training summary weights path differs from config")
        if (
            config_path is None
            or _resolve(project_root, structured.get("config")) != config_path
        ):
            raise ValueError("DINOv3 training summary config path differs from config")
        return

    # The historical v1 summary predates the structured identity block. When a
    # current writer supplies it for ResNet, still reject contradictory values.
    if isinstance(structured, Mapping) and (
        structured.get("encoder_name") != RESNET18_ENCODER
        or structured.get("weights_sha256") != expected_weights
        or structured.get("config_sha256") is not None
    ):
        raise ValueError("ResNet training summary visual identity differs from config")


def training_summary_vision_payload(
    config: Mapping[str, Any], *, project_root: Path
) -> dict[str, Any]:
    """Build the structured visual identity used by diagnostic summary writers."""

    initialization = _mapping(config, "initialization")
    weights, config_path = vision_artifact_paths(config, project_root=project_root)
    return {
        "encoder_name": str(initialization["vision_backbone"]),
        "weights": str(weights),
        "weights_sha256": str(initialization["expected_vision_weights_sha256"]),
        "config": None if config_path is None else str(config_path),
        "config_sha256": (
            None
            if config_path is None
            else str(initialization["expected_vision_config_sha256"])
        ),
    }


def validate_loaded_checkpoint_vision(
    config: Mapping[str, Any], model: Any, metadata: Mapping[str, Any]
) -> None:
    """Validate a reconstructed teacher and its schema against the selected config."""

    initialization = _mapping(config, "initialization")
    model_config = _mapping(config, "model")
    schema = _mapping(metadata, "schema")
    format_version = str(schema.get("format_version", ""))
    if format_version not in allowed_checkpoint_formats(config):
        raise ValueError(
            f"M1 checkpoint format {format_version!r} is invalid for "
            f"{initialization['vision_backbone']!r}"
        )
    expected_weights = str(initialization["expected_vision_weights_sha256"])
    encoder = getattr(model, "vision_encoder", None)
    if encoder is None or getattr(encoder, "artifact_sha256", None) != expected_weights:
        raise ValueError("M1 checkpoint uses a different visual weights artifact")
    if schema.get("vision_source_sha256") != expected_weights:
        raise ValueError("M1 checkpoint schema uses a different visual source hash")
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError("M1 checkpoint visual encoder is not frozen")

    if not is_dinov3_config(config):
        encoder_config = getattr(encoder, "config", None)
        raw_size = int(model_config["vision_input_size"])
        if (
            encoder_config is None
            or int(getattr(encoder_config, "crop_size", -1)) != raw_size
            or int(getattr(encoder_config, "resize_shorter_side", -1)) != raw_size
        ):
            raise ValueError("ResNet checkpoint preprocessing differs from config")
        if format_version == CHECKPOINT_FORMAT_VERSION:
            identity = schema.get("vision_encoder")
            if not isinstance(identity, Mapping) or (
                identity.get("family") != "resnet18"
                or identity.get("encoder_name") != RESNET18_ENCODER
                or identity.get("weights_sha256") != expected_weights
                or int(identity.get("input_size", -1)) != raw_size
            ):
                raise ValueError("M1 v2 ResNet schema identity differs from config")
        return

    expected_identity = {
        "family": "dinov3",
        "encoder_name": str(initialization["vision_backbone"]),
        "weights_sha256": expected_weights,
        "config_sha256": str(initialization["expected_vision_config_sha256"]),
        "model_id": str(initialization["vision_model_id"]),
        "revision": str(initialization["vision_revision"]),
        "preprocess_id": str(initialization["vision_preprocess"]),
        "input_size": int(model_config["vision_encoder_input_size"]),
        "output_dim": int(model_config["vision_patch_dim"]),
    }
    identity = schema.get("vision_encoder")
    if not isinstance(identity, Mapping) or any(
        identity.get(key) != value for key, value in expected_identity.items()
    ):
        raise ValueError("DINOv3 checkpoint schema identity differs from config")
    encoder_config = getattr(encoder, "config", None)
    observed_encoder = {
        "encoder_name": getattr(encoder_config, "encoder_name", None),
        "model_id": getattr(encoder_config, "model_id", None),
        "revision": getattr(encoder_config, "revision", None),
        "preprocess_id": getattr(encoder_config, "preprocess_id", None),
        "input_size": int(getattr(encoder_config, "input_size", -1)),
        "config_sha256": getattr(encoder, "config_sha256", None),
        "output_dim": int(getattr(encoder, "output_dim", -1)),
    }
    if any(
        observed_encoder.get(key) != expected_identity[key] for key in observed_encoder
    ):
        raise ValueError("loaded DINOv3 encoder identity differs from config")


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"M1 {key!r} must be a mapping")
    return result


def _resolve(project_root: Path, value: Any) -> Path:
    if value is None:
        raise ValueError("M1 artifact path is missing")
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "allowed_checkpoint_formats",
    "is_dinov3_config",
    "training_summary_vision_payload",
    "validate_loaded_checkpoint_vision",
    "validate_source_artifacts",
    "validate_training_summary_vision",
    "vision_artifact_paths",
]
