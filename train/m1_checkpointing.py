"""Portable, fail-closed checkpoint I/O for the Phase M1 latent WAM."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
import uuid

import numpy as np
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
from models.wam_multimodal import (
    DINOV3_ENCODER_SPECS,
    FrozenDINOv3Config,
    FrozenDINOv3Encoder,
    FrozenResNet18Config,
    FrozenResNet18Encoder,
    LatentWAM,
    LatentWAMConfig,
)


LEGACY_CHECKPOINT_FORMAT_VERSION = "wam.multimodal_joint_wam/1"
CHECKPOINT_FORMAT_VERSION = "wam.multimodal_joint_wam/2"
LATENT_WAM_WEIGHTS = "latent_wam.safetensors"
ACTION_FLOW_WEIGHTS = "action_flow.safetensors"
LEGACY_WORLD_WEIGHTS = "legacy_world_model.safetensors"
LEGACY_FLOW_WEIGHTS = "legacy_action_flow.safetensors"
VISION_SOURCE_WEIGHTS = "vision_source_weights.pth"
DINOV3_SOURCE_WEIGHTS = "vision_source_weights.safetensors"
DINOV3_SOURCE_CONFIG = "vision_source_config.json"
NORMALIZATION_FILE = "normalization.npz"
_BASE_HASHED_ARTIFACTS = (
    LATENT_WAM_WEIGHTS,
    ACTION_FLOW_WEIGHTS,
    LEGACY_WORLD_WEIGHTS,
    LEGACY_FLOW_WEIGHTS,
    NORMALIZATION_FILE,
    "config.yaml",
    "dataset_manifest.json",
    "metrics.json",
    "provenance.json",
)
_LEGACY_HASHED_ARTIFACTS = (
    LATENT_WAM_WEIGHTS,
    ACTION_FLOW_WEIGHTS,
    LEGACY_WORLD_WEIGHTS,
    LEGACY_FLOW_WEIGHTS,
    VISION_SOURCE_WEIGHTS,
    NORMALIZATION_FILE,
    "config.yaml",
    "dataset_manifest.json",
    "metrics.json",
    "provenance.json",
)


def save_m1_checkpoint(
    directory: str | Path,
    model: LatentWAM,
    action_flow: StatefulActionFlow,
    legacy_world_model: RWMARWorldModel,
    legacy_action_flow: StatefulActionFlow,
    normalization: NormalizationStats,
    *,
    experiment_config: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    provenance: Mapping[str, Any],
    schema_version: str,
    train_seed: int,
    model_variant: str,
) -> Path:
    """Save every runtime dependency, including the verified backbone source."""

    canonical_variant = _canonical_model_variant(model)
    if str(model_variant) != canonical_variant:
        raise ValueError(
            f"checkpoint variant {model_variant!r} does not match model "
            f"contract {canonical_variant!r}"
        )
    _validate_contract(
        model,
        action_flow,
        legacy_world_model,
        legacy_action_flow,
        normalization=normalization,
    )
    if model.vision_encoder is None:
        raise ValueError("formal M1 checkpoints must bundle the visual teacher")
    action_flow.freeze_anchor()
    legacy_action_flow.freeze_anchor()
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    vision_payload, vision_identity, vision_artifacts = _bundle_vision_encoder(
        target, model.vision_encoder
    )
    _atomic_safetensors(
        model,
        target / LATENT_WAM_WEIGHTS,
        "latent_wam",
        excluded_prefixes=("vision_encoder.",),
    )
    _atomic_safetensors(action_flow, target / ACTION_FLOW_WEIGHTS, "action_flow")
    _atomic_safetensors(
        legacy_world_model, target / LEGACY_WORLD_WEIGHTS, "legacy_world_model"
    )
    _atomic_safetensors(
        legacy_action_flow, target / LEGACY_FLOW_WEIGHTS, "legacy_action_flow"
    )
    _atomic_normalization(normalization, target / NORMALIZATION_FILE)
    payload = dict(experiment_config)
    payload.update(
        {
            "latent_wam_config": model.config.to_dict(),
            "world_model_config": asdict(model.world_model.config),
            "action_flow_config": asdict(action_flow.config),
            "legacy_world_model_config": asdict(legacy_world_model.config),
            "legacy_action_flow_config": asdict(legacy_action_flow.config),
            "vision_encoder_config": vision_payload,
            "train_seed": int(train_seed),
            "model_variant": str(model_variant),
        }
    )
    _atomic_text(
        target / "config.yaml", yaml.safe_dump(_plain(payload), sort_keys=False)
    )
    _atomic_json(target / "dataset_manifest.json", dataset_manifest)
    _atomic_json(target / "metrics.json", metrics)
    _atomic_json(target / "provenance.json", provenance)
    runtime_inputs = ["task", "past_executed_actions"]
    if model.config.use_vision:
        runtime_inputs.append("images.fixed")
    if model.config.use_state:
        runtime_inputs.append("proprioception")
    schema = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "schema_version": str(schema_version),
        "model": "multimodal_joint_wam",
        "model_variant": canonical_variant,
        "train_seed": int(train_seed),
        "runtime_inputs": runtime_inputs,
        "forbidden_runtime_inputs": [
            "privileged_state",
            "cue_id",
            "cue_variant",
            "rendered_cue_variant",
            "event_truth",
            "future_images",
        ],
        "fallback_required": False,
        "single_active_expert": True,
        "action_chunk_horizon": int(action_flow.config.horizon),
        "future_visual_horizons": list(model.future_horizons),
        "embedded_frozen_action_prior_anchor": True,
        "frozen_visual_backbone": True,
        "normalization_sha256": normalization.sha256(),
        "vision_encoder": vision_identity,
        "vision_contract_sha256": _canonical_mapping_sha256(vision_identity),
        "vision_source_sha256": model.vision_encoder.artifact_sha256,
        "latent_wam_excluded_prefixes": ["vision_encoder."],
        "weight_files": {
            "latent_wam": LATENT_WAM_WEIGHTS,
            "action_flow": ACTION_FLOW_WEIGHTS,
            "legacy_world_model": LEGACY_WORLD_WEIGHTS,
            "legacy_action_flow": LEGACY_FLOW_WEIGHTS,
        },
        "artifact_sha256": {
            name: _sha256(target / name)
            for name in (*_BASE_HASHED_ARTIFACTS, *vision_artifacts)
        },
    }
    _atomic_json(target / "schema.json", schema)
    return target


def load_m1_checkpoint(
    directory: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_schema_version: str | None = None,
) -> tuple[
    LatentWAM,
    StatefulActionFlow,
    RWMARWorldModel,
    StatefulActionFlow,
    dict[str, Any],
]:
    """Strictly reconstruct M1 using files located only in ``directory``."""

    source = Path(directory)
    if not (source / "schema.json").is_file():
        raise FileNotFoundError("M1 checkpoint is missing ['schema.json']")
    if (source / "schema.json").is_symlink():
        raise ValueError(
            "M1 checkpoint cannot depend on symlink artifacts: ['schema.json']"
        )
    schema = _read_json(source / "schema.json")
    format_version = str(schema.get("format_version", ""))
    hashes = schema.get("artifact_sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("M1 artifact hash manifest is invalid")
    artifact_names = tuple(str(name) for name in hashes)
    if any(
        Path(name).name != name or name in {"", ".", ".."} for name in artifact_names
    ):
        raise ValueError("M1 artifact manifest contains an unsafe path")
    missing = [name for name in artifact_names if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"M1 checkpoint is missing {missing}")
    symlinks = [name for name in artifact_names if (source / name).is_symlink()]
    if symlinks:
        raise ValueError(
            f"M1 checkpoint cannot depend on symlink artifacts: {symlinks}"
        )
    _validate_schema(schema, expected_schema_version=expected_schema_version)
    for name in artifact_names:
        if schema["artifact_sha256"].get(name) != _sha256(source / name):
            raise ValueError(f"M1 checkpoint artifact fingerprint mismatch: {name}")
    normalization = NormalizationStats.load(source / NORMALIZATION_FILE)
    if normalization.sha256() != schema["normalization_sha256"]:
        raise ValueError("M1 normalization fingerprint mismatch")
    payload = yaml.safe_load((source / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("M1 config root must be a mapping")
    world = RWMARWorldModel(
        RWMARConfig(**_mapping(payload, "world_model_config")), normalization
    )
    vision_payload = _mapping(payload, "vision_encoder_config")
    if format_version == LEGACY_CHECKPOINT_FORMAT_VERSION:
        vision = _load_legacy_resnet_encoder(source, vision_payload)
    else:
        vision = _load_v2_vision_encoder(source, vision_payload, schema)
    model = LatentWAM(
        LatentWAMConfig.from_dict(_mapping(payload, "latent_wam_config")),
        world,
        vision,
    )
    canonical_variant = _canonical_model_variant(model)
    if payload.get("model_variant") != canonical_variant:
        raise ValueError("M1 config model_variant does not match latent model config")
    if schema.get("model_variant") != canonical_variant:
        raise ValueError("M1 schema model_variant does not match latent model config")
    if int(payload.get("train_seed", -1)) != int(schema.get("train_seed", -2)):
        raise ValueError("M1 train seed differs between config and schema")
    action_flow = StatefulActionFlow(
        StatefulActionFlowConfig(**_mapping(payload, "action_flow_config")),
        normalization,
    )
    legacy_world = RWMARWorldModel(
        RWMARConfig(**_mapping(payload, "legacy_world_model_config")), normalization
    )
    legacy_flow = StatefulActionFlow(
        StatefulActionFlowConfig(**_mapping(payload, "legacy_action_flow_config")),
        normalization,
    )
    if format_version == LEGACY_CHECKPOINT_FORMAT_VERSION:
        incompatible = model.load_state_dict(
            load_file(source / LATENT_WAM_WEIGHTS, device=str(device)), strict=True
        )
    else:
        incompatible = model.load_state_dict(
            load_file(source / LATENT_WAM_WEIGHTS, device=str(device)), strict=False
        )
        expected_missing = {
            name
            for name in model.state_dict()
            if name.startswith("vision_encoder.")
            and not name.endswith(".num_batches_tracked")
        }
        if set(incompatible.missing_keys) != expected_missing:
            raise RuntimeError(
                "strict M1 latent WAM reload has invalid excluded vision keys: "
                f"missing={sorted(incompatible.missing_keys)}, "
                f"expected={sorted(expected_missing)}"
            )
    if incompatible.unexpected_keys or (
        format_version == LEGACY_CHECKPOINT_FORMAT_VERSION and incompatible.missing_keys
    ):
        raise RuntimeError(f"strict M1 latent WAM reload failed: {incompatible}")
    model.to(device).eval()
    modules = (
        (action_flow, ACTION_FLOW_WEIGHTS, "action flow"),
        (legacy_world, LEGACY_WORLD_WEIGHTS, "legacy world model"),
        (legacy_flow, LEGACY_FLOW_WEIGHTS, "legacy action flow"),
    )
    for module, filename, label in modules:
        incompatible = module.load_state_dict(
            load_file(source / filename, device=str(device)), strict=True
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"strict M1 {label} reload failed: {incompatible}")
        module.to(device).eval()
    action_flow.freeze_anchor()
    legacy_flow.freeze_anchor()
    _validate_contract(
        model,
        action_flow,
        legacy_world,
        legacy_flow,
        normalization=normalization,
    )
    if any(parameter.requires_grad for parameter in model.vision_encoder.parameters()):
        raise RuntimeError("reloaded M1 backbone is not frozen")
    expected_runtime_inputs = ["task", "past_executed_actions"]
    if model.config.use_vision:
        expected_runtime_inputs.append("images.fixed")
    if model.config.use_state:
        expected_runtime_inputs.append("proprioception")
    if schema.get("runtime_inputs") != expected_runtime_inputs:
        raise ValueError("M1 runtime input contract does not match model modalities")
    if schema.get("vision_source_sha256") != vision.artifact_sha256:
        raise ValueError("M1 schema vision fingerprint mismatch")
    if format_version == CHECKPOINT_FORMAT_VERSION:
        observed_identity = _vision_identity(vision)
        if schema.get("vision_encoder") != observed_identity:
            raise ValueError("M1 schema vision identity mismatch")
        if schema.get("vision_contract_sha256") != _canonical_mapping_sha256(
            observed_identity
        ):
            raise ValueError("M1 schema vision contract fingerprint mismatch")
    return (
        model,
        action_flow,
        legacy_world,
        legacy_flow,
        {
            "experiment_config": dict(payload),
            "schema": schema,
            "dataset_manifest": _read_json(source / "dataset_manifest.json"),
            "metrics": _read_json(source / "metrics.json"),
            "provenance": _read_json(source / "provenance.json"),
            "normalization": normalization,
        },
    )


def checkpoint_tree_sha256(directory: str | Path) -> str:
    """Return the canonical M0-compatible digest of a checkpoint tree."""

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(root)
    tree: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"checkpoint tree cannot contain symlinks: {path}")
        if path.is_file():
            tree[str(path.relative_to(root))] = _sha256(path)
    if not tree:
        raise ValueError("checkpoint tree is empty")
    serialized = json.dumps(
        dict(sorted(tree.items())), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _bundle_vision_encoder(
    target: Path, encoder: torch.nn.Module
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    if isinstance(encoder, FrozenResNet18Encoder):
        _atomic_link_or_copy(
            Path(encoder.config.weights_path), target / VISION_SOURCE_WEIGHTS
        )
        if _sha256(target / VISION_SOURCE_WEIGHTS) != encoder.artifact_sha256:
            raise ValueError("bundled M1 ResNet source hash changed during save")
        payload = {
            "family": "resnet18",
            "encoder_name": "resnet18_imagenet1k_v1",
            "weights_file": VISION_SOURCE_WEIGHTS,
            "expected_sha256": encoder.artifact_sha256,
            "resize_shorter_side": encoder.config.resize_shorter_side,
            "crop_size": encoder.config.crop_size,
        }
        artifacts = (VISION_SOURCE_WEIGHTS,)
    elif isinstance(encoder, FrozenDINOv3Encoder):
        _atomic_link_or_copy(
            Path(encoder.config.weights_path), target / DINOV3_SOURCE_WEIGHTS
        )
        _atomic_link_or_copy(
            Path(encoder.config.config_path), target / DINOV3_SOURCE_CONFIG
        )
        if _sha256(target / DINOV3_SOURCE_WEIGHTS) != encoder.artifact_sha256:
            raise ValueError("bundled M1 DINOv3 source hash changed during save")
        payload = {
            "family": "dinov3",
            "encoder_name": encoder.config.encoder_name,
            "model_id": encoder.config.model_id,
            "revision": encoder.config.revision,
            "weights_file": DINOV3_SOURCE_WEIGHTS,
            "config_file": DINOV3_SOURCE_CONFIG,
            "expected_weights_sha256": encoder.artifact_sha256,
            "expected_config_sha256": encoder.config_sha256,
            "input_size": encoder.config.input_size,
            "preprocess_id": encoder.config.preprocess_id,
            "inference_batch_size": encoder.config.inference_batch_size,
        }
        artifacts = (DINOV3_SOURCE_WEIGHTS, DINOV3_SOURCE_CONFIG)
    else:
        raise TypeError(f"unsupported frozen vision encoder {type(encoder).__name__}")
    return payload, _vision_identity(encoder), artifacts


def _load_legacy_resnet_encoder(
    source: Path, payload: Mapping[str, Any]
) -> FrozenResNet18Encoder:
    if Path(str(payload.get("weights_file", ""))).name != VISION_SOURCE_WEIGHTS:
        raise ValueError("legacy M1 vision source must resolve inside its checkpoint")
    return FrozenResNet18Encoder(
        FrozenResNet18Config(
            weights_path=source / VISION_SOURCE_WEIGHTS,
            expected_sha256=str(payload["expected_sha256"]),
            resize_shorter_side=int(payload["resize_shorter_side"]),
            crop_size=int(payload["crop_size"]),
        )
    )


def _load_v2_vision_encoder(
    source: Path,
    payload: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> torch.nn.Module:
    family = payload.get("family")
    if family == "resnet18":
        if payload.get("encoder_name") != "resnet18_imagenet1k_v1":
            raise ValueError("M1 ResNet encoder identity is invalid")
        return _load_legacy_resnet_encoder(source, payload)
    if family != "dinov3":
        raise ValueError("M1 v2 vision encoder family is invalid")
    if payload.get("weights_file") != DINOV3_SOURCE_WEIGHTS:
        raise ValueError("M1 DINOv3 weights must resolve inside its checkpoint")
    if payload.get("config_file") != DINOV3_SOURCE_CONFIG:
        raise ValueError("M1 DINOv3 config must resolve inside its checkpoint")
    encoder_name = str(payload.get("encoder_name", ""))
    if encoder_name not in DINOV3_ENCODER_SPECS:
        raise ValueError("M1 DINOv3 encoder alias is unknown")
    encoder = FrozenDINOv3Encoder(
        FrozenDINOv3Config(
            encoder_name=encoder_name,
            model_id=str(payload["model_id"]),
            revision=str(payload["revision"]),
            config_path=source / DINOV3_SOURCE_CONFIG,
            weights_path=source / DINOV3_SOURCE_WEIGHTS,
            expected_weights_sha256=str(payload["expected_weights_sha256"]),
            expected_config_sha256=str(payload["expected_config_sha256"]),
            input_size=int(payload["input_size"]),
            preprocess_id=str(payload["preprocess_id"]),
            inference_batch_size=int(payload["inference_batch_size"]),
        )
    )
    if schema.get("vision_encoder") != _vision_identity(encoder):
        raise ValueError("M1 DINOv3 config and schema identities differ")
    return encoder


def _vision_identity(encoder: torch.nn.Module) -> dict[str, Any]:
    if isinstance(encoder, FrozenResNet18Encoder):
        return {
            "family": "resnet18",
            "encoder_name": "resnet18_imagenet1k_v1",
            "weights_sha256": encoder.artifact_sha256,
            "config_sha256": None,
            "model_id": "torchvision/resnet18-imagenet1k-v1",
            "revision": "f37072fd",
            "preprocess_id": "imagenet_rgb_resize_center_crop_antialias_v1",
            "input_size": int(encoder.config.crop_size),
            "output_dim": int(encoder.output_dim),
            "patch_size": 32,
            "register_tokens": 0,
            "spatial_tokens": "layer4_flattened_row_major",
            "pooled_latent": "spatial_patch_mean",
            "implementation": "fe_pc_wam.self_contained_resnet18_v1",
        }
    if isinstance(encoder, FrozenDINOv3Encoder):
        return {
            "family": "dinov3",
            "encoder_name": encoder.config.encoder_name,
            "weights_sha256": encoder.artifact_sha256,
            "config_sha256": encoder.config_sha256,
            "model_id": encoder.config.model_id,
            "revision": encoder.config.revision,
            "preprocess_id": encoder.config.preprocess_id,
            "input_size": int(encoder.config.input_size),
            "output_dim": int(encoder.output_dim),
            "patch_size": int(encoder.patch_size),
            "register_tokens": int(encoder.register_tokens),
            "spatial_tokens": "last_hidden_state_excluding_cls_and_registers",
            "pooled_latent": "normalized_cls_token",
            "implementation": str(encoder.implementation),
        }
    raise TypeError(f"unsupported frozen vision encoder {type(encoder).__name__}")


def _canonical_mapping_sha256(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        _plain(dict(payload)), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _canonical_model_variant(model: LatentWAM) -> str:
    """Map architecture switches to the five preregistered M1 contrasts."""

    config = model.config
    if config.use_state and not config.use_vision:
        if config.capacity_control != "none":
            raise ValueError("state-only M1 must not include a capacity control head")
        return "state_only"
    if config.use_vision and not config.use_state:
        if config.capacity_control != "none":
            raise ValueError("vision-only M1 must not include a capacity control head")
        return "vision_only"
    if config.use_state and config.use_vision:
        return {
            "none": "state_vision_no_future",
            "future_head": "state_vision_future",
            "action_mlp": "parameter_matched_mlp",
        }[config.capacity_control]
    raise ValueError("M1 model has no deployable modality")


def _validate_contract(
    model: LatentWAM,
    action_flow: StatefulActionFlow,
    legacy_world: RWMARWorldModel,
    legacy_flow: StatefulActionFlow,
    *,
    normalization: NormalizationStats | None = None,
) -> None:
    if model.planning_feature_dim != action_flow.config.feature_dim:
        raise ValueError("M1 planning/action feature dimensions differ")
    if action_flow.config.action_dim != model.world_model.config.action_dim:
        raise ValueError("M1 action dimensions differ")
    if legacy_flow.config.feature_dim != legacy_world.planning_feature_dim:
        raise ValueError("legacy planning/action feature dimensions differ")
    if legacy_flow.config.action_dim != legacy_world.config.action_dim:
        raise ValueError("legacy action dimensions differ")
    if action_flow.config.action_dim != legacy_flow.config.action_dim:
        raise ValueError("M1 and legacy action dimensions differ")
    if action_flow.config.horizon != 8 or legacy_flow.config.horizon != 8:
        raise ValueError("Phase M1 requires 8-step action chunks")
    if legacy_world.config != model.world_model.config:
        raise ValueError("M1 and legacy world model architectures differ")
    for flow in (action_flow, legacy_flow):
        if any(parameter.requires_grad for parameter in flow.anchor_prior.parameters()):
            raise ValueError("M1 action anchors must be frozen")
    if normalization is not None:
        _validate_normalization(
            model, action_flow, legacy_world, legacy_flow, normalization
        )


def _validate_normalization(
    model: LatentWAM,
    action_flow: StatefulActionFlow,
    legacy_world: RWMARWorldModel,
    legacy_flow: StatefulActionFlow,
    stats: NormalizationStats,
) -> None:
    """Ensure serialized stats are the exact stats embedded in every module."""

    comparisons: list[tuple[str, torch.Tensor, np.ndarray]] = []
    for label, world in (
        ("M1 world", model.world_model),
        ("legacy world", legacy_world),
    ):
        comparisons.extend(
            (
                (f"{label} state_mean", world.features.state_mean, stats.state_mean),
                (f"{label} state_std", world.features.state_std, stats.state_std),
                (f"{label} action_mean", world.features.action_mean, stats.action_mean),
                (f"{label} action_std", world.features.action_std, stats.action_std),
                (f"{label} delta_mean", world.delta_mean, stats.delta_mean),
                (f"{label} delta_std", world.delta_std, stats.delta_std),
            )
        )
    for label, flow in (("M1 flow", action_flow), ("legacy flow", legacy_flow)):
        comparisons.extend(
            (
                (f"{label} action_mean", flow.action_mean, stats.action_mean),
                (f"{label} action_std", flow.action_std, stats.action_std),
            )
        )
    for label, actual, expected in comparisons:
        expected_tensor = torch.as_tensor(
            expected, dtype=actual.dtype, device=actual.device
        )
        if actual.shape != expected_tensor.shape or not torch.equal(
            actual, expected_tensor
        ):
            raise ValueError(f"{label} differs from checkpoint normalization")


def _validate_schema(
    schema: Mapping[str, Any], *, expected_schema_version: str | None
) -> None:
    format_version = schema.get("format_version")
    if format_version not in {
        LEGACY_CHECKPOINT_FORMAT_VERSION,
        CHECKPOINT_FORMAT_VERSION,
    }:
        raise ValueError("unsupported M1 checkpoint format")
    if schema.get("model") != "multimodal_joint_wam":
        raise ValueError("checkpoint does not contain an M1 model")
    if (
        expected_schema_version is not None
        and schema.get("schema_version") != expected_schema_version
    ):
        raise ValueError("M1 data schema mismatch")
    required_true = (
        "single_active_expert",
        "embedded_frozen_action_prior_anchor",
        "frozen_visual_backbone",
    )
    if any(schema.get(name) is not True for name in required_true):
        raise ValueError("M1 checkpoint invariant is disabled")
    if schema.get("fallback_required") is not False:
        raise ValueError("M1 checkpoint requires an external fallback")
    if schema.get("action_chunk_horizon") != 8:
        raise ValueError("M1 checkpoint action chunk must have horizon 8")
    if schema.get("future_visual_horizons") != [1, 2, 4, 8]:
        raise ValueError("M1 future visual horizon contract changed")
    expected_weights = {
        "latent_wam": LATENT_WAM_WEIGHTS,
        "action_flow": ACTION_FLOW_WEIGHTS,
        "legacy_world_model": LEGACY_WORLD_WEIGHTS,
        "legacy_action_flow": LEGACY_FLOW_WEIGHTS,
    }
    if schema.get("weight_files") != expected_weights:
        raise ValueError("M1 weight file contract is invalid")
    forbidden = schema.get("forbidden_runtime_inputs")
    required_forbidden = {
        "privileged_state",
        "cue_id",
        "cue_variant",
        "rendered_cue_variant",
        "event_truth",
        "future_images",
    }
    if not isinstance(forbidden, list) or not required_forbidden.issubset(forbidden):
        raise ValueError("M1 forbidden runtime input contract is incomplete")
    hashes = schema.get("artifact_sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("M1 artifact hash manifest is invalid")
    if format_version == LEGACY_CHECKPOINT_FORMAT_VERSION:
        expected_artifacts = set(_LEGACY_HASHED_ARTIFACTS)
    else:
        identity = schema.get("vision_encoder")
        if not isinstance(identity, Mapping):
            raise ValueError("M1 v2 vision identity is missing")
        family = identity.get("family")
        expected_artifacts = set(_BASE_HASHED_ARTIFACTS)
        if family == "resnet18":
            expected_artifacts.add(VISION_SOURCE_WEIGHTS)
        elif family == "dinov3":
            expected_artifacts.update({DINOV3_SOURCE_WEIGHTS, DINOV3_SOURCE_CONFIG})
        else:
            raise ValueError("M1 v2 vision family is invalid")
        if schema.get("latent_wam_excluded_prefixes") != ["vision_encoder."]:
            raise ValueError("M1 v2 excluded vision-state contract is invalid")
        if schema.get("vision_contract_sha256") != _canonical_mapping_sha256(identity):
            raise ValueError("M1 v2 vision contract fingerprint is invalid")
    if set(hashes) != expected_artifacts:
        raise ValueError("M1 artifact hash manifest is invalid")
    if any(not _is_sha256(value) for value in hashes.values()):
        raise ValueError("M1 artifact hash is invalid")


def _mapping(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"M1 config field {name!r} must be a mapping")
    return dict(value)


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous()
        for name, value in module.state_dict().items()
    }


def _atomic_safetensors(
    module: torch.nn.Module,
    path: Path,
    family: str,
    *,
    excluded_prefixes: tuple[str, ...] = (),
) -> None:
    temporary = _temporary(path)
    try:
        state = _cpu_state_dict(module)
        if excluded_prefixes:
            state = {
                name: value
                for name, value in state.items()
                if not name.startswith(excluded_prefixes)
            }
        if not state:
            raise ValueError(f"{family} checkpoint state cannot be empty")
        save_file(
            state,
            temporary,
            metadata={
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "model_family": family,
            },
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_copy(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    temporary = _temporary(target)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_link_or_copy(source: Path, target: Path) -> None:
    """Atomically hard-link immutable artifacts, falling back to a copy."""

    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    temporary = _temporary(target)
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        os.replace(temporary, target)
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
    _atomic_text(
        path,
        json.dumps(_plain(payload), indent=2, sort_keys=True, allow_nan=False),
    )


def _atomic_text(path: Path, value: str) -> None:
    temporary = _temporary(path)
    try:
        temporary.write_text(value, encoding="utf-8")
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


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def _sha256(path: Path) -> str:
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


__all__ = [
    "ACTION_FLOW_WEIGHTS",
    "CHECKPOINT_FORMAT_VERSION",
    "LATENT_WAM_WEIGHTS",
    "NORMALIZATION_FILE",
    "checkpoint_tree_sha256",
    "load_m1_checkpoint",
    "save_m1_checkpoint",
]
