from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.torch import load_file, save_file
import torch
from torch import nn
from torch.nn import functional as F
from transformers import DINOv3ViTConfig, DINOv3ViTModel
import yaml

from models.wam import (
    NormalizationStats,
    RWMARConfig,
    RWMARWorldModel,
    StatefulActionFlow,
    StatefulActionFlowConfig,
)
from models.wam_multimodal import (
    DINOV3_PREPROCESS_ID,
    DINOv3EncoderSpec,
    FrozenDINOv3Config,
    FrozenDINOv3Encoder,
    IMAGENET_RGB_MEAN,
    IMAGENET_RGB_STD,
    LatentWAM,
    LatentWAMConfig,
    PerceiverResampler,
    PerceiverResamplerConfig,
    canonical_json_sha256,
    sha256_file,
)
from models.wam_multimodal import vision_encoder as vision_encoder_module
from train.m1_checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    DINOV3_SOURCE_CONFIG,
    DINOV3_SOURCE_WEIGHTS,
    LATENT_WAM_WEIGHTS,
    load_m1_checkpoint,
    save_m1_checkpoint,
)


_TINY_ENCODER = "dinov3_unit_vit16"
_TINY_MODEL_ID = "offline/dinov3-unit-vit16"
_TINY_REVISION = "1" * 40
_TINY_DIM = 32
_TINY_PATCH = 16
_TINY_REGISTERS = 4


@dataclass(frozen=True)
class _TinyDINOv3Artifact:
    config_path: Path
    config_sha256: str
    weights_path: Path
    weights_sha256: str
    state_dict: dict[str, torch.Tensor]

    def encoder_config(self, **overrides: Any) -> FrozenDINOv3Config:
        values: dict[str, Any] = {
            "encoder_name": _TINY_ENCODER,
            "model_id": _TINY_MODEL_ID,
            "revision": _TINY_REVISION,
            "config_path": self.config_path,
            "weights_path": self.weights_path,
            "expected_weights_sha256": self.weights_sha256,
            "expected_config_sha256": self.config_sha256,
            "input_size": 32,
            "preprocess_id": DINOV3_PREPROCESS_ID,
            "inference_batch_size": 2,
        }
        values.update(overrides)
        return FrozenDINOv3Config(**values)

    def encoder(self, **overrides: Any) -> FrozenDINOv3Encoder:
        return FrozenDINOv3Encoder(self.encoder_config(**overrides))


@pytest.fixture()
def tiny_dinov3_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _TinyDINOv3Artifact:
    spec = DINOv3EncoderSpec(
        name=_TINY_ENCODER,
        model_id=_TINY_MODEL_ID,
        output_dim=_TINY_DIM,
        patch_size=_TINY_PATCH,
        register_tokens=_TINY_REGISTERS,
        default_revision=_TINY_REVISION,
    )
    monkeypatch.setitem(
        vision_encoder_module.DINOV3_ENCODER_SPECS,
        _TINY_ENCODER,
        spec,
    )
    config = DINOv3ViTConfig(
        image_size=32,
        patch_size=_TINY_PATCH,
        num_channels=3,
        hidden_size=_TINY_DIM,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        num_register_tokens=_TINY_REGISTERS,
        attention_dropout=0.0,
        drop_path_rate=0.0,
    )
    config.architectures = ["DINOv3ViTModel"]
    config_path = tmp_path / "config.json"
    config.to_json_file(config_path, use_diff=False)
    with torch.random.fork_rng():
        torch.manual_seed(1701)
        backbone = DINOv3ViTModel(config).eval()
    state = {
        name: value.detach().cpu().contiguous().clone()
        for name, value in backbone.state_dict().items()
    }
    weights_path = tmp_path / "model.safetensors"
    save_file(state, weights_path)
    return _TinyDINOv3Artifact(
        config_path=config_path,
        config_sha256=canonical_json_sha256(
            json.loads(config_path.read_text(encoding="utf-8"))
        ),
        weights_path=weights_path,
        weights_sha256=sha256_file(weights_path),
        state_dict=state,
    )


def test_dinov3_loads_strictly_from_verified_local_artifacts(
    tiny_dinov3_artifact: _TinyDINOv3Artifact,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_remote_load(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("DINOv3 unit test attempted a remote model load")

    monkeypatch.setattr(DINOv3ViTModel, "from_pretrained", forbidden_remote_load)
    encoder = tiny_dinov3_artifact.encoder()

    assert encoder.family == "dinov3"
    assert encoder.encoder_name == _TINY_ENCODER
    assert encoder.config.model_id == _TINY_MODEL_ID
    assert encoder.config.revision == _TINY_REVISION
    assert encoder.output_dim == _TINY_DIM
    assert encoder.patch_size == _TINY_PATCH
    assert encoder.register_tokens == _TINY_REGISTERS
    assert encoder.patch_count == 4
    assert encoder.artifact_sha256 == tiny_dinov3_artifact.weights_sha256
    assert encoder.config_sha256 == tiny_dinov3_artifact.config_sha256
    assert set(encoder.backbone.state_dict()) == set(tiny_dinov3_artifact.state_dict)
    for name, expected in tiny_dinov3_artifact.state_dict.items():
        assert torch.equal(encoder.backbone.state_dict()[name], expected), name

    with pytest.raises(ValueError, match="official model id"):
        tiny_dinov3_artifact.encoder_config(model_id="offline/wrong-model")
    with pytest.raises(ValueError, match="full lowercase commit SHA"):
        tiny_dinov3_artifact.encoder_config(revision="main")


def test_dinov3_loads_official_unprefixed_safetensors_strictly(
    tiny_dinov3_artifact: _TinyDINOv3Artifact,
    tmp_path: Path,
) -> None:
    official_state = _official_unprefixed_state(tiny_dinov3_artifact.state_dict)
    weights_path = tmp_path / "official_unprefixed.safetensors"
    save_file(official_state, weights_path)

    encoder = FrozenDINOv3Encoder(
        tiny_dinov3_artifact.encoder_config(
            weights_path=weights_path,
            expected_weights_sha256=sha256_file(weights_path),
        )
    )

    loaded_state = encoder.backbone.state_dict()
    assert set(loaded_state) == set(tiny_dinov3_artifact.state_dict)
    for name, expected in tiny_dinov3_artifact.state_dict.items():
        assert torch.equal(loaded_state[name], expected), name


@pytest.mark.parametrize(
    "mutation",
    ("missing", "unexpected", "shape_tampered", "prefix_collision"),
)
def test_dinov3_official_unprefixed_safetensors_remain_strict(
    tiny_dinov3_artifact: _TinyDINOv3Artifact,
    tmp_path: Path,
    mutation: str,
) -> None:
    official_state = _official_unprefixed_state(tiny_dinov3_artifact.state_dict)
    if mutation == "missing":
        official_state.pop("layer.0.norm1.weight")
        expected_error = "Missing key"
    elif mutation == "unexpected":
        official_state["layer.0.unexpected.weight"] = torch.ones(1)
        expected_error = "Unexpected key"
    elif mutation == "shape_tampered":
        official_state["norm.weight"] = official_state["norm.weight"][:-1].clone()
        expected_error = "size mismatch"
    else:
        official_state["model.layer.0.norm1.weight"] = official_state[
            "layer.0.norm1.weight"
        ].clone()
        expected_error = "key normalization collision"
    weights_path = tmp_path / f"official_unprefixed_{mutation}.safetensors"
    save_file(official_state, weights_path)

    with pytest.raises(RuntimeError, match=expected_error):
        FrozenDINOv3Encoder(
            tiny_dinov3_artifact.encoder_config(
                weights_path=weights_path,
                expected_weights_sha256=sha256_file(weights_path),
            )
        )


def test_dinov3_uses_cls_and_excludes_register_tokens_from_spatial_output(
    tiny_dinov3_artifact: _TinyDINOv3Artifact,
) -> None:
    encoder = tiny_dinov3_artifact.encoder(inference_batch_size=8)
    images = torch.arange(2 * 3 * 32 * 32, dtype=torch.int64)
    images = images.remainder(256).to(torch.uint8).reshape(2, 3, 32, 32)

    prepared, leading_shape = encoder.preprocess(images)
    with torch.inference_mode():
        hidden = encoder.backbone(pixel_values=prepared).last_hidden_state
    output = encoder(images)

    assert leading_shape == (2,)
    assert hidden.shape == (2, 1 + _TINY_REGISTERS + 4, _TINY_DIM)
    torch.testing.assert_close(output.pooled_latent, hidden[:, 0, :])
    torch.testing.assert_close(
        output.spatial_tokens,
        hidden[:, 1 + _TINY_REGISTERS :, :],
    )
    assert output.spatial_tokens.shape == (2, 4, _TINY_DIM)


def test_dinov3_preprocess_and_leading_dimensions_are_exact(
    tiny_dinov3_artifact: _TinyDINOv3Artifact,
) -> None:
    encoder = tiny_dinov3_artifact.encoder(inference_batch_size=3)
    raw = torch.arange(2 * 2 * 2 * 3 * 20 * 28, dtype=torch.int64)
    raw = raw.remainder(256).to(torch.uint8).reshape(2, 2, 2, 3, 20, 28)
    prepared, leading_shape = encoder.preprocess(raw)
    expected = F.interpolate(
        raw.reshape(-1, 3, 20, 28).float() / 255.0,
        size=(32, 32),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    mean = torch.tensor(IMAGENET_RGB_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_RGB_STD).view(1, 3, 1, 1)
    expected = (expected - mean) / std

    assert leading_shape == (2, 2, 2)
    torch.testing.assert_close(prepared, expected, rtol=0.0, atol=0.0)
    integer_output = encoder(raw)
    float_output = encoder(raw.float() / 255.0)
    assert integer_output.spatial_tokens.shape == (2, 2, 2, 4, _TINY_DIM)
    assert integer_output.pooled_latent.shape == (2, 2, 2, _TINY_DIM)
    torch.testing.assert_close(
        integer_output.spatial_tokens,
        float_output.spatial_tokens,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        integer_output.pooled_latent,
        float_output.pooled_latent,
        rtol=0.0,
        atol=0.0,
    )

    invalid = torch.zeros(1, 3, 16, 16, dtype=torch.float32)
    invalid[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        encoder.preprocess(invalid)
    with pytest.raises(ValueError, match=r"scaled to \[0,1\]"):
        encoder.preprocess(torch.full((1, 3, 16, 16), 1.01))
    with pytest.raises(TypeError, match="uint8 or floating"):
        encoder.preprocess(torch.zeros(1, 3, 16, 16, dtype=torch.int64))


def test_dinov3_remains_frozen_eval_and_microbatching_is_consistent(
    tiny_dinov3_artifact: _TinyDINOv3Artifact,
) -> None:
    one_at_a_time = tiny_dinov3_artifact.encoder(inference_batch_size=1)
    all_at_once = tiny_dinov3_artifact.encoder(inference_batch_size=16)
    one_at_a_time.train(True)

    assert one_at_a_time.training is False
    assert one_at_a_time.backbone.training is False
    assert all(not module.training for module in one_at_a_time.backbone.modules())
    assert all(not parameter.requires_grad for parameter in one_at_a_time.parameters())

    images = torch.randint(0, 256, (7, 3, 32, 32), dtype=torch.uint8)
    small = one_at_a_time(images)
    large = all_at_once(images)
    repeated = one_at_a_time(images)
    torch.testing.assert_close(
        small.spatial_tokens,
        large.spatial_tokens,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        small.pooled_latent,
        large.pooled_latent,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        small.spatial_tokens,
        repeated.spatial_tokens,
        rtol=0.0,
        atol=0.0,
    )
    assert not small.spatial_tokens.requires_grad
    assert not small.pooled_latent.requires_grad


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model_type", "vit"),
        ("hidden_size", 48),
        ("patch_size", 8),
        ("num_register_tokens", 0),
        ("num_channels", 1),
        ("architectures", ["NotDINOv3"]),
    ),
)
def test_dinov3_rejects_wrong_hash_and_architecture_identity(
    tiny_dinov3_artifact: _TinyDINOv3Artifact,
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        FrozenDINOv3Encoder(
            tiny_dinov3_artifact.encoder_config(expected_weights_sha256="0" * 64)
        )

    payload = json.loads(tiny_dinov3_artifact.config_path.read_text(encoding="utf-8"))
    payload[field] = value
    invalid_config = tmp_path / f"invalid_{field}.json"
    invalid_config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="architecture identity mismatch"):
        FrozenDINOv3Encoder(
            tiny_dinov3_artifact.encoder_config(
                config_path=invalid_config,
                expected_config_sha256=canonical_json_sha256(payload),
            )
        )


def test_dinov3_strict_weight_loading_rejects_missing_tensor(
    tiny_dinov3_artifact: _TinyDINOv3Artifact,
    tmp_path: Path,
) -> None:
    state = dict(tiny_dinov3_artifact.state_dict)
    state.pop(next(iter(state)))
    incomplete = tmp_path / "incomplete.safetensors"
    save_file(state, incomplete)
    with pytest.raises(RuntimeError, match="Missing key"):
        FrozenDINOv3Encoder(
            tiny_dinov3_artifact.encoder_config(
                weights_path=incomplete,
                expected_weights_sha256=sha256_file(incomplete),
            )
        )


def test_perceiver_projects_teacher_width_independently_from_model_width() -> None:
    config = PerceiverResamplerConfig(
        input_dim=32,
        width=64,
        num_latents=4,
        num_layers=1,
        num_heads=8,
        mlp_ratio=2,
        raw_patch_grid=2,
        raw_patch_hidden_dim=8,
        raw_shortcut_hidden_dim=16,
    )
    resampler = PerceiverResampler(config)
    images = torch.randint(0, 256, (1, 1, 1, 3, 32, 32), dtype=torch.uint8)
    teacher = torch.randn(1, 1, 1, 4, 32)
    valid = torch.ones(1, 1, 1, dtype=torch.bool)

    adapted = resampler.visual_adapter(images, teacher, valid)
    output = resampler(
        adapted.context,
        context_valid_mask=adapted.context_valid_mask,
    )
    output.square().mean().backward()

    assert adapted.context.shape == (1, 8, 64)
    assert output.shape == (1, 4, 64)
    projection = resampler.visual_adapter.teacher_projection
    assert isinstance(projection, nn.Sequential)
    linear = projection[-1]
    assert isinstance(linear, nn.Linear)
    assert linear.weight.grad is not None
    assert torch.isfinite(linear.weight.grad).all()
    with pytest.raises(ValueError, match="adapter input_dim"):
        resampler.visual_adapter(images, teacher[..., :31], valid)


def test_latent_wam_uses_dynamic_dinov3_future_latent_width(
    tiny_dinov3_artifact: _TinyDINOv3Artifact,
) -> None:
    encoder = tiny_dinov3_artifact.encoder(inference_batch_size=1)
    stats = _normalization()
    world = RWMARWorldModel(
        RWMARConfig(
            history_horizon=2,
            train_forecast_horizon=8,
            planning_horizon=8,
            encoder_hidden_dim=16,
            gru_hidden_dim=8,
            gru_layers=1,
        ),
        stats,
    )
    resampler = PerceiverResamplerConfig(
        input_dim=_TINY_DIM,
        width=512,
        num_latents=16,
        num_layers=3,
        num_heads=8,
        mlp_ratio=2,
        raw_patch_grid=2,
        raw_patch_hidden_dim=8,
        raw_shortcut_hidden_dim=16,
    )
    config = LatentWAMConfig(
        task_vocabulary=("visual_event_stop",),
        use_state=False,
        use_vision=True,
        capacity_control="future_head",
        task_embedding_dim=8,
        fusion_hidden_dim=64,
        future_hidden_dim=64,
        future_action_hidden_dim=16,
        future_latent_dim=_TINY_DIM,
        resampler=resampler,
    )
    model = LatentWAM(config, world, encoder)
    model.train()
    images = torch.randint(0, 256, (1, 3, 32, 32), dtype=torch.uint8)
    task_index = model.task_indices("visual_event_stop")
    candidate_actions = torch.zeros(1, 8, 8)

    output = model(
        None,
        None,
        None,
        images,
        task_index,
        candidate_actions,
    )
    loss = (
        output.encoding.planning_features.square().mean()
        + output.future_visual_latents.square().mean()
    )
    loss.backward()

    assert encoder.training is False
    assert encoder.backbone.training is False
    assert output.encoding.teacher_current_pooled_latent.shape == (1, _TINY_DIM)
    assert output.encoding.visual_tokens.shape == (1, 16, 512)
    assert output.future_visual_latents is not None
    assert output.future_visual_latents.shape == (1, 4, _TINY_DIM)
    assert torch.isfinite(output.future_visual_latents).all()
    assert all(parameter.grad is None for parameter in encoder.parameters())
    teacher_projection = model.resampler.visual_adapter.teacher_projection
    assert isinstance(teacher_projection, nn.Sequential)
    assert isinstance(teacher_projection[-1], nn.Linear)
    assert teacher_projection[-1].weight.grad is not None

    mismatched = LatentWAMConfig(
        task_vocabulary=("visual_event_stop",),
        use_state=False,
        capacity_control="none",
        future_latent_dim=64,
        resampler=PerceiverResamplerConfig(input_dim=64),
    )
    with pytest.raises(ValueError, match="output_dim must match"):
        LatentWAM(mismatched, world, encoder)


def test_dinov3_v2_checkpoint_is_self_contained_and_strictly_exact(
    tiny_dinov3_artifact: _TinyDINOv3Artifact,
    tmp_path: Path,
) -> None:
    (
        checkpoint,
        model,
        flow,
        legacy_world,
        legacy_flow,
    ) = _save_tiny_dinov3_checkpoint(tmp_path, tiny_dinov3_artifact)
    schema = json.loads((checkpoint / "schema.json").read_text(encoding="utf-8"))
    identity = schema["vision_encoder"]

    assert schema["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert schema["model_variant"] == "state_vision_future"
    assert schema["runtime_inputs"] == [
        "task",
        "past_executed_actions",
        "images.fixed",
        "proprioception",
    ]
    assert schema["frozen_visual_backbone"] is True
    assert schema["latent_wam_excluded_prefixes"] == ["vision_encoder."]
    assert identity == {
        "family": "dinov3",
        "encoder_name": _TINY_ENCODER,
        "weights_sha256": tiny_dinov3_artifact.weights_sha256,
        "config_sha256": tiny_dinov3_artifact.config_sha256,
        "model_id": _TINY_MODEL_ID,
        "revision": _TINY_REVISION,
        "preprocess_id": DINOV3_PREPROCESS_ID,
        "input_size": 32,
        "output_dim": _TINY_DIM,
        "patch_size": _TINY_PATCH,
        "register_tokens": _TINY_REGISTERS,
        "spatial_tokens": "last_hidden_state_excluding_cls_and_registers",
        "pooled_latent": "normalized_cls_token",
        "implementation": model.vision_encoder.implementation,
    }
    assert schema["vision_contract_sha256"] == _canonical_mapping_sha256(identity)
    assert schema["vision_source_sha256"] == tiny_dinov3_artifact.weights_sha256
    assert set(schema["artifact_sha256"]) >= {
        LATENT_WAM_WEIGHTS,
        DINOV3_SOURCE_CONFIG,
        DINOV3_SOURCE_WEIGHTS,
    }
    latent_state = load_file(checkpoint / LATENT_WAM_WEIGHTS, device="cpu")
    assert latent_state
    assert not any(name.startswith("vision_encoder.") for name in latent_state)
    assert (checkpoint / DINOV3_SOURCE_CONFIG).is_file()
    assert (checkpoint / DINOV3_SOURCE_WEIGHTS).is_file()

    fixed_inputs = _fixed_model_inputs()
    expected_output = _forward_snapshot(model, fixed_inputs)
    tiny_dinov3_artifact.config_path.unlink()
    tiny_dinov3_artifact.weights_path.unlink()

    loaded_model, loaded_flow, loaded_world, loaded_legacy_flow, metadata = (
        load_m1_checkpoint(
            checkpoint,
            expected_schema_version="wam.multimodal/1.1",
        )
    )
    assert metadata["schema"] == schema
    assert loaded_model.vision_encoder.config.config_path.parent == checkpoint
    assert loaded_model.vision_encoder.config.weights_path.parent == checkpoint
    _assert_module_state_exact(model, loaded_model)
    _assert_module_state_exact(flow, loaded_flow)
    _assert_module_state_exact(legacy_world, loaded_world)
    _assert_module_state_exact(legacy_flow, loaded_legacy_flow)
    actual_output = _forward_snapshot(loaded_model, fixed_inputs)
    assert set(actual_output) == set(expected_output)
    for name, expected in expected_output.items():
        assert torch.equal(actual_output[name], expected), name


def test_dinov3_v2_checkpoint_rejects_semantic_config_tampering(
    tiny_dinov3_artifact: _TinyDINOv3Artifact,
    tmp_path: Path,
) -> None:
    checkpoint, *_ = _save_tiny_dinov3_checkpoint(
        tmp_path,
        tiny_dinov3_artifact,
    )
    bundled_config = checkpoint / DINOV3_SOURCE_CONFIG
    raw_config = json.loads(bundled_config.read_text(encoding="utf-8"))
    raw_config["hidden_size"] = 48
    bundled_config.write_text(
        json.dumps(raw_config, sort_keys=True),
        encoding="utf-8",
    )
    tampered_config_sha256 = canonical_json_sha256(raw_config)

    payload_path = checkpoint / "config.yaml"
    payload = yaml.safe_load(payload_path.read_text(encoding="utf-8"))
    payload["vision_encoder_config"]["expected_config_sha256"] = tampered_config_sha256
    payload_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    schema_path = checkpoint / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["vision_encoder"]["config_sha256"] = tampered_config_sha256
    schema["vision_contract_sha256"] = _canonical_mapping_sha256(
        schema["vision_encoder"]
    )
    schema["artifact_sha256"][DINOV3_SOURCE_CONFIG] = sha256_file(bundled_config)
    schema["artifact_sha256"]["config.yaml"] = sha256_file(payload_path)
    schema_path.write_text(
        json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="architecture identity mismatch"):
        load_m1_checkpoint(checkpoint)


def test_dinov3_v2_checkpoint_rejects_schema_identity_tampering(
    tiny_dinov3_artifact: _TinyDINOv3Artifact,
    tmp_path: Path,
) -> None:
    checkpoint, *_ = _save_tiny_dinov3_checkpoint(
        tmp_path,
        tiny_dinov3_artifact,
    )
    schema_path = checkpoint / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["vision_encoder"]["model_id"] = "offline/tampered-dinov3"
    schema["vision_contract_sha256"] = _canonical_mapping_sha256(
        schema["vision_encoder"]
    )
    schema_path.write_text(
        json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="config and schema identities differ"):
        load_m1_checkpoint(checkpoint)


def _save_tiny_dinov3_checkpoint(
    root: Path,
    artifact: _TinyDINOv3Artifact,
) -> tuple[
    Path,
    LatentWAM,
    StatefulActionFlow,
    RWMARWorldModel,
    StatefulActionFlow,
]:
    stats = _normalization()
    world = _tiny_world(stats)
    encoder = artifact.encoder(inference_batch_size=2)
    model = LatentWAM(
        LatentWAMConfig(
            task_vocabulary=("visual_event_stop",),
            use_state=True,
            use_vision=True,
            capacity_control="future_head",
            task_embedding_dim=8,
            fusion_hidden_dim=64,
            future_hidden_dim=64,
            future_action_hidden_dim=16,
            future_latent_dim=_TINY_DIM,
            resampler=PerceiverResamplerConfig(
                input_dim=_TINY_DIM,
                width=512,
                num_latents=16,
                num_layers=3,
                num_heads=8,
                mlp_ratio=2,
                raw_patch_grid=2,
                raw_patch_hidden_dim=8,
                raw_shortcut_hidden_dim=16,
            ),
        ),
        world,
        encoder,
    ).eval()
    flow = _tiny_flow(world, stats).eval()
    legacy_world = _tiny_world(stats).eval()
    legacy_flow = _tiny_flow(legacy_world, stats).eval()
    checkpoint = root / "checkpoint"
    save_m1_checkpoint(
        checkpoint,
        model,
        flow,
        legacy_world,
        legacy_flow,
        stats,
        experiment_config={"test_fixture": "tiny_offline_dinov3"},
        dataset_manifest={"manifest_sha256": "a" * 64},
        metrics={"loss": 0.0},
        provenance={"source": "unit-test"},
        schema_version="wam.multimodal/1.1",
        train_seed=101,
        model_variant="state_vision_future",
    )
    return checkpoint, model, flow, legacy_world, legacy_flow


def _fixed_model_inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(404)
    return {
        "states": torch.randn(1, 2, 22, generator=generator),
        "past_actions": torch.randn(1, 1, 8, generator=generator),
        "valid_mask": torch.ones(1, 2, dtype=torch.bool),
        "images": torch.randint(
            0,
            256,
            (1, 2, 1, 3, 32, 32),
            generator=generator,
            dtype=torch.uint8,
        ),
        "candidate_actions": torch.randn(1, 8, 8, generator=generator),
    }


def _forward_snapshot(
    model: LatentWAM,
    values: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    model.eval()
    with torch.inference_mode():
        output = model(
            values["states"],
            values["past_actions"],
            values["valid_mask"],
            values["images"],
            model.task_indices("visual_event_stop"),
            values["candidate_actions"],
        )
    assert output.world_predictions is not None
    assert output.future_visual_latents is not None
    return {
        "planning_features": output.encoding.planning_features.detach().clone(),
        "visual_tokens": output.encoding.visual_tokens.detach().clone(),
        "teacher_pooled": (
            output.encoding.teacher_current_pooled_latent.detach().clone()
        ),
        "future_visual_latents": output.future_visual_latents.detach().clone(),
        "next_state_mean": (output.world_predictions.next_state_mean.detach().clone()),
    }


def _assert_module_state_exact(expected: nn.Module, actual: nn.Module) -> None:
    expected_state = expected.state_dict()
    actual_state = actual.state_dict()
    assert set(actual_state) == set(expected_state)
    for name, value in expected_state.items():
        assert torch.equal(actual_state[name].cpu(), value.detach().cpu()), name


def _tiny_world(stats: NormalizationStats) -> RWMARWorldModel:
    return RWMARWorldModel(
        RWMARConfig(
            history_horizon=2,
            train_forecast_horizon=8,
            planning_horizon=8,
            encoder_hidden_dim=16,
            gru_hidden_dim=8,
            gru_layers=1,
        ),
        stats,
    )


def _tiny_flow(
    world: RWMARWorldModel,
    stats: NormalizationStats,
) -> StatefulActionFlow:
    return StatefulActionFlow(
        StatefulActionFlowConfig(
            feature_dim=world.planning_feature_dim,
            hidden_dim=16,
            hidden_layers=1,
            time_embedding_dim=4,
            anchor_hidden_dim=8,
            anchor_hidden_layers=1,
        ),
        stats,
    )


def _canonical_mapping_sha256(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _official_unprefixed_state(
    wrapper_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Mirror the official artifact's unprefixed encoder-layer key space."""

    assert any(name.startswith("model.") for name in wrapper_state)
    state = {
        name.removeprefix("model."): value.detach().cpu().contiguous().clone()
        for name, value in wrapper_state.items()
    }
    assert not any(name.startswith("model.") for name in state)
    assert len(state) == len(wrapper_state)
    return state


def _normalization() -> NormalizationStats:
    return NormalizationStats(
        state_mean=np.zeros(22, dtype=np.float32),
        state_std=np.ones(22, dtype=np.float32),
        action_mean=np.zeros(8, dtype=np.float32),
        action_std=np.ones(8, dtype=np.float32),
        delta_mean=np.zeros(22, dtype=np.float32),
        delta_std=np.ones(22, dtype=np.float32),
        reward_mean=np.zeros(1, dtype=np.float32),
        reward_std=np.ones(1, dtype=np.float32),
    )
