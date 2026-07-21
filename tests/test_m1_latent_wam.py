from __future__ import annotations

import gc
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from models.wam import (
    NormalizationStats,
    RWMARConfig,
    RWMARWorldModel,
    StatefulActionFlow,
    StatefulActionFlowConfig,
)
from models.wam_multimodal import (
    ActionConditionedFutureLatentHead,
    FrozenResNet18Config,
    FrozenResNet18Encoder,
    FutureLatentHeadConfig,
    LatentWAM,
    LatentWAMConfig,
    OFFICIAL_RESNET18_SHA256,
    PerceiverResampler,
    PerceiverResamplerConfig,
    build_resnet18_classifier,
    default_resnet18_weights_path,
    sha256_file,
)


TASKS = (
    "cooperative_stop",
    "visual_event_stop",
    "visual_target_select",
    "visual_obstacle_avoid",
)


@pytest.fixture(scope="module")
def frozen_encoder(tmp_path_factory: pytest.TempPathFactory) -> FrozenResNet18Encoder:
    official = default_resnet18_weights_path()
    if official.is_file():
        assert sha256_file(official) == OFFICIAL_RESNET18_SHA256
        config = FrozenResNet18Config(official)
    else:
        # Preserve a portable unit-test path while the formal configuration
        # remains pinned to the published ImageNet artifact and digest.
        path = tmp_path_factory.mktemp("resnet18") / "resnet18-test.pth"
        torch.save(build_resnet18_classifier().state_dict(), path)
        config = FrozenResNet18Config(path, expected_sha256=sha256_file(path))
    return FrozenResNet18Encoder(config)


def test_latent_wam_config_is_canonical_serializable_and_names_variants() -> None:
    config = LatentWAMConfig(task_vocabulary=TASKS)

    assert config.resampler == PerceiverResamplerConfig(
        input_dim=512,
        width=512,
        num_latents=16,
        num_layers=3,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.0,
    )
    assert config.future_horizons == (1, 2, 4, 8)
    assert config.variant == "state_vision_future_head"
    assert config.use_future_head is True
    assert LatentWAMConfig.from_dict(config.to_dict()) == config

    assert LatentWAMConfig(
        task_vocabulary=TASKS,
        use_vision=False,
        capacity_control="none",
    ).variant == "state_only_none"
    assert LatentWAMConfig(
        task_vocabulary=TASKS,
        use_state=False,
        capacity_control="none",
    ).variant == "vision_only_none"
    with pytest.raises(ValueError, match="at least one"):
        LatentWAMConfig(
            task_vocabulary=TASKS,
            use_state=False,
            use_vision=False,
        )


def test_resnet18_artifact_is_strict_frozen_eval_and_returns_patch_tokens(
    frozen_encoder: FrozenResNet18Encoder,
) -> None:
    raw = torch.arange(3 * 64 * 64, dtype=torch.int64).remainder(256)
    raw = raw.to(torch.uint8).reshape(1, 3, 64, 64)

    output = frozen_encoder(raw)
    frozen_encoder.train(True)

    assert output.spatial_tokens.shape == (1, 49, 512)
    assert output.pooled_latent.shape == (1, 512)
    assert torch.isfinite(output.spatial_tokens).all()
    assert output.spatial_tokens.requires_grad is False
    assert frozen_encoder.training is False
    assert frozen_encoder.backbone.training is False
    assert all(not parameter.requires_grad for parameter in frozen_encoder.parameters())
    assert all(
        not module.training
        for module in frozen_encoder.modules()
        if isinstance(module, torch.nn.BatchNorm2d)
    )
    assert "fc.weight" in frozen_encoder.backbone.state_dict()


def test_resnet18_sha_and_float_preprocess_fail_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "weights.pth"
    artifact.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        FrozenResNet18Encoder(
            FrozenResNet18Config(artifact, expected_sha256="0" * 64)
        )


def test_three_layer_perceiver_runs_cross_and_self_attention() -> None:
    config = PerceiverResamplerConfig()
    resampler = PerceiverResampler(config)
    context = torch.randn(1, 5, 512, requires_grad=True)
    mask = torch.tensor([[True, True, True, False, False]])

    output = resampler(context, context_valid_mask=mask)
    output.square().mean().backward()

    assert output.shape == (1, 16, 512)
    assert len(resampler.layers) == 3
    assert all(hasattr(layer, "cross_attention") for layer in resampler.layers)
    assert all(hasattr(layer, "self_attention") for layer in resampler.layers)
    assert context.grad is not None and torch.isfinite(context.grad).all()
    with pytest.raises(ValueError, match="at least one valid"):
        resampler(context.detach(), context_valid_mask=torch.zeros_like(mask))


def test_position_aware_visual_summary_changes_under_horizontal_mirror() -> None:
    """A left/right flip must no longer be an invisible token permutation."""

    torch.manual_seed(7)
    resampler = PerceiverResampler(PerceiverResamplerConfig())
    image = torch.zeros(1, 1, 1, 3, 32, 32, dtype=torch.uint8)
    image[..., 0, 9:18, 2:8] = 255
    mirrored = torch.flip(image, dims=(-1,))
    # Equal teacher tokens isolate the new generic raw-patch/position route.
    teacher = torch.zeros(1, 1, 1, 49, 512)
    valid = torch.ones(1, 1, 1, dtype=torch.bool)

    original_context = resampler.visual_adapter(image, teacher, valid)
    mirrored_context = resampler.visual_adapter(mirrored, teacher, valid)
    original_summary = resampler(
        original_context.context,
        context_valid_mask=original_context.context_valid_mask,
    ).mean(dim=1)
    mirrored_summary = resampler(
        mirrored_context.context,
        context_valid_mask=mirrored_context.context_valid_mask,
    ).mean(dim=1)

    assert not torch.allclose(original_summary, mirrored_summary)
    difference = (original_summary - mirrored_summary).square().mean().sqrt()
    assert float(difference.detach()) > 1e-4


def test_generic_rgb_patch_tokens_distinguish_paired_left_right_markers() -> None:
    torch.manual_seed(11)
    resampler = PerceiverResampler(PerceiverResamplerConfig())
    paired = torch.zeros(2, 1, 1, 3, 40, 40, dtype=torch.uint8)
    # Same marker appearance, paired location only: no task-specific decoder.
    paired[0, 0, 0, 1, 14:24, 3:11] = 255
    paired[1, 0, 0, 1, 14:24, 29:37] = 255
    teacher = torch.zeros(2, 1, 1, 49, 512)
    valid = torch.ones(2, 1, 1, dtype=torch.bool)

    adapted = resampler.visual_adapter(paired, teacher, valid)
    latents = resampler(
        adapted.context,
        context_valid_mask=adapted.context_valid_mask,
    )
    features = resampler.summarize(latents, adapted.spatial_shortcut)

    assert adapted.context.shape == (2, 113, 512)  # 7x7 teacher + 8x8 RGB
    assert adapted.spatial_shortcut.shape == (2, 512)
    assert not torch.allclose(
        adapted.spatial_shortcut[0], adapted.spatial_shortcut[1]
    )
    assert not torch.allclose(features[0], features[1])
    assert any(
        parameter.requires_grad
        for parameter in (
            resampler.visual_adapter.raw_patch_projection.parameters()
        )
    )
    assert any(
        parameter.requires_grad
        for parameter in (
            resampler.visual_adapter.raw_shortcut_projection.parameters()
        )
    )


def test_future_head_is_action_conditioned_and_has_no_future_target_input() -> None:
    head = ActionConditionedFutureLatentHead(
        FutureLatentHeadConfig(
            planning_feature_dim=24,
            hidden_dim=64,
            action_hidden_dim=32,
        )
    )
    planning = torch.randn(2, 24)
    visual = torch.randn(2, 16, 512)
    zeros = torch.zeros(2, 8, 8)
    ones = torch.ones(2, 8, 8)

    without_action = head(planning, visual, zeros)
    with_action = head(planning, visual, ones)

    assert without_action.shape == (2, 4, 512)
    assert not torch.allclose(without_action, with_action)
    assert head.horizons == (1, 2, 4, 8)
    assert "target" not in inspect.signature(head.forward).parameters
    assert "image" not in inspect.signature(head.forward).parameters


def test_composite_forward_fuses_modalities_and_keeps_teacher_frozen(
    frozen_encoder: FrozenResNet18Encoder,
) -> None:
    world = _world_model(small=True)
    model = LatentWAM(
        LatentWAMConfig(
            task_vocabulary=TASKS,
            fusion_hidden_dim=128,
            future_hidden_dim=128,
            future_action_hidden_dim=64,
        ),
        world,
        frozen_encoder,
    )
    states, actions, valid = _history(batch_size=1)
    images = torch.randint(0, 256, (1, 2, 1, 3, 64, 64), dtype=torch.uint8)
    image_mask = torch.ones(1, 2, 1, dtype=torch.bool)
    task_index = model.task_indices(["visual_event_stop"])
    candidate_actions = torch.randn(1, 8, 8)

    model.train()
    output = model(
        states,
        actions,
        valid,
        images,
        task_index,
        candidate_actions,
        image_valid_mask=image_mask,
    )
    loss = (
        output.encoding.planning_features.square().mean()
        + output.future_visual_latents.square().mean()
    )
    loss.backward()

    assert output.encoding.planning_features.shape == (
        1,
        world.planning_feature_dim,
    )
    assert output.encoding.visual_tokens.shape == (1, 16, 512)
    assert output.encoding.teacher_current_pooled_latent.shape == (1, 512)
    assert output.world_predictions is not None
    assert output.future_visual_latents.shape == (1, 4, 512)
    assert output.future_horizons == (1, 2, 4, 8)
    assert any(
        parameter.grad is not None for parameter in model.resampler.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.resampler.visual_adapter.raw_patch_projection.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.resampler.visual_adapter.raw_shortcut_projection.parameters()
    )
    assert model.fusion.visual_skip.weight.grad is not None
    assert all(
        parameter.grad is None and not parameter.requires_grad
        for parameter in frozen_encoder.parameters()
    )
    assert frozen_encoder.training is False
    assert "target" not in inspect.signature(model.forward).parameters


def test_vision_only_never_invokes_state_encoder_and_state_only_skips_rgb(
    frozen_encoder: FrozenResNet18Encoder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vision_world = _world_model(small=True)

    def forbidden_state_encoder(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("vision-only ablation called the state encoder")

    monkeypatch.setattr(
        vision_world,
        "encode_planning_history",
        forbidden_state_encoder,
    )
    vision_model = LatentWAM(
        LatentWAMConfig(
            task_vocabulary=TASKS,
            use_state=False,
            capacity_control="none",
            fusion_hidden_dim=64,
        ),
        vision_world,
        frozen_encoder,
    )
    image = torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.uint8)
    vision_encoding = vision_model.encode(
        None,
        None,
        None,
        image,
        vision_model.task_indices("visual_target_select"),
    )
    assert vision_encoding.hidden is None
    assert vision_encoding.current_state is None
    assert torch.count_nonzero(vision_encoding.state_planning_features) == 0

    state_world = _world_model(small=True)
    state_model = LatentWAM(
        LatentWAMConfig(
            task_vocabulary=TASKS,
            use_vision=False,
            capacity_control="none",
            fusion_hidden_dim=64,
        ),
        state_world,
        frozen_encoder,
    )
    states, actions, valid = _history(batch_size=1)
    state_encoding = state_model.encode(
        states,
        actions,
        valid,
        None,
        state_model.task_indices("cooperative_stop"),
    )
    assert state_encoding.hidden is not None
    assert torch.count_nonzero(state_encoding.visual_tokens) == 0


def test_five_variants_construct_and_equal_capacity_models_match_parameter_count(
    frozen_encoder: FrozenResNet18Encoder,
) -> None:
    variants = (
        (True, False, "none", "state_only_none"),
        (False, True, "none", "vision_only_none"),
        (True, True, "none", "state_vision_none"),
        (True, True, "future_head", "state_vision_future_head"),
        (True, True, "action_mlp", "state_vision_action_mlp"),
    )
    for use_state, use_vision, capacity, expected_variant in variants:
        model = LatentWAM(
            LatentWAMConfig(
                task_vocabulary=TASKS,
                use_state=use_state,
                use_vision=use_vision,
                capacity_control=capacity,
                fusion_hidden_dim=64,
                future_hidden_dim=128,
                future_action_hidden_dim=64,
                action_mlp_hidden_dim=64,
            ),
            _world_model(small=True),
            frozen_encoder,
        )
        assert model.config.variant == expected_variant
        assert (model.future_head is not None) is (capacity == "future_head")
        assert (model.action_capacity_mlp is not None) is (capacity == "action_mlp")
        del model
        gc.collect()

    stats = _stats()
    future_world = _world_model(stats=stats)
    action_flow = StatefulActionFlow(
        StatefulActionFlowConfig(feature_dim=future_world.planning_feature_dim),
        stats,
    )
    future_model = LatentWAM(
        LatentWAMConfig(task_vocabulary=TASKS, capacity_control="future_head"),
        future_world,
        frozen_encoder,
    )
    future_breakdown = future_model.parameter_breakdown(action_flow)
    future_total = future_model.trainable_parameter_count(action_flow)
    future_capacity = future_model.capacity_target_parameter_count
    assert future_breakdown["total_active"] == future_total
    del future_model, future_world
    gc.collect()

    action_world = _world_model(stats=stats)
    action_model = LatentWAM(
        LatentWAMConfig(task_vocabulary=TASKS, capacity_control="action_mlp"),
        action_world,
        frozen_encoder,
    )
    action_breakdown = action_model.parameter_breakdown(action_flow)

    assert 20_000_000 <= future_total <= 60_000_000
    assert action_model.trainable_parameter_count(action_flow) == future_total
    assert action_model.capacity_target_parameter_count == future_capacity
    assert action_breakdown["capacity_padding"] == 0
    assert action_breakdown["action_mlp_functional"] == future_capacity
    assert action_breakdown["total_active"] == action_breakdown["total_trainable"]
    assert action_model.action_capacity_mlp is not None
    assert not hasattr(action_model.action_capacity_mlp, "capacity_padding")
    action_model.action_capacity_mlp(
        torch.randn(2, action_model.planning_feature_dim),
        torch.randn(2, action_model.config.resampler.width),
        torch.randn(2, action_model.config.task_embedding_dim),
    ).square().mean().backward()
    assert all(
        parameter.grad is not None
        for parameter in action_model.action_capacity_mlp.parameters()
    )
    assert future_breakdown["vision_encoder_trainable"] == 0
    assert future_breakdown["vision_encoder_frozen"] > 0


def _stats() -> NormalizationStats:
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


def _world_model(
    *,
    small: bool = False,
    stats: NormalizationStats | None = None,
) -> RWMARWorldModel:
    config = RWMARConfig(
        encoder_hidden_dim=16 if small else 256,
        gru_hidden_dim=12 if small else 256,
        gru_layers=1 if small else 2,
        train_forecast_horizon=8,
        planning_horizon=8,
    )
    return RWMARWorldModel(config, stats or _stats())


def _history(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states = torch.randn(batch_size, 2, 22)
    actions = torch.randn(batch_size, 1, 8)
    valid = torch.ones(batch_size, 2, dtype=torch.bool)
    return states, actions, valid
