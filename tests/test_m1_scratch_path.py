from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from models.wam import (
    ActionChunkConfig,
    AffineActionCodec,
    AffineActionCodecConfig,
    NormalizationStats,
    RWMARConfig,
)
from models.wam_multimodal import (
    LatentWAMConfig,
    PerceiverResamplerConfig,
    VisionEncoderOutput,
)
from policies.scratch_m1 import ScratchM1Policy, ScratchM1PolicyConfig
from scripts.train_liftbarrier_m1_scratch import _resolve_input_pipeline
from train.m1_scratch_builder import (
    ScratchActionFlowConfig,
    ScratchM1BuildConfig,
    build_scratch_m1,
)
from train.m1_scratch_checkpointing import (
    load_scratch_m1_checkpoint,
    save_scratch_m1_checkpoint,
)
from train.m1_scratch_training import (
    ScratchM1StageConfig,
    build_scratch_optimizer,
    scratch_stage_required_keys,
    train_scratch_m1_stage,
    validate_scratch_stage_order,
)
from train.m1_training import M1FlowObjectiveConfig, M1LossWeights


class _TinyFrozenVision(nn.Module):
    output_dim = 512
    artifact_sha256 = "a" * 64
    config_sha256 = "b" * 64

    def __init__(self) -> None:
        super().__init__()
        projection = torch.linspace(-0.5, 0.5, 3 * self.output_dim).reshape(
            3, self.output_dim
        )
        self.register_buffer("projection", projection)

    def forward(self, images: torch.Tensor) -> VisionEncoderOutput:
        leading = images.shape[:-3]
        flattened = images.reshape(-1, *images.shape[-3:]).float() / 255.0
        patches = F.adaptive_avg_pool2d(flattened, (2, 2))
        patches = patches.flatten(2).transpose(1, 2) @ self.projection
        pooled = flattened.mean(dim=(-2, -1)) @ self.projection
        return VisionEncoderOutput(
            spatial_tokens=patches.reshape(*leading, 4, self.output_dim).detach(),
            pooled_latent=pooled.reshape(*leading, self.output_dim).detach(),
        )


def _stats(state_dim: int = 4, action_dim: int = 4) -> NormalizationStats:
    return NormalizationStats(
        state_mean=np.zeros(state_dim, dtype=np.float32),
        state_std=np.ones(state_dim, dtype=np.float32),
        action_mean=np.zeros(action_dim, dtype=np.float32),
        action_std=np.ones(action_dim, dtype=np.float32),
        delta_mean=np.zeros(state_dim, dtype=np.float32),
        delta_std=np.ones(state_dim, dtype=np.float32),
        reward_mean=np.zeros(1, dtype=np.float32),
        reward_std=np.ones(1, dtype=np.float32),
    )


def _codec(action_dim: int = 4) -> AffineActionCodecConfig:
    return AffineActionCodecConfig(
        codec_id="test.raw-action/1",
        low=tuple(-2.0 - index for index in range(action_dim)),
        high=tuple(2.0 + index for index in range(action_dim)),
        raw_domain="test_raw_controller",
    )


def _build_config(seed: int = 17) -> ScratchM1BuildConfig:
    return ScratchM1BuildConfig(
        seed=seed,
        world=RWMARConfig(
            state_dim=4,
            action_dim=4,
            history_horizon=4,
            train_forecast_horizon=8,
            planning_horizon=8,
            encoder_hidden_dim=8,
            gru_hidden_dim=8,
            gru_layers=1,
            yaw_indices=(),
            gripper_closed_indices=(),
        ),
        action_flow=ScratchActionFlowConfig(
            action_dim=4,
            horizon=8,
            hidden_dim=16,
            hidden_layers=1,
            time_embedding_dim=8,
            anchor_hidden_dim=8,
            anchor_hidden_layers=1,
            anchor_mode="none",
        ),
        latent_wam=LatentWAMConfig(
            task_vocabulary=("lift_barrier",),
            action_dim=4,
            task_embedding_dim=8,
            fusion_hidden_dim=16,
            future_hidden_dim=16,
            future_action_hidden_dim=8,
            future_latent_dim=512,
            capacity_control="future_head",
            resampler=PerceiverResamplerConfig(
                input_dim=512,
                width=512,
                num_latents=16,
                num_layers=3,
                num_heads=8,
                mlp_ratio=1,
                raw_patch_grid=2,
                raw_patch_hidden_dim=8,
                raw_shortcut_hidden_dim=16,
                max_visual_history=2,
                max_visual_cameras=1,
            ),
        ),
    )


def _batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    return {
        "states": torch.randn(1, 4, 4, generator=generator),
        "state_valid_mask": torch.ones(1, 4, dtype=torch.bool),
        "past_actions": torch.rand(1, 3, 4, generator=generator) - 0.5,
        "images": torch.randint(
            0, 256, (1, 2, 1, 3, 8, 8), generator=generator, dtype=torch.uint8
        ),
        "task_index": torch.zeros(1, dtype=torch.long),
        "action_targets": torch.rand(1, 8, 4, generator=generator) - 0.5,
        "future_states": torch.randn(1, 8, 4, generator=generator),
        "future_images": torch.randint(
            0, 256, (1, 4, 1, 3, 8, 8), generator=generator, dtype=torch.uint8
        ),
        "future_image_novelty_mask": torch.ones(1, 4, 1, dtype=torch.bool),
        "future_horizons": torch.tensor([[1, 2, 4, 8]], dtype=torch.long),
    }


def _stages() -> tuple[ScratchM1StageConfig, ...]:
    no_future = M1LossWeights(
        flow_matching=0.5,
        action_endpoint=1.0,
        action_smoothness=0.05,
        future_visual_latent=0.0,
        future_state=0.0,
    )
    return (
        ScratchM1StageConfig(
            name="dynamics",
            objective="dynamics_warmup",
            steps=1,
            world_learning_rate=1e-3,
            losses=M1LossWeights(
                flow_matching=0.0,
                action_endpoint=0.0,
                action_smoothness=0.0,
                future_visual_latent=0.0,
                future_state=1.0,
            ),
        ),
        ScratchM1StageConfig(
            name="flow",
            objective="action_flow_warmup",
            steps=1,
            action_flow_learning_rate=1e-3,
            losses=no_future,
        ),
        ScratchM1StageConfig(
            name="fusion",
            objective="multimodal_fusion",
            steps=1,
            action_flow_learning_rate=1e-3,
            multimodal_learning_rate=1e-3,
            losses=no_future,
        ),
        ScratchM1StageConfig(
            name="future",
            objective="future_joint",
            steps=1,
            world_learning_rate=1e-4,
            action_flow_learning_rate=1e-3,
            multimodal_learning_rate=1e-3,
            future_learning_rate=1e-3,
            losses=M1LossWeights(
                flow_matching=0.5,
                action_endpoint=1.0,
                action_smoothness=0.05,
                future_visual_latent=0.1,
                future_state=0.1,
            ),
        ),
    )


def test_affine_action_codec_round_trip_and_rejects_out_of_bounds() -> None:
    codec = AffineActionCodec(_codec())
    raw = np.asarray([[-2.0, 0.0, 0.0, 5.0], [2.0, -3.0, 4.0, -5.0]], dtype=np.float32)
    canonical = codec.encode(raw)
    np.testing.assert_allclose(canonical[0], [-1.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(codec.decode(canonical, clip=False), raw)
    with pytest.raises(ValueError, match="outside"):
        codec.encode(np.asarray([[2.1, 0.0, 0.0, 0.0]], dtype=np.float32))

    liftbarrier = AffineActionCodec(
        AffineActionCodecConfig.load(
            Path("configs/action_codecs/liftbarrier_pd_joint_pos_16d.json")
        )
    )
    assert liftbarrier.action_dim == 16
    center = 0.5 * (
        np.asarray(liftbarrier.config.low, dtype=np.float32)
        + np.asarray(liftbarrier.config.high, dtype=np.float32)
    )
    np.testing.assert_allclose(liftbarrier.encode(center), np.zeros(16))
    np.testing.assert_allclose(
        liftbarrier.decode(np.ones(16, dtype=np.float32)),
        np.asarray(liftbarrier.config.high, dtype=np.float32),
        atol=2e-7,
    )


def test_scratch_input_pipeline_resolves_high_throughput_controls() -> None:
    pipeline = _resolve_input_pipeline(
        {
            "batch_size": 32,
            "num_workers": "auto",
            "prefetch_factor": 4,
            "persistent_workers": True,
            "pin_memory": True,
            "multiprocessing_context": "spawn",
            "in_order": True,
            "hdf5_cache_size": 32,
            "preload_to_ram": True,
            "preload_shared_memory": True,
            "preload_max_available_fraction": 0.5,
            "precision": "fp32",
            "torch_float32_matmul_precision": "high",
            "allow_tf32": True,
            "cudnn_benchmark": True,
        },
        device=torch.device("cuda"),
    )
    assert 2 <= pipeline.num_workers <= 12
    assert pipeline.prefetch_factor == 4
    assert pipeline.persistent_workers is True
    assert pipeline.pin_memory is True
    assert pipeline.preload_to_ram is True
    assert pipeline.preload_shared_memory is True
    assert pipeline.multiprocessing_context == "spawn"
    assert pipeline.allow_tf32 is True


def test_scratch_build_stages_checkpoint_and_policy(tmp_path: Path) -> None:
    build = _build_config()
    bundle = build_scratch_m1(
        build,
        _stats(),
        _codec(),
        vision_encoder=_TinyFrozenVision(),
    )
    repeated = build_scratch_m1(
        build,
        _stats(),
        _codec(),
        vision_encoder=_TinyFrozenVision(),
    )
    different = build_scratch_m1(
        _build_config(seed=18),
        _stats(),
        _codec(),
        vision_encoder=_TinyFrozenVision(),
    )
    assert bundle.initialization_hashes == repeated.initialization_hashes
    assert bundle.initialization_hashes != different.initialization_hashes
    assert not bundle.action_flow.has_anchor
    assert bundle.action_flow.config.anchor_mode == "none"
    assert all(
        not parameter.requires_grad
        for parameter in bundle.model.vision_encoder.parameters()
    )

    stages = _stages()
    validate_scratch_stage_order(stages)
    optimizer = build_scratch_optimizer(bundle.model, bundle.action_flow)
    objective = M1FlowObjectiveConfig(
        execution_steps=2,
        solver_steps=1,
        warm_start_probability=0.0,
        warm_start_noise_std=0.0,
        policy_fixed_action_dims=(),
    )
    reports = []
    progress_updates: list[tuple[int, float]] = []
    for index, stage in enumerate(stages):
        reports.append(
            train_scratch_m1_stage(
                bundle.model,
                bundle.action_flow,
                [_batch()],
                optimizer,
                stage,
                device=torch.device("cpu"),
                flow_objective=objective,
                seed=40 + index,
                progress=lambda completed, metrics: progress_updates.append(
                    (completed, metrics["total"])
                ),
            )
        )
    assert [report["objective"] for report in reports] == [
        stage.objective for stage in stages
    ]
    assert [completed for completed, _ in progress_updates] == [1, 1, 1, 1]
    assert all(np.isfinite(loss) for _, loss in progress_updates)

    required = [scratch_stage_required_keys(bundle.model, stage) for stage in stages]
    assert required[0] == {
        "states",
        "state_valid_mask",
        "past_actions",
        "action_targets",
        "future_states",
    }
    assert "images" not in required[0]
    assert "images" not in required[1]
    assert "images" in required[2]
    assert "future_images" not in required[2]
    assert {"future_images", "future_image_novelty_mask", "future_states"} <= required[3]

    checkpoint = tmp_path / "scratch"
    save_scratch_m1_checkpoint(
        checkpoint,
        bundle,
        dataset_lineage={"manifest_sha256": "c" * 64},
        stage_state={"completed_stage_count": 4},
    )
    loaded, metadata = load_scratch_m1_checkpoint(
        checkpoint,
        vision_encoder=_TinyFrozenVision(),
    )
    assert metadata["schema"]["legacy_weight_files"] == []
    assert metadata["schema"]["action_anchor_mode"] == "none"
    for name, value in bundle.action_flow.state_dict().items():
        torch.testing.assert_close(value, loaded.action_flow.state_dict()[name])

    policy = ScratchM1Policy.from_bundle(
        loaded,
        ScratchM1PolicyConfig(
            action_chunk=ActionChunkConfig(
                action_dim=4,
                horizon=8,
                execution_steps=2,
                solver_steps=1,
            ),
            camera_order=("global",),
            visual_history_frames=2,
        ),
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    raw_action = policy.act(
        {
            "task": {"id": "lift_barrier", "text": "lift the barrier"},
            "proprioception": np.zeros(4, dtype=np.float32),
            "images": {"global": image},
            "image_frame_indices": {"global": 0},
            "past_executed_actions": np.zeros((0, 4), dtype=np.float32),
        }
    )
    assert raw_action.shape == (4,)
    assert np.all(raw_action >= np.asarray(_codec().low, dtype=np.float32))
    assert np.all(raw_action <= np.asarray(_codec().high, dtype=np.float32))
    assert policy.last_diagnostics["model_action_domain"] == "canonical_unit_action"
    assert policy.last_diagnostics["legacy_bypass_used"] is False
