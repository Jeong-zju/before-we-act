from __future__ import annotations

import torch

from models.static_rgb_act import (
    StaticRGBMoEACT,
    StaticRGBMoEACTConfig,
    TemporalChunkEnsembler,
    Top2SparseMoE,
)
from scripts.train_static_rgb_act_moe import (
    _TaskBalancedBatchSampler,
    _local_batch,
)


def _config() -> StaticRGBMoEACTConfig:
    return StaticRGBMoEACTConfig(
        horizon=6,
        vision_dim=16,
        d_model=32,
        encoder_layers=1,
        decoder_layers=2,
        heads=4,
        ffn_dim=64,
        latent_dim=8,
        experts=4,
        dropout=0.0,
    )


def test_top2_moe_routes_and_backpropagates_router_loss():
    torch.manual_seed(3)
    moe = Top2SparseMoE(_config())
    value = torch.randn(2, 5, 32, requires_grad=True)

    output, balance = moe(value)
    (output.square().mean() + 0.01 * balance).backward()

    assert output.shape == value.shape
    assert torch.isfinite(output).all()
    assert float(balance.detach()) > 0.0
    assert moe.router.weight.grad is not None


def test_static_rgb_act_output_and_zero_latent_inference_contract():
    config = _config()
    model = StaticRGBMoEACT(config).eval()
    vision = torch.randn(2, 12, config.vision_dim)
    state = torch.randn(2, config.state_dim)

    first = model(vision, state)[0]
    second = model(vision, state)[0]

    assert first.shape == (2, config.horizon, config.action_dim)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_temporal_ensemble_prefers_newer_chunks():
    ensemble = TemporalChunkEnsembler(horizon=3, decay=0.5)
    ensemble.push(torch.tensor([[[0.0], [1.0], [1.0]]]))
    ensemble.advance()
    ensemble.push(torch.tensor([[[3.0], [3.0], [3.0]]]))

    current = ensemble.current()

    expected = (
        torch.exp(torch.tensor(-0.5)) * 1.0 + 3.0
    ) / (torch.exp(torch.tensor(-0.5)) + 1.0)
    torch.testing.assert_close(current, expected.reshape(1, 1))


def test_existing_static_camera_batch_expands_only_present_agents():
    batch = {
        "states": torch.randn(2, 1, 72),
        "action_targets": torch.randn(2, 6, 32),
        "action_target_valid_mask": torch.ones(2, 6, dtype=torch.bool),
        "embodiment_index": torch.tensor([1, 3]),
        "images": torch.zeros(2, 1, 5, 3, 8, 8, dtype=torch.uint8),
        "image_valid_mask": torch.ones(2, 1, 5, dtype=torch.bool),
    }

    local = _local_batch(batch)

    assert local["images"].shape == (6, 3, 8, 8)
    assert local["state"].shape == (6, 18)
    assert local["actions"].shape == (6, 6, 8)
    assert local["valid"].shape == (6, 6)


def test_task_balanced_sampler_resume_is_exact_suffix():
    class Dataset:
        contracts = (object(), object())

        @staticmethod
        def task_indices(task_index):
            return (range(0, 10), range(10, 30))[task_index]

    full = list(
        _TaskBalancedBatchSampler(
            Dataset(),
            batch_size=4,
            first_update=1,
            final_update=5,
            seed=17,
        )
    )
    resumed = list(
        _TaskBalancedBatchSampler(
            Dataset(),
            batch_size=4,
            first_update=3,
            final_update=5,
            seed=17,
        )
    )

    assert resumed == full[2:]
