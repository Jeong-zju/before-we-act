from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import yaml

from models.static_rgb_act import (
    DenseFeedForward,
    LatestChunkSelector,
    StaticRGBMoEACT,
    StaticRGBMoEACTConfig,
    TemporalChunkEnsembler,
    Top2SparseMoE,
    build_chunk_aggregator,
)
from scripts.train_static_rgb_act_moe import (
    _TaskBalancedBatchSampler,
    _frozen_vision_tokens,
    _local_batch,
)


ROOT = Path(__file__).resolve().parents[1]


def _config(**overrides) -> StaticRGBMoEACTConfig:
    values = {
        "horizon": 6,
        "vision_dim": 16,
        "d_model": 32,
        "encoder_layers": 1,
        "decoder_layers": 2,
        "heads": 4,
        "ffn_dim": 64,
        "latent_dim": 8,
        "experts": 4,
        "dropout": 0.0,
        "dense_ffn_dim": 128,
    }
    values.update(overrides)
    return StaticRGBMoEACTConfig(**values)


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


def test_dense_act_decoder_backpropagates_without_router_auxiliary_loss():
    config = _config(decoder_kind="dense")
    model = StaticRGBMoEACT(config)
    vision = torch.randn(2, 12, config.vision_dim)
    state = torch.randn(2, config.state_dim)
    actions = torch.randn(2, config.horizon, config.action_dim)

    prediction, _, _, router_aux = model(vision, state, actions)
    (prediction.square().mean() + 0.01 * (router_aux - 1.0)).backward()

    assert prediction.shape == actions.shape
    torch.testing.assert_close(router_aux, torch.ones_like(router_aux))
    first_ffn = model.decoder.layers[0].moe
    assert isinstance(first_ffn, DenseFeedForward)
    assert first_ffn.net[0].out_features == config.dense_ffn_dim
    assert first_ffn.net[0].weight.grad is not None


def test_legacy_model_config_defaults_to_sparse_moe():
    raw = _config().to_dict()
    raw.pop("decoder_kind")
    raw.pop("dense_ffn_dim")

    restored = StaticRGBMoEACTConfig.from_dict(raw)

    assert restored.decoder_kind == "sparse_moe"
    assert restored.dense_ffn_dim == 3072


@pytest.mark.parametrize("decoder_kind", ["", "moe", "unknown"])
def test_model_config_rejects_unknown_decoder(decoder_kind):
    with pytest.raises(ValueError, match="decoder_kind"):
        _config(decoder_kind=decoder_kind)


def test_frozen_vision_features_are_valid_inputs_for_act_backpropagation():
    config = _config()

    class Vision(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(3, config.vision_dim)

        def forward(self, images):
            pooled = images.float().mean(dim=(-1, -2))
            tokens = self.projection(pooled)[:, None].expand(-1, 12, -1)
            return SimpleNamespace(spatial_tokens=tokens)

    vision = Vision()
    model = StaticRGBMoEACT(config)
    images = torch.randn(2, 3, 8, 8)
    tokens = _frozen_vision_tokens(vision, images)

    assert not torch.is_inference(tokens)
    assert not tokens.requires_grad

    prediction = model(
        tokens,
        torch.randn(2, config.state_dim),
        torch.randn(2, config.horizon, config.action_dim),
    )[0]
    prediction.square().mean().backward()

    assert vision.projection.weight.grad is None
    assert model.vision_projection[1].weight.grad is not None


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


def test_latest_chunk_ablation_uses_action_zero_from_newest_replan():
    selector = LatestChunkSelector(horizon=3)
    selector.push(torch.tensor([[[1.0], [2.0], [3.0]]]))
    selector.advance()
    selector.push(torch.tensor([[[7.0], [8.0], [9.0]]]))

    torch.testing.assert_close(selector.current(), torch.tensor([[7.0]]))
    selector.reset()
    with pytest.raises(RuntimeError, match="no prediction"):
        selector.current()


def test_chunk_aggregator_factory_has_explicit_ablation_modes():
    temporal = build_chunk_aggregator(
        mode="temporal_ensemble", horizon=3, decay=0.5
    )
    latest = build_chunk_aggregator(mode="latest_chunk", horizon=3)

    assert isinstance(temporal, TemporalChunkEnsembler)
    assert isinstance(latest, LatestChunkSelector)
    with pytest.raises(ValueError, match="chunk aggregation"):
        build_chunk_aggregator(mode="implicit", horizon=3)


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


def test_ablation_configs_change_only_declared_factors():
    config_root = ROOT / "configs/static_act"
    full = _yaml(config_root / "lpd_static_dino_act_moe.yaml")
    dense = _yaml(
        config_root / "lpd_static_dino_act_dense_compute_matched.yaml"
    )
    latest = _yaml(config_root / "lpd_static_dino_act_moe_latest_chunk.yaml")

    assert dense["data"] == full["data"]
    assert dense["vision"] == full["vision"]
    assert dense["model"] == {
        **full["model"],
        "decoder_kind": "dense",
    }
    assert dense["training"] == {
        **full["training"],
        "router_aux_weight": 0.0,
    }
    assert dense["inference"] == full["inference"]

    assert latest["data"] == full["data"]
    assert latest["vision"] == full["vision"]
    assert latest["model"] == full["model"]
    assert latest["training"] == full["training"]
    assert latest["checkpoint"] == full["checkpoint"]
    assert latest["inference"] == {
        **full["inference"],
        "chunk_aggregation": "latest_chunk",
    }


def _yaml(path: Path):
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
