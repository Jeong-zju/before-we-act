from __future__ import annotations

import copy

import pytest
import torch

from models.wam_multimodal.horizon_causal_future_predictor import (
    HorizonCausalActionAggregator,
    HorizonCausalActiveTeamFutureProvider,
)
from models.wam_multimodal.local_future_predictor import (
    LocalActionConditionedFuturePredictor,
    LocalFuturePredictorConfig,
)
from models.wam_multimodal.protected_team_future_predictor import (
    ProtectedTeamFuturePredictor,
    ProtectedTeamFuturePredictorConfig,
)
from scripts.evaluate_s4_r8_prefix_causality import _exact_interventions


def _local_config() -> LocalFuturePredictorConfig:
    return LocalFuturePredictorConfig(
        max_agents=4,
        state_dim=5,
        action_dim=2,
        action_horizon=8,
        future_horizons=(1, 3, 5, 8),
        visual_grid_tokens=2,
        visual_latent_dim=6,
        d_model=12,
        ffn_dim=24,
        layers=1,
        heads=3,
        dropout=0.0,
    )


def _source(config: LocalFuturePredictorConfig) -> ProtectedTeamFuturePredictor:
    own = LocalActionConditionedFuturePredictor(config)
    source = ProtectedTeamFuturePredictor(
        config,
        ProtectedTeamFuturePredictorConfig(
            layers=1,
            heads=3,
            ffn_dim=24,
            dropout=0.0,
            team_mixer="shared",
        ),
    )
    source.load_protected_own(own.state_dict())
    return source


@pytest.mark.parametrize("kind", ["prefix_mean", "causal_prefix_attention"])
def test_action_aggregator_is_strictly_suffix_invariant(kind: str) -> None:
    generator = torch.Generator().manual_seed(801)
    tokens = torch.randn(2, 4, 8, 12, generator=generator)
    aggregator = HorizonCausalActionAggregator(
        d_model=12,
        action_horizon=8,
        future_horizons=(1, 3, 5, 8),
        kind=kind,
        rank=4,
    )
    baseline = aggregator(tokens)
    for horizon_index, horizon in enumerate((1, 3, 5, 8)):
        suffix_changed = tokens.clone()
        suffix_changed[..., horizon:, :] += 1000.0
        observed = aggregator(suffix_changed)[..., horizon_index, :]
        assert torch.equal(observed, baseline[..., horizon_index, :])
        prefix_changed = tokens.clone()
        prefix_changed[..., horizon - 1, :] += 1.0
        observed = aggregator(prefix_changed)[..., horizon_index, :]
        assert not torch.equal(observed, baseline[..., horizon_index, :])


def test_p1_zero_initialization_is_elementwise_equal_to_p0() -> None:
    generator = torch.Generator().manual_seed(802)
    tokens = torch.randn(2, 4, 8, 12, generator=generator)
    p0 = HorizonCausalActionAggregator(
        d_model=12,
        action_horizon=8,
        future_horizons=(1, 3, 5, 8),
        kind="prefix_mean",
        rank=4,
    )
    p1 = HorizonCausalActionAggregator(
        d_model=12,
        action_horizon=8,
        future_horizons=(1, 3, 5, 8),
        kind="causal_prefix_attention",
        rank=4,
    )
    assert p1.audit()["output_projection_zero_initialized"] is True
    assert torch.equal(p0(tokens), p1(tokens))
    p1(tokens).square().mean().backward()
    assert p1.output_weight.grad is not None
    assert torch.count_nonzero(p1.output_weight.grad) > 0


def test_full_provider_has_canonical_shapes_masks_and_step0_exactness() -> None:
    config = _local_config()
    source = _source(config)
    base_local = copy.deepcopy(source.protected_own)
    p0 = HorizonCausalActiveTeamFutureProvider(
        copy.deepcopy(base_local),
        source,
        action_prefix_aggregator="prefix_mean",
        action_prefix_rank=4,
    ).eval()
    p1 = HorizonCausalActiveTeamFutureProvider(
        copy.deepcopy(base_local),
        source,
        action_prefix_aggregator="causal_prefix_attention",
        action_prefix_rank=4,
    ).eval()
    generator = torch.Generator().manual_seed(803)
    state = torch.randn(2, 4, 5, generator=generator)
    visual = torch.randn(2, 4, 2, 6, generator=generator)
    shared = torch.randn(2, 2, 6, generator=generator)
    actions = torch.randn(2, 4, 8, 2, generator=generator)
    valid = torch.tensor([[True, True, False, True], [True, False, False, False]])
    with torch.no_grad():
        left = p0(state, visual, shared, actions, valid)
        right = p1(state, visual, shared, actions, valid)
    for field in (
        "own_state",
        "own_visual",
        "peer_state",
        "peer_visual",
        "shared_visual",
    ):
        assert torch.equal(getattr(left, field), getattr(right, field)), field
    assert left.own_state.shape == (2, 4, 4, 5)
    assert left.own_visual.shape == (2, 4, 4, 2, 6)
    assert left.peer_state.shape == (2, 4, 4, 4, 5)
    assert left.peer_visual.shape == (2, 4, 4, 4, 2, 6)
    assert left.shared_visual.shape == (2, 4, 4, 2, 6)
    assert torch.count_nonzero(left.own_state[0, 2]) == 0
    assert torch.count_nonzero(left.peer_state[0, :, 2]) == 0
    assert torch.count_nonzero(left.peer_state[0, 2]) == 0


def test_full_provider_rejects_unknown_aggregator() -> None:
    config = _local_config()
    source = _source(config)
    with pytest.raises(ValueError, match="unregistered"):
        HorizonCausalActiveTeamFutureProvider(
            copy.deepcopy(source.protected_own),
            source,
            action_prefix_aggregator="whole_chunk_mean",
        )


def test_special_exact_audit_addresses_the_horizon_axis_for_every_source() -> None:
    config = LocalFuturePredictorConfig(
        max_agents=4,
        state_dim=5,
        action_dim=2,
        action_horizon=100,
        future_horizons=(1, 25, 50, 100),
        visual_grid_tokens=2,
        visual_latent_dim=6,
        d_model=12,
        ffn_dim=24,
        layers=1,
        heads=3,
        dropout=0.0,
    )
    source = _source(config)
    provider = HorizonCausalActiveTeamFutureProvider(
        copy.deepcopy(source.protected_own),
        source,
        action_prefix_aggregator="causal_prefix_attention",
        action_prefix_rank=4,
    ).eval()
    generator = torch.Generator().manual_seed(804)
    inputs = {
        "state": torch.randn(2, 4, 5, generator=generator),
        "local_visual": torch.randn(2, 4, 2, 6, generator=generator),
        "shared_visual": torch.randn(2, 2, 6, generator=generator),
    }
    actions = torch.randn(2, 4, 100, 2, generator=generator)
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    team = actions[:, None].expand(-1, 4, -1, -1, -1)
    with torch.inference_mode():
        baseline = provider(
            inputs["state"],
            inputs["local_visual"],
            inputs["shared_visual"],
            actions,
            valid,
            actions_by_focal=team,
        )
        for index, horizon in enumerate((1, 25, 50, 100)):
            result = _exact_interventions(
                provider,
                inputs,
                actions,
                team,
                valid,
                baseline,
                horizon_index=index,
                horizon=horizon,
            )
            assert set(result) == {"own", "peer", "shared"}
            assert all(row["suffix_max_abs_diff"] == 0.0 for row in result.values())
            assert all(row["prefix_max_abs_diff"] > 0.0 for row in result.values())
