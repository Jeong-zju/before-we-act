from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from models.static_rgb_act import StaticRGBMoEACTConfig
from models.wam_multimodal.agent_factorized_flow_wam import AgentFactorizedFlowWAM
from models.wam_multimodal.cross_agent_world_conditioned_flow import (
    CrossAgentWorldConditionedFlow,
    PredictedFutureLatents,
    WorldToFlowAdapterConfig,
)
from models.wam_multimodal.local_future_predictor import (
    LocalActionConditionedFuturePredictor,
    LocalFuturePredictorConfig,
)
from models.wam_multimodal.protected_team_future_predictor import (
    ProtectedTeamFuturePredictor,
    ProtectedTeamFuturePredictorConfig,
)
from models.wam_multimodal.utility_calibrated_world_flow import (
    ActiveParentVelocityCache,
    S4ActiveTeamFutureProvider,
    ScaleAlignedActiveWorldFlow,
    UtilityCalibratedWorldFlow,
    world_utility_coupling_loss,
)
from models.wam_multimodal.world_evidence_router import (
    FutureEvidenceRouter,
    LowRankEvidenceAdapterBank,
    S4WorldEvidenceProvider,
    UtilityCalibratedResidual,
    WorldEvidenceRouterConfig,
    group_index,
)


def _config() -> WorldEvidenceRouterConfig:
    return WorldEvidenceRouterConfig()


def _futures(batch_size: int = 1) -> PredictedFutureLatents:
    generator = torch.Generator().manual_seed(704)
    return PredictedFutureLatents(
        own_state=torch.randn(batch_size, 4, 4, 18, generator=generator),
        own_visual=torch.randn(batch_size, 4, 4, 4, 256, generator=generator),
        peer_state=torch.randn(batch_size, 4, 4, 4, 18, generator=generator),
        peer_visual=torch.randn(
            batch_size, 4, 4, 4, 4, 256, generator=generator
        ),
        shared_visual=torch.randn(
            batch_size, 4, 4, 4, 256, generator=generator
        ),
    )


def _valid() -> Tensor:
    return torch.tensor([[True, True, False, True]])


def test_s4_provider_preserves_tokens_and_enforces_exact_source_masks() -> None:
    provider = S4WorldEvidenceProvider(_config())
    evidence = provider.pack(_futures(), _valid())
    assert evidence.tokens.shape == (1, 4, 3, 4, 4, 5, 384)
    assert evidence.mask.shape == (1, 4, 3, 4, 4, 5)
    assert evidence.mask.dtype == torch.bool
    # Own opens only the focal source-agent slot.
    for focal in range(4):
        for source_agent in range(4):
            assert bool(evidence.mask[0, focal, 0, source_agent].any()) is (
                bool(_valid()[0, focal]) and source_agent == focal
            )
    # Peer keeps every other valid agent and explicitly masks self/invalid slots.
    for focal in (0, 1, 3):
        assert not bool(evidence.mask[0, focal, 1, focal].any())
        assert not bool(evidence.mask[0, focal, 1, 2].any())
        for source_agent in ({0, 1, 3} - {focal}):
            assert bool(evidence.mask[0, focal, 1, source_agent].all())
    # Shared owns exactly one common source-agent slot and has no state token.
    assert not bool(evidence.mask[:, :, 2, 1:].any())
    assert not bool(evidence.mask[:, :, 2, 0, :, 0].any())
    assert bool(evidence.mask[0, 0, 2, 0, :, 1:].all())
    assert not bool(evidence.mask[0, 2].any())
    assert torch.count_nonzero(evidence.tokens[~evidence.mask]) == 0
    # Distinct peer source-agent tokens remain distinct rather than being pooled.
    assert not torch.equal(
        evidence.tokens[0, 0, 1, 1], evidence.tokens[0, 0, 1, 3]
    )


def test_rank32_adapter_and_dense_router_are_masked_and_detach_inputs() -> None:
    config = _config()
    provider = S4WorldEvidenceProvider(config)
    evidence = provider.pack(_futures(), _valid())
    evidence.tokens.retain_grad()
    q = torch.randn(1, 4, 3, 384, requires_grad=True)
    adapter = LowRankEvidenceAdapterBank(config)
    adapted = adapter(q, evidence)
    assert adapted.z.shape == (1, 4, 3, 12, 384)
    assert adapted.group_mask.shape == (1, 4, 12)
    assert all(layer.out_features == 32 for layer in adapter.query_projections)
    assert all(layer.in_features == 32 for layer in adapter.output_projections)
    assert not bool(adapted.group_mask[0, 2].any())
    assert torch.count_nonzero(adapted.z[0, 2]) == 0

    router = FutureEvidenceRouter(config)
    routed = router(q, evidence, group_mask=adapted.group_mask)
    assert routed.logits.shape == routed.pi.shape == (1, 4, 3, 12)
    assert torch.isneginf(routed.logits[0, 2]).all()
    assert torch.count_nonzero(routed.pi[0, 2]) == 0
    torch.testing.assert_close(
        routed.pi[0, (0, 1, 3)].sum(dim=-1),
        torch.ones(3, 3),
    )
    weighted = torch.arange(12, dtype=routed.pi.dtype)
    loss = (routed.pi * weighted).sum()
    loss.backward()
    assert q.grad is None
    assert evidence.tokens.grad is None
    assert _gradient_norm(router) > 0


def _small_local_config() -> LocalFuturePredictorConfig:
    return LocalFuturePredictorConfig(
        max_agents=4,
        state_dim=5,
        action_dim=2,
        action_horizon=4,
        future_horizons=(1, 4),
        visual_grid_tokens=2,
        visual_latent_dim=6,
        d_model=12,
        ffn_dim=24,
        layers=1,
        heads=3,
        dropout=0.0,
    )


def _r5_p0_source(
    local_config: LocalFuturePredictorConfig,
) -> ProtectedTeamFuturePredictor:
    own = LocalActionConditionedFuturePredictor(local_config)
    source = ProtectedTeamFuturePredictor(
        local_config,
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


def test_active_team_provider_uses_one_trainable_local_clone_without_protected_path() -> None:
    config = _small_local_config()
    r5_source = _r5_p0_source(config)
    active_local = copy.deepcopy(r5_source.protected_own)
    provider = S4ActiveTeamFutureProvider(active_local, r5_source)
    assert provider.local_predictor is active_local
    assert provider.training and provider.local_predictor.training
    assert all(parameter.requires_grad for parameter in provider.parameters())
    assert all("protected_own" not in name for name, _ in provider.named_parameters())
    generator = torch.Generator().manual_seed(706)
    state = torch.randn(1, 4, 5, generator=generator)
    visual = torch.randn(1, 4, 2, 6, generator=generator)
    shared = torch.randn(1, 2, 6, generator=generator)
    actions = torch.randn(1, 4, 4, 2, generator=generator)
    valid = _valid()
    expected_own = active_local(state, visual, actions, valid, valid)
    prediction = provider(state, visual, shared, actions, valid)
    assert torch.equal(prediction.own_state, expected_own[0])
    assert torch.equal(prediction.own_visual, expected_own[1])
    assert prediction.peer_state.shape == (1, 4, 4, 2, 5)
    assert prediction.peer_visual.shape == (1, 4, 4, 2, 2, 6)
    assert prediction.shared_visual.shape == (1, 4, 2, 2, 6)
    loss = (
        prediction.own_state.square().mean()
        + prediction.peer_state.square().mean()
        + prediction.shared_visual.square().mean()
    )
    loss.backward()
    assert _gradient_norm(provider.local_predictor) > 0
    assert _gradient_norm(provider.shared_mixer) > 0
    assert _gradient_norm(r5_source) == 0


def test_scale_aligned_parent_is_trainable_and_returns_one_complete_cache() -> None:
    flow_config = StaticRGBMoEACTConfig(
        state_dim=5,
        action_dim=2,
        horizon=4,
        vision_dim=6,
        d_model=12,
        encoder_layers=1,
        decoder_layers=1,
        heads=3,
        ffn_dim=24,
        latent_dim=4,
        experts=2,
        dropout=0.0,
        decoder_kind="dense",
        dense_ffn_dim=24,
    )
    local_config = _small_local_config()
    legacy = CrossAgentWorldConditionedFlow(
        AgentFactorizedFlowWAM(flow_config),
        LocalActionConditionedFuturePredictor(local_config),
        WorldToFlowAdapterConfig(
            flow_dim=12,
            state_dim=5,
            visual_dim=6,
            hidden_dim=12,
            action_dim=2,
            max_gate=0.25,
        ),
        future_scope="local",
        injection=True,
    )
    r5_source = _r5_p0_source(local_config)
    legacy_requires_grad = {
        name: parameter.requires_grad for name, parameter in legacy.named_parameters()
    }
    active = ScaleAlignedActiveWorldFlow.from_legacy_reference(legacy, r5_source)
    assert active.training and active.future_predictor.local_predictor.training
    assert all(parameter.requires_grad for parameter in active.parameters())
    assert {
        name: parameter.requires_grad for name, parameter in legacy.named_parameters()
    } == legacy_requires_grad
    with torch.no_grad():
        active.legacy_adapter.gate_alpha.fill_(0.4)
    generator = torch.Generator().manual_seed(707)
    inputs = (
        torch.randn(1, 4, 3, 6, generator=generator),
        torch.randn(1, 4, 5, generator=generator),
        torch.randn(1, 4, 2, 6, generator=generator),
        torch.randn(1, 2, 6, generator=generator),
        torch.randn(1, 4, 4, 2, generator=generator),
        torch.rand(1, generator=generator),
        _valid(),
    )
    velocity, _auxiliary, cache = active.velocity(*inputs, return_cache=True)
    zero_velocity, _ = active.velocity(*inputs, force_gate_zero=True)
    assert not torch.equal(velocity, zero_velocity)
    assert torch.equal(cache.active_parent_velocity, velocity)
    assert cache.flow_features.shape == (1, 4, 4, 12)
    assert cache.clean_actions.shape == (1, 4, 4, 2)
    assert isinstance(cache.predicted_futures, PredictedFutureLatents)
    loss = velocity.square().mean() + cache.predicted_futures.peer_state.square().mean()
    loss.backward()
    assert _gradient_norm(active.base_flow) > 0
    assert _gradient_norm(active.future_predictor.local_predictor) > 0
    assert _gradient_norm(active.future_predictor.shared_mixer) > 0
    assert _gradient_norm(active.legacy_adapter) > 0


class _SyntheticActiveParent(nn.Module):
    def __init__(self, futures: PredictedFutureLatents, queries: int = 3) -> None:
        super().__init__()
        self.base_velocity = nn.Parameter(torch.linspace(-0.2, 0.2, 8))
        self.old_world_residual = nn.Parameter(torch.full((8,), 0.1))
        self.flow_query = nn.Parameter(torch.randn(queries, 384) * 0.02)
        self.futures = futures
        self.last_force_gate_zero = False

    def velocity(
        self,
        _base_vision_tokens: Tensor,
        current_state: Tensor,
        _current_visual_latent: Tensor,
        _shared_visual_latent: Tensor,
        action_inputs: Tensor,
        flow_time: Tensor,
        valid_agent_mask: Tensor,
        *,
        force_gate_zero: bool = False,
        return_cache: bool = False,
    ) -> tuple[Tensor, dict[str, object], ActiveParentVelocityCache] | tuple[
        Tensor, dict[str, object]
    ]:
        self.last_force_gate_zero = force_gate_zero
        batch_size, agents, queries = action_inputs.shape[:3]
        base = self.base_velocity.view(1, 1, 1, 8).expand(
            batch_size, agents, queries, -1
        )
        velocity = base if force_gate_zero else base + self.old_world_residual
        velocity = velocity * valid_agent_mask[:, :, None, None]
        flow_features = self.flow_query[None, None].expand(
            batch_size, agents, -1, -1
        )
        clean_actions = action_inputs + (
            1 - flow_time[:, None, None, None]
        ) * velocity
        auxiliary: dict[str, object] = {"old_gate_zero": force_gate_zero}
        if not return_cache:
            return velocity, auxiliary
        return velocity, auxiliary, ActiveParentVelocityCache(
            active_parent_velocity=velocity,
            flow_features=flow_features,
            clean_actions=clean_actions,
            predicted_futures=self.futures,
        )


class _CountingProvider(S4WorldEvidenceProvider):
    def __init__(self, config: WorldEvidenceRouterConfig) -> None:
        super().__init__(config)
        self.calls = 0

    def forward(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        self.calls += 1
        return super().forward(*args, **kwargs)  # type: ignore[arg-type]


def _flow_model() -> tuple[
    UtilityCalibratedWorldFlow, _SyntheticActiveParent, _CountingProvider
]:
    config = _config()
    parent = _SyntheticActiveParent(_futures())
    provider = _CountingProvider(config)
    return UtilityCalibratedWorldFlow(parent, provider, config), parent, provider


def _flow_inputs() -> tuple[Tensor, ...]:
    generator = torch.Generator().manual_seed(705)
    return (
        torch.randn(1, 4, 2, 16, generator=generator),
        torch.randn(1, 4, 18, generator=generator),
        torch.randn(1, 4, 4, 256, generator=generator),
        torch.randn(1, 4, 256, generator=generator),
        torch.randn(1, 4, 3, 8, generator=generator),
        torch.rand(1, generator=generator),
        _valid(),
    )


def test_zero_gate_and_all_world_zero_are_exact_active_parent_paths() -> None:
    model, parent, provider = _flow_model()
    inputs = _flow_inputs()
    expected_active = parent.velocity(*inputs)[0]
    skipped, skipped_diagnostics = model.velocity(
        *inputs, force_world_evidence_gate_zero=True
    )
    assert provider.calls == 0
    assert torch.equal(skipped, expected_active)
    assert skipped_diagnostics["router_pi"] is None

    executed, executed_diagnostics = model.velocity(
        *inputs,
        force_world_evidence_gate_zero=True,
        execute_evidence_when_gate_zero=True,
    )
    assert provider.calls == 1
    assert torch.equal(executed, expected_active)
    assert isinstance(executed_diagnostics["router_pi"], Tensor)

    expected_base = parent.velocity(*inputs, force_gate_zero=True)[0]
    all_zero, _ = model.velocity(*inputs, force_all_world_gates_zero=True)
    assert parent.last_force_gate_zero is True
    assert torch.equal(all_zero, expected_base)

    normal, diagnostics = model.velocity(*inputs)
    # The query-wise gate is exactly zero at initialization.
    assert torch.equal(normal, expected_active)
    assert torch.count_nonzero(diagnostics["new_gate"]) == 0  # type: ignore[arg-type]
    assert torch.count_nonzero(model.residual.query_gate.weight) == 0
    assert torch.count_nonzero(model.residual.query_gate.bias) == 0


def test_forced_group_errors_wuc_scope_and_normal_gradient_scope() -> None:
    model, parent, _provider = _flow_model()
    inputs = _flow_inputs()
    with torch.no_grad():
        model.residual.query_gate.bias.fill_(0.4)
    forced, forced_diagnostics = model.velocity(
        *inputs, forced_group=group_index("peer", 25)
    )
    effective_pi = forced_diagnostics["effective_pi"]
    assert isinstance(effective_pi, Tensor)
    assert torch.count_nonzero(effective_pi[..., group_index("peer", 25)]) > 0
    assert torch.count_nonzero(effective_pi.sum(dim=-1) > 1) == 0
    assert not torch.equal(forced, forced_diagnostics["active_parent_velocity"])
    gate = forced_diagnostics["new_gate"]
    assert isinstance(gate, Tensor)
    assert float(gate.detach().abs().max()) <= 0.25

    model.zero_grad(set_to_none=True)
    normal, diagnostics = model.velocity(*inputs)
    target_velocity = torch.randn_like(normal)
    audit = model.forced_evidence_audit(
        diagnostics,
        target_velocity,
        inputs[-1],
    )
    assert audit.velocity_errors.shape == (1, 4, 3, 12)
    assert audit.utility_target.requires_grad is False
    assert bool(audit.valid_query_mask[0, (0, 1, 3)].all())
    torch.testing.assert_close(
        audit.utility_target[audit.valid_query_mask].sum(dim=-1),
        torch.ones(int(audit.valid_query_mask.sum())),
    )
    router_pi = diagnostics["router_pi"]
    group_mask = diagnostics["group_mask"]
    assert isinstance(router_pi, Tensor) and isinstance(group_mask, Tensor)
    wuc = world_utility_coupling_loss(
        router_pi,
        audit.utility_target,
        group_mask,
        inputs[-1],
    )
    wuc.backward()
    assert _gradient_norm(model.router) > 0
    assert _gradient_norm(model.evidence_adapter) == 0
    assert _gradient_norm(model.evidence_provider) == 0
    assert _gradient_norm(model.residual) == 0
    assert _gradient_norm(parent) == 0

    model.zero_grad(set_to_none=True)
    normal, _ = model.velocity(*inputs)
    normal.square().mean().backward()
    assert _gradient_norm(parent) > 0
    assert _gradient_norm(model.evidence_provider) > 0
    assert _gradient_norm(model.evidence_adapter) > 0
    assert _gradient_norm(model.router) > 0
    assert _gradient_norm(model.residual) > 0


def _gradient_norm(module: nn.Module) -> float:
    return sum(
        float(parameter.grad.detach().float().norm())
        for parameter in module.parameters()
        if parameter.grad is not None
    )
