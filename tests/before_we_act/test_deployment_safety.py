from __future__ import annotations

import pytest
import torch

from before_we_act.deployment_safety import (
    DeploymentProgressWatchdog,
    ResidualSafetyConfig,
    calibrated_residual_safety,
)
from before_we_act.predictive_team_belief_policy import DirectBeliefResidual


def configured_residual(**overrides) -> DirectBeliefResidual:
    values = {
        "enabled": True,
        "max_residual_l2": 0.25,
        "max_belief_entropy": 0.6,
        "max_temporal_residual_delta_l2": 0.1,
    }
    values.update(overrides)
    module = DirectBeliefResidual(
        32, 8, safety=ResidualSafetyConfig(**values)
    ).eval()
    with torch.no_grad():
        module.output.weight.fill_(0.2)
        module.output.bias.fill_(0.2)
    return module


def inputs():
    torch.manual_seed(11)
    action = torch.randn(2, 5, 32)
    belief = torch.randn(2, 4, 32)
    sigma = torch.full_like(belief, 0.2)
    reliability = torch.ones(2, 1, 1)
    return action, belief, sigma, reliability


def test_residual_safety_hard_rejects_uncertainty_and_clips_norm() -> None:
    module = configured_residual()
    action, belief, sigma, reliability = inputs()
    safe, safe_gate = module(action, belief, sigma, reliability)
    unsafe_sigma = sigma.clone()
    unsafe_sigma[0] = 0.9
    guarded, guarded_gate = module(
        action, belief, unsafe_sigma, reliability
    )

    assert safe.float().norm(dim=-1).max() <= 0.250001
    assert torch.count_nonzero(guarded[0]) == 0
    assert torch.count_nonzero(guarded_gate[0]) == 0
    assert torch.count_nonzero(guarded[1]) > 0
    assert torch.count_nonzero(safe_gate) > 0


def test_safety_gate_adds_no_trainable_parameter_or_checkpoint_key() -> None:
    plain = DirectBeliefResidual(32, 8)
    guarded = configured_residual()

    assert plain.state_dict().keys() == guarded.state_dict().keys()
    assert sum(value.numel() for value in plain.parameters()) == sum(
        value.numel() for value in guarded.parameters()
    )


def test_residual_safety_rejects_temporally_unstable_action_mapping() -> None:
    module = configured_residual(max_temporal_residual_delta_l2=1e-6)
    action, belief, sigma, reliability = inputs()
    residual, gate = module(
        action,
        belief,
        sigma,
        reliability,
        previous_belief_mu=-belief,
        temporal_safety_active=torch.tensor([True, False]),
    )

    assert torch.count_nonzero(residual[0]) == 0
    assert torch.count_nonzero(gate[0]) == 0
    assert torch.count_nonzero(residual[1]) > 0


def test_progress_watchdog_requires_persistent_residual_induced_stall() -> None:
    config = ResidualSafetyConfig(
        enabled=True, progress_patience_steps=2, progress_recovery_steps=3
    )
    watchdog = DeploymentProgressWatchdog(config)
    assert watchdog.choose_base(
        candidate_inactive=True, base_inactive=False
    ) == (False, "candidate")
    assert watchdog.choose_base(
        candidate_inactive=True, base_inactive=False
    ) == (True, "residual_induced_stall")
    assert watchdog.choose_base(
        candidate_inactive=False, base_inactive=False
    ) == (True, "recovery_window")


def test_progress_watchdog_does_not_blame_shared_or_deliberate_waiting() -> None:
    config = ResidualSafetyConfig(
        enabled=True, progress_patience_steps=2, progress_recovery_steps=3
    )
    watchdog = DeploymentProgressWatchdog(config)
    for _ in range(5):
        assert watchdog.choose_base(
            candidate_inactive=True, base_inactive=True
        ) == (False, "candidate")


def test_calibration_uses_held_out_quantiles_without_success_labels() -> None:
    config = calibrated_residual_safety(
        {
            "belief_entropy": {"p99": 0.5},
            "target_residual_l2": {"p99": 0.2},
            "temporal_residual_delta_l2": {"p99": 0.03},
        }
    )
    assert config.enabled
    assert config.max_residual_l2 == pytest.approx(0.22)
    assert config.max_belief_entropy == pytest.approx(0.55)
    assert config.max_temporal_residual_delta_l2 == pytest.approx(0.0601)
