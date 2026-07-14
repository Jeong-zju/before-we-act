from __future__ import annotations

import torch

from models.research_v2 import (
    BeliefEncoderV2,
    BeliefEncoderV2Config,
    BlockTransitionWorldModelV2,
    DirectParallelWorldModelV2,
    WorldModelV2Config,
)
from train.research_v2_losses import plan_distribution_loss_v2


def _world_config() -> WorldModelV2Config:
    return WorldModelV2Config(
        horizon=4,
        block_length=2,
        belief_dim=16,
        model_dim=32,
        context_layers=1,
        transition_layers=2,
        heads=4,
        ffn_dim=64,
        dropout=0.0,
    )


def test_world_step_zero_cannot_see_later_actions_and_quantiles_are_monotone():
    torch.manual_seed(4)
    belief = torch.randn(2, 4, 16)
    own = torch.randn(2, 4, 4)
    peer = torch.randn(2, 4, 4)
    later_own = own.clone()
    later_peer = peer.clone()
    later_own[:, 1:] += 20.0
    later_peer[:, 1:] -= 20.0

    for model_class in (DirectParallelWorldModelV2, BlockTransitionWorldModelV2):
        model = model_class(_world_config()).eval()
        baseline = model(belief, own, peer)
        changed = model(belief, later_own, later_peer)
        torch.testing.assert_close(baseline["features"][:, 0], changed["features"][:, 0])
        assert torch.all(
            baseline["return_quantiles"][:, 1:]
            >= baseline["return_quantiles"][:, :-1]
        )


def test_belief_roles_cross_attend_to_temporal_evidence():
    model = BeliefEncoderV2(
        BeliefEncoderV2Config(
            history=3,
            local_dim=5,
            model_dim=16,
            num_heads=4,
            temporal_layers=1,
            role_layers=1,
            ffn_dim=32,
            dropout=0.0,
        )
    )
    assert isinstance(model.role_cross_attention, torch.nn.MultiheadAttention)


def test_plan_loss_skips_targets_outside_estimated_support_without_nan():
    output = {
        "code_logits": torch.randn(3, 4, requires_grad=True),
        "residual_mu_by_code": torch.randn(3, 4, 2, requires_grad=True),
        "residual_logvar_by_code": torch.zeros(3, 4, 2, requires_grad=True),
    }
    losses = plan_distribution_loss_v2(
        output,
        target_code=torch.tensor([0, 2, 3]),
        target_residual=torch.randn(3, 2),
        active_code_mask=torch.tensor([True, True, False, False]),
    )
    assert torch.isfinite(losses["loss"])
    torch.testing.assert_close(losses["active_target_fraction"], torch.tensor(1.0 / 3.0))
    losses["loss"].backward()

    all_inactive = plan_distribution_loss_v2(
        output,
        target_code=torch.tensor([2, 3, 2]),
        target_residual=torch.randn(3, 2),
        active_code_mask=torch.tensor([True, True, False, False]),
    )
    assert torch.isfinite(all_inactive["loss"])
    assert all_inactive["active_target_fraction"].item() == 0.0
