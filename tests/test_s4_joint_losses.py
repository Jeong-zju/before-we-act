from __future__ import annotations

import math

import pytest
import torch

from train.s4_joint_losses import (
    S4JointLoss,
    s4_flow_loss,
    s4_joint_losses,
    s4_peer_state_loss,
    s4_shared_visual_loss,
)


def test_flow_reduction_gives_two_and_four_agent_teams_equal_weight() -> None:
    prediction = torch.zeros(2, 4, 2, 2)
    prediction[0, :2, 0] = 1.0
    prediction[1, :, 0] = 3.0
    prediction[:, :, 1] = 100.0
    prediction.requires_grad_()
    valid_agents = torch.tensor(
        [[True, True, False, False], [True, True, True, True]]
    )
    valid_horizon = torch.tensor([[True, False], [True, False]])

    loss = s4_flow_loss(
        prediction,
        torch.zeros_like(prediction),
        valid_agents,
        valid_horizon,
    )

    # Team losses are 1 and 9.  A flattened-agent reduction would be 19/3.
    torch.testing.assert_close(loss, torch.tensor(5.0))
    loss.backward()
    assert prediction.grad is not None
    assert not bool(prediction.grad[:, :, 1].any())
    assert not bool(prediction.grad[0, 2:].any())
    assert bool(prediction.grad[0, :2, 0].any())
    assert bool(prediction.grad[1, :, 0].any())


def _visual_prediction(shape: tuple[int, ...]) -> torch.Tensor:
    value = torch.zeros(*shape, 2)
    value[..., 1] = 1.0
    return value.requires_grad_()


def _visual_target(shape: tuple[int, ...]) -> torch.Tensor:
    value = torch.zeros(*shape, 2)
    value[..., 0] = 1.0
    return value


def test_joint_loss_combines_fixed_components_and_masks_peer_self() -> None:
    batch, agents, horizon, futures, grid = 2, 4, 2, 2, 1
    valid_agents = torch.tensor(
        [[True, True, False, False], [True, True, True, True]]
    )
    future_valid = valid_agents[:, :, None].expand(-1, -1, futures)
    flow = torch.ones(batch, agents, horizon, 2, requires_grad=True)
    own_state = torch.ones(batch, agents, futures, 3, requires_grad=True)
    own_visual = _visual_prediction((batch, agents, futures, grid))
    peer_state = torch.ones(
        batch, agents, agents, futures, 3, requires_grad=True
    )
    peer_visual = _visual_prediction(
        (batch, agents, agents, futures, grid)
    )
    shared_visual = _visual_prediction((batch, agents, futures, grid))

    result = s4_joint_losses(
        flow_prediction=flow,
        flow_target=torch.zeros_like(flow),
        flow_valid_mask=torch.ones(batch, horizon, dtype=torch.bool),
        valid_agent_mask=valid_agents,
        own_state_prediction=own_state,
        own_state_target=torch.zeros_like(own_state),
        own_state_valid_mask=future_valid,
        own_visual_prediction=own_visual,
        own_visual_target=_visual_target(
            (batch, agents, futures, grid)
        ),
        own_visual_valid_mask=future_valid,
        peer_state_prediction=peer_state,
        peer_state_target=torch.zeros(batch, agents, futures, 3),
        peer_state_valid_mask=future_valid,
        peer_visual_prediction=peer_visual,
        peer_visual_target=_visual_target(
            (batch, agents, futures, grid)
        ),
        peer_visual_valid_mask=future_valid,
        shared_visual_prediction=shared_visual,
        shared_visual_target=_visual_target((batch, futures, grid)),
        shared_visual_valid_mask=torch.ones(
            batch, futures, dtype=torch.bool
        ),
    )

    assert isinstance(result, S4JointLoss)
    assert result["loss"] is result.total
    torch.testing.assert_close(result.flow, torch.tensor(1.0))
    torch.testing.assert_close(result.own_state, torch.tensor(0.5))
    torch.testing.assert_close(result.peer_state, torch.tensor(0.5))
    torch.testing.assert_close(result.state, torch.tensor(0.5))
    torch.testing.assert_close(result.own_visual, torch.tensor(1.0))
    torch.testing.assert_close(result.peer_visual, torch.tensor(1.0))
    torch.testing.assert_close(result.shared_visual, torch.tensor(1.0))
    torch.testing.assert_close(result.visual, torch.tensor(1.0))
    torch.testing.assert_close(result.total, torch.tensor(1.375))

    result.total.backward()
    assert flow.grad is not None and bool(flow.grad[valid_agents].any())
    assert own_state.grad is not None and bool(own_state.grad[valid_agents].any())
    assert own_visual.grad is not None and bool(own_visual.grad[valid_agents].any())
    assert peer_state.grad is not None
    assert peer_visual.grad is not None
    assert shared_visual.grad is not None
    diagonal = torch.arange(agents)
    assert not bool(peer_state.grad[:, diagonal, diagonal].any())
    assert not bool(peer_visual.grad[:, diagonal, diagonal].any())
    assert not bool(peer_state.grad[0, 2:].any())
    assert bool(peer_state.grad[0, :2].any())


def test_empty_optional_targets_stay_zero_without_reweighting_batch() -> None:
    valid_agents = torch.ones(2, 2, dtype=torch.bool)
    peer_prediction = torch.ones(2, 2, 2, 1, 1, requires_grad=True)
    peer_valid = torch.tensor(
        [[[False], [False]], [[True], [True]]]
    )
    peer = s4_peer_state_loss(
        peer_prediction,
        torch.zeros(2, 2, 1, 1),
        peer_valid,
        valid_agents,
    )
    # Team 0 has no optional target and contributes zero; team 1 contributes
    # Smooth-L1(1,0)=0.5.  The fixed batch mean is therefore 0.25.
    torch.testing.assert_close(peer, torch.tensor(0.25))

    shared_prediction = _visual_prediction((2, 2, 1, 1))
    shared = s4_shared_visual_loss(
        shared_prediction,
        _visual_target((2, 1, 1)),
        torch.tensor([[False], [True]]),
        valid_agents,
    )
    torch.testing.assert_close(shared, torch.tensor(0.5))
    combined = peer + shared
    assert math.isfinite(float(combined.detach()))
    combined.backward()
    assert peer_prediction.grad is not None
    assert not bool(peer_prediction.grad[0].any())
    assert bool(peer_prediction.grad[1].any())
    assert shared_prediction.grad is not None
    assert not bool(shared_prediction.grad[0].any())
    assert bool(shared_prediction.grad[1].any())


def test_absent_peer_and_shared_keep_fixed_state_visual_divisors() -> None:
    valid_agents = torch.tensor([[True, True]])
    flow = torch.ones(1, 2, 1, 1, requires_grad=True)
    own_state = torch.ones(1, 2, 1, 1, requires_grad=True)
    own_visual = _visual_prediction((1, 2, 1, 1))
    result = s4_joint_losses(
        flow_prediction=flow,
        flow_target=torch.zeros_like(flow),
        flow_valid_mask=torch.ones(1, 1, dtype=torch.bool),
        valid_agent_mask=valid_agents,
        own_state_prediction=own_state,
        own_state_target=torch.zeros_like(own_state),
        own_state_valid_mask=torch.ones(1, 2, 1, dtype=torch.bool),
        own_visual_prediction=own_visual,
        own_visual_target=_visual_target((1, 2, 1, 1)),
        own_visual_valid_mask=torch.ones(1, 2, 1, dtype=torch.bool),
    )

    torch.testing.assert_close(result.peer_state, torch.tensor(0.0))
    torch.testing.assert_close(result.peer_visual, torch.tensor(0.0))
    torch.testing.assert_close(result.shared_visual, torch.tensor(0.0))
    torch.testing.assert_close(result.state, torch.tensor(0.25))
    torch.testing.assert_close(result.visual, torch.tensor(1.0 / 3.0))
    torch.testing.assert_close(
        result.total,
        torch.tensor(1.0 + 0.25 * 0.25 + 0.25 / 3.0),
    )
    result.total.backward()
    assert flow.grad is not None and bool(flow.grad.any())
    assert own_state.grad is not None and bool(own_state.grad.any())
    assert own_visual.grad is not None and bool(own_visual.grad.any())


def test_joint_loss_rejects_non_boolean_masks_and_partial_optional_inputs() -> None:
    with pytest.raises(TypeError, match="bool"):
        s4_flow_loss(
            torch.zeros(1, 1, 1, 1),
            torch.zeros(1, 1, 1, 1),
            torch.ones(1, 1),
            torch.ones(1, 1, dtype=torch.bool),
        )

    with pytest.raises(ValueError, match="all-or-none"):
        s4_joint_losses(
            flow_prediction=torch.zeros(1, 1, 1, 1),
            flow_target=torch.zeros(1, 1, 1, 1),
            flow_valid_mask=torch.ones(1, 1, dtype=torch.bool),
            valid_agent_mask=torch.ones(1, 1, dtype=torch.bool),
            own_state_prediction=torch.zeros(1, 1, 1, 1),
            own_state_target=torch.zeros(1, 1, 1, 1),
            own_state_valid_mask=torch.ones(1, 1, 1, dtype=torch.bool),
            own_visual_prediction=torch.ones(1, 1, 1, 1, 2),
            own_visual_target=torch.ones(1, 1, 1, 1, 2),
            own_visual_valid_mask=torch.ones(1, 1, 1, dtype=torch.bool),
            peer_state_prediction=torch.zeros(1, 1, 1, 1, 1),
        )
