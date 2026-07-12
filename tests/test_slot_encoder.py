import inspect

import pytest
import torch

from models.slot_encoder import (
    LocalBeliefSlotEncoder,
    LocalBeliefSlotEncoderConfig,
    compute_local_belief_auxiliary_losses,
)


def test_local_belief_encoder_has_fixed_roles_and_strict_local_api():
    cfg = LocalBeliefSlotEncoderConfig(
        history=5,
        local_dim=7,
        object_dim=3,
        slot_dim=16,
        hidden_dim=32,
        num_heads=4,
        num_history_layers=1,
        num_slot_layers=1,
        dropout=0.0,
        privileged_aux_dims={"object_pose": 3},
        privileged_aux_roles={"object_pose": "object-belief"},
    )
    model = LocalBeliefSlotEncoder(cfg).eval()
    local = torch.randn(3, 5, 7)
    history_mask = torch.tensor(
        [[False, True, True, True, True], [True, True, True, False, False], [True, True, True, True, True]]
    )
    object_observation = torch.randn(3, 5, 3)
    object_valid = torch.tensor(
        [[False, True, True, False, True], [True, False, True, False, False], [False, False, False, False, False]]
    )
    object_age = torch.rand(3, 5) * 10.0
    object_confidence = torch.rand(3, 5)

    forward_parameters = inspect.signature(model.forward).parameters
    assert "phase_history" not in forward_parameters
    assert "teammate_pose" not in forward_parameters
    assert "teammate_state" not in forward_parameters
    out = model(
        local,
        history_mask,
        torch.tensor([0, 1, 0]),
        object_observation=object_observation,
        object_valid=object_valid,
        object_age=object_age,
        object_confidence=object_confidence,
    )

    assert model.ROLE_NAMES == ("self", "object-belief", "teammate-belief", "task-context")
    assert out["slots"].shape == (3, 4, 16)
    assert out["self_slot"].shape == (3, 16)
    assert out["object_belief_slot"].shape == (3, 16)
    assert out["teammate_belief_slot"].shape == (3, 16)
    assert out["task_context_slot"].shape == (3, 16)
    assert out["privileged_predictions"]["object_pose"].shape == (3, 3)
    expected_probe = model.privileged_aux_heads["object_pose"](out["object_belief_slot"])
    torch.testing.assert_close(out["privileged_predictions"]["object_pose"], expected_probe)


def test_local_belief_encoder_masks_padding_and_missing_object_values():
    cfg = LocalBeliefSlotEncoderConfig(
        history=4,
        local_dim=5,
        object_dim=2,
        slot_dim=16,
        hidden_dim=32,
        num_heads=4,
        num_history_layers=1,
        num_slot_layers=1,
        dropout=0.0,
    )
    model = LocalBeliefSlotEncoder(cfg).eval()
    local = torch.randn(2, 4, 5)
    history_mask = torch.tensor([[False, True, True, True], [True, True, False, False]])
    object_observation = torch.randn(2, 4, 2)
    object_valid = torch.tensor([[False, True, False, True], [True, False, False, False]])
    object_age = torch.rand(2, 4)
    # confidence=0 is fully missing, so arbitrary placeholders must be ignored.
    object_confidence = torch.where(object_valid, torch.rand(2, 4), torch.zeros(2, 4))
    kwargs = {
        "object_valid": object_valid,
        "object_age": object_age,
        "object_confidence": object_confidence,
    }

    reference = model(
        local,
        history_mask,
        torch.tensor([0, 1]),
        object_observation=object_observation,
        **kwargs,
    )["slots"]

    changed_local = local.clone()
    changed_local[~history_mask] = 1e6
    changed_object = object_observation.clone()
    changed_object[(object_confidence == 0) | ~history_mask] = -1e6
    changed = model(
        changed_local,
        history_mask,
        torch.tensor([0, 1]),
        object_observation=changed_object,
        **kwargs,
    )["slots"]
    torch.testing.assert_close(changed, reference)


def test_local_belief_privileged_targets_are_loss_only():
    cfg = LocalBeliefSlotEncoderConfig(
        history=3,
        local_dim=4,
        slot_dim=8,
        hidden_dim=16,
        num_heads=2,
        num_history_layers=1,
        num_slot_layers=1,
        dropout=0.0,
        privileged_aux_dims={"sim_object_pose": 3, "sim_force": 1},
        privileged_aux_roles={"sim_object_pose": "object-belief", "sim_force": "self"},
    )
    model = LocalBeliefSlotEncoder(cfg)
    batch = {
        "local_history": torch.randn(2, 3, 4),
        "history_mask": torch.ones(2, 3, dtype=torch.bool),
        "agent_id": torch.tensor([0, 1]),
    }
    slots_before_loss = model.eval()(**batch)["slots"].detach().clone()
    model.train()
    losses = compute_local_belief_auxiliary_losses(
        model,
        batch,
        {
            "sim_object_pose": torch.randn(2, 3),
            "sim_force": torch.randn(2, 1),
        },
    )
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert model.privileged_aux_heads["sim_object_pose"].net[-1].weight.grad is not None

    # Merely changing a privileged label cannot change slot construction.
    slots_after_loss_setup = model.eval()(**batch)["slots"]
    torch.testing.assert_close(slots_after_loss_setup, slots_before_loss)


def test_local_belief_encoder_rejects_ambiguous_missingness():
    cfg = LocalBeliefSlotEncoderConfig(
        history=3,
        local_dim=4,
        object_dim=2,
        slot_dim=8,
        hidden_dim=16,
        num_heads=2,
        num_history_layers=1,
        num_slot_layers=1,
    )
    model = LocalBeliefSlotEncoder(cfg)
    local = torch.randn(1, 3, 4)
    mask = torch.ones(1, 3, dtype=torch.bool)
    object_observation = torch.randn(1, 3, 2)

    with pytest.raises(ValueError, match="requires valid, age, and confidence"):
        model(local, mask, torch.tensor([0]), object_observation=object_observation)

    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        model(
            local,
            mask,
            torch.tensor([0]),
            object_observation=object_observation,
            object_valid=torch.ones(1, 3, dtype=torch.bool),
            object_age=torch.zeros(1, 3),
            object_confidence=torch.full((1, 3), 1.1),
        )
