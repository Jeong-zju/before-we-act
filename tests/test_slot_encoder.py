import torch

from models.slot_encoder import AgentObjectSlotEncoder, SlotEncoderConfig, compute_slot_losses


def test_slot_encoder_forward_shapes():
    cfg = SlotEncoderConfig(history=8, local_dim=17, slot_dim=32, hidden_dim=64, num_object_slots=2, num_layers=2, num_heads=4, plan_codebook_size=16)
    model = AgentObjectSlotEncoder(cfg)

    batch = {
        "local_history": torch.randn(4, 8, 17),
        "agent_id": torch.tensor([0, 1, 0, 1]),
        "phase_history": torch.randint(0, 9, (4, 8)),
        "self_pose": torch.randn(4, 3),
        "other_rel_pose": torch.randn(4, 3),
        "object_rel_pose": torch.randn(4, 3),
        "contact": torch.randint(0, 2, (4,)).float(),
        "force_proxy": torch.rand(4),
        "phase": torch.randint(0, 9, (4,)),
        "plan_token": torch.randint(0, 16, (4,)),
    }

    out = model(batch["local_history"], batch["agent_id"], batch["phase_history"])
    assert out["slots"].shape == (4, 4, 32)
    assert out["self_slot"].shape == (4, 32)
    assert out["other_slot"].shape == (4, 32)
    assert out["object_slots"].shape == (4, 2, 32)
    assert out["pred_self_pose"].shape == (4, 3)
    assert out["pred_other_rel_pose"].shape == (4, 3)
    assert out["pred_object_rel_pose"].shape == (4, 3)
    assert out["pred_phase_logits"].shape == (4, 9)
    assert out["pred_plan_logits"].shape == (4, 16)

    losses = compute_slot_losses(model, batch)
    assert torch.isfinite(losses["loss"])


def test_slot_encoder_encode_api():
    cfg = SlotEncoderConfig(history=4, local_dim=17, slot_dim=16, hidden_dim=32, num_object_slots=2, num_layers=1, num_heads=4, plan_codebook_size=8)
    model = AgentObjectSlotEncoder(cfg)

    enc = model.encode_slots(
        torch.randn(2, 4, 17),
        torch.tensor([0, 1]),
        torch.randint(0, 9, (2, 4)),
    )
    assert enc["slots"].shape == (2, 4, 16)
