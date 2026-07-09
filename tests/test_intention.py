import torch

from models.intention import IntentionConfig, IntentionInferenceModel, compute_intention_losses


def test_intention_forward_shapes():
    cfg = IntentionConfig(
        slot_dim=32,
        slots_per_agent=4,
        plan_codebook_size=16,
        plan_latent_dim=16,
        model_dim=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=256,
    )
    model = IntentionInferenceModel(cfg)
    B = 3

    out = model(
        ego_slots=torch.randn(B, 4, 32),
        ego_plan_codes=torch.randint(0, 16, (B,)),
        ego_plan_residuals=torch.randn(B, 16),
        ego_id=torch.tensor([0, 1, 0]),
        phase_history=torch.randint(0, 9, (B, 8)),
        rel_target_pose=torch.randn(B, 3),
        object_rel_pose=torch.randn(B, 3),
    )

    assert out["target_code_logits"].shape == (B, 16)
    assert out["target_residual_mu"].shape == (B, 16)
    assert out["target_residual_logvar"].shape == (B, 16)
    assert out["uncertainty"].shape == (B,)


def test_intention_loss_is_finite():
    cfg = IntentionConfig(
        slot_dim=32,
        slots_per_agent=4,
        plan_codebook_size=16,
        plan_latent_dim=16,
        model_dim=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=256,
    )
    model = IntentionInferenceModel(cfg)
    B = 4

    batch = {
        "ego_id": torch.tensor([0, 1, 0, 1]),
        "phase_history": torch.randint(0, 9, (B, 8)),
        "rel_target_pose": torch.randn(B, 3),
        "object_rel_pose": torch.randn(B, 3),
    }
    targets = {
        "ego_slots": torch.randn(B, 4, 32),
        "ego_plan_codes": torch.randint(0, 16, (B,)),
        "ego_plan_residuals": torch.randn(B, 16),
        "target_plan_codes": torch.randint(0, 16, (B,)),
        "target_plan_residuals": torch.randn(B, 16),
    }

    losses = compute_intention_losses(model, batch, targets)
    assert torch.isfinite(losses["loss"])
    assert losses["pred_code"].shape == (B,)
