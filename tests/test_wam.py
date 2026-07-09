import torch

from models.wam import LatentWorldActionModel, WAMConfig, compute_wam_losses, count_parameters


def test_wam_forward_shapes():
    cfg = WAMConfig(
        horizon=8,
        num_agents=2,
        slots_per_agent=4,
        slot_dim=32,
        plan_codebook_size=16,
        plan_latent_dim=16,
        action_dim=8,
        model_dim=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=256,
        use_checkpoint=False,
    )
    model = LatentWorldActionModel(cfg)
    B = 3

    current_slots = torch.randn(B, 2, 4, 32)
    plan_codes = torch.randint(0, 16, (B, 2))
    plan_residuals = torch.randn(B, 2, 16)

    out = model(current_slots, plan_codes, plan_residuals)
    assert out["pred_slots"].shape == (B, 8, 2, 4, 32)
    assert out["pred_actions"].shape == (B, 8, 8)
    assert out["pred_contact_logits"].shape == (B, 8)
    assert out["pred_force"].shape == (B, 8)
    assert out["pred_progress"].shape == (B, 8)
    assert count_parameters(model) > 0


def test_wam_loss_is_finite():
    cfg = WAMConfig(
        horizon=8,
        num_agents=2,
        slots_per_agent=4,
        slot_dim=32,
        plan_codebook_size=16,
        plan_latent_dim=16,
        action_dim=8,
        model_dim=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=256,
        use_checkpoint=False,
    )
    model = LatentWorldActionModel(cfg)
    B = 2

    batch = {
        "future_actions": torch.randn(B, 8, 8),
        "target_contact": torch.randint(0, 2, (B, 8)).float(),
        "target_force": torch.rand(B, 8),
        "target_progress": torch.rand(B, 8),
    }
    targets = {
        "current_slots": torch.randn(B, 2, 4, 32),
        "future_slots": torch.randn(B, 8, 2, 4, 32),
        "plan_codes": torch.randint(0, 16, (B, 2)),
        "plan_residuals": torch.randn(B, 2, 16),
    }
    losses = compute_wam_losses(model, batch, targets)
    assert torch.isfinite(losses["loss"])
