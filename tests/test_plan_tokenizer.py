import torch

from models.plan_tokenizer import PlanTokenizer, PlanTokenizerConfig, compute_losses


def test_plan_tokenizer_forward_shapes():
    cfg = PlanTokenizerConfig(horizon=16, action_dim=4, traj_dim=5, num_phases=9, latent_dim=32, hidden_dim=64, codebook_size=16)
    model = PlanTokenizer(cfg)

    batch = {
        "actions": torch.randn(4, 16, 4),
        "trajectory": torch.randn(4, 16, 5),
        "phase": torch.randint(0, 9, (4, 16)),
    }

    out = model(batch["actions"], batch["trajectory"])
    assert out["recon_actions"].shape == (4, 16, 4)
    assert out["recon_trajectory"].shape == (4, 16, 5)
    assert out["phase_logits"].shape == (4, 16, 9)
    assert out["code_indices"].shape == (4,)

    losses = compute_losses(model, batch)
    assert torch.isfinite(losses["loss"])
    assert losses["code_indices"].shape == (4,)


def test_plan_tokenizer_encode_decode_api():
    cfg = PlanTokenizerConfig(horizon=8, action_dim=4, traj_dim=5, num_phases=9, latent_dim=16, hidden_dim=32, codebook_size=8)
    model = PlanTokenizer(cfg)

    actions = torch.randn(2, 8, 4)
    trajectory = torch.randn(2, 8, 5)

    enc = model.encode_future_segment(actions, trajectory)
    dec = model.decode_plan_latent(enc["code_indices"], enc["residual"])

    assert enc["code_indices"].shape == (2,)
    assert enc["residual"].shape == (2, 16)
    assert dec["recon_actions"].shape == (2, 8, 4)
    assert dec["recon_trajectory"].shape == (2, 8, 5)
