import torch

from models.free_energy import FreeEnergyConfig, FreeEnergyEvaluator


def test_free_energy_shapes_and_finite():
    cfg = FreeEnergyConfig()
    evaluator = FreeEnergyEvaluator(cfg)

    B, H = 4, 16
    rollout = {
        "pred_progress": torch.rand(B, H),
        "pred_contact_logits": torch.randn(B, H),
        "pred_force": torch.rand(B, H),
        "pred_actions": torch.randn(B, H, 8),
    }
    uncertainty = torch.rand(B)

    scores = evaluator.total_score(rollout, uncertainty=uncertainty)

    for key in ["G", "L_goal", "L_safety", "L_collab", "U_intent", "C_ctrl"]:
        assert scores[key].shape == (B,)
        assert torch.isfinite(scores[key]).all()


def test_safety_cost_increases_with_contact():
    cfg = FreeEnergyConfig(alpha_safety=2.0)
    evaluator = FreeEnergyEvaluator(cfg)

    B, H = 2, 8
    base = {
        "pred_progress": torch.ones(B, H),
        "pred_contact_logits": torch.ones(B, H) * -8.0,
        "pred_force": torch.zeros(B, H),
        "pred_actions": torch.zeros(B, H, 8),
    }
    unsafe = {k: v.clone() for k, v in base.items()}
    unsafe["pred_contact_logits"] = torch.ones(B, H) * 8.0

    s0 = evaluator.total_score(base)["L_safety"]
    s1 = evaluator.total_score(unsafe)["L_safety"]
    assert (s1 > s0).all()
