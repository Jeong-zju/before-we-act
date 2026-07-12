"""Dynamics model tests."""

import inspect
import math

import torch

from models.communication import (
    CommunicationConfig,
    VPICommunicationTrigger,
    reply_plan_diagnostics,
)
from models.decentralized import (
    EgoLocalWAM,
    EgoLocalWAMConfig,
    LocalIntentionConfig,
    LocalIntentionPosterior,
)
from models.free_energy import (
    FreeEnergyConfig,
    FreeEnergyEvaluator,
    multi_hypothesis_expected_free_energy,
)


def test_ego_local_wam_uses_only_ego_slots_and_returns_ego_first_rollout():
    cfg = EgoLocalWAMConfig(
        horizon=5,
        slots_per_agent=4,
        slot_dim=16,
        plan_codebook_size=8,
        plan_latent_dim=6,
        model_dim=32,
        num_layers=1,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
    )
    model = EgoLocalWAM(cfg)
    B = 3
    out = model(
        ego_slots=torch.randn(B, 4, 16),
        plan_codes=torch.randint(0, 8, (B, 2)),
        plan_residuals=torch.randn(B, 2, 6),
        teammate_hypothesis_weight=torch.tensor([0.2, 0.3, 0.5]),
    )

    assert "teammate_slots" not in inspect.signature(model.forward).parameters
    assert out["pred_ego_slots"].shape == (B, 5, 4, 16)
    assert out["pred_actions"].shape == (B, 5, 8)
    assert out["pred_physical_outcome"].shape == (B, 5, 3)
    assert out["pred_contact_logits"].shape == (B, 5)
    assert torch.isfinite(out["pred_actions"]).all()


def test_wam_conditional_dynamics_do_not_depend_on_posterior_probability():
    cfg = EgoLocalWAMConfig(
        horizon=3,
        slots_per_agent=4,
        slot_dim=8,
        plan_codebook_size=4,
        plan_latent_dim=5,
        model_dim=16,
        num_layers=1,
        num_heads=4,
        ffn_dim=32,
        dropout=0.0,
    )
    model = EgoLocalWAM(cfg).eval()
    slots = torch.randn(1, 4, 8)
    codes = torch.tensor([[1, 2]])
    residuals = torch.randn(1, 2, 5)
    low_q = model(slots, codes, residuals, torch.tensor([0.1]))
    high_q = model(slots, codes, residuals, torch.tensor([0.9]))
    for key in ("pred_ego_slots", "pred_actions", "pred_physical_outcome"):
        assert torch.equal(low_q[key], high_q[key])


def test_local_intention_uncertainty_is_derived_from_posterior_statistics():
    cfg = LocalIntentionConfig(
        slots_per_agent=4,
        slot_dim=16,
        plan_codebook_size=8,
        plan_latent_dim=6,
        message_metadata_dim=3,
        model_dim=32,
        num_layers=1,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
    )
    model = LocalIntentionPosterior(cfg)
    B = 2
    out = model(
        ego_slots=torch.randn(B, 4, 16),
        ego_plan_code=torch.tensor([1, 2]),
        ego_plan_residual=torch.randn(B, 6),
        agent_id=torch.tensor([0, 1]),
        received_message_metadata=torch.zeros(B, 3),
    )

    expected_uncertainty = out["normalized_code_entropy"] + out["residual_variance"]
    assert not hasattr(model, "uncertainty_head")
    assert out["code_logits"].shape == (B, 8)
    assert out["residual_mu_by_code"].shape == (B, 8, 6)
    assert out["target_residual_mu"].shape == (B, 6)
    assert torch.allclose(out["uncertainty"], expected_uncertainty, atol=1e-6)
    assert torch.isfinite(out["uncertainty"]).all()

    hypotheses = model.topk_hypotheses(
        ego_slots=torch.randn(B, 4, 16),
        ego_plan_code=torch.tensor([1, 2]),
        ego_plan_residual=torch.randn(B, 6),
        agent_id=torch.tensor([0, 1]),
        received_message_metadata=torch.zeros(B, 3),
        k=3,
    )
    assert hypotheses["plan_codes"].shape == (B, 3)
    assert torch.allclose(hypotheses["hypothesis_weights"].sum(-1), torch.ones(B))


def test_multi_hypothesis_efe_uses_min_expected_and_expected_min():
    # Candidate 0 is right under hypothesis 0; candidate 1 is right under 1.
    # Before a reveal either candidate costs 5.  After a reveal the cost is 0.
    G = torch.tensor([[[0.0, 10.0], [10.0, 0.0]]])
    weights = torch.tensor([[0.5, 0.5]])
    out = multi_hypothesis_expected_free_energy(G, weights)

    assert torch.allclose(out["expected_G_by_candidate"], torch.tensor([[5.0, 5.0]]))
    assert torch.allclose(out["G_no"], torch.tensor([5.0]))
    assert torch.allclose(out["G_reveal"], torch.tensor([0.0]))
    assert torch.allclose(out["VPI"], torch.tensor([5.0]))


def test_free_energy_scores_candidate_hypothesis_grid():
    evaluator = FreeEnergyEvaluator(FreeEnergyConfig())
    B, K, M, H = 2, 3, 4, 5
    rollout = {
        "pred_progress": torch.rand(B, K, M, H),
        "pred_contact_logits": torch.randn(B, K, M, H),
        "pred_force": torch.rand(B, K, M, H),
        "pred_actions": torch.randn(B, K, M, H, 8),
    }
    weights = torch.rand(B, M)
    scores = evaluator.total_score_hypotheses(rollout, weights)

    assert scores["G"].shape == (B, K, M)
    assert scores["expected_G_by_candidate"].shape == (B, K)
    assert scores["G_no"].shape == (B,)
    assert scores["G_reveal"].shape == (B,)
    assert (scores["VPI"] >= 0).all()


def test_vpi_request_does_not_receive_true_message_code():
    cfg = CommunicationConfig(
        lambda_bits=0.0,
        lambda_delay=0.0,
        lambda_redundancy=99.0,
    )
    trigger = VPICommunicationTrigger(cfg)
    G = torch.tensor([[[0.0, 10.0], [10.0, 0.0]]])
    weights = torch.tensor([[0.5, 0.5]])
    out = trigger.decide_request(G, weights, request_cost=1.0)

    parameters = inspect.signature(trigger.decide_request).parameters
    assert "message_code" not in parameters
    assert "true_code" not in parameters
    assert out["trigger"].tolist() == [True]
    assert torch.allclose(out["G_no"], torch.tensor([5.0]))
    assert torch.allclose(out["G_reveal"], torch.tensor([0.0]))
    assert torch.all(out["bits"] == out["request_bits"] + out["reply_bits"])


def test_reply_diagnostics_reports_surprise_replan_and_action_change():
    residual_mu = torch.zeros(2, 2, 3)
    residual_logvar = torch.zeros_like(residual_mu)
    diagnostics = reply_plan_diagnostics(
        prior_code_probabilities=torch.tensor([[0.8, 0.2], [0.5, 0.5]]),
        reply_code=torch.tensor([1, 0]),
        prior_plan_index=torch.tensor([0, 1]),
        revised_plan_index=torch.tensor([1, 1]),
        prior_actions=torch.zeros(2, 3, 8),
        revised_actions=torch.stack([torch.ones(3, 8), torch.zeros(3, 8)]),
        prior_residual_mu_by_code=residual_mu,
        prior_residual_logvar_by_code=residual_logvar,
        reply_residual=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )

    assert torch.allclose(diagnostics["code_surprise"][0], torch.tensor(-math.log(0.2)))
    assert diagnostics["residual_mahalanobis_sq"].tolist() == [1.0, 0.0]
    assert torch.allclose(
        diagnostics["plan_surprise"],
        diagnostics["code_surprise"] + diagnostics["residual_surprise"],
    )
    assert diagnostics["replanned"].tolist() == [True, False]
    assert diagnostics["action_change_l2"][0] > 0
    assert diagnostics["action_change_l2"][1] == 0
