from __future__ import annotations

import torch
from types import SimpleNamespace

from before_we_act.contracts import ConsequencePrediction, PlannerDecision, TeamBeliefState
from before_we_act.planner.base import CANDIDATE_KINDS, WorldUtility


def test_r14_registry_is_exactly_four_routes():
    assert CANDIDATE_KINDS == {
        "p0": "world_in_world_revision",
        "p1": "dinowm_cem",
        "p2": "tdmpc2_mpc",
        "p3": "mbrl_lib_cem",
    }


def test_planner_decision_contract():
    decision = PlannerDecision(
        actions=torch.zeros(1, 4, 100, 8),
        selected_source="w12_base_index_0",
        fallback=True,
        utility_gain=0.0,
        latency_ms=1.0,
        reason="test",
    )
    assert decision.validate() is decision


def test_planner_decision_rejects_nonfinite():
    actions = torch.zeros(1, 4, 100, 8)
    actions[0, 0, 0, 0] = float("nan")
    decision = PlannerDecision(actions, "x", False, 0.0, 1.0, "test")
    try:
        decision.validate()
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("non-finite planner action was accepted")


def test_world_utility_accepts_proposal_agent_horizon_action_shape():
    class FakeWorld:
        def __call__(self, **values):
            proposals = values["candidate_actions"].shape[1]
            return ConsequencePrediction(
                latent_by_horizon=torch.zeros(1, proposals, 3, 1, 1),
                qpos_delta_by_horizon=torch.zeros(1, proposals, 3, 4, 9),
                progress_by_horizon=torch.zeros(1, proposals, 3),
                failure_logits_by_horizon=torch.zeros(1, proposals, 3),
                uncertainty_by_horizon=torch.zeros(1, proposals, 3),
                valid_mask=torch.ones(1, proposals, dtype=torch.bool),
            ).validate()

    config = SimpleNamespace(planner={
        "progress_weight": 1.0,
        "failure_weight": 0.2,
        "uncertainty_weight": 0.05,
        "trust_region_weight": 1.0,
    })
    belief = TeamBeliefState(
        tokens=torch.zeros(1, 16, 96),
        agent_tokens=torch.zeros(1, 4, 96),
        consensus_token=torch.zeros(1, 96),
        uncertainty=torch.zeros(1, 1),
        agent_mask=torch.ones(1, 4, dtype=torch.bool),
    ).validate()
    base = torch.zeros(4, 100, 8)
    candidates = torch.stack((base, base + 0.01))
    scores = WorldUtility(FakeWorld(), config)(belief, candidates, base)
    assert scores.shape == (2,)
    assert scores[0] > scores[1]
