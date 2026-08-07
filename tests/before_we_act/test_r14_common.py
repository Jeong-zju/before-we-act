from __future__ import annotations

import torch

from before_we_act.contracts import PlannerDecision
from before_we_act.planner.base import CANDIDATE_KINDS


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
