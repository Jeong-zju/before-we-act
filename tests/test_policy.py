"""Decentralized policy tests."""

from __future__ import annotations

import inspect
import math
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from models.communication import CommunicationConfig, VPICommunicationTrigger
from models.free_energy import FreeEnergyConfig, FreeEnergyEvaluator
from models.plan_tokenizer import PlanCodeSupport
from policies.decentralized import (
    DecentralizedPairCoordinator,
    DecentralizedPolicyConfig,
    LocalAgentPlanner,
    LocalPlannerInput,
    PlanMessage,
    SelectivePlanRouter,
)


class FakeTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(
            horizon=2,
            action_dim=4,
            latent_dim=2,
            codebook_size=5,
        )
        self.decoded_codes: list[torch.Tensor] = []

    @torch.no_grad()
    def decode_plan_latent(self, code_indices, residual):
        self.decoded_codes.append(code_indices.detach().cpu().clone())
        value = code_indices.float().view(-1, 1, 1) / 4.0
        actions = value.expand(-1, self.cfg.horizon, self.cfg.action_dim).clone()
        # Keep a small learned-residual contribution so the controller cannot
        # accidentally replace every residual with a zero vector unnoticed.
        actions[..., 0] += 0.01 * residual[:, :1]
        return {"recon_actions": actions}


class FakeIntention(nn.Module):
    def __init__(self, residual_variance: float = 0.25):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(
            slots_per_agent=4,
            slot_dim=3,
            plan_codebook_size=5,
            plan_latent_dim=2,
            message_metadata_dim=4,
        )
        self.residual_variance = float(residual_variance)
        self.received_metadata: list[torch.Tensor] = []

    def forward(
        self,
        ego_slots,
        ego_plan_code,
        ego_plan_residual,
        agent_id,
        received_message_metadata,
    ):
        del ego_plan_code, ego_plan_residual, agent_id
        self.received_metadata.append(received_message_metadata.detach().cpu().clone())
        B = ego_slots.shape[0]
        # Most unmasked probability is intentionally on unsupported code 0.
        # Active codes 1 and 4 become uniform only after the deployable mask.
        probabilities = ego_slots.new_tensor([0.96, 0.02, 0.0, 0.0, 0.02]).expand(B, -1)
        mu = torch.zeros(B, 5, 2, device=ego_slots.device, dtype=ego_slots.dtype)
        mu[:, 1] = ego_slots.new_tensor([0.2, 0.3])
        mu[:, 4] = ego_slots.new_tensor([0.8, 0.9])
        logvar = torch.full_like(mu, math.log(self.residual_variance))
        uncertainty = torch.full(
            (B,),
            self.residual_variance,
            device=ego_slots.device,
            dtype=ego_slots.dtype,
        )
        return {
            "code_probabilities": probabilities,
            "residual_mu_by_code": mu,
            "residual_logvar_by_code": logvar,
            "uncertainty": uncertainty,
        }


class MatchingPlanWAM(nn.Module):
    """Costs 0 for matching codes and 10 for mismatching codes."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(
            horizon=2,
            slots_per_agent=4,
            slot_dim=3,
            plan_codebook_size=5,
            plan_latent_dim=2,
            action_dim_per_agent=4,
        )
        self.calls: list[dict[str, torch.Tensor]] = []

    @torch.no_grad()
    def rollout(
        self,
        ego_slots,
        plan_codes,
        plan_residuals,
        teammate_hypothesis_weight,
    ):
        self.calls.append(
            {
                "ego_slots": ego_slots.detach().cpu().clone(),
                "plan_codes": plan_codes.detach().cpu().clone(),
                "plan_residuals": plan_residuals.detach().cpu().clone(),
                "weights": teammate_hypothesis_weight.detach().cpu().clone(),
            }
        )
        N = ego_slots.shape[0]
        mismatch_cost = (plan_codes[:, 0] != plan_codes[:, 1]).float() * 10.0
        progress = (10.0 - mismatch_cost).view(N, 1).expand(N, self.cfg.horizon)
        return {
            "pred_ego_slots": ego_slots[:, None].expand(-1, self.cfg.horizon, -1, -1),
            "pred_actions": torch.zeros(N, self.cfg.horizon, 8, device=ego_slots.device),
            "pred_progress": progress,
            "pred_force": torch.zeros(N, self.cfg.horizon, device=ego_slots.device),
            "pred_contact_logits": torch.full(
                (N, self.cfg.horizon), -10.0, device=ego_slots.device
            ),
        }


def make_support() -> PlanCodeSupport:
    counts = torch.tensor([0, 50, 0, 0, 50])
    return PlanCodeSupport(
        codebook_size=5,
        min_count=1,
        counts=counts,
        probabilities=counts.float() / counts.sum(),
        residual_mean=torch.tensor(
            [[0.1, 0.1], [0.2, 0.3], [0.4, 0.4], [0.6, 0.6], [0.8, 0.9]]
        ),
        residual_std=torch.full((5, 2), 0.01),
    )


def make_components(residual_variance: float = 0.25):
    tokenizer = FakeTokenizer()
    intention = FakeIntention(residual_variance=residual_variance)
    wam = MatchingPlanWAM()
    support = make_support()
    free_energy = FreeEnergyEvaluator(
        FreeEnergyConfig(
            goal_y=10.0,
            alpha_goal=1.0,
            alpha_safety=0.0,
            alpha_collab=0.0,
            alpha_unc=0.0,
            alpha_ctrl=0.0,
            terminal_goal_weight=1.0,
            mean_goal_weight=0.0,
        )
    )
    communication = VPICommunicationTrigger(
        CommunicationConfig(
            codebook_size=5,
            residual_dim=2,
            lambda_bits=0.0,
            lambda_delay=0.0,
            delay_steps=1.0,
        )
    )
    return tokenizer, wam, intention, support, free_energy, communication


def local_inputs():
    return (
        LocalPlannerInput(torch.zeros(4, 3), torch.zeros(4)),
        LocalPlannerInput(torch.ones(4, 3), torch.zeros(4)),
    )


def make_pair(*, mode="selective", cooldown=3, valid=1, residual_variance=0.25):
    tokenizer, wam, intention, support, free_energy, communication = make_components(
        residual_variance
    )
    config = DecentralizedPolicyConfig(
        num_candidates=2,
        num_teammate_hypotheses=2,
        residual_sigma_points=3,
        communication_mode=mode,
        cooldown_steps=cooldown,
        plan_valid_steps=valid,
        seed=4,
    )
    pair = DecentralizedPairCoordinator.from_shared_components(
        tokenizer=tokenizer,
        wam=wam,
        intention=intention,
        support=support,
        free_energy=free_energy,
        communication=communication,
        config=config,
    )
    return pair, (tokenizer, wam, intention, support)


def test_information_firewall_packet_schema_and_prepare_signature():
    assert {field.name for field in fields(LocalPlannerInput)} == {
        "ego_slots",
        "message_metadata",
    }
    packet_fields = {field.name for field in fields(PlanMessage)}
    assert packet_fields == {
        "sender_id",
        "sequence",
        "start_step",
        "valid_until_step",
        "code",
        "residual",
        "confidence",
    }
    forbidden = {"state", "pose", "observation", "action", "action_chunk", "slots"}
    assert packet_fields.isdisjoint(forbidden)
    prepare_parameters = set(inspect.signature(LocalAgentPlanner.prepare).parameters)
    assert prepare_parameters == {"self", "local_input"}


def test_pair_shares_models_but_keeps_controller_state_separate():
    pair, components = make_pair(mode="no_comm")
    tokenizer, wam, intention, support = components
    planner0, planner1 = pair.planners
    for planner in pair.planners:
        assert planner.tokenizer is tokenizer
        assert planner.wam is wam
        assert planner.intention is intention
        assert planner.support is support
    assert planner0._generator is not planner1._generator

    decision = pair.step(local_inputs())
    assert decision.joint_action.shape == (8,)
    assert decision.routed_messages == 0
    assert planner0.step_index == planner1.step_index == 1
    planner0.reset()
    assert planner0.step_index == 0
    assert planner1.step_index == 1


def test_candidates_and_teammate_hypotheses_are_masked_to_artifact_support():
    pair, (_, wam, _, support) = make_pair(mode="no_comm")
    decision = pair.step(local_inputs())
    active = set(support.active_codes.tolist())
    assert active == {1, 4}
    assert {agent.plan_code for agent in decision.agents}.issubset(active)
    assert wam.calls
    for call in wam.calls:
        assert set(call["plan_codes"].reshape(-1).tolist()).issubset(active)
        # Posterior probabilities are aggregated outside the dynamics model.
        assert torch.all(call["weights"] == 1)
    for agent in decision.agents:
        assert set(agent.diagnostics["candidate_codes"]) == active
        assert set(agent.diagnostics["hypothesis_codes"]).issubset(active)
        assert torch.linalg.vector_norm(agent.plan_residual).item() > 0


def test_selective_reply_rerolls_and_repairs_action_before_execution():
    pair, _ = make_pair(mode="selective", cooldown=3, valid=1)
    decision = pair.step(local_inputs())

    assert decision.routed_messages == 2
    assert all(agent.communicated for agent in decision.agents)
    for agent in decision.agents:
        diagnostics = agent.diagnostics
        assert diagnostics["request_sent"] is True
        assert diagnostics["reply_received"] is True
        assert diagnostics["VPI"] > 0
        assert diagnostics["G_before"] == 10.0
        assert diagnostics["G_after"] == 0.0
        assert diagnostics["replanned"] is True
        assert diagnostics["action_change_l2"] > 0
        assert diagnostics["plan_surprise"] > 0
        assert diagnostics["actual_round_trip_bits"] == (
            diagnostics["actual_request_bits"] + diagnostics["actual_reply_bits"]
        )
        assert diagnostics["actual_request_bits"] > 0
        assert diagnostics["actual_reply_bits"] > 0
        assert diagnostics["actual_delay_steps"] == 1.0


def test_router_never_reads_peer_plan_for_a_non_requesting_agent():
    router = SelectivePlanRouter()
    calls = {0: 0, 1: 0}

    def supplier(agent_id):
        def make_message():
            calls[agent_id] += 1
            return PlanMessage(
                sender_id=agent_id,
                sequence=0,
                start_step=0,
                valid_until_step=0,
                code=1,
                residual=torch.tensor([0.2, 0.3]),
                confidence=0.5,
            )

        return make_message

    deliveries = router.route(
        requests={0: False, 1: True},
        reply_suppliers={0: supplier(0), 1: supplier(1)},
    )
    assert set(deliveries) == {1}
    assert calls == {0: 1, 1: 0}


def test_valid_message_cache_avoids_repeat_request_then_expires_under_cooldown():
    pair, (_, _, intention, _) = make_pair(mode="always_reply", cooldown=3, valid=1)

    first = pair.step(local_inputs())
    second = pair.step(local_inputs())
    third = pair.step(local_inputs())
    fourth = pair.step(local_inputs())

    assert first.routed_messages == 2
    assert second.routed_messages == 0
    assert all(agent.diagnostics["cached_message_used"] for agent in second.agents)
    assert all(not agent.diagnostics["request_sent"] for agent in second.agents)
    assert third.routed_messages == 0
    assert all(not agent.diagnostics["cached_message_used"] for agent in third.agents)
    assert all(agent.diagnostics["cooldown_remaining"] == 1 for agent in third.agents)
    assert fourth.routed_messages == 2

    # Calls are interleaved agent0/agent1 for each step.  Cache metadata records
    # availability and age, while expired content is not used as a hypothesis.
    second_step_metadata = intention.received_metadata[2:4]
    third_step_metadata = intention.received_metadata[4:6]
    assert all(metadata[0, 0].item() == 1.0 for metadata in second_step_metadata)
    assert all(metadata[0, 1].item() == 1.0 for metadata in second_step_metadata)
    assert all(metadata[0, 2].item() == 0.5 for metadata in second_step_metadata)
    assert all(metadata[0, 0].item() == 0.0 for metadata in third_step_metadata)
    assert all(metadata[0, 1].item() == 2.0 for metadata in third_step_metadata)


def test_residual_predictive_variance_expands_sigma_point_rollout():
    low_pair, (_, low_wam, _, _) = make_pair(mode="no_comm", residual_variance=0.01)
    high_pair, (_, high_wam, _, _) = make_pair(mode="no_comm", residual_variance=4.0)
    low = low_pair.step(local_inputs())
    high = high_pair.step(local_inputs())

    low_peer_residuals = low_wam.calls[0]["plan_residuals"][:, 1]
    high_peer_residuals = high_wam.calls[0]["plan_residuals"][:, 1]
    assert high_peer_residuals.std() > low_peer_residuals.std()
    assert high.agents[0].diagnostics["hypothesis_codes"] == [1, 1, 1, 4, 4, 4]
    # The actual intention model reports predictive variance in uncertainty;
    # the controller retains it in the local FE computation.
    assert (
        high.agents[0].diagnostics["intention_uncertainty"]
        > low.agents[0].diagnostics["intention_uncertainty"]
    )
    assert high.agents[0].diagnostics["G_no"] == low.agents[0].diagnostics["G_no"]


def test_no_comm_mode_never_transmits_even_when_vpi_is_positive():
    pair, _ = make_pair(mode="no_comm")
    decision = pair.step(local_inputs())
    assert decision.routed_messages == 0
    assert all(agent.diagnostics["VPI"] > 0 for agent in decision.agents)
    assert all(agent.diagnostics["request_sent"] is False for agent in decision.agents)
    assert all(agent.diagnostics["actual_round_trip_bits"] == 0 for agent in decision.agents)


def test_periodic_and_random_communication_baselines_do_not_use_reply_content_to_trigger():
    periodic, _ = make_pair(mode="periodic", cooldown=0, valid=0)
    periodic.planners[0].config = DecentralizedPolicyConfig(
        num_candidates=2,
        num_teammate_hypotheses=2,
        residual_sigma_points=3,
        communication_mode="periodic",
        cooldown_steps=0,
        plan_valid_steps=0,
        periodic_interval=2,
        seed=4,
    )
    periodic.planners[1].config = periodic.planners[0].config
    assert periodic.step(local_inputs()).routed_messages == 2
    assert periodic.step(local_inputs()).routed_messages == 0

    random_pair, _ = make_pair(mode="random", cooldown=0, valid=0)
    random_config = DecentralizedPolicyConfig(
        num_candidates=2,
        num_teammate_hypotheses=2,
        residual_sigma_points=3,
        communication_mode="random",
        cooldown_steps=0,
        plan_valid_steps=0,
        random_request_probability=0.0,
        seed=4,
    )
    for planner in random_pair.planners:
        planner.config = random_config
    assert random_pair.step(local_inputs()).routed_messages == 0


def test_periodic_quota_scheduler_matches_arbitrary_rate_and_resets_deterministically():
    pair, _ = make_pair(mode="periodic", cooldown=0, valid=0)
    config = DecentralizedPolicyConfig(
        num_candidates=2,
        num_teammate_hypotheses=2,
        residual_sigma_points=3,
        communication_mode="periodic",
        cooldown_steps=0,
        plan_valid_steps=0,
        periodic_request_rate=0.37,
        seed=4,
    )
    planner = pair.planners[0]
    planner.config = config
    trigger = {"trigger": torch.tensor([False])}

    def request_sequence() -> list[bool]:
        return [
            planner._request_decision(trigger, valid_message=None)[0]
            for _ in range(101)
        ]

    planner.reset(seed=17)
    first = request_sequence()
    planner.reset(seed=17)
    repeated = request_sequence()

    assert first == repeated
    assert sum(first) == math.floor(101 * 0.37)
    request_indices = [index for index, request in enumerate(first) if request]
    assert set(b - a for a, b in zip(request_indices, request_indices[1:])) <= {2, 3}


def test_periodic_quota_scheduler_handles_zero_one_and_validates_rate():
    pair, _ = make_pair(mode="periodic", cooldown=0, valid=0)
    planner = pair.planners[0]
    trigger = {"trigger": torch.tensor([False])}

    planner.config = DecentralizedPolicyConfig(
        communication_mode="periodic",
        periodic_request_rate=0.0,
    )
    assert not any(
        planner._request_decision(trigger, valid_message=None)[0] for _ in range(20)
    )

    planner.config = DecentralizedPolicyConfig(
        communication_mode="periodic",
        periodic_request_rate=1.0,
    )
    planner.reset(seed=9)
    assert all(
        planner._request_decision(trigger, valid_message=None)[0] for _ in range(20)
    )

    for invalid_rate in (-0.01, 1.01):
        with pytest.raises(ValueError, match="periodic_request_rate"):
            DecentralizedPolicyConfig(periodic_request_rate=invalid_rate)
