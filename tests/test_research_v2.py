from __future__ import annotations

import copy

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import data.decentralized_dataset as decentralized_dataset_module
from data.collect import (
    _research_v2_action_programs,
    _research_v2_plan_pairs,
    collect_one_episode,
)
from data.policies import ScriptedPolicy
from data.research_v2 import ResearchV2Dataset, save_research_v2_episode
from envs.two_robot_carry_env import CarryEnvConfig, TwoRobotCarryNarrowPassageEnv
from eval.research_v2 import (
    binary_calibration_metrics,
    data_scaling_gate,
    proposal_oracle_coverage,
    return_quantile_metrics,
)
from models.research_v2 import (
    BeliefEncoderV2,
    BeliefEncoderV2Config,
    BlockTransitionWorldModelV2,
    DirectParallelWorldModelV2,
    IntentionPosteriorV2,
    PlanDistributionV2Config,
    PlanProposalV2,
    PlanTokenizerV2,
    PlanTokenizerV2Config,
    WorldModelV2Config,
    count_deployable_parameters,
)
from models.research_v2_decision import candidate_hypothesis_risk, counterfactual_vpi
from policies.research_v2 import (
    DecentralizedPairCoordinatorV2,
    DeterministicRequestArbiterV2,
    LocalPlannerV2,
    MessageCodecV2,
    PlannerV2Config,
)
from policies.research_v2_runtime import LocalRuntimeV2
from train.research_v2_checkpoint import (
    IncompatibleResearchV2Checkpoint,
    load_research_v2_checkpoint,
)


def test_research_v2_base_parameter_budget_and_clean_forward_signatures():
    modules = {
        "plan": PlanTokenizerV2(PlanTokenizerV2Config()),
        "belief": BeliefEncoderV2(BeliefEncoderV2Config()),
        "proposal": PlanProposalV2(PlanDistributionV2Config()),
        "intention": IntentionPosteriorV2(PlanDistributionV2Config()),
        "world": BlockTransitionWorldModelV2(WorldModelV2Config()),
    }
    counts = count_deployable_parameters(modules)
    assert 55_000_000 <= counts["total"] <= 70_000_000
    direct = DirectParallelWorldModelV2(WorldModelV2Config())
    direct_count = sum(parameter.numel() for parameter in direct.parameters())
    assert abs(counts["world"] - direct_count) / direct_count <= 0.05
    for module in modules.values():
        names = getattr(module, "INPUT_NAMES", ())
        assert all("privileged" not in name and "truth" not in name for name in names)


def test_v2_belief_uses_current_local_history_only_and_fixed_roles():
    cfg = BeliefEncoderV2Config(
        history=3,
        local_dim=5,
        model_dim=16,
        num_heads=4,
        temporal_layers=1,
        role_layers=1,
        ffn_dim=32,
        dropout=0.0,
    )
    model = BeliefEncoderV2(cfg).eval()
    kwargs = {
        "local_history": torch.randn(2, 3, 5),
        "history_mask": torch.ones(2, 3, dtype=torch.bool),
        "ego_id": torch.tensor([0, 1]),
        "object_observation": torch.randn(2, 3, 3),
        "object_valid": torch.ones(2, 3, dtype=torch.bool),
        "object_age": torch.zeros(2, 3),
        "object_confidence": torch.ones(2, 3),
    }
    out = model(**kwargs)
    assert out["belief"].shape == (2, 4, 16)


def test_block_world_is_action_conditioned_and_later_blocks_cannot_change_earlier_output():
    cfg = WorldModelV2Config(
        horizon=4,
        block_length=2,
        belief_dim=16,
        model_dim=32,
        context_layers=1,
        transition_layers=1,
        heads=4,
        ffn_dim=64,
        dropout=0.0,
    )
    model = BlockTransitionWorldModelV2(cfg).eval()
    belief = torch.randn(2, 4, 16)
    own = torch.randn(2, 4, 4)
    peer = torch.randn(2, 4, 4)
    baseline = model(belief, own, peer)
    changed_late = own.clone()
    changed_late[:, 2:] += 10.0
    late = model(belief, changed_late, peer)
    torch.testing.assert_close(baseline["features"][:, :2], late["features"][:, :2])
    changed_early = own.clone()
    changed_early[:, :2] += 10.0
    early = model(belief, changed_early, peer)
    assert not torch.allclose(baseline["features"][:, :2], early["features"][:, :2])
    direct = DirectParallelWorldModelV2(cfg).eval()(belief, own, peer)
    assert set(direct) == set(baseline) - {"block_boundary_belief"}


def test_v2_risk_keeps_posterior_outside_conditional_world_model():
    quantiles = torch.tensor(
        [[[[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], [[0.0, 1.0, 2.0], [4.0, 5.0, 6.0]]]]]
    )
    constraints = torch.zeros(1, 1, 2, 2)
    actions = torch.zeros(1, 2, 4, 4)
    risk = candidate_hypothesis_risk(
        ensemble_return_quantiles=quantiles,
        ensemble_constraint_logits=constraints,
        ego_actions=actions,
    )
    first = counterfactual_vpi(risk["G"], torch.tensor([[0.5, 0.5]]))
    second = counterfactual_vpi(risk["G"], torch.tensor([[0.9, 0.1]]))
    torch.testing.assert_close(risk["G"], risk["G"].clone())
    assert not torch.equal(first["G_no"], second["G_no"])


def _tiny_planner(agent_id: int, state: dict | None = None) -> LocalPlannerV2:
    tokenizer = PlanTokenizerV2(
        PlanTokenizerV2Config(horizon=4, codebook_size=8, residual_dim=16, hidden_dim=32)
    )
    distribution_cfg = PlanDistributionV2Config(
        belief_dim=16,
        codebook_size=8,
        residual_dim=16,
        model_dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
    )
    proposal = PlanProposalV2(distribution_cfg)
    intention = IntentionPosteriorV2(distribution_cfg)
    world = BlockTransitionWorldModelV2(
        WorldModelV2Config(
            horizon=4,
            block_length=2,
            belief_dim=16,
            model_dim=32,
            context_layers=1,
            transition_layers=1,
            heads=4,
            ffn_dim=64,
            dropout=0.0,
        )
    )
    modules = {"tokenizer": tokenizer, "proposal": proposal, "intention": intention, "world": world}
    if state is not None:
        for name, module in modules.items():
            module.load_state_dict(state[name])
    else:
        state = {name: copy.deepcopy(module.state_dict()) for name, module in modules.items()}
    planner = LocalPlannerV2(
        agent_id,
        tokenizer=tokenizer,
        proposal=proposal,
        intention=intention,
        world_ensemble=[world],
        active_code_mask=torch.ones(8, dtype=torch.bool),
        residual_prior_by_code=torch.zeros(8, 16),
        action_mean=torch.zeros(4),
        action_std=torch.ones(4),
        artifact_hash="same-artifact",
        codec=MessageCodecV2(payload_dim=16),
        config=PlannerV2Config(
            num_candidates=2,
            num_hypotheses=2,
            communication_cost=-1.0,
            cooldown_steps=0,
        ),
    )
    planner.test_state = state
    return planner


def test_simultaneous_requests_are_content_blind_and_commitment_consistent():
    planner0 = _tiny_planner(0)
    planner1 = _tiny_planner(1, planner0.test_state)
    assert planner0.world_ensemble[0] is not planner1.world_ensemble[0]
    pair = DecentralizedPairCoordinatorV2(planner0, planner1)
    decision = pair.step((torch.randn(4, 16), torch.randn(4, 16)))
    assert decision.routed_messages == 1
    assert decision.requester == 0  # even public step
    responder = decision.agents[1]
    assert responder.locked_as_responder is True
    assert responder.reply_received is False
    assert decision.agents[0].reply_received is True
    assert DeterministicRequestArbiterV2.requester({0: True, 1: True}, 0, 1) == 1
    cached = planner0.prepare(torch.randn(4, 16))
    assert cached.request is True
    assert planner0._pending is not None
    assert (
        planner0._pending.posterior_diagnostics["intention_conditioning"]
        == "proposal_marginalized_over_all_candidates"
    )
    planner0.finalize()


def test_research_v2_matched_branches_are_ego_first_and_current_target_is_t(
    tmp_path, monkeypatch
):
    env = TwoRobotCarryNarrowPassageEnv(
        CarryEnvConfig(scenario="nominal", episode_len=110, seed=4)
    )
    episode, _, spec = collect_one_episode(
        env,
        ScriptedPolicy(seed=4),
        4,
        collect_matched_branches=True,
    )
    path = tmp_path / "episode_000000.hdf5"
    episode.metadata.update({"split": "train", "episode_index": 0})
    save_research_v2_episode(path, episode, spec, episode.research_v2_branch_groups)
    dataset = ResearchV2Dataset(tmp_path, history=8, horizon=16)
    indices = [
        index
        for index, sample in enumerate(dataset.index)
        if sample.decision_t == episode.research_v2_branch_groups[0].decision_t
    ]
    ego0 = dataset[indices[0]]
    ego1 = dataset[indices[1]]
    with h5py.File(path, "r") as file:
        group = file["research_v2/branches/group_0000"]
        canonical = torch.from_numpy(np.asarray(group["actions"], dtype=np.float32))
        current_object = torch.from_numpy(
            np.asarray(
                file["privileged/observations/object_pose_ego"][
                    episode.research_v2_branch_groups[0].decision_t, 0
                ],
                dtype=np.float32,
            )
        )
    torch.testing.assert_close(ego0["branch_matched_action"], canonical)
    torch.testing.assert_close(ego1["branch_matched_action"], canonical[:, [1, 0]])
    torch.testing.assert_close(ego0["target_current_object_pose"], current_object)
    assert "branch_group_id" not in ResearchV2Dataset.INPUT_KEYS

    selected_keys = (
        "local_history",
        "history_mask",
        "ego_id",
        "object_observation_history",
        "target_current_object_pose",
        "branch_group_id",
        "branch_matched_action",
        "branch_valid_mask",
        "branch_target_reward",
    )
    selected = dataset.get_selected(indices[0], selected_keys)
    assert set(selected) == set(selected_keys)
    for key in selected_keys:
        torch.testing.assert_close(selected[key], ego0[key], rtol=0, atol=0)

    projected = dataset.project(selected_keys)
    projected_batch = next(iter(DataLoader(projected, batch_size=2)))
    assert set(projected_batch) == set(selected_keys)
    assert projected.index is dataset.index

    def unexpected_read(*_args, **_kwargs):
        raise AssertionError("plan projection must not read local future/matched branches")

    monkeypatch.setattr(
        decentralized_dataset_module,
        "_read_selected_local_observations",
        unexpected_read,
    )
    monkeypatch.setattr(dataset, "_read_selected_matched_branches", unexpected_read)
    plan_only = dataset.get_selected(
        indices[0], ("ego_future_action", "target_maneuver")
    )
    assert set(plan_only) == {"ego_future_action", "target_maneuver"}
    torch.testing.assert_close(plan_only["ego_future_action"], ego0["ego_future_action"])
    torch.testing.assert_close(plan_only["target_maneuver"], ego0["target_maneuver"])


def test_research_v2_branch_program_is_deterministic_and_covers_all_action_dimensions():
    env = TwoRobotCarryNarrowPassageEnv(
        CarryEnvConfig(scenario="nominal", episode_len=110, seed=31)
    )
    env.reset(seed=31)
    pairs = _research_v2_plan_pairs(env, decision_t=32, group_id=0)
    first_delta, first_grip = _research_v2_action_programs(
        env, pairs, decision_t=32, group_id=0, horizon=16
    )
    second_delta, second_grip = _research_v2_action_programs(
        env, pairs.copy(), decision_t=32, group_id=0, horizon=16
    )
    np.testing.assert_allclose(first_delta, second_delta)
    np.testing.assert_allclose(first_grip, second_grip, equal_nan=True)
    assert first_delta.shape == (8, 16, 2, 4)
    assert first_grip.shape == (8, 16, 2)
    np.testing.assert_allclose(first_delta[4], 0.0)  # scripted control candidate
    assert np.ptp(first_delta[..., 0]) > 0.5  # lateral
    assert np.ptp(first_delta[..., 1]) > 0.2  # longitudinal
    assert np.ptp(first_delta[..., 2]) > 0.5  # yaw
    assert np.any(np.isfinite(first_grip)) and np.nanmin(first_grip) == 0.0


def test_v2_checkpoint_loader_rejects_frozen_v1_artifact():
    with pytest.raises(IncompatibleResearchV2Checkpoint, match="V1/legacy"):
        load_research_v2_checkpoint("checkpoints/private_gates_v1/plan.pt")


def test_local_runtime_api_has_no_joint_or_privileged_observation_argument():
    import inspect

    parameters = inspect.signature(LocalRuntimeV2.observe).parameters
    assert set(parameters) == {"self", "packet"}


def test_research_v2_calibration_and_scaling_gates_are_explicit():
    quantiles = return_quantile_metrics(
        np.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]), np.asarray([1.0, 2.0])
    )
    assert quantiles["quantile_crossing_rate"] == 0.0
    binary = binary_calibration_metrics(np.asarray([0.1, 0.9]), np.asarray([0.0, 1.0]))
    assert binary["brier"] == pytest.approx(0.01)
    assert proposal_oracle_coverage(np.asarray([[1, 2], [3, 4]]), np.asarray([2, 0])) == 0.5
    gate = data_scaling_gate(
        d1_branch_regret=1.0,
        d2_branch_regret=0.96,
        d1_success_rate=0.8,
        d2_success_rate=0.8,
        d1_constraint_ece=0.05,
        d2_constraint_ece=0.06,
    )
    assert gate["passed"] is True
    assert MessageCodecV2(payload_dim=0).reply_bits == 78
    assert MessageCodecV2(payload_dim=16).reply_bits == 206


def test_independently_loaded_weight_copies_are_output_equivalent():
    first = _tiny_planner(0)
    second = _tiny_planner(0, first.test_state)
    belief = torch.randn(4, 16)
    prepared_first = first.prepare(belief)
    prepared_second = second.prepare(belief.clone())
    assert prepared_first.provisional_message.code == prepared_second.provisional_message.code
    torch.testing.assert_close(
        prepared_first.provisional_message.residual_payload,
        prepared_second.provisional_message.residual_payload,
    )
    assert prepared_first.vpi == pytest.approx(prepared_second.vpi)
