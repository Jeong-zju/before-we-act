from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from eval.closed_loop import ClosedLoopEpisode, aggregate_closed_loop, gate_d_report
from models.wam import (
    NormalizationStats,
    RWMARConfig,
    RWMARWorldModel,
    RWMUEnsemble,
    RWMUEnsembleConfig,
    WAMPlanningHeadConfig,
    WAMPlanningHeads,
    WorldModelSequenceInputs,
)
from policies import (
    MPPIConfig,
    MPPIPlan,
    MPPISafetyConfig,
    RiskAwareMPPI,
    WAMMPPIActionPolicy,
)
from train.trajectory_dataset import InMemoryProprioSequenceDataset
from train.wam_mppi_checkpointing import (
    load_wam_mppi_heads_checkpoint,
    save_wam_mppi_heads_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


def test_phase3_config_freezes_formal_mppi_and_gate_d_names():
    payload = yaml.safe_load(
        (ROOT / "configs/wam/phase3_wam_mppi_v2.yaml").read_text(encoding="utf-8")
    )
    assert payload["phase"] == "phase3_wam_mppi"
    assert payload["planner"]["num_samples"] == 512
    assert payload["planner"]["particles_per_candidate"] == 5
    assert payload["planner"]["execute_steps"] == 1
    assert payload["planner"]["initial_std"] == 0.1
    assert payload["planner"]["prior_action_max_delta"] == 0.15
    assert payload["planner"]["terminal_value_weight"] == 0.0
    assert payload["safety"]["sticky_latency_fallback"] is True
    assert payload["safety"]["discard_over_budget_plans"] is True
    assert payload["evaluation"]["episodes"] == 500
    assert payload["evaluation"]["gate_d_protocol"] == "standard_noninferiority_v2"
    assert payload["evaluation"]["gate_d_success_regression_max"] == 0.01
    assert payload["evaluation"]["gate_d_mppi_execution_rate_min"] == 0.50
    assert payload["checkpoint"]["directory"].endswith("phase3_wam_mppi_v1")


def test_planning_features_heads_and_imagined_step_are_finite():
    ensemble = _ensemble(2)
    member = ensemble.members[0]
    history = _history()
    hidden, state, features = member.encode_planning_history(history)
    heads = WAMPlanningHeads(
        WAMPlanningHeadConfig(
            feature_dim=member.planning_feature_dim,
            action_dim=8,
            hidden_dim=16,
            hidden_layers=1,
        )
    )
    output = heads(features)
    action = heads.sample_action(features)
    next_hidden, next_state, _ = member.imagine_step(hidden, state, action)

    assert output.action_mean.shape == (1, 8)
    assert output.value.shape == (1, 1)
    assert bool(torch.isfinite(heads.action_nll(features, action)).all())
    assert next_hidden.shape == hidden.shape
    assert next_state.shape == state.shape


def test_planning_dataset_adds_mc_returns_and_success_filtered_prior(tmp_path: Path):
    path = tmp_path / "episode_000000.hdf5"
    _write_episode(path, rewards=np.asarray([1.0, 2.0], dtype=np.float32))
    dataset = InMemoryProprioSequenceDataset(
        paths=[path],
        history_horizon=3,
        forecast_horizon=1,
        planning_discount=0.9,
        action_prior_behavior_weights={"scripted_oracle_v1": 1.0},
        action_prior_require_success=True,
    )

    assert dataset[0]["returns_to_go"].item() == pytest.approx(2.8)
    assert dataset[1]["returns_to_go"].item() == pytest.approx(2.0)
    assert dataset[0]["action_prior_weights"].item() == 1.0
    assert dataset.planning_metadata["eligible_episodes"] == 1


def test_vectorized_risk_aware_mppi_returns_bounded_first_action():
    ensemble = _ensemble(2)
    member = ensemble.members[0]
    heads = WAMPlanningHeads(
        WAMPlanningHeadConfig(
            feature_dim=member.planning_feature_dim,
            action_dim=8,
            hidden_dim=16,
            hidden_layers=1,
        )
    )
    planner = RiskAwareMPPI(
        ensemble,
        heads,
        MPPIConfig(
            planning_horizon=2,
            num_samples=6,
            num_elites=2,
            iterations=1,
            num_policy_trajectories=2,
            particles_per_candidate=2,
            candidate_batch_size=3,
        ),
        fixed_actions={3: 1.0, 7: 1.0},
    )
    result = planner.plan(_history())

    assert result.action.shape == (8,)
    assert result.sequence.shape == (2, 8)
    assert float(result.action.abs().max()) <= 1.0
    assert result.action[3].item() == 1.0
    assert result.action[7].item() == 1.0
    assert np.isfinite(result.diagnostics["score"])
    assert result.diagnostics["prior_action_max_delta"] <= 0.15 + 1e-6


def test_policy_rejects_privileged_state_and_prior_fallback_is_bounded():
    ensemble = _ensemble(2)
    member = ensemble.members[0]
    heads = WAMPlanningHeads(
        WAMPlanningHeadConfig(
            feature_dim=member.planning_feature_dim,
            hidden_dim=16,
            hidden_layers=1,
        )
    )
    planner = RiskAwareMPPI(
        ensemble,
        heads,
        MPPIConfig(
            planning_horizon=1,
            num_samples=2,
            num_elites=1,
            iterations=1,
            num_policy_trajectories=1,
            particles_per_candidate=1,
            candidate_batch_size=2,
        ),
        fixed_actions={3: 1.0, 7: 1.0},
    )
    policy = WAMMPPIActionPolicy(
        planner, mode="action_prior", safety=MPPISafetyConfig()
    )
    observation = {"proprioception": _history().states[0, -1].numpy()}
    action = policy.act(observation)

    assert action.shape == (8,)
    assert np.isfinite(action).all()
    assert action[3] == 1.0 and action[7] == 1.0
    assert policy.last_diagnostics["privileged_state_seen"] is False
    with pytest.raises(RuntimeError, match="privileged_state"):
        policy.act({**observation, "privileged_state": {}})


def test_latency_fallback_descends_full_to_reduced_to_prior():
    ensemble = _ensemble(2)
    member = ensemble.members[0]
    heads = WAMPlanningHeads(
        WAMPlanningHeadConfig(
            feature_dim=member.planning_feature_dim,
            hidden_dim=16,
            hidden_layers=1,
        )
    )
    policy = WAMMPPIActionPolicy(
        RiskAwareMPPI(
            ensemble,
            heads,
            MPPIConfig(
                planning_horizon=1,
                num_samples=2,
                num_elites=1,
                iterations=1,
                num_policy_trajectories=1,
                particles_per_candidate=1,
                candidate_batch_size=2,
            ),
        ),
        safety=MPPISafetyConfig(latency_budget_ms=10.0),
    )

    policy._update_latency_profile(20.0, "mppi_full")
    assert policy._planner_profile == "reduced"
    policy._update_latency_profile(20.0, "mppi_reduced")
    assert policy._planner_profile == "prior"
    policy._update_latency_profile(1.0, "action_prior_latency_fallback")
    assert policy._planner_profile == "prior"


def test_over_budget_plan_is_discarded_on_the_same_step(monkeypatch: pytest.MonkeyPatch):
    ensemble = _ensemble(2)
    member = ensemble.members[0]
    heads = WAMPlanningHeads(
        WAMPlanningHeadConfig(
            feature_dim=member.planning_feature_dim,
            hidden_dim=16,
            hidden_layers=1,
        )
    )
    planner = RiskAwareMPPI(
        ensemble,
        heads,
        MPPIConfig(
            planning_horizon=1,
            num_samples=2,
            num_elites=1,
            iterations=1,
            num_policy_trajectories=1,
            particles_per_candidate=1,
            candidate_batch_size=2,
        ),
    )
    clock_values = iter((0.0, 0.020))
    policy = WAMMPPIActionPolicy(
        planner,
        safety=MPPISafetyConfig(latency_budget_ms=10.0),
        clock=lambda: next(clock_values),
    )
    plan = MPPIPlan(
        action=torch.ones(8),
        sequence=torch.ones(1, 8),
        diagnostics={
            "score": 1.0,
            "expected_return": 10.0,
            "epistemic": 0.0,
            "action_ood": 0.0,
            "failure_probability": 0.0,
            "predicted_robot_distance": 0.0,
        },
    )
    monkeypatch.setattr(planner, "plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(
        policy, "_prior_action", lambda history: np.zeros(8, dtype=np.float32)
    )

    action = policy.act({"proprioception": np.zeros(22, dtype=np.float32)})

    np.testing.assert_array_equal(action, np.zeros(8, dtype=np.float32))
    assert policy.last_diagnostics["planned_mode"] == "mppi_full"
    assert policy.last_diagnostics["executed_mode"] == "action_prior_deadline_fallback"
    assert policy.last_diagnostics["plan_executed"] is False
    assert policy.last_diagnostics["deadline_exceeded"] is True
    assert policy._planner_profile == "reduced"


def test_weighted_outcome_logits_are_prior_corrected():
    ensemble = _ensemble(2)
    member = ensemble.members[0]
    planner = RiskAwareMPPI(
        ensemble,
        WAMPlanningHeads(
            WAMPlanningHeadConfig(
                feature_dim=member.planning_feature_dim,
                hidden_dim=16,
                hidden_layers=1,
            )
        ),
        MPPIConfig(),
        outcome_positive_weights={"done": 9.0, "failure": 99.0},
    )

    probability = planner._outcome_probability(torch.tensor([np.log(9.0)]), "done")

    assert probability.item() == pytest.approx(0.5)


def test_phase3_heads_checkpoint_binds_exact_phase2_fingerprint(tmp_path: Path):
    phase2 = tmp_path / "phase2"
    (phase2 / "members").mkdir(parents=True)
    (phase2 / "schema.json").write_text("{}", encoding="utf-8")
    (phase2 / "members/member_00.safetensors").write_bytes(b"member")
    heads = WAMPlanningHeads(
        WAMPlanningHeadConfig(feature_dim=17, hidden_dim=8, hidden_layers=1)
    )
    checkpoint = tmp_path / "phase3"
    save_wam_mppi_heads_checkpoint(
        checkpoint,
        heads,
        phase2_checkpoint=phase2,
        experiment_config={"phase": "phase3_wam_mppi"},
        dataset_manifest={"partitions": {}},
        metrics={"test": {}},
        provenance={"seed": 1},
        schema_version="wam.proprio/1.0",
        normalization_sha256="abc",
    )
    loaded, metadata = load_wam_mppi_heads_checkpoint(
        checkpoint,
        phase2_checkpoint=phase2,
        expected_schema_version="wam.proprio/1.0",
        expected_normalization_sha256="abc",
    )

    for name, value in heads.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name])
    assert metadata["schema"]["forbidden_runtime_inputs"] == [
        "privileged_state",
        "braking_agent",
        "braking_time",
    ]
    (phase2 / "members/member_00.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="fingerprint"):
        load_wam_mppi_heads_checkpoint(checkpoint, phase2_checkpoint=phase2)


def test_gate_d_requires_full_500_seed_gain_and_leakage_audit():
    episodes = [_episode(success=True, mode="mppi_full") for _ in range(500)]
    mppi = aggregate_closed_loop(episodes)
    prior = aggregate_closed_loop(
        [_episode(success=index < 400, mode="action_prior") for index in range(500)]
    )
    stationary = aggregate_closed_loop(
        [_episode(success=False, mode="safe_stop") for _ in range(500)]
    )
    oracle = aggregate_closed_loop(
        [_episode(success=True, mode="scripted_oracle") for _ in range(500)]
    )
    report = gate_d_report(
        {
            "wam_mppi": mppi,
            "action_prior": prior,
            "stationary": stationary,
            "scripted_oracle": oracle,
        },
        full_evaluation=True,
    )

    assert report["passed"]
    assert report["checks"]["standard_task_success_noninferiority"][
        "absolute_improvement"
    ] == pytest.approx(0.2)


def _member_config() -> RWMARConfig:
    return RWMARConfig(
        history_horizon=3,
        train_forecast_horizon=2,
        planning_horizon=2,
        encoder_hidden_dim=16,
        gru_hidden_dim=12,
        gru_layers=1,
    )


def _stats() -> NormalizationStats:
    return NormalizationStats(
        state_mean=np.zeros(22, dtype=np.float32),
        state_std=np.ones(22, dtype=np.float32),
        action_mean=np.zeros(8, dtype=np.float32),
        action_std=np.ones(8, dtype=np.float32),
        delta_mean=np.zeros(22, dtype=np.float32),
        delta_std=np.ones(22, dtype=np.float32),
        reward_mean=np.zeros(1, dtype=np.float32),
        reward_std=np.ones(1, dtype=np.float32),
    )


def _ensemble(size: int) -> RWMUEnsemble:
    members = []
    for seed in range(size):
        torch.manual_seed(seed + 1)
        members.append(RWMARWorldModel(_member_config(), _stats()))
    return RWMUEnsemble(members, RWMUEnsembleConfig(size), _stats())


def _history() -> WorldModelSequenceInputs:
    states = torch.zeros(1, 3, 22)
    states[..., [7, 18]] = 1.0
    return WorldModelSequenceInputs(
        states=states,
        past_actions=torch.zeros(1, 2, 8),
        valid_mask=torch.ones(1, 3, dtype=torch.bool),
    )


def _write_episode(path: Path, rewards: np.ndarray) -> None:
    steps = len(rewards)
    with h5py.File(path, "w") as file:
        file.attrs.update(
            {
                "schema_profile": "wam_proprio",
                "schema_version": "wam.proprio/1.0",
                "num_steps": steps,
                "episode_index": 0,
                "seed": 0,
                "behavior_id": "scripted_oracle_v1",
                "total_reward": float(rewards.sum()),
            }
        )
        data = file.create_group("data")
        data.create_dataset("observation/state", data=np.zeros((steps, 22), np.float32))
        data.create_dataset("next_observation/state", data=np.zeros((steps, 22), np.float32))
        data.create_dataset("commanded_action", data=np.zeros((steps, 8), np.float32))
        data.create_dataset("executed_action", data=np.zeros((steps, 8), np.float32))
        data.create_dataset("reward", data=rewards)
        for name in ("done", "success", "failure"):
            values = np.zeros(steps, dtype=np.bool_)
            if name in {"done", "success"}:
                values[-1] = True
            data.create_dataset(name, data=values)
        data.create_dataset("response_progress", data=np.zeros(steps, np.float32))
        data.create_dataset("coordination_error", data=np.zeros(steps, np.float32))


def _episode(*, success: bool, mode: str) -> ClosedLoopEpisode:
    return ClosedLoopEpisode(
        policy="test",
        seed=1,
        steps=10,
        success=success,
        failure=not success,
        failure_reason="none" if success else "response_timeout",
        total_reward=20.0 if success else -10.0,
        response_delay_seconds=0.2 if success else -1.0,
        mean_coordination_error=0.1,
        gradual_brake_steps=8,
        stop_hold_steps=8 if success else 0,
        pre_brake_motion_valid=success,
        planner_latency_ms=(10.0,),
        planner_modes=(mode,),
        planner_attempted_modes=(mode,) if mode.startswith("mppi") else (),
        deadline_misses=0,
        discarded_plans=0,
        fallback_reasons=(),
        predicted_returns=(),
        actions_finite_and_bounded=True,
        privileged_state_seen=False,
    )
