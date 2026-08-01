from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.static_rgb_act import StaticRGBMoEACTConfig
from models.wam_multimodal import (
    AgentFactorizedFlowWAM,
    CrossAgentWorldConditionedFlow,
    LocalActionConditionedFuturePredictor,
    LocalFuturePredictorConfig,
    WorldToFlowAdapterConfig,
)
from scripts.accept_s3_r6 import build_final_acceptance, build_pair_acceptance
from scripts.build_lpd_gate_summary import FORMAT_VERSION as GATE_FORMAT
from scripts.s3_r6_runtime import initialize, render_monitor, update_status
from scripts.serve_robofactory_m2_rollout import _validate_client
from scripts.train_s3_r6_world_action_flow import CHECKPOINT_FORMAT
from train.s3_model_registry import S3_R6_MODEL_KINDS, validate_s3_r6_candidate
from train.world_action_flow_training import (
    grouped_flow_matching_batch,
    grouped_masked_flow_mse,
)


def _round(kind: str) -> dict[str, object]:
    micro, candidate, scope, injection = S3_R6_MODEL_KINDS[kind]
    return {
        "round_id": "s3-r6",
        "micro_round": micro,
        "candidate_id": candidate,
        "model_kind": kind,
        "future_scope": scope,
        "injection": injection,
    }


def _flow_config() -> StaticRGBMoEACTConfig:
    return StaticRGBMoEACTConfig(
        state_dim=5,
        action_dim=2,
        horizon=4,
        vision_dim=6,
        d_model=12,
        encoder_layers=1,
        decoder_layers=1,
        heads=3,
        ffn_dim=24,
        latent_dim=4,
        experts=2,
        dropout=0.0,
        decoder_kind="dense",
        dense_ffn_dim=24,
    )


def _local_config() -> LocalFuturePredictorConfig:
    return LocalFuturePredictorConfig(
        max_agents=3,
        state_dim=5,
        action_dim=2,
        action_horizon=4,
        future_horizons=(1, 4),
        visual_grid_tokens=2,
        visual_latent_dim=6,
        d_model=12,
        ffn_dim=24,
        layers=1,
        heads=3,
        dropout=0.0,
    )


def _model(*, injection: bool) -> CrossAgentWorldConditionedFlow:
    return CrossAgentWorldConditionedFlow(
        AgentFactorizedFlowWAM(_flow_config()),
        LocalActionConditionedFuturePredictor(_local_config()),
        WorldToFlowAdapterConfig(
            flow_dim=12,
            state_dim=5,
            visual_dim=6,
            hidden_dim=12,
            action_dim=2,
            max_gate=0.25,
        ),
        future_scope="local",
        injection=injection,
    )


def _inputs() -> tuple[torch.Tensor, ...]:
    batch, agents = 2, 3
    return (
        torch.randn(batch, agents, 5, 6),
        torch.randn(batch, agents, 5),
        torch.randn(batch, agents, 2, 6),
        torch.randn(batch, 2, 6),
        torch.randn(batch, agents, 4, 2),
        torch.rand(batch),
        torch.tensor([[True, True, False], [True, True, True]]),
    )


def test_s3_model_kind_allowlist_is_exact_and_fail_closed() -> None:
    for kind, expected in S3_R6_MODEL_KINDS.items():
        assert validate_s3_r6_candidate(_round(kind)) == (
            expected[0],
            expected[1],
            kind,
            expected[2],
            expected[3],
        )
    invalid = _round("s3_r6l_protected_local_gated")
    invalid["model_kind"] = "s3_r6_unregistered"
    with pytest.raises(ValueError, match="unregistered"):
        validate_s3_r6_candidate(invalid)
    mismatched = _round("s3_r6j_protected_team_gated")
    mismatched["future_scope"] = "local"
    with pytest.raises(ValueError, match="allowlist"):
        validate_s3_r6_candidate(mismatched)


def test_zero_gate_is_exact_base_and_only_adapter_is_trainable() -> None:
    model = _model(injection=True)
    values = _inputs()
    observed, diagnostics = model.velocity(*values)
    raw, state, _local, _shared, actions, tau, valid = values
    expected = model.base_flow(
        raw.flatten(0, 1),
        state.flatten(0, 1),
        actions.flatten(0, 1),
        tau[:, None].expand(-1, 3).reshape(-1),
    )[0].reshape_as(observed)
    expected = expected * valid[:, :, None, None]
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert float(diagnostics["gate"]) == 0.0
    assert all(not parameter.requires_grad for parameter in model.base_flow.parameters())
    assert all(
        not parameter.requires_grad for parameter in model.future_predictor.parameters()
    )
    loss = observed.square().mean()
    loss.backward()
    assert model.adapter.gate_alpha.grad is not None
    assert float(model.adapter.gate_alpha.grad.abs()) > 0.0
    assert all(parameter.grad is None for parameter in model.base_flow.parameters())


def test_offpath_candidate_never_executes_future_predictor() -> None:
    model = _model(injection=False)

    def fail(*_args, **_kwargs):
        raise AssertionError("off-path control must not execute the future predictor")

    model._predict_futures = fail  # type: ignore[method-assign]
    velocity, diagnostics = model.velocity(*_inputs())
    assert velocity.shape == (2, 3, 4, 2)
    assert float(diagnostics["gate"]) == 0.0
    assert model.trainable_parameters() == ()


def test_grouped_flow_objective_respects_agent_and_horizon_masks() -> None:
    target = torch.randn(2, 3, 4, 2)
    inputs, velocity, tau = grouped_flow_matching_batch(
        target, generator=torch.Generator().manual_seed(7)
    )
    assert inputs.shape == target.shape
    assert velocity.shape == target.shape
    assert tau.shape == (2,)
    prediction = velocity.clone()
    prediction[0, 2] = 1000
    prediction[1, :, 3] = 1000
    loss = grouped_masked_flow_mse(
        prediction,
        velocity,
        torch.tensor([[True, True, False], [True, True, True]]),
        torch.tensor([[True, True, True, True], [True, True, True, False]]),
    )
    assert float(loss) == 0.0


def _gate(successes: tuple[int, int]) -> dict[str, object]:
    def task(count: int) -> dict[str, object]:
        episodes = [
            {"seed": 900 + index, "success": index < count}
            for index in range(4)
        ]
        return {"successes": count, "success_rate": count / 4, "episodes": episodes}

    return {
        "format_version": GATE_FORMAT,
        "mode": "gate",
        "candidate": {"policy_kind": "s3_flow"},
        "seed_protocol": {"seed_start": 900, "episodes_per_task": 4},
        "lift_barrier": task(successes[0]),
        "long_pipeline_delivery": task(successes[1]),
    }


def _checkpoint(kind: str) -> dict[str, object]:
    micro, candidate, scope, injection = S3_R6_MODEL_KINDS[kind]
    return {
        "format_version": CHECKPOINT_FORMAT,
        "method": {
            "micro_round": micro,
            "candidate_id": candidate,
            "model_kind": kind,
            "future_scope": scope,
            "injection": injection,
        },
        "structural_invariants": {
            "protected_own_elementwise_exact": True,
            "protected_parent_model_hashes_unchanged": True,
            "parent_files_unchanged": True,
            "parents_excluded_from_optimizer": True,
            "gate_zero_base_action_elementwise_exact": False,
        },
    }


def test_s3_acceptance_uses_only_per_task_no_regression_plus_structure() -> None:
    accepted = build_pair_acceptance(
        "R6L",
        _gate((1, 2)),
        _gate((1, 3)),
        _checkpoint("s3_r6l_protected_local_aux"),
        _checkpoint("s3_r6l_protected_local_gated"),
    )
    assert accepted["passed"] is True
    # A report-only gate-zero diagnostic is deliberately false above and does
    # not alter the stage rule.
    rejected = build_pair_acceptance(
        "R6L",
        _gate((2, 2)),
        _gate((1, 4)),
        _checkpoint("s3_r6l_protected_local_aux"),
        _checkpoint("s3_r6l_protected_local_gated"),
    )
    assert rejected["passed"] is False
    assert rejected["tasks"]["lift_barrier"]["passed_no_regression"] is False

    final = build_final_acceptance(accepted, {**accepted, "micro_round": "R6J"})
    assert final["passed_for_r7"] is True


def test_s3_checkpoint_is_in_closed_loop_server_allowlist() -> None:
    source = "s3_r6j_protected_gated_residual_flow_p1"
    contract = {"task_id": "lift_barrier", "future_path": False}
    client = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "task_vocabulary": ["lift_barrier"],
        "future_path": False,
        "policy": {"action_source": source},
    }
    assert _validate_client(client, contract=contract)["policy"]["action_source"] == source


def test_s3_monitor_names_program_heartbeat_progress_and_special_rule(tmp_path: Path) -> None:
    worktrees = []
    for candidate in ("R6L-P0", "R6L-P1", "R6J-P0", "R6J-P1"):
        path = tmp_path / candidate
        path.mkdir()
        worktrees.append(f"{candidate}={path}")
    root = tmp_path / "run"
    initialize(
        root,
        run_id="s3-test",
        session="permanent",
        window_prefix="s3-test",
        monitor_window="s3-test-monitor",
        base_repo=tmp_path,
        worktrees=worktrees,
    )
    update_status(
        root,
        candidate="R6L-P1",
        phase="training",
        program="train_s3_r6_world_action_flow.py",
        detail="adapter/gate-only",
        gpu_index=1,
        total_updates=10000,
        exit_code=None,
    )
    rendered = render_monitor(root)
    assert "train_s3_r6_world_action_flow.py" in rendered
    assert "HEARTBEAT" in rendered
    assert "P1>=P0" in rendered
    assert "phase1 GPU0=R6L-P0" in rendered


def test_s3_s0_preparation_covers_robofactory_with_protected_fifos() -> None:
    root = Path(__file__).resolve().parents[1]
    wrapper = (root / "scripts/prepare_s3_r6_from_s0.sh").read_text()
    prepare = (root / "scripts/prepare_s3_r6_shared.sh").read_text()
    launcher = (root / "scripts/launch_s3_r6_2gpu_tmux.sh").read_text()
    assert "prepare_s0_shared.sh" in wrapper
    assert "prepare_s2_r4_shared.sh" in wrapper
    assert 'chmod 600 "${SECRET_FIFO}"' in wrapper
    assert "HF_TOKEN_INPUT" in wrapper
    assert "S3_R6_ROBOFACTORY_ROOT" in prepare
    assert "ROBOFACTORY_SENTINEL" in launcher
    assert 'unlink "${READY_FILE}"' in launcher


def test_s3_monitor_reports_live_rollout_task_episode_step(tmp_path: Path) -> None:
    worktrees = []
    for candidate in ("R6L-P0", "R6L-P1", "R6J-P0", "R6J-P1"):
        path = tmp_path / candidate
        path.mkdir()
        worktrees.append(f"{candidate}={path}")
    root = tmp_path / "run"
    initialize(
        root,
        run_id="s3-test",
        session="permanent",
        window_prefix="s3-test",
        monitor_window="s3-test-monitor",
        base_repo=tmp_path,
        worktrees=worktrees,
    )
    rollout = (
        root
        / "candidates/r6l_p0/validation/gate_s3-test/lift_barrier"
        / "rollout_status.json"
    )
    rollout.parent.mkdir(parents=True)
    rollout.write_text(
        '{"task_id":"lift_barrier","stage":"rollout","episode_current":2,'
        '"episodes_total":20,"step":125,"max_steps":500,"successes":1}'
    )
    rendered = render_monitor(root)
    assert "task=lift_barrier episode=2/20 step=125/500 success=1" in rendered
    assert "stage=rollout" in rendered
