from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from deployment.bicoord_care import branch_collection as module


class _Env:
    eval_success = False
    stage_eval_score = 0.0


def test_restore_runtime_and_env_prefers_fresh_seed_rebuild_over_physx_unpack(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def rebuild() -> None:
        calls.append("rebuild")

    env = SimpleNamespace(
        _bicoord_rebuild_at_anchor=rebuild,
        get_obs=lambda: {"anchor": True},
    )

    def forbidden_restore(*_args, **_kwargs):
        raise AssertionError("opaque PhysX restore must not run for rebuilt branches")

    monkeypatch.setattr(module, "restore_state", forbidden_restore)

    class Runtime:
        def restore_state(self, _state) -> None:
            calls.append("runtime")

    observed = module._restore_runtime_and_env(
        env,
        Runtime(),
        {"unused": True},
        {"runtime": True},
        branch_seed=None,
    )

    assert calls == ["rebuild", "runtime"]
    assert observed == {"anchor": True}


def test_step_metrics_active_uses_command_against_pre_step_qpos(monkeypatch) -> None:
    previous = np.zeros((2, 7), dtype=np.float32)
    # The post-step qpos is deliberately below the active threshold.  The
    # command itself is above it, matching the upstream CARE metric contract.
    observation_qpos = np.full((2, 7), 0.001, dtype=np.float32)
    command = np.zeros((2, 7), dtype=np.float32)
    command[0, 0] = 0.025

    monkeypatch.setattr(module, "_qpos", lambda _observation: observation_qpos)
    monkeypatch.setattr(module, "_success_progress", lambda _env: (False, 0.0))
    monkeypatch.setattr(
        module,
        "_physical_safety",
        lambda _env, _baseline: {
            "hard_safety_violation": False,
            "robot_collision": False,
        },
    )

    row = module._step_metrics(
        _Env(),
        object(),
        previous,
        command,
        {},
        0,
    )

    assert row["active"] == [True, False]
    assert row["all_joint_changes_below_0_02"] is False
    # The diagnostic qpos remains the simulator observation; only the active
    # utility gate is grounded in the executed command.
    assert row["qpos"] == observation_qpos.tolist()


def test_step_metrics_rejects_invalid_action_shape(monkeypatch) -> None:
    previous = np.zeros((2, 7), dtype=np.float32)
    monkeypatch.setattr(module, "_qpos", lambda _observation: previous)
    monkeypatch.setattr(module, "_success_progress", lambda _env: (False, 0.0))
    monkeypatch.setattr(
        module,
        "_physical_safety",
        lambda _env, _baseline: {
            "hard_safety_violation": False,
            "robot_collision": False,
        },
    )

    try:
        module._step_metrics(_Env(), object(), previous, np.zeros((7,), dtype=np.float32), {}, 0)
    except RuntimeError as error:
        assert "finite [2,7] actions" in str(error)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("invalid action shape was accepted")


def test_deadlock_run_begins_after_first_observed_tick() -> None:
    rows = [
        {
            "progress": 0.0,
            "all_joint_changes_below_0_02": True,
            "success": False,
        }
        for _ in range(8)
    ]

    # Seven comparable stagnant transitions are not an eight-tick deadlock.
    assert module._deadlock_mask(rows, start_progress=0.0) == [False] * 8

    rows.append(dict(rows[-1]))
    # With nine observations there are now eight consecutive comparable
    # transitions; only rows 1..8 belong to the deadlock run.
    assert module._deadlock_mask(rows, start_progress=0.0) == [False] + [True] * 8


def test_fidelity_diagnostic_exposes_discrete_and_continuous_differences() -> None:
    metrics = [
        {
            "qpos": [[0.0] * 7, [0.0] * 7],
            "progress": 0.0,
            "active": [False, False],
            "all_joint_changes_below_0_02": True,
        }
    ]
    replay_metrics = [dict(metrics[0], active=[True, False])]
    actions = np.zeros((1, 2, 7), dtype=np.float32)
    replay_actions = actions.copy()
    replay_actions[0, 0, 0] = 1e-4
    row = {
        "status": "VALID",
        "executed_actions": actions.tolist(),
        "metrics": metrics,
        "outcomes": {
            str(horizon): {
                "utility_main": 0.0,
                "bounded_utility_vector": [0.0] * 8,
            }
            for horizon in module.HORIZONS
        },
    }
    replay = {
        "status": "VALID",
        "executed_actions": replay_actions.tolist(),
        "metrics": replay_metrics,
        "outcomes": {
            str(horizon): {
                "utility_main": 0.0340909090909 if horizon == 16 else 0.0,
                "bounded_utility_vector": [0.0] * 8,
            }
            for horizon in module.HORIZONS
        },
    }

    diagnostic = module._fidelity_diagnostic(row, replay)

    assert diagnostic["executed_action_max_abs_error"] == pytest.approx(1e-4)
    assert diagnostic["executed_action_first_difference"] == 0
    assert diagnostic["active_label_difference_steps"] == [0]
    assert diagnostic["all_joint_changes_label_difference_steps"] == []
    assert diagnostic["utility_by_horizon"]["16"]["utility_abs_error"] > 0.03


def test_fidelity_summary_rejects_hidden_physical_drift_even_when_utility_matches() -> None:
    metrics = [
        {
            "qpos": [[0.0] * 7, [0.0] * 7],
            "progress": 0.0,
            "active": [False, False],
            "all_joint_changes_below_0_02": True,
        }
    ]
    row = {
        "repeat_id": 0,
        "executed_actions": [[[0.0] * 7, [0.0] * 7]],
        "metrics": metrics,
        "outcomes": {
            str(horizon): {"utility_main": 0.0, "bounded_utility_vector": [0.0] * 8}
            for horizon in module.HORIZONS
        },
    }
    replay = {
        **row,
        "executed_actions": [[[1e-5] + [0.0] * 6, [0.0] * 7]],
    }
    summary = module._fidelity_summary(row, replay)
    assert summary["utility_max_abs_error"] == 0.0
    assert summary["executed_action_max_abs_error"] == pytest.approx(1e-5)
    assert summary["passed"] is False
