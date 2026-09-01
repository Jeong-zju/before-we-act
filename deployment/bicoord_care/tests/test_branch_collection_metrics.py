from __future__ import annotations

import numpy as np

from deployment.bicoord_care import branch_collection as module


class _Env:
    eval_success = False
    stage_eval_score = 0.0


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
