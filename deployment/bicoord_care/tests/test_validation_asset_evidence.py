from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from deployment.bicoord_care import evaluate_b0h, evaluate_bcore, paired_evaluate
from deployment.bicoord_care.bcore_data import (
    BICOORD_CARE_MEMORY_TOKENS,
    BICOORD_CARE_MEMORY_WIDTH,
)
from deployment.bicoord_care.config import ACTION_DIM, ACTION_HORIZON


OVERLAY = {
    "task": "sweep_block",
    "applied": True,
    "overlay": "/run/asset_contract/overlay/082_smallshovel/model_data3.json",
    "receipt": "/run/asset_contract/asset_contract.json",
    "receipt_sha256": "a" * 64,
    "contact_points_pose_sha256": "b" * 64,
    "actors": {
        "shovel": {
            "after_sha256": "b" * 64,
            "contact_points_pose_count": 1,
            "scale_preserved": True,
            "changed_fields": ["contact_points_pose"],
        }
    },
    "copied_fields": ["contact_points_pose"],
    "derived_fields": ["contact_points_pose"],
    "source_fields": ["contact_pose", "trans_matrix"],
    "legacy_conversion": True,
    "task_source_modified": False,
}


def _native_action() -> np.ndarray:
    value = np.zeros(ACTION_DIM, dtype=np.float32)
    value[-1] = 0.5
    return value


class _OneStepEnv:
    def __init__(self) -> None:
        self._bicoord_asset_overlay = OVERLAY
        self.eval_success = False
        self.stage_eval_score = 0.0
        self.closed = False

    def get_obs(self) -> dict[str, object]:
        return {"observation": True}

    def take_action(self, _action: object) -> None:
        self.eval_success = True
        self.stage_eval_score = 1.0

    def check_success(self) -> bool:
        return self.eval_success

    def close_env(self) -> None:
        self.closed = True


def test_b0h_episode_receipt_carries_detached_runtime_overlay(
    tmp_path: Path, monkeypatch
) -> None:
    env = _OneStepEnv()
    monkeypatch.setattr(evaluate_b0h, "_bench_env", lambda *_args: env)

    class Runtime:
        last_prediction_diagnostics: dict[str, int] = {}

        def reset(self) -> None:
            return None

        def act(self, _observation: object, _task: str) -> dict[int, np.ndarray]:
            return {0: _native_action()[None], 1: _native_action()[None]}

    row = evaluate_b0h._run_episode(
        Runtime(), tmp_path, "sweep_block", 100_000, 2, tmp_path / "b0h.jsonl"
    )

    assert row["asset_overlay"] == OVERLAY
    assert row["asset_overlay"] is not OVERLAY
    assert env.closed is True
    OVERLAY["actors"]["shovel"]["after_sha256"] = "c" * 64
    assert row["asset_overlay"]["actors"]["shovel"]["after_sha256"] == "b" * 64
    OVERLAY["actors"]["shovel"]["after_sha256"] = "b" * 64


def test_bcore_episode_receipt_carries_detached_runtime_overlay(
    tmp_path: Path, monkeypatch
) -> None:
    env = _OneStepEnv()
    monkeypatch.setattr(evaluate_bcore, "_make_env", lambda *_args: env)

    class Runtime:
        def reset(self, _task: str) -> None:
            return None

        def act(
            self, _observation: object, _task: str, *, belief_enabled: bool
        ) -> tuple[dict[int, np.ndarray], dict[str, int]]:
            assert belief_enabled is True
            return {0: _native_action(), 1: _native_action()}, {}

    row = evaluate_bcore._run_episode(
        Runtime(), tmp_path, "sweep_block", 100_000, 2, tmp_path / "bcore.jsonl"
    )

    assert row["asset_overlay"] == OVERLAY
    assert row["asset_overlay"] is not OVERLAY
    assert env.closed is True


def test_paired_mode_episode_receipt_carries_detached_runtime_overlay(
    tmp_path: Path,
) -> None:
    env = _OneStepEnv()
    action = _native_action()
    reference = np.broadcast_to(
        action, (2, ACTION_HORIZON, ACTION_DIM)
    ).copy()
    context = SimpleNamespace(
        memory=np.zeros(
            (2, BICOORD_CARE_MEMORY_TOKENS, BICOORD_CARE_MEMORY_WIDTH),
            dtype=np.float32,
        ),
        memory_mask=np.ones(
            (2, BICOORD_CARE_MEMORY_TOKENS), dtype=bool
        ),
        reference_plan=reference,
        base_plan=reference.copy(),
        current_qpos=np.stack((action, action)),
        diagnostics={},
    )

    class Runtime:
        device = SimpleNamespace(type="cpu")

        def reset(self, _task: str) -> None:
            return None

        def act_with_context(self, *_args: Any, **_kwargs: Any) -> Any:
            return context

        def record_executed_actions(self, _actions: np.ndarray) -> None:
            return None

    row = paired_evaluate._episode(
        env=env,
        runtime=Runtime(),
        care=SimpleNamespace(),
        calibration=SimpleNamespace(),
        task="sweep_block",
        seed=100_000,
        max_steps=2,
        selector_enabled=False,
        progress_path=tmp_path / "selector_off.jsonl",
        action_min=np.full(ACTION_DIM, -1.0, dtype=np.float32),
        action_max=np.full(ACTION_DIM, 1.0, dtype=np.float32),
        initial_observation=env.get_obs(),
    )

    assert row["asset_overlay"] == OVERLAY
    assert row["asset_overlay"] is not OVERLAY

