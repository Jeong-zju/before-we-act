from __future__ import annotations

from copy import deepcopy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from before_we_act.mars_care_branch_collector import (
    INTERVENTION_STEP_CHOICES,
    MarsSimulatorSnapshot,
    branch_action,
    intervention_action,
    restore_snapshot,
)


@pytest.mark.parametrize("duration", INTERVENTION_STEP_CHOICES)
def test_fixed_candidate_prefix_returns_to_receding_reference(duration: int) -> None:
    candidate = np.arange(100 * 8, dtype=np.float32).reshape(100, 8)
    reference = np.full(8, -17.0, dtype=np.float32)
    for step in range(20):
        actual = intervention_action(
            reference,
            candidate,
            branch_step=step,
            intervention_steps=duration,
        )
        expected = candidate[step] if step < duration else reference
        np.testing.assert_array_equal(actual, expected)
        assert not np.shares_memory(actual, candidate)


def test_duration_one_is_the_original_one_step_intervention() -> None:
    candidate = np.arange(100 * 8, dtype=np.float32).reshape(100, 8)
    first = np.full(8, 3.0, dtype=np.float32)
    later = np.full(8, 4.0, dtype=np.float32)
    np.testing.assert_array_equal(
        intervention_action(first, candidate, branch_step=0, intervention_steps=1),
        candidate[0],
    )
    np.testing.assert_array_equal(
        intervention_action(later, candidate, branch_step=1, intervention_steps=1),
        later,
    )


@pytest.mark.parametrize("duration", INTERVENTION_STEP_CHOICES)
def test_candidate_zero_stays_exact_receding_reference(duration: int) -> None:
    candidate = np.arange(100 * 8, dtype=np.float32).reshape(100, 8)
    for step in range(20):
        reference = np.full(8, -17.0 + step, dtype=np.float32)
        actual = branch_action(
            0,
            reference,
            candidate,
            branch_step=step,
            intervention_steps=duration,
        )
        np.testing.assert_array_equal(actual, reference)
        assert not np.shares_memory(actual, reference)


def test_duration_contract_rejects_unregistered_or_malformed_inputs() -> None:
    candidate = np.zeros((100, 8), dtype=np.float32)
    reference = np.zeros(8, dtype=np.float32)
    with pytest.raises(ValueError, match="1/4/8/16"):
        intervention_action(reference, candidate, branch_step=0, intervention_steps=2)
    with pytest.raises(ValueError, match="non-negative"):
        intervention_action(reference, candidate, branch_step=-1, intervention_steps=1)
    with pytest.raises(ValueError, match="shape"):
        intervention_action(reference, candidate[:99], branch_step=0, intervention_steps=1)


def test_restore_separates_consumed_qpos_parity_from_rerender_noise(monkeypatch) -> None:
    captured = {
        "agent": {"panda-0": {"qpos": np.full((1, 8), 0.25, dtype=np.float32)}},
        "sensor_data": {
            "head_camera_agent0": {
                "rgb": np.zeros((1, 2, 3, 3), dtype=np.uint8)
            }
        },
    }
    rerendered = deepcopy(captured)
    rerendered["sensor_data"]["head_camera_agent0"]["rgb"].fill(16)

    class Base:
        _episode_rng = None

        def __init__(self) -> None:
            self._elapsed_steps = torch.tensor([9])

        def set_state_dict(self, value) -> None:
            assert value == {"physical": "snapshot"}

        def get_obs(self):
            return deepcopy(rerendered)

    base = Base()
    env = SimpleNamespace(base_env=base, unwrapped=base, _elapsed_steps=torch.tensor([9]))
    snapshot = MarsSimulatorSnapshot(
        state={"physical": "snapshot"},
        elapsed_steps=torch.tensor([9]),
        wrapper_elapsed_steps=torch.tensor([9]),
        observation=deepcopy(captured),
        runtime=SimpleNamespace(arms=(0,)),
        python_rng=random.getstate(),
        numpy_rng=deepcopy(np.random.get_state()),
        torch_rng=torch.get_rng_state().clone(),
        cuda_rng=[],
        episode_rng=None,
        start_metrics={},
    )
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda _value: None)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _value: None)
    observation, _runtime, restore_error, rerender_error = restore_snapshot(
        env, snapshot, seed=7
    )
    assert restore_error == 0.0
    assert rerender_error == 16.0
    np.testing.assert_array_equal(
        observation["sensor_data"]["head_camera_agent0"]["rgb"],
        captured["sensor_data"]["head_camera_agent0"]["rgb"],
    )
