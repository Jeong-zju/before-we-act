"""Chunk commitment must follow the caller's candidate shape, not MARS's.

RoboFactory and MARS act on [K, 100, 8]; BiCoord acts on [K, 100, 7]. The
commitment state machine reads both the candidate arity and the chunk shape
from the tensor it is handed, so porting it needs no per-benchmark edit.
"""
from __future__ import annotations

import numpy as np
import pytest

from before_we_act.care_chunk_commitment import (
    advance_chunk_commitments,
    apply_chunk_commitments,
)


def _stack(candidates: int, horizon: int, action_dim: int) -> np.ndarray:
    values = np.arange(candidates * horizon * action_dim, dtype=np.float32)
    return values.reshape(candidates, horizon, action_dim)


@pytest.mark.parametrize(
    ("candidates", "action_dim"),
    [(6, 8), (6, 7), (4, 7), (8, 8)],
)
def test_commitment_round_trips_for_any_candidate_shape(
    candidates: int, action_dim: int
) -> None:
    horizon, steps = 100, 8
    stacks = [_stack(candidates, horizon, action_dim)]
    selected, best_lower = [0], [0.0]
    commitments: dict[int, dict[str, object]] = {}

    # Step 0: the selector picks a non-reference candidate and opens a window.
    selected[0] = candidates - 1
    best_lower[0] = 0.5
    active = apply_chunk_commitments(
        stacks, selected, best_lower, commitments, intervention_steps=steps
    )
    assert active == set()
    decisions, committed = advance_chunk_commitments(
        stacks, selected, best_lower, [0], commitments, active,
        intervention_steps=steps,
    )
    assert (decisions, committed) == (1, 0)
    assert commitments[0]["plan"].shape == (horizon, action_dim)

    # The window then replays its frozen plan for the remaining steps.
    for offset in range(1, steps):
        fresh = [_stack(candidates, horizon, action_dim)]
        active = apply_chunk_commitments(
            fresh, [0], [0.0], commitments, intervention_steps=steps
        )
        assert active == {0}
        _decisions, committed = advance_chunk_commitments(
            fresh, [candidates - 1], [0.5], [0], commitments, active,
            intervention_steps=steps,
        )
        assert committed == 1
    assert commitments == {}


def test_commitment_rejects_a_plan_from_a_different_action_space() -> None:
    stacks = [_stack(6, 100, 7)]
    commitments = {
        0: {
            "candidate_id": 1,
            "plan": np.zeros((100, 8), dtype=np.float32),
            "next_step": 1,
            "best_lower": 0.5,
        }
    }

    with pytest.raises(ValueError, match="plan/offset drifted"):
        apply_chunk_commitments(stacks, [0], [0.0], commitments, intervention_steps=8)


def test_commitment_rejects_a_candidate_id_outside_the_stack() -> None:
    stacks = [_stack(4, 100, 7)]
    commitments = {
        0: {
            "candidate_id": 4,
            "plan": np.zeros((100, 7), dtype=np.float32),
            "next_step": 1,
            "best_lower": 0.5,
        }
    }

    with pytest.raises(ValueError, match="identity is invalid"):
        apply_chunk_commitments(stacks, [0], [0.0], commitments, intervention_steps=8)
