"""Local H-step candidate commitment semantics for CARE deployment.

The candidate arity and chunk shape are read from the candidate tensor that
the caller already passes, so this module carries no benchmark-specific
dimensions: RoboFactory and MARS act on ``[K, 100, 8]`` while BiCoord acts
on ``[K, 100, 7]``.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def apply_chunk_commitments(
    candidates: Sequence[np.ndarray],
    selected: list[int],
    best_lower: list[float],
    commitments: dict[int, dict[str, Any]],
    *,
    intervention_steps: int,
) -> set[int]:
    active = set(commitments)
    for row, state in commitments.items():
        candidate_id = int(state["candidate_id"])
        offset = int(state["next_step"])
        plan = np.asarray(state["plan"], dtype=np.float32)
        if not 0 <= row < len(candidates):
            raise ValueError("CARE commitment identity is invalid")
        stack = np.asarray(candidates[row])
        if stack.ndim != 3:
            raise ValueError("CARE candidate stack must be [candidate,step,action]")
        if not 1 <= candidate_id < stack.shape[0]:
            raise ValueError("CARE commitment identity is invalid")
        if plan.shape != stack.shape[1:] or not 0 <= offset < int(intervention_steps):
            raise ValueError("CARE commitment plan/offset drifted")
        selected[row] = candidate_id
        best_lower[row] = float(state["best_lower"])
        candidates[row][candidate_id, 0] = plan[offset]
    return active


def advance_chunk_commitments(
    candidates: Sequence[np.ndarray],
    selected: Sequence[int],
    best_lower: Sequence[float],
    applied_rows: Sequence[int],
    commitments: dict[int, dict[str, Any]],
    active_rows: set[int],
    *,
    intervention_steps: int,
) -> tuple[int, int]:
    applied = {int(row) for row in applied_rows}
    if not active_rows.issubset(applied):
        raise RuntimeError("decentralized CARE suppressed an active chunk")
    decisions = committed_steps = 0
    for row in sorted(applied):
        if row in active_rows:
            state = commitments[row]
            state["next_step"] = int(state["next_step"]) + 1
            committed_steps += 1
            if int(state["next_step"]) >= int(intervention_steps):
                del commitments[row]
        elif int(selected[row]) != 0:
            decisions += 1
            if int(intervention_steps) > 1:
                commitments[row] = {
                    "candidate_id": int(selected[row]),
                    "plan": np.asarray(
                        candidates[row][int(selected[row])], dtype=np.float32
                    ).copy(),
                    "next_step": 1,
                    "best_lower": float(best_lower[row]),
                }
    return decisions, committed_steps


__all__ = ["advance_chunk_commitments", "apply_chunk_commitments"]
