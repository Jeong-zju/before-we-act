"""Pure helpers for the DuoBench local joint-residual action contract.

The policy predicts a chunk relative to the joint observation at the instant
the chunk was proposed.  Older chunks therefore cannot be averaged as raw
residuals at a later control step: first rebase every chunk onto the current
joint observation, then apply the temporal-ensemble weights.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


JOINT_DIM = 7


def encode_anchor_relative_chunk(
    absolute_actions: np.ndarray,
    anchor_qpos: np.ndarray,
) -> np.ndarray:
    """Encode ``[H, 8]`` absolute commands relative to one anchor state."""
    actions = np.asarray(absolute_actions, dtype=np.float32)
    anchor = np.asarray(anchor_qpos, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] < JOINT_DIM + 1:
        raise ValueError(f"expected [H,A>=8] actions, got {actions.shape}")
    if anchor.ndim != 1 or anchor.shape[-1] < JOINT_DIM:
        raise ValueError(f"expected [Q>=7] anchor qpos, got {anchor.shape}")
    encoded = actions.copy()
    encoded[:, :JOINT_DIM] -= anchor[None, :JOINT_DIM]
    return encoded


def current_relative_temporal_ensemble(
    active_chunks: Sequence[tuple[int, np.ndarray, np.ndarray]],
    current_step: int,
    current_qpos: np.ndarray,
    *,
    decay: float = 0.01,
) -> np.ndarray:
    """Rebase anchor-relative chunks and ensemble their current commands.

    Each entry is ``(proposal_step, chunk, proposal_qpos)`` where ``chunk`` is
    ``[agents, horizon, action_dim]``.  Joint entries are returned relative to
    ``current_qpos``; gripper entries stay in their absolute ``[0, 1]`` domain.
    """
    if not active_chunks:
        raise ValueError("temporal ensemble requires at least one active chunk")
    current = np.asarray(current_qpos, dtype=np.float32)
    if current.ndim != 2 or current.shape[-1] < JOINT_DIM:
        raise ValueError(f"expected [agents,Q>=7] current qpos, got {current.shape}")

    rebased: list[np.ndarray] = []
    ages: list[int] = []
    for proposal_step, chunk, proposal_qpos in active_chunks:
        age = int(current_step) - int(proposal_step)
        values = np.asarray(chunk, dtype=np.float32)
        reference = np.asarray(proposal_qpos, dtype=np.float32)
        if values.ndim != 3 or values.shape[0] != current.shape[0]:
            raise ValueError(f"expected [agents,H,A] chunk, got {values.shape}")
        if age < 0 or age >= values.shape[1]:
            raise ValueError(f"chunk age {age} outside horizon {values.shape[1]}")
        if reference.shape != current.shape:
            raise ValueError(
                f"proposal/current qpos shapes differ: {reference.shape}/{current.shape}"
            )
        command = values[:, age].copy()
        command[:, :JOINT_DIM] += (
            reference[:, :JOINT_DIM] - current[:, :JOINT_DIM]
        )
        rebased.append(command)
        ages.append(age)

    weights = np.exp(-float(decay) * np.asarray(ages, dtype=np.float64))
    weights /= weights.sum()
    return np.sum(
        np.stack(rebased, axis=0) * weights[:, None, None], axis=0
    ).astype(np.float32)
