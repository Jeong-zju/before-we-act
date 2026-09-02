"""Canonical CARE candidate plans for DuoBench.

The six candidates deliberately mirror the registered RoboFactory CARE
candidate family.  They are transformations of the *frozen reference plan*;
the belief head only scores these plans and never generates a residual action.

DuoBench policies encode joints relative to the qpos at proposal time while
the gripper remains absolute.  Candidate construction is therefore performed
in physical (absolute-joint) coordinates and converted back to that encoding
at the boundary.  Keeping this module pure makes the exact same construction
usable by branch collection and closed-loop deployment.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import numpy as np

from deployment.duo_act.action_target import (
    CONTROLLER_JOINT_HIGH,
    CONTROLLER_JOINT_LOW,
    canonicalize_controller_action_with_audit,
)


ACTION_DIM: Final = 8
JOINT_DIM: Final = 7
CANDIDATE_COUNT: Final = 6
CANDIDATE_NAMES: Final = (
    "reference",
    "base",
    "hold_one_step",
    "time_warp_0.75",
    "time_warp_1.25",
    "freeze_gripper",
)


@dataclass(frozen=True)
class CandidateAudit:
    valid: bool
    failures: tuple[str, ...]
    first_joint_delta_linf: float
    changed_values: int
    max_abs_canonicalization: float


def _require_chunk(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != ACTION_DIM or not len(result):
        raise ValueError(f"{name} must be finite [horizon,{ACTION_DIM}], got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a non-finite action")
    return result


def _require_qpos(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (ACTION_DIM,) or not np.isfinite(result).all():
        raise ValueError(f"proposal qpos must be finite [{ACTION_DIM}], got {result.shape}")
    return result


def decoded_absolute_chunk(encoded: np.ndarray, proposal_qpos: np.ndarray) -> np.ndarray:
    """Decode anchor-relative joint actions without changing the gripper."""

    result = _require_chunk(encoded, "encoded candidate").copy()
    qpos = _require_qpos(proposal_qpos)
    result[:, :JOINT_DIM] += qpos[None, :JOINT_DIM]
    return result


def encoded_relative_chunk(absolute: np.ndarray, proposal_qpos: np.ndarray) -> np.ndarray:
    """Encode absolute joint actions relative to the proposal-time qpos."""

    result = _require_chunk(absolute, "absolute candidate").copy()
    qpos = _require_qpos(proposal_qpos)
    result[:, :JOINT_DIM] -= qpos[None, :JOINT_DIM]
    return result


def canonicalize_encoded_chunk(
    encoded: np.ndarray,
    proposal_qpos: np.ndarray,
    joint_low: np.ndarray,
    joint_high: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Return the command that can actually reach the DuoBench controller."""

    absolute = decoded_absolute_chunk(encoded, proposal_qpos)
    low = np.asarray(joint_low, dtype=np.float32)
    high = np.asarray(joint_high, dtype=np.float32)
    if low.shape != (JOINT_DIM,) or high.shape != (JOINT_DIM,) or np.any(low >= high):
        raise ValueError("DuoBench joint bounds must be matching finite [7] vectors")
    # Formal DuoBench always passes the pinned controller range.  Keep this
    # pure kernel usable with bounded fake environments in contract tests; the
    # fallback still applies the explicitly supplied environment bounds.
    if np.array_equal(low, CONTROLLER_JOINT_LOW) and np.array_equal(
        high, CONTROLLER_JOINT_HIGH
    ):
        # Use the same auditable controller boundary as data preparation and
        # closed-loop emission.  This includes binary-gripper canonicalization.
        absolute, audit = canonicalize_controller_action_with_audit(absolute)
        changed = int(audit["changed_values"])
        maximum = float(audit["max_abs_delta"])
    else:
        before = absolute.copy()
        absolute[:, :JOINT_DIM] = np.clip(absolute[:, :JOINT_DIM], low, high)
        absolute[:, JOINT_DIM] = (absolute[:, JOINT_DIM] >= 0.5).astype(np.float32)
        delta = np.abs(absolute.astype(np.float64) - before.astype(np.float64))
        changed = int(np.count_nonzero(delta))
        maximum = float(delta.max(initial=0.0))
    return (
        encoded_relative_chunk(absolute, proposal_qpos),
        changed,
        maximum,
    )


def time_warp_absolute(
    reference: np.ndarray,
    current_qpos: np.ndarray,
    scale: float,
    current_gripper: float,
) -> np.ndarray:
    """Time-warp an absolute plan using the RoboFactory CARE convention."""

    plan = _require_chunk(reference, "absolute reference")
    qpos = _require_qpos(current_qpos)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("CARE time-warp scale must be positive")
    horizon = len(plan)
    joint_knots = np.concatenate((qpos[None, :JOINT_DIM], plan[:, :JOINT_DIM]), axis=0)
    grip_knots = np.concatenate((np.asarray([current_gripper], np.float32), plan[:, JOINT_DIM]))
    result = np.empty_like(plan)
    for index in range(horizon):
        tau = min((index + 1) * float(scale), float(horizon))
        lower = int(math.floor(tau))
        upper = min(lower + 1, horizon)
        fraction = tau - lower
        result[index, :JOINT_DIM] = (
            (1.0 - fraction) * joint_knots[lower]
            + fraction * joint_knots[upper]
        )
        result[index, JOINT_DIM] = grip_knots[lower]
    return result


def candidate_plan(
    candidate_id: int,
    reference_encoded: np.ndarray,
    base_encoded: np.ndarray,
    proposal_qpos: np.ndarray,
    *,
    joint_low: np.ndarray,
    joint_high: np.ndarray,
    current_gripper: float | None = None,
) -> tuple[np.ndarray, CandidateAudit]:
    """Build one canonical proposal-time encoded CARE candidate.

    Candidate 0 is the frozen reference, candidate 1 is the frozen policy's
    belief-off/base plan, and candidates 2--5 are the registered delay,
    slow/fast time-warp, and gripper-freeze transformations.
    """

    if not 0 <= int(candidate_id) < CANDIDATE_COUNT:
        raise ValueError(f"unknown CARE candidate {candidate_id}")
    reference = _require_chunk(reference_encoded, "reference plan")
    base = _require_chunk(base_encoded, "base plan")
    if base.shape != reference.shape:
        raise ValueError("reference/base CARE plan shapes differ")
    qpos = _require_qpos(proposal_qpos)
    reference, ref_changed, ref_max = canonicalize_encoded_chunk(
        reference, qpos, joint_low, joint_high
    )
    base, base_changed, base_max = canonicalize_encoded_chunk(
        base, qpos, joint_low, joint_high
    )
    reference_absolute = decoded_absolute_chunk(reference, qpos)
    base_absolute = decoded_absolute_chunk(base, qpos)
    grip = float(reference_absolute[0, JOINT_DIM] if current_gripper is None else current_gripper)
    grip = float(grip >= 0.5)

    if candidate_id == 0:
        absolute = reference_absolute.copy()
    elif candidate_id == 1:
        absolute = base_absolute.copy()
    elif candidate_id == 2:
        hold = np.concatenate((qpos[:JOINT_DIM], np.asarray([grip], np.float32)))
        absolute = np.concatenate((hold[None], reference_absolute[:-1]), axis=0)
    elif candidate_id == 3:
        absolute = time_warp_absolute(reference_absolute, qpos, 0.75, grip)
    elif candidate_id == 4:
        absolute = time_warp_absolute(reference_absolute, qpos, 1.25, grip)
    else:
        absolute = reference_absolute.copy()
        absolute[:, JOINT_DIM] = grip

    encoded = encoded_relative_chunk(absolute, qpos)
    encoded, changed, maximum = canonicalize_encoded_chunk(
        encoded, qpos, joint_low, joint_high
    )
    physical = decoded_absolute_chunk(encoded, qpos)
    failures: list[str] = []
    if not np.isfinite(encoded).all():
        failures.append("non_finite")
    if np.any(physical[:, :JOINT_DIM] < np.asarray(joint_low) - 1e-6) or np.any(
        physical[:, :JOINT_DIM] > np.asarray(joint_high) + 1e-6
    ):
        failures.append("joint_domain")
    if not np.isin(physical[:, JOINT_DIM], (0.0, 1.0)).all():
        failures.append("gripper_domain")
    first_delta = float(np.max(np.abs(physical[0, :JOINT_DIM] - qpos[:JOINT_DIM])))
    return encoded.astype(np.float32), CandidateAudit(
        valid=not failures,
        failures=tuple(failures),
        first_joint_delta_linf=first_delta,
        changed_values=ref_changed + base_changed + changed,
        max_abs_canonicalization=max(ref_max, base_max, maximum),
    )


def candidate_family(
    reference_encoded: np.ndarray,
    base_encoded: np.ndarray,
    proposal_qpos: np.ndarray,
    *,
    joint_low: np.ndarray,
    joint_high: np.ndarray,
    current_gripper: float | None = None,
) -> tuple[np.ndarray, tuple[CandidateAudit, ...]]:
    """Build all six candidates in their stable registered order."""

    rows: list[np.ndarray] = []
    audits: list[CandidateAudit] = []
    for candidate_id in range(CANDIDATE_COUNT):
        plan, audit = candidate_plan(
            candidate_id,
            reference_encoded,
            base_encoded,
            proposal_qpos,
            joint_low=joint_low,
            joint_high=joint_high,
            current_gripper=current_gripper,
        )
        rows.append(plan)
        audits.append(audit)
    return np.stack(rows).astype(np.float32), tuple(audits)


__all__ = [
    "ACTION_DIM",
    "CANDIDATE_COUNT",
    "CANDIDATE_NAMES",
    "CandidateAudit",
    "candidate_family",
    "candidate_plan",
    "canonicalize_encoded_chunk",
    "decoded_absolute_chunk",
    "encoded_relative_chunk",
    "time_warp_absolute",
]
