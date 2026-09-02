"""Select which CARE candidate family a collector or runtime builds.

Two families coexist during the migration:

``fixed``
    The original transforms in :mod:`before_we_act.care_branch_collector`.  They
    perturb only the first action of a chunk, which is why the calibrated
    selector could never clear its lower bound.  Retained so archived corpora
    stay reproducible.

``behavior``
    The behavior-level family in
    :mod:`before_we_act.care_behavior_candidates`, held across the executed
    commitment window.

Both are exposed behind one signature so a collector picks a family by name and
its call sites stay unchanged.  The chosen family is recorded in the branch
manifest: mixing families inside one corpus would make the advantages
incomparable.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from before_we_act.care_behavior_candidates import (
    CANDIDATE_COUNT as BEHAVIOR_CANDIDATE_COUNT,
    CANDIDATE_NAMES as BEHAVIOR_CANDIDATE_NAMES,
    BehaviorCandidateConfig,
    behavior_candidate_plan,
    validate_behavior_candidate,
)


FIXED_FAMILY = "fixed"
BEHAVIOR_FAMILY = "behavior"
CANDIDATE_FAMILIES = (FIXED_FAMILY, BEHAVIOR_FAMILY)
FIXED_CANDIDATE_NAMES = (
    "reference",
    "base",
    "hold_one_step",
    "time_warp_0.75",
    "time_warp_1.25",
    "freeze_gripper",
)


def candidate_names(family: str) -> tuple[str, ...]:
    if family == FIXED_FAMILY:
        return FIXED_CANDIDATE_NAMES
    if family == BEHAVIOR_FAMILY:
        return BEHAVIOR_CANDIDATE_NAMES
    raise ValueError(f"unknown CARE candidate family: {family}")


def candidate_count(family: str) -> int:
    return len(candidate_names(family))


def _bounds(action_space: Any) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(action_space.low, dtype=np.float32),
        np.asarray(action_space.high, dtype=np.float32),
    )


def build_candidate(
    family: str,
    candidate_id: int,
    *,
    reference: np.ndarray,
    base: np.ndarray,
    current_qpos: np.ndarray,
    current_grip: float,
    config: BehaviorCandidateConfig | None = None,
    joint_low: np.ndarray | None = None,
    joint_high: np.ndarray | None = None,
) -> np.ndarray:
    """Return one candidate chunk from the requested family."""

    if family == FIXED_FAMILY:
        from before_we_act.care_branch_collector import candidate_plan

        return candidate_plan(candidate_id, reference, base, current_qpos, current_grip)
    if family == BEHAVIOR_FAMILY:
        if config is None:
            raise ValueError("the behavior family needs a BehaviorCandidateConfig")
        return behavior_candidate_plan(
            candidate_id,
            reference,
            current_qpos,
            current_grip,
            config,
            joint_low=joint_low,
            joint_high=joint_high,
        )
    raise ValueError(f"unknown CARE candidate family: {family}")


def validate_candidate_for_family(
    family: str,
    candidate_id: int,
    plan: np.ndarray,
    *,
    reference: np.ndarray,
    base: np.ndarray,
    current_qpos: np.ndarray,
    current_grip: float,
    action_space: Any,
    config: BehaviorCandidateConfig | None = None,
) -> tuple[bool, list[str]]:
    """Check a candidate against the legality rules of its own family."""

    if family == FIXED_FAMILY:
        from before_we_act.care_branch_collector import validate_candidate

        return validate_candidate(
            candidate_id,
            plan,
            reference,
            base,
            current_qpos,
            current_grip,
            action_space,
        )
    if family == BEHAVIOR_FAMILY:
        if config is None:
            raise ValueError("the behavior family needs a BehaviorCandidateConfig")
        low, high = _bounds(action_space)
        return validate_behavior_candidate(
            candidate_id, plan, reference, current_qpos, low, high, config
        )
    raise ValueError(f"unknown CARE candidate family: {family}")


def build_candidate_set(
    family: str,
    *,
    reference: np.ndarray,
    base: np.ndarray,
    current_qpos: np.ndarray,
    current_grip: float,
    config: BehaviorCandidateConfig | None = None,
) -> np.ndarray:
    """Return the whole family as ``[candidate, step, action]``."""

    return np.stack(
        [
            build_candidate(
                family,
                index,
                reference=reference,
                base=base,
                current_qpos=current_qpos,
                current_grip=current_grip,
                config=config,
            )
            for index in range(candidate_count(family))
        ]
    )


def family_manifest(
    family: str, config: BehaviorCandidateConfig | None
) -> dict[str, Any]:
    """The provenance a branch corpus must carry so families are never mixed."""

    row: dict[str, Any] = {
        "candidate_family": family,
        "candidate_names": list(candidate_names(family)),
        "candidate_count": candidate_count(family),
    }
    if family == BEHAVIOR_FAMILY:
        if config is None:
            raise ValueError("the behavior family needs a BehaviorCandidateConfig")
        row["candidate_family_config"] = config.to_dict()
    return row


__all__ = [
    "BEHAVIOR_CANDIDATE_COUNT",
    "BEHAVIOR_FAMILY",
    "CANDIDATE_FAMILIES",
    "FIXED_CANDIDATE_NAMES",
    "FIXED_FAMILY",
    "build_candidate",
    "build_candidate_set",
    "candidate_count",
    "candidate_names",
    "family_manifest",
    "validate_candidate_for_family",
]
