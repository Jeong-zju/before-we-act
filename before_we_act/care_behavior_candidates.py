"""Behavior-level CARE candidates shared by every benchmark.

The original candidate family only perturbed the *first* action of a chunk:
hold one step, interpolate the first row two ways, or freeze the gripper.  Under
the one-step intervention those alternatives differed from the nominal action by
a fraction of a single joint step, so the realized advantage of the best
candidate was around ``1e-4`` and no alternative could ever clear a calibrated
lower bound.

These candidates instead express decisions a teammate can actually respond to --
wait, yield the workspace, grasp earlier or later, move slower -- and they are
held for the whole executed commitment window rather than a single step.

Only the first ``intervention_steps`` rows are executed open loop; the policy
re-plans afterwards and the chunk tail is superseded.  Each transform therefore
concentrates its behavior in that prefix, and the legality envelope is checked
there.  The tail still has to be finite and inside the action bounds.

Every transform reads only the nominal chunk, the acting robot's current joint
pose, and its current gripper command.  No teammate state, no privileged
simulator state, and no benchmark-specific dimension: the layout is
``[joint_0 .. joint_{d-2}, gripper]``, so ``joints = action_dim - 1`` covers
RoboFactory and MARS (8) as well as any other arm with a trailing gripper.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import numpy as np


CANDIDATE_NAMES: tuple[str, ...] = (
    "reference",
    "wait",
    "retreat",
    "grip_delay",
    "grip_advance",
    "slow",
)
CANDIDATE_COUNT = len(CANDIDATE_NAMES)
FAMILY_VERSION = "before-we-act.care-behavior-candidates/1"


@dataclass(frozen=True)
class BehaviorCandidateConfig:
    """Frozen shape and magnitude contract for the behavior candidate family."""

    action_horizon: int
    action_dim: int
    intervention_steps: int = 8
    wait_steps: int = 4
    retreat_scale: float = 0.5
    grip_shift_steps: int = 4
    slow_scale: float = 0.6

    def __post_init__(self) -> None:
        if self.action_dim < 2:
            raise ValueError("CARE actions need at least one joint and one gripper")
        if self.action_horizon < 2:
            raise ValueError("CARE action horizon must span at least two steps")
        if not 1 <= self.intervention_steps <= self.action_horizon:
            raise ValueError("CARE commitment must lie inside the action horizon")
        if not 1 <= self.wait_steps <= self.intervention_steps:
            raise ValueError("wait must fit inside the executed commitment window")
        if not 1 <= self.grip_shift_steps <= self.intervention_steps:
            raise ValueError("grip shift must fit inside the executed commitment window")
        if not 0.0 < self.retreat_scale <= 1.0:
            raise ValueError("retreat scale must lie in (0, 1]")
        if not 0.0 < self.slow_scale < 1.0:
            raise ValueError("slow scale must lie in (0, 1)")

    @property
    def joints(self) -> int:
        return self.action_dim - 1

    @property
    def gripper(self) -> int:
        return self.action_dim - 1

    def to_dict(self) -> dict[str, object]:
        return {"family_version": FAMILY_VERSION, **asdict(self)}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "BehaviorCandidateConfig":
        row = {key: item for key, item in value.items() if key != "family_version"}
        return cls(**row)  # type: ignore[arg-type]


def _validated_inputs(
    reference: np.ndarray,
    current_qpos: np.ndarray,
    current_grip: float,
    config: BehaviorCandidateConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    plan = np.asarray(reference, dtype=np.float32)
    expected = (config.action_horizon, config.action_dim)
    if plan.shape != expected or not np.isfinite(plan).all():
        raise ValueError(
            f"CARE reference chunk must be finite {expected}, got {plan.shape}"
        )
    pose = np.asarray(current_qpos, dtype=np.float32).reshape(-1)
    if pose.shape[0] < config.joints or not np.isfinite(pose[: config.joints]).all():
        raise ValueError(
            f"CARE current pose needs {config.joints} finite joints, got {pose.shape}"
        )
    grip = float(current_grip)
    if not math.isfinite(grip):
        raise ValueError("CARE current gripper command must be finite")
    return plan, pose[: config.joints].copy(), grip


def _hold_row(
    pose: np.ndarray, grip: float, config: BehaviorCandidateConfig
) -> np.ndarray:
    row = np.empty(config.action_dim, dtype=np.float32)
    row[: config.joints] = pose
    row[config.gripper] = grip
    return row


def _wait(
    plan: np.ndarray,
    pose: np.ndarray,
    grip: float,
    config: BehaviorCandidateConfig,
) -> np.ndarray:
    """Hold the current pose, then resume the nominal plan that many steps late."""

    steps = config.wait_steps
    held = np.repeat(_hold_row(pose, grip, config)[None], steps, axis=0)
    return np.concatenate((held, plan[:-steps]), axis=0)


def _retreat(
    plan: np.ndarray,
    pose: np.ndarray,
    grip: float,
    config: BehaviorCandidateConfig,
    joint_low: np.ndarray | None = None,
    joint_high: np.ndarray | None = None,
) -> np.ndarray:
    """Back away from where the nominal plan was heading, then rejoin it.

    The retreat is scaled by the motion the nominal plan would have covered over
    the commitment window, so a nearly stationary arm barely moves -- there is
    nothing to yield -- while a fast arm yields proportionally.  Backing off over
    the whole window keeps the executed rate below the nominal rate.

    This is the one transform that leaves the convex hull of the nominal plan and
    the current pose: every other candidate resamples values the reference
    already visits.  Bounds are therefore applied to the retreat *target* rather
    than to the finished chunk, so the candidate is in range by construction --
    some benchmarks deliberately execute candidates without clipping.  Near a
    joint limit the retreat shortens instead of being rejected, which is also
    the physical answer: there is no room to yield into.
    """

    window = config.intervention_steps
    direction = plan[window - 1, : config.joints] - pose
    goal = pose - config.retreat_scale * direction
    if joint_low is not None and joint_high is not None:
        low = np.asarray(joint_low, dtype=np.float32).reshape(-1)[: config.joints]
        high = np.asarray(joint_high, dtype=np.float32).reshape(-1)[: config.joints]
        if low.shape != goal.shape or np.any(low > high):
            raise ValueError("CARE retreat bounds must be matching finite joint limits")
        goal = np.clip(goal, low, high)

    result = np.empty_like(plan)
    ramp = (np.arange(1, window + 1, dtype=np.float32) / float(window))[:, None]
    result[:window, : config.joints] = pose[None] + ramp * (goal - pose)[None]
    result[:window, config.gripper] = grip

    # The tail is superseded by re-planning; ease back onto the nominal timeline
    # so the stored chunk stays a plausible continuation rather than a jump.
    rejoin = min(window, config.action_horizon - window)
    if rejoin > 0:
        ramp = (np.arange(1, rejoin + 1, dtype=np.float32) / float(rejoin))[:, None]
        result[window : window + rejoin, : config.joints] = (
            goal[None] + ramp * (plan[0, : config.joints] - goal)[None]
        )
        result[window : window + rejoin, config.gripper] = grip
    remainder = config.action_horizon - window - rejoin
    if remainder > 0:
        result[window + rejoin :] = plan[:remainder]
    return result


def _grip_shift(
    plan: np.ndarray,
    grip: float,
    config: BehaviorCandidateConfig,
    *,
    steps: int,
) -> np.ndarray:
    """Move the gripper trajectory in time while the arm follows the nominal plan.

    A positive shift delays the grasp or release; a negative shift brings it
    forward.  The arm channels are untouched, so this isolates timing from
    trajectory.
    """

    result = plan.copy()
    index = np.arange(config.action_horizon) - int(steps)
    channel = np.where(
        index < 0,
        np.float32(grip),
        plan[np.clip(index, 0, config.action_horizon - 1), config.gripper],
    )
    result[:, config.gripper] = channel.astype(np.float32)
    return result


def _slow(
    plan: np.ndarray,
    pose: np.ndarray,
    grip: float,
    config: BehaviorCandidateConfig,
) -> np.ndarray:
    """Traverse the nominal path at a reduced rate.

    Joint targets are interpolated along the nominal knots so the path is
    preserved and only its timing changes.  The gripper takes the knot value at
    the floor of the warped index: it is a drive target, not a pose to blend.
    """

    joints = config.joints
    arm_knots = np.concatenate((pose[None], plan[:, :joints]), axis=0)
    grip_knots = np.concatenate((np.asarray([grip], dtype=np.float32), plan[:, config.gripper]))
    result = np.empty_like(plan)
    horizon = config.action_horizon
    for step in range(horizon):
        tau = min((step + 1) * config.slow_scale, float(horizon))
        lower = int(math.floor(tau))
        upper = min(lower + 1, horizon)
        fraction = tau - lower
        result[step, :joints] = (1.0 - fraction) * arm_knots[lower] + fraction * arm_knots[upper]
        result[step, config.gripper] = grip_knots[lower]
    return result


def behavior_candidate_plan(
    candidate_id: int,
    reference: np.ndarray,
    current_qpos: np.ndarray,
    current_grip: float,
    config: BehaviorCandidateConfig,
    *,
    joint_low: np.ndarray | None = None,
    joint_high: np.ndarray | None = None,
) -> np.ndarray:
    """Return one behavior-level candidate chunk.

    Candidate zero is the nominal chunk, returned element-for-element so the
    fail-closed path stays exact.
    """

    plan, pose, grip = _validated_inputs(reference, current_qpos, current_grip, config)
    if candidate_id == 0:
        return plan.copy()
    if candidate_id == 1:
        result = _wait(plan, pose, grip, config)
    elif candidate_id == 2:
        result = _retreat(plan, pose, grip, config, joint_low, joint_high)
    elif candidate_id == 3:
        result = _grip_shift(plan, grip, config, steps=config.grip_shift_steps)
    elif candidate_id == 4:
        result = _grip_shift(plan, grip, config, steps=-config.grip_shift_steps)
    elif candidate_id == 5:
        result = _slow(plan, pose, grip, config)
    else:
        raise ValueError(f"unknown CARE behavior candidate: {candidate_id}")
    if result.shape != plan.shape or not np.isfinite(result).all():
        raise ValueError(f"CARE candidate {candidate_id} produced an invalid chunk")
    return result.astype(np.float32, copy=False)


def behavior_candidate_set(
    reference: np.ndarray,
    current_qpos: np.ndarray,
    current_grip: float,
    config: BehaviorCandidateConfig,
    *,
    joint_low: np.ndarray | None = None,
    joint_high: np.ndarray | None = None,
) -> np.ndarray:
    """Return the whole family as ``[candidate, step, action]``."""

    return np.stack(
        [
            behavior_candidate_plan(
                index,
                reference,
                current_qpos,
                current_grip,
                config,
                joint_low=joint_low,
                joint_high=joint_high,
            )
            for index in range(CANDIDATE_COUNT)
        ]
    )


def validate_behavior_candidate(
    candidate_id: int,
    plan: np.ndarray,
    reference: np.ndarray,
    current_qpos: np.ndarray,
    action_low: Sequence[float],
    action_high: Sequence[float],
    config: BehaviorCandidateConfig,
    *,
    rate_tolerance: float = 1.25,
) -> tuple[bool, list[str]]:
    """Check the executed prefix against the nominal rate and action bounds.

    The rate envelope is measured over the commitment window that is actually
    executed open loop.  The chunk tail is superseded by re-planning, so it is
    only required to be finite and inside the action bounds.
    """

    failures: list[str] = []
    value = np.asarray(plan, dtype=np.float32)
    expected = (config.action_horizon, config.action_dim)
    if value.shape != expected or not np.isfinite(value).all():
        return False, ["shape_or_finite"]

    low = np.asarray(action_low, dtype=np.float32)
    high = np.asarray(action_high, dtype=np.float32)
    if low.shape != (config.action_dim,) or high.shape != (config.action_dim,):
        raise ValueError("CARE action bounds must match the action dimension")
    if np.any(value < low[None] - 1e-6) or np.any(value > high[None] + 1e-6):
        failures.append("action_domain")

    if candidate_id != 0:
        nominal = np.asarray(reference, dtype=np.float32)
        pose = np.asarray(current_qpos, dtype=np.float32).reshape(-1)[: config.joints]
        window = config.intervention_steps
        nominal_path = np.concatenate((pose[None], nominal[:window, : config.joints]))
        nominal_rate = np.max(np.abs(np.diff(nominal_path, axis=0)), axis=0)
        candidate_path = np.concatenate((pose[None], value[:window, : config.joints]))
        candidate_rate = np.max(np.abs(np.diff(candidate_path, axis=0)), axis=0)
        if np.any(candidate_rate > rate_tolerance * nominal_rate + 1e-5):
            failures.append("joint_rate_envelope")

    return not failures, failures


__all__ = [
    "CANDIDATE_COUNT",
    "CANDIDATE_NAMES",
    "FAMILY_VERSION",
    "BehaviorCandidateConfig",
    "behavior_candidate_plan",
    "behavior_candidate_set",
    "validate_behavior_candidate",
]
