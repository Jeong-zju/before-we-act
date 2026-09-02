"""Model-independent CARE branch-family contracts for DuoBench.

This module contains the pieces that must stay identical when the frozen
reference provider changes (ACT, DINO--Transformer, or another policy): anchor
sampling, branch-seed derivation, prefix outcome construction, and family
gates.  It deliberately does not import a policy or a simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HORIZONS = (8, 16, 32, 64)
BRANCH_CANDIDATES = 6
BRANCH_REPEATS = (0, 1)
BRANCH_REGIMES = ("reactive", "replay")

# Keep the registered CARE utility geometry.  The first two terms measure
# task progress/success; the remaining terms penalize unsafe or uncoordinated
# behavior.  Values are bounded before entering the scorer.
MAIN_WEIGHTS = np.asarray((0.30, 0.30, 0.12, 0.08, 0.06, 0.06, 0.03, 0.05), dtype=np.float64)


def derive_branch_seed(snapshot_id: str, repeat_id: int, *, protocol: str = "duobench-care-branch-v2") -> int:
    """Derive distinct, reproducible simulator seeds for each repeat."""

    if int(repeat_id) < 0:
        raise ValueError("repeat_id must be non-negative")
    digest = hashlib.sha256(f"{protocol}|{snapshot_id}|repeat={int(repeat_id)}".encode()).digest()
    # Keep the value in numpy/gym's portable uint32 seed range.
    return int.from_bytes(digest[:4], "big")


def stratified_anchor_steps(
    episode_length: int,
    *,
    max_steps: int,
    count: int = 30,
    horizon: int = 64,
    critical_count: int = 20,
) -> tuple[dict[str, Any], ...]:
    """Return stable critical+uniform mid/late anchors for one trajectory.

    Anchors are always non-terminal and leave at least ``horizon`` controls in
    the source episode.  The returned rows are metadata only; a collector is
    responsible for replaying the reference policy to the selected state and
    taking an exact simulator snapshot there.
    """

    length = int(episode_length)
    limit = int(max_steps)
    total = int(count)
    critical = min(int(critical_count), total)
    if length <= horizon + 1 or limit <= horizon:
        raise ValueError(f"episode too short for CARE anchors: length={length}, max_steps={limit}")
    usable = max(1, min(length - horizon, limit - horizon))
    rows: list[dict[str, Any]] = []
    uniform_count = total - critical
    for ordinal in range(total):
        if ordinal < critical:
            # Transition-rich mid/late support (35%--80%), matching the
            # official RoboFactory compact protocol.
            phase = 0.35 + 0.45 * ((ordinal + 0.5) / max(critical, 1))
            stratum = "critical"
            stratum_index = ordinal
            stratum_count = critical
        else:
            # Broad support includes early and late non-terminal states.
            j = ordinal - critical
            phase = 0.10 + 0.75 * ((j + 0.5) / max(uniform_count, 1))
            stratum = "uniform"
            stratum_index = j
            stratum_count = uniform_count
        anchor = min(usable, max(1, int(round(phase * usable))))
        rows.append(
            {
                "ordinal": ordinal,
                "anchor_step": anchor,
                "sampling_stratum": stratum,
                "stratum_index": stratum_index,
                "stratum_count": stratum_count,
                "available_steps_after_anchor": max(0, length - anchor),
                "horizon": int(horizon),
            }
        )
    return tuple(rows)


def _scalar(value: Any, default: float = 0.0) -> float:
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return float(default)
        return float(arr.reshape(-1)[0])
    except Exception:
        return float(default)


def _bool(value: Any) -> bool:
    try:
        return bool(np.asarray(value).all())
    except Exception:
        return bool(value)


def _prefix_rows(rows: Sequence[Mapping[str, Any]], horizon: int) -> list[Mapping[str, Any]]:
    """Take exactly a prefix, padding only after genuine success."""

    if not rows:
        raise ValueError("a CARE branch must contain at least one step")
    selected = list(rows[: int(horizon)])
    if len(selected) < int(horizon) and _bool(selected[-1].get("success", False)):
        terminal = dict(selected[-1])
        terminal.update(
            {
                "collision_or_drop": False,
                "robot_conflict": False,
                "duplicate_work": False,
                "active": (False, False),
                "all_joint_changes_below_threshold": True,
            }
        )
        selected.extend(dict(terminal) for _ in range(int(horizon) - len(selected)))
    return selected


def _deadlock_mask(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    progress = np.asarray([_scalar(row.get("progress", 0.0)) for row in rows], dtype=np.float64)
    stagnant = np.asarray(
        [
            index > 0
            and abs(progress[index] - progress[index - 1]) <= 1e-4
            and _bool(row.get("all_joint_changes_below_threshold", row.get("all_joint_changes_below_0_02", False)))
            and not _bool(row.get("success", False))
            for index, row in enumerate(rows)
        ],
        dtype=bool,
    )
    result = np.zeros(len(rows), dtype=bool)
    start = 0
    while start < len(rows):
        if not stagnant[start]:
            start += 1
            continue
        stop = start
        while stop < len(rows) and stagnant[stop]:
            stop += 1
        if stop - start >= 8:
            result[start:stop] = True
        start = stop
    return result


def prefix_outcome(
    rows: Sequence[Mapping[str, Any]],
    *,
    start_progress: float,
    horizon: int,
) -> dict[str, Any]:
    """Construct one bounded CARE outcome from only the requested prefix.

    Unlike the original DuoBench adapter, terminal/safety values are never
    copied from a later 64-step rollout into an 8/16/32-step label.
    """

    selected = _prefix_rows(rows, horizon)
    observed = min(len(rows), int(horizon))
    progress = np.asarray([_scalar(row.get("progress", 0.0)) for row in selected], dtype=np.float64)
    success = np.asarray([_bool(row.get("success", False)) for row in selected], dtype=bool)
    collision = np.asarray([_bool(row.get("collision_or_drop", row.get("hard_safety_violation", False))) for row in selected], dtype=bool)
    conflict = np.asarray([_bool(row.get("robot_conflict", False)) for row in selected], dtype=bool)
    duplicate = np.asarray([_bool(row.get("duplicate_work", False)) for row in selected], dtype=bool)
    active_values = []
    for row in selected:
        active = row.get("active", (False, False))
        a = np.asarray(active, dtype=np.float64).reshape(-1)
        active_values.append(np.pad(a[:2], (0, max(0, 2 - len(a)))))
    active_fraction = np.asarray(active_values, dtype=np.float64).mean(0) if active_values else np.zeros(2)
    first_success = next((index + 1 for index, value in enumerate(success) if value), None)
    deadlock = _deadlock_mask(selected)
    vector = np.asarray(
        (
            np.clip(float(progress[-1]) - float(start_progress), -1.0, 1.0),
            float(first_success is not None),
            -float(collision.mean()),
            -float(conflict.mean()),
            -float(duplicate.mean()),
            -float(deadlock.mean()),
            -float(abs(active_fraction[0] - active_fraction[1])),
            -float((first_success if first_success is not None else int(horizon)) / max(int(horizon), 1)),
        ),
        dtype=np.float32,
    )
    hard_safety = bool(collision.any())
    utility = float(np.dot(MAIN_WEIGHTS, vector.astype(np.float64)) - 3.0 * hard_safety)
    return {
        "requested_steps": int(horizon),
        "observed_steps": int(observed),
        "bounded_utility_vector": vector.tolist(),
        "utility_main": utility,
        "hard_safety_violation": hard_safety,
        "first_success_step": first_success,
        "progress": float(progress[-1]),
        "success": bool(success[-1]),
        "active_fraction": active_fraction.tolist(),
    }


def outcome_table(
    rows: Sequence[Mapping[str, Any]], *, start_progress: float, horizons: Iterable[int] = HORIZONS
) -> dict[str, dict[str, Any]]:
    return {str(int(horizon)): prefix_outcome(rows, start_progress=start_progress, horizon=int(horizon)) for horizon in horizons}


def stable_tree_hash(value: Any) -> str:
    """Hash nested simulator state/observation values independent of dict order."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if hasattr(item, "detach"):
            item = item.detach().cpu().numpy()
        if isinstance(item, Mapping):
            digest.update(b"map{")
            for key in sorted(item, key=str):
                key_bytes = str(key).encode("utf-8")
                digest.update(f"key:{len(key_bytes)}:".encode())
                digest.update(key_bytes)
                digest.update(b"=")
                update(item[key])
            digest.update(b"}")
        elif isinstance(item, (tuple, list)):
            digest.update(f"seq:{len(item)}[".encode())
            for child in item: update(child)
            digest.update(b"]")
        elif item is None:
            digest.update(b"none")
        elif isinstance(item, (bool, np.bool_)):
            digest.update(b"bool:1" if bool(item) else b"bool:0")
        elif isinstance(item, (int, np.integer)):
            # NumPy represents integers outside the fixed-width range as an
            # object scalar.  Hashing an object array's raw bytes hashes its
            # process-local pointer, not the integer value.  PCG64 RNG states
            # routinely contain such 128-bit Python integers.
            encoded = str(int(item)).encode("ascii")
            digest.update(f"int:{len(encoded)}:".encode())
            digest.update(encoded)
        elif isinstance(item, str):
            encoded = item.encode("utf-8")
            digest.update(f"str:{len(encoded)}:".encode())
            digest.update(encoded)
        elif isinstance(item, (bytes, bytearray, memoryview)):
            encoded = bytes(item)
            digest.update(f"bytes:{len(encoded)}:".encode())
            digest.update(encoded)
        else:
            try:
                arr = np.ascontiguousarray(item)
                digest.update(f"array:{arr.dtype}:{arr.shape}:".encode())
                if arr.dtype.hasobject:
                    # Recurse over values so no PyObject addresses enter the
                    # digest.  Shape and order remain part of the contract.
                    for child in arr.flat:
                        update(child)
                else:
                    digest.update(arr.tobytes())
            except Exception:
                encoded = repr(item).encode("utf-8")
                digest.update(f"repr:{type(item).__qualname__}:{len(encoded)}:".encode())
                digest.update(encoded)
    update(value)
    return digest.hexdigest()


@dataclass(frozen=True)
class BranchGate:
    """Structural and causal checks applied before a family is trainable."""

    require_candidate0_trace_match: bool = True
    require_distinct_repeats: bool = True
    min_nonzero_total_labels: int = 1

    def check(self, family: Mapping[str, Any]) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        branches = list(family.get("branches", ()))
        expected = {(c, regime, repeat) for c in range(BRANCH_CANDIDATES) for regime in BRANCH_REGIMES for repeat in BRANCH_REPEATS}
        seen = {(int(row.get("candidate_id", -1)), str(row.get("regime")), int(row.get("repeat_id", -1))) for row in branches}
        if len(branches) != 24 or seen != expected:
            errors.append("branch_key_contract")
        if self.require_distinct_repeats:
            seeds = [int(row.get("branch_seed", -1)) for row in branches if int(row.get("candidate_id", -1)) == 0 and str(row.get("regime")) == "reactive"]
            if len(seeds) >= 2 and len(set(seeds)) != len(seeds):
                errors.append("repeat_seeds_not_distinct")
        if self.require_candidate0_trace_match:
            by = {(int(row.get("candidate_id", -1)), str(row.get("regime")), int(row.get("repeat_id", -1))): row for row in branches}
            for repeat in BRANCH_REPEATS:
                left = by.get((0, "reactive", repeat), {}).get("action_trace_sha256")
                right = by.get((0, "replay", repeat), {}).get("action_trace_sha256")
                if left and right and left != right:
                    errors.append(f"candidate0_trace_mismatch_repeat{repeat}")
        nonzero = 0
        for row in branches:
            for outcome in row.get("outcomes", {}).values():
                vector = np.asarray(outcome.get("bounded_utility_vector", ()), dtype=np.float64)
                if vector.shape != (8,) or not np.isfinite(vector).all():
                    errors.append("outcome_vector_contract")
                if len(vector) and np.any(np.abs(vector) > 1e-7):
                    nonzero += 1
        if nonzero < self.min_nonzero_total_labels:
            warnings.append("family_outcomes_all_zero")
        return {
            "status": "PASSED" if not errors else "FAILED",
            "errors": sorted(set(errors)),
            "warnings": sorted(set(warnings)),
            "branch_count": len(branches),
            "nonzero_outcome_vectors": int(nonzero),
        }


__all__ = [
    "BRANCH_CANDIDATES",
    "BRANCH_REGIMES",
    "BRANCH_REPEATS",
    "BranchGate",
    "HORIZONS",
    "derive_branch_seed",
    "outcome_table",
    "prefix_outcome",
    "stable_tree_hash",
    "stratified_anchor_steps",
]
