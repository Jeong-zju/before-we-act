"""Shared MARS-Control per-arm action contract.

The four MARS-Control environments expose an eight dimensional Panda
``pd_joint_pos`` action.  Historically the ACT adapter trained on the raw
planner trace while CARE clipped that trace before computing statistics and
targets.  That seemingly small difference changes both the normalization and
the learned policy.  This module is the single source of truth for the
physical action contract and is deliberately independent of either policy.

The contract is intentionally conservative:

* values are checked for finiteness and a trailing width of eight;
* canonicalization is an element-wise clip in physical ``pd_joint_pos``
  space, followed by a float32 cast;
* normalized actions are decoded, canonicalized, and encoded again with the
  same operation used by the environment adapter;
* checkpoint metadata and live action-space bounds can be checked fail-closed;
* audit helpers report how much of an otherwise immutable raw corpus would be
  changed.  The raw HDF5 files are never rewritten by this module.

No candidate, belief, scorer, or selector logic lives here.  Consequently,
using this contract cannot change CARE's theoretical pipeline; it only makes
the train/decode/runtime semantics reproducible across ACT and CARE.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np


ACTION_CONTRACT_VERSION = "before-we-act.mars-action-contract/1"
ACTION_DIM = 8
ACTION_HORIZON = 100
ACTION_ENCODING = "absolute_pd_joint_pos"

# ManiSkill/RoboFactory Panda ``pd_joint_pos`` limits.  Keep these literals in
# one place and expose read-only arrays so a caller cannot mutate process-wide
# policy semantics by accident.
ACTION_LOW = np.asarray(
    (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -1.0),
    dtype=np.float32,
)
ACTION_HIGH = np.asarray(
    (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1.0),
    dtype=np.float32,
)
ACTION_LOW.setflags(write=False)
ACTION_HIGH.setflags(write=False)

# Backwards-compatible names used by the first MARS CARE adapter.
PD_ACTION_LOW = ACTION_LOW
PD_ACTION_HIGH = ACTION_HIGH


def _validate_bounds() -> None:
    if ACTION_LOW.shape != (ACTION_DIM,) or ACTION_HIGH.shape != (ACTION_DIM,):
        raise RuntimeError("MARS action contract bounds have the wrong shape")
    if not np.isfinite(ACTION_LOW).all() or not np.isfinite(ACTION_HIGH).all():
        raise RuntimeError("MARS action contract bounds must be finite")
    if not np.all(ACTION_LOW < ACTION_HIGH):
        raise RuntimeError("MARS action contract has invalid bounds")


_validate_bounds()


def _as_numeric_array(value: Any, *, name: str = "action") -> np.ndarray:
    """Convert an action-like value to an array and enforce shape/finiteness."""

    try:
        array = np.asarray(value)
    except Exception as error:  # pragma: no cover - defensive object arrays
        raise ValueError(f"{name} cannot be converted to an array") from error
    if array.ndim == 0 or array.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"{name} must have trailing width {ACTION_DIM}, got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric, got dtype {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _stats_arrays(
    mean: Any, std: Any, *, name: str = "action normalization"
) -> tuple[np.ndarray, np.ndarray]:
    mean_array = np.asarray(mean, dtype=np.float32)
    std_array = np.asarray(std, dtype=np.float32)
    if mean_array.shape != (ACTION_DIM,) or std_array.shape != (ACTION_DIM,):
        raise ValueError(
            f"{name} mean/std must both have shape ({ACTION_DIM},), "
            f"got {mean_array.shape}/{std_array.shape}"
        )
    if not np.isfinite(mean_array).all() or not np.isfinite(std_array).all():
        raise ValueError(f"{name} mean/std must be finite")
    if np.any(std_array <= 0):
        raise ValueError(f"{name} standard deviation must be positive")
    return mean_array, std_array


def canonicalize_action(value: Any) -> np.ndarray:
    """Return the exact physical action sent to RoboFactory.

    ``value`` may have any leading dimensions, but its final dimension must be
    eight.  The result is always a float32 NumPy array.  Rejecting non-finite
    values instead of letting ``np.clip`` turn them into a seemingly legal
    command is intentional and is part of the fail-closed deployment policy.
    """

    array = _as_numeric_array(value)
    # Clip in the source precision first, then make the storage dtype explicit.
    # For float64 traces this preserves the comparison against the exact
    # float32 limits while ensuring every downstream consumer receives the same
    # representation.
    return np.clip(array, ACTION_LOW, ACTION_HIGH).astype(np.float32, copy=False)


def clip_pd_action(value: Any) -> np.ndarray:
    """Compatibility alias for older MARS adapters."""

    return canonicalize_action(value)


def canonicalize_normalized_action(
    normalized: Any, action_mean: Any, action_std: Any
) -> np.ndarray:
    """Decode, clip in physical space, and re-encode an action tensor.

    This is useful at the policy/runtime boundary: model outputs can be
    slightly outside the demonstrated distribution, but clipping must happen
    in physical joint space rather than in normalized coordinates.  Applying
    this function twice is idempotent up to float32 representation.
    """

    array = _as_numeric_array(normalized, name="normalized action")
    mean, std = _stats_arrays(action_mean, action_std)
    # Keep arithmetic in float32 to match the ACT and CARE inference paths.
    decoded = array.astype(np.float32, copy=False) * std + mean
    if not np.isfinite(decoded).all():
        raise ValueError("decoded action became non-finite")
    canonical = canonicalize_action(decoded)
    encoded = (canonical - mean) / std
    if not np.isfinite(encoded).all():
        raise ValueError("canonical normalized action became non-finite")
    return encoded.astype(np.float32, copy=False)


def canonicalize_torch_action(value: Any) -> Any:
    """Torch equivalent of :func:`canonicalize_action` without a hard import.

    The function preserves the input device and returns a tensor with the
    input floating dtype (or float32 for non-floating inputs).  It is kept
    separate from the NumPy API so data-audit code remains lightweight.
    """

    try:
        import torch
    except ImportError as error:  # pragma: no cover - torch is a project dep
        raise RuntimeError("canonicalize_torch_action requires torch") from error
    if not torch.is_tensor(value):
        raise ValueError("torch action must be a torch.Tensor")
    if value.ndim == 0 or value.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"torch action must have trailing width {ACTION_DIM}, got {tuple(value.shape)}"
        )
    if not torch.is_floating_point(value):
        value = value.float()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("torch action must contain only finite values")
    low = torch.as_tensor(ACTION_LOW, device=value.device, dtype=value.dtype)
    high = torch.as_tensor(ACTION_HIGH, device=value.device, dtype=value.dtype)
    return torch.clamp(value, min=low, max=high)


def canonicalize_normalized_torch_action(
    normalized: Any, action_mean: Any, action_std: Any
) -> Any:
    """Torch equivalent of :func:`canonicalize_normalized_action`."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("canonicalize_normalized_torch_action requires torch") from error
    if not torch.is_tensor(normalized):
        raise ValueError("normalized action must be a torch.Tensor")
    if normalized.ndim == 0 or normalized.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"normalized action must have trailing width {ACTION_DIM}, "
            f"got {tuple(normalized.shape)}"
        )
    if not bool(torch.isfinite(normalized).all()):
        raise ValueError("normalized action must be finite")
    mean = torch.as_tensor(action_mean, device=normalized.device, dtype=normalized.dtype)
    std = torch.as_tensor(action_std, device=normalized.device, dtype=normalized.dtype)
    if mean.shape != (ACTION_DIM,) or std.shape != (ACTION_DIM,):
        raise ValueError("torch action normalization mean/std must have width 8")
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
        raise ValueError("torch action normalization mean/std must be finite")
    if bool((std <= 0).any()):
        raise ValueError("torch action normalization standard deviation must be positive")
    decoded = normalized * std + mean
    if not bool(torch.isfinite(decoded).all()):
        raise ValueError("decoded torch action became non-finite")
    low = torch.as_tensor(ACTION_LOW, device=normalized.device, dtype=normalized.dtype)
    high = torch.as_tensor(ACTION_HIGH, device=normalized.device, dtype=normalized.dtype)
    return (torch.clamp(decoded, min=low, max=high) - mean) / std


def validate_action_space_bounds(space: Any) -> bool:
    """Require a live Gym action space to match the static contract exactly."""

    if not hasattr(space, "low") or not hasattr(space, "high"):
        raise ValueError("action space must expose low and high bounds")
    try:
        low = np.asarray(space.low, dtype=np.float32)
        high = np.asarray(space.high, dtype=np.float32)
    except Exception as error:
        raise ValueError("action-space bounds are not numeric") from error
    if low.shape != (ACTION_DIM,) or high.shape != (ACTION_DIM,):
        raise ValueError(
            f"action-space bounds must have shape ({ACTION_DIM},), "
            f"got {low.shape}/{high.shape}"
        )
    if not np.array_equal(low, ACTION_LOW) or not np.array_equal(high, ACTION_HIGH):
        raise ValueError(
            "live action-space bounds differ from the MARS action contract"
        )
    return True


def _contract_payload() -> dict[str, Any]:
    # Include byte-level encodings in the hashed payload so JSON float printer
    # changes cannot silently alter the identity of this contract.
    low_le = np.asarray(ACTION_LOW, dtype="<f4")
    high_le = np.asarray(ACTION_HIGH, dtype="<f4")
    return {
        "version": ACTION_CONTRACT_VERSION,
        "action_dim": ACTION_DIM,
        "horizon": ACTION_HORIZON,
        "encoding": ACTION_ENCODING,
        "low": ACTION_LOW.tolist(),
        "high": ACTION_HIGH.tolist(),
        "low_float32_hex": low_le.tobytes().hex(),
        "high_float32_hex": high_le.tobytes().hex(),
        "canonicalization": "finite_check_then_elementwise_clip_physical_float32",
    }


def action_contract_hash() -> str:
    payload = json.dumps(
        _contract_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def contract_metadata() -> dict[str, Any]:
    """Return JSON-safe static metadata suitable for receipts/checkpoints."""

    payload = _contract_payload()
    payload["sha256"] = action_contract_hash()
    payload["bounds_dtype"] = "float32"
    return payload


def checkpoint_action_contract(**extra: Any) -> dict[str, Any]:
    """Build checkpoint metadata, allowing only explicit audit annotations."""

    value = contract_metadata()
    if extra:
        value["annotations"] = dict(extra)
    return value


def validate_checkpoint_action_contract(
    checkpoint: Mapping[str, Any], *, require: bool = True
) -> dict[str, Any]:
    """Validate embedded action metadata and return a normalized copy.

    Older checkpoints intentionally fail closed when ``require`` is true.  A
    caller may set ``require=False`` only for a read-only migration audit; it
    then receives an empty dictionary for a missing field, never an implicit
    compatibility claim.
    """

    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    metadata = checkpoint.get("action_contract")
    if metadata is None:
        if require:
            raise ValueError("checkpoint is missing action_contract metadata")
        return {}
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint action_contract must be a mapping")
    expected = contract_metadata()
    if str(metadata.get("version")) != ACTION_CONTRACT_VERSION:
        raise ValueError("checkpoint action contract version differs")
    if str(metadata.get("sha256")) != expected["sha256"]:
        raise ValueError("checkpoint action contract hash differs")
    if int(metadata.get("action_dim", -1)) != ACTION_DIM:
        raise ValueError("checkpoint action contract dimension differs")
    if str(metadata.get("encoding")) != ACTION_ENCODING:
        raise ValueError("checkpoint action encoding differs")
    # Validate bounds as well as the hash.  This catches hand-written metadata
    # that copied the hash but changed a visible field.
    for key, static in (("low", ACTION_LOW), ("high", ACTION_HIGH)):
        observed = np.asarray(metadata.get(key), dtype=np.float32)
        if observed.shape != static.shape or not np.array_equal(observed, static):
            raise ValueError(f"checkpoint action contract {key} bounds differ")
    return dict(metadata)


def validate_action_stats(
    stats: Mapping[str, Any], *, mean_key: str = "a_mean", std_key: str = "a_std"
) -> tuple[np.ndarray, np.ndarray]:
    """Validate action normalization vectors used by ACT/CARE checkpoints."""

    if not isinstance(stats, Mapping):
        raise ValueError("normalization stats must be a mapping")
    if mean_key not in stats or std_key not in stats:
        raise ValueError(f"normalization stats missing {mean_key}/{std_key}")
    return _stats_arrays(stats[mean_key], stats[std_key])


def normalization_stats_hash(
    stats: Mapping[str, Any], *, mean_key: str = "a_mean", std_key: str = "a_std"
) -> str:
    """Hash validated float32 action statistics for provenance receipts."""

    mean, std = validate_action_stats(stats, mean_key=mean_key, std_key=std_key)
    payload = (
        ACTION_CONTRACT_VERSION.encode("utf-8")
        + mean.astype(np.float32, copy=False).tobytes()
        + std.astype(np.float32, copy=False).tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def audit_action_array(value: Any, *, source: str | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Canonicalize an array and return counts useful for an immutable audit."""

    raw = _as_numeric_array(value)
    outside = (raw < ACTION_LOW) | (raw > ACTION_HIGH)
    canonical = canonicalize_action(raw)
    changed = int(np.count_nonzero(outside))
    if changed:
        delta = np.abs(
            canonical.astype(np.float64) - raw.astype(np.float64, copy=False)
        )
        max_abs_change = float(np.max(delta[outside]))
    else:
        max_abs_change = 0.0
    per_dimension = outside.reshape(-1, ACTION_DIM).sum(axis=0).astype(int).tolist()
    report: dict[str, Any] = {
        "contract_version": ACTION_CONTRACT_VERSION,
        "contract_sha256": action_contract_hash(),
        "raw_values": int(raw.size),
        "out_of_bounds_values": changed,
        "changed_values": changed,
        "out_of_bounds_fraction": float(changed / raw.size) if raw.size else 0.0,
        "max_abs_change": max_abs_change,
        "out_of_bounds_by_dimension": per_dimension,
    }
    if source is not None:
        report["source"] = str(source)
    return canonical, report


__all__ = [
    "ACTION_CONTRACT_VERSION",
    "ACTION_DIM",
    "ACTION_ENCODING",
    "ACTION_HIGH",
    "ACTION_HORIZON",
    "ACTION_LOW",
    "PD_ACTION_HIGH",
    "PD_ACTION_LOW",
    "action_contract_hash",
    "audit_action_array",
    "canonicalize_action",
    "canonicalize_normalized_action",
    "canonicalize_normalized_torch_action",
    "canonicalize_torch_action",
    "checkpoint_action_contract",
    "clip_pd_action",
    "contract_metadata",
    "normalization_stats_hash",
    "validate_action_space_bounds",
    "validate_action_stats",
    "validate_checkpoint_action_contract",
]
