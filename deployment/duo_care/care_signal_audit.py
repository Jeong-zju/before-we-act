"""Audits for non-degenerate DuoBench CARE branch signals.

The audit is intentionally independent of a particular reference policy.  It
checks the *contract* between a proposal provider, the frozen branch labels,
and the belief head.  A policy may be weak, but a formal CARE run must not
silently turn into fitting an all-zero target or an all-safe classifier.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from deployment.duo_dino_reference.bcore_data import (
    DUO_CARE_MEMORY_TOKENS,
    DUO_CARE_MEMORY_WIDTH,
)


HORIZONS = (8, 16, 32, 64)
CANDIDATES = 6
REPEATS = 2
COMPONENTS = 3


def _nonzero(value: np.ndarray, tol: float) -> np.ndarray:
    return np.abs(np.asarray(value, dtype=np.float64)) > float(tol)


def _family_axis(prepared: Mapping[str, Any], key: str, n: int) -> np.ndarray:
    value = prepared.get(key)
    if value is None:
        return np.zeros(n, dtype=np.int64)
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.shape != (n,):
        raise ValueError(f"prepared {key} must be [{n}], got {value.shape}")
    return value.astype(np.int64, copy=False)


def _pairwise_count(target: np.ndarray, tol: float) -> int:
    """Count non-tied candidate pairs over families/repeats/components."""

    # target is [N, C, R, K] after selecting one horizon.
    delta = target[:, :, None, :, :] - target[:, None, :, :, :]
    # Comparing candidate axes leaves [N,C,C,R,K].  Only non-reference
    # candidates are used by CARE's scorer, but include all pairs in the
    # diagnostic so a broken reference baseline is visible too.
    return int(np.count_nonzero(np.abs(delta) > float(tol)) // 2)


def audit_prepared(
    path: str | Path,
    *,
    tolerance: float = 1e-7,
    require_nonzero_horizons: Iterable[int] = HORIZONS,
    strict: bool = True,
) -> dict[str, Any]:
    """Audit a prepared CARE tensor and return a JSON-safe report.

    ``strict=True`` marks zero-signal horizons as errors.  This is suitable
    for formal preparation; smoke fixtures can call with ``strict=False``.
    """

    raw = torch.load(path, map_location="cpu", weights_only=False)
    required = ("memory", "memory_mask", "candidate_chunks", "targets", "hard_safety", "usable")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"prepared CARE artifact is missing {missing}")
    targets = raw["targets"].detach().cpu().numpy() if torch.is_tensor(raw["targets"]) else np.asarray(raw["targets"])
    chunks = raw["candidate_chunks"].detach().cpu().numpy() if torch.is_tensor(raw["candidate_chunks"]) else np.asarray(raw["candidate_chunks"])
    safety = raw["hard_safety"].detach().cpu().numpy() if torch.is_tensor(raw["hard_safety"]) else np.asarray(raw["hard_safety"])
    memory = raw["memory"].detach().cpu().numpy() if torch.is_tensor(raw["memory"]) else np.asarray(raw["memory"])
    mask = raw["memory_mask"].detach().cpu().numpy() if torch.is_tensor(raw["memory_mask"]) else np.asarray(raw["memory_mask"])
    usable = raw["usable"].detach().cpu().numpy() if torch.is_tensor(raw["usable"]) else np.asarray(raw["usable"])
    errors: list[str] = []
    warnings: list[str] = []
    expected = {
        "memory": (None, DUO_CARE_MEMORY_TOKENS, DUO_CARE_MEMORY_WIDTH),
        "memory_mask": (None, DUO_CARE_MEMORY_TOKENS),
        "candidate_chunks": (None, CANDIDATES, 100, 8),
        "targets": (None, len(HORIZONS), CANDIDATES, REPEATS, COMPONENTS),
        "hard_safety": (None, len(HORIZONS), CANDIDATES, REPEATS),
        "usable": (None, len(HORIZONS)),
    }
    values = {"memory": memory, "memory_mask": mask, "candidate_chunks": chunks, "targets": targets, "hard_safety": safety, "usable": usable}
    n = None
    for key, shape in expected.items():
        actual = values[key].shape
        if len(actual) != len(shape) or any(w is not None and int(a) != int(w) for a, w in zip(actual, shape)):
            errors.append(f"{key}_shape:{actual}!={shape}")
        elif n is None:
            n = int(actual[0])
        elif int(actual[0]) != n:
            errors.append(f"{key}_family_count:{actual[0]}!={n}")
    if n is None:
        n = 0
    if not np.isfinite(targets).all():
        errors.append("targets_nonfinite")
    if not np.isfinite(chunks).all():
        errors.append("candidate_chunks_nonfinite")
    if not np.isfinite(memory).all():
        errors.append("memory_nonfinite")
    if not np.isfinite(safety).all():
        errors.append("hard_safety_nonfinite")
    split = _family_axis(raw, "split_id", n)
    task = _family_axis(raw, "task_id", n)
    nonzero = _nonzero(targets, tolerance)
    horizon_report: dict[str, Any] = {}
    for hi, horizon in enumerate(HORIZONS):
        usable_count = int(np.count_nonzero(usable[:, hi])) if usable.ndim == 2 and usable.shape[0] == n else 0
        # Candidate zero is definitionally zero.  The useful signal is the
        # non-reference candidates' total-utility labels.
        nz = nonzero[:, hi, 1:, :, 2] if targets.ndim == 5 and targets.shape[1] > hi else np.zeros((n, 5, 2), bool)
        pair_count = _pairwise_count(targets[:, hi], tolerance) if targets.ndim == 5 and targets.shape[1] > hi else 0
        row = {
            "horizon": horizon,
            "usable_families": usable_count,
            "nonzero_nonreference_total_labels": int(np.count_nonzero(nz)),
            "nonzero_family_fraction": float(np.mean(np.any(nz, axis=(1, 2)))) if n else 0.0,
            "pairwise_non_ties": pair_count,
        }
        horizon_report[str(horizon)] = row
        if horizon in tuple(require_nonzero_horizons) and usable_count and row["nonzero_nonreference_total_labels"] == 0:
            (errors if strict else warnings).append(f"horizon_{horizon}_all_zero_total_labels")
        if horizon in tuple(require_nonzero_horizons) and usable_count and pair_count == 0:
            (errors if strict else warnings).append(f"horizon_{horizon}_no_pairwise_signal")

    # Candidate diversity is measured after action normalization.  A branch
    # family can legitimately have a near-identical candidate in one state,
    # but an entire corpus of identical candidates indicates a broken adapter.
    scales = raw.get("action_std", torch.ones(8))
    scales = scales.detach().cpu().numpy() if torch.is_tensor(scales) else np.asarray(scales)
    scales = np.maximum(np.asarray(scales, dtype=np.float64), 1e-6)
    pair_rms: dict[str, float] = {}
    if chunks.ndim == 4 and chunks.shape[1] == CANDIDATES:
        for left in range(CANDIDATES):
            for right in range(left + 1, CANDIDATES):
                value = (chunks[:, left] - chunks[:, right]) / scales[None, None, :]
                pair_rms[f"{left}-{right}"] = float(np.sqrt(np.mean(value * value)))
    collapsed_pairs = [key for key, value in pair_rms.items() if value <= tolerance]
    if collapsed_pairs:
        warnings.append(f"candidate_pairs_collapsed:{','.join(collapsed_pairs)}")
    safety_positive = float(np.mean(safety > 0.5)) if safety.size else 0.0
    if safety.size and safety_positive == 0.0:
        warnings.append("hard_safety_labels_all_zero")
    if safety.size and safety_positive == 1.0:
        warnings.append("hard_safety_labels_all_one")
    split_report: dict[str, Any] = {}
    for value in sorted(set(split.tolist())):
        indices = split == value
        split_report[str(int(value))] = {
            "families": int(np.count_nonzero(indices)),
            "nonzero_total_labels_h16": int(np.count_nonzero(nonzero[indices, 1, :, :, 2])) if targets.ndim == 5 else 0,
            "pairwise_non_ties_h16": _pairwise_count(targets[indices, 1], tolerance) if targets.ndim == 5 and np.count_nonzero(indices) else 0,
        }
    report = {
        "status": "PASSED" if not errors else "FAILED",
        "path": str(Path(path).resolve()),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "families": n,
        "errors": errors,
        "warnings": warnings,
        "horizons": horizon_report,
        "candidate_pair_normalized_rms": pair_rms,
        "hard_safety_positive_fraction": safety_positive,
        "split": split_report,
        "task_count": int(len(set(task.tolist()))) if n else 0,
    }
    return report


def audit_family_json(
    path: str | Path,
    *,
    tolerance: float = 1e-6,
    strict: bool = True,
) -> dict[str, Any]:
    """Audit one raw branch-family JSON before labels are frozen."""

    family = json.loads(Path(path).read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []
    branches = list(family.get("branches", ()))
    if int(family.get("branch_count", -1)) != 24 or len(branches) != 24:
        errors.append("branch_count_not_24")
    keys = [(int(row.get("candidate_id", -1)), str(row.get("regime")), int(row.get("repeat_id", -1))) for row in branches]
    counts = Counter(keys)
    expected = {(candidate, regime, repeat) for candidate in range(CANDIDATES) for regime in ("reactive", "replay") for repeat in range(REPEATS)}
    missing = sorted(expected - set(counts))
    duplicate = sorted(key for key, count in counts.items() if count != 1)
    if missing:
        errors.append(f"missing_branch_keys:{missing}")
    if duplicate:
        errors.append(f"duplicate_branch_keys:{duplicate}")
    seeds = {(str(row.get("regime")), int(row.get("repeat_id", -1))): int(row.get("branch_seed", -1)) for row in branches}
    if len({seeds.get((regime, repeat), -1) for regime in ("reactive", "replay") for repeat in range(REPEATS)}) < REPEATS:
        warnings.append("repeat_branch_seeds_not_distinct")
    for row in branches:
        outcomes = row.get("outcomes", {})
        for horizon in HORIZONS:
            value = outcomes.get(str(horizon))
            if not isinstance(value, Mapping):
                errors.append(f"missing_outcome:{row.get('candidate_id')}/{row.get('regime')}/{row.get('repeat_id')}/{horizon}")
                continue
            vector = np.asarray(value.get("bounded_utility_vector", ()), dtype=np.float64)
            if vector.shape != (8,) or not np.isfinite(vector).all():
                errors.append(f"bad_utility_vector:{row.get('candidate_id')}/{horizon}")
    # If traces are available, candidate 0 must be reproducible between the
    # reactive and replay counterfactuals.  A mismatch means the two branches
    # did not start from the same simulator/runtime state.
    by_key = {(int(row.get("candidate_id", -1)), str(row.get("regime")), int(row.get("repeat_id", -1))): row for row in branches}
    for repeat in range(REPEATS):
        a = by_key.get((0, "reactive", repeat), {}).get("action_trace_sha256")
        b = by_key.get((0, "replay", repeat), {}).get("action_trace_sha256")
        if a and b and a != b:
            (errors if strict else warnings).append(f"candidate0_reactive_replay_trace_mismatch_repeat{repeat}")
    report = {
        "status": "PASSED" if not errors else "FAILED",
        "path": str(Path(path).resolve()),
        "snapshot_id": family.get("snapshot_id"),
        "task": family.get("task"),
        "errors": errors,
        "warnings": warnings,
        "branch_count": len(branches),
        "unique_branch_keys": len(set(keys)),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data", type=Path)
    parser.add_argument("--family", type=Path, action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--non-strict", action="store_true")
    args = parser.parse_args()
    if args.prepared_data is None and not args.family:
        parser.error("provide --prepared-data and/or --family")
    result: dict[str, Any] = {"format_version": "before-we-act.duo-care-signal-audit/1"}
    reports: list[dict[str, Any]] = []
    if args.prepared_data is not None:
        reports.append(audit_prepared(args.prepared_data, tolerance=args.tolerance, strict=not args.non_strict))
    for path in args.family or ():
        reports.append(audit_family_json(path, tolerance=args.tolerance, strict=not args.non_strict))
    result["reports"] = reports
    result["status"] = "PASSED" if all(row["status"] == "PASSED" for row in reports) else "FAILED"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASSED" else 2)


if __name__ == "__main__":
    main()


__all__ = ["audit_family_json", "audit_prepared"]
