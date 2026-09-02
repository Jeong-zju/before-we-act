"""Deterministic event-aware anchor sampling for a CARE ablation.

The formal DuoBench protocol intentionally uses ``stratified_anchor_steps``.
This module is a separate, data-only ablation: it changes only which anchor
indices are proposed, never the branch kernel, candidates, seeds, or labels.
Event scores must be computed from the frozen candidate-zero rollout *before*
any counterfactual branch is executed.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from deployment.duo_care.branch_signal import stratified_anchor_steps


def _valid_limit(episode_length: int, max_steps: int, horizon: int) -> int:
    length = int(episode_length)
    limit = int(max_steps)
    if length <= int(horizon) + 1 or limit <= int(horizon):
        raise ValueError("episode is too short for CARE event-aware anchors")
    return max(1, min(length - int(horizon), limit - int(horizon)))


def _event_score(row: Any) -> float:
    """Convert a scalar or metadata row into a finite, non-negative score."""

    if isinstance(row, Mapping):
        # These names are intentionally generic and benchmark-independent.
        values = [
            row.get("event_score", 0.0),
            row.get("stage_change", 0.0),
            row.get("contact_change", 0.0),
            row.get("action_delta", 0.0),
            row.get("belief_surprise", 0.0),
        ]
        value = max((abs(float(x)) for x in values), default=0.0)
    else:
        value = abs(float(row))
    return value if np.isfinite(value) else 0.0


def event_aware_hybrid_anchor_steps(
    episode_length: int,
    *,
    max_steps: int,
    event_scores: Sequence[float | Mapping[str, Any]],
    uncertainty_scores: Sequence[float | Mapping[str, Any]] | None = None,
    count: int = 30,
    horizon: int = 64,
    event_count: int | None = None,
    uncertainty_count: int | None = None,
    min_gap: int = 8,
) -> tuple[dict[str, Any], ...]:
    """Return a fixed 1/3 event, 1/3 uncertainty, 1/3 uniform mix.

    ``event_scores[t]`` describes the candidate-zero transition ending at
    step ``t``.  Selection uses only indices up to the non-terminal limit and
    deterministic tie-breaking (score descending, then index ascending).
    Event/uncertainty points are selected with a minimum spacing; if too few
    distinct points exist, the remaining quota is filled deterministically. The
    returned metadata records the policy and score, making the ablation
    auditable and preventing accidental use as the formal protocol.
    """

    total = int(count)
    if total < 3:
        raise ValueError("event-aware sampling requires at least three anchors")
    scores = np.asarray([_event_score(row) for row in event_scores], dtype=np.float64)
    if scores.ndim != 1 or scores.size < 1:
        raise ValueError("event_scores must be a non-empty vector")
    limit = min(_valid_limit(episode_length, max_steps, horizon), int(scores.size))
    uncertainty_source = event_scores if uncertainty_scores is None else uncertainty_scores
    uncertainty = np.asarray(
        [_event_score(row) for row in uncertainty_source], dtype=np.float64
    )
    if uncertainty.ndim != 1 or uncertainty.size < limit:
        raise ValueError(f"uncertainty_scores must contain at least {limit} entries")
    event_quota = int(event_count if event_count is not None else total // 3)
    event_quota = min(max(event_quota, 1), total - 2)
    uncertainty_quota = int(
        uncertainty_count if uncertainty_count is not None else total // 3
    )
    uncertainty_quota = min(max(uncertainty_quota, 1), total - event_quota - 1)
    uniform_quota = total - event_quota - uncertainty_quota

    ranked = sorted(
        (float(scores[index - 1]), index)
        for index in range(1, limit + 1)
    )
    ranked.reverse()  # score descending; index tie-break fixed below
    ranked.sort(key=lambda pair: (-pair[0], pair[1]))
    event_selected: list[int] = []
    for score, index in ranked:
        if len(event_selected) >= event_quota:
            break
        if all(abs(index - other) >= int(min_gap) for other in event_selected):
            event_selected.append(index)
    event_selected.sort()

    # Fill missing event quota from the highest-ranked unused locations.
    for _score, index in ranked:
        if len(event_selected) >= event_quota:
            break
        if index not in event_selected:
            event_selected.append(index)
    event_selected.sort()

    uncertainty_ranked = sorted(
        ((float(uncertainty[index - 1]), index) for index in range(1, limit + 1)),
        key=lambda pair: (-pair[0], pair[1]),
    )
    uncertainty_selected: list[int] = []
    for _score, index in uncertainty_ranked:
        if len(uncertainty_selected) >= uncertainty_quota:
            break
        if index in event_selected:
            continue
        if all(abs(index - other) >= int(min_gap) for other in uncertainty_selected):
            uncertainty_selected.append(index)
    for _score, index in uncertainty_ranked:
        if len(uncertainty_selected) >= uncertainty_quota:
            break
        if index not in event_selected and index not in uncertainty_selected:
            uncertainty_selected.append(index)
    uncertainty_selected.sort()

    fixed = stratified_anchor_steps(
        limit + int(horizon),
        max_steps=limit + int(horizon),
        count=uniform_quota,
        horizon=horizon,
        critical_count=0,
    )
    rows: list[dict[str, Any]] = []
    ordinal = 0
    for index in event_selected:
        rows.append(
            {
                "ordinal": ordinal,
                "anchor_step": int(index),
                "sampling_stratum": "event",
                "event_score": float(scores[index - 1]),
                "sampling_policy": "event_aware_hybrid_v1",
            }
        )
        ordinal += 1
    for index in uncertainty_selected:
        rows.append(
            {
                "ordinal": ordinal,
                "anchor_step": int(index),
                "sampling_stratum": "uncertainty",
                "uncertainty_score": float(uncertainty[index - 1]),
                "sampling_policy": "event_aware_hybrid_v1",
            }
        )
        ordinal += 1
    for row in fixed:
        item = dict(row)
        item["ordinal"] = ordinal
        item["sampling_policy"] = "event_aware_hybrid_v1"
        rows.append(item)
        ordinal += 1
    # Keep stratum order (event, uncertainty, uniform) stable so an ordinal
    # pre-registers its stratum even though each branch seed has its own trace.
    dedup: dict[int, dict[str, Any]] = {}
    for row in rows:
        dedup.setdefault(int(row["anchor_step"]), row)
    rows = list(dedup.values())
    if len(rows) < total:
        # Deterministic fallback over the complete legal range.  This branch
        # is normally only hit when an event point overlaps a stratified point.
        for step in np.linspace(1, limit, num=max(total * 4, 32), dtype=int):
            step = int(step)
            if step in dedup:
                continue
            item = {
                "ordinal": len(rows),
                "anchor_step": step,
                "sampling_stratum": "uniform_fallback",
                "event_score": float(scores[step - 1]),
                "sampling_policy": "event_aware_hybrid_v1",
            }
            rows.append(item)
            dedup[step] = item
            if len(rows) == total:
                break
    if len(rows) != total:
        raise RuntimeError(f"event-aware sampler produced {len(rows)} anchors, expected {total}")
    for index, row in enumerate(rows):
        row["ordinal"] = index
    return tuple(rows)


def compare_signal_reports(
    fixed: Iterable[Mapping[str, Any]], hybrid: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Compare family-level signal reports without selecting a winner."""

    def summarize(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        values = list(rows)
        if not values:
            raise ValueError("signal report is empty")
        families = len(values)
        h16 = [row.get("16", {}) for row in values]
        effective = sum(bool(row.get("families_with_signal", 0)) for row in h16)
        nonzero = sum(int(row.get("nonzero_candidate_advantages", 0)) for row in h16)
        pairwise = sum(int(row.get("pairwise_non_ties", 0)) for row in h16)
        return {
            "families": families,
            "effective_family_fraction_h16": effective / families,
            "nonzero_candidate_advantages_h16": nonzero,
            "pairwise_non_ties_h16": pairwise,
        }

    fixed_summary = summarize(fixed)
    hybrid_summary = summarize(hybrid)
    return {
        "protocol": "event_aware_hybrid_ablation_v1",
        "formal_protocol_unchanged": True,
        "fixed": fixed_summary,
        "hybrid": hybrid_summary,
        "delta": {
            key: hybrid_summary[key] - fixed_summary[key]
            for key in fixed_summary
            if key != "families"
        },
    }


__all__ = ["event_aware_hybrid_anchor_steps", "compare_signal_reports"]
