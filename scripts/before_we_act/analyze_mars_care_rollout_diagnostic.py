#!/usr/bin/env python3
"""Summarize observer-only MARS CARE rollout telemetry into failure evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


FORMAT_VERSION = "before-we-act.care-mars-rollout-failure-audit/1"
TASKS = (
    "place_cube_in_cup",
    "strike_cube_hard",
    "three_robots_place_shoes",
    "four_robots_stack_cube",
)
MODES = ("selector_off", "care")


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if value.get("type") == "step":
                yield value


def _finite(values: Iterable[Any]) -> list[float]:
    result = []
    for value in values:
        if value is not None and math.isfinite(float(value)):
            result.append(float(value))
    return result


def _quantiles(values: Iterable[Any]) -> dict[str, float | None]:
    finite = _finite(values)
    if not finite:
        return {"min": None, "median": None, "p95": None, "max": None}
    return {
        "min": min(finite),
        "median": float(np.quantile(finite, 0.5)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": max(finite),
    }


def summarize_episode(path: Path) -> dict[str, Any]:
    stage_counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    first_events: dict[str, int] = {}
    progress: list[float] = []
    gate: list[float] = []
    reliability: list[float] = []
    sigma: list[float] = []
    belief_events: list[float] = []
    best_lower: list[float] = []
    legal = total_candidates = 0
    unsafe = illegal = clipped = 0
    proposed = applied = 0
    candidate_reference_l2: list[float] = []
    action_reference_l2: list[float] = []
    action_step_l2: list[float] = []
    previous_actions: dict[str, np.ndarray] = {}
    step_count = 0
    last: dict[str, Any] | None = None

    for row in rows(path):
        last = row
        step = int(row["step"])
        step_count += 1
        metrics = row["privileged_metrics"]
        stage_counts[str(metrics["stage_id"])] += 1
        progress.append(float(metrics["progress"]))
        for name, active in row.get("events", {}).items():
            if active is True and name not in {"strike_proxy_is_not_contact"}:
                first_events.setdefault(str(name), step)
        diag = row.get("care_diagnostics", {})
        gate.extend(_finite((diag.get("gate"),)))
        reliability.extend(_finite((diag.get("reliability"),)))
        sigma.extend(_finite((diag.get("sigma"),)))
        belief_events.extend(_finite((diag.get("events"),)))
        reasons.update(str(value) for value in row.get("selection_reason", ()))
        best_lower.extend(_finite(row.get("best_lower", ())))
        legality = np.asarray(row.get("candidate_legality", ()), dtype=bool)
        legal += int(legality.sum())
        total_candidates += int(legality.size)
        illegal += int(np.asarray(row.get("illegal_mask", ()), dtype=bool).sum())
        unsafe += int(np.asarray(row.get("learned_unsafe_mask", ()), dtype=bool).sum())
        assembly = row.get("assembly", {})
        proposed += len(assembly.get("proposed_rows", ()))
        applied += len(assembly.get("applied_rows", ()))
        clipped += sum(
            int(item.get("elements_clipped", 0))
            for item in row.get("action_bounds", {}).values()
        )

        reference = row.get("reference_first_action", {})
        applied_actions = row.get("action_applied", {})
        for arm, (candidate_rows, reference_key) in enumerate(
            zip(row.get("candidate_first_actions", ()), sorted(reference))
        ):
            del arm
            reference_action = np.asarray(reference[reference_key], dtype=np.float64)
            for candidate in np.asarray(candidate_rows, dtype=np.float64)[1:]:
                candidate_reference_l2.append(float(np.linalg.norm(candidate - reference_action)))
            actual = np.asarray(applied_actions[reference_key], dtype=np.float64)
            action_reference_l2.append(float(np.linalg.norm(actual - reference_action)))
            if reference_key in previous_actions:
                action_step_l2.append(float(np.linalg.norm(actual - previous_actions[reference_key])))
            previous_actions[reference_key] = actual

    if last is None:
        raise ValueError(f"rollout telemetry contains no steps: {path}")
    return {
        "telemetry": str(path.resolve()),
        "steps": step_count,
        "success": bool(last["privileged_metrics"].get("success", False)),
        "final_stage": str(last["privileged_metrics"]["stage_id"]),
        "final_progress": float(last["privileged_metrics"]["progress"]),
        "maximum_progress": max(progress),
        "stage_counts": dict(stage_counts),
        "first_event_steps": first_events,
        "selection_reason_counts": dict(reasons),
        "proposed_overrides": proposed,
        "applied_overrides": applied,
        "candidate_legal_fraction": legal / max(total_candidates, 1),
        "illegal_mask_count": illegal,
        "learned_unsafe_count": unsafe,
        "action_elements_clipped": clipped,
        "candidate_first_action_l2_from_reference": _quantiles(candidate_reference_l2),
        "applied_action_l2_from_reference": _quantiles(action_reference_l2),
        "applied_action_step_l2": _quantiles(action_step_l2),
        "best_lower": _quantiles(best_lower),
        "belief_gate": _quantiles(gate),
        "belief_reliability": _quantiles(reliability),
        "belief_sigma": _quantiles(sigma),
        "belief_event_count": _quantiles(belief_events),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_summary = json.loads((args.root / "summary.json").read_text())
    episodes: dict[str, Any] = {}
    for task in TASKS:
        episodes[task] = {}
        for mode in MODES:
            parent = args.root / "records" / task / mode
            paths = sorted(parent.glob("seed_*/telemetry.jsonl"))
            if len(paths) != 1:
                raise RuntimeError(f"expected one {task}/{mode} telemetry, got {paths}")
            episodes[task][mode] = summarize_episode(paths[0])
        episodes[task]["paired_action_trace_equal"] = bool(
            source_summary["tasks"][task]["paired_action_trace_equal"]
        )

    result = {
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "observer_only": True,
        "privileged_state_returned_to_policy": False,
        "source_summary": str((args.root / "summary.json").resolve()),
        "all_paired_action_traces_equal": bool(source_summary["all_paired_action_traces_equal"]),
        "care_overrides": int(source_summary["care_overrides"]),
        "episodes": episodes,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
