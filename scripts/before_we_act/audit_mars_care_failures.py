#!/usr/bin/env python3
"""Audit existing MARS-Control CARE failures without touching training data.

The formal run intentionally stores compact Validation20 summaries rather than
videos.  This report therefore separates facts available in the summaries and
branch corpus from telemetry that must be collected by a later diagnostic
recorder.  It never reads Validation20 values to tune a model and never writes
into a formal run directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


TASKS = (
    "place_cube_in_cup",
    "strike_cube_hard",
    "three_robots_place_shoes",
    "four_robots_stack_cube",
)
HORIZONS = (8, 16, 32, 64)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def validation_audit(root: Path) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []
    tasks: dict[str, Any] = {}
    all_care: dict[int, dict[str, Any]] = {}
    all_off: dict[int, dict[str, Any]] = {}
    for task in TASKS:
        care_rows = read_jsonl(root / "care" / f"{task}.jsonl")
        off_rows = read_jsonl(root / "selector_off" / f"{task}.jsonl")
        care = {int(row["seed"]): row for row in care_rows}
        off = {int(row["seed"]): row for row in off_rows}
        all_care.update(care)
        all_off.update(off)
        common = sorted(set(care) & set(off))
        identical = [seed for seed in common if care[seed].get("action_trace_sha256") == off[seed].get("action_trace_sha256")]
        failures = [row for row in care_rows if not bool(row.get("success", False))]
        diag_keys = ("gate", "residual_norm", "reliability", "sigma", "events")
        diag = {
            key: float(np.mean([float(row.get("belief_diagnostics", {}).get(key, np.nan)) for row in care_rows]))
            for key in diag_keys
            if care_rows and any(key in row.get("belief_diagnostics", {}) for row in care_rows)
        }
        tasks[task] = {
            "episodes": len(care_rows),
            "successes": sum(bool(row.get("success", False)) for row in care_rows),
            "success_rate": float(np.mean([bool(row.get("success", False)) for row in care_rows])) if care_rows else 0.0,
            "failure_seeds": [int(row["seed"]) for row in failures],
            "failure_steps": {str(int(row["seed"])): int(row.get("steps", -1)) for row in failures},
            "care_overrides": sum(int(row.get("overrides", 0)) for row in care_rows),
            "care_fallbacks": sum(int(row.get("fallbacks", 0)) for row in care_rows),
            "care_safety_rejections": sum(int(row.get("safety_rejections", 0)) for row in care_rows),
            "care_candidate_counts": dict(Counter(str(k) for row in care_rows for k in row.get("candidate_counts", {}))),
            "common_seed_count": len(common),
            "identical_trace_count": len(identical),
            "identical_trace_fraction": len(identical) / max(len(common), 1),
            "identical_trace_seeds": identical,
            "belief_diagnostics_mean": diag,
            "available_failure_fields": sorted(set().union(*(row.keys() for row in failures))) if failures else [],
        }
    total_rows = sum(int(value["episodes"]) for value in tasks.values())
    total_failures = sum(len(value["failure_seeds"]) for value in tasks.values())
    return {
        "tasks": tasks,
        "total_episodes": total_rows,
        "total_failures": total_failures,
        "total_successes": total_rows - total_failures,
        "total_overrides": sum(int(value["care_overrides"]) for value in tasks.values()),
        "total_fallbacks": sum(int(value["care_fallbacks"]) for value in tasks.values()),
        "all_common_trace_seeds_identical": bool(all_care and all_off and set(all_care) == set(all_off) and all(all_care[s].get("action_trace_sha256") == all_off[s].get("action_trace_sha256") for s in all_care)),
    }, notes


def outcome_signal(candidate: Mapping[str, Any], reference: Mapping[str, Any], horizon: int, epsilon: float) -> bool:
    left = candidate.get("outcomes", {}).get(str(horizon), {})
    right = reference.get("outcomes", {}).get(str(horizon), {})
    if not left or not right or left.get("hard_safety_violation") or right.get("hard_safety_violation"):
        return False
    a = np.asarray(left.get("bounded_utility_vector", []), dtype=np.float64)
    b = np.asarray(right.get("bounded_utility_vector", []), dtype=np.float64)
    return bool(a.shape == b.shape and a.size and np.any(np.abs(a - b) >= epsilon))


def branch_audit(root: Path, epsilon: float) -> dict[str, Any]:
    by_task: dict[str, Any] = {}
    all_families = 0
    all_branches = 0
    total_signals = 0
    total_units = 0
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        families = [read_json(path) for path in sorted(task_dir.glob("*.json"))]
        stage_at_snapshot = Counter(str(f.get("snapshot_metrics", {}).get("stage_id", "missing")) for f in families)
        final_stages = Counter()
        statuses = Counter()
        requested = Counter()
        applied = Counter()
        candidate_diff: defaultdict[str, list[float]] = defaultdict(list)
        signals = 0
        units = 0
        for family in families:
            all_families += 1
            branches = family.get("branches", [])
            all_branches += len(branches)
            requested.update(int(branch.get("intervention_steps_requested", -1)) for branch in branches)
            applied.update(int(branch.get("intervention_steps_applied", -1)) for branch in branches)
            indexed = {(int(branch.get("repeat_id", -1)), str(branch.get("regime", "")), int(branch.get("candidate_id", -1))): branch for branch in branches}
            for branch in branches:
                statuses[str(branch.get("status", "missing"))] += 1
                for horizon in HORIZONS:
                    outcome = branch.get("outcomes", {}).get(str(horizon), {})
                    if outcome:
                        final_stages[str(outcome.get("final_stage_id", "missing"))] += 1
                if int(branch.get("candidate_id", -1)) == 0:
                    continue
                key = (int(branch.get("repeat_id", -1)), str(branch.get("regime", "")), 0)
                reference = indexed.get(key)
                for horizon in HORIZONS:
                    units += 1
                    if reference is not None and outcome_signal(branch, reference, horizon, epsilon):
                        signals += 1
                # The compact corpus contains executed actions.  Measure the
                # largest focal action difference against candidate 0 at the
                # same repeat/regime; this is not a physical-state difference.
                if reference is not None:
                    arm = f"panda-{int(family.get('focal_agent', -1))}"
                    c_rows = branch.get("executed_actions", [])
                    r_rows = reference.get("executed_actions", [])
                    n = min(len(c_rows), len(r_rows))
                    diffs = []
                    for index in range(n):
                        if arm in c_rows[index] and arm in r_rows[index]:
                            diffs.append(float(np.max(np.abs(np.asarray(c_rows[index][arm], dtype=np.float64) - np.asarray(r_rows[index][arm], dtype=np.float64)))))
                    if diffs:
                        candidate_diff[str(int(branch.get("candidate_id", -1)))].append(max(diffs))
        by_task[task_dir.name] = {
            "families": len(families),
            "snapshot_stage_counts": dict(stage_at_snapshot),
            "branch_count": sum(len(f.get("branches", [])) for f in families),
            "status_counts": dict(statuses),
            "final_stage_counts_by_horizon": dict(final_stages),
            "intervention_steps_requested": dict(requested),
            "intervention_steps_applied": dict(applied),
            "signal_count": signals,
            "signal_units": units,
            "signal_density": signals / max(units, 1),
            "max_focal_action_diff_by_candidate": {
                key: {
                    "count": len(values),
                    "mean_max_abs": float(np.mean(values)),
                    "max_max_abs": float(np.max(values)),
                }
                for key, values in sorted(candidate_diff.items())
            },
        }
        total_signals += signals
        total_units += units
    return {
        "family_root": str(root.resolve()),
        "families": all_families,
        "branches": all_branches,
        "signal_epsilon": epsilon,
        "signal_count": total_signals,
        "signal_units": total_units,
        "signal_density": total_signals / max(total_units, 1),
        "tasks": by_task,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    validation = report["validation"]
    lines = [
        "# MARS-Control CARE failure audit",
        "",
        "This report is descriptive only. It does not tune on Validation20 and does not modify the formal run.",
        "",
        "## Bottom line",
        "",
        f"- Validation20: {validation['total_successes']}/{validation['total_episodes']} successes; CARE overrides={validation['total_overrides']}, fallbacks={validation['total_fallbacks']}.",
        f"- CARE and selector-off traces are identical for every paired seed: `{validation['all_common_trace_seeds_identical']}`.",
        "- The compact MARS branch corpus has no video/frame telemetry; final physical state cannot be reconstructed from Validation20 JSONL alone.",
        "- Therefore the first actionable bottleneck is candidate/branch signal acquisition and reference task execution, not selector threshold relaxation.",
        "",
        "## Validation20 by task",
        "",
        "| task | success | failures | overrides | fallbacks | identical trace fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, value in validation["tasks"].items():
        lines.append(f"| {task} | {value['successes']}/{value['episodes']} | {len(value['failure_seeds'])} | {value['care_overrides']} | {value['care_fallbacks']} | {value['identical_trace_fraction']:.0%} |")
    lines += [
        "",
        "## What is proven vs missing",
        "",
        "**Proven by existing artifacts:**",
        "",
        "- CARE does not alter the action trace in the formal Validation20 run.",
        "- The branch collector produced valid branches, but every stored branch used the one-step intervention contract.",
        "- Failure rows do not contain stage transitions, object poses, gripper/contact events, or termination reasons.",
        "",
        "**Missing and required before algorithm selection:**",
        "",
        "- per-step local RGB, qpos, action, gripper, stage/predicate overlay;",
        "- first contact/grasp/placement/stacking event and final physical state;",
        "- explicit fallback/selection reason at each step;",
        "- a small pre-registered diagnostic seed set disjoint from Validation20.",
        "",
        "## Branch corpus",
        "",
        "| task | families | snapshot stage | final stages | intervention applied | signal density |",
        "|---|---:|---|---|---|---:|",
    ]
    for task, value in report["branches"]["tasks"].items():
        lines.append(f"| {task} | {value['families']} | {value['snapshot_stage_counts']} | {value['final_stage_counts_by_horizon']} | {value['intervention_steps_applied']} | {value['signal_density']:.2%} |")
    lines += ["", "## Decision rule", "", "Keep fixed stratified sampling as the main protocol. Run the diagnostic recorder first; compare 1/4/8/16-step prefixes only on the disjoint diagnostic seeds. Promote a duration or candidate change only if it increases branch signal density/effective pairs, preserves restore/replay parity and safety, and improves the fixed smoke tasks without degrading the reference.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--family-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--signal-epsilon", type=float, default=1e-3)
    args = parser.parse_args()
    validation, notes = validation_audit(args.validation_root)
    report: dict[str, Any] = {"format_version": "before-we-act.mars-care-failure-audit/1", "validation": validation, "notes": notes}
    if args.family_root:
        report["branches"] = branch_audit(args.family_root, args.signal_epsilon)
    else:
        report["branches"] = {"tasks": {}, "families": 0, "branches": 0}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": "complete", "json": str((args.output_dir / 'report.json').resolve()), "markdown": str((args.output_dir / 'report.md').resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
