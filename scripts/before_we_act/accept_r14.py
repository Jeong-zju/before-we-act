#!/usr/bin/env python3
"""Authoritative R14 component provenance plus complete W12-paired Gate20."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


TASKS = (
    "lift_barrier", "camera_alignment", "three_robots_stack_cube",
    "long_pipeline_delivery", "take_photo",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mappings(values, name):
    result = {}
    for item in values:
        task, separator, path = item.partition("=")
        if separator != "=" or task not in TASKS or task in result:
            raise ValueError(f"invalid {name} mapping {item!r}")
        result[task] = read(path)
    if tuple(result) != TASKS:
        raise ValueError(f"{name} mappings must use frozen task order")
    return result


def gate(label, passed, detail):
    return {"id": label, "passed": bool(passed), "detail": detail}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True)
    for name in ("source", "license", "patch", "dependency", "action-effect", "parity", "preflight", "separation"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--gate20", action="append", default=[])
    parser.add_argument("--baseline", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    candidate = mappings(args.gate20, "Gate20")
    baseline = mappings(args.baseline, "baseline")
    receipts = {
        key.replace("_", "-"): read(getattr(args, key))
        for key in ("source", "license", "patch", "dependency", "action_effect", "parity", "preflight", "separation")
    }
    acceptance = []
    for key, payload in receipts.items():
        acceptance.append(gate(key, payload.get("passed") is True, f"{key} receipt passed"))
    task_rows, candidate_total, baseline_total = {}, 0, 0
    all_complete = all_paired = protected_exact = True
    total_interventions = total_fallbacks = 0
    p95 = []
    for task in TASKS:
        current, parent = candidate[task], baseline[task]
        current_rows, parent_rows = current.get("rows", []), parent.get("rows", [])
        complete = current.get("episodes") == 20 and parent.get("episodes") == 20
        paired = complete and [row.get("seed") for row in current_rows] == [row.get("seed") for row in parent_rows]
        if not all(isinstance(row.get("success"), bool) for row in current_rows + parent_rows):
            paired = False
        all_complete &= complete
        all_paired &= paired
        current_success = sum(row["success"] for row in current_rows) if paired else 0
        parent_success = sum(row["success"] for row in parent_rows) if paired else 0
        candidate_total += current_success
        baseline_total += parent_success
        paired_wins = paired_losses = 0
        if paired:
            for row, base in zip(current_rows, parent_rows):
                paired_wins += int(row["success"] and not base["success"])
                paired_losses += int(base["success"] and not row["success"])
        if task != "three_robots_stack_cube":
            protected_exact &= current.get("route") == "exact_w12_fallback" and current_success == parent_success
        total_interventions += int(current.get("planner", {}).get("interventions", 0))
        total_fallbacks += int(current.get("planner", {}).get("fallbacks", 0))
        latency = current.get("latency_ms", {}).get("p95")
        if latency is not None:
            p95.append(float(latency))
        task_rows[task] = {
            "episodes": current.get("episodes"),
            "baseline": parent_success,
            "candidate": current_success,
            "delta": current_success - parent_success,
            "paired_wins": paired_wins,
            "paired_losses": paired_losses,
            "route": current.get("route"),
        }
    acceptance.extend((
        gate("frozen_w12_baseline", baseline_total == 77, f"baseline={baseline_total}/100"),
        gate("complete_paired_gate20", all_complete and all_paired, "five tasks x 20 paired seeds"),
        gate("protected_exact_w12", protected_exact, "four protected tasks preserve W12"),
        gate("strict_quality_improvement", candidate_total > 77, f"candidate={candidate_total}/100 > W12=77/100"),
    ))
    passed = all(row["passed"] for row in acceptance)
    result = {
        "schema_version": 1,
        "round": "R14",
        "candidate_id": args.candidate,
        "branch": args.branch,
        "commit": args.commit,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASSED" if passed else "FAILED",
        "acceptance": acceptance,
        "gate20": {
            "baseline_total_successes": baseline_total,
            "candidate_total_successes": candidate_total,
            "macro_success": candidate_total / 100,
            "tasks": task_rows,
            "paired_wins": sum(row["paired_wins"] for row in task_rows.values()),
            "paired_losses": sum(row["paired_losses"] for row in task_rows.values()),
            "p95_latency_ms": max(p95) if p95 else None,
        },
        "planner": {"interventions": total_interventions, "fallbacks": total_fallbacks},
        "quality_gate": "complete same-seed 5x20 and total successes strictly greater than W12 77/100",
        "receipts": receipts,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"candidate": args.candidate, "status": result["status"], "gate20": candidate_total}, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
