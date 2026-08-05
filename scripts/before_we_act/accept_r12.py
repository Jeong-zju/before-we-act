#!/usr/bin/env python3
"""Authoritative R12 component validity plus mandatory paired Gate20 acceptance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


TASKS = (
    "lift_barrier",
    "camera_alignment",
    "three_robots_stack_cube",
    "long_pipeline_delivery",
    "take_photo",
)


def read(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check(identifier: str, passed: bool, evidence: str, detail: str = ""):
    return {"id": identifier, "passed": bool(passed), "evidence": evidence, "detail": detail}


def rows_by_seed(payload):
    return {int(row["seed"]): bool(row["success"]) for row in payload["rows"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("p0", "p1", "p2", "p3"), required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--patch", required=True)
    parser.add_argument("--dependency", required=True)
    parser.add_argument("--parity", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--offline", required=True)
    parser.add_argument("--core-free", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--baseline", action="append", default=[])
    parser.add_argument("--gate20", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.gate20) != 5:
        raise ValueError("R12 acceptance requires exactly five Gate20 task mappings")
    gate20 = {}
    for item in args.gate20:
        task, separator, path = item.partition("=")
        if separator != "=" or task not in TASKS or task in gate20:
            raise ValueError(f"invalid Gate20 mapping {item!r}")
        gate20[task] = read(path)
    if tuple(gate20) != TASKS:
        raise ValueError("R12 Gate20 task order differs")
    source, license_result, patch = read(args.source), read(args.license), read(args.patch)
    dependency, parity = read(args.dependency), read(args.parity)
    preflight, offline, core_free = read(args.preflight), read(args.offline), read(args.core_free)
    baseline = read(args.baseline_summary)
    baseline_rows = {}
    for item in args.baseline:
        task, separator, path = item.partition("=")
        if separator != "=" or task not in TASKS or task in baseline_rows:
            raise ValueError(f"invalid baseline mapping {item!r}")
        baseline_rows[task] = rows_by_seed(read(path))
    if set(baseline_rows) != set(TASKS):
        raise ValueError("R12 acceptance requires five immutable baseline reports")
    baseline_tasks = baseline["tasks"]
    task_results = {}
    all_complete = True
    seed_protocols = set()
    latency_p95 = []
    for task in TASKS:
        payload = gate20[task]
        rows = rows_by_seed(payload)
        all_complete = all_complete and payload.get("episodes") == 20 and len(rows) == 20
        seed_protocols.add(payload.get("seed_protocol", {}).get("sha256"))
        p95 = payload.get("latency_ms", {}).get("p95")
        if p95 is not None:
            latency_p95.append(float(p95))
        baseline_success = int(baseline_tasks[task]["baseline_successes"])
        candidate_success = int(payload["successes"])
        base_rows = baseline_rows[task]
        if set(rows) - set(base_rows):
            raise ValueError(f"{task} W10 baseline lacks paired Gate20 seeds")
        task_results[task] = {
            "episodes": int(payload["episodes"]),
            "baseline": baseline_success,
            "candidate": candidate_success,
            "delta": candidate_success - baseline_success,
            "paired_wins": sum((not base_rows[seed]) and rows[seed] for seed in rows),
            "paired_losses": sum(base_rows[seed] and (not rows[seed]) for seed in rows),
        }
    baseline_total = sum(row["baseline"] for row in task_results.values())
    candidate_total = sum(row["candidate"] for row in task_results.values())
    if baseline_total != 74 or float(baseline["macro_success_rate"]) != 0.74:
        raise ValueError("frozen W10 Gate20 baseline is not 74/100")
    hard_checks = [
        check("official_source_commit_pinned", source.get("passed") and bool(source.get("resolved_commit")), args.source),
        check("license_verified_and_preserved", license_result.get("passed"), args.license),
        check("minimal_component_patch_audited", patch.get("passed"), args.patch),
        check("no_full_repo_runtime_dependency", dependency.get("passed"), args.dependency),
        check("upstream_component_parity", parity.get("passed"), args.parity),
        check("train_save_strict_restore_normalization_mask", preflight.get("passed"), args.preflight),
        check(
            "formal_2000_updates_and_offline_smoke",
            offline.get("checkpoint_update") == 2_000
            and offline.get("finite")
            and offline.get("absent_agent_zero")
            and offline.get("normalized_abs_max", 99) <= 5.0,
            args.offline,
        ),
        check("physical_core_free_runtime", core_free.get("passed") and offline.get("core_free_runtime"), args.core_free),
        check(
            "complete_paired_gate20",
            all_complete and len(seed_protocols) == 5,
            "five task Gate20 reports",
            f"candidate={candidate_total}/100 baseline={baseline_total}/100",
        ),
    ]
    benchmark_check = check(
        "strictly_better_than_w10",
        all_complete and candidate_total > baseline_total,
        args.baseline_summary,
        f"candidate={candidate_total}/100 must be > W10={baseline_total}/100",
    )
    checks = hard_checks + [benchmark_check]
    result = {
        "schema_version": 1,
        "round": "R12",
        "candidate_id": args.candidate,
        "branch": args.branch,
        "commit": args.commit,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "acceptance_rules": {
            "engineering_hard_gates": [row["id"] for row in hard_checks],
            "quality_gate": "complete five-task Gate20 and candidate total successes strictly > W10 74/100",
            "loss_threshold": None,
        },
        "gate20": {
            "baseline_total_successes": baseline_total,
            "candidate_total_successes": candidate_total,
            "baseline_macro": baseline_total / 100,
            "candidate_macro": candidate_total / 100,
            "macro_delta": (candidate_total - baseline_total) / 100,
            "tasks": task_results,
        },
        "offline": offline,
        "latency_p95_ms_max_task": max(latency_p95) if latency_p95 else None,
        "acceptance": checks,
    }
    result["valid_component"] = all(row["passed"] for row in hard_checks)
    result["qualified"] = result["valid_component"] and benchmark_check["passed"]
    result["passed"] = result["qualified"]
    result["status"] = "PASSED" if result["passed"] else "FAILED"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"candidate": args.candidate, "status": result["status"], "gate20": candidate_total}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
