#!/usr/bin/env python3
"""Authoritative full-data/high-resolution R12-R4 Gate20 acceptance."""
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
TRAIN_COUNTS = {
    "lift_barrier": 8255,
    "camera_alignment": 11764,
    "three_robots_stack_cube": 48892,
    "long_pipeline_delivery": 88493,
    "take_photo": 23044,
}
VALIDATION_COUNTS = {
    "lift_barrier": 1015,
    "camera_alignment": 1457,
    "three_robots_stack_cube": 6138,
    "long_pipeline_delivery": 10981,
    "take_photo": 2884,
}


def read(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check(identifier: str, passed: bool, evidence: str, detail: str = ""):
    return {
        "id": identifier,
        "passed": bool(passed),
        "evidence": evidence,
        "detail": detail,
    }


def rows_by_seed(payload):
    return {int(row["seed"]): bool(row["success"]) for row in payload["rows"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("p0", "p1", "p2", "p3"), required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bridge-checkpoint", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--patch", required=True)
    parser.add_argument("--dependency", required=True)
    parser.add_argument("--parity", required=True)
    parser.add_argument("--cache-equivalence", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--offline", required=True)
    parser.add_argument("--core-free", required=True)
    parser.add_argument("--training-identity", required=True)
    parser.add_argument("--full-index", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--baseline", action="append", default=[])
    parser.add_argument("--gate20", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.gate20) != 5 or len(args.baseline) != 5:
        raise ValueError("R12-R4 acceptance requires five Gate20/baseline mappings")
    gate20, baseline_rows = {}, {}
    for item in args.gate20:
        task, separator, path = item.partition("=")
        if separator != "=" or task not in TASKS or task in gate20:
            raise ValueError(f"invalid Gate20 mapping {item!r}")
        gate20[task] = read(path)
    for item in args.baseline:
        task, separator, path = item.partition("=")
        if separator != "=" or task not in TASKS or task in baseline_rows:
            raise ValueError(f"invalid baseline mapping {item!r}")
        baseline_rows[task] = rows_by_seed(read(path))
    if tuple(gate20) != TASKS or set(baseline_rows) != set(TASKS):
        raise ValueError("R12-R4 task order/coverage differs")

    source, license_result, patch = read(args.source), read(args.license), read(args.patch)
    dependency, parity = read(args.dependency), read(args.parity)
    cache_equivalence = read(args.cache_equivalence)
    preflight, offline, core_free = read(args.preflight), read(args.offline), read(args.core_free)
    identity, full_index = read(args.training_identity), read(args.full_index)
    baseline = read(args.baseline_summary)
    task_results, seed_protocols, latency_p95 = {}, set(), []
    all_complete = True
    for task in TASKS:
        payload = gate20[task]
        rows = rows_by_seed(payload)
        all_complete = all_complete and payload.get("episodes") == 20 and len(rows) == 20
        seed_protocols.add(payload.get("seed_protocol", {}).get("sha256"))
        value = payload.get("latency_ms", {}).get("p95")
        if value is not None:
            latency_p95.append(float(value))
        baseline_success = int(baseline["tasks"][task]["baseline_successes"])
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
    observation = full_index.get("observation", {})
    high_resolution = (
        observation.get("input_height") == 480
        and observation.get("input_width") == 640
        and observation.get("require_native_input_shape") is True
        and observation.get("encoder_patch_grid") == [30, 40]
        and observation.get("spatial_grid") == [6, 8]
        and observation.get("compression_stage")
        == "adaptive_average_after_full_resolution_dinov3"
    )
    hard_checks = [
        check("official_source_commit_pinned", source.get("passed") and bool(source.get("resolved_commit")), args.source),
        check("license_verified_and_preserved", license_result.get("passed"), args.license),
        check("minimal_component_patch_audited", patch.get("passed"), args.patch),
        check("no_full_repo_runtime_dependency", dependency.get("passed"), args.dependency),
        check("upstream_component_parity", parity.get("passed"), args.parity),
        check("strict_restore_bridge_gradient_action_effect", preflight.get("passed"), args.preflight),
        check(
            "native_high_resolution_before_compression",
            high_resolution
            and full_index.get("cache_semantics")
            == "native_480x640_rgb_encoded_before_post_dino_6x8_pooling"
            and cache_equivalence.get("passed") is True,
            args.cache_equivalence,
        ),
        check(
            "full_timestep_counts",
            full_index.get("step_counts", {}).get("train") == TRAIN_COUNTS
            and full_index.get("step_counts", {}).get("validation") == VALIDATION_COUNTS,
            args.full_index,
        ),
        check(
            "causal_source_and_no_recovery",
            identity.get("recovery_cache") is None
            and identity.get("source_aware_history") is True
            and identity.get("full_step_counts") == full_index.get("step_counts"),
            args.training_identity,
        ),
        check(
            "two_stage_10k_plus_120k_complete",
            identity.get("bridge_updates") == 10_000
            and identity.get("joint_updates") == 120_000
            and Path(args.bridge_checkpoint).is_file()
            and Path(args.checkpoint).is_file()
            and offline.get("update") == 130_000,
            args.training_identity,
        ),
        check(
            "full_validation_finite",
            offline.get("validation_rows") == 22_475
            and offline.get("expected_validation_rows") == 22_475
            and offline.get("all_outputs_finite") is True,
            args.offline,
        ),
        check("physical_core_free_runtime", core_free.get("passed") and identity.get("core_free_runtime"), args.core_free),
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
        "round": "R12-R4",
        "candidate_id": args.candidate,
        "branch": args.branch,
        "commit": args.commit,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "bridge_checkpoint": str(Path(args.bridge_checkpoint).resolve()),
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
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"candidate": args.candidate, "status": result["status"], "gate20": candidate_total}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
