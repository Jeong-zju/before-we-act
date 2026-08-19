#!/usr/bin/env python3
"""验收 A5 状态恢复/资源试跑；不读取非基准候选的效用。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

from before_we_act.care_branch_collector import atomic_json, sha256_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = []
    artifacts = []
    missing = []
    for family in manifest["families"]:
        stem = args.family_root / family["task"] / family["snapshot_id"]
        json_path = stem.with_suffix(".json")
        npz_path = stem.with_suffix(".npz")
        if not json_path.is_file() or not npz_path.is_file():
            missing.append(family["snapshot_id"])
            continue
        row = json.loads(json_path.read_text(encoding="utf-8"))
        if row.get("resource_only") is not True or row.get("branch_count") != 24:
            raise RuntimeError(f"invalid resource-only family: {json_path}")
        rows.append(row)
        for path in (json_path, npz_path):
            artifacts.append(
                {
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not rows:
        raise RuntimeError("pilot produced no complete snapshot family")
    restore_errors = []
    rerender_diagnostics = []
    fidelity_errors = []
    branch_seconds = []
    invalid = 0
    invalid_branch_statuses = 0
    invalid_by_candidate: dict[str, dict[str, object]] = {}
    invalid_by_task: dict[str, dict[str, int]] = {}
    branch_status_counts: Counter[str] = Counter()
    restore_failed_families = []
    fidelity_failed_families = []
    replay_policy_violations = 0
    replay_action_errors = []
    reference_canonicalization_changes = []
    belief_off_canonicalization_changes = []
    reference_repeat_values: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        probe = row["restore_probe"]
        if not bool(probe["passed"]):
            restore_failed_families.append(row["snapshot_id"])
        rerender_diagnostics.append(
            float(probe["restore_rerender_diagnostic_max_abs_error"])
        )
        restore_errors.extend(
            float(probe[key])
            for key in (
                "restore_observation_max_abs_error",
                "reference_action_max_abs_error",
                "qpos_max_abs_error",
                "observation_max_abs_error",
                "reward_max_abs_error",
            )
        )
        family_fidelity = [
            float(item["utility_max_abs_error"])
            for item in row["reference_reactive_replay_fidelity"]
        ]
        fidelity_errors.extend(family_fidelity)
        if max(family_fidelity, default=math.inf) > 1e-4:
            fidelity_failed_families.append(row["snapshot_id"])
        task_legality = invalid_by_task.setdefault(
            str(row["task"]), {"invalid": 0, "total": 0}
        )
        for item in row["candidate_legality"]:
            candidate = invalid_by_candidate.setdefault(
                str(item["candidate_id"]),
                {"invalid": 0, "total": 0, "failure_reasons": Counter()},
            )
            failed = int(not bool(item["valid"]))
            invalid += failed
            candidate["invalid"] = int(candidate["invalid"]) + failed
            candidate["total"] = int(candidate["total"]) + 1
            candidate["failure_reasons"].update(item["failures"])
            task_legality["invalid"] += failed
            task_legality["total"] += 1
        for branch in row["branches"]:
            branch_seconds.append(float(branch["wall_seconds"]))
            reference_canonicalization_changes.append(
                float(branch["reference_canonicalization_max_abs_change"])
            )
            belief_off_canonicalization_changes.append(
                float(branch["belief_off_canonicalization_max_abs_change"])
            )
            branch_status_counts[str(branch["status"])] += 1
            invalid_branch_statuses += int(
                branch["status"] not in {"VALID", "SUCCESS_TERMINATION"}
            )
            if branch["regime"] == "replay":
                replay_policy_violations += int(
                    branch["policy_evaluations"]["teammates"] != 0
                )
                replay_action_errors.append(
                    float(branch["replay_teammate_action_max_abs_error"])
                )
            if branch["candidate_id"] == 0:
                for horizon, outcome in branch["outcomes"].items():
                    key = (row["task"], branch["regime"], horizon)
                    reference_repeat_values.setdefault(key, []).append(
                        float(outcome["utility_main"])
                    )
    repeat_differences = []
    for values in reference_repeat_values.values():
        for start in range(0, len(values), 2):
            if start + 1 < len(values):
                repeat_differences.append(abs(values[start] - values[start + 1]))
    family_seconds = [float(row["wall_seconds"]) for row in rows]
    total_bytes = sum(item["size_bytes"] for item in artifacts)
    mean_family_seconds = statistics.mean(family_seconds)
    projected_families = 14_800
    projected_branches = projected_families * 24
    projected = {
        "formal_snapshot_families": projected_families,
        "formal_short_branches": projected_branches,
        "single_worker_wall_hours": mean_family_seconds * projected_families / 3600.0,
        "four_worker_idealized_wall_hours": mean_family_seconds * projected_families / 3600.0 / 4.0,
        "artifact_bytes": total_bytes / len(rows) * projected_families,
        "warning": "线性外推只用于租机和磁盘准备，不是正式运行时保证。",
    }
    checks = {
        "all_families_present": len(rows) == int(manifest["family_count"]),
        "all_restore_probes_passed": all(row["restore_probe"]["passed"] for row in rows),
        "restore_max_abs_lte_1e_6": max(restore_errors, default=math.inf) <= 1e-6,
        "candidate0_reactive_replay_utility_lte_1e_4": max(
            fidelity_errors, default=math.inf
        ) <= 1e-4,
        "replay_never_evaluated_teammate_policy": replay_policy_violations == 0,
        "replay_actions_exact": max(replay_action_errors, default=math.inf) == 0.0,
        "all_six_candidates_legal": invalid == 0,
        "all_branch_statuses_valid": invalid_branch_statuses == 0,
        "all_families_have_24_branches": all(row["branch_count"] == 24 for row in rows),
    }
    passed = all(checks.values())
    receipt = {
        "format_version": "before-we-act.a5r2-care-resource-pilot-receipt/3",
        "stage_id": "A5R2-CARE-BRANCHES-PILOT",
        "completed_at_utc": utc_now(),
        "status": (
            "PASSED_A5R2_RESTORE_RESOURCE_PILOT_FULL_COLLECTION_NOT_AUTHORIZED"
            if passed
            else "FAILED_A5R2_RESTORE_RESOURCE_PILOT"
        ),
        "passed": passed,
        "resource_only": True,
        "formal_gate_a_evaluated": False,
        "formal_gate_b_evaluated": False,
        "care_training_authorized": False,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "checks": checks,
        "measurements": {
            "planned_snapshot_families": int(manifest["family_count"]),
            "snapshot_families": len(rows),
            "missing_snapshot_family_count": len(missing),
            "missing_snapshot_family_ids": missing,
            "short_branches": sum(int(row["branch_count"]) for row in rows),
            "restore_max_abs_error": max(restore_errors, default=math.nan),
            "restore_rerender_diagnostic_max_abs_error": max(
                rerender_diagnostics, default=math.nan
            ),
            "candidate0_reactive_replay_utility_max_abs_error": max(
                fidelity_errors, default=math.nan
            ),
            "reference_canonicalization_max_abs_change": max(
                reference_canonicalization_changes, default=math.nan
            ),
            "belief_off_canonicalization_max_abs_change": max(
                belief_off_canonicalization_changes, default=math.nan
            ),
            "reference_repeat_noise_abs_p95": percentile(repeat_differences, 0.95),
            "invalid_candidate_count": invalid,
            "candidate_legality_by_id": {
                key: {
                    "invalid": int(value["invalid"]),
                    "total": int(value["total"]),
                    "failure_reasons": dict(value["failure_reasons"]),
                }
                for key, value in sorted(invalid_by_candidate.items(), key=lambda item: int(item[0]))
            },
            "candidate_legality_by_task": invalid_by_task,
            "invalid_branch_status_count": invalid_branch_statuses,
            "branch_status_counts": dict(branch_status_counts),
            "restore_failed_family_ids": restore_failed_families,
            "reference_fidelity_failed_family_ids": fidelity_failed_families,
            "branch_wall_seconds_mean": statistics.mean(branch_seconds),
            "branch_wall_seconds_p95": percentile(branch_seconds, 0.95),
            "family_wall_seconds_mean": mean_family_seconds,
            "family_wall_seconds_p95": percentile(family_seconds, 0.95),
            "gpu_peak_memory_bytes_max": max(
                int(row["gpu_peak_memory_bytes"]) for row in rows
            ),
            "artifact_bytes_total": total_bytes,
            "artifact_bytes_per_family_mean": total_bytes / len(rows),
        },
        "formal_collection_projection": projected,
        "artifacts": artifacts,
        "claim_limits": [
            "本回执只证明采集器能恢复状态并按合同运行两种队友模式。",
            "基准与关闭信念修正计划先经过公开记录的统一物理动作边界映射；候选构造后没有再次裁剪。",
            "分支首帧来自快照中逐字节保存的部署观测；同一物理状态重新渲染的像素差单独报告，不冒充恢复后的策略输入。",
            "预注册锚点若落在回合终止之后，按缺失状态报告，不事后替换状态。",
            "试跑中的非基准候选效用没有被汇总，也不能用于 Gate A 或 Gate B。",
            "正式 14,800 个状态的采集仍需新的执行授权和具备持久备份的存储位置。",
        ],
    }
    atomic_json(args.output, receipt)
    print(json.dumps(receipt | {"artifacts": "saved"}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
