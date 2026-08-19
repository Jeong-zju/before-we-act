#!/usr/bin/env python3
"""验收完整的 CARE 正式分支数据；不读取非原动作效用。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping

from scripts.before_we_act.rescore_care_branch_pilot import (
    SCIENTIFIC_RESOLUTION,
    outcome_discrete_equal,
    percentile,
)


SUPPORTED_STAGES = {
    "A5R4-CARE-BRANCHES-FORMAL": {
        "contract_stage": "A4R4-CARE-FORMAL-COLLECTION",
        "family_format": "before-we-act.a5r4-care-formal-branch-family/1",
        "receipt_format": "before-we-act.a5r4-care-formal-collection-receipt/1",
        "passed_status": "PASSED_A5R4_FORMAL_COLLECTION_READY_FOR_GATES",
        "failed_status": "FAILED_A5R4_FORMAL_COLLECTION",
    },
    "A5R5-CARE-GATE-FIRST-BRANCHES": {
        "contract_stage": "A4R5-CARE-GATE-FIRST-COLLECTION",
        "family_format": "before-we-act.a5r5-care-gate-first-branch-family/1",
        "receipt_format": "before-we-act.a5r5-care-gate-first-collection-receipt/1",
        "passed_status": "PASSED_A5R5_GATE_FIRST_COLLECTION_READY_FOR_GATES",
        "failed_status": "FAILED_A5R5_GATE_FIRST_COLLECTION",
    },
    "A5R6-CARE-COMPACT-BRANCHES": {
        "contract_stage": "A4R6-CARE-COMPACT-COLLECTION",
        "family_format": "before-we-act.a5r6-care-compact-branch-family/1",
        "receipt_format": "before-we-act.a5r6-care-compact-collection-receipt/1",
        "passed_status": "PASSED_A5R6_COMPACT_COLLECTION_READY_FOR_CORE_GATES",
        "failed_status": "FAILED_A5R6_COMPACT_COLLECTION",
    },
    "A5R7-CARE-COMMON-SUPPORT-BRANCHES": {
        "contract_stage": "A4R7-CARE-COMMON-SUPPORT-COLLECTION",
        "family_format": "before-we-act.a5r7-care-common-support-branch-family/1",
        "receipt_format": "before-we-act.a5r7-care-common-support-collection-receipt/1",
        "passed_status": "PASSED_A5R7_COMMON_SUPPORT_COLLECTION_READY_FOR_SUPPORTED_HORIZON_GATES",
        "failed_status": "FAILED_A5R7_COMMON_SUPPORT_COLLECTION",
        "common_support": True,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    manifest_stage = str(manifest.get("stage_id"))
    stage = SUPPORTED_STAGES.get(manifest_stage)
    if stage is None or contract.get("stage_id") != stage["contract_stage"]:
        raise RuntimeError("正式清单或合同阶段错误")
    contract_sha = sha256_file(args.contract)
    if manifest["contract_sha256"] != contract_sha:
        raise RuntimeError("正式清单的合同发生漂移")

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
        if (
            row.get("stage_id") != manifest_stage
            or row.get("format_version") != stage["family_format"]
            or row.get("resource_only") is not False
            or row.get("contract_sha256") != contract_sha
            or int(row.get("branch_count", -1)) != 24
        ):
            raise RuntimeError(f"正式状态族来源或格式错误：{json_path}")
        rows.append(row)
        for path in (json_path, npz_path):
            artifacts.append(
                {
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    invalid_candidates = 0
    invalid_branch_statuses = 0
    branch_statuses: Counter[str] = Counter()
    replay_policy_calls = 0
    replay_action_errors = []
    origin_errors = []
    origin_sources_exact = True
    probe_terminal_success_exact = True
    utility_errors = []
    discrete_mismatches = []
    common_support_errors = []
    common_support_steps: Counter[int] = Counter()
    supported_horizon_families: Counter[int] = Counter()
    repeat_values: dict[tuple[str, str, str], list[tuple[str, int, float]]] = defaultdict(list)
    allocation: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        allocation[(str(row["split"]), str(row["sampling_stratum"]), str(row["task"]))] += 1
        invalid_candidates += sum(int(not bool(item["valid"])) for item in row["candidate_legality"])
        probe = row["restore_probe"]
        origin_errors.append(float(probe["restore_observation_max_abs_error"]))
        origin_sources_exact &= probe.get("restore_observation_source") == "captured_snapshot"
        probe_terminal_success_exact &= bool(probe["terminal_and_success_exact"])
        if stage.get("common_support"):
            support_rows = row.get("repeat_common_replay_support", [])
            support_by_repeat = {
                int(item["repeat_id"]): int(item["common_replay_support_steps"])
                for item in support_rows
            }
            if set(support_by_repeat) != {0, 1}:
                common_support_errors.append(
                    {"snapshot_id": row["snapshot_id"], "error": "repeat_support_missing"}
                )
            family_supported = {
                horizon
                for horizon in (8, 16, 32, 64)
                if all(support_by_repeat.get(repeat_id, -1) >= horizon for repeat_id in (0, 1))
            }
            for horizon in family_supported:
                supported_horizon_families[horizon] += 1
        for branch in row["branches"]:
            branch_statuses[str(branch["status"])] += 1
            invalid_branch_statuses += int(branch["status"] not in {"VALID", "SUCCESS_TERMINATION"})
            origin_errors.append(float(branch["restore_observation_max_abs_error"]))
            origin_sources_exact &= branch.get("restore_observation_source") == "captured_snapshot"
            if branch["regime"] == "replay":
                replay_policy_calls += int(branch["policy_evaluations"]["teammates"])
                replay_action_errors.append(float(branch["replay_teammate_action_max_abs_error"]))
            if stage.get("common_support"):
                repeat_id = int(branch["repeat_id"])
                support = support_by_repeat.get(repeat_id, -1)
                common_support_steps[support] += 1
                expected_horizons = [
                    value for value in (8, 16, 32, 64) if value <= support
                ]
                if (
                    int(branch.get("common_replay_support_steps", -1)) != support
                    or branch.get("supported_outcome_horizons") != expected_horizons
                    or branch.get("unsupported_outcome_horizons")
                    != [value for value in (8, 16, 32, 64) if value > support]
                    or sorted(map(int, branch["outcomes"])) != expected_horizons
                    or int(branch["steps"]) > support
                ):
                    common_support_errors.append(
                        {
                            "snapshot_id": row["snapshot_id"],
                            "repeat_id": repeat_id,
                            "candidate_id": int(branch["candidate_id"]),
                            "regime": branch["regime"],
                            "error": "branch_support_mismatch",
                        }
                    )
            if int(branch["candidate_id"]) == 0:
                for horizon, outcome in branch["outcomes"].items():
                    repeat_values[(str(row["task"]), str(branch["regime"]), str(horizon))].append(
                        (str(row["snapshot_id"]), int(branch["repeat_id"]), float(outcome["utility_main"]))
                    )
        for repeat_id in (0, 1):
            selected = [
                branch for branch in row["branches"]
                if int(branch["candidate_id"]) == 0 and int(branch["repeat_id"]) == repeat_id
            ]
            reactive = next(branch for branch in selected if branch["regime"] == "reactive")
            replay = next(branch for branch in selected if branch["regime"] == "replay")
            for horizon in sorted(set(reactive["outcomes"]) & set(replay["outcomes"]), key=int):
                left, right = reactive["outcomes"][horizon], replay["outcomes"][horizon]
                utility_errors.append(abs(float(left["utility_main"]) - float(right["utility_main"])))
                if not outcome_discrete_equal(left, right):
                    discrete_mismatches.append(
                        {"snapshot_id": row["snapshot_id"], "repeat_id": repeat_id, "horizon": int(horizon)}
                    )

    repeat_noise_by_task_horizon = {}
    for key, values in repeat_values.items():
        grouped: dict[str, dict[int, float]] = defaultdict(dict)
        for snapshot_id, repeat_id, utility in values:
            grouped[snapshot_id][repeat_id] = utility
        differences = [
            abs(items[0] - items[1]) for items in grouped.values() if set(items) == {0, 1}
        ]
        repeat_noise_by_task_horizon["|".join(key)] = percentile(differences, 0.95)
    repeat_noise_q95_max = max(repeat_noise_by_task_horizon.values(), default=math.inf)

    expected_allocation = {
        (item["split"], item["stratum"], item["task"]): int(item["count"])
        for item in manifest["allocation"]
    }
    expected_family_count = int(manifest["family_count"])
    expected_branch_count = int(manifest["planned_branch_count"])
    checks = {
        "all_expected_families_present": (
            not missing and len(rows) == expected_family_count
        ),
        "all_expected_branches_present": (
            sum(int(row["branch_count"]) for row in rows)
            == expected_branch_count
        ),
        "allocation_exact": dict(allocation) == expected_allocation,
        "captured_policy_observation_source_exact": origin_sources_exact,
        "restored_proprioception_max_abs_lte_1e_6": max(origin_errors, default=math.inf) <= 1e-6,
        "restore_probe_terminal_and_success_exact": probe_terminal_success_exact,
        "replay_never_evaluated_teammate_policy": replay_policy_calls == 0,
        "replay_actions_exact": max(replay_action_errors, default=math.inf) == 0.0,
        "candidate0_discrete_outcomes_exact": not discrete_mismatches,
        "candidate0_reactive_replay_utility_lte_0_02": max(utility_errors, default=math.inf) <= SCIENTIFIC_RESOLUTION,
        "reference_repeat_noise_each_task_horizon_q95_lte_0_02": repeat_noise_q95_max <= SCIENTIFIC_RESOLUTION,
        "all_six_candidates_legal": invalid_candidates == 0,
        "all_branch_statuses_valid": invalid_branch_statuses == 0,
    }
    if stage.get("common_support"):
        checks["common_replay_support_metadata_exact"] = not common_support_errors
    passed = all(checks.values())
    usage = shutil.disk_usage(args.family_root)
    receipt = {
        "format_version": stage["receipt_format"],
        "stage_id": manifest_stage,
        "completed_at_utc": utc_now(),
        "status": stage["passed_status"] if passed else stage["failed_status"],
        "passed": passed,
        "contract": str(args.contract.resolve()),
        "contract_sha256": contract_sha,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "formal_gate_a_evaluated": False,
        "formal_gate_b_evaluated": False,
        "care_training_authorized": False,
        "checks": checks,
        "measurements": {
            "snapshot_families": len(rows),
            "short_branches": sum(int(row["branch_count"]) for row in rows),
            "missing_family_count": len(missing),
            "missing_family_ids_first_100": missing[:100],
            "invalid_candidate_count": invalid_candidates,
            "invalid_branch_status_count": invalid_branch_statuses,
            "branch_status_counts": dict(branch_statuses),
            "candidate0_discrete_mismatch_count": len(discrete_mismatches),
            "candidate0_discrete_mismatches_first_100": discrete_mismatches[:100],
            "candidate0_reactive_replay_utility_max_abs_error": max(utility_errors, default=math.nan),
            "reference_repeat_noise_q95_max": repeat_noise_q95_max,
            "reference_repeat_noise_q95_by_task_regime_horizon": repeat_noise_by_task_horizon,
            "replay_teammate_action_max_abs_error": max(replay_action_errors, default=math.nan),
            "replay_teammate_policy_evaluation_count": replay_policy_calls,
            "common_replay_support_steps_by_branch": dict(sorted(common_support_steps.items())),
            "families_with_both_repeats_supported_by_horizon": {
                str(value): supported_horizon_families[value]
                for value in (8, 16, 32, 64)
            },
            "common_replay_support_errors_first_100": common_support_errors[:100],
            "artifact_count": len(artifacts),
            "artifact_bytes_total": sum(item["size_bytes"] for item in artifacts),
            "remote_disk_free_bytes_after_collection": usage.free,
        },
        "artifacts": artifacts,
        "storage": {
            "remote_only": True,
            "local_transfer_performed": False,
            "workspace_is_persistent_volume": False,
            "warning": "Recycle or destroy erases the non-volume remote workspace.",
        },
        "claim_limits": [
            "This receipt validates collection integrity only.",
            "No non-reference candidate utility was read by this summarizer.",
            "Gate A and Gate B remain unevaluated.",
            "CARE training remains forbidden until both gates pass."
        ],
    }
    atomic_json(args.output, receipt)
    print(json.dumps({"output": str(args.output), "passed": passed, "checks": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
