#!/usr/bin/env python3
"""按 A4R3 的“实质等价”规则重评不可变的 A5R2 资源试跑。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCIENTIFIC_RESOLUTION = 0.02
EXPECTED_CONTRACT_STAGE = "A4R3-CARE-CONTRACT"
EXPECTED_SOURCE_STAGE = "A5R2-CARE-BRANCHES-PILOT"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.inf
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def discrete_projection(value: Any) -> Any:
    """保留会改变任务语义的离散值，忽略连续测量的微小漂移。"""
    if isinstance(value, Mapping):
        return {
            str(key): discrete_projection(item)
            for key, item in value.items()
            if not isinstance(item, float)
        }
    if isinstance(value, list):
        return [
            discrete_projection(item)
            for item in value
            if not isinstance(item, float)
        ]
    return value


def candidate0_pairs(row: Mapping[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs = []
    repeats = sorted(
        {int(branch["repeat_id"]) for branch in row["branches"] if int(branch["candidate_id"]) == 0}
    )
    for repeat_id in repeats:
        selected = [
            branch
            for branch in row["branches"]
            if int(branch["candidate_id"]) == 0
            and int(branch["repeat_id"]) == repeat_id
        ]
        reactive = [branch for branch in selected if branch["regime"] == "reactive"]
        replay = [branch for branch in selected if branch["regime"] == "replay"]
        if len(reactive) != 1 or len(replay) != 1:
            raise RuntimeError(
                f"{row['snapshot_id']} repeat {repeat_id} 缺少唯一的原动作双分支"
            )
        pairs.append((reactive[0], replay[0]))
    return pairs


def outcome_discrete_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    exact_keys = (
        "hard_safety_violation",
        "first_success_step",
        "final_stage_id",
        "observed_steps",
    )
    if any(left.get(key) != right.get(key) for key in exact_keys):
        return False
    return discrete_projection(left.get("final_factorized_predicates", {})) == discrete_projection(
        right.get("final_factorized_predicates", {})
    )


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    source = json.loads(args.source_receipt.read_text(encoding="utf-8"))
    if contract.get("stage_id") != EXPECTED_CONTRACT_STAGE:
        raise RuntimeError("合同阶段不是 A4R3")
    if source.get("stage_id") != EXPECTED_SOURCE_STAGE:
        raise RuntimeError("源回执不是不可变的 A5R2 资源试跑")
    expected_source_hash = contract["data_reuse"]["source_receipt_sha256"]
    if sha256_file(args.source_receipt) != expected_source_hash:
        raise RuntimeError("A5R2 源回执哈希与 A4R3 合同不一致")

    family_paths = sorted(args.family_root.rglob("*.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in family_paths]
    if len(rows) != int(source["measurements"]["snapshot_families"]):
        raise RuntimeError("A5R2 状态族数量与源回执不一致")

    source_artifacts = {
        str(item["path"]).split("/families/", 1)[-1]: item
        for item in source["artifacts"]
        if "/families/" in str(item["path"])
    }
    local_artifacts = [
        path
        for path in sorted(args.family_root.rglob("*"))
        if path.is_file() and path.suffix in {".json", ".npz"}
    ]
    artifact_mismatches = []
    for path in local_artifacts:
        relative = str(path.relative_to(args.family_root))
        expected = source_artifacts.get(relative)
        if (
            expected is None
            or int(expected["size_bytes"]) != path.stat().st_size
            or str(expected["sha256"]) != sha256_file(path)
        ):
            artifact_mismatches.append(relative)

    origin_errors: list[float] = []
    post_step_qpos_errors: list[float] = []
    post_step_action_errors: list[float] = []
    post_step_observation_errors: list[float] = []
    post_step_reward_errors: list[float] = []
    rerender_errors: list[float] = []
    utility_errors: list[float] = []
    repeat_noise: list[float] = []
    discrete_mismatches: list[dict[str, Any]] = []
    replay_action_errors: list[float] = []
    replay_policy_calls = 0
    origin_source_exact = True
    probe_terminal_success_exact = True

    for row in rows:
        probe = row["restore_probe"]
        origin_errors.append(float(probe["restore_observation_max_abs_error"]))
        post_step_qpos_errors.append(float(probe["qpos_max_abs_error"]))
        post_step_action_errors.append(float(probe["reference_action_max_abs_error"]))
        post_step_observation_errors.append(float(probe["observation_max_abs_error"]))
        post_step_reward_errors.append(float(probe["reward_max_abs_error"]))
        rerender_errors.append(float(probe["restore_rerender_diagnostic_max_abs_error"]))
        origin_source_exact &= probe.get("restore_observation_source") == "captured_snapshot"
        probe_terminal_success_exact &= bool(probe["terminal_and_success_exact"])

        by_regime_horizon: dict[tuple[str, str], list[float]] = {}
        for branch in row["branches"]:
            origin_errors.append(float(branch["restore_observation_max_abs_error"]))
            origin_source_exact &= branch.get("restore_observation_source") == "captured_snapshot"
            if branch["regime"] == "replay":
                replay_policy_calls += int(branch["policy_evaluations"]["teammates"])
                replay_action_errors.append(float(branch["replay_teammate_action_max_abs_error"]))
            if int(branch["candidate_id"]) == 0:
                for horizon, outcome in branch["outcomes"].items():
                    by_regime_horizon.setdefault((str(branch["regime"]), str(horizon)), []).append(
                        float(outcome["utility_main"])
                    )
        for values in by_regime_horizon.values():
            if len(values) != 2:
                raise RuntimeError(f"{row['snapshot_id']} 的原动作重复次数不是 2")
            repeat_noise.append(abs(values[0] - values[1]))

        for repeat_id, (reactive, replay) in enumerate(candidate0_pairs(row)):
            horizons = sorted(set(reactive["outcomes"]) & set(replay["outcomes"]), key=int)
            if not horizons:
                raise RuntimeError(f"{row['snapshot_id']} 的原动作双分支没有共同结果时域")
            for horizon in horizons:
                left = reactive["outcomes"][horizon]
                right = replay["outcomes"][horizon]
                utility_errors.append(abs(float(left["utility_main"]) - float(right["utility_main"])))
                if not outcome_discrete_equal(left, right):
                    discrete_mismatches.append(
                        {
                            "snapshot_id": row["snapshot_id"],
                            "repeat_id": repeat_id,
                            "horizon": int(horizon),
                        }
                    )

    repeat_noise_q95 = percentile(repeat_noise, 0.95)
    checks = {
        "source_artifacts_match_a5r2_receipt": not artifact_mismatches
        and len(local_artifacts) == len(source_artifacts),
        "all_source_families_present": len(rows) == int(source["measurements"]["planned_snapshot_families"]),
        "all_source_families_have_24_branches": all(int(row["branch_count"]) == 24 for row in rows),
        "captured_policy_observation_source_exact": origin_source_exact,
        "restored_proprioception_max_abs_lte_1e_6": max(origin_errors, default=math.inf) <= 1e-6,
        "restore_probe_terminal_and_success_exact": probe_terminal_success_exact,
        "replay_never_evaluated_teammate_policy": replay_policy_calls == 0,
        "replay_actions_exact": max(replay_action_errors, default=math.inf) == 0.0,
        "candidate0_discrete_outcomes_exact": not discrete_mismatches,
        "candidate0_reactive_replay_utility_lte_0_02": max(utility_errors, default=math.inf)
        <= SCIENTIFIC_RESOLUTION,
        "reference_repeat_noise_q95_lte_0_02": repeat_noise_q95 <= SCIENTIFIC_RESOLUTION,
        "all_six_candidates_legal": bool(source["checks"]["all_six_candidates_legal"]),
        "all_branch_statuses_valid": bool(source["checks"]["all_branch_statuses_valid"]),
    }
    passed = all(checks.values())

    receipt = {
        "format_version": "before-we-act.a5r3-care-material-equivalence-rescore/1",
        "stage_id": "A5R3-CARE-MATERIAL-EQUIVALENCE-RESCORE",
        "completed_at_utc": utc_now(),
        "status": (
            "PASSED_A5R3_RESOURCE_PILOT_FORMAL_COLLECTION_NOT_EXECUTED"
            if passed
            else "FAILED_A5R3_RESOURCE_PILOT"
        ),
        "passed": passed,
        "contract": str(args.contract.resolve()),
        "contract_sha256": sha256_file(args.contract),
        "source_receipt": str(args.source_receipt.resolve()),
        "source_receipt_sha256": sha256_file(args.source_receipt),
        "source_branches_modified": False,
        "formal_collection_ready_subject_to_user_and_storage_authorization": passed,
        "formal_collection_executed": False,
        "formal_gate_a_evaluated": False,
        "formal_gate_b_evaluated": False,
        "care_training_authorized": False,
        "scientific_resolution": SCIENTIFIC_RESOLUTION,
        "checks": checks,
        "measurements": {
            "snapshot_families": len(rows),
            "short_branches": sum(int(row["branch_count"]) for row in rows),
            "candidate0_reactive_replay_utility_max_abs_error": max(utility_errors, default=math.nan),
            "reference_repeat_noise_abs_q95": repeat_noise_q95,
            "candidate0_discrete_mismatch_count": len(discrete_mismatches),
            "candidate0_discrete_mismatches": discrete_mismatches,
            "initial_restored_proprioception_max_abs_error": max(origin_errors, default=math.nan),
            "post_step_qpos_diagnostic_max_abs_error": max(post_step_qpos_errors, default=math.nan),
            "post_step_reference_action_diagnostic_max_abs_error": max(post_step_action_errors, default=math.nan),
            "post_step_observation_diagnostic_max_abs_error": max(post_step_observation_errors, default=math.nan),
            "post_step_reward_diagnostic_max_abs_error": max(post_step_reward_errors, default=math.nan),
            "restore_rerender_diagnostic_max_abs_error": max(rerender_errors, default=math.nan),
            "replay_teammate_action_max_abs_error": max(replay_action_errors, default=math.nan),
            "replay_teammate_policy_evaluation_count": replay_policy_calls,
            "invalid_candidate_count": int(source["measurements"]["invalid_candidate_count"]),
            "invalid_branch_status_count": int(source["measurements"]["invalid_branch_status_count"]),
            "source_artifact_count": len(local_artifacts),
            "source_artifact_mismatches": artifact_mismatches,
        },
        "source_artifact_root": str(args.family_root.resolve()),
        "claim_limits": [
            "A5R2 did not pass its original 1e-6/1e-4 rule; A5R3 transparently replaces that unsuitable rule.",
            "Post-step qpos, action, pixel, and raw-reward differences remain reported as simulator-noise diagnostics.",
            "Any effect no larger than 0.02 is treated as unresolved simulator noise and cannot support candidate or team-response claims.",
            "No non-reference candidate utility was read or summarized during this rescore.",
            "Gate A, Gate B, formal collection, and CARE training remain unexecuted.",
        ],
    }
    atomic_json(args.output, receipt)
    print(json.dumps({"output": str(args.output), "passed": passed, "checks": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
