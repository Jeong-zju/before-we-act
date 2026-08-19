#!/usr/bin/env python3
"""Create immutable sidecar quality labels for A5R7 without reading candidate outcomes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from scripts.before_we_act.rescore_care_branch_pilot import (
    SCIENTIFIC_RESOLUTION,
    outcome_discrete_equal,
)


EXPECTED_STAGE = "A5R7-CARE-COMMON-SUPPORT-BRANCHES"
EXPECTED_CONTRACT_STAGE = "A4R7-CARE-COMMON-SUPPORT-COLLECTION"
LABEL_STAGE = "A5R7Q1-CARE-SIMULATOR-QUALITY-LABELS"
HORIZONS = (8, 16, 32, 64)
VALID_STATUSES = {"VALID", "SUCCESS_TERMINATION"}


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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def verify_source_artifacts(
    source_receipt: Mapping[str, Any], family_root: Path
) -> dict[str, Any]:
    """Verify every immutable family JSON/NPZ against the frozen A5R7 receipt."""
    root = family_root.resolve()
    verified = []
    for item in source_receipt.get("artifacts", []):
        path = Path(str(item["path"])).resolve()
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(f"source artifact is outside family root: {path}") from error
        if not path.is_file():
            raise RuntimeError(f"missing immutable A5R7 artifact: {path}")
        size_bytes = path.stat().st_size
        digest = sha256_file(path)
        if size_bytes != int(item["size_bytes"]) or digest != str(item["sha256"]):
            raise RuntimeError(f"immutable A5R7 artifact drifted: {path}")
        verified.append((str(relative), size_bytes, digest))
    if len(verified) != 360:
        raise RuntimeError(f"expected 360 immutable A5R7 artifacts, found {len(verified)}")
    aggregate = hashlib.sha256()
    for relative, size_bytes, digest in sorted(verified):
        aggregate.update(f"{relative}\0{size_bytes}\0{digest}\n".encode("utf-8"))
    return {"artifact_count": len(verified), "aggregate_sha256": aggregate.hexdigest()}


def branch_by_key(
    row: Mapping[str, Any], candidate_id: int, regime: str, repeat_id: int
) -> Mapping[str, Any]:
    selected = [
        branch
        for branch in row["branches"]
        if int(branch["candidate_id"]) == candidate_id
        and str(branch["regime"]) == regime
        and int(branch["repeat_id"]) == repeat_id
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"{row['snapshot_id']} missing unique branch {(candidate_id, regime, repeat_id)}"
        )
    return selected[0]


def label_horizon(row: Mapping[str, Any], horizon: int) -> dict[str, Any]:
    support_by_repeat = {
        int(item["repeat_id"]): int(item["common_replay_support_steps"])
        for item in row.get("repeat_common_replay_support", [])
    }
    supported = set(support_by_repeat) == {0, 1} and all(
        support_by_repeat[repeat_id] >= horizon for repeat_id in (0, 1)
    )
    measurements: dict[str, Any] = {
        "support_steps_by_repeat": {
            str(repeat_id): support_by_repeat.get(repeat_id) for repeat_id in (0, 1)
        }
    }
    if not supported:
        return {
            "label": "DO_NOT_USE_UNSUPPORTED_HORIZON",
            "use_for_gate_analysis": False,
            "use_for_training": False,
            "reasons": ["COMMON_SUPPORT_SHORTER_THAN_HORIZON"],
            "measurements": measurements,
        }

    all_branches = list(row["branches"])
    coverage_exact = len(all_branches) == 24 and all(
        str(horizon) in branch.get("outcomes", {}) for branch in all_branches
    )
    branch_statuses_valid = all(
        branch.get("status") in VALID_STATUSES for branch in all_branches
    )
    branch_candidates_valid = all(
        bool(branch.get("candidate_valid")) for branch in all_branches
    ) and all(bool(item.get("valid")) for item in row.get("candidate_legality", []))
    if not (coverage_exact and branch_statuses_valid and branch_candidates_valid):
        reasons = []
        if not coverage_exact:
            reasons.append("BRANCH_OR_OUTCOME_MISSING")
        if not branch_statuses_valid:
            reasons.append("INVALID_BRANCH_STATUS")
        if not branch_candidates_valid:
            reasons.append("INVALID_CANDIDATE")
        return {
            "label": "DO_NOT_USE_INVALID_BRANCH",
            "use_for_gate_analysis": False,
            "use_for_training": False,
            "reasons": reasons,
            "measurements": measurements,
        }

    probe = row["restore_probe"]
    restore_sources_exact = probe.get("restore_observation_source") == "captured_snapshot" and all(
        branch.get("restore_observation_source") == "captured_snapshot"
        for branch in all_branches
    )
    restore_error_max = max(
        [float(probe["restore_observation_max_abs_error"])]
        + [float(branch["restore_observation_max_abs_error"]) for branch in all_branches]
    )
    restore_valid = (
        restore_sources_exact
        and restore_error_max <= 1e-6
        and bool(probe["terminal_and_success_exact"])
    )

    references: dict[tuple[str, int], Mapping[str, Any]] = {}
    mode_discrete_equal: dict[str, bool] = {}
    mode_utility_difference: dict[str, float] = {}
    for repeat_id in (0, 1):
        reactive = branch_by_key(row, 0, "reactive", repeat_id)["outcomes"][str(horizon)]
        replay = branch_by_key(row, 0, "replay", repeat_id)["outcomes"][str(horizon)]
        references[("reactive", repeat_id)] = reactive
        references[("replay", repeat_id)] = replay
        mode_discrete_equal[str(repeat_id)] = outcome_discrete_equal(reactive, replay)
        mode_utility_difference[str(repeat_id)] = abs(
            float(reactive["utility_main"]) - float(replay["utility_main"])
        )
    repeat_utility_difference = {
        regime: abs(
            float(references[(regime, 0)]["utility_main"])
            - float(references[(regime, 1)]["utility_main"])
        )
        for regime in ("reactive", "replay")
    }
    measurements.update(
        {
            "restore_observation_max_abs_error": restore_error_max,
            "reference_mode_discrete_equal_by_repeat": mode_discrete_equal,
            "reference_mode_utility_abs_difference_by_repeat": mode_utility_difference,
            "reference_repeat_utility_abs_difference_by_regime": repeat_utility_difference,
        }
    )
    reasons = []
    if not restore_valid:
        reasons.append("RESTORE_ORIGIN_OR_PROBE_MISMATCH")
    if not all(mode_discrete_equal.values()):
        reasons.append("REFERENCE_MODE_DISCRETE_MISMATCH")
    if max(mode_utility_difference.values()) > SCIENTIFIC_RESOLUTION:
        reasons.append("REFERENCE_MODE_UTILITY_MISMATCH")
    if max(repeat_utility_difference.values()) > SCIENTIFIC_RESOLUTION:
        reasons.append("REFERENCE_REPEAT_UTILITY_MISMATCH")
    if reasons:
        return {
            "label": "DO_NOT_USE_SIMULATOR_ANOMALY",
            "use_for_gate_analysis": False,
            "use_for_training": False,
            "reasons": reasons,
            "measurements": measurements,
        }
    return {
        "label": "USE",
        "use_for_gate_analysis": True,
        "use_for_training": False,
        "reasons": [],
        "measurements": measurements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--label-contract", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.summary.exists():
        raise RuntimeError("refusing to overwrite existing A5R7 quality labels")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_contract = json.loads(args.source_contract.read_text(encoding="utf-8"))
    source_receipt = json.loads(args.source_receipt.read_text(encoding="utf-8"))
    label_contract = json.loads(args.label_contract.read_text(encoding="utf-8"))
    if manifest.get("stage_id") != EXPECTED_STAGE:
        raise RuntimeError("wrong A5R7 source manifest stage")
    if source_contract.get("stage_id") != EXPECTED_CONTRACT_STAGE:
        raise RuntimeError("wrong A4R7 source contract stage")
    if label_contract.get("stage_id") != LABEL_STAGE:
        raise RuntimeError("wrong A5R7Q1 label contract stage")
    source = label_contract["source"]
    expected_hashes = {
        "contract_sha256": sha256_file(args.source_contract),
        "manifest_sha256": sha256_file(args.manifest),
        "receipt_sha256": sha256_file(args.source_receipt),
    }
    if any(source[key] != value for key, value in expected_hashes.items()):
        raise RuntimeError("A5R7 label source artifact drifted")
    if manifest.get("contract_sha256") != expected_hashes["contract_sha256"]:
        raise RuntimeError("A5R7 manifest does not match source contract")
    if int(manifest.get("family_count", -1)) != 180 or int(manifest.get("planned_branch_count", -1)) != 4320:
        raise RuntimeError("A5R7 manifest scale drifted")

    source_verification_before = verify_source_artifacts(source_receipt, args.family_root)
    try:
        summary_relative = args.summary.resolve().relative_to(args.output_root.resolve())
    except ValueError as error:
        raise RuntimeError("summary must be located inside output root") from error
    temporary_root = args.output_root.with_name(
        f".{args.output_root.name}.{os.getpid()}.tmp"
    )
    if temporary_root.exists():
        raise RuntimeError(f"temporary output already exists: {temporary_root}")
    temporary_root.mkdir(parents=True)
    counts: Counter[tuple[int, str]] = Counter()
    counts_by_task: Counter[tuple[str, int, str]] = Counter()
    counts_by_stratum: Counter[tuple[str, int, str]] = Counter()
    reason_counts: Counter[str] = Counter()
    artifacts = []
    for family in manifest["families"]:
        source_path = args.family_root / family["task"] / f"{family['snapshot_id']}.json"
        if not source_path.is_file():
            raise RuntimeError(f"missing A5R7 family: {source_path}")
        row = json.loads(source_path.read_text(encoding="utf-8"))
        if (
            row.get("stage_id") != EXPECTED_STAGE
            or row.get("contract_sha256") != expected_hashes["contract_sha256"]
            or int(row.get("branch_count", -1)) != 24
        ):
            raise RuntimeError(f"A5R7 family provenance drifted: {source_path}")
        horizons = {str(horizon): label_horizon(row, horizon) for horizon in HORIZONS}
        sidecar = {
            "format_version": "before-we-act.a5r7q1-care-simulator-quality-sidecar/1",
            "stage_id": LABEL_STAGE,
            "created_at_utc": utc_now(),
            "snapshot_id": row["snapshot_id"],
            "task": row["task"],
            "split": row["split"],
            "sampling_stratum": row["sampling_stratum"],
            "source_family": str(source_path.resolve()),
            "source_family_sha256": sha256_file(source_path),
            "label_contract": str(args.label_contract.resolve()),
            "label_contract_sha256": sha256_file(args.label_contract),
            "nonreference_outcomes_read": False,
            "raw_data_modified": False,
            "horizons": horizons,
            "usable_horizons": [int(key) for key, value in horizons.items() if value["label"] == "USE"],
        }
        final_output = args.output_root / family["task"] / f"{family['snapshot_id']}.quality.json"
        output = temporary_root / family["task"] / f"{family['snapshot_id']}.quality.json"
        atomic_json(output, sidecar)
        artifacts.append(
            {
                "path": str(final_output.resolve()),
                "size_bytes": output.stat().st_size,
                "sha256": sha256_file(output),
            }
        )
        for horizon, value in horizons.items():
            key = (int(horizon), str(value["label"]))
            counts[key] += 1
            counts_by_task[(str(row["task"]), *key)] += 1
            counts_by_stratum[(str(row["sampling_stratum"]), *key)] += 1
            reason_counts.update(value["reasons"])

    source_verification_after = verify_source_artifacts(source_receipt, args.family_root)
    if source_verification_after != source_verification_before:
        raise RuntimeError("immutable A5R7 artifacts changed during quality labeling")
    summary = {
        "format_version": "before-we-act.a5r7q1-care-simulator-quality-summary/1",
        "stage_id": LABEL_STAGE,
        "created_at_utc": utc_now(),
        "status": "COMPLETED_IMMUTABLE_SIDECAR_LABELING",
        "label_contract": str(args.label_contract.resolve()),
        "label_contract_sha256": sha256_file(args.label_contract),
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": expected_hashes["manifest_sha256"],
        "source_receipt": str(args.source_receipt.resolve()),
        "source_receipt_sha256": expected_hashes["receipt_sha256"],
        "family_count": len(artifacts),
        "family_horizon_label_count": len(artifacts) * len(HORIZONS),
        "raw_data_modified": False,
        "raw_data_deleted": False,
        "source_artifact_verification_before": source_verification_before,
        "source_artifact_verification_after": source_verification_after,
        "nonreference_outcomes_read": False,
        "counts_by_horizon_and_label": {
            str(horizon): {
                label: counts[(horizon, label)]
                for label in (
                    "USE",
                    "DO_NOT_USE_SIMULATOR_ANOMALY",
                    "DO_NOT_USE_UNSUPPORTED_HORIZON",
                    "DO_NOT_USE_INVALID_BRANCH",
                )
            }
            for horizon in HORIZONS
        },
        "counts_by_task_horizon_and_label": {
            "|".join(map(str, key)): value for key, value in sorted(counts_by_task.items())
        },
        "counts_by_stratum_horizon_and_label": {
            "|".join(map(str, key)): value for key, value in sorted(counts_by_stratum.items())
        },
        "reason_counts": dict(sorted(reason_counts.items())),
        "permitted_use": label_contract["permitted_use"],
        "posthoc_disclosure": label_contract["posthoc_disclosure"],
        "artifacts": artifacts,
    }
    atomic_json(temporary_root / summary_relative, summary)
    os.replace(temporary_root, args.output_root)
    print(json.dumps({key: summary[key] for key in ("status", "family_count", "family_horizon_label_count", "counts_by_horizon_and_label", "reason_counts")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
