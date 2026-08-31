from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .action_target import (
    ACTION_TARGET_CONTRACT_ID,
    ACTION_TARGET_CONTRACT_SHA256,
    ACTION_TARGET_CONTRACT_SCHEMA,
    canonicalize_controller_action_with_audit,
    validate_action_target_contract,
    validate_controller_action,
)
from .dataset import TASKS
from .protocol import (
    FORMAL_CONTROLLER_CORRECTION_ENTRIES,
    FORMAL_CONTROLLER_CORRECTIONS_BY_JOINT,
    FORMAL_CONTROLLER_CORRECTIONS_BY_TASK,
    FORMAL_DATASET_REVISION,
    FORMAL_RCS_API_OUT_OF_RANGE_ENTRIES_DIAGNOSTIC,
    FORMAL_SIM_PARQUET_SHA256,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _check_action_audit(
    data: Path,
    manifest: dict,
    checks: dict[str, bool],
) -> tuple[dict, dict]:
    """Recompute the target audit instead of trusting manifest extrema."""

    contract = manifest.get("action_target_contract")
    try:
        if not isinstance(contract, dict):
            raise ValueError("missing contract")
        validate_action_target_contract(contract)
    except ValueError:
        checks["action_target_contract"] = False
    else:
        checks["action_target_contract"] = bool(
            contract.get("schema") == ACTION_TARGET_CONTRACT_SCHEMA
        )
    receipt_path = data / "action_target_audit.json"
    checks["action_target_receipt"] = receipt_path.is_file() and receipt_path.stat().st_size > 0
    if not checks["action_target_receipt"]:
        return {}, {}
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError):
        checks["action_target_receipt"] = False
        return {}, {}
    checks["action_target_receipt"] &= bool(
        receipt.get("schema") == "before-we-act.duobench.action-target-audit/1"
        and receipt.get("status") == "PASSED"
        and receipt.get("dataset_revision") == FORMAL_DATASET_REVISION
        and receipt.get("contract_id") == ACTION_TARGET_CONTRACT_ID
        and receipt.get("contract_sha256") == ACTION_TARGET_CONTRACT_SHA256
        and isinstance(receipt.get("tasks"), dict)
    )
    manifest_receipt = manifest.get("action_target_audit", {})
    checks["action_target_receipt_hash"] = bool(
        isinstance(manifest_receipt, dict)
        and manifest_receipt.get("path") == receipt_path.name
        and manifest_receipt.get("sha256") == _sha256_file(receipt_path)
        and manifest_receipt.get("contract_id") == ACTION_TARGET_CONTRACT_ID
        and manifest_receipt.get("contract_sha256") == ACTION_TARGET_CONTRACT_SHA256
    )
    recomputed: dict[str, dict] = {}
    checks["raw_action_provenance"] = True
    checks["controller_equivalent_reconstruction"] = True
    checks["action_receipt_counts"] = True
    checks["formal_action_counts"] = True
    checks["source_parquet_identity"] = True
    for task in TASKS:
        raw_path = data / task / "raw_action.npy"
        action_path = data / task / "action.npy"
        if not raw_path.is_file() or not action_path.is_file():
            checks["raw_action_provenance"] = False
            continue
        raw = np.load(raw_path, mmap_mode="r")
        action = np.load(action_path, mmap_mode="r")
        checks["raw_action_provenance"] &= raw.shape == action.shape and raw.ndim == 2 and raw.shape[1] == 16
        if not checks["raw_action_provenance"]:
            continue
        try:
            canonical, summary = canonicalize_controller_action_with_audit(
                np.asarray(raw).reshape(-1, 2, 8)
            )
            validate_controller_action(canonical)
        except (ValueError, FloatingPointError):
            checks["raw_action_provenance"] = False
            continue
        checks["controller_equivalent_reconstruction"] &= bool(
            np.array_equal(np.asarray(action), canonical.reshape(-1, 16))
        )
        summary.update(
            {
                "raw_ik_array_sha256": _sha256_array(np.asarray(raw)),
                "controller_equivalent_array_sha256": _sha256_array(np.asarray(action)),
            }
        )
        recomputed[task] = summary
        expected = receipt.get("tasks", {}).get(task)
        checks["action_receipt_counts"] &= isinstance(expected, dict) and all(
            expected.get(key) == summary.get(key)
            for key in (
                "raw_ik_array_sha256",
                "controller_equivalent_array_sha256",
                "changed_values",
                "changed_joint_values",
                "changed_gripper_values",
                "out_of_controller_range_entries",
                "out_of_controller_range_by_joint",
            )
        )
        checks["formal_action_counts"] &= bool(
            summary["out_of_controller_range_by_joint"]
            == FORMAL_CONTROLLER_CORRECTIONS_BY_TASK[task]
        )
        checks["source_parquet_identity"] &= bool(
            isinstance(expected, dict)
            and expected.get("source_parquet_sha256")
            == FORMAL_SIM_PARQUET_SHA256[task]
            and expected.get("contract_id") == ACTION_TARGET_CONTRACT_ID
            and expected.get("contract_sha256")
            == ACTION_TARGET_CONTRACT_SHA256
            and expected.get("action_encoding")
            == "absolute_joint7_binary_gripper1"
        )
    if len(recomputed) == len(TASKS):
        corrected_by_joint = np.sum(
            [
                np.asarray(
                    recomputed[task]["out_of_controller_range_by_joint"],
                    dtype=np.int64,
                )
                for task in TASKS
            ],
            axis=0,
        ).astype(int).tolist()
        corrected_entries = sum(
            int(recomputed[task]["out_of_controller_range_entries"])
            for task in TASKS
        )
        rcs_diagnostic_entries = sum(
            int(recomputed[task]["outside_rcs_api_limits_diagnostic_entries"])
            for task in TASKS
        )
        checks["formal_action_counts"] &= bool(
            corrected_by_joint == FORMAL_CONTROLLER_CORRECTIONS_BY_JOINT
            and corrected_entries == FORMAL_CONTROLLER_CORRECTION_ENTRIES
            and rcs_diagnostic_entries
            == FORMAL_RCS_API_OUT_OF_RANGE_ENTRIES_DIAGNOSTIC
        )
    else:
        checks["formal_action_counts"] = False
    return receipt, recomputed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    checks = {
        "eleven_tasks": tuple(manifest["tasks"]) == TASKS,
        "all_550_episodes": manifest["total_episodes"] == 550,
        "normalization_finite": True,
        "normalization_nonzero": True,
        "dimensions": True,
        "episode_contiguous": True,
        "gripper_binary": True,
        "image_uint8": True,
        "action_target_contract": False,
        "action_target_receipt": False,
        "action_target_receipt_hash": False,
        "raw_action_provenance": False,
        "controller_equivalent_reconstruction": False,
        "action_receipt_counts": False,
        "formal_action_counts": False,
        "source_parquet_identity": False,
        "causal_recording_alignment": False,
        "causal_normalization_recomputed": False,
    }
    receipt, recomputed_audits = _check_action_audit(args.data, manifest, checks)
    for values in manifest["normalization"].values():
        if isinstance(values, list):
            checks["normalization_finite"] &= bool(np.isfinite(values).all())
    checks["normalization_nonzero"] &= bool(np.asarray(manifest["normalization"]["qpos_std"]).min() >= 1e-4)
    checks["normalization_nonzero"] &= bool(np.asarray(manifest["normalization"]["action_std"]).min() >= 1e-4)
    alignment = manifest.get("recording_alignment", {})
    checks["causal_recording_alignment"] = bool(
        alignment.get("source_row_semantics")
        == "post_action_observation_and_same_row_executed_action"
        and alignment.get("policy_decision_pair")
        == "observation_row_i_to_action_row_i_plus_1"
        and alignment.get("action_lag_rows") == 1
        and alignment.get("source_dataset_immutable") is True
        and manifest.get("total_policy_samples")
        == manifest.get("total_frames") - manifest.get("total_episodes")
    )
    details = {}
    aligned_qposes: list[np.ndarray] = []
    aligned_actions: list[np.ndarray] = []
    for task in TASKS:
        arrays = {name: np.load(args.data / task / f"{name}.npy", mmap_mode="r") for name in ("state", "action", "head", "left", "right", "episodes")}
        n = len(arrays["state"])
        checks["dimensions"] &= arrays["state"].shape == (n, 16) and arrays["action"].shape == (n, 16)
        checks["dimensions"] &= all(len(arrays[name]) == n for name in ("head", "left", "right", "episodes"))
        unique, first = np.unique(arrays["episodes"], return_index=True)
        checks["episode_contiguous"] &= len(unique) == 50 and bool(np.all(np.diff(first) > 0))
        gripper = np.asarray(arrays["action"][:, (7, 15)])
        checks["gripper_binary"] &= bool(np.isin(gripper, (0, 1)).all())
        checks["image_uint8"] &= all(arrays[name].dtype == np.uint8 for name in ("head", "left", "right"))
        ends = np.r_[first[1:], n]
        state_local = arrays["state"].reshape(-1, 2, 8)
        action_local = arrays["action"].reshape(-1, 2, 8)
        for start, end in zip(first, ends, strict=True):
            aligned_qposes.append(np.asarray(state_local[start : end - 1]))
            aligned_actions.append(np.asarray(action_local[start + 1 : end]))
        details[task] = {
            "frames": n,
            "episodes": len(unique),
            "max_steps": manifest["tasks"][task]["validation_max_steps"],
            "action_target_audit": recomputed_audits.get(task, {}),
        }
    qpos = np.concatenate(aligned_qposes).astype(np.float64)
    action = np.concatenate(aligned_actions).astype(np.float64)
    norm = manifest["normalization"]
    checks["causal_normalization_recomputed"] = bool(
        np.allclose(qpos.mean((0, 1)), norm["qpos_mean"], rtol=0, atol=1e-12)
        and np.allclose(
            np.maximum(qpos.std((0, 1)), 1e-4), norm["qpos_std"], rtol=0, atol=1e-12
        )
        and np.allclose(action.mean((0, 1)), norm["action_mean"], rtol=0, atol=1e-12)
        and np.allclose(
            np.maximum(action.std((0, 1)), 1e-4), norm["action_std"], rtol=0, atol=1e-12
        )
        and norm.get("action_lag_rows") == 1
    )
    report = {
        "schema": "duobench-act-audit-v2",
        "passed": all(checks.values()),
        "checks": checks,
        "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
        "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
        "action_target_receipt_sha256": _sha256_file(args.data / "action_target_audit.json")
        if (args.data / "action_target_audit.json").is_file()
        else None,
        "tasks": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
