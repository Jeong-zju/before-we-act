#!/usr/bin/env python3
"""Collect and evaluate the one-time SSC-V7 M3-R4-C sealed test."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import h5py
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts.before_we_act import audit_ssc_v7_m2 as m2  # noqa: E402
from scripts.before_we_act import run_ssc_v7_m3 as m3  # noqa: E402
from scripts.before_we_act import run_ssc_v7_m3_r4 as r4  # noqa: E402
from scripts.before_we_act import run_ssc_v7_m3_r4_b as r4b  # noqa: E402
from scripts.before_we_act import run_ssc_v7_m3_r4_b_supplement as supplement  # noqa: E402
from scripts.before_we_act import run_ssc_v7_m3_r4_successor as successor  # noqa: E402


STAGE_ID = "SSC-V7-M3-R4-C-SEALED-TEST"
FROZEN_STATUS = "FROZEN_R4_C_BEFORE_TEST_GENERATION"
TASKS = tuple(r4b.TASKS)
SOURCE_CONDITIONS = tuple(r4b.CONDITIONS)
SUPPLEMENT_CONDITIONS = tuple(supplement.CONDITIONS)
ORACLE_CONDITION = "oracle_direct"
ALL_CONDITIONS = SOURCE_CONDITIONS + SUPPLEMENT_CONDITIONS + (ORACLE_CONDITION,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("collect-test-task", "merge-test", "evaluate-once")
    )
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--seed-contract",
        type=Path,
        default=Path(
            "/workspace/bwa_runs/ssc-v7-social-state-cooperation-v2/"
            "pre_registration/contracts/seed_contract.json"
        ),
    )
    parser.add_argument(
        "--w10-seed-root",
        type=Path,
        default=Path("/workspace/bwa_runs/w10-six-task-v1/seeds/validation"),
    )
    parser.add_argument(
        "--robofactory-root", type=Path, default=Path("/workspace/RoboFactory")
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_gate(path: Path) -> dict[str, Any]:
    gate = read_json(path)
    if gate.get("stage_id") != STAGE_ID or gate.get("status") != FROZEN_STATUS:
        raise RuntimeError("R4-C gate identity/status is not frozen")
    unsigned = {key: value for key, value in gate.items() if key != "integrity"}
    expected = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    if expected != str(gate["integrity"]["payload_sha256"]):
        raise RuntimeError("R4-C gate payload hash mismatch")
    gate["_runtime_gate_sha256"] = sha256_file(path)
    return gate


def verify_frozen_file(item: Mapping[str, Any]) -> None:
    path = Path(str(item["path"]))
    if sha256_file(path) != str(item["sha256"]):
        raise RuntimeError(f"frozen artifact hash mismatch: {path}")


def preflight(gate: Mapping[str, Any]) -> None:
    implementation = gate["implementation"]
    checks = {
        "script": sha256_file(REPOSITORY / str(implementation["script"]))
        == str(implementation["script_sha256"]),
        "test": sha256_file(REPOSITORY / str(implementation["test"]))
        == str(implementation["test_sha256"]),
        "implementation_is_ancestor": subprocess.call(
            (
                "git",
                "-C",
                str(REPOSITORY),
                "merge-base",
                "--is-ancestor",
                str(implementation["commit"]),
                "HEAD",
            )
        )
        == 0,
        "repository_clean": subprocess.check_output(
            ("git", "-C", str(REPOSITORY), "status", "--porcelain"), text=True
        ).strip()
        == "",
    }
    if not all(checks.values()):
        raise RuntimeError(f"R4-C implementation preflight failed: {checks}")
    for item in gate["frozen_artifacts"]:
        verify_frozen_file(item)
    source_receipt = read_json(Path(str(gate["source"]["supplement_receipt"])))
    if not source_receipt.get("r4_c_authorized"):
        raise RuntimeError("signal-first R4-B amendment did not authorize R4-C")
    if int(source_receipt.get("test_paths_opened", -1)) != 0:
        raise RuntimeError("source opened a test path before R4-C")


def existing_episode_identity(gate: Mapping[str, Any]) -> tuple[set[int], set[str]]:
    seeds: set[int] = set()
    hashes: set[str] = set()
    for key in ("training_manifest", "confirmation_manifest"):
        manifest = read_json(Path(str(gate["data"][key])))
        for item in manifest["episodes"]:
            seeds.add(int(item["seed"]))
            hashes.add(str(item["hdf5_sha256"]))
    return seeds, hashes


def collect_test_task(args: argparse.Namespace, gate: Mapping[str, Any]) -> None:
    if args.task is None:
        raise ValueError("collect-test-task requires --task")
    if args.output_root.exists():
        raise FileExistsError(f"fresh R4-C task output required: {args.output_root}")
    args.output_root.mkdir(parents=True)
    (args.output_root / "logs").mkdir()
    collection = gate["test_collection"]
    first_candidate = int(collection["first_unused_candidate_index_by_task"][args.task])
    required = int(collection["successful_episodes_per_task"])
    candidates = [
        int(value) for value in collection["candidate_seeds_by_task"][args.task]
    ]
    if len(candidates) < required:
        raise RuntimeError("not enough frozen unused candidate seeds")
    existing_seeds, _ = existing_episode_identity(gate)
    if any(int(seed) in existing_seeds for seed in candidates):
        raise RuntimeError("R4-C candidate seed overlaps train/confirmation data")

    m2.EpisodeWriter = m3.CompactEpisodeWriter
    from robofactory.planner import solutions

    spec = m2.TASKS[args.task]
    environment = m2.make_env(args.task, args.robofactory_root)
    wrapper = m2.M2AuditWrapper(environment, args.task, args.output_root)
    solver = getattr(solutions, str(spec["solver"]))
    attempts: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc)
    try:
        for offset, seed in enumerate(candidates):
            candidate_index = first_candidate + offset
            if len(episodes) >= required:
                break
            print(
                f"[R4-C sealed collection] {args.task} "
                f"candidate={candidate_index} seed={seed}",
                flush=True,
            )
            wrapper.begin_attempt(int(seed), candidate_index)
            log_path = args.output_root / "logs" / (
                f"{args.task}.candidate_{candidate_index:03d}.seed_{seed}.log"
            )
            try:
                with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log):
                    result = solver(wrapper, seed=int(seed), debug=False, vis=False)
                success = bool(result != -1 and m2.scalar_bool(result[-1]["success"]))
                item = wrapper.finish_attempt(success, args.output_root)
                attempts.append(
                    {
                        "candidate_index": candidate_index,
                        "seed": int(seed),
                        "success": success,
                        "log_path": str(log_path),
                        "log_sha256": sha256_file(log_path),
                    }
                )
                if item is not None:
                    item["success_rank"] = len(episodes)
                    episodes.append(item)
            except Exception:
                wrapper.abort_attempt()
                raise
        if len(episodes) != required:
            raise RuntimeError(
                f"{args.task} produced {len(episodes)}/{required} successful episodes"
            )
    finally:
        wrapper.abort_attempt()
        environment.close()
    receipt = {
        "format_version": "ssc-v7.m3_r4_c.test_task_collection/1",
        "stage_id": STAGE_ID,
        "task": args.task,
        "purpose": "fresh_sealed_r4_c_test",
        "gate_sha256": gate["_runtime_gate_sha256"],
        "first_candidate_index": first_candidate,
        "required_successes": required,
        "episodes": episodes,
        "attempts": attempts,
        "elapsed_wall_seconds": (
            datetime.now(timezone.utc) - started
        ).total_seconds(),
        "completed_at_utc": utc_now(),
        "test_content_loaded_for_metrics": False,
        "test_open_events": 0,
    }
    write_json(args.output_root / "task_collection_receipt.json", receipt)
    print(f"SSC_V7_M3_R4_C_{args.task.upper()}_TEST_COLLECTED_AND_UNREAD")


def merge_test(args: argparse.Namespace, gate: Mapping[str, Any]) -> None:
    if args.data_root is None:
        raise ValueError("merge-test requires --data-root")
    if args.output_root.exists():
        raise FileExistsError(f"fresh sealed test root required: {args.output_root}")
    existing_seeds, existing_hashes = existing_episode_identity(gate)
    episodes: list[dict[str, Any]] = []
    source_receipts: list[dict[str, Any]] = []
    required = int(gate["test_collection"]["successful_episodes_per_task"])
    seen_seeds: set[int] = set()
    seen_hashes: set[str] = set()
    for task in TASKS:
        receipt_path = args.data_root / task / "task_collection_receipt.json"
        receipt = read_json(receipt_path)
        if (
            receipt.get("stage_id") != STAGE_ID
            or receipt.get("task") != task
            or receipt.get("purpose") != "fresh_sealed_r4_c_test"
            or receipt.get("gate_sha256") != gate["_runtime_gate_sha256"]
            or receipt.get("test_content_loaded_for_metrics") is not False
        ):
            raise RuntimeError(f"wrong R4-C task receipt identity: {receipt_path}")
        if len(receipt["episodes"]) != required:
            raise RuntimeError(f"wrong R4-C episode count for {task}")
        source_receipts.append(
            {"task": task, "path": str(receipt_path), "sha256": sha256_file(receipt_path)}
        )
        for rank, raw in enumerate(receipt["episodes"]):
            item = deepcopy(raw)
            seed = int(item["seed"])
            hdf5_hash = str(item["hdf5_sha256"])
            if int(item["success_rank"]) != rank:
                raise RuntimeError("non-canonical R4-C success order")
            if seed in existing_seeds or seed in seen_seeds:
                raise RuntimeError("R4-C seed overlap")
            if hdf5_hash in existing_hashes or hdf5_hash in seen_hashes:
                raise RuntimeError("R4-C episode overlap")
            for field, hash_field in (
                ("hdf5_path", "hdf5_sha256"),
                ("sidecar_path", "sidecar_sha256"),
            ):
                path = Path(str(item[field]))
                if not path.is_file() or sha256_file(path) != str(item[hash_field]):
                    raise RuntimeError(f"R4-C artifact mismatch: {path}")
            seen_seeds.add(seed)
            seen_hashes.add(hdf5_hash)
            item["split"] = "read_only_test"
            item["source_stage_id"] = STAGE_ID
            episodes.append(item)
    manifest = {
        "format_version": "ssc-v7.m3_r4_c.sealed_test_manifest/1",
        "stage_id": STAGE_ID,
        "purpose": "fresh_sealed_r4_c_test",
        "created_at_utc": utc_now(),
        "frozen_gate": str(args.gate),
        "frozen_gate_sha256": gate["_runtime_gate_sha256"],
        "source_receipts": source_receipts,
        "split_counts": {task: {"read_only_test": required} for task in TASKS},
        "episodes": episodes,
        "non_overlapping_with_train_and_confirmation": True,
        "test_is_sealed": True,
        "test_has_been_read": False,
        "test_open_events": 0,
    }
    args.output_root.mkdir(parents=True)
    manifest_path = args.output_root / "test_manifest.json"
    write_json(manifest_path, manifest)
    write_json(
        args.output_root / "test_manifest_receipt.json",
        {
            "decision_code": "SSC_V7_M3_R4_C_TEST_GENERATED_SEALED_UNREAD",
            "gate_sha256": gate["_runtime_gate_sha256"],
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "episode_count": len(episodes),
            "per_task_episode_count": required,
            "test_open_events": 0,
        },
    )
    print("SSC_V7_M3_R4_C_TEST_GENERATED_SEALED_UNREAD")


@dataclass
class TestBundle:
    cached: successor.CachedBundle
    raw_arb: np.ndarray
    audit: dict[str, Any]


def load_test_once(
    manifest_path: Path, data: r4b.StageData, action_norms: Mapping[str, Any]
) -> TestBundle:
    base, audit = m3.load_sealed_test(manifest_path)
    manifest = read_json(manifest_path)
    features: list[np.ndarray] = []
    identities: list[tuple[str, int, int]] = []
    for episode in manifest["episodes"]:
        if str(episode["split"]) != "read_only_test":
            continue
        labels = r4.label_rows(Path(str(episode["sidecar_path"])))
        episode_id = str(episode["hdf5_sha256"])
        with h5py.File(str(episode["hdf5_path"]), "r") as stream:
            action_count = int(stream["data/action/commanded"].shape[0])
            agent_count = int(stream.attrs["agent_count"])
        signatures = {
            own_slot: [r4.relation_signature(label, own_slot) for label in labels]
            for own_slot in range(agent_count)
        }
        for frame in m3.uniform_indices(16, action_count - 16, 64):
            label = labels[frame]
            if int(label["ambiguity_code"]) != 0 or not all(
                bool(value) for value in label["label_validity_mask"].values()
            ):
                continue
            for own_slot in range(agent_count):
                features.append(
                    r4.arb_tokens(labels, frame, own_slot, signatures[own_slot])
                )
                identities.append((episode_id, frame, own_slot))
    expected = list(
        zip(
            base.episode_ids.astype(str).tolist(),
            base.frame_indices.astype(int).tolist(),
            base.agent_slots.astype(int).tolist(),
            strict=True,
        )
    )
    if identities != expected:
        raise RuntimeError("R4-C ARB extraction order differs from test probe rows")
    raw_arb = np.stack(features).astype(np.float32)
    normalized = (raw_arb - data.arb_mean) / data.arb_std
    targets = m3.normalized_targets(base, action_norms)
    cached = successor.CachedBundle(
        base=base,
        normalized_target=targets.astype(np.float32),
        arb=normalized.astype(np.float32),
        sanitized=np.zeros_like(normalized, dtype=np.float32),
    )
    audit = dict(audit)
    audit["arb_shape"] = list(raw_arb.shape)
    audit["test_open_events"] = 1
    return TestBundle(cached=cached, raw_arb=raw_arb, audit=audit)


def load_predictor(
    item: Mapping[str, Any], predictor_receipt: Mapping[str, Any], device: str
) -> Any:
    kind = str(item["kind"])
    details = predictor_receipt["candidate" if kind == "legal" else "time_only_control"]
    payload = m3.load_torch_checkpoint(Path(str(item["path"])), "cpu")
    model = r4b.build_predictor(
        int(payload["input_width"]), 0, int(payload["hidden_width"])
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, np.asarray(details["final_shrinkage_alpha"], dtype=np.float32)


def predictor_outputs(
    gate: Mapping[str, Any], data: r4b.StageData, test: successor.CachedBundle, device: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source_root = Path(str(gate["source"]["r4_b_root"]))
    predictor_receipt = read_json(source_root / "predictors/predictor_receipt.json")
    prior = data.train_y.mean(axis=0)
    output: dict[str, np.ndarray] = {}
    details: dict[str, Any] = {}
    for kind in ("legal", "time_only"):
        item = next(
            entry
            for entry in gate["predictor_checkpoints"]
            if str(entry["kind"]) == kind
        )
        model, alphas = load_predictor(item, predictor_receipt, device)
        values = r4b.predictor_inputs(kind, test)
        logits = r4b.predict_logits(model, values, device)
        probabilities = r4b.calibrated_probabilities(logits, prior, alphas)
        output["candidate" if kind == "legal" else "time_only"] = probabilities
        output[
            "candidate_reliability" if kind == "legal" else "time_only_reliability"
        ] = r4b.predictor_reliability(probabilities)
        details[kind] = {
            "rows": len(probabilities),
            "checkpoint_sha256": item["sha256"],
        }
        del model
    return output, details


def condition_inputs(
    gate: Mapping[str, Any],
    source_gate: Mapping[str, Any],
    supplement_gate: Mapping[str, Any],
    data: r4b.StageData,
    test_bundle: TestBundle,
    cached_predictions: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    bundle = test_bundle.cached
    hc_seed = int(source_gate["source"]["hc_noise_seed"])
    values: dict[str, np.ndarray] = {}
    for condition in SOURCE_CONDITIONS:
        features, reliability = r4b.condition_prediction(
            condition, cached_predictions, bundle, data, source_gate
        )
        values[condition] = np.concatenate(
            (
                r4.hc_input(bundle, hc_seed),
                features.reshape(len(bundle), -1),
                reliability,
            ),
            axis=1,
        ).astype(np.float32)
    for condition in SUPPLEMENT_CONDITIONS:
        features, reliability = supplement.condition_features(
            condition, cached_predictions, bundle, data, supplement_gate
        )
        values[condition] = np.concatenate(
            (
                r4.hc_input(bundle, hc_seed),
                features.reshape(len(bundle), -1),
                reliability,
            ),
            axis=1,
        ).astype(np.float32)
    values[ORACLE_CONDITION] = np.concatenate(
        (
            r4.hc_input(bundle, hc_seed),
            bundle.arb.reshape(len(bundle), -1),
            np.ones((len(bundle), 1), dtype=np.float32),
        ),
        axis=1,
    ).astype(np.float32)
    return values


def checkpoint_item(
    gate: Mapping[str, Any], condition: str, seed_index: int
) -> Mapping[str, Any]:
    return next(
        item
        for item in gate["action_checkpoints"]
        if str(item["condition"]) == condition
        and int(item["seed_index"]) == seed_index
    )


def evaluate_models(
    gate: Mapping[str, Any],
    data: r4b.StageData,
    bundle: successor.CachedBundle,
    inputs: Mapping[str, np.ndarray],
    device: str,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    _, hc_payload, _ = r4b.load_source_hc(r4b.load_gate(Path(str(gate["source"]["r4_b_gate"]))))
    hc_seed = int(gate["source"]["hc_noise_seed"])
    hc_model = r4.HCWrapper.create(hc_payload).to(device).eval()
    hc_metrics = m3.evaluate_model(
        hc_model,
        r4.hc_input(bundle, hc_seed),
        bundle.base,
        bundle.normalized_target,
        device,
    )
    loaded: dict[str, list[dict[str, Any]]] = {condition: [] for condition in ALL_CONDITIONS}
    for condition in ALL_CONDITIONS:
        for seed_index in range(3):
            item = checkpoint_item(gate, condition, seed_index)
            payload = m3.load_torch_checkpoint(Path(str(item["path"])), "cpu")
            model = successor.DirectResidualFactory.create(hc_payload, seed_index).to(device)
            model.load_state_dict(payload["state_dict"])
            model.eval()
            loaded[condition].append(
                m3.evaluate_model(
                    model,
                    inputs[condition],
                    bundle.base,
                    bundle.normalized_target,
                    device,
                )
            )
            del model
    return hc_metrics, loaded


def gate_off_audit(
    gate: Mapping[str, Any],
    bundle: successor.CachedBundle,
    candidate_input: np.ndarray,
) -> dict[str, Any]:
    import torch

    source_gate = r4b.load_gate(Path(str(gate["source"]["r4_b_gate"])))
    _, hc_payload, _ = r4b.load_source_hc(source_gate)
    values = candidate_input[: min(1024, len(bundle))].copy()
    values[:, -1] = 0.0
    hc_values = r4.hc_input(
        bundle, int(source_gate["source"]["hc_noise_seed"])
    )[: len(values)]
    baseline = r4.HCWrapper.create(hc_payload).eval()
    with torch.no_grad():
        expected = baseline(torch.from_numpy(hc_values))
    maximum = 0.0
    for seed_index in range(3):
        item = checkpoint_item(gate, "arb_hat_direct", seed_index)
        payload = m3.load_torch_checkpoint(Path(str(item["path"])), "cpu")
        model = successor.DirectResidualFactory.create(hc_payload, seed_index).eval()
        model.load_state_dict(payload["state_dict"])
        with torch.no_grad():
            actual = model(torch.from_numpy(values))
        maximum = max(maximum, float((actual - expected).abs().max()))
    return {
        "rows_checked": len(values),
        "seed_count": 3,
        "max_abs_difference_from_frozen_hc": maximum,
        "exact_fallback": maximum == 0.0,
    }


def signal_first_checks(
    candidate: Mapping[str, Any],
    per_seed: list[Mapping[str, Any]],
    retention: float,
    calibration: Mapping[str, Any],
    gate_off: Mapping[str, Any],
    test_audit: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    source_authorized: bool,
    expected_episode_count: int,
) -> tuple[dict[str, bool], list[str]]:
    stable_harms = r4.stable_task_harms(
        candidate, float(acceptance["stable_task_harm_threshold_abs"])
    )
    checks = {
        "source_signal_first_r4_b_authorized": source_authorized,
        "candidate_ci_lower_positive": float(candidate["ci95"][0]) > 0.0,
        "at_least_two_positive_tasks": len(candidate["positive_tasks"])
        >= int(acceptance["positive_tasks_min"]),
        "no_stable_task_harm_at_3pct": not stable_harms,
        "at_least_two_of_three_seeds_positive": sum(
            float(item["macro_gain"]) > 0.0 for item in per_seed
        )
        >= 2,
        "no_seed_stably_harmed_at_3pct": all(
            float(item["ci95"][1])
            > -float(acceptance["stable_task_harm_threshold_abs"])
            for item in per_seed
        ),
        "retains_at_least_half_oracle_direct_gain": retention
        >= float(acceptance["oracle_gain_retention_min"]),
        "test_mean_brier_beats_train_rate_constant": float(
            calibration["mean_predictor_brier"]
        )
        < float(calibration["mean_constant_rate_brier"]),
        "test_reliability_bins_directional": bool(
            calibration["error_rises_as_reliability_falls"]
        ),
        "gate_off_exactly_returns_hc": bool(gate_off["exact_fallback"]),
        "fresh_test_has_expected_episode_count": int(test_audit["episode_count"])
        == expected_episode_count,
        "test_opened_exactly_once": int(test_audit["test_open_events"]) == 1,
    }
    return checks, stable_harms


def evaluate_once(args: argparse.Namespace, gate: Mapping[str, Any]) -> None:
    output = args.output_root / "formal/r4_c_sealed_test_receipt.json"
    marker = args.output_root / "formal/test_opened_marker.json"
    if output.exists() or marker.exists():
        raise FileExistsError("R4-C sealed test is strictly one-time and was already opened")
    manifest_path = Path(str(gate["test_manifest_path"]))
    manifest_receipt_path = manifest_path.parent / "test_manifest_receipt.json"
    manifest_receipt = read_json(manifest_receipt_path)
    if (
        manifest_receipt.get("gate_sha256") != gate["_runtime_gate_sha256"]
        or sha256_file(manifest_path) != str(manifest_receipt["manifest_sha256"])
        or int(manifest_receipt.get("test_open_events", -1)) != 0
    ):
        raise RuntimeError("sealed test manifest/receipt boundary mismatch")
    write_json(
        marker,
        {
            "stage_id": STAGE_ID,
            "opened_at_utc": utc_now(),
            "gate_sha256": gate["_runtime_gate_sha256"],
            "manifest_sha256": sha256_file(manifest_path),
            "test_open_events": 1,
            "evaluation_retry_authorized": False,
        },
    )

    source_gate = r4b.load_gate(Path(str(gate["source"]["r4_b_gate"])))
    supplement_gate = supplement.load_gate(
        Path(str(gate["source"]["supplement_gate"]))
    )
    data = r4b.load_stage_data(source_gate)
    normalization = read_json(Path(str(gate["data"]["normalization"])))
    test = load_test_once(manifest_path, data, normalization["action"])
    predictions, predictor_details = predictor_outputs(
        gate, data, test.cached, args.device
    )
    inputs = condition_inputs(
        gate, source_gate, supplement_gate, data, test, predictions
    )
    hc_metrics, loaded = evaluate_models(
        gate, data, test.cached, inputs, args.device
    )
    medians = {
        condition: r4.median_metrics(metrics) for condition, metrics in loaded.items()
    }
    statistics_seed = int(gate["statistics_seed"])
    candidate = m3.summarize_gain(
        hc_metrics, medians["arb_hat_direct"], statistics_seed
    )
    per_seed = [
        m3.summarize_gain(
            hc_metrics,
            loaded["arb_hat_direct"][seed_index],
            statistics_seed + 20 + seed_index,
        )
        for seed_index in range(3)
    ]
    oracle = m3.summarize_gain(
        hc_metrics, medians[ORACLE_CONDITION], statistics_seed + 40
    )
    retention = (
        float(candidate["macro_gain"]) / float(oracle["macro_gain"])
        if float(oracle["macro_gain"]) > 0.0
        else math.nan
    )
    controls = {
        condition: m3.summarize_gain(
            medians[condition],
            medians["arb_hat_direct"],
            statistics_seed + 100 + offset,
        )
        for offset, condition in enumerate(
            tuple(item for item in ALL_CONDITIONS if item not in {"arb_hat_direct", ORACLE_CONDITION})
        )
    }
    test_targets = test.raw_arb.reshape(len(test.cached), -1)
    calibration = r4b.calibration_summary(
        data.train_y,
        test_targets,
        predictions["candidate"],
        predictions["candidate_reliability"],
    )
    gate_off = gate_off_audit(gate, test.cached, inputs["arb_hat_direct"])
    source_supplement = read_json(Path(str(gate["source"]["supplement_receipt"])))
    checks, stable_harms = signal_first_checks(
        candidate,
        per_seed,
        retention,
        calibration,
        gate_off,
        test.audit,
        gate["acceptance"],
        bool(source_supplement["r4_c_authorized"]),
        len(TASKS) * int(gate["test_collection"]["successful_episodes_per_task"]),
    )
    passed = all(checks.values())
    decision = (
        "PASSED_M3_R4_C_SIGNAL_FIRST_SEALED_TEST"
        if passed
        else "FAILED_M3_R4_C_SIGNAL_FIRST_SEALED_TEST"
    )
    receipt = {
        "format_version": "ssc-v7.m3_r4_c.sealed_test_receipt/1",
        "stage_id": STAGE_ID,
        "completed_at_utc": utc_now(),
        "gate_sha256": gate["_runtime_gate_sha256"],
        "test_manifest": str(manifest_path),
        "test_manifest_sha256": sha256_file(manifest_path),
        "test_open_events": 1,
        "test_episode_paths_opened": int(test.audit["test_paths_opened"]),
        "decision_code": decision,
        "passed": passed,
        "m3_r4_passed": passed,
        "m4_authorized": passed,
        "b_core_authorized": False,
        "arb_hat_direct_vs_hc": candidate,
        "arb_hat_direct_per_seed_vs_hc": per_seed,
        "oracle_direct_vs_hc": oracle,
        "oracle_gain_retention": retention,
        "arb_hat_vs_diagnostic_controls": controls,
        "diagnostic_controls_are_non_blocking": True,
        "calibration": calibration,
        "gate_off_audit": gate_off,
        "stable_task_harms": stable_harms,
        "checks": checks,
        "predictor_inference": predictor_details,
        "test_audit": test.audit,
        "interpretation_policy": gate["interpretation_policy"],
    }
    write_json(output, receipt)
    print(decision)


def main() -> None:
    args = parse_args()
    gate = load_gate(args.gate)
    preflight(gate)
    if args.command == "collect-test-task":
        collect_test_task(args, gate)
    elif args.command == "merge-test":
        merge_test(args, gate)
    elif args.command == "evaluate-once":
        evaluate_once(args, gate)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
