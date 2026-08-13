#!/usr/bin/env python3
"""Run the SSC-V7 M3-R4 successor A1 confirmation and standardized A2 2x2.

This executable never opens the sealed test split and never starts R4-B.  It
supports a preparation phase (fresh confirmation collection and train-only
convergence pilots) and a separately frozen formal confirmation phase.
"""

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
import random
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts.before_we_act import audit_ssc_v7_m2 as m2  # noqa: E402
from scripts.before_we_act import run_ssc_v7_m3 as m3  # noqa: E402
from scripts.before_we_act import run_ssc_v7_m3_r4 as r4  # noqa: E402


STAGE_ID = "SSC-V7-M3-R4-SUCCESSOR-A1"
PREPARATION_STATUS = "FROZEN_SUCCESSOR_PREPARATION_BEFORE_CONFIRMATION_COLLECTION"
FORMAL_STATUS = "FROZEN_SUCCESSOR_EXECUTABLE_BEFORE_CONFIRMATION_METRICS"
TASKS = tuple(m3.TASKS)
TOKEN_COUNT = r4.TOKEN_COUNT
TOKEN_WIDTH = r4.TOKEN_WIDTH
FEATURE_WIDTH = TOKEN_COUNT * TOKEN_WIDTH
PRIMARY_STEPS = r4.PRIMARY_STEPS
ACTION_WIDTH = r4.ACTION_WIDTH
OUTPUT_WIDTH = r4.OUTPUT_WIDTH
HC_INPUT_WIDTH = r4.HC_INPUT_WIDTH
DIRECT_HIDDEN_WIDTH = 97

A1_CONDITIONS = (
    "oracle_arb_query",
    "zero_arb_query",
    "noise_arb_query",
    "label_shuffled_arb_query",
    "episode_shuffled_arb_query",
)
A2_EXTRA_CONDITIONS = (
    "arb_direct",
    "sanitized_legacy_direct",
    "sanitized_legacy_query",
)
FORMAL_CONDITIONS = A1_CONDITIONS + A2_EXTRA_CONDITIONS
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "collect-confirmation-task",
            "merge-confirmation",
            "build-formal-cache",
            "train-hc",
            "train-branch",
            "aggregate",
            "parameter-audit",
        ),
    )
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--condition", choices=FORMAL_CONDITIONS)
    parser.add_argument("--seed-index", type=int, choices=(0, 1, 2))
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


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def load_gate(path: Path) -> dict[str, Any]:
    gate = read_json(path)
    if gate.get("stage_id") != STAGE_ID:
        raise RuntimeError("wrong successor gate identity")
    if gate.get("status") not in {PREPARATION_STATUS, FORMAL_STATUS}:
        raise RuntimeError("successor gate is not frozen")
    unsigned = {key: value for key, value in gate.items() if key != "integrity"}
    payload_hash = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    if payload_hash != str(gate["integrity"]["payload_sha256"]):
        raise RuntimeError("successor gate payload hash mismatch")
    gate["_runtime_gate_sha256"] = sha256_file(path)
    return gate


def preflight(gate: Mapping[str, Any]) -> None:
    implementation = gate["implementation"]
    checks = {
        "successor_script_hash": sha256_file(
            REPOSITORY / str(implementation["script"])
        )
        == str(implementation["script_sha256"]),
        "successor_test_hash": sha256_file(REPOSITORY / str(implementation["test"]))
        == str(implementation["test_sha256"]),
        "base_r4_script_hash": sha256_file(
            REPOSITORY / str(implementation["base_r4_script"])
        )
        == str(implementation["base_r4_script_sha256"]),
        "implementation_is_ancestor": subprocess.call(
            (
                "git",
                "-C",
                str(REPOSITORY),
                "merge-base",
                "--is-ancestor",
                str(implementation["implementation_commit"]),
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
        raise RuntimeError(f"successor preflight failed: {checks}")


def require_status(gate: Mapping[str, Any], expected: str) -> None:
    if gate.get("status") != expected:
        raise RuntimeError(f"command requires gate status {expected}")


def int_agents(values: Iterable[Any]) -> set[int]:
    result: set[int] = set()
    for value in values:
        try:
            result.add(int(value))
        except (TypeError, ValueError):
            continue
    return result


def sanitized_legacy_tokens(label: Mapping[str, Any], own_slot: int) -> np.ndarray:
    """Current relational B without progress, identity, future, P, or T fields."""
    self_contact = self_grasp = self_control = False
    peer_contact = peer_grasp = peer_control = False
    shared = contested = dropped = False
    peer_custodian = self_custodian = missing_custodian = False
    multi_contact = multi_grasp = multi_control = False
    any_object = False
    object_states = label["grasp_contact_custody_state"]
    for state in object_states.values():
        any_object = True
        contacts = int_agents(state.get("contact_agents", []))
        grasps = int_agents(state.get("grasp_agents", []))
        controllers = int_agents(state.get("controller_agents", []))
        self_contact |= own_slot in contacts
        self_grasp |= own_slot in grasps
        self_control |= own_slot in controllers
        peer_contact |= bool(contacts - {own_slot})
        peer_grasp |= bool(grasps - {own_slot})
        peer_control |= bool(controllers - {own_slot})
        shared |= bool(state.get("shared_control", False))
        custodian = state.get("current_custodian")
        try:
            custodian = None if custodian is None else int(custodian)
        except (TypeError, ValueError):
            custodian = None
        self_custodian |= custodian == own_slot
        peer_custodian |= custodian is not None and custodian != own_slot
        missing_custodian |= custodian is None and bool(contacts | grasps | controllers)
        multi_contact |= len(contacts) >= 2
        multi_grasp |= len(grasps) >= 2
        multi_control |= len(controllers) >= 2

    risk = label["collision_drop_contention_risk"]
    contested = bool(risk.get("contested_objects", []))
    dropped = bool(risk.get("dropped_objects", []))
    peers = [
        item
        for item in label["per_agent_contribution"]
        if int(item["agent_slot"]) != own_slot
    ]
    peer_active = [bool(item.get("active", False)) for item in peers]
    peer_roles = {
        str(role)
        for item in peers
        for role in item.get("roles", [])
        if str(role) != "none"
    }
    valid = label["label_validity_mask"]
    values = np.asarray(
        [
            [self_contact, self_grasp, self_control, self_custodian, any_object, bool(peers), shared, not missing_custodian],
            [peer_contact, peer_grasp, peer_control, peer_custodian, any(peer_active), any(not x for x in peer_active), sum(peer_active) >= 2, bool(peer_roles)],
            [shared, multi_contact, multi_grasp, multi_control, self_custodian, peer_custodian, missing_custodian, contested],
            [bool(risk.get("robot_collision", False)), bool(risk.get("robot_proximity_risk", False)), contested, dropped, multi_contact, multi_grasp, shared, missing_custodian],
            ["support" in peer_roles, "receiver" in peer_roles, "handler" in peer_roles, "button_operator" in peer_roles, "custodian" in peer_roles, peer_contact, peer_grasp, peer_control],
            [bool(valid.get("grasp_contact_custody_state", False)), bool(valid.get("collision_drop_contention_risk", False)), bool(valid.get("per_agent_contribution", False)), int(label.get("ambiguity_code", 1)) == 0, any_object, bool(peers), not dropped, True],
        ],
        dtype=np.float32,
    )
    if values.shape != (TOKEN_COUNT, TOKEN_WIDTH):
        raise AssertionError(values.shape)
    return values


@dataclass
class SuccessorBundle:
    base: m3.ProbeData
    arb: np.ndarray
    sanitized: np.ndarray

    def subset(self, indices: np.ndarray) -> "SuccessorBundle":
        return SuccessorBundle(
            self.base.subset(indices), self.arb[indices], self.sanitized[indices]
        )

    def __len__(self) -> int:
        return len(self.base)


def load_successor_bundle(
    manifest_path: Path, splits: set[str]
) -> tuple[SuccessorBundle, dict[str, Any]]:
    if "read_only_test" in splits:
        raise RuntimeError("successor implementation is forbidden from opening test")
    arb_bundle, audit = r4.load_bundle(manifest_path, splits)
    manifest = read_json(manifest_path)
    features: list[np.ndarray] = []
    identities: list[tuple[str, int, int]] = []
    for episode in manifest["episodes"]:
        if str(episode["split"]) not in splits:
            continue
        labels = r4.label_rows(Path(str(episode["sidecar_path"])))
        episode_id = str(episode["hdf5_sha256"])
        with h5py.File(str(episode["hdf5_path"]), "r") as stream:
            action_count = int(stream["data/action/commanded"].shape[0])
            agent_count = int(stream.attrs["agent_count"])
        for frame in m3.uniform_indices(16, action_count - 16, 64):
            label = labels[frame]
            if int(label["ambiguity_code"]) != 0 or not all(
                bool(value) for value in label["label_validity_mask"].values()
            ):
                continue
            for own_slot in range(agent_count):
                features.append(sanitized_legacy_tokens(label, own_slot))
                identities.append((episode_id, frame, own_slot))
    expected = list(
        zip(
            arb_bundle.base.episode_ids.astype(str).tolist(),
            arb_bundle.base.frame_indices.astype(int).tolist(),
            arb_bundle.base.agent_slots.astype(int).tolist(),
            strict=True,
        )
    )
    if identities != expected:
        raise RuntimeError("sanitized feature order does not match probe rows")
    sanitized = np.stack(features).astype(np.float32)
    result_audit = dict(audit)
    result_audit["sanitized_shape"] = list(sanitized.shape)
    result_audit["forbidden_fields_opened"] = 0
    return SuccessorBundle(arb_bundle.base, arb_bundle.arb, sanitized), result_audit


def collect_confirmation_task(args: argparse.Namespace, gate: Mapping[str, Any]) -> None:
    require_status(gate, PREPARATION_STATUS)
    if args.task is None:
        raise ValueError("collect-confirmation-task requires --task")
    if args.output_root.exists():
        raise FileExistsError(f"fresh collection output required: {args.output_root}")
    args.output_root.mkdir(parents=True)
    (args.output_root / "logs").mkdir()
    seed_contract = read_json(args.seed_contract)
    expanded = m2.expanded_seed_manifest(seed_contract, args.w10_seed_root)
    pool = expanded["per_task"][args.task]["expert_candidate_pool"]
    collection = gate["confirmation_collection"]
    first_candidate = int(collection["first_unused_candidate_index_by_task"][args.task])
    required = int(collection["successful_episodes_per_task"])
    candidates = pool[first_candidate:]
    if len(candidates) < required:
        raise RuntimeError("not enough frozen unused candidate seeds")

    base_manifest = read_json(Path(str(gate["data"]["training_manifest"])))
    existing_seeds = {int(item["seed"]) for item in base_manifest["episodes"]}
    if any(int(seed) in existing_seeds for seed in candidates):
        raise RuntimeError("confirmation candidate seed overlaps the existing dataset")

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
                f"[successor confirmation] {args.task} "
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
        "format_version": "ssc-v7.m3_r4_successor.confirmation_collection/1",
        "stage_id": STAGE_ID,
        "task": args.task,
        "purpose": "fresh_successor_confirmation",
        "gate_sha256": gate["_runtime_gate_sha256"],
        "first_candidate_index": first_candidate,
        "required_successes": required,
        "episodes": episodes,
        "attempts": attempts,
        "elapsed_wall_seconds": (
            datetime.now(timezone.utc) - started
        ).total_seconds(),
        "completed_at_utc": utc_now(),
        "test_paths_opened": 0,
        "r4_b_started": False,
    }
    write_json(args.output_root / "task_collection_receipt.json", receipt)
    print(f"SSC_V7_M3_R4_SUCCESSOR_{args.task.upper()}_CONFIRMATION_COLLECTED")


def merge_confirmation(args: argparse.Namespace, gate: Mapping[str, Any]) -> None:
    require_status(gate, PREPARATION_STATUS)
    if args.data_root is None:
        raise ValueError("merge-confirmation requires --data-root")
    if args.output_root.exists():
        raise FileExistsError(f"fresh confirmation manifest root required: {args.output_root}")
    base_path = Path(str(gate["data"]["training_manifest"]))
    if sha256_file(base_path) != str(gate["data"]["training_manifest_sha256"]):
        raise RuntimeError("training manifest hash mismatch")
    base = read_json(base_path)
    base_seeds = {int(item["seed"]) for item in base["episodes"]}
    base_hdf5 = {str(item["hdf5_sha256"]) for item in base["episodes"]}
    episodes: list[dict[str, Any]] = []
    source_receipts: list[dict[str, Any]] = []
    required = int(gate["confirmation_collection"]["successful_episodes_per_task"])
    for task in TASKS:
        receipt_path = args.data_root / task / "task_collection_receipt.json"
        receipt = read_json(receipt_path)
        if (
            receipt.get("stage_id") != STAGE_ID
            or receipt.get("task") != task
            or receipt.get("purpose") != "fresh_successor_confirmation"
            or receipt.get("gate_sha256") != gate["_runtime_gate_sha256"]
        ):
            raise RuntimeError(f"wrong confirmation receipt identity: {receipt_path}")
        if len(receipt["episodes"]) != required:
            raise RuntimeError(f"wrong confirmation episode count for {task}")
        source_receipts.append(
            {"task": task, "path": str(receipt_path), "sha256": sha256_file(receipt_path)}
        )
        for rank, raw in enumerate(receipt["episodes"]):
            item = deepcopy(raw)
            if int(item["success_rank"]) != rank:
                raise RuntimeError("non-canonical confirmation success order")
            if int(item["seed"]) in base_seeds:
                raise RuntimeError("confirmation seed overlaps old data")
            if str(item["hdf5_sha256"]) in base_hdf5:
                raise RuntimeError("confirmation episode overlaps old data")
            for field in ("hdf5_path", "sidecar_path"):
                if not Path(str(item[field])).is_file():
                    raise FileNotFoundError(item[field])
            if sha256_file(Path(str(item["hdf5_path"]))) != str(item["hdf5_sha256"]):
                raise RuntimeError("confirmation HDF5 hash mismatch")
            if sha256_file(Path(str(item["sidecar_path"]))) != str(item["sidecar_sha256"]):
                raise RuntimeError("confirmation sidecar hash mismatch")
            item["split"] = "confirmation"
            item["source_stage_id"] = STAGE_ID
            episodes.append(item)
    manifest = {
        "format_version": "ssc-v7.m3_r4_successor.confirmation_manifest/1",
        "stage_id": STAGE_ID,
        "purpose": "fresh_successor_confirmation",
        "created_at_utc": utc_now(),
        "preparation_gate": str(args.gate),
        "preparation_gate_sha256": gate["_runtime_gate_sha256"],
        "source_receipts": source_receipts,
        "split_counts": {task: {"confirmation": required} for task in TASKS},
        "episodes": episodes,
        "non_overlapping_with_training_manifest": True,
        "existing_tune_used": False,
        "read_only_test_used": False,
        "test_paths_opened": 0,
    }
    args.output_root.mkdir(parents=True)
    manifest_path = args.output_root / "confirmation_manifest.json"
    write_json(manifest_path, manifest)
    receipt = {
        "decision_code": "SSC_V7_M3_R4_SUCCESSOR_CONFIRMATION_DATA_FROZEN",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "episode_count": len(episodes),
        "per_task_episode_count": required,
        "test_paths_opened": 0,
        "r4_b_started": False,
    }
    write_json(args.output_root / "confirmation_manifest_receipt.json", receipt)
    print(receipt["decision_code"])


def feature_statistics(values: np.ndarray) -> dict[str, Any]:
    mean = values.mean(axis=0)
    std = np.maximum(values.std(axis=0), 1e-3)
    return {"mean": mean.astype(float).tolist(), "std": std.astype(float).tolist()}


def normalize_features(values: np.ndarray, statistics: Mapping[str, Any]) -> np.ndarray:
    mean = np.asarray(statistics["mean"], dtype=np.float32)
    std = np.asarray(statistics["std"], dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def save_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez(temporary, **values)
    os.replace(temporary, path)


def bundle_arrays(
    bundle: SuccessorBundle,
    action_norms: Mapping[str, Any],
    arb_norms: Mapping[str, Any],
    sanitized_norms: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    return {
        "legal": bundle.base.legal.astype(np.float32),
        "e0": bundle.base.e0.astype(np.float32),
        "social": bundle.base.social.astype(np.float32),
        "time": bundle.base.time.astype(np.float32),
        "target": bundle.base.target.astype(np.float32),
        "normalized_target": m3.normalized_targets(bundle.base, action_norms),
        "target_mask": bundle.base.target_mask.astype(np.float32),
        "tasks": bundle.base.tasks,
        "episode_ids": bundle.base.episode_ids,
        "frame_indices": bundle.base.frame_indices.astype(np.int32),
        "agent_slots": bundle.base.agent_slots.astype(np.int16),
        "arb": normalize_features(bundle.arb, arb_norms),
        "sanitized": normalize_features(bundle.sanitized, sanitized_norms),
    }


def build_formal_cache(args: argparse.Namespace, gate: Mapping[str, Any]) -> None:
    require_status(gate, FORMAL_STATUS)
    if args.output_root.exists():
        raise FileExistsError(f"fresh formal cache required: {args.output_root}")
    train_manifest = Path(str(gate["data"]["training_manifest"]))
    confirmation_manifest = Path(str(gate["data"]["confirmation_manifest"]))
    if sha256_file(train_manifest) != str(gate["data"]["training_manifest_sha256"]):
        raise RuntimeError("training manifest hash mismatch")
    if sha256_file(confirmation_manifest) != str(
        gate["data"]["confirmation_manifest_sha256"]
    ):
        raise RuntimeError("confirmation manifest hash mismatch")
    train, train_audit = load_successor_bundle(train_manifest, {"train"})
    confirmation, confirmation_audit = load_successor_bundle(
        confirmation_manifest, {"confirmation"}
    )
    train_ids = set(train.base.episode_ids.astype(str).tolist())
    confirmation_ids = set(confirmation.base.episode_ids.astype(str).tolist())
    if train_ids & confirmation_ids:
        raise RuntimeError("formal train and confirmation episodes overlap")
    action_norms = m3.normalizer(train.base)
    arb_norms = feature_statistics(train.arb)
    sanitized_norms = feature_statistics(train.sanitized)
    args.output_root.mkdir(parents=True)
    train_path = args.output_root / "train.npz"
    confirmation_path = args.output_root / "confirmation.npz"
    normalization_path = args.output_root / "normalization.json"
    write_json(
        normalization_path,
        {
            "fit_split": "train only",
            "action": action_norms,
            "arb": arb_norms,
            "sanitized_legacy_B": sanitized_norms,
        },
    )
    save_npz(
        train_path,
        bundle_arrays(train, action_norms, arb_norms, sanitized_norms),
    )
    save_npz(
        confirmation_path,
        bundle_arrays(confirmation, action_norms, arb_norms, sanitized_norms),
    )
    receipt = {
        "format_version": "ssc-v7.m3_r4_successor.formal_cache/1",
        "stage_id": STAGE_ID,
        "completed_at_utc": utc_now(),
        "gate_sha256": gate["_runtime_gate_sha256"],
        "train_manifest_sha256": sha256_file(train_manifest),
        "confirmation_manifest_sha256": sha256_file(confirmation_manifest),
        "train_audit": train_audit,
        "confirmation_audit": confirmation_audit,
        "train_episode_count": len(train_ids),
        "confirmation_episode_count": len(confirmation_ids),
        "episode_overlap_count": 0,
        "normalization_fit_on_train_only": True,
        "normalization": str(normalization_path),
        "normalization_sha256": sha256_file(normalization_path),
        "train_cache": str(train_path),
        "train_cache_sha256": sha256_file(train_path),
        "confirmation_cache": str(confirmation_path),
        "confirmation_cache_sha256": sha256_file(confirmation_path),
        "existing_tune_used": False,
        "test_paths_opened": 0,
        "r4_b_started": False,
    }
    write_json(args.output_root / "cache_receipt.json", receipt)
    print("SSC_V7_M3_R4_SUCCESSOR_FORMAL_CACHE_FROZEN")


@dataclass
class CachedBundle:
    base: m3.ProbeData
    normalized_target: np.ndarray
    arb: np.ndarray
    sanitized: np.ndarray

    def __len__(self) -> int:
        return len(self.base)


def load_cached_bundle(path: Path) -> CachedBundle:
    with np.load(path, allow_pickle=False) as values:
        base = m3.ProbeData(
            legal=values["legal"].copy(),
            e0=values["e0"].copy(),
            social=values["social"].copy(),
            time=values["time"].copy(),
            target=values["target"].copy(),
            target_mask=values["target_mask"].copy(),
            tasks=values["tasks"].copy(),
            episode_ids=values["episode_ids"].copy(),
            frame_indices=values["frame_indices"].copy(),
            agent_slots=values["agent_slots"].copy(),
        )
        return CachedBundle(
            base=base,
            normalized_target=values["normalized_target"].copy(),
            arb=values["arb"].copy(),
            sanitized=values["sanitized"].copy(),
        )


def load_formal_cache(
    root: Path, gate: Mapping[str, Any]
) -> tuple[CachedBundle, CachedBundle, dict[str, Any]]:
    receipt_path = root / "formal_cache" / "cache_receipt.json"
    receipt = read_json(receipt_path)
    if receipt.get("gate_sha256") != gate["_runtime_gate_sha256"]:
        raise RuntimeError("formal cache belongs to another gate")
    for path_key, hash_key in (
        ("train_cache", "train_cache_sha256"),
        ("confirmation_cache", "confirmation_cache_sha256"),
        ("normalization", "normalization_sha256"),
    ):
        if sha256_file(Path(str(receipt[path_key]))) != str(receipt[hash_key]):
            raise RuntimeError(f"formal cache hash mismatch: {path_key}")
    return (
        load_cached_bundle(Path(str(receipt["train_cache"]))),
        load_cached_bundle(Path(str(receipt["confirmation_cache"]))),
        receipt,
    )


def deranged_indices(indices: np.ndarray, seed: int) -> np.ndarray:
    if len(indices) < 2:
        raise RuntimeError("derangement needs at least two rows")
    rng = np.random.default_rng(seed)
    for _ in range(64):
        shuffled = rng.permutation(indices)
        if np.all(shuffled != indices):
            return shuffled
    return np.roll(indices, 1)


def label_shuffle(values: np.ndarray, data: m3.ProbeData, seed: int) -> np.ndarray:
    result = np.empty_like(values)
    for task_index, task in enumerate(TASKS):
        indices = np.flatnonzero(data.tasks == task)
        result[indices] = values[deranged_indices(indices, seed + task_index * 7919)]
    return result


def episode_shuffle(values: np.ndarray, data: m3.ProbeData, seed: int) -> np.ndarray:
    result = np.empty_like(values)
    rng = np.random.default_rng(seed)
    for task in TASKS:
        episode_ids = sorted(set(data.episode_ids[data.tasks == task].tolist()))
        source_ids = episode_ids.copy()
        for _ in range(64):
            rng.shuffle(source_ids)
            if all(source != target for source, target in zip(source_ids, episode_ids)):
                break
        else:
            source_ids = source_ids[1:] + source_ids[:1]
        for target_id, source_id in zip(episode_ids, source_ids, strict=True):
            target_indices = np.flatnonzero(data.episode_ids == target_id)
            source_indices = np.flatnonzero(data.episode_ids == source_id)
            target_order = target_indices[
                np.lexsort(
                    (data.agent_slots[target_indices], data.frame_indices[target_indices])
                )
            ]
            source_order = source_indices[
                np.lexsort(
                    (data.agent_slots[source_indices], data.frame_indices[source_indices])
                )
            ]
            mapped = np.rint(
                np.linspace(0, len(source_order) - 1, len(target_order))
            ).astype(np.int64)
            result[target_order] = values[source_order[mapped]]
    return result


def side_input(bundle: CachedBundle, hc_seed: int, features: np.ndarray) -> np.ndarray:
    if features.shape != (len(bundle), TOKEN_COUNT, TOKEN_WIDTH):
        raise ValueError(f"wrong side feature shape: {features.shape}")
    reliability = np.ones((len(bundle), 1), dtype=np.float32)
    return np.concatenate(
        (
            r4.hc_input(bundle, hc_seed),
            features.reshape(len(bundle), -1),
            reliability,
        ),
        axis=1,
    ).astype(np.float32)


class DirectResidualFactory:
    @staticmethod
    def create(payload: Mapping[str, Any], seed: int) -> Any:
        torch = r4.torch_setup(seed)

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.hc = r4.build_frozen_hc(payload)
                self.fusion = torch.nn.Sequential(
                    torch.nn.LayerNorm(256 + FEATURE_WIDTH),
                    torch.nn.Linear(
                        256 + FEATURE_WIDTH, DIRECT_HIDDEN_WIDTH
                    ),
                    torch.nn.SiLU(),
                )
                self.residual = torch.nn.Linear(
                    DIRECT_HIDDEN_WIDTH, PRIMARY_STEPS * ACTION_WIDTH
                )
                self.gate = torch.nn.Linear(DIRECT_HIDDEN_WIDTH, 1)
                torch.nn.init.zeros_(self.residual.weight)
                torch.nn.init.zeros_(self.residual.bias)

            def forward(self, values: Any) -> Any:
                base, hidden = r4.hc_forward_hidden(self.hc, values)
                first = HC_INPUT_WIDTH
                features = values[:, first : first + FEATURE_WIDTH]
                reliability = values[:, first + FEATURE_WIDTH : first + FEATURE_WIDTH + 1]
                state = self.fusion(torch.cat((hidden, features), dim=1))
                delta = reliability * torch.sigmoid(self.gate(state)) * self.residual(state)
                tail = torch.zeros(
                    (values.shape[0], OUTPUT_WIDTH - delta.shape[1]),
                    device=values.device,
                    dtype=values.dtype,
                )
                return base + torch.cat((delta, tail), dim=1)

        return Model()


def trainable_parameter_count(model: Any) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def parameter_counts(seed: int) -> dict[str, Any]:
    hc = m3.build_action_model(HC_INPUT_WIDTH, 256, seed)
    payload = {
        "state_dict": hc.state_dict(),
        "input_width": HC_INPUT_WIDTH,
        "hidden_width": 256,
    }
    query = r4.ArbResidualFactory.create(payload, seed + 1)
    direct = DirectResidualFactory.create(payload, seed + 1)
    query_count = trainable_parameter_count(query)
    direct_count = trainable_parameter_count(direct)
    spread = abs(query_count - direct_count) / max(query_count, direct_count)
    return {
        "query_attention_residual": query_count,
        "direct_residual_mlp": direct_count,
        "relative_spread": spread,
        "within_5pct": spread <= 0.05,
        "direct_hidden_width": DIRECT_HIDDEN_WIDTH,
    }


def parameter_audit(args: argparse.Namespace, gate: Mapping[str, Any]) -> None:
    counts = parameter_counts(int(gate["seeds"]["hc_initialization_seed"]))
    expected = gate["architecture"]["trainable_parameter_counts"]
    if counts != expected:
        raise RuntimeError(f"parameter count mismatch: actual={counts} expected={expected}")
    output = args.output_root / "parameter_audit.json"
    if output.exists():
        raise FileExistsError(f"fresh parameter audit required: {output}")
    write_json(
        output,
        {
            "stage_id": STAGE_ID,
            "gate_sha256": gate["_runtime_gate_sha256"],
            "counts": counts,
            "test_paths_opened": 0,
            "r4_b_started": False,
        },
    )
    print("SSC_V7_M3_R4_SUCCESSOR_PARAMETER_MATCH_PASSED")


def train_hc(args: argparse.Namespace, gate: Mapping[str, Any]) -> None:
    require_status(gate, FORMAL_STATUS)
    output = args.output_root / "formal" / "hc"
    if output.exists():
        raise FileExistsError(f"fresh successor HC output required: {output}")
    output.mkdir(parents=True)
    train, confirmation, cache_receipt = load_formal_cache(args.output_root, gate)
    hc_seed = int(gate["seeds"]["hc_noise_seed"])
    training_config = gate["training"]
    model, training = m3.train_action_model(
        r4.hc_input(train, hc_seed),
        train.normalized_target,
        train.base.target_mask,
        r4.hc_input(confirmation, hc_seed),
        confirmation.base,
        confirmation.normalized_target,
        args.device,
        float(training_config["hc_learning_rate"]),
        256,
        int(gate["seeds"]["hc_initialization_seed"]),
        int(gate["seeds"]["hc_sampler_seed"]),
        int(training_config["max_epochs"]),
        int(training_config["patience"]),
    )
    converged = (
        int(training["epochs_run"]) - 1 - int(training["best_epoch"])
        >= int(training_config["patience"])
    )
    checkpoint = output / "action_HC.pt"
    r4.save_checkpoint(
        checkpoint,
        model,
        {"input_width": HC_INPUT_WIDTH, "hidden_width": 256, "condition": "HC"},
    )
    metric = m3.evaluate_model(
        model,
        r4.hc_input(confirmation, hc_seed),
        confirmation.base,
        confirmation.normalized_target,
        args.device,
    )
    metrics_path = output / "confirmation_metrics.json"
    write_json(metrics_path, metric)
    receipt = {
        "format_version": "ssc-v7.m3_r4_successor.hc_receipt/1",
        "stage_id": STAGE_ID,
        "completed_at_utc": utc_now(),
        "gate_sha256": gate["_runtime_gate_sha256"],
        "cache_receipt_sha256": sha256_file(
            args.output_root / "formal_cache" / "cache_receipt.json"
        ),
        "training": training,
        "converged_by_patience": converged,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "metrics": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "decision_code": (
            "SSC_V7_M3_R4_SUCCESSOR_HC_FROZEN"
            if converged
            else "EXPLORATORY_HC_BEST_CHECKPOINT_AT_CAP"
        ),
        "strict_gate_eligible": converged,
        "signal_first_execution_may_continue": True,
        "confirmation_episode_count": cache_receipt["confirmation_episode_count"],
        "test_paths_opened": 0,
        "r4_b_started": False,
    }
    write_json(output / "hc_receipt.json", receipt)
    print(receipt["decision_code"])


def load_hc(
    root: Path, gate: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = read_json(root / "formal" / "hc" / "hc_receipt.json")
    if receipt.get("gate_sha256") != gate["_runtime_gate_sha256"]:
        raise RuntimeError("successor HC belongs to another gate")
    checkpoint = Path(str(receipt["checkpoint"]))
    if sha256_file(checkpoint) != str(receipt["checkpoint_sha256"]):
        raise RuntimeError("successor HC checkpoint hash mismatch")
    return receipt, m3.load_torch_checkpoint(checkpoint, "cpu")


def branch_features(
    condition: str,
    bundle: CachedBundle,
    gate: Mapping[str, Any],
    seed_index: int,
) -> np.ndarray:
    if condition in {"oracle_arb_query", "arb_direct"}:
        return bundle.arb
    if condition == "zero_arb_query":
        return np.zeros_like(bundle.arb)
    if condition == "noise_arb_query":
        noise_seed = int(gate["seeds"]["noise_input"][seed_index])
        noise = np.random.default_rng(noise_seed).normal(
            0.0, 1.0, size=(TOKEN_COUNT, TOKEN_WIDTH)
        ).astype(np.float32)
        return np.broadcast_to(noise, bundle.arb.shape).copy()
    if condition == "label_shuffled_arb_query":
        return label_shuffle(
            bundle.arb,
            bundle.base,
            int(gate["seeds"]["label_shuffle"][seed_index]),
        )
    if condition == "episode_shuffled_arb_query":
        return episode_shuffle(
            bundle.arb,
            bundle.base,
            int(gate["seeds"]["episode_shuffle"][seed_index]),
        )
    if condition in {"sanitized_legacy_direct", "sanitized_legacy_query"}:
        return bundle.sanitized
    raise ValueError(condition)


def train_branch(args: argparse.Namespace, gate: Mapping[str, Any]) -> None:
    require_status(gate, FORMAL_STATUS)
    if args.condition is None or args.seed_index is None:
        raise ValueError("train-branch requires --condition and --seed-index")
    output = (
        args.output_root
        / "formal"
        / "branches"
        / args.condition
        / f"seed_{args.seed_index}"
    )
    if output.exists():
        raise FileExistsError(f"fresh branch output required: {output}")
    output.mkdir(parents=True)
    train, confirmation, _ = load_formal_cache(args.output_root, gate)
    hc_receipt, hc_payload = load_hc(args.output_root, gate)
    init_seed = int(gate["seeds"]["residual_initialization"][args.seed_index])
    sampler_seed = int(gate["seeds"]["residual_sampler"][args.seed_index])
    factory = (
        DirectResidualFactory
        if args.condition in {
            "arb_direct",
            "sanitized_legacy_direct",
        }
        else r4.ArbResidualFactory
    )
    model = factory.create(hc_payload, init_seed)
    train_features = branch_features(args.condition, train, gate, args.seed_index)
    confirmation_features = branch_features(
        args.condition, confirmation, gate, args.seed_index
    )
    hc_seed = int(gate["seeds"]["hc_noise_seed"])
    train_x = side_input(train, hc_seed, train_features)
    confirmation_x = side_input(confirmation, hc_seed, confirmation_features)
    training_config = gate["training"]
    model, training = r4.train_residual(
        model,
        train_x,
        train.normalized_target,
        train.base.target_mask,
        confirmation_x,
        confirmation.base,
        confirmation.normalized_target,
        args.device,
        float(training_config["residual_learning_rate"]),
        sampler_seed,
        int(training_config["max_epochs"]),
        int(training_config["patience"]),
    )
    checkpoint = output / "action_residual.pt"
    r4.save_checkpoint(
        checkpoint,
        model,
        {
            "condition": args.condition,
            "seed_index": args.seed_index,
            "input_width": int(train_x.shape[1]),
        },
    )
    metric = m3.evaluate_model(
        model,
        confirmation_x,
        confirmation.base,
        confirmation.normalized_target,
        args.device,
    )
    metrics_path = output / "confirmation_metrics.json"
    write_json(metrics_path, metric)
    converged = bool(training["converged_by_patience"])
    receipt = {
        "format_version": "ssc-v7.m3_r4_successor.branch_receipt/1",
        "stage_id": STAGE_ID,
        "completed_at_utc": utc_now(),
        "condition": args.condition,
        "seed_index": args.seed_index,
        "initialization_seed": init_seed,
        "sampler_seed": sampler_seed,
        "gate_sha256": gate["_runtime_gate_sha256"],
        "hc_checkpoint_sha256": hc_receipt["checkpoint_sha256"],
        "training": training,
        "trainable_parameter_count": trainable_parameter_count(model),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "metrics": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "decision_code": (
            "SSC_V7_M3_R4_SUCCESSOR_BRANCH_CONVERGED"
            if converged
            else "EXPLORATORY_BRANCH_BEST_CHECKPOINT_AT_CAP"
        ),
        "strict_gate_eligible": converged,
        "signal_first_execution_may_continue": True,
        "test_paths_opened": 0,
        "r4_b_started": False,
    }
    write_json(output / "branch_receipt.json", receipt)
    print(f"{receipt['decision_code']} {args.condition} seed={args.seed_index}")


def mean_metrics(metrics: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    if not metrics:
        raise ValueError("empty metric collection")
    episode_ids = tuple(metrics[0]["episode_errors"])
    episodes: dict[str, dict[str, Any]] = {}
    for episode_id in episode_ids:
        values = [item["episode_errors"][episode_id] for item in metrics]
        if any(value["task"] != values[0]["task"] for value in values):
            raise RuntimeError("metric task mismatch")
        episodes[episode_id] = {
            "task": values[0]["task"],
            "primary_16_nrmse": float(
                np.mean([value["primary_16_nrmse"] for value in values])
            ),
            "diagnostic_100_masked_nrmse": float(
                np.mean([value["diagnostic_100_masked_nrmse"] for value in values])
            ),
            "gripper_16_rmse": float(
                np.mean([value["gripper_16_rmse"] for value in values])
            ),
            "rows": int(values[0]["rows"]),
        }
    per_task = {
        task: float(
            np.mean(
                [
                    value["primary_16_nrmse"]
                    for value in episodes.values()
                    if value["task"] == task
                ]
            )
        )
        for task in TASKS
    }
    return {
        "episode_errors": episodes,
        "task_macro_primary_16_nrmse": float(np.mean(list(per_task.values()))),
        "per_task_primary_16_nrmse": per_task,
        "aggregation": label,
    }


def load_branch_results(
    root: Path, gate: Mapping[str, Any]
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    metrics: dict[str, list[dict[str, Any]]] = {}
    receipts: dict[str, list[dict[str, Any]]] = {}
    for condition in FORMAL_CONDITIONS:
        metrics[condition] = []
        receipts[condition] = []
        for seed_index in range(3):
            branch_root = (
                root / "formal" / "branches" / condition / f"seed_{seed_index}"
            )
            receipt = read_json(branch_root / "branch_receipt.json")
            if receipt.get("gate_sha256") != gate["_runtime_gate_sha256"]:
                raise RuntimeError("branch gate hash mismatch")
            metric_path = Path(str(receipt["metrics"]))
            if sha256_file(metric_path) != str(receipt["metrics_sha256"]):
                raise RuntimeError("branch metric hash mismatch")
            receipts[condition].append(receipt)
            metrics[condition].append(read_json(metric_path))
    return metrics, receipts


def interaction_summary(
    arb_direct: Mapping[str, Any],
    arb_query: Mapping[str, Any],
    legacy_direct: Mapping[str, Any],
    legacy_query: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for episode_id, direct in arb_direct["episode_errors"].items():
        aq = arb_query["episode_errors"][episode_id]
        ld = legacy_direct["episode_errors"][episode_id]
        lq = legacy_query["episode_errors"][episode_id]
        arb_effect = (direct["primary_16_nrmse"] - aq["primary_16_nrmse"]) / max(
            direct["primary_16_nrmse"], 1e-12
        )
        legacy_effect = (
            ld["primary_16_nrmse"] - lq["primary_16_nrmse"]
        ) / max(ld["primary_16_nrmse"], 1e-12)
        rows[episode_id] = {
            "task": direct["task"],
            "gain": float(arb_effect - legacy_effect),
        }
    value, lower, upper, per_task = m3.bootstrap_gain(rows, seed)
    return {
        "macro_interaction": value,
        "ci95": [lower, upper],
        "per_task": per_task,
        "episode_interactions": rows,
        "definition": "ARB query-vs-direct relative gain minus sanitized-legacy query-vs-direct relative gain",
    }


def aggregate(args: argparse.Namespace, gate: Mapping[str, Any]) -> None:
    require_status(gate, FORMAL_STATUS)
    output = args.output_root / "formal" / "successor_a1_a2_receipt.json"
    if output.exists():
        raise FileExistsError(f"fresh aggregate output required: {output}")
    hc_receipt, _ = load_hc(args.output_root, gate)
    hc_metrics_path = Path(str(hc_receipt["metrics"]))
    if sha256_file(hc_metrics_path) != str(hc_receipt["metrics_sha256"]):
        raise RuntimeError("HC metric hash mismatch")
    hc_metrics = read_json(hc_metrics_path)
    loaded, branch_receipts = load_branch_results(args.output_root, gate)
    medians = {
        condition: r4.median_metrics(values) for condition, values in loaded.items()
    }
    statistics_seed = int(gate["seeds"]["statistics_seed"])
    oracle = m3.summarize_gain(
        hc_metrics, medians["oracle_arb_query"], statistics_seed
    )
    control_names = (
        "zero_arb_query",
        "noise_arb_query",
        "label_shuffled_arb_query",
        "episode_shuffled_arb_query",
    )
    comparisons = {
        condition: m3.summarize_gain(
            medians[condition],
            medians["oracle_arb_query"],
            statistics_seed + offset,
        )
        for offset, condition in enumerate(control_names, start=1)
    }
    seed_summaries = [
        m3.summarize_gain(hc_metrics, metric, statistics_seed + 20 + seed_index)
        for seed_index, metric in enumerate(loaded["oracle_arb_query"])
    ]
    acceptance = gate["a1_acceptance"]
    stable_harms = r4.stable_task_harms(
        oracle, float(acceptance["stable_task_harm_threshold_abs"])
    )
    a1_converged = bool(hc_receipt["strict_gate_eligible"]) and all(
        bool(receipt["strict_gate_eligible"])
        for condition in A1_CONDITIONS
        for receipt in branch_receipts[condition]
    )
    strict_checks = {
        "all_required_training_converged": a1_converged,
        "oracle_gain_at_least_3pct": float(oracle["macro_gain"])
        >= float(acceptance["oracle_gain_min"]),
        "oracle_ci_lower_positive": float(oracle["ci95"][0]) > 0.0,
        "at_least_two_positive_tasks": len(oracle["positive_tasks"])
        >= int(acceptance["positive_tasks_min"]),
        "no_stable_task_harm_at_3pct": not stable_harms,
        "at_least_two_of_three_seeds_positive": sum(
            float(item["macro_gain"]) > 0.0 for item in seed_summaries
        )
        >= 2,
        "no_seed_stably_harmed_at_3pct": all(
            float(item["ci95"][1])
            > -float(acceptance["stable_task_harm_threshold_abs"])
            for item in seed_summaries
        ),
        **{
            f"beats_{condition}_ci": float(summary["ci95"][0]) > 0.0
            for condition, summary in comparisons.items()
        },
    }
    if not a1_converged:
        formal_a1 = "INCONCLUSIVE_M3_R4_A1"
    elif all(strict_checks.values()):
        formal_a1 = "PASSED_M3_R4_A1_CONFIRMED_ORACLE_UTILITY"
    else:
        formal_a1 = "FAILED_M3_R4_A1_NO_CONFIRMED_ORACLE_UTILITY"

    point_control_wins = {
        condition: float(summary["macro_gain"]) > 0.0
        for condition, summary in comparisons.items()
    }
    seed_positive_count = sum(
        float(item["macro_gain"]) > 0.0 for item in seed_summaries
    )
    if (
        float(oracle["macro_gain"]) > 0.0
        and len(oracle["positive_tasks"]) >= 2
        and seed_positive_count >= 2
        and sum(point_control_wins.values()) >= 3
    ):
        signal_code = "PROMISING_DIRECTIONAL_ARB_SIGNAL"
    elif float(oracle["macro_gain"]) > 0.0:
        signal_code = "MIXED_BUT_POSITIVE_ARB_SIGNAL"
    else:
        signal_code = "NO_POSITIVE_ARB_SIGNAL_IN_THIS_RUN"

    arb_average = mean_metrics(
        [medians["oracle_arb_query"], medians["arb_direct"]],
        "mean across frozen query and direct fusion for ARB",
    )
    legacy_average = mean_metrics(
        [
            medians["sanitized_legacy_query"],
            medians["sanitized_legacy_direct"],
        ],
        "mean across frozen query and direct fusion for sanitized legacy B",
    )
    query_average = mean_metrics(
        [medians["oracle_arb_query"], medians["sanitized_legacy_query"]],
        "mean across frozen ARB and sanitized-legacy representations for query fusion",
    )
    direct_average = mean_metrics(
        [medians["arb_direct"], medians["sanitized_legacy_direct"]],
        "mean across frozen ARB and sanitized-legacy representations for direct fusion",
    )
    representation_effect = m3.summarize_gain(
        legacy_average, arb_average, statistics_seed + 100
    )
    fusion_effect = m3.summarize_gain(
        direct_average, query_average, statistics_seed + 101
    )
    interaction = interaction_summary(
        medians["arb_direct"],
        medians["oracle_arb_query"],
        medians["sanitized_legacy_direct"],
        medians["sanitized_legacy_query"],
        statistics_seed + 102,
    )
    a2_converged = all(
        bool(receipt["strict_gate_eligible"])
        for condition in (
            "oracle_arb_query",
            "arb_direct",
            "sanitized_legacy_query",
            "sanitized_legacy_direct",
        )
        for receipt in branch_receipts[condition]
    )
    a2_claims = {
        "all_2x2_training_converged": a2_converged,
        "arb_representation_superiority_supported": a2_converged
        and float(representation_effect["ci95"][0]) > 0.0,
        "query_fusion_superiority_supported": a2_converged
        and float(fusion_effect["ci95"][0]) > 0.0,
    }
    receipt = {
        "format_version": "ssc-v7.m3_r4_successor.a1_a2_receipt/1",
        "stage_id": STAGE_ID,
        "completed_at_utc": utc_now(),
        "gate_sha256": gate["_runtime_gate_sha256"],
        "interpretation_policy": {
            "primary": "exploratory signal strength and mechanism pattern",
            "secondary": "strict thresholds constrain formal claims only",
            "anti_overreaction": "A strict miss does not by itself reject ARB or stop follow-up research.",
        },
        "a1": {
            "exploratory_signal_code": signal_code,
            "oracle_vs_hc": oracle,
            "oracle_vs_controls": comparisons,
            "oracle_per_seed_vs_hc": seed_summaries,
            "point_estimate_control_wins": point_control_wins,
            "stable_task_harms": stable_harms,
            "strict_checks": strict_checks,
            "formal_decision_code": formal_a1,
        },
        "a2_standardized_2x2": {
            "cell_metrics": {
                "ARB__query_attention": medians["oracle_arb_query"],
                "ARB__direct_residual": medians["arb_direct"],
                "sanitized_legacy_B__query_attention": medians[
                    "sanitized_legacy_query"
                ],
                "sanitized_legacy_B__direct_residual": medians[
                    "sanitized_legacy_direct"
                ],
            },
            "representation_main_effect": representation_effect,
            "fusion_main_effect": fusion_effect,
            "interaction": interaction,
            "claims": a2_claims,
        },
        "r4_b_status": "LOCKED_BY_OWNER_INSTRUCTION_NOT_STARTED",
        "r4_b_started": False,
        "test_paths_opened": 0,
        "m4_authorized": False,
        "b_core_authorized": False,
    }
    write_json(output, receipt)
    print(f"{signal_code} / {formal_a1}")


def main() -> None:
    args = parse_args()
    gate = load_gate(args.gate)
    preflight(gate)
    if args.command == "collect-confirmation-task":
        collect_confirmation_task(args, gate)
    elif args.command == "merge-confirmation":
        merge_confirmation(args, gate)
    elif args.command == "build-formal-cache":
        build_formal_cache(args, gate)
    elif args.command == "train-hc":
        train_hc(args, gate)
    elif args.command == "train-branch":
        train_branch(args, gate)
    elif args.command == "aggregate":
        aggregate(args, gate)
    elif args.command == "parameter-audit":
        parameter_audit(args, gate)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
