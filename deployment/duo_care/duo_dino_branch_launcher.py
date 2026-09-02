"""Four-GPU formal CARE branch collection for the DuoBench B-core.

The legacy :mod:`deployment.duo_care.branch_launcher` predates the Duo DINO
reference and loads an ACT/CARE checkpoint.  This entry point is intentionally
separate and has a deliberately narrow interface: a selected
``PredictiveTeamBeliefPolicy`` is the only reference provider accepted by the
launcher.  The supplied B0-H checkpoint is used solely as provenance (the
selected B-core contains the frozen B0-H tensors); it is never instantiated as
the CARE reference.

One subprocess owns one task and one isolated GPU.  Tasks are launched in
waves of at most four, so a partially completed run can be resumed without
re-running finished families.  The branch kernel itself remains the registered
RoboFactory CARE kernel (six candidates, reactive/replay regimes, two repeats).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from deployment.duo_dino_reference.bcore_data import (
    DUO_CARE_MEMORY_SEMANTICS,
    DUO_CARE_MEMORY_TOKENS,
    DUO_CARE_MEMORY_WIDTH,
    validate_b0h_payload,
)
from deployment.duo_dino_reference.bcore_runtime import validate_bcore_payload
from deployment.duo_dino_reference.data import TASKS, load_manifest
from deployment.duo_dino_reference.preprocessing import (
    DINO_NORMALIZATION_ID,
    IMAGE_PREPROCESS_ID,
)
from deployment.duo_care.branch_collection_v2 import (
    KernelConfig,
    ReferenceTerminatedBeforeAnchor,
    advance_to_anchor,
    clip_anchor_for_reference,
    collect_from_anchor,
)
from deployment.duo_care.branch_signal import HORIZONS, stratified_anchor_steps
from deployment.duo_care.care_signal_audit import audit_family_json
from deployment.duo_care.duobench_adapter import (
    DuoBenchEnvironment,
    DuoBcoreProposalProvider,
)
from deployment.duo_act.action_target import (
    ACTION_TARGET_CONTRACT_ID,
    ACTION_TARGET_CONTRACT_SHA256,
)


FORMAT_VERSION = "before-we-act.care-duobench-family-manifest/1"
COLLECTION_VERSION = "before-we-act.care-duobench-dino-bcore-branch-collection/1"
REFERENCE_POLICY_FAMILY = "PredictiveTeamBeliefPolicy"
BASE_POLICY_FAMILY = "TemporalHistoryPolicy"
METHOD_FAMILY = "CARE"
MAX_GPUS = 4
DEFAULT_SEED_START = 20261001
ANCHOR_SAMPLING_CONTRACT = "fixed_stratified_reference_reachability_clip_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def _manifest_root(value: Path) -> Path:
    """Accept either a prepared-data directory or its manifest path."""

    value = value.resolve()
    if value.is_dir():
        return value
    if value.name == "manifest.json":
        return value.parent
    raise FileNotFoundError(f"prepared DuoBench root is not a directory: {value}")


def _require_metadata(mapping: Mapping[str, Any], key: str, expected: Any, context: str) -> None:
    value = mapping.get(key)
    if value != expected:
        raise ValueError(f"{context} differs at {key}: {value!r} != {expected!r}")


def validate_selected_inputs(
    *,
    bcore_checkpoint: Path,
    b0h_checkpoint: Path,
    prepared_data: Path,
    visual_cache: Path,
    dino_model: Path,
) -> dict[str, Any]:
    """Validate all immutable inputs before any worker is started.

    In particular, this function rejects a B0-H payload passed as the B-core
    checkpoint, and verifies that the B-core's frozen-backbone provenance points
    at the exact supplied B0-H bytes.
    """

    prepared_root = _manifest_root(prepared_data)
    manifest = load_manifest(prepared_root, require_formal=True)
    if int(manifest.get("total_episodes", -1)) != 550:
        raise ValueError("formal Duo branch collection requires all 550 demonstrations")
    b0h_checkpoint = b0h_checkpoint.resolve(strict=True)
    bcore_checkpoint = bcore_checkpoint.resolve(strict=True)
    dino_model = dino_model.resolve(strict=True)
    visual_cache = visual_cache.resolve(strict=True)
    b0h = torch.load(b0h_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(b0h, Mapping):
        raise ValueError("B0-H checkpoint is not a mapping")
    b0h_config = dict(validate_b0h_payload(b0h))
    _require_metadata(
        b0h_config, "image_preprocess_id", IMAGE_PREPROCESS_ID, "B0-H config"
    )
    _require_metadata(
        b0h_config,
        "dino_normalization_id",
        DINO_NORMALIZATION_ID,
        "B0-H config",
    )
    b0h_sha = _sha256(b0h_checkpoint)

    bcore = torch.load(bcore_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(bcore, Mapping):
        raise ValueError("B-core checkpoint is not a mapping")
    # ``validate_bcore_payload`` checks the independent deployment format and
    # architecture signatures.  The explicit format comparison below keeps a
    # B0-H checkpoint from ever being accepted as a reference by accident.
    bcore_format = bcore.get("format") or bcore.get("format_version")
    if bcore_format == "before-we-act.duobench.dino-b0h/1":
        raise ValueError("a B0-H checkpoint cannot be used as the CARE reference")
    bcore_config = dict(validate_bcore_payload(bcore))
    for key, expected in (
        ("policy_family", REFERENCE_POLICY_FAMILY),
        ("reference_policy_family", REFERENCE_POLICY_FAMILY),
        ("method_family", METHOD_FAMILY),
        ("benchmark_adapter", "DuoBench"),
        ("vision_backbone", "dinov3_vitb16_frozen"),
        ("image_preprocess_id", IMAGE_PREPROCESS_ID),
        ("dino_normalization_id", DINO_NORMALIZATION_ID),
        ("strictly_decentralized", True),
        ("strict_local", True),
        ("act_provider_allowed", False),
        ("teacher_present", False),
    ):
        value = bcore.get(key, bcore_config.get(key))
        if value != expected:
            raise ValueError(f"selected B-core differs at {key}: {value!r} != {expected!r}")
    source_sha = (
        bcore.get("source_b0h_checkpoint_sha256")
        or bcore_config.get("source_b0h_checkpoint_sha256")
    )
    if source_sha != b0h_sha:
        raise ValueError("selected B-core was not derived from the supplied formal B0-H")
    n2_config = bcore_config.get("n2_config")
    if not isinstance(n2_config, Mapping) or (
        int(n2_config.get("n_belief_tokens", -1))
        + int(n2_config.get("event_capacity", -1))
        != DUO_CARE_MEMORY_TOKENS
        or int(n2_config.get("d_model", -1)) != DUO_CARE_MEMORY_WIDTH
    ):
        raise ValueError("selected B-core does not provide the registered CARE memory")

    cache_receipt = visual_cache / "cache_receipt.json"
    if not cache_receipt.is_file():
        raise FileNotFoundError(f"Duo DINO visual cache receipt is missing: {cache_receipt}")
    cache = json.loads(cache_receipt.read_text(encoding="utf-8"))
    if cache.get("status") not in ("PASSED", "SMOKE"):
        raise ValueError("Duo DINO visual cache is incomplete")
    if cache.get("image_preprocess_id") != IMAGE_PREPROCESS_ID:
        raise ValueError("Duo DINO visual cache preprocess contract differs")

    return {
        "prepared_root": str(prepared_root),
        "prepared_manifest_sha256": _sha256(prepared_root / "manifest.json"),
        "b0h_checkpoint_sha256": b0h_sha,
        "bcore_checkpoint_sha256": _sha256(bcore_checkpoint),
        "bcore_config": bcore_config,
        "visual_cache_receipt_sha256": _sha256(cache_receipt),
        "dino_model": str(dino_model),
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "memory_semantics": DUO_CARE_MEMORY_SEMANTICS,
        "care_memory_tokens": DUO_CARE_MEMORY_TOKENS,
    }


def _family_signal(family: Mapping[str, Any], tolerance: float = 1e-7) -> dict[str, Any]:
    by = {
        (int(row["candidate_id"]), str(row["regime"]), int(row["repeat_id"])): row
        for row in family.get("branches", ())
    }
    report: dict[str, Any] = {}
    for horizon in HORIZONS:
        deltas: list[float] = []
        for regime in ("reactive", "replay"):
            for repeat in (0, 1):
                reference = float(
                    by[(0, regime, repeat)]["outcomes"][str(horizon)]["utility_main"]
                )
                deltas.extend(
                    float(
                        by[(candidate, regime, repeat)]["outcomes"][str(horizon)][
                            "utility_main"
                        ]
                    )
                    - reference
                    for candidate in range(1, 6)
                )
        values = np.asarray(deltas, dtype=np.float64)
        report[str(horizon)] = {
            "nonzero_candidate_advantages": int(np.count_nonzero(np.abs(values) > tolerance)),
            "families_with_signal": int(bool(np.any(np.abs(values) > tolerance))),
            "advantage_linf": float(np.max(np.abs(values), initial=0.0)),
        }
    return report


def _valid_existing_family(
    json_path: Path,
    npz_path: Path,
    *,
    bcore_sha: str,
) -> bool:
    if not json_path.is_file() or not npz_path.is_file():
        return False
    try:
        family = json.loads(json_path.read_text(encoding="utf-8"))
        if (
            family.get("reference_policy_family") != REFERENCE_POLICY_FAMILY
            or family.get("base_policy_family") != BASE_POLICY_FAMILY
            or family.get("method_family") != METHOD_FAMILY
            or family.get("bcore_checkpoint_sha256") != bcore_sha
            or family.get("image_preprocess_id") != IMAGE_PREPROCESS_ID
            or family.get("strict_dino_contract") is not True
            or family.get("act_provider_allowed") is not False
            or family.get("memory_semantics") != DUO_CARE_MEMORY_SEMANTICS
            or family.get("care_memory_tokens") != DUO_CARE_MEMORY_TOKENS
            or family.get("action_target_contract_id") != ACTION_TARGET_CONTRACT_ID
            or family.get("action_target_contract_sha256")
            != ACTION_TARGET_CONTRACT_SHA256
        ):
            return False
        with np.load(npz_path, allow_pickle=False) as arrays:
            return (
                arrays["memory"].shape
                == (DUO_CARE_MEMORY_TOKENS, DUO_CARE_MEMORY_WIDTH)
                and arrays["memory_mask"].shape == (DUO_CARE_MEMORY_TOKENS,)
                and arrays["candidate_chunks"].shape == (6, 100, 8)
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def collect_task(
    *,
    bcore_checkpoint: Path,
    b0h_checkpoint: Path,
    prepared_data: Path,
    visual_cache: Path,
    dino_model: Path,
    output: Path,
    task: str,
    families_per_task: int,
    seed_start: int,
    image_size: int,
    smoke: bool = False,
) -> dict[str, Any]:
    """Collect one task's exact-snapshot families in the worker process."""

    if task not in TASKS:
        raise ValueError(f"unknown DuoBench task: {task}")
    prepared_root = _manifest_root(prepared_data)
    manifest = load_manifest(prepared_root, require_formal=True)
    task_manifest = manifest.get("tasks", {}).get(task, {})
    maximum = int(task_manifest.get("validation_max_steps", 0))
    if maximum <= 0:
        raise ValueError(f"prepared manifest has no validation horizon for {task}")
    provider = DuoBcoreProposalProvider(
        bcore_checkpoint,
        b0h_checkpoint=b0h_checkpoint,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        dino_model=str(dino_model),
        image_height=image_size,
        image_width=image_size,
    )
    bcore_sha = _sha256(bcore_checkpoint)
    b0h_sha = _sha256(b0h_checkpoint)
    anchors = stratified_anchor_steps(
        maximum,
        max_steps=maximum,
        count=int(families_per_task),
        horizon=max(HORIZONS),
        critical_count=min(20, int(families_per_task)),
    )
    task_root = output / task
    task_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    started = time.perf_counter()
    env = DuoBenchEnvironment(task, image_size=image_size)
    try:
        for anchor_row in anchors:
            ordinal = int(anchor_row["ordinal"])
            seed = int(seed_start + TASKS.index(task) * 100_000 + ordinal)
            focal = ordinal % 2
            requested_anchor_step = int(anchor_row["anchor_step"])
            anchor_step = requested_anchor_step
            reference_length: int | None = None
            anchor_clipped = False
            while True:
                snapshot_id = hashlib.sha256(
                    f"duobench-care-dino-bcore-v1|{task}|{seed}|{anchor_step}|arm={focal}".encode()
                ).hexdigest()
                json_path = task_root / f"{snapshot_id}.json"
                npz_path = task_root / f"{snapshot_id}.npz"
                if _valid_existing_family(json_path, npz_path, bcore_sha=bcore_sha):
                    break
                try:
                    anchor = advance_to_anchor(
                        env,
                        provider,
                        task=task,
                        episode_seed=seed,
                        anchor_step=anchor_step,
                        focal_agent=focal,
                        sampling_stratum=str(anchor_row["sampling_stratum"]),
                        snapshot_id=snapshot_id,
                        config=KernelConfig(),
                    )
                    break
                except ReferenceTerminatedBeforeAnchor as error:
                    anchor_step, reference_length = clip_anchor_for_reference(
                        requested_anchor=requested_anchor_step,
                        terminal_step=error.terminal_step,
                        branch_horizon=max(HORIZONS),
                    )
                    anchor_clipped = True
                    print(
                        json.dumps(
                            {
                                "event": "anchor_clipped_for_reference_reachability",
                                "task": task,
                                "ordinal": ordinal,
                                "episode_seed": seed,
                                "requested_anchor_step": requested_anchor_step,
                                "anchor_step": anchor_step,
                                "reference_length": reference_length,
                                "sampling_stratum": anchor_row["sampling_stratum"],
                            }
                        ),
                        flush=True,
                    )
            if not _valid_existing_family(json_path, npz_path, bcore_sha=bcore_sha):
                family, arrays = collect_from_anchor(
                    env, provider, anchor, config=KernelConfig()
                )
                family.update(
                    {
                        "collection_format": COLLECTION_VERSION,
                        "reference_policy_family": REFERENCE_POLICY_FAMILY,
                        "base_policy_family": BASE_POLICY_FAMILY,
                        "method_family": METHOD_FAMILY,
                        "vision": "dinov3_vitb16_frozen",
                        "vision_backbone": "dinov3_vitb16_frozen",
                        "image_preprocess_id": IMAGE_PREPROCESS_ID,
                        "preprocess_id": IMAGE_PREPROCESS_ID,
                        "dino_normalization_id": DINO_NORMALIZATION_ID,
                        "source_policy_action_encoding": "absolute_joint7_binary_gripper1",
                        "action_encoding": "joint_residual7_gripper_absolute1",
                        "strictly_decentralized": True,
                        "strict_local": True,
                        "strict_dino_contract": True,
                        "act_provider_allowed": False,
                        "formal_provider": True,
                        "memory_semantics": DUO_CARE_MEMORY_SEMANTICS,
                        "care_memory_tokens": DUO_CARE_MEMORY_TOKENS,
                        "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
                        "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
                        "bcore_checkpoint": str(bcore_checkpoint.resolve()),
                        "bcore_checkpoint_sha256": bcore_sha,
                        "b0h_checkpoint": str(b0h_checkpoint.resolve()),
                        "b0h_checkpoint_sha256": b0h_sha,
                        "requested_anchor_step": requested_anchor_step,
                        "reference_length": reference_length,
                        "anchor_clipped_for_reachability": anchor_clipped,
                        "anchor_sampling_contract": ANCHOR_SAMPLING_CONTRACT,
                    }
                )
                _atomic_npz(npz_path, arrays)
                _atomic_json(json_path, family)
            audit = audit_family_json(json_path, strict=True)
            if audit["status"] != "PASSED":
                raise RuntimeError(f"branch family audit failed for {snapshot_id}: {audit['errors']}")
            family = json.loads(json_path.read_text(encoding="utf-8"))
            signal = _family_signal(family)
            signals.append(signal)
            records.append(
                {
                    "ordinal": ordinal,
                    "snapshot_id": snapshot_id,
                    "task": task,
                    "episode_seed": seed,
                    "anchor_step": anchor_step,
                    "requested_anchor_step": requested_anchor_step,
                    "reference_length": reference_length,
                    "anchor_clipped_for_reachability": anchor_clipped,
                    "anchor_sampling_contract": ANCHOR_SAMPLING_CONTRACT,
                    "focal_agent": focal,
                    "sampling_stratum": anchor_row["sampling_stratum"],
                    "path": str(json_path.resolve()),
                    "npz": str(npz_path.resolve()),
                    "json_sha256": _sha256(json_path),
                    "npz_sha256": _sha256(npz_path),
                    "signal": signal,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "bcore_family_complete",
                        "task": task,
                        "ordinal": ordinal,
                        "anchor_step": anchor_step,
                        "focal_agent": focal,
                        "signal_h16": signal["16"]["nonzero_candidate_advantages"],
                    }
                ),
                flush=True,
            )
    finally:
        env.close()

    aggregate = {
        str(horizon): {
            "nonzero_candidate_advantages": int(
                sum(row[str(horizon)]["nonzero_candidate_advantages"] for row in signals)
            ),
            "families_with_signal": int(
                sum(row[str(horizon)]["families_with_signal"] for row in signals)
            ),
            "advantage_linf": float(
                max((row[str(horizon)]["advantage_linf"] for row in signals), default=0.0)
            ),
        }
        for horizon in HORIZONS
    }
    errors: list[str] = []
    if not smoke:
        for horizon in (8, 16, 32):
            if aggregate[str(horizon)]["nonzero_candidate_advantages"] == 0:
                errors.append(f"horizon_{horizon}_all_candidate_advantages_zero")
    receipt = {
        "schema": "before-we-act.care-duobench-dino-bcore-task/1",
        "status": "PASSED" if not errors else "FAILED",
        "task": task,
        "families": len(records),
        "branches": 24 * len(records),
        "reference_policy_family": REFERENCE_POLICY_FAMILY,
        "base_policy_family": BASE_POLICY_FAMILY,
        "method_family": METHOD_FAMILY,
        "vision": "dinov3_vitb16_frozen",
        "vision_backbone": "dinov3_vitb16_frozen",
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "source_policy_action_encoding": "absolute_joint7_binary_gripper1",
        "action_encoding": "joint_residual7_gripper_absolute1",
        "strictly_decentralized": True,
        "strict_local": True,
        "strict_dino_contract": True,
        "act_provider_allowed": False,
        "memory_semantics": DUO_CARE_MEMORY_SEMANTICS,
        "care_memory_tokens": DUO_CARE_MEMORY_TOKENS,
        "care_memory_width": DUO_CARE_MEMORY_WIDTH,
        "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
        "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
        "bcore_checkpoint_sha256": bcore_sha,
        "b0h_checkpoint_sha256": b0h_sha,
        "signal": aggregate,
        "errors": errors,
        "records": records,
        "wall_seconds": time.perf_counter() - started,
    }
    _atomic_json(output / f"{task}.receipt.json", receipt)
    if errors:
        raise RuntimeError(f"formal B-core branch signal gate failed: {errors}")
    return receipt


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def merge_task_shards(
    *, output: Path, tasks: Sequence[str], families_per_task: int, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge task workers and emit the supervisor-consumed manifest."""

    family_root = output / "families"
    family_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        source_root = output / "shards" / task / task
        source_files = sorted(source_root.glob("*.json"))
        source_files = [path for path in source_files if not path.name.endswith(".receipt.json")]
        if len(source_files) != int(families_per_task):
            raise RuntimeError(
                f"task {task} emitted {len(source_files)} families, expected {families_per_task}"
            )
        destination_root = family_root / task
        for source in source_files:
            destination = destination_root / source.name
            source_npz = source.with_suffix(".npz")
            if not source_npz.is_file():
                raise FileNotFoundError(source_npz)
            _copy_atomic(source, destination)
            _copy_atomic(source_npz, destination.with_suffix(".npz"))
            family = json.loads(destination.read_text(encoding="utf-8"))
            if family.get("reference_policy_family") != REFERENCE_POLICY_FAMILY:
                raise ValueError(f"merged family {destination} is not B-core sourced")
            rows.append(
                {
                    "snapshot_id": family["snapshot_id"],
                    "task": task,
                    "path": str(destination.resolve()),
                    "npz": str(destination.with_suffix(".npz").resolve()),
                    "family_sha256": _sha256(destination),
                    "npz_sha256": _sha256(destination.with_suffix(".npz")),
                    "episode_seed": int(family["episode_seed"]),
                    "anchor_step": int(family["anchor_step"]),
                    "focal_agent": int(family["focal_agent"]),
                }
            )
    expected = int(families_per_task) * len(tasks)
    if len(rows) != expected:
        raise RuntimeError(f"merged {len(rows)} families, expected {expected}")
    rows.sort(key=lambda row: (TASKS.index(row["task"]), row["snapshot_id"]))
    manifest = {
        "format_version": FORMAT_VERSION,
        "collection_format": COLLECTION_VERSION,
        "status": "COMPLETE",
        "tasks": list(tasks),
        "branches_per_family": 24,
        "family_count": len(rows),
        "families_per_task": int(families_per_task),
        "families": rows,
        "reference_policy_family": REFERENCE_POLICY_FAMILY,
        "base_policy_family": BASE_POLICY_FAMILY,
        "method_family": METHOD_FAMILY,
        "vision": "dinov3_vitb16_frozen",
        "vision_backbone": "dinov3_vitb16_frozen",
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "source_policy_action_encoding": "absolute_joint7_binary_gripper1",
        "action_encoding": "joint_residual7_gripper_absolute1",
        "strictly_decentralized": True,
        "strict_local": True,
        "strict_dino_contract": True,
        "act_provider_allowed": False,
        "memory_semantics": provenance["memory_semantics"],
        "care_memory_tokens": provenance["care_memory_tokens"],
        "care_memory_width": DUO_CARE_MEMORY_WIDTH,
        "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
        "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
        "bcore_checkpoint_sha256": provenance["bcore_checkpoint_sha256"],
        "provider_checkpoint_sha256": provenance["bcore_checkpoint_sha256"],
        "b0h_checkpoint_sha256": provenance["b0h_checkpoint_sha256"],
        "prepared_manifest_sha256": provenance["prepared_manifest_sha256"],
        "visual_cache_receipt_sha256": provenance["visual_cache_receipt_sha256"],
        "dino_model": provenance["dino_model"],
        "created_at_utc": time.time(),
    }
    _atomic_json(family_root / "manifest.json", manifest)
    return manifest


def _worker_command(args: argparse.Namespace, task: str, worker_output: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "deployment.duo_care.duo_dino_branch_launcher",
        "--worker-task",
        task,
        "--bcore-checkpoint",
        str(args.bcore_checkpoint),
        "--b0h-checkpoint",
        str(args.b0h_checkpoint),
        "--prepared-data",
        str(args.prepared_data),
        "--visual-cache",
        str(args.visual_cache),
        "--dino-model",
        str(args.dino_model),
        "--output",
        str(worker_output),
        "--families-per-task",
        str(args.families_per_task),
        "--seed-start",
        str(args.seed_start),
        "--image-size",
        str(args.image_size),
    ]
    if args.smoke:
        command.append("--smoke")
    return command


def _run_workers(args: argparse.Namespace, tasks: Sequence[str]) -> None:
    """Run a dynamic GPU queue with isolated visible GPU IDs.

    The old implementation waited for the slowest task in each wave.  A long
    task (for example carry_pot) could therefore leave three 5090s idle.  This
    scheduler launches the next pending task as soon as a slot exits, while
    preserving deterministic task order, one-GPU isolation, and fail-closed
    cancellation semantics.
    """

    workers = min(int(args.workers), MAX_GPUS)
    if workers < 1:
        raise ValueError("--workers must be positive")
    log_root = args.output / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    pending = list(tasks)
    active: dict[int, tuple[str, subprocess.Popen[bytes], Any]] = {}
    failures: list[str] = []
    repo_root = str(Path(__file__).resolve().parents[2])

    def launch(slot: int, task: str) -> None:
        worker_output = args.output / "shards" / task
        command = _worker_command(args, task, worker_output)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(slot)
        environment["MUJOCO_EGL_DEVICE_ID"] = "0"
        environment["PYTHONPATH"] = repo_root + os.pathsep + environment.get("PYTHONPATH", "")
        log = (log_root / f"{task}.log").open("a", encoding="utf-8")
        process = subprocess.Popen(command, cwd=repo_root, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        active[slot] = (task, process, log)

    try:
        for slot in range(min(workers, len(pending))):
            launch(slot, pending.pop(0))
        while active:
            progressed = False
            for slot, (task, process, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                progressed = True
                log.close()
                del active[slot]
                if code != 0:
                    failures.append(f"{task}:exit={code}")
                elif pending:
                    launch(slot, pending.pop(0))
            if failures:
                raise RuntimeError("Duo B-core branch worker failed: " + ", ".join(failures))
            if not progressed:
                time.sleep(0.2)
    except BaseException:
        for _slot, (_task, process, log) in active.items():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                log.close()
            except Exception:
                pass
        raise


def _worker_main(args: argparse.Namespace) -> None:
    if args.worker_task is None:
        raise ValueError("--worker-task is required for worker mode")
    provenance = validate_selected_inputs(
        bcore_checkpoint=args.bcore_checkpoint,
        b0h_checkpoint=args.b0h_checkpoint,
        prepared_data=args.prepared_data,
        visual_cache=args.visual_cache,
        dino_model=args.dino_model,
    )
    collect_task(
        bcore_checkpoint=args.bcore_checkpoint,
        b0h_checkpoint=args.b0h_checkpoint,
        prepared_data=args.prepared_data,
        visual_cache=args.visual_cache,
        dino_model=args.dino_model,
        output=args.output,
        task=args.worker_task,
        families_per_task=args.families_per_task,
        seed_start=args.seed_start,
        image_size=args.image_size,
        smoke=args.smoke,
    )
    _atomic_json(
        args.output / "worker_provenance.json",
        {
            "status": "PASSED",
            "task": args.worker_task,
            "reference_policy_family": REFERENCE_POLICY_FAMILY,
            **{key: value for key, value in provenance.items() if key != "bcore_config"},
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcore-checkpoint", type=Path, required=True)
    parser.add_argument("--b0h-checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--dino-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families-per-task", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--task", action="append", choices=TASKS)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker-task", choices=TASKS, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.families_per_task < 1:
        raise ValueError("--families-per-task must be positive")
    if args.image_size <= 0 or args.image_size % 16:
        raise ValueError("--image-size must be a positive multiple of 16")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker_task is not None:
        _worker_main(args)
        return
    tasks = tuple(args.task or TASKS)
    provenance = validate_selected_inputs(
        bcore_checkpoint=args.bcore_checkpoint,
        b0h_checkpoint=args.b0h_checkpoint,
        prepared_data=args.prepared_data,
        visual_cache=args.visual_cache,
        dino_model=args.dino_model,
    )
    _run_workers(args, tasks)
    manifest = merge_task_shards(
        output=args.output,
        tasks=tasks,
        families_per_task=args.families_per_task,
        provenance=provenance,
    )
    receipt = {
        "schema": COLLECTION_VERSION,
        "status": "COMPLETE",
        "tasks": list(tasks),
        "family_count": int(manifest["family_count"]),
        "branches": int(manifest["family_count"]) * 24,
        "branches_per_family": 24,
        "reference_policy_family": REFERENCE_POLICY_FAMILY,
        "base_policy_family": BASE_POLICY_FAMILY,
        "method_family": METHOD_FAMILY,
        "vision": "dinov3_vitb16_frozen",
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "strictly_decentralized": True,
        "strict_local": True,
        "act_provider_allowed": False,
        "bcore_checkpoint_sha256": provenance["bcore_checkpoint_sha256"],
        "b0h_checkpoint_sha256": provenance["b0h_checkpoint_sha256"],
        "manifest": str((args.output / "families" / "manifest.json").resolve()),
    }
    _atomic_json(args.output / "collection_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "BASE_POLICY_FAMILY",
    "COLLECTION_VERSION",
    "FORMAT_VERSION",
    "METHOD_FAMILY",
    "REFERENCE_POLICY_FAMILY",
    "collect_task",
    "main",
    "merge_task_shards",
    "validate_selected_inputs",
]
