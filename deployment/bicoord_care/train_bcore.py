"""Train one real BiCoord CARE B-core/TUNE seed.

The frozen B0-H policy supplies decoded action queries and complete base
actions through :mod:`cache_bcore`.  This stage optimizes CARE's unchanged
``TeamBeliefExperiment`` (runtime belief core, removable privileged teacher,
belief residual, and matched direct-history control).  Deployment export then
attaches the learned runtime tensors to the original
``PredictiveTeamBeliefPolicy`` and physically excludes every teacher tensor.

Formal runs are fixed at three seeds, 120,000 updates/seed, effective batch
48, all 1,800 demonstrations, and offline diagnostics every 5,000 updates.
Smoke runs execute the identical graph for 1--10 updates and also emit a real
teacher-free deployment checkpoint for closed-loop interface testing.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from before_we_act.predictive_team_belief_policy import PredictiveTeamBeliefPolicy
from before_we_act.predictive_team_belief_training import (
    TeamBeliefExperiment,
    paired_permutation,
)
from before_we_act.team_belief.losses import (
    TeamBeliefLossWeights,
    compute_team_belief_losses,
)

from .bcore_data import (
    BCORE_CACHE_SCHEMA,
    BCORE_DEPLOYMENT_FORMAT,
    BCORE_SEEDS,
    BCORE_TRAINING_FORMAT,
    BCORE_UPDATES,
    BICOORD_BELIEF_CONFIG,
    BICOORD_CARE_MEMORY_SEMANTICS,
    BICOORD_CARE_MEMORY_TOKENS,
    BICOORD_CARE_MEMORY_WIDTH,
    BICOORD_FUTURE_OFFSETS_STEPS,
    BICOORD_SOURCE_FREQUENCY_HZ,
    DATA_SEED,
    BiCoordPairedSituationBatchSampler,
    BiCoordTeamBeliefDataset,
    fixed_diagnostic_requests,
    validate_b0h_payload,
)
from .config import (
    ACTION_DIM,
    ACTION_ENCODING,
    ACTION_HORIZON,
    DATASET_REVISION,
    D_MODEL,
    DECODER_LAYERS,
    EFFECTIVE_BATCH,
    ENCODER_LAYERS,
    EPISODES_PER_TASK,
    GRIPPER_ENCODING,
    GRIPPER_NATIVE_RANGE,
    HISTORY_LAYERS,
    HISTORY_STEPS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    ROLES,
    ROLE_RANK,
    STATE_DIM,
    TASKS,
    TASK_TEXT,
    TOTAL_EPISODES,
    VISION_BACKBONE,
)
from .data import (
    BiCoordEpisode,
    discover_bicoord_episodes,
    load_normalization_receipt,
)
from .hdf5_data import (
    BiCoordHDF5Reader,
    episode_number,
    sha256_file as hdf5_sha256_file,
    validate_hdf5_schema,
)
from .preprocessing import DINO_NORMALIZATION_ID, IMAGE_PREPROCESS_ID
from .stage_common import (
    artifact,
    atomic_json,
    publish_result,
    read_json,
    require_stage_result,
    sha256_file,
)


EVAL_EVERY = 5_000
FINAL_SUFFICIENCY_WINDOW = 20_000
LR_DROP_UPDATE = 80_000
DEFAULT_LR = 2.0e-4
DEFAULT_WEIGHT_DECAY = 1.0e-4

# Unchanged registered B-core objective weights from the CARE MARS/Duo path.
WEIGHTS = TeamBeliefLossWeights(
    action=1.0,
    action_posterior_kl=0.0,
    teacher_alignment=0.1,
    future_latent=0.01,
    teacher_reconstruction=0.01,
    teammate_delta=0.1,
    teammate_action=0.1,
    exchange_consistency=0.05,
    anti_collapse=0.01,
    action_pairing=1.0,
    action_pairing_margin_fraction=0.1,
    action_pairing_margin_cap=0.01,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _device_batch(raw: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in raw.items()
    }


def _shuffle_permutation(task: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Deterministic within-task/phase negatives for action pairing."""

    result = torch.arange(len(task), device=task.device)
    for task_value in torch.unique(task).tolist():
        phases = torch.unique(phase[task == task_value]).tolist()
        for phase_value in phases:
            rows = torch.nonzero(
                (task == task_value) & (phase == phase_value), as_tuple=False
            ).flatten()
            if len(rows) > 1:
                result[rows] = rows.roll(1)
    return result


def _json_belief_config() -> dict[str, Any]:
    value = asdict(BICOORD_BELIEF_CONFIG)
    value["future_offsets_steps"] = list(BICOORD_BELIEF_CONFIG.future_offsets_steps)
    value["future_offsets_seconds"] = list(
        BICOORD_BELIEF_CONFIG.future_offsets_seconds
    )
    return value


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _diagnostic_rows(payload: Mapping[str, Any]) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list):
        return rows
    for item in evaluations:
        if not isinstance(item, Mapping):
            continue
        try:
            update = int(item["update"])
            score = float(item["validation"]["macro"]["b_core"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(score):
            rows.append((update, score))
    return rows


def _selected_diagnostic(payload: Mapping[str, Any], *, formal: bool) -> tuple[int, float]:
    rows = _diagnostic_rows(payload)
    if formal:
        rows = [
            row
            for row in rows
            if BCORE_UPDATES - FINAL_SUFFICIENCY_WINDOW <= row[0] <= BCORE_UPDATES
        ]
    if not rows:
        raise ValueError("B-core checkpoint has no finite offline diagnostic in the sufficiency window")
    return min(rows, key=lambda row: (row[1], row[0]))


@torch.no_grad()
def evaluate_offline(
    model: TeamBeliefExperiment,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate frozen cached labels only; no simulator outcome enters here."""

    model.eval()
    values: dict[str, list[float]] = {
        "b0h": [],
        "b_core": [],
        "b_shuffle": [],
        "direct_reactive": [],
    }
    auxiliary: dict[str, list[float]] = {}
    residual_target: list[torch.Tensor] = []
    residual_output: list[torch.Tensor] = []
    for raw in loader:
        batch = _device_batch(raw, device)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model(batch)
            negative = _shuffle_permutation(batch["task_index"], batch["phase_bin"])
            shuffled_residual, _ = model.belief_residual(
                batch["decoded_action_hidden"],
                output.candidate.belief.mu[negative],
                output.candidate.belief.sigma[negative],
                output.candidate.belief.reliability[negative],
            )
            shuffled = batch["base_action"] + shuffled_residual
        predictions = {
            "b0h": batch["base_action"],
            "b_core": output.candidate.prediction,
            "b_shuffle": shuffled,
            "direct_reactive": output.direct_prediction,
        }
        for name, prediction in predictions.items():
            squared = (prediction - batch["action"]).float().square().mean(-1)
            per_row = (squared * batch["action_mask"]).sum(-1) / batch[
                "action_mask"
            ].sum(-1).clamp_min(1)
            values[name].extend(per_row.cpu().tolist())
        residual_target.append(
            (batch["action"] - batch["base_action"])[batch["action_mask"]]
            .float()
            .cpu()
        )
        residual_output.append(
            output.candidate.belief_residual[batch["action_mask"]].float().cpu()
        )
        losses = compute_team_belief_losses(
            output.candidate,
            batch["action"],
            batch["action_mask"],
            batch["teammate_delta"],
            batch["teacher_future_anchor_mask"],
            batch["teammate_action"],
            batch["teammate_action_mask"],
            WEIGHTS,
        )
        for key, value in losses.items():
            auxiliary.setdefault(key, []).append(float(value.detach().cpu()))
    result = {
        "macro": {
            key: float(np.mean(rows)) if rows else float("nan")
            for key, rows in values.items()
        },
        "auxiliary": {
            key: float(np.mean(rows)) if rows else float("nan")
            for key, rows in auxiliary.items()
        },
        "residual_target_rms": float(
            torch.cat(residual_target).square().mean().sqrt()
        )
        if residual_target
        else 0.0,
        "residual_output_rms": float(
            torch.cat(residual_output).square().mean().sqrt()
        )
        if residual_output
        else 0.0,
        "rows": len(values["b0h"]),
        "source": "fixed_offline_diagnostic_requests",
        "closed_loop_results_used": False,
    }
    if not all(_finite(value) for value in result["macro"].values()):
        raise FloatingPointError("non-finite B-core offline diagnostic")
    return result


def _normalization_stats(payload: Mapping[str, Any]) -> dict[str, Any]:
    stats = payload.get("stats")
    if not isinstance(stats, Mapping):
        raise ValueError("B0-H checkpoint has no normalization stats")
    expected_metadata = {
        "action_encoding": ACTION_ENCODING,
        "gripper_encoding": GRIPPER_ENCODING,
        "gripper_native_range": list(GRIPPER_NATIVE_RANGE),
    }
    for name, expected in expected_metadata.items():
        observed = stats.get(name)
        if name == "gripper_native_range":
            try:
                observed = list(map(float, observed))
            except (TypeError, ValueError):
                observed = None
        if observed != expected:
            raise ValueError(
                f"B0-H normalization metadata differs at {name}: "
                f"{observed!r} != {expected!r}"
            )
    result: dict[str, Any] = dict(expected_metadata)
    for name, dimension, positive in (
        ("q_mean", STATE_DIM, False),
        ("q_std", STATE_DIM, True),
        ("a_mean", ACTION_DIM, False),
        ("a_std", ACTION_DIM, True),
        ("q_min", STATE_DIM, False),
        ("q_max", STATE_DIM, False),
        ("a_min", ACTION_DIM, False),
        ("a_max", ACTION_DIM, False),
    ):
        value = np.asarray(stats.get(name), dtype=np.float32)
        if value.shape != (dimension,) or not np.isfinite(value).all():
            raise ValueError(f"invalid B0-H normalization vector {name}")
        if positive and np.any(value <= 0):
            raise ValueError(f"non-positive B0-H normalization scale {name}")
        result[name] = value.tolist()
    for prefix in ("q", "a"):
        minimum = np.asarray(result[f"{prefix}_min"], dtype=np.float32)
        maximum = np.asarray(result[f"{prefix}_max"], dtype=np.float32)
        if np.any(minimum > maximum):
            raise ValueError(f"B0-H {prefix} source ranges are inverted")
        low, high = GRIPPER_NATIVE_RANGE
        if float(minimum[-1]) != low or float(maximum[-1]) != high:
            raise ValueError(
                f"B0-H {prefix} gripper range is not the audited [0,1]"
            )
    return result


def _validate_normalization_receipt_contract(receipt: Mapping[str, Any]) -> None:
    expected = {
        "action_encoding": ACTION_ENCODING,
        "state_encoding": ACTION_ENCODING,
        "gripper_encoding": GRIPPER_ENCODING,
        "gripper_native_range": list(GRIPPER_NATIVE_RANGE),
        "gripper_thresholding": False,
        "gripper_reparameterization": False,
    }
    for name, value in expected.items():
        observed = receipt.get(name)
        if name == "gripper_native_range":
            try:
                observed = list(map(float, observed))
            except (TypeError, ValueError):
                observed = None
        if observed != value:
            raise ValueError(
                f"BiCoord normalization receipt differs at {name}: "
                f"{observed!r} != {value!r}"
            )
    for name in ("qpos_min", "qpos_max", "action_min", "action_max"):
        values = np.asarray(receipt.get(name), dtype=np.float32)
        if values.shape != (STATE_DIM,) or not np.isfinite(values).all():
            raise ValueError(f"invalid BiCoord normalization source range {name}")
    low, high = GRIPPER_NATIVE_RANGE
    if not (
        float(receipt["qpos_min"][-1]) == low
        and float(receipt["action_min"][-1]) == low
        and float(receipt["qpos_max"][-1]) == high
        and float(receipt["action_max"][-1]) == high
    ):
        raise ValueError("BiCoord normalization gripper range is not [0,1]")


def _deployment_required_prefixes(state: Mapping[str, Any]) -> None:
    prefixes = (
        "vision.",
        "history_encoder.",
        "history_action.",
        "decoder.",
        "hidden_residual.",
        "belief_core.",
        "direct_belief_residual.",
    )
    keys = tuple(str(key) for key in state)
    missing = [prefix for prefix in prefixes if not any(key.startswith(prefix) for key in keys)]
    if missing:
        raise ValueError(f"B-core deployment state is missing original modules: {missing}")
    if any(key.startswith("belief_core.teacher_branch.") for key in keys):
        raise ValueError("privileged teacher tensors remain in B-core deployment state")


def validate_deployment_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Fail closed on relabeled, teacher-bearing, or resized deployments."""

    if (payload.get("format") or payload.get("format_version")) != BCORE_DEPLOYMENT_FORMAT:
        raise ValueError("checkpoint is not a BiCoord B-core deployment")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("BiCoord B-core deployment has no config mapping")
    expected = {
        "policy_family": "PredictiveTeamBeliefPolicy",
        "reference_policy_family": "PredictiveTeamBeliefPolicy",
        "method_family": "CARE",
        "architecture": "PredictiveTeamBeliefPolicy_direct_belief_residual",
        "benchmark_adapter": "BiCoord",
        "vision_backbone": VISION_BACKBONE,
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "action_encoding": ACTION_ENCODING,
        "gripper_encoding": GRIPPER_ENCODING,
        "gripper_native_range": list(GRIPPER_NATIVE_RANGE),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "horizon": ACTION_HORIZON,
        "history_steps": HISTORY_STEPS,
        "d_model": D_MODEL,
        "enc_layers": ENCODER_LAYERS,
        "dec_layers": DECODER_LAYERS,
        "roles": ROLES,
        "role_rank": ROLE_RANK,
        "history_layers": HISTORY_LAYERS,
        "source_frequency_hz": BICOORD_SOURCE_FREQUENCY_HZ,
        "strictly_decentralized": True,
        "strict_local": True,
        "shared_weights": True,
        "shared_checkpoint_for_both_arms": True,
        "arm_id_input": False,
        "peer_runtime_input": False,
        "teacher_present": False,
        "strict_dino_contract": True,
        "state_clipping": False,
        "action_clipping": False,
        "gripper_thresholding": False,
        "gripper_reparameterization": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"BiCoord B-core deployment differs at {key}: "
                f"{config.get(key)!r} != {value!r}"
            )
    for key in (
        "policy_family",
        "reference_policy_family",
        "method_family",
        "benchmark_adapter",
        "teacher_present",
        "strictly_decentralized",
    ):
        if payload.get(key) != expected[key]:
            raise ValueError(f"BiCoord B-core top-level contract differs at {key}")
    if tuple(config.get("future_offsets_steps", ())) != BICOORD_FUTURE_OFFSETS_STEPS:
        raise ValueError("BiCoord B-core future step anchors differ")
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("BiCoord B-core deployment has no model state")
    _deployment_required_prefixes(state)
    _normalization_stats(payload)
    return config


def export_deployment(
    training_checkpoint: Path,
    b0h_checkpoint: Path,
    output: Path,
    *,
    normalization: Path,
    bcore_cache: Path,
    visual_cache: Path,
    dino_model: str | Path | None = None,
) -> dict[str, Any]:
    """Attach trained B-core tensors to the unchanged upstream policy."""

    training = torch.load(training_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(training, Mapping) or (
        training.get("format") or training.get("format_version")
    ) != BCORE_TRAINING_FORMAT:
        raise ValueError("B-core export source is not a training checkpoint")
    b0h = torch.load(b0h_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(b0h, Mapping):
        raise ValueError("B0-H checkpoint is not a mapping")
    b0h_config = validate_b0h_payload(b0h)
    model_name = str(dino_model or b0h_config.get("dino_model") or "")
    if not model_name:
        raise ValueError("B-core deployment requires the pinned DINO model path")
    policy = PredictiveTeamBeliefPolicy(
        BICOORD_BELIEF_CONFIG,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        horizon=ACTION_HORIZON,
        d_model=D_MODEL,
        enc_layers=ENCODER_LAYERS,
        dec_layers=DECODER_LAYERS,
        roles=ROLES,
        role_rank=ROLE_RANK,
        history_layers=HISTORY_LAYERS,
        dino_model=model_name,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        strict_dino_contract=True,
        include_teacher=False,
        residual_safety={"enabled": False},
    )
    incompatible = policy.load_state_dict(b0h["model"], strict=False)
    expected_missing = {
        key
        for key in policy.state_dict()
        if key.startswith(("belief_core.", "direct_belief_residual."))
    }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "B0-H cannot be attached to the original PredictiveTeamBeliefPolicy: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    training_state = training.get("model")
    if not isinstance(training_state, Mapping):
        raise ValueError("B-core training checkpoint has no model state")
    core_state = {
        str(key).removeprefix("belief_core."): value
        for key, value in training_state.items()
        if str(key).startswith("belief_core.")
        and not str(key).startswith("belief_core.teacher_branch.")
    }
    residual_state = {
        str(key).removeprefix("belief_residual."): value
        for key, value in training_state.items()
        if str(key).startswith("belief_residual.")
    }
    if not core_state or not residual_state:
        raise ValueError("B-core training checkpoint lacks runtime core/residual tensors")
    policy.belief_core.load_state_dict(core_state, strict=True)
    policy.direct_belief_residual.load_state_dict(residual_state, strict=True)
    policy.eval()
    state = dict(policy.deployment_state_dict())
    _deployment_required_prefixes(state)
    if policy.belief_core.teacher_branch is not None:
        raise RuntimeError("teacher branch exists on deployment policy object")

    bcore_receipt = bcore_cache / "cache_receipt.json"
    visual_receipt = visual_cache / "cache_receipt.json"
    stats = _normalization_stats(b0h)
    normalization_value = load_normalization_receipt(
        normalization, require_formal=bool(training.get("provenance", {}).get("all_1800_demonstrations", False))
    )
    _validate_normalization_receipt_contract(normalization_value)
    receipt_to_stats = {
        "q_mean": normalization_value["qpos_mean"],
        "q_std": normalization_value["qpos_std"],
        "a_mean": normalization_value["action_mean"],
        "a_std": normalization_value["action_std"],
        "q_min": normalization_value["qpos_min"],
        "q_max": normalization_value["qpos_max"],
        "a_min": normalization_value["action_min"],
        "a_max": normalization_value["action_max"],
    }
    for name, expected_values in receipt_to_stats.items():
        if not np.array_equal(
            np.asarray(stats[name], dtype=np.float32),
            np.asarray(expected_values, dtype=np.float32),
        ):
            raise ValueError(
                f"B0-H checkpoint and normalization receipt differ at {name}"
            )
    training_provenance = training.get("provenance")
    if not isinstance(training_provenance, Mapping):
        raise ValueError("B-core training checkpoint has no provenance")
    all_demonstrations = bool(
        training_provenance.get("all_1800_demonstrations", False)
    )
    config = {
        **dict(b0h_config),
        "policy_family": "PredictiveTeamBeliefPolicy",
        "reference_policy_family": "PredictiveTeamBeliefPolicy",
        "method_family": "CARE",
        "architecture": "PredictiveTeamBeliefPolicy_direct_belief_residual",
        "benchmark_adapter": "BiCoord",
        "vision": VISION_BACKBONE,
        "vision_backbone": VISION_BACKBONE,
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "action_encoding": ACTION_ENCODING,
        "gripper_encoding": GRIPPER_ENCODING,
        "gripper_native_range": list(GRIPPER_NATIVE_RANGE),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "horizon": ACTION_HORIZON,
        "history_steps": HISTORY_STEPS,
        "d_model": D_MODEL,
        "enc_layers": ENCODER_LAYERS,
        "dec_layers": DECODER_LAYERS,
        "roles": ROLES,
        "role_rank": ROLE_RANK,
        "history_layers": HISTORY_LAYERS,
        "strictly_decentralized": True,
        "strict_local": True,
        "shared_weights": True,
        "shared_checkpoint_for_both_arms": True,
        "global_view": "shared_head_camera",
        "local_view": "own_wrist_camera_only",
        "arm_id_input": False,
        "peer_runtime_input": False,
        "peer_qpos_action_wrist_allowed": False,
        "act_provider_allowed": False,
        "teacher_present": False,
        "strict_dino_contract": True,
        "source_frequency_hz": BICOORD_SOURCE_FREQUENCY_HZ,
        "future_offsets_steps": list(BICOORD_FUTURE_OFFSETS_STEPS),
        "future_offsets_seconds": list(BICOORD_BELIEF_CONFIG.future_offsets_seconds),
        "recording_alignment": {
            "observation_row_offset": 0,
            "action_row_offset": 1,
            "action_lag_rows": 1,
        },
        "state_clipping": False,
        "action_clipping": False,
        "gripper_thresholding": False,
        "gripper_reparameterization": False,
        "normalization_population": (
            "all_1800_demos_both_local_arms"
            if all_demonstrations
            else "smoke_subset_both_local_arms"
        ),
        "all_1800_demonstrations": all_demonstrations,
        "n2_config": _json_belief_config(),
        "memory_semantics": BICOORD_CARE_MEMORY_SEMANTICS,
        "care_memory_tokens": BICOORD_CARE_MEMORY_TOKENS,
        "care_memory_width": BICOORD_CARE_MEMORY_WIDTH,
        "residual_safety": {"enabled": False},
        "dino_model": model_name,
        "source_b0h_checkpoint_sha256": sha256_file(b0h_checkpoint),
        "source_bcore_training_checkpoint_sha256": sha256_file(training_checkpoint),
        "normalization_receipt_sha256": sha256_file(normalization),
        "bcore_cache_receipt_sha256": sha256_file(bcore_receipt),
        "visual_cache_receipt_sha256": sha256_file(visual_receipt),
    }
    payload: dict[str, Any] = {
        "format": BCORE_DEPLOYMENT_FORMAT,
        "format_version": BCORE_DEPLOYMENT_FORMAT,
        "model": state,
        "stats": stats,
        "update": int(training.get("update", -1)),
        "seed": int(training.get("provenance", {}).get("seed", -1)),
        "policy_family": "PredictiveTeamBeliefPolicy",
        "reference_policy_family": "PredictiveTeamBeliefPolicy",
        "method_family": "CARE",
        "architecture": "PredictiveTeamBeliefPolicy_direct_belief_residual",
        "benchmark_adapter": "BiCoord",
        "teacher_present": False,
        "strictly_decentralized": True,
        "action_encoding": ACTION_ENCODING,
        "gripper_encoding": GRIPPER_ENCODING,
        "gripper_native_range": list(GRIPPER_NATIVE_RANGE),
        "gripper_thresholding": False,
        "gripper_reparameterization": False,
        "strict_local": True,
        "shared_checkpoint_for_both_arms": True,
        "config": config,
        "source_b0h_checkpoint_sha256": sha256_file(b0h_checkpoint),
        "training_checkpoint_sha256": sha256_file(training_checkpoint),
        "normalization_receipt_sha256": sha256_file(normalization),
        "bcore_cache_receipt_sha256": sha256_file(bcore_receipt),
        "visual_cache_receipt_sha256": sha256_file(visual_receipt),
        "created_at_utc": _now(),
    }
    validate_deployment_payload(payload)
    _atomic_save(output, payload)
    reloaded = torch.load(output, map_location="cpu", weights_only=False)
    validate_deployment_payload(reloaded)
    return payload


def _artifact_paths(
    run: Path,
    result: Mapping[str, Any],
    *,
    kinds: Sequence[str] | None = None,
) -> list[Path]:
    """Resolve only hash-verified files from one completed stage result."""

    allowed = None if kinds is None else set(kinds)
    rows: list[Path] = []
    for item in result.get("artifacts", []):
        if not isinstance(item, Mapping):
            continue
        if allowed is not None and item.get("kind") not in allowed:
            continue
        path = Path(str(item.get("path", "")))
        if not path.is_absolute():
            path = run / path
        try:
            path = path.expanduser().resolve(strict=True)
        except FileNotFoundError:
            continue
        if path.is_file() and sha256_file(path) == item.get("sha256"):
            rows.append(path)
    return list(dict.fromkeys(rows))


def _require_published_path(
    run: Path,
    stage: str,
    expected: Path,
    *,
    config_sha256: str,
    kinds: Sequence[str] | None = None,
) -> Path:
    """Require an exact layout path published by the matching DAG stage."""

    result = require_stage_result(run, stage, config_sha256=config_sha256)
    expected = expected.expanduser().resolve(strict=True)
    verified = _artifact_paths(run, result, kinds=kinds)
    if expected not in verified:
        raise ValueError(
            f"{stage} did not publish the required artifact {expected}; "
            f"verified={verified}"
        )
    return expected


def _require_b0h_checkpoint(
    run: Path,
    stage: str,
    *,
    config_sha256: str,
) -> Path:
    result = require_stage_result(run, stage, config_sha256=config_sha256)
    verified = _artifact_paths(
        run,
        result,
        kinds=("checkpoint", "b0h_checkpoint", "training_checkpoint"),
    )
    declared = result.get("checkpoint") or result.get("final_checkpoint")
    if declared:
        candidate = Path(str(declared))
        if not candidate.is_absolute():
            candidate = run / candidate
        try:
            candidate = candidate.expanduser().resolve(strict=True)
        except FileNotFoundError:
            candidate = Path()
        selected = [candidate] if candidate in verified else []
    else:
        selected = verified
    root = (run / "artifacts" / stage).resolve()
    selected = [path for path in selected if root in path.parents]
    if len(selected) != 1:
        raise ValueError(
            f"{stage} must publish one canonical B0-H checkpoint below {root}; "
            f"found={selected}"
        )
    return selected[0]


def _require_exact_argument(
    value: Path | None,
    expected: Path,
    *,
    label: str,
) -> Path:
    expected = expected.expanduser().resolve()
    if value is not None and Path(value).expanduser().resolve() != expected:
        raise ValueError(
            f"supervisor {label} path differs from the frozen layout: "
            f"{Path(value).expanduser().resolve()} != {expected}"
        )
    return expected


def _resolve_paths(args: argparse.Namespace) -> None:
    if args.run is not None:
        run = args.run.expanduser().resolve()
        config_sha256 = str(args.config_sha256)
        if len(config_sha256) != 64:
            raise ValueError("supervisor config SHA-256 must contain 64 characters")
        try:
            int(config_sha256, 16)
        except ValueError as error:
            raise ValueError("supervisor config SHA-256 is not hexadecimal") from error

        normalization = _require_exact_argument(
            args.normalization,
            run / "artifacts" / "dataset_audit" / "normalization.json",
            label="normalization",
        )
        args.normalization = _require_published_path(
            run,
            "dataset_audit",
            normalization,
            config_sha256=config_sha256,
            kinds=("normalization",),
        )

        visual_root = _require_exact_argument(
            args.visual_cache,
            run / "artifacts" / "dino_cache",
            label="DINO cache",
        )
        _require_published_path(
            run,
            "dino_cache",
            visual_root / "cache_receipt.json",
            config_sha256=config_sha256,
        )
        args.visual_cache = visual_root

        smoke = args.stage == "smoke"
        cache_stage = "bcore_smoke_cache" if smoke else "bcore_cache"
        cache_root = _require_exact_argument(
            args.bcore_cache,
            run / "artifacts" / cache_stage,
            label="B-core cache",
        )
        _require_published_path(
            run,
            cache_stage,
            cache_root / "cache_receipt.json",
            config_sha256=config_sha256,
        )
        args.bcore_cache = cache_root

        b0h_stage = "b0h_smoke_train" if smoke else "b0h_formal"
        checkpoint = _require_b0h_checkpoint(
            run, b0h_stage, config_sha256=config_sha256
        )
        if (
            args.b0h_checkpoint is not None
            and Path(args.b0h_checkpoint).expanduser().resolve() != checkpoint
        ):
            raise ValueError(
                f"supervisor B0-H checkpoint differs from {b0h_stage}: "
                f"{args.b0h_checkpoint} != {checkpoint}"
            )
        args.b0h_checkpoint = checkpoint

        expected_output = (
            run / "artifacts" / "bcore_smoke_train"
            if smoke
            else run / "artifacts" / "bcore_train_3seeds" / f"seed_{args.seed}"
        )
        args.output = _require_exact_argument(
            args.output, expected_output, label="B-core training output"
        )
    if args.data_root is None and args.dataset is not None:
        args.data_root = args.dataset
    required = {
        "data_root": args.data_root,
        "normalization": args.normalization,
        "visual_cache": args.visual_cache,
        "bcore_cache": args.bcore_cache,
        "b0h_checkpoint": args.b0h_checkpoint,
        "dino_model": args.dino_model,
        "output": args.output,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"B-core training paths are missing: {missing}")
    for name in ("data_root", "visual_cache", "bcore_cache", "output"):
        setattr(args, name, Path(getattr(args, name)).expanduser().resolve())
    for name in ("normalization", "b0h_checkpoint"):
        setattr(args, name, Path(getattr(args, name)).expanduser().resolve())
    args.dino_model = Path(args.dino_model).expanduser().resolve()


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation", nargs="?", choices=("smoke-train", "formal-train", "train")
    )
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--benchmark-repo", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--dino-model", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--config-sha256", default="")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--data-root", "--prepared-data", dest="data_root", type=Path)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--visual-cache", type=Path)
    parser.add_argument("--bcore-cache", type=Path)
    parser.add_argument("--b0h-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=BCORE_SEEDS[0])
    parser.add_argument("--updates", type=int)
    parser.add_argument(
        "--global-batch", "--batch-size", dest="global_batch", type=int, default=EFFECTIVE_BATCH
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    args.stage = (
        "smoke" if args.smoke or args.operation == "smoke-train" else "formal"
    )
    args.updates = int(
        args.updates
        if args.updates is not None
        else (5 if args.stage == "smoke" else BCORE_UPDATES)
    )
    _resolve_paths(args)
    return args


def _sidecar(path: Path, directory: str, episode_id: int) -> Path | None:
    for name in (f"episode{episode_id}.json", f"episode_{episode_id}.json"):
        candidate = path.parent.parent / directory / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def _smoke_source_rows(receipt: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Validate the exact one-episode-per-task cache selection receipt."""

    expected_head = {
        "schema": BCORE_CACHE_SCHEMA,
        "status": "PASSED",
        "formal": False,
        "cache_complete": True,
        "episodes": len(TASKS),
        "smoke_episode_selection": "minimum_episode_id_then_path_per_task",
    }
    for key, expected in expected_head.items():
        if receipt.get(key) != expected:
            raise ValueError(
                f"B-core smoke cache receipt differs at {key}: "
                f"{receipt.get(key)!r} != {expected!r}"
            )
    counts = receipt.get("episodes_per_task")
    expected_counts = {task: 1 for task in TASKS}
    if not isinstance(counts, Mapping) or {
        task: int(counts.get(task, -1)) for task in TASKS
    } != expected_counts:
        raise ValueError("B-core smoke cache must contain exactly one episode per task")
    sources = receipt.get("episode_sources")
    if not isinstance(sources, list) or len(sources) != len(TASKS):
        raise ValueError("B-core smoke cache lacks its 18 exact episode sources")
    if any(not isinstance(row, Mapping) for row in sources):
        raise ValueError("B-core smoke episode source is not a mapping")
    tasks = [str(row.get("task", "")) for row in sources]
    identities = [str(row.get("source_identity", "")) for row in sources]
    receipt_identities = receipt.get("episode_source_identities")
    if tasks != list(TASKS):
        raise ValueError("B-core smoke episode sources are not in frozen task order")
    if (
        not isinstance(receipt_identities, list)
        or list(map(str, receipt_identities)) != identities
        or len(set(identities)) != len(TASKS)
        or any(len(value) != 64 for value in identities)
    ):
        raise ValueError("B-core smoke source identities are missing or duplicated")
    return list(sources)


def _smoke_episodes(data_root: Path, cache_root: Path) -> list[BiCoordEpisode]:
    """Reconstruct only the 18 episodes actually materialized by smoke cache."""

    data_root = data_root.expanduser().resolve(strict=True)
    receipt = read_json(cache_root / "cache_receipt.json")
    sources = _smoke_source_rows(receipt)
    episodes: list[BiCoordEpisode] = []
    for expected_task, row in zip(TASKS, sources):
        path = Path(str(row.get("path", "")))
        if not path.is_absolute():
            path = data_root / path
        path = path.expanduser().resolve(strict=True)
        if data_root not in path.parents:
            raise ValueError(f"B-core smoke source escapes the dataset root: {path}")
        episode_id = int(row.get("episode_id", -1))
        if episode_id < 0 or episode_number(path) != episode_id:
            raise ValueError(f"B-core smoke source episode ID differs: {path}")
        source_identity = str(row.get("source_identity", ""))
        observed_identity = hdf5_sha256_file(path)
        if observed_identity != source_identity:
            raise ValueError(f"B-core smoke source hash changed: {path}")
        metadata = validate_hdf5_schema(path, check_images=False)
        marker = cache_root / expected_task / f"{source_identity}.complete.json"
        decoded = cache_root / expected_task / f"{source_identity}.decoded.npy"
        base = cache_root / expected_task / f"{source_identity}.base_action.npy"
        if not marker.is_file() or not decoded.is_file() or not base.is_file():
            raise ValueError(f"B-core smoke cache files are incomplete: {source_identity}")
        marker_value = read_json(marker)
        marker_expected = {
            "status": "PASSED",
            "task": expected_task,
            "episode_id": episode_id,
            "source_identity": source_identity,
            "decisions": int(metadata["length"]) - 1,
        }
        for key, expected in marker_expected.items():
            if marker_value.get(key) != expected:
                raise ValueError(
                    f"B-core smoke marker differs at {key}: "
                    f"{marker_value.get(key)!r} != {expected!r}"
                )
        stage = _sidecar(path, "stages", episode_id)
        instruction = _sidecar(path, "instructions", episode_id)
        episodes.append(
            BiCoordEpisode(
                path=str(path),
                task=expected_task,
                task_text=TASK_TEXT[expected_task],
                episode_id=episode_id,
                length=int(metadata["length"]),
                hdf5_sha256=source_identity,
                stage_path=str(stage) if stage else None,
                instruction_path=str(instruction) if instruction else None,
            )
        )
    if len(episodes) != len(TASKS) or [row.task for row in episodes] != list(TASKS):
        raise AssertionError("B-core smoke episode reconstruction lost task coverage")
    return episodes


def _validate_inputs(args: argparse.Namespace) -> tuple[list[Any], dict[str, Any]]:
    if args.stage == "formal":
        if args.updates != BCORE_UPDATES:
            raise ValueError("formal BiCoord B-core is fixed at 120000 updates")
        if args.seed not in BCORE_SEEDS:
            raise ValueError(f"formal B-core seed must be one of {BCORE_SEEDS}")
        if args.eval_every != EVAL_EVERY:
            raise ValueError("formal B-core offline diagnostics are fixed every 5000 updates")
    elif not 1 <= args.updates <= 10:
        raise ValueError("BiCoord B-core smoke is capped at ten updates")
    if args.global_batch != EFFECTIVE_BATCH:
        raise ValueError("BiCoord B-core effective batch is frozen at 48")
    if args.workers < 0 or min(args.save_every, args.eval_every, args.log_every) < 1:
        raise ValueError("worker count/intervals are invalid")
    for path in (
        args.data_root,
        args.visual_cache,
        args.bcore_cache,
        args.dino_model,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    for path in (args.normalization, args.b0h_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    episodes = (
        discover_bicoord_episodes(
            args.data_root,
            require_formal=True,
            verify_schema=True,
        )
        if args.stage == "formal"
        else _smoke_episodes(args.data_root, args.bcore_cache)
    )
    if not episodes:
        raise ValueError("BiCoord B-core dataset cannot be empty")
    if args.stage == "formal" and len(episodes) != TOTAL_EPISODES:
        raise ValueError("formal BiCoord B-core requires all 1800 demonstrations")
    if args.stage == "smoke" and (
        len(episodes) != len(TASKS)
        or [episode.task for episode in episodes] != list(TASKS)
    ):
        raise ValueError("B-core smoke requires exactly one cached episode per task")
    normalization = load_normalization_receipt(
        args.normalization, require_formal=args.stage == "formal"
    )
    _validate_normalization_receipt_contract(normalization)
    if normalization.get("dataset_revision") not in (None, DATASET_REVISION):
        raise ValueError("BiCoord normalization belongs to another dataset revision")
    counts = {task: sum(episode.task == task for episode in episodes) for task in TASKS}
    if args.stage == "formal" and any(counts[task] != EPISODES_PER_TASK for task in TASKS):
        raise ValueError("BiCoord B-core requires 100 demonstrations per task")
    if args.stage == "smoke" and any(counts[task] != 1 for task in TASKS):
        raise ValueError("BiCoord B-core smoke cache coverage differs by task")
    return episodes, normalization


def _provenance(args: argparse.Namespace, *, episodes: Sequence[Any]) -> dict[str, Any]:
    all_demonstrations = len(episodes) == TOTAL_EPISODES
    return {
        "stage": args.stage,
        "seed": int(args.seed),
        "protocol_updates": BCORE_UPDATES,
        "effective_batch": EFFECTIVE_BATCH,
        "data_seed": DATA_SEED,
        "dataset_repo_id": "GradiusTwinbee/BiCoord",
        "dataset_revision": DATASET_REVISION,
        "policy_episode_count": len(episodes),
        "policy_training_split": (
            "all_1800_demonstrations_no_holdout"
            if all_demonstrations
            else "smoke_subset_no_holdout"
        ),
        "all_1800_demonstrations": all_demonstrations,
        "b0h_checkpoint_sha256": sha256_file(args.b0h_checkpoint),
        "normalization_receipt_sha256": sha256_file(args.normalization),
        "visual_cache_receipt_sha256": sha256_file(
            args.visual_cache / "cache_receipt.json"
        ),
        "bcore_cache_receipt_sha256": sha256_file(
            args.bcore_cache / "cache_receipt.json"
        ),
        "belief_config": _json_belief_config(),
        "loss_weights": asdict(WEIGHTS),
        "supervisor_config_sha256": str(args.config_sha256),
    }


def _training_config(args: argparse.Namespace, provenance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format_version": BCORE_TRAINING_FORMAT,
        "policy_family": "PredictiveTeamBeliefPolicy",
        "reference_policy_family": "PredictiveTeamBeliefPolicy",
        "method_family": "CARE",
        "architecture": "PredictiveTeamBeliefPolicy_direct_belief_residual",
        "benchmark_adapter": "BiCoord",
        "vision": VISION_BACKBONE,
        "vision_backbone": VISION_BACKBONE,
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "action_encoding": ACTION_ENCODING,
        "gripper_encoding": GRIPPER_ENCODING,
        "gripper_native_range": list(GRIPPER_NATIVE_RANGE),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "horizon": ACTION_HORIZON,
        "history_steps": HISTORY_STEPS,
        "d_model": D_MODEL,
        "enc_layers": ENCODER_LAYERS,
        "dec_layers": DECODER_LAYERS,
        "roles": ROLES,
        "role_rank": ROLE_RANK,
        "history_layers": HISTORY_LAYERS,
        "strictly_decentralized": True,
        "strict_local": True,
        "shared_weights": True,
        "shared_checkpoint_for_both_arms": True,
        "arm_id_input": False,
        "peer_runtime_input": False,
        "act_provider_allowed": False,
        "strict_dino_contract": True,
        "teacher_training_only": True,
        "source_frequency_hz": BICOORD_SOURCE_FREQUENCY_HZ,
        "future_offsets_steps": list(BICOORD_FUTURE_OFFSETS_STEPS),
        "future_offsets_seconds": list(BICOORD_BELIEF_CONFIG.future_offsets_seconds),
        "recording_alignment": {
            "observation_row_offset": 0,
            "action_row_offset": 1,
            "action_lag_rows": 1,
        },
        "state_clipping": False,
        "action_clipping": False,
        "gripper_thresholding": False,
        "gripper_reparameterization": False,
        "normalization_population": (
            "all_1800_demos_both_local_arms"
            if provenance["all_1800_demonstrations"]
            else "smoke_subset_both_local_arms"
        ),
        "all_1800_demonstrations": bool(provenance["all_1800_demonstrations"]),
        "held_out_demonstrations": 0,
        "sampling": "18_base_pairs_plus_6_rotating_extra_pairs",
        "seed": int(args.seed),
        "protocol_updates": BCORE_UPDATES,
        "update_target": int(args.updates),
        "effective_batch": EFFECTIVE_BATCH,
        "tasks": list(TASKS),
        "n2_config": _json_belief_config(),
        "loss_weights": asdict(WEIGHTS),
        "offline_diagnostic_every": EVAL_EVERY,
        "closed_loop_results_used_for_training_or_selection": False,
        "source_b0h_checkpoint_sha256": provenance["b0h_checkpoint_sha256"],
        "bcore_cache_receipt_sha256": provenance["bcore_cache_receipt_sha256"],
    }


def _publish_training_result(
    args: argparse.Namespace,
    *,
    latest: Path,
    deployment: Path,
    receipt: Path,
    selected_update: int,
    selected_score: float,
) -> None:
    if args.result is None:
        return
    stage = "bcore_smoke_train" if args.stage == "smoke" else "bcore_train_3seeds"
    publish_result(
        args,
        stage=stage,
        include_model_contract=True,
        artifacts=(
            artifact(latest, kind="training_checkpoint"),
            artifact(deployment, kind="deployment_checkpoint"),
            artifact(receipt, kind="checkpoint_receipt"),
            artifact(args.output / "status.json", kind="status"),
        ),
        seed=int(args.seed),
        update=int(args.updates),
        selected_update=int(selected_update),
        selected_offline_score_b_core_mse=float(selected_score),
        checkpoint=str(deployment.resolve()),
        checkpoint_sha256=sha256_file(deployment),
        all_1800_demonstrations=bool(
            read_json(args.output / "config.json").get("all_1800_demonstrations", False)
        ),
        effective_batch=EFFECTIVE_BATCH,
        closed_loop_results_used_for_selection=False,
        teacher_present=False,
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    episodes, _normalization = _validate_inputs(args)
    if args.stage == "formal" and not torch.cuda.is_available():
        raise RuntimeError("formal BiCoord B-core training requires a CUDA GPU lease")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(max(1, min(12, os.cpu_count() or 12)))
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    latest = args.output / "checkpoint_latest.pt"
    resume_path = args.resume
    if resume_path is None and args.auto_resume and latest.is_file():
        resume_path = latest
    saved: Mapping[str, Any] | None = None
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        candidate = torch.load(resume_path, map_location="cpu", weights_only=False)
        if not isinstance(candidate, Mapping) or (
            candidate.get("format") or candidate.get("format_version")
        ) != BCORE_TRAINING_FORMAT:
            raise ValueError("resume checkpoint has wrong BiCoord B-core format")
        saved = candidate
    start = int(saved.get("update", 0)) if saved else 0
    if not 0 <= start <= args.updates:
        raise ValueError("resume update is outside requested B-core budget")

    dataset = BiCoordTeamBeliefDataset(
        episodes,
        args.normalization,
        args.visual_cache,
        args.bcore_cache,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        cache_limit=max(16, args.workers * 4),
    )
    sampler = BiCoordPairedSituationBatchSampler(
        episodes,
        updates=BCORE_UPDATES,
        data_seed=DATA_SEED,
        start_update=start,
    )
    if saved:
        sampler.validate_cursor(saved["sample_cursor"])
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    requests = fixed_diagnostic_requests(episodes)
    diagnostic_batches = [
        requests[first : first + EFFECTIVE_BATCH]
        for first in range(0, len(requests), EFFECTIVE_BATCH)
    ]
    diagnostic = DataLoader(
        dataset,
        batch_sampler=diagnostic_batches,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = TeamBeliefExperiment(BICOORD_BELIEF_CONFIG).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1.0 if step < LR_DROP_UPDATE else 0.1
    )
    provenance = _provenance(args, episodes=episodes)
    evaluations: list[dict[str, Any]] = []
    if saved:
        if saved.get("provenance") != provenance:
            raise ValueError("BiCoord B-core resume provenance drift")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        evaluations = list(saved.get("evaluations", []))
    config = _training_config(args, provenance)
    atomic_json(args.output / "config.json", config)
    atomic_json(
        args.output / "status.json",
        {
            "status": "TRAINING",
            "stage": args.stage,
            "seed": args.seed,
            "update": start,
            "target_updates": args.updates,
            "started_at_utc": _now(),
        },
    )

    started = time.time()
    last_metrics: dict[str, Any] = dict(saved.get("last_metrics", {})) if saved else {}
    for update, raw in enumerate(loader, start=start + 1):
        if update > args.updates:
            break
        step_seed = args.seed + 10_000_019 * update
        random.seed(step_seed)
        np.random.seed(step_seed % (2**32))
        torch.manual_seed(step_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(step_seed)
        batch = _device_batch(raw, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model(batch)
            partner = paired_permutation(batch["pair_id"])
            swapped = replace(
                output.candidate,
                belief=replace(
                    output.candidate.belief,
                    mu=output.candidate.belief.mu[partner],
                ),
            )
            negative = _shuffle_permutation(batch["task_index"], batch["phase_bin"])
            counterfactual_residual, _ = model.belief_residual(
                batch["decoded_action_hidden"],
                output.candidate.belief.mu[negative],
                output.candidate.belief.sigma[negative],
                output.candidate.belief.reliability[negative],
            )
            residual_target = batch["action"] - batch["base_action"]
            losses = compute_team_belief_losses(
                output.candidate,
                batch["action"],
                batch["action_mask"],
                batch["teammate_delta"],
                batch["teacher_future_anchor_mask"],
                batch["teammate_action"],
                batch["teammate_action_mask"],
                WEIGHTS,
                swapped_output=swapped,
                counterfactual_prediction=batch["base_action"]
                + counterfactual_residual,
                counterfactual_residual_target=residual_target[negative],
                counterfactual_action_mask=batch["action_mask"][negative],
            )
            direct = (
                (output.direct_prediction - batch["action"])
                .float()
                .square()
                .mean(-1)
                * batch["action_mask"]
            ).sum() / batch["action_mask"].sum().clamp_min(1)
            loss = losses["total"] + direct
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite BiCoord B-core loss at update {update}")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient):
            raise FloatingPointError(
                f"non-finite BiCoord B-core gradient at update {update}"
            )
        optimizer.step()
        scheduler.step()
        last_metrics = {
            key: float(value.detach().cpu()) for key, value in losses.items()
        }
        last_metrics.update(
            {
                "direct_reactive": float(direct.detach().cpu()),
                "combined": float(loss.detach().cpu()),
                "gradient_norm": float(gradient.detach().cpu()),
                "learning_rate": scheduler.get_last_lr()[0],
                "update": update,
                "target_updates": args.updates,
                "elapsed_seconds": time.time() - started,
            }
        )
        if update == start + 1 or update % args.log_every == 0 or update == args.updates:
            print(json.dumps(last_metrics, sort_keys=True), flush=True)
            with (args.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(last_metrics, sort_keys=True) + "\n")
            atomic_json(args.output / "heartbeat.json", last_metrics)
            atomic_json(
                args.output / "status.json",
                {
                    "status": "TRAINING",
                    "stage": args.stage,
                    "seed": args.seed,
                    **last_metrics,
                },
            )
        should_eval = update % args.eval_every == 0 or update == args.updates
        if should_eval:
            metrics = evaluate_offline(model, diagnostic, device)
            evaluations.append({"update": update, "validation": metrics})
            print(json.dumps({"evaluation": evaluations[-1]}, sort_keys=True), flush=True)
        if update % args.save_every == 0 or update == args.updates:
            checkpoint = {
                "format": BCORE_TRAINING_FORMAT,
                "format_version": BCORE_TRAINING_FORMAT,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update": update,
                "evaluations": evaluations,
                "last_metrics": last_metrics,
                "sample_cursor": sampler.cursor_receipt(update),
                "provenance": provenance,
                "config": config,
            }
            numbered = args.output / f"checkpoint_{update:06d}.pt"
            _atomic_save(numbered, checkpoint)
            _atomic_save(latest, checkpoint)

    if not latest.is_file():
        raise RuntimeError("BiCoord B-core training produced no checkpoint")
    completed = torch.load(latest, map_location="cpu", weights_only=False)
    if int(completed.get("update", -1)) != args.updates:
        raise RuntimeError("BiCoord B-core checkpoint did not reach requested update")
    selected_update, selected_score = _selected_diagnostic(
        completed, formal=args.stage == "formal"
    )
    selected_checkpoint = args.output / f"checkpoint_{selected_update:06d}.pt"
    if not selected_checkpoint.is_file():
        if selected_update != args.updates:
            raise FileNotFoundError(selected_checkpoint)
        selected_checkpoint = latest
    deployment = args.output / "deployment_checkpoint.pt"
    export_deployment(
        selected_checkpoint,
        args.b0h_checkpoint,
        deployment,
        normalization=args.normalization,
        bcore_cache=args.bcore_cache,
        visual_cache=args.visual_cache,
        dino_model=args.dino_model,
    )
    final_copy = args.output / "final.pt"
    _atomic_save(
        final_copy,
        torch.load(deployment, map_location="cpu", weights_only=False),
    )
    status_value = {
        "status": "PASSED_SMOKE" if args.stage == "smoke" else "COMPLETED",
        "stage": args.stage,
        "seed": args.seed,
        "update": args.updates,
        "target_updates": args.updates,
        "selected_update": selected_update,
        "selected_offline_score_b_core_mse": selected_score,
        "deployment_checkpoint": str(deployment.resolve()),
        "deployment_checkpoint_sha256": sha256_file(deployment),
        "closed_loop_results_used_for_selection": False,
        "teacher_present": False,
        "completed_at_utc": _now(),
    }
    atomic_json(args.output / "status.json", status_value)
    receipt_path = args.output / "checkpoint_receipt.json"
    receipt = {
        "schema": "before-we-act.bicoord.dino-bcore-training-checkpoint/1",
        "status": "PASSED_SMOKE" if args.stage == "smoke" else "PASSED",
        "format": BCORE_TRAINING_FORMAT,
        "stage": args.stage,
        "seed": args.seed,
        "update": args.updates,
        "selected_update": selected_update,
        "selected_offline_score_b_core_mse": selected_score,
        "policy_family": "PredictiveTeamBeliefPolicy",
        "reference_policy_family": "PredictiveTeamBeliefPolicy",
        "method_family": "CARE",
        "benchmark_adapter": "BiCoord",
        "vision_backbone": VISION_BACKBONE,
        "action_encoding": ACTION_ENCODING,
        "gripper_encoding": GRIPPER_ENCODING,
        "gripper_native_range": list(GRIPPER_NATIVE_RANGE),
        "gripper_thresholding": False,
        "gripper_reparameterization": False,
        "strictly_decentralized": True,
        "strict_local": True,
        "shared_weights": True,
        "teacher_present": False,
        "all_1800_demonstrations": bool(
            completed.get("provenance", {}).get("all_1800_demonstrations", False)
        ),
        "source_b0h_checkpoint_sha256": sha256_file(args.b0h_checkpoint),
        "training_checkpoint_sha256": sha256_file(selected_checkpoint),
        "deployment_checkpoint_sha256": sha256_file(deployment),
        "closed_loop_results_used_for_selection": False,
        "created_at_utc": _now(),
    }
    atomic_json(receipt_path, receipt)
    _publish_training_result(
        args,
        latest=latest,
        deployment=deployment,
        receipt=receipt_path,
        selected_update=selected_update,
        selected_score=selected_score,
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    train(_arguments(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FINAL_SUFFICIENCY_WINDOW",
    "WEIGHTS",
    "evaluate_offline",
    "export_deployment",
    "train",
    "validate_deployment_payload",
]
