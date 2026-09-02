"""Paired selector-off/CARE Validation20 for the DuoBench B-core.

This is the formal closed-loop companion to
``duo_dino_branch_launcher``.  Every task/seed is evaluated twice with the
same seed: selector-off executes candidate zero, the complete frozen B-core
reference proposal, while CARE scores the *same* B-core proposal and applies the frozen
offline selector.  No ACT model, random projection, or legacy CAREPolicy is
loaded here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from before_we_act.care_belief import (
    CAREBeliefConfig,
    CAREBeliefHead,
    CARECalibration,
    select_care_candidate,
)
from deployment.duo_dino_reference.bcore_runtime import DuoBcoreRuntime, validate_bcore_payload
from deployment.duo_dino_reference.bcore_data import (
    DUO_CARE_MEMORY_SEMANTICS,
    DUO_CARE_MEMORY_TOKENS,
    DUO_CARE_MEMORY_WIDTH,
)
from deployment.duo_dino_reference.data import TASKS, load_manifest
from deployment.duo_dino_reference.evaluate import (
    _progress,
    _task_success,
    make_env,
)
from deployment.duo_dino_reference.preprocessing import (
    DINO_NORMALIZATION_ID,
    IMAGE_PREPROCESS_ID,
)
from deployment.duo_care.candidates import (
    candidate_family,
    decoded_absolute_chunk,
)
from deployment.duo_act.action_target import (
    ACTION_TARGET_CONTRACT_ID,
    ACTION_TARGET_CONTRACT_SHA256,
    CONTROLLER_JOINT_HIGH,
    CONTROLLER_JOINT_LOW,
    canonicalize_controller_action_with_audit,
    summarize_action_canonicalization,
)


SUMMARY_SCHEMA = "before-we-act.care-duobench-paired-validation20/1"
TASK_SCHEMA = "before-we-act.care-duobench-paired-validation20-task/1"
SMOKE_SCHEMA = "before-we-act.care-duobench-paired-validation-smoke/1"
SMOKE_TASK_SCHEMA = "before-we-act.care-duobench-paired-validation-smoke-task/1"
EPISODE_SCHEMA = "before-we-act.care-duobench-paired-episode/1"
REFERENCE_POLICY_FAMILY = "PredictiveTeamBeliefPolicy"
BASE_POLICY_FAMILY = "TemporalHistoryPolicy"
METHOD_FAMILY = "CARE"
MAX_GPUS = 4


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


def _find(value: object, key: str) -> object | None:
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find(child, key)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for child in value:
            found = _find(child, key)
            if found is not None:
                return found
    return None


def validate_selected_inputs(
    *,
    bcore_checkpoint: Path,
    care_checkpoint: Path,
    prepared_data: Path,
    dino_model: Path,
) -> dict[str, Any]:
    """Fail closed on any non-formal reference/scorer checkpoint."""

    prepared_data = prepared_data.resolve()
    if not prepared_data.is_dir():
        raise FileNotFoundError(prepared_data)
    manifest = load_manifest(prepared_data, require_formal=True)
    if int(manifest.get("total_episodes", -1)) != 550:
        raise ValueError("Duo paired validation requires all 550 demonstrations")
    bcore_checkpoint = bcore_checkpoint.resolve(strict=True)
    care_checkpoint = care_checkpoint.resolve(strict=True)
    dino_model = dino_model.resolve(strict=True)
    bcore_payload = torch.load(bcore_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(bcore_payload, Mapping):
        raise ValueError("B-core checkpoint is not a mapping")
    if (bcore_payload.get("format") or bcore_payload.get("format_version")) == "before-we-act.duobench.dino-b0h/1":
        raise ValueError("paired validation cannot use B0-H as the B-core reference")
    bcore_config = dict(validate_bcore_payload(bcore_payload))
    bcore_sha = _sha256(bcore_checkpoint)
    required_bcore = {
        "policy_family": REFERENCE_POLICY_FAMILY,
        "reference_policy_family": REFERENCE_POLICY_FAMILY,
        "method_family": METHOD_FAMILY,
        "benchmark_adapter": "DuoBench",
        "vision_backbone": "dinov3_vitb16_frozen",
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "strict_dino_contract": True,
        "strictly_decentralized": True,
        "strict_local": True,
        "act_provider_allowed": False,
        "teacher_present": False,
        "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
        "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
    }
    for key, expected in required_bcore.items():
        value = bcore_payload.get(key, bcore_config.get(key))
        if value != expected:
            raise ValueError(f"B-core differs at {key}: {value!r} != {expected!r}")
    n2_config = bcore_config.get("n2_config")
    if not isinstance(n2_config, Mapping):
        raise ValueError("B-core has no N2 config for CARE memory validation")
    if (
        int(n2_config.get("n_belief_tokens", -1))
        + int(n2_config.get("event_capacity", -1))
        != DUO_CARE_MEMORY_TOKENS
        or int(n2_config.get("d_model", -1)) != DUO_CARE_MEMORY_WIDTH
    ):
        raise ValueError("B-core CARE memory token/width contract differs")

    care = torch.load(care_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(care, Mapping):
        raise ValueError("CARE deployment checkpoint is not a mapping")
    if (care.get("format_version") or care.get("format")) != "before-we-act.care-duobench-deployment-checkpoint/1":
        raise ValueError("paired validation requires the formal Duo CARE deployment checkpoint")
    care_config = care.get("config")
    if not isinstance(care_config, Mapping):
        raise ValueError("CARE deployment has no config mapping")
    required_care = {
        "policy_family": "CAREBeliefHead",
        "reference_policy_family": REFERENCE_POLICY_FAMILY,
        "method_family": METHOD_FAMILY,
        "architecture": "CAREBeliefHead_distributional_candidate_scorer",
        "benchmark_adapter": "DuoBench",
        "vision_backbone": "dinov3_vitb16_frozen",
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "strict_dino_contract": True,
        "strictly_decentralized": True,
        "strict_local": True,
        "act_provider_allowed": False,
        "teacher_present": False,
        "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
        "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
    }
    for key, expected in required_care.items():
        value = care.get(key, care_config.get(key))
        if value != expected:
            raise ValueError(f"CARE deployment differs at {key}: {value!r} != {expected!r}")
    for key, expected in {
        "care_memory_semantics": DUO_CARE_MEMORY_SEMANTICS,
        "care_memory_tokens": DUO_CARE_MEMORY_TOKENS,
    }.items():
        value = care.get(key, care_config.get(key))
        if value != expected:
            raise ValueError(f"CARE deployment differs at {key}: {value!r} != {expected!r}")
    reference_hash = (
        care.get("reference_checkpoint_sha256")
        or care.get("source_bcore_checkpoint_sha256")
        or _find(care_config, "reference_checkpoint_sha256")
    )
    if reference_hash != bcore_sha:
        raise ValueError("CARE deployment was not calibrated against the selected B-core")
    if not isinstance(care.get("model"), Mapping):
        raise ValueError("CARE deployment has no scorer state")
    if not isinstance(care.get("calibration"), Mapping):
        raise ValueError("CARE deployment has no offline calibration")
    return {
        "prepared_manifest_sha256": _sha256(prepared_data / "manifest.json"),
        "bcore_checkpoint_sha256": bcore_sha,
        "care_checkpoint_sha256": _sha256(care_checkpoint),
        "dino_model": str(dino_model),
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
    }


def _action_spaces(env: Any) -> tuple[np.ndarray, np.ndarray]:
    del env
    return CONTROLLER_JOINT_LOW.copy(), CONTROLLER_JOINT_HIGH.copy()


def _payload_from_absolute(
    env: Any,
    absolute: np.ndarray,
    *,
    with_audit: bool = False,
) -> Any:
    del env
    value = np.asarray(absolute, dtype=np.float32)
    if value.shape != (2, 8) or not np.isfinite(value).all():
        raise ValueError(f"absolute action must be finite [2,8], got {value.shape}")
    value, audit = canonicalize_controller_action_with_audit(value)
    payload = {
        "left": {
            "joints": value[0, :7].astype(np.float32),
            "gripper": np.asarray([value[0, 7]], dtype=np.float32),
        },
        "right": {
            "joints": value[1, :7].astype(np.float32),
            "gripper": np.asarray([value[1, 7]], dtype=np.float32),
        },
    }
    if with_audit:
        return payload, value, audit
    # Preserve the original private-helper return type for diagnostic callers;
    # the formal rollout opts into the auditable triple below.
    return payload


def _valid_recovered_episode(
    row: object,
    *,
    task: str,
    mode: str,
    seed: int,
    max_steps: int,
    bcore_checkpoint_sha256: str,
    care_checkpoint_sha256: str,
) -> bool:
    """Validate a JSONL row before allowing resume to skip execution.

    JSONL is deliberately treated as an untrusted crash-recovery cache.  A
    stale row from another checkpoint, seed protocol, or legacy evaluator must
    never be promoted into a formal Validation20 summary merely because its
    ``task`` and ``seed`` happen to match.
    """

    if not isinstance(row, Mapping):
        return False
    expected_policy = (
        "PredictiveTeamBeliefPolicy" if mode == "selector_off" else "CAREBeliefHead"
    )
    expected = {
        "schema": EPISODE_SCHEMA,
        "task": task,
        "mode": mode,
        "seed": int(seed),
        "max_steps": int(max_steps),
        "strictly_decentralized": True,
        "strict_local": True,
        "per_robot_independent_inputs": True,
        "act_provider_allowed": False,
        "policy_family": expected_policy,
        "reference_policy_family": REFERENCE_POLICY_FAMILY,
        "method_family": METHOD_FAMILY,
        "vision": "dinov3_vitb16_frozen",
        "vision_backbone": "dinov3_vitb16_frozen",
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "strict_dino_contract": True,
        "action_encoding": "absolute_joint7_binary_gripper1",
        "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
        "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
        "candidate_encoding": "joint_residual7_gripper_absolute1",
        "selector_off_semantics": "candidate0_complete_bcore_reference",
        "override_contract": "at_most_one_focal_arm_per_control_step",
        "care_memory_semantics": DUO_CARE_MEMORY_SEMANTICS,
        "care_memory_tokens": DUO_CARE_MEMORY_TOKENS,
        "bcore_checkpoint_sha256": bcore_checkpoint_sha256,
        "care_checkpoint_sha256": care_checkpoint_sha256,
    }
    if any(row.get(key) != value for key, value in expected.items()):
        return False
    if not isinstance(row.get("success"), bool):
        return False
    try:
        steps = int(row.get("steps", -1))
    except (TypeError, ValueError):
        return False
    if not 1 <= steps <= int(max_steps):
        return False
    trace = row.get("action_trace_sha256")
    if not isinstance(trace, str) or len(trace) != 64:
        return False
    try:
        int(trace, 16)
    except ValueError:
        return False
    selected_rows = row.get("selected_candidates")
    if not isinstance(selected_rows, list) or len(selected_rows) != steps:
        return False
    for index, decision in enumerate(selected_rows):
        if not isinstance(decision, Mapping) or decision.get("step") != index:
            return False
        selected = decision.get("selected_candidates")
        proposed = decision.get("proposed_candidates")
        if not isinstance(selected, list) or not isinstance(proposed, list):
            return False
        if len(selected) != 2 or len(proposed) != 2:
            return False
        if any(not isinstance(value, int) or not 0 <= value < 6 for value in selected + proposed):
            return False
        # Selector-off has no proposed intervention.  CARE is allowed to score
        # both arms, but only one focal override may reach the environment.
        if mode == "selector_off" and any(proposed):
            return False
        if sum(value != 0 for value in selected) > 1:
            return False
        focal = decision.get("focal_arm")
        if focal is not None and focal not in (0, 1):
            return False
        selected_arm = next((arm for arm, value in enumerate(selected) if value), None)
        if selected_arm != focal:
            return False
        if bool(decision.get("one_focal_override_applied")) != (focal is not None):
            return False
    if mode == "selector_off":
        try:
            if float(row.get("override_rate", -1.0)) != 0.0:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _load_models(
    bcore_checkpoint: Path,
    care_checkpoint: Path,
    *,
    dino_model: Path,
    device: torch.device,
) -> tuple[DuoBcoreRuntime, CAREBeliefHead, CARECalibration, Mapping[str, Any]]:
    runtime = DuoBcoreRuntime.from_checkpoint(
        bcore_checkpoint, device=device, dino_model=str(dino_model)
    )
    saved = torch.load(care_checkpoint, map_location="cpu", weights_only=False)
    scorer = CAREBeliefHead(CAREBeliefConfig.from_mapping(saved["config"])).to(device)
    scorer.load_state_dict(saved["model"], strict=True)
    scorer.eval()
    calibration = CARECalibration.from_mapping(saved["calibration"])
    return runtime, scorer, calibration, saved


@torch.inference_mode()
def run_episode(
    *,
    task: str,
    seed: int,
    mode: str,
    max_steps: int,
    runtime: DuoBcoreRuntime,
    scorer: CAREBeliefHead,
    calibration: CARECalibration,
    env: Any,
    bcore_checkpoint_sha256: str,
    care_checkpoint_sha256: str,
) -> dict[str, Any]:
    if mode not in {"selector_off", "care"}:
        raise ValueError(mode)
    if max_steps <= 0:
        raise ValueError("task max_steps must be positive")
    seed = int(seed)
    np.random.seed(seed % (2**32))
    random_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    observation, reset_info = env.reset(seed=seed)
    runtime.reset(task)
    trace = hashlib.sha256()
    progress: list[float] = []
    selected_rows: list[dict[str, Any]] = []
    raw_commands: list[np.ndarray] = []
    canonical_commands: list[np.ndarray] = []
    success = False
    terminated = truncated = False
    started = time.perf_counter()
    horizon_index = int(scorer.config.horizons.index(calibration.primary_horizon))
    low, high = _action_spaces(env)
    for step in range(max_steps):
        prediction = runtime.predict_chunks(
            observation, task, belief_enabled=True, append_observation=True
        )
        # Candidate zero is the complete B-core proposal after the same
        # absolute temporal ensemble used by B-core Validation20.  Candidate
        # one is the complete embedded B0-H (belief-off) proposal.  CARE scores
        # transformations of these plans; it never generates an action.
        reference_plans = runtime.ensemble.add_and_plan(
            runtime.step_index, prediction.prediction
        )
        base_plans = runtime.base_ensemble.add_and_plan(
            runtime.step_index, prediction.base_prediction
        )
        encoded_candidates = []
        for arm in range(2):
            reference_encoded = reference_plans[arm].copy()
            base_encoded = base_plans[arm].copy()
            reference_encoded[:, :7] -= prediction.qpos[arm, None, :7]
            base_encoded[:, :7] -= prediction.qpos[arm, None, :7]
            candidates, candidate_audit = candidate_family(
                reference_encoded,
                base_encoded,
                prediction.qpos[arm],
                joint_low=low,
                joint_high=high,
                current_gripper=float(prediction.qpos[arm, 7]),
            )
            if not all(audit.valid for audit in candidate_audit):
                raise RuntimeError(
                    f"illegal CARE candidate family for {task}/arm={arm}: "
                    f"{candidate_audit}"
                )
            encoded_candidates.append(candidates)
        if mode == "selector_off":
            # Selector-off is candidate zero: the frozen B-core reference.
            # Belief-off/B0-H remains candidate one and is not the paired
            # control condition in the registered RoboFactory CARE protocol.
            proposed = np.zeros(2, dtype=np.int64)
            lower = np.zeros(2, dtype=np.float32)
            unsafe = np.zeros((2, 6), dtype=bool)
        else:
            candidate_tensor = torch.as_tensor(
                np.stack(encoded_candidates), dtype=torch.float32, device=runtime.device
            )
            memory = torch.as_tensor(
                prediction.memory, dtype=torch.float32, device=runtime.device
            )
            memory_mask = torch.as_tensor(
                prediction.memory_mask, dtype=torch.bool, device=runtime.device
            )
            if tuple(memory.shape) != (
                2,
                DUO_CARE_MEMORY_TOKENS,
                DUO_CARE_MEMORY_WIDTH,
            ) or tuple(memory_mask.shape) != (2, DUO_CARE_MEMORY_TOKENS):
                raise RuntimeError(
                    "paired CARE memory contract differs: "
                    f"{tuple(memory.shape)}/{tuple(memory_mask.shape)}"
                )
            horizon = torch.full(
                (2,), horizon_index, dtype=torch.long, device=runtime.device
            )
            output = scorer(memory, memory_mask, candidate_tensor, horizon)
            selected_tensor, lower_tensor, unsafe_tensor = select_care_candidate(
                output, calibration, variant=scorer.config.variant
            )
            proposed = selected_tensor.detach().cpu().numpy().astype(np.int64)
            lower = lower_tensor.detach().float().cpu().numpy()
            unsafe = unsafe_tensor.detach().cpu().numpy().astype(bool)
        # Preserve the registered RoboFactory action--response factorization:
        # no more than one focal arm may be overridden at a control step.  Each
        # arm is scored from its own local B-core memory; the arbitration only
        # compares the two scalar conservative lower bounds.
        selected = np.zeros(2, dtype=np.int64)
        eligible = [arm for arm in range(2) if int(proposed[arm]) != 0]
        focal_arm = max(eligible, key=lambda arm: float(lower[arm])) if eligible else None
        if focal_arm is not None:
            selected[focal_arm] = int(proposed[focal_arm])
        absolute = np.stack(
            [
                decoded_absolute_chunk(
                    encoded_candidates[arm][int(selected[arm])], prediction.qpos[arm]
                )[0]
                for arm in range(2)
            ]
        )
        payload, command, action_audit = _payload_from_absolute(
            env, absolute, with_audit=True
        )
        raw_commands.append(np.asarray(absolute, dtype=np.float32).copy())
        canonical_commands.append(command.copy())
        trace.update(command.astype(np.float32).tobytes())
        runtime.append_absolute_action(command)
        observation, reward, terminated, truncated, info = env.step(payload)
        progress.append(_progress(info if isinstance(info, Mapping) else {}, reward))
        selected_rows.append(
            {
                "step": int(step),
                "selected_candidates": selected.tolist(),
                "proposed_candidates": proposed.tolist(),
                "focal_arm": None if focal_arm is None else int(focal_arm),
                "one_focal_override_applied": bool(focal_arm is not None),
                "lower_bounds": lower.tolist(),
                "unsafe_candidates": unsafe.sum(1).tolist(),
                "action_canonicalization": {
                    "changed_values": int(action_audit["changed_values"]),
                    "max_abs_delta": float(action_audit["max_abs_delta"]),
                    "out_of_controller_range_by_joint": list(
                        action_audit["out_of_controller_range_by_joint"]
                    ),
                    "reasons": dict(action_audit["reasons"]),
                },
            }
        )
        success = _task_success(info if isinstance(info, dict) else {}, terminated)
        if success or bool(np.asarray(terminated).all()) or bool(np.asarray(truncated).all()):
            break
    # Keep caller RNG state isolated; each paired arm/mode starts from the
    # explicitly seeded stream above.
    torch.random.set_rng_state(random_state)
    action_canonicalization = summarize_action_canonicalization(
        np.stack(raw_commands), np.stack(canonical_commands)
    )
    return {
        "schema": EPISODE_SCHEMA,
        "task": task,
        "seed": seed,
        "mode": mode,
        "success": bool(success),
        "steps": len(selected_rows),
        "max_steps": int(max_steps),
        "final_stage_progress": float(progress[-1]) if progress else 0.0,
        "max_stage_progress": float(max(progress)) if progress else 0.0,
        "selected_candidates": selected_rows,
        "override_rate": float(
            np.mean(
                [
                    int(any(candidate != 0 for candidate in row["selected_candidates"]))
                    for row in selected_rows
                ]
            )
        )
        if mode == "care" and selected_rows
        else 0.0,
        "action_trace_sha256": trace.hexdigest(),
        "wall_seconds": float(time.perf_counter() - started),
        "strictly_decentralized": True,
        "strict_local": True,
        "per_robot_independent_inputs": True,
        "act_provider_allowed": False,
        "policy_family": (
            "PredictiveTeamBeliefPolicy" if mode == "selector_off" else "CAREBeliefHead"
        ),
        "reference_policy_family": REFERENCE_POLICY_FAMILY,
        "method_family": METHOD_FAMILY,
        "vision": "dinov3_vitb16_frozen",
        "vision_backbone": "dinov3_vitb16_frozen",
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "strict_dino_contract": True,
        "action_encoding": "absolute_joint7_binary_gripper1",
        "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
        "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
        "action_canonicalization": action_canonicalization,
        "candidate_encoding": "joint_residual7_gripper_absolute1",
        "selector_off_semantics": "candidate0_complete_bcore_reference",
        "override_contract": "at_most_one_focal_arm_per_control_step",
        "care_memory_semantics": DUO_CARE_MEMORY_SEMANTICS,
        "care_memory_tokens": DUO_CARE_MEMORY_TOKENS,
        "bcore_checkpoint_sha256": str(bcore_checkpoint_sha256),
        "care_checkpoint_sha256": str(care_checkpoint_sha256),
    }


def evaluate_task(
    *,
    bcore_checkpoint: Path,
    care_checkpoint: Path,
    prepared_data: Path,
    dino_model: Path,
    duobench_root: Path | None,
    output: Path,
    task: str,
    episodes: int,
    seed_start: int,
    max_steps: int,
    device: str,
    smoke: bool = False,
) -> dict[str, Any]:
    if task not in TASKS:
        raise ValueError(task)
    if episodes < 1:
        raise ValueError("--episodes must be positive")
    runtime, scorer, calibration, _saved = _load_models(
        bcore_checkpoint,
        care_checkpoint,
        dino_model=dino_model,
        device=torch.device(device),
    )
    bcore_checkpoint_sha256 = _sha256(bcore_checkpoint.resolve(strict=True))
    care_checkpoint_sha256 = _sha256(care_checkpoint.resolve(strict=True))
    env = make_env(task, duobench_root=duobench_root)
    rows: list[dict[str, Any]] = []
    recovered: dict[tuple[str, int], dict[str, Any]] = {}
    jsonl = output.with_suffix(".jsonl")
    if jsonl.is_file():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                key = (str(row["mode"]), int(row["seed"]))
                if key[0] not in {"selector_off", "care"}:
                    continue
                if _valid_recovered_episode(
                    row,
                    task=task,
                    mode=key[0],
                    seed=key[1],
                    max_steps=max_steps,
                    bcore_checkpoint_sha256=bcore_checkpoint_sha256,
                    care_checkpoint_sha256=care_checkpoint_sha256,
                ):
                    recovered[key] = row
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    try:
        for index in range(episodes):
            seed = int(seed_start + index)
            for mode in ("selector_off", "care"):
                row = recovered.get((mode, seed))
                if row is None:
                    row = run_episode(
                        task=task,
                        seed=seed,
                        mode=mode,
                        max_steps=max_steps,
                        runtime=runtime,
                        scorer=scorer,
                        calibration=calibration,
                        env=env,
                        bcore_checkpoint_sha256=bcore_checkpoint_sha256,
                        care_checkpoint_sha256=care_checkpoint_sha256,
                    )
                    if not _valid_recovered_episode(
                        row,
                        task=task,
                        mode=mode,
                        seed=seed,
                        max_steps=max_steps,
                        bcore_checkpoint_sha256=bcore_checkpoint_sha256,
                        care_checkpoint_sha256=care_checkpoint_sha256,
                    ):
                        raise RuntimeError(
                            f"new paired episode violates the frozen contract: "
                            f"{task}/{seed}/{mode}"
                        )
                    with jsonl.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                    print(json.dumps({"task": task, "seed": seed, "mode": mode, "success": row["success"]}), flush=True)
                rows.append(row)
    finally:
        env.close()
    pairs: list[dict[str, Any]] = []
    for index in range(episodes):
        seed = int(seed_start + index)
        off = next(row for row in rows if row["mode"] == "selector_off" and int(row["seed"]) == seed)
        care = next(row for row in rows if row["mode"] == "care" and int(row["seed"]) == seed)
        pairs.append(
            {
                "task": task,
                "seed": seed,
                "selector_off_success": bool(off["success"]),
                "care_success": bool(care["success"]),
                "success_delta": int(bool(care["success"])) - int(bool(off["success"])),
                "progress_delta": float(care["final_stage_progress"] - off["final_stage_progress"]),
                "harmful_override": bool(
                    care["final_stage_progress"] < off["final_stage_progress"]
                    and care["override_rate"] > 0.0
                ),
                "strictly_decentralized": True,
                "strict_local": True,
                "per_robot_independent_inputs": True,
                "act_provider_allowed": False,
                "selector_off_semantics": "candidate0_complete_bcore_reference",
                "override_contract": "at_most_one_focal_arm_per_control_step",
                "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
                "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
                "care_memory_semantics": DUO_CARE_MEMORY_SEMANTICS,
                "care_memory_tokens": DUO_CARE_MEMORY_TOKENS,
                "max_steps": int(max_steps),
                "bcore_checkpoint_sha256": bcore_checkpoint_sha256,
                "care_checkpoint_sha256": care_checkpoint_sha256,
            }
        )
    result = {
        "schema": SMOKE_TASK_SCHEMA if smoke else TASK_SCHEMA,
        "status": "complete",
        "task": task,
        "episodes": int(episodes),
        "max_steps": int(max_steps),
        "seed_start": int(seed_start),
        "seeds": [int(seed_start + index) for index in range(episodes)],
        "benchmark_adapter": "DuoBench",
        "method_family": METHOD_FAMILY,
        "reference_policy_family": REFERENCE_POLICY_FAMILY,
        "base_policy_family": BASE_POLICY_FAMILY,
        "reference_architecture": "PredictiveTeamBeliefPolicy_direct_belief_residual",
        "vision": "dinov3_vitb16_frozen",
        "vision_backbone": "dinov3_vitb16_frozen",
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "strict_dino_contract": True,
        "strictly_decentralized": True,
        "strict_local": True,
        "per_robot_independent_inputs": True,
        "act_provider_allowed": False,
        "selector_off_semantics": "candidate0_complete_bcore_reference",
        "override_contract": "at_most_one_focal_arm_per_control_step",
        "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
        "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
        "care_memory_semantics": DUO_CARE_MEMORY_SEMANTICS,
        "care_memory_tokens": DUO_CARE_MEMORY_TOKENS,
        "bcore_checkpoint_sha256": bcore_checkpoint_sha256,
        "care_checkpoint_sha256": care_checkpoint_sha256,
        "selector_off_success_rate": float(
            np.mean([row["success"] for row in rows if row["mode"] == "selector_off"])
        ),
        "care_success_rate": float(
            np.mean([row["success"] for row in rows if row["mode"] == "care"])
        ),
        "paired_success_improvement": float(np.mean([row["success_delta"] for row in pairs])),
        "selector_off_mean_progress": float(
            np.mean([row["final_stage_progress"] for row in rows if row["mode"] == "selector_off"])
        ),
        "care_mean_progress": float(
            np.mean([row["final_stage_progress"] for row in rows if row["mode"] == "care"])
        ),
        "override_rate": float(np.mean([row["override_rate"] for row in rows if row["mode"] == "care"])),
        "harmful_override_rate": float(np.mean([row["harmful_override"] for row in pairs])),
        "rows": rows,
        "pairs": pairs,
    }
    _atomic_json(output, result)
    return result


def _worker_command(args: argparse.Namespace, task: str, output: Path, max_steps: int) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "deployment.duo_care.duo_dino_paired_launcher",
        "--worker-task",
        task,
        "--bcore-checkpoint",
        str(args.bcore_checkpoint),
        "--care-checkpoint",
        str(args.care_checkpoint),
        "--prepared-data",
        str(args.prepared_data),
        "--dino-model",
        str(args.dino_model),
        "--duobench-root",
        str(args.duobench_root) if args.duobench_root else "",
        "--output",
        str(output),
        "--episodes",
        str(args.episodes),
        "--seed-start",
        str(args.seed_start),
        "--task-max-steps-json",
        str(args.task_max_steps_json),
        "--max-steps",
        str(max_steps),
        "--device",
        "cuda:0",
    ]
    if not args.duobench_root:
        # Remove the optional path pair when no explicit source is supplied.
        command = [value for index, value in enumerate(command) if not (value == "--duobench-root" or (index and command[index - 1] == "--duobench-root"))]
    if args.smoke:
        command.append("--smoke")
    return command


def _run_workers(args: argparse.Namespace, tasks: Sequence[str], max_steps: Mapping[str, int]) -> None:
    workers = min(int(args.workers), MAX_GPUS)
    if workers < 1:
        raise ValueError("--workers must be positive")
    log_root = args.output / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    repo_root = str(Path(__file__).resolve().parents[2])
    for first in range(0, len(tasks), workers):
        active: list[tuple[str, subprocess.Popen[bytes], Any]] = []
        try:
            for slot, task in enumerate(tasks[first : first + workers]):
                output = args.output / "tasks" / f"{task}.json"
                command = _worker_command(args, task, output, int(max_steps[task]))
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(slot)
                env["MUJOCO_EGL_DEVICE_ID"] = "0"
                env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
                log = (log_root / f"{task}.log").open("a", encoding="utf-8")
                process = subprocess.Popen(
                    command,
                    cwd=repo_root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active.append((task, process, log))
            failures: list[str] = []
            for task, process, log in active:
                code = process.wait()
                log.close()
                if code != 0:
                    failures.append(f"{task}:exit={code}")
            if failures:
                raise RuntimeError("paired worker failure: " + ", ".join(failures))
        except BaseException:
            for _task, process, log in active:
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


def _parse_max_steps(value: str, prepared_data: Path) -> dict[str, int]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("--task-max-steps-json must be valid JSON") from error
    if not isinstance(parsed, Mapping) or set(parsed) != set(TASKS):
        raise ValueError("task max-step map must cover exactly all 11 DuoBench tasks")
    manifest = load_manifest(prepared_data, require_formal=True)
    result = {task: int(parsed[task]) for task in TASKS}
    for task in TASKS:
        expected = int(manifest["tasks"][task]["validation_max_steps"])
        if result[task] != expected:
            raise ValueError(f"max steps for {task} differ from the frozen manifest: {result[task]} != {expected}")
    return result


def _validate_task_result(
    row: object,
    *,
    task: str,
    episodes: int,
    seed_start: int,
    max_steps: int,
    smoke: bool,
    bcore_checkpoint_sha256: str,
    care_checkpoint_sha256: str,
) -> None:
    """Fail closed on a worker result before aggregation."""

    if not isinstance(row, Mapping):
        raise RuntimeError(f"paired task result is not a mapping: {task}")
    expected = {
        "schema": SMOKE_TASK_SCHEMA if smoke else TASK_SCHEMA,
        "status": "complete",
        "task": task,
        "episodes": int(episodes),
        "max_steps": int(max_steps),
        "seed_start": int(seed_start),
        "seeds": [int(seed_start + index) for index in range(episodes)],
        "benchmark_adapter": "DuoBench",
        "method_family": METHOD_FAMILY,
        "reference_policy_family": REFERENCE_POLICY_FAMILY,
        "base_policy_family": BASE_POLICY_FAMILY,
        "reference_architecture": "PredictiveTeamBeliefPolicy_direct_belief_residual",
        "vision": "dinov3_vitb16_frozen",
        "vision_backbone": "dinov3_vitb16_frozen",
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "strictly_decentralized": True,
        "strict_local": True,
        "per_robot_independent_inputs": True,
        "act_provider_allowed": False,
        "selector_off_semantics": "candidate0_complete_bcore_reference",
        "override_contract": "at_most_one_focal_arm_per_control_step",
        "strict_dino_contract": True,
        "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
        "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
        "care_memory_semantics": DUO_CARE_MEMORY_SEMANTICS,
        "care_memory_tokens": DUO_CARE_MEMORY_TOKENS,
        "bcore_checkpoint_sha256": bcore_checkpoint_sha256,
        "care_checkpoint_sha256": care_checkpoint_sha256,
    }
    differences = {
        key: (row.get(key), value)
        for key, value in expected.items()
        if row.get(key) != value
    }
    if differences:
        raise RuntimeError(f"paired task result contract differs for {task}: {differences}")
    rows = row.get("rows")
    expected_keys = [
        (mode, int(seed_start + index))
        for index in range(episodes)
        for mode in ("selector_off", "care")
    ]
    if not isinstance(rows, list) or len(rows) != 2 * episodes:
        raise RuntimeError(f"paired rollout rows are incomplete for {task}")
    actual_keys = [(item.get("mode"), item.get("seed")) for item in rows]
    if actual_keys != expected_keys:
        raise RuntimeError(f"paired rollout seed/mode order differs for {task}")
    for item in rows:
        if not _valid_recovered_episode(
            item,
            task=task,
            mode=str(item["mode"]),
            seed=int(item["seed"]),
            max_steps=max_steps,
            bcore_checkpoint_sha256=bcore_checkpoint_sha256,
            care_checkpoint_sha256=care_checkpoint_sha256,
        ):
            raise RuntimeError(f"paired rollout provenance differs for {task}")
    pairs = row.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != episodes:
        raise RuntimeError(f"paired seed rows are incomplete for {task}")
    expected_seeds = [int(seed_start + index) for index in range(episodes)]
    if [pair.get("seed") for pair in pairs] != expected_seeds:
        raise RuntimeError(f"paired seed order differs for {task}")
    required_pair = {
        "task": task,
        "max_steps": int(max_steps),
        "strictly_decentralized": True,
        "strict_local": True,
        "per_robot_independent_inputs": True,
        "act_provider_allowed": False,
        "selector_off_semantics": "candidate0_complete_bcore_reference",
        "override_contract": "at_most_one_focal_arm_per_control_step",
        "strict_dino_contract": True,
        "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
        "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
        "care_memory_semantics": DUO_CARE_MEMORY_SEMANTICS,
        "care_memory_tokens": DUO_CARE_MEMORY_TOKENS,
        "bcore_checkpoint_sha256": bcore_checkpoint_sha256,
        "care_checkpoint_sha256": care_checkpoint_sha256,
    }
    for pair in pairs:
        if not isinstance(pair, Mapping) or any(
            pair.get(key) != value for key, value in required_pair.items()
        ):
            raise RuntimeError(f"paired seed provenance differs for {task}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcore-checkpoint", type=Path, required=True)
    parser.add_argument("--care-checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--dino-model", type=Path, required=True)
    parser.add_argument("--duobench-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=20260830)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--task-max-steps-json", required=True)
    parser.add_argument("--task", action="append", choices=TASKS)
    parser.add_argument("--worker-task", choices=TASKS, help=argparse.SUPPRESS)
    parser.add_argument("--max-steps", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda:0", help=argparse.SUPPRESS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    args.output = args.output.resolve()
    # Parent invocations write a directory; worker invocations write one JSON
    # file.  Do not pre-create the latter as a directory or the atomic result
    # write in ``evaluate_task`` can never succeed.
    if args.worker_task is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
    if args.worker_task is not None:
        if args.max_steps is None:
            raise ValueError("worker mode requires --max-steps")
        task_max_steps = _parse_max_steps(args.task_max_steps_json, args.prepared_data)
        if int(task_max_steps[args.worker_task]) != int(args.max_steps):
            raise ValueError(
                f"worker max steps for {args.worker_task} differ from frozen map: "
                f"{args.max_steps} != {task_max_steps[args.worker_task]}"
            )
        validate_selected_inputs(
            bcore_checkpoint=args.bcore_checkpoint,
            care_checkpoint=args.care_checkpoint,
            prepared_data=args.prepared_data,
            dino_model=args.dino_model,
        )
        evaluate_task(
            bcore_checkpoint=args.bcore_checkpoint,
            care_checkpoint=args.care_checkpoint,
            prepared_data=args.prepared_data,
            dino_model=args.dino_model,
            duobench_root=args.duobench_root,
            output=args.output,
            task=args.worker_task,
            episodes=args.episodes,
            seed_start=args.seed_start,
            max_steps=args.max_steps,
            device=args.device,
            smoke=args.smoke,
        )
        return

    provenance = validate_selected_inputs(
        bcore_checkpoint=args.bcore_checkpoint,
        care_checkpoint=args.care_checkpoint,
        prepared_data=args.prepared_data,
        dino_model=args.dino_model,
    )
    max_steps = _parse_max_steps(args.task_max_steps_json, args.prepared_data)
    tasks = tuple(args.task or TASKS)
    if tasks != TASKS:
        raise RuntimeError(
            "formal paired validation must cover all 11 DuoBench tasks in frozen order"
        )
    _run_workers(args, tasks, max_steps)
    task_results = []
    for task in tasks:
        path = args.output / "tasks" / f"{task}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        row = json.loads(path.read_text(encoding="utf-8"))
        _validate_task_result(
            row,
            task=task,
            episodes=args.episodes,
            seed_start=args.seed_start,
            max_steps=max_steps[task],
            smoke=args.smoke,
            bcore_checkpoint_sha256=provenance["bcore_checkpoint_sha256"],
            care_checkpoint_sha256=provenance["care_checkpoint_sha256"],
        )
        task_results.append(row)
    rows = [row for result in task_results for row in result["rows"]]
    pairs = [pair for result in task_results for pair in result["pairs"]]
    off = [row for row in rows if row["mode"] == "selector_off"]
    care = [row for row in rows if row["mode"] == "care"]
    expected_pairs = args.episodes * len(tasks)
    if len(pairs) != expected_pairs:
        raise RuntimeError(f"paired result count differs: {len(pairs)} != {expected_pairs}")
    summary = {
        "status": "complete",
        "format_version": SMOKE_SCHEMA if args.smoke else SUMMARY_SCHEMA,
        "benchmark_adapter": "DuoBench",
        "reference_policy_family": REFERENCE_POLICY_FAMILY,
        "base_policy_family": BASE_POLICY_FAMILY,
        "method_family": METHOD_FAMILY,
        "vision": "dinov3_vitb16_frozen",
        "vision_backbone": "dinov3_vitb16_frozen",
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "strict_dino_contract": True,
        "strictly_decentralized": True,
        "strict_local": True,
        "act_provider_allowed": False,
        "per_robot_independent_inputs": True,
        "selector_off_semantics": "candidate0_complete_bcore_reference",
        "override_contract": "at_most_one_focal_arm_per_control_step",
        "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
        "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
        "care_memory_semantics": DUO_CARE_MEMORY_SEMANTICS,
        "care_memory_tokens": DUO_CARE_MEMORY_TOKENS,
        "bcore_checkpoint_sha256": provenance["bcore_checkpoint_sha256"],
        "care_checkpoint_sha256": provenance["care_checkpoint_sha256"],
        "prepared_manifest_sha256": provenance["prepared_manifest_sha256"],
        "episodes_per_task": int(args.episodes),
        "seed_start_per_task": int(args.seed_start),
        "task_specific_max_steps": max_steps,
        "total_pairs": len(pairs),
        "selector_off_success_rate": float(np.mean([row["success"] for row in off])) if off else 0.0,
        "care_success_rate": float(np.mean([row["success"] for row in care])) if care else 0.0,
        "paired_success_improvement": float(np.mean([pair["success_delta"] for pair in pairs])) if pairs else 0.0,
        "selector_off_mean_progress": float(np.mean([row["final_stage_progress"] for row in off])) if off else 0.0,
        "care_mean_progress": float(np.mean([row["final_stage_progress"] for row in care])) if care else 0.0,
        "override_rate": float(np.mean([row["override_rate"] for row in care])) if care else 0.0,
        "harmful_override_rate": float(np.mean([pair["harmful_override"] for pair in pairs])) if pairs else 0.0,
        "tasks": {
            result["task"]: {
                "episodes": int(result["episodes"]),
                "max_steps": int(result["max_steps"]),
                "selector_off_success_rate": result["selector_off_success_rate"],
                "care_success_rate": result["care_success_rate"],
                "paired_success_improvement": result["paired_success_improvement"],
                "selector_off_mean_progress": result["selector_off_mean_progress"],
                "care_mean_progress": result["care_mean_progress"],
                "override_rate": result["override_rate"],
                "harmful_override_rate": result["harmful_override_rate"],
            }
            for result in task_results
        },
        "pairs": pairs,
        "task_results": {
            result["task"]: str((args.output / "tasks" / f"{result['task']}.json").resolve())
            for result in task_results
        },
    }
    _atomic_json(args.output / "summary.json", summary)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "status",
                    "total_pairs",
                    "selector_off_success_rate",
                    "care_success_rate",
                    "paired_success_improvement",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "SUMMARY_SCHEMA",
    "TASK_SCHEMA",
    "evaluate_task",
    "main",
    "run_episode",
    "validate_selected_inputs",
]
