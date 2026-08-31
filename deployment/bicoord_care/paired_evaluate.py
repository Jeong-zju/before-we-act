"""Paired selector-off/CARE closed-loop evaluation for BiCoord.

For every expert-valid seed this worker executes selector-off first, restores
the exact initial SAPIEN/controller/RNG state, verifies the restore hash, and
then executes CARE.  The two arms share the frozen B-core and CARE weights but
are scored independently from their own belief/event memory and local action
candidates; there is no cross-arm lower-bound arbitration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
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

from .bcore_data import (
    BICOORD_CARE_MEMORY_SEMANTICS,
    BICOORD_CARE_MEMORY_TOKENS,
    BICOORD_CARE_MEMORY_WIDTH,
)
from .bicoord_snapshot import capture_state, restore_state, state_sha256
from .branch_collection import CANDIDATES, candidate_plan
from .config import (
    ACTION_DIM,
    ACTION_ENCODING,
    ACTION_HORIZON,
    TASKS,
    VALIDATION_EPISODES,
    VALIDATION_MAX_STEPS,
)
from .data import load_normalization_receipt, project_local_observation
from .evaluate_bcore import _checkpoint, _make_env, _official_seeds, _runtime
from .stage_common import (
    artifact,
    assert_common_paths,
    atomic_json,
    common_parser,
    canonical_sha256,
    publish_result,
    require_hashed_artifact,
    require_stage_result,
    sha256_file,
)


PAIRED_RECEIPT_SCHEMA = "before-we-act.bicoord.care-paired-progress/1"
DEPLOYMENT_FORMAT = "before-we-act.bicoord.care-deployment/1"
TRAINING_FORMAT = "before-we-act.care-bicoord-training/1"
REGISTERED_SELECTOR = {
    "selector_delta": 0.0,
    "hard_safety_probability_max": 0.25,
    "nominal_simultaneous_coverage": 0.9,
    "primary_horizon": 16,
}


def _close_env(env: Any) -> None:
    try:
        env.close_env()
    except Exception:
        try:
            env.close()
        except Exception:
            pass


def _care_checkpoint(args: argparse.Namespace, *, formal: bool) -> Path:
    if formal:
        dependency = require_stage_result(
            args.run,
            "offline_selection_calibration",
            config_sha256=args.config_sha256,
        )
        return require_hashed_artifact(
            dependency, kind="care_deployment_checkpoint"
        )
    dependency = require_stage_result(
        args.run, "belief_smoke_train", config_sha256=args.config_sha256
    )
    return require_hashed_artifact(dependency, kind="belief_checkpoint")


def _validate_config(config: CAREBeliefConfig, action_std: np.ndarray) -> None:
    expected = {
        "d_model": BICOORD_CARE_MEMORY_WIDTH,
        "action_dim": ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "candidates": CANDIDATES,
        "outcome_components": 3,
    }
    for name, value in expected.items():
        if getattr(config, name) != value:
            raise RuntimeError(f"CARE checkpoint differs at {name}")
    configured = np.asarray(config.action_std, dtype=np.float32)
    if configured.shape != (ACTION_DIM,) or not np.array_equal(configured, action_std):
        raise RuntimeError("CARE action encoder scale differs from B-core action normalization")


def _load_care(
    path: Path,
    *,
    device: torch.device,
    reference_checkpoint_sha256: str,
    action_std: np.ndarray,
    formal: bool,
) -> tuple[CAREBeliefHead, CARECalibration, Mapping[str, Any]]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(saved, Mapping):
        raise ValueError("CARE checkpoint is not a mapping")
    expected_format = DEPLOYMENT_FORMAT if formal else TRAINING_FORMAT
    if saved.get("format_version") != expected_format:
        raise ValueError(f"CARE checkpoint format differs: {saved.get('format_version')!r}")
    if saved.get("benchmark_adapter") != "BiCoord" or saved.get("method_family") != "CARE":
        raise ValueError("CARE checkpoint benchmark/method provenance differs")
    if saved.get("policy_family") != "CAREBeliefHead" or saved.get("reference_policy") != "B-core/TUNE":
        raise ValueError("CARE checkpoint policy provenance differs")
    # Both smoke and formal CARE rollouts must be cryptographically bound to
    # the exact B-core/TUNE checkpoint used to build their branch data.  Smoke
    # is an interface gate, not a provenance exception: accepting an
    # unbound/mismatched smoke head could make the later formal resume appear
    # healthy while exercising a different reference policy.
    if saved.get("reference_checkpoint_sha256") != reference_checkpoint_sha256:
        raise ValueError("CARE checkpoint is not bound to the selected B-core checkpoint")
    config = CAREBeliefConfig.from_mapping(saved["config"])
    _validate_config(config, action_std)
    model = CAREBeliefHead(config).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval().requires_grad_(False)
    if formal:
        value = saved.get("calibration")
        if not isinstance(value, Mapping):
            raise ValueError("CARE deployment has no calibration")
        calibration = CARECalibration.from_mapping(value)
        for name, expected in REGISTERED_SELECTOR.items():
            if getattr(calibration, name) != expected:
                raise ValueError(f"CARE calibration differs at {name}")
        if not math.isfinite(calibration.lower_correction) or calibration.lower_correction < 0:
            raise ValueError("CARE conformal lower correction is invalid")
        if saved.get("oof_calibration_complete") is not True:
            raise ValueError("CARE deployment lacks complete OOF calibration provenance")
    else:
        calibration = CARECalibration(
            lower_correction=0.0,
            selector_delta=REGISTERED_SELECTOR["selector_delta"],
            hard_safety_probability_max=REGISTERED_SELECTOR[
                "hard_safety_probability_max"
            ],
            nominal_simultaneous_coverage=REGISTERED_SELECTOR[
                "nominal_simultaneous_coverage"
            ],
            primary_horizon=REGISTERED_SELECTOR["primary_horizon"],
        )
    return model, calibration, saved


def _normalization_audit(args: argparse.Namespace, runtime: Any) -> dict[str, Any]:
    path = args.run / "artifacts" / "dataset_audit" / "normalization.json"
    value = load_normalization_receipt(path, require_formal=True)
    pairs = {
        "qpos_mean": runtime.q_mean,
        "qpos_std": runtime.q_std,
        "action_mean": runtime.a_mean,
        "action_std": runtime.a_std,
    }
    for name, runtime_value in pairs.items():
        source = np.asarray(value[name], dtype=np.float32)
        if not np.array_equal(source, np.asarray(runtime_value, dtype=np.float32)):
            raise RuntimeError(f"paired evaluation normalization differs at {name}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "qpos_min": value["qpos_min"],
        "qpos_max": value["qpos_max"],
        "action_min": value["action_min"],
        "action_max": value["action_max"],
        "state_clipping": False,
        "action_clipping": False,
        "image_preprocessing_owner": "BiCoordBcoreRuntime/frozen_DINOv3",
    }


def _observation_sha256(observation: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for arm in (0, 1):
        local = project_local_observation(observation, arm)
        for name in ("head_rgb", "wrist_rgb", "state"):
            value = np.ascontiguousarray(np.asarray(local[name]))
            digest.update(str(arm).encode())
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(repr(tuple(value.shape)).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def _success_progress(env: Any) -> tuple[bool, float]:
    try:
        checked = bool(env.check_success())
    except Exception:
        checked = False
    success = bool(getattr(env, "eval_success", False) or checked)
    progress = float(getattr(env, "stage_eval_score", float(success)))
    if not math.isfinite(progress):
        raise RuntimeError("BiCoord paired progress is non-finite")
    return success, float(np.clip(progress, 0, 1))


def _candidate_tensor(context: Any) -> np.ndarray:
    rows = []
    for arm in (0, 1):
        rows.append(
            np.stack(
                [
                    candidate_plan(
                        candidate,
                        context.reference_plan[arm],
                        context.base_plan[arm],
                        context.current_qpos[arm],
                    )
                    for candidate in range(CANDIDATES)
                ]
            )
        )
    value = np.stack(rows).astype(np.float32)
    if value.shape != (2, CANDIDATES, ACTION_HORIZON, ACTION_DIM):
        raise RuntimeError("paired CARE candidate tensor differs")
    return value


def _append_progress(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _reset_incomplete_progress(*paths: Path) -> None:
    """Discard only an uncommitted seed attempt before replaying that pair."""
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            stream.flush()
            os.fsync(stream.fileno())


def _combine_seed_progress(rows: Sequence[Mapping[str, Any]], mode: str, target: Path) -> None:
    if mode not in {"selector_off", "care"}:
        raise ValueError(mode)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for row in rows:
                source = Path(str(row[f"{mode}_progress"]))
                if not source.is_file() or sha256_file(source) != row.get(f"{mode}_progress_sha256"):
                    raise RuntimeError(f"paired {mode} seed progress changed: {source}")
                value = source.read_text(encoding="utf-8")
                if not value.strip():
                    raise RuntimeError(f"paired {mode} seed progress is empty: {source}")
                output.write(value)
                if not value.endswith("\n"):
                    output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@torch.inference_mode()
def _episode(
    *,
    env: Any,
    runtime: Any,
    care: CAREBeliefHead,
    calibration: CARECalibration,
    task: str,
    seed: int,
    max_steps: int,
    selector_enabled: bool,
    progress_path: Path,
    action_min: np.ndarray,
    action_max: np.ndarray,
    initial_observation: Mapping[str, Any],
) -> dict[str, Any]:
    mode = "care" if selector_enabled else "selector_off"
    runtime.reset(task)
    # The caller supplies the observation captured from the exact paired
    # initial state.  Do not advance a camera/RNG stream with an extra get_obs.
    observation = initial_observation
    action_digest = hashlib.sha256()
    reference_digest = hashlib.sha256()
    override_steps = 0
    candidate_counts = {arm: {candidate: 0 for candidate in range(CANDIDATES)} for arm in (0, 1)}
    safety_rejections = {0: 0, 1: 0}
    uncertainty_fallbacks = {0: 0, 1: 0}
    out_of_source_range = 0
    success = False
    progress = 0.0
    inference_seconds: list[float] = []
    for step in range(max_steps):
        started = time.perf_counter()
        context = runtime.act_with_context(
            observation, task, belief_enabled=True, commit=False
        )
        if (
            context.memory.shape
            != (2, BICOORD_CARE_MEMORY_TOKENS, BICOORD_CARE_MEMORY_WIDTH)
            or context.memory_mask.shape != (2, BICOORD_CARE_MEMORY_TOKENS)
        ):
            raise RuntimeError("paired evaluation received the wrong CARE memory")
        candidates = _candidate_tensor(context)
        actions = context.reference_plan[:, 0].copy()
        selected = np.zeros(2, dtype=np.int64)
        lower_bounds = np.zeros(2, dtype=np.float32)
        unsafe = np.zeros((2, CANDIDATES), dtype=bool)
        if selector_enabled:
            horizon = care.config.horizons.index(calibration.primary_horizon)
            candidate_tensor = torch.from_numpy(candidates).to(runtime.device)
            memory = torch.from_numpy(context.memory).to(runtime.device)
            memory_mask = torch.from_numpy(context.memory_mask).to(runtime.device)
            horizon_index = torch.full(
                (2,), horizon, dtype=torch.long, device=runtime.device
            )
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=runtime.device.type == "cuda"
            ):
                output = care(memory, memory_mask, candidate_tensor, horizon_index)
            chosen, best_lower, unsafe_tensor = select_care_candidate(
                output, calibration, variant=care.config.variant
            )
            selected = chosen.detach().cpu().numpy().astype(np.int64)
            lower_bounds = best_lower.detach().float().cpu().numpy()
            unsafe = unsafe_tensor.detach().cpu().numpy()
            # Each local stream makes its own decision.  There is deliberately
            # no max/lower-bound arbitration between arms.
            for arm in (0, 1):
                candidate = int(selected[arm])
                safety_rejections[arm] += int(unsafe[arm, 1:].sum())
                if candidate == 0:
                    uncertainty_fallbacks[arm] += 1
                else:
                    actions[arm] = candidates[arm, candidate, 0]
                    candidate_counts[arm][candidate] += 1
            if bool(np.any(selected != 0)):
                override_steps += 1
        if actions.shape != (2, ACTION_DIM) or not np.isfinite(actions).all():
            raise RuntimeError("paired CARE produced a non-finite native action")
        out_of_source_range += int(
            np.count_nonzero(
                (actions < action_min[None]) | (actions > action_max[None])
            )
        )
        for arm in (0, 1):
            action_digest.update(actions[arm].astype(np.float32).tobytes())
            reference_digest.update(
                context.reference_plan[arm, 0].astype(np.float32).tobytes()
            )
        runtime.record_executed_actions(actions)
        env.take_action(actions.reshape(-1))
        observation = env.get_obs()
        success, progress = _success_progress(env)
        elapsed = time.perf_counter() - started
        inference_seconds.append(elapsed)
        _append_progress(
            progress_path,
            {
                "task": task,
                "seed": int(seed),
                "mode": mode,
                "step": step + 1,
                "max_steps": max_steps,
                "progress": progress,
                "success": success,
                "selected_candidates": selected.tolist(),
                "selected_lower_bounds": lower_bounds.tolist(),
                "per_arm_independent_selector": True,
                "cross_arm_arbitration": False,
                "action_clipped": False,
            },
        )
        if success:
            break
    steps = len(inference_seconds)
    if steps < 1:
        raise RuntimeError("paired evaluation executed no controller step")
    return {
        "task": task,
        "seed": int(seed),
        "mode": mode,
        "success": success,
        "steps": steps,
        "max_steps": max_steps,
        "final_progress": progress,
        "override_steps": override_steps,
        "override_rate": override_steps / steps,
        "candidate_counts": {
            str(arm): {str(key): value for key, value in rows.items() if value}
            for arm, rows in candidate_counts.items()
        },
        "predicted_safety_rejections": {str(key): value for key, value in safety_rejections.items()},
        "uncertainty_fallback_steps": {str(key): value for key, value in uncertainty_fallbacks.items()},
        "commands_outside_training_population_range": out_of_source_range,
        "action_clipping": False,
        "state_clipping": False,
        "per_arm_independent_selector": True,
        "cross_arm_lower_bound_arbitration": False,
        "strictly_decentralized": True,
        "action_trace_sha256": action_digest.hexdigest(),
        "reference_action_trace_sha256": reference_digest.hexdigest(),
        "mean_inference_seconds": float(np.mean(inference_seconds)),
        "p95_inference_seconds": float(np.quantile(inference_seconds, 0.95)),
    }


def _paired_seed(
    *,
    args: argparse.Namespace,
    runtime: Any,
    care: CAREBeliefHead,
    calibration: CARECalibration,
    task: str,
    seed: int,
    max_steps: int,
    selector_off_progress: Path,
    care_progress: Path,
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> dict[str, Any]:
    env = _make_env(args.benchmark_repo, task, seed)
    try:
        runtime.reset(task)
        initial_observation = env.get_obs()
        initial_observation_hash = _observation_sha256(initial_observation)
        initial_state = capture_state(env)
        initial_state_hash = state_sha256(initial_state)
        selector_off = _episode(
            env=env,
            runtime=runtime,
            care=care,
            calibration=calibration,
            task=task,
            seed=seed,
            max_steps=max_steps,
            selector_enabled=False,
            progress_path=selector_off_progress,
            action_min=action_min,
            action_max=action_max,
            initial_observation=initial_observation,
        )
        restore_state(env, initial_state)
        restored_state_hash = state_sha256(capture_state(env))
        if restored_state_hash != initial_state_hash:
            raise RuntimeError("paired selector-off/CARE initial simulator state differs")
        restored_observation = env.get_obs()
        restored_observation_hash = _observation_sha256(restored_observation)
        if restored_observation_hash != initial_observation_hash:
            raise RuntimeError("paired selector-off/CARE initial policy observation differs")
        # Rerendering is a diagnostic.  Restore once more so CARE starts from
        # exactly the same post-observation simulator/RNG state as control.
        restore_state(env, initial_state)
        if state_sha256(capture_state(env)) != initial_state_hash:
            raise RuntimeError("paired state drifted after the rerender probe")
        care_row = _episode(
            env=env,
            runtime=runtime,
            care=care,
            calibration=calibration,
            task=task,
            seed=seed,
            max_steps=max_steps,
            selector_enabled=True,
            progress_path=care_progress,
            action_min=action_min,
            action_max=action_max,
            initial_observation=initial_observation,
        )
        return {
            "task": task,
            "seed": int(seed),
            "paired": True,
            "execution_order": ["selector_off", "care"],
            "same_initial_simulator_state": True,
            "same_initial_observation": True,
            "initial_state_sha256": initial_state_hash,
            "restored_state_sha256": restored_state_hash,
            "initial_observation_sha256": initial_observation_hash,
            "restored_observation_sha256": restored_observation_hash,
            "selector_off": selector_off,
            "care": care_row,
        }
    finally:
        _close_env(env)


def _seed_manifest_receipt(args: argparse.Namespace) -> tuple[str, str]:
    stage = "seed_discovery_smoke" if args.operation == "smoke-paired" else "seed_discovery"
    dependency = require_stage_result(args.run, stage, config_sha256=args.config_sha256)
    path = Path(str(dependency.get("seed_manifest", "")))
    digest = dependency.get("seed_manifest_sha256")
    if not path.is_file() or not isinstance(digest, str) or sha256_file(path) != digest:
        raise RuntimeError("paired evaluation seed manifest changed")
    return str(path.resolve()), digest


def _pair_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    """Fields which bind a resumable pair to immutable run inputs."""
    return {
        "task": row.get("task"),
        "seed": int(row.get("seed", -1)),
        "max_steps": int(row.get("max_steps", -1)),
        "reference_checkpoint_sha256": row.get("reference_checkpoint_sha256"),
        "care_checkpoint_sha256": row.get("care_checkpoint_sha256"),
        "seed_manifest_sha256": row.get("seed_manifest_sha256"),
        "normalization_sha256": row.get("normalization_sha256"),
        "operation": row.get("operation"),
    }


def _write_pair(path: Path, row: Mapping[str, Any], *, identity: Mapping[str, Any]) -> None:
    value = dict(row)
    value["pair_identity"] = dict(identity)
    value["pair_identity_sha256"] = canonical_sha256(identity)
    value["pair_payload_sha256"] = canonical_sha256(value)
    atomic_json(path, value)


def _read_completed_pair(path: Path, *, identity: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid resumable paired result {path}: {error}") from error
    if not isinstance(value, Mapping) or value.get("paired") is not True:
        raise RuntimeError(f"resumable paired result is not a complete pair: {path}")
    expected_payload_sha = value.get("pair_payload_sha256")
    unsigned = dict(value); unsigned.pop("pair_payload_sha256", None)
    if not isinstance(expected_payload_sha, str) or expected_payload_sha != canonical_sha256(unsigned):
        raise RuntimeError(f"resumable paired result payload hash differs: {path}")
    if dict(value.get("pair_identity", {})) != dict(identity) or value.get("pair_identity_sha256") != canonical_sha256(identity):
        raise RuntimeError(f"resumable paired result provenance differs: {path}")
    if not isinstance(value.get("selector_off"), Mapping) or not isinstance(value.get("care"), Mapping):
        raise RuntimeError(f"resumable paired result is incomplete: {path}")
    for mode in ("selector_off", "care"):
        progress = Path(str(value.get(f"{mode}_progress", "")))
        digest = value.get(f"{mode}_progress_sha256")
        if not progress.is_file() or not isinstance(digest, str) or sha256_file(progress) != digest:
            raise RuntimeError(f"resumable paired {mode} progress differs: {path}")
    return dict(value)


def run(args: argparse.Namespace) -> dict[str, Any]:
    assert_common_paths(args)
    task = getattr(args, "task", None)
    if task not in TASKS:
        raise ValueError(f"invalid paired BiCoord task: {task}")
    formal = args.operation == "validation20-paired"
    expected_episodes = VALIDATION_EPISODES if formal else 1
    episodes = int(getattr(args, "episodes", expected_episodes))
    if episodes != expected_episodes:
        raise ValueError(f"{args.operation} requires {expected_episodes} paired seeds")
    max_steps = int(getattr(args, "max_steps", 0) or VALIDATION_MAX_STEPS[task])
    if max_steps != VALIDATION_MAX_STEPS[task]:
        raise ValueError(f"{task}: paired max steps must be {VALIDATION_MAX_STEPS[task]}")
    reference_checkpoint = _checkpoint(args)
    reference_sha = sha256_file(reference_checkpoint)
    runtime = _runtime(reference_checkpoint, args)
    normalization = _normalization_audit(args, runtime)
    care_checkpoint = _care_checkpoint(args, formal=formal)
    care, calibration, care_saved = _load_care(
        care_checkpoint,
        device=runtime.device,
        reference_checkpoint_sha256=reference_sha,
        action_std=np.asarray(runtime.a_std, dtype=np.float32),
        formal=formal,
    )
    seed_manifest_path, seed_manifest_sha = _seed_manifest_receipt(args)
    seeds = _official_seeds(args, task, count=episodes)
    progress_root = args.run / "progress" / args.operation
    selector_off_progress = progress_root / f"{task}.selector_off.jsonl"
    care_progress = progress_root / f"{task}.care.jsonl"
    pair_root = progress_root / "pairs" / task
    pair_root.mkdir(parents=True, exist_ok=True)
    pair_identity_base = {
        "task": task,
        "max_steps": max_steps,
        "reference_checkpoint_sha256": reference_sha,
        "care_checkpoint_sha256": sha256_file(care_checkpoint),
        "seed_manifest_sha256": seed_manifest_sha,
        "normalization_sha256": normalization["sha256"],
        "operation": args.operation,
    }
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        pair_path = pair_root / f"seed_{int(seed)}.json"
        identity = {**pair_identity_base, "seed": int(seed)}
        seed_selector_progress = progress_root / f"{task}.seed_{int(seed)}.selector_off.jsonl"
        seed_care_progress = progress_root / f"{task}.seed_{int(seed)}.care.jsonl"
        if pair_path.is_file():
            # A completed pair is independently persisted before the worker
            # emits its final receipt.  Reuse it only when every immutable
            # input (including checkpoint and seed-manifest hashes) agrees.
            row = _read_completed_pair(pair_path, identity=identity)
        else:
            _reset_incomplete_progress(seed_selector_progress, seed_care_progress)
            row = _paired_seed(
                args=args,
                runtime=runtime,
                care=care,
                calibration=calibration,
                task=task,
                seed=seed,
                max_steps=max_steps,
                selector_off_progress=seed_selector_progress,
                care_progress=seed_care_progress,
                action_min=np.asarray(normalization["action_min"], dtype=np.float32),
                action_max=np.asarray(normalization["action_max"], dtype=np.float32),
            )
            row.update(
                {
                    "operation": args.operation,
                    "reference_checkpoint_sha256": reference_sha,
                    "care_checkpoint_sha256": sha256_file(care_checkpoint),
                    "seed_manifest_sha256": seed_manifest_sha,
                    "normalization_sha256": normalization["sha256"],
                    "max_steps": max_steps,
                    "selector_off_progress": str(seed_selector_progress.resolve()),
                    "care_progress": str(seed_care_progress.resolve()),
                    "selector_off_progress_sha256": sha256_file(seed_selector_progress),
                    "care_progress_sha256": sha256_file(seed_care_progress),
                }
            )
            _write_pair(pair_path, row, identity=identity)
            row = _read_completed_pair(pair_path, identity=identity)
        # Legacy/hand-created pair files are not accepted unless they carry
        # the per-seed progress receipts needed for crash-safe aggregation.
        if "selector_off_progress" not in row or "care_progress" not in row:
            raise RuntimeError(f"paired seed file lacks progress provenance: {pair_path}")
        row["pair_file"] = str(pair_path.resolve())
        row["pair_file_sha256"] = sha256_file(pair_path)
        rows.append(row)
        print(
            json.dumps(
                {
                    "task": task,
                    "seed": seed,
                    "completed_pairs": len(rows),
                    "target_pairs": episodes,
                    "selector_off_success": row["selector_off"]["success"],
                    "care_success": row["care"]["success"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if len(rows) != episodes or [int(row["seed"]) for row in rows] != seeds:
        raise RuntimeError("paired evaluation seed coverage differs")
    _combine_seed_progress(rows, "selector_off", selector_off_progress)
    _combine_seed_progress(rows, "care", care_progress)
    pair_manifest = progress_root / f"{task}.pairs.json"
    atomic_json(
        pair_manifest,
        {
            "schema": "before-we-act.bicoord.care-pair-manifest/1",
            "status": "PASSED",
            "task": task,
            "operation": args.operation,
            "episodes": episodes,
            "identity": pair_identity_base,
            "pairs": [
                {
                    "seed": int(row["seed"]),
                    "path": row["pair_file"],
                    "sha256": row["pair_file_sha256"],
                    "pair_identity_sha256": row.get("pair_identity_sha256"),
                }
                for row in rows
            ],
        },
    )
    receipt = progress_root / f"{task}.receipt.json"
    receipt_value = {
        "schema": PAIRED_RECEIPT_SCHEMA,
        "status": "PASSED",
        "task": task,
        "operation": args.operation,
        "episodes": episodes,
        "completed": episodes,
        "rollouts": episodes * 2,
        "max_steps": max_steps,
        "seeds": seeds,
        "seed_source": "policy_independent_expert_valid_manifest",
        "seed_manifest": seed_manifest_path,
        "seed_manifest_sha256": seed_manifest_sha,
        "paired": True,
        "selector_off_control": True,
        "execution_order": ["selector_off", "care"],
        "all_pairs_same_initial_simulator_state": all(
            row["same_initial_simulator_state"] for row in rows
        ),
        "all_pairs_same_initial_observation": all(
            row["same_initial_observation"] for row in rows
        ),
        "per_arm_independent_selector": True,
        "cross_arm_lower_bound_arbitration": False,
        "reference_checkpoint": str(reference_checkpoint.resolve()),
        "reference_checkpoint_sha256": reference_sha,
        "care_checkpoint": str(care_checkpoint.resolve()),
        "care_checkpoint_sha256": sha256_file(care_checkpoint),
        "normalization": normalization,
        "care_memory_tokens": BICOORD_CARE_MEMORY_TOKENS,
        "care_memory_width": BICOORD_CARE_MEMORY_WIDTH,
        "care_memory_semantics": BICOORD_CARE_MEMORY_SEMANTICS,
        "selector_calibration": {
            "lower_correction": calibration.lower_correction,
            **REGISTERED_SELECTOR,
        },
        "selector_off_successes": sum(int(row["selector_off"]["success"]) for row in rows),
        "care_successes": sum(int(row["care"]["success"]) for row in rows),
        "selector_off_progress": str(selector_off_progress.resolve()),
        "selector_off_progress_sha256": sha256_file(selector_off_progress),
        "care_progress": str(care_progress.resolve()),
        "care_progress_sha256": sha256_file(care_progress),
        "pair_manifest": str(pair_manifest.resolve()),
        "pair_manifest_sha256": sha256_file(pair_manifest),
        "rows": rows,
    }
    atomic_json(receipt, receipt_value)
    stage = "paired_validation20" if formal else "paired_validation_smoke"
    return publish_result(
        args,
        stage=stage,
        include_model_contract=True,
        artifacts=[
            artifact(selector_off_progress, kind="selector_off_progress"),
            artifact(care_progress, kind="care_progress"),
            artifact(receipt, kind="paired_progress_receipt"),
            artifact(pair_manifest, kind="paired_seed_manifest"),
            *(
                artifact(Path(str(row["pair_file"])), kind="paired_seed_result")
                for row in rows
            ),
        ],
        task=task,
        episodes=episodes,
        completed=episodes,
        rollouts=episodes * 2,
        successes=sum(int(row["care"]["success"]) for row in rows),
        selector_off_successes=sum(int(row["selector_off"]["success"]) for row in rows),
        care_successes=sum(int(row["care"]["success"]) for row in rows),
        max_steps=max_steps,
        seeds=seeds,
        seed_source="policy_independent_expert_valid_manifest",
        seed_manifest=seed_manifest_path,
        seed_manifest_sha256=seed_manifest_sha,
        progress_receipt=str(receipt.resolve()),
        progress_receipt_sha256=sha256_file(receipt),
        selector_off_progress=str(selector_off_progress.resolve()),
        selector_off_progress_sha256=sha256_file(selector_off_progress),
        care_progress=str(care_progress.resolve()),
        care_progress_sha256=sha256_file(care_progress),
        pair_manifest=str(pair_manifest.resolve()),
        pair_manifest_sha256=sha256_file(pair_manifest),
        paired_rows=rows,
        paired=True,
        selector_off_control=True,
        execution_order=["selector_off", "care"],
        same_initial_state_verified=True,
        per_arm_independent_selector=True,
        cross_arm_lower_bound_arbitration=False,
        reference_checkpoint=str(reference_checkpoint.resolve()),
        reference_checkpoint_sha256=reference_sha,
        care_checkpoint=str(care_checkpoint.resolve()),
        care_checkpoint_sha256=sha256_file(care_checkpoint),
        action_encoding=ACTION_ENCODING,
        action_horizon=ACTION_HORIZON,
        action_clipping=False,
        state_clipping=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = common_parser(__doc__, ("smoke-paired", "validation20-paired"))
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--record-progress", action="store_true")
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEPLOYMENT_FORMAT", "PAIRED_RECEIPT_SCHEMA", "REGISTERED_SELECTOR", "run"]
