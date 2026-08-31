"""Collect physical same-snapshot CARE branch families on BiCoord.

Every label in this module comes from a SAPIEN rollout.  A family freezes the
simulator, controller, RNG, and B-core local-history state, proves repeatable
restore, and then executes the registered ``6 candidates x 2 response regimes
x 2 repeats`` design.  Offline demonstration error is deliberately absent:
it is not a counterfactual outcome and must never authorize CARE training.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from before_we_act.care_training_data import ORDINARY_WEIGHTS

from .bcore_data import (
    BICOORD_CARE_MEMORY_SEMANTICS,
    BICOORD_CARE_MEMORY_TOKENS,
    BICOORD_CARE_MEMORY_WIDTH,
    BICOORD_FUTURE_OFFSETS_STEPS,
)
from .bicoord_snapshot import (
    SNAPSHOT_TOLERANCE,
    capture_state,
    restore_probe,
    restore_state,
    state_sha256,
)
from .config import ACTION_DIM, ACTION_HORIZON, TASKS, VALIDATION_MAX_STEPS
from .data import load_normalization_receipt, project_local_observation
from .evaluate_bcore import _checkpoint, _make_env, _official_seeds, _runtime
from .stage_common import (
    artifact,
    assert_common_paths,
    atomic_json,
    common_parser,
    publish_result,
    require_stage_result,
    sha256_file,
)


BRANCH_SCHEMA = "before-we-act.bicoord.care-physical-branch-family/1"
SHARD_SCHEMA = "before-we-act.bicoord.care-physical-branch-shard/1"
HORIZONS = (8, 16, 32, 64)
CANDIDATES = 6
BRANCHES_PER_FAMILY = CANDIDATES * 2 * 2
INTERVENTION_STEPS = 1
MAX_BRANCH_STEPS = max(HORIZONS)


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **values)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _close_env(env: Any) -> None:
    try:
        env.close_env()
    except Exception:
        try:
            env.close()
        except Exception:
            pass


def _normalization_audit(args: argparse.Namespace, runtime: Any) -> dict[str, Any]:
    path = args.run / "artifacts" / "dataset_audit" / "normalization.json"
    normalization = load_normalization_receipt(path, require_formal=True)
    expected = {
        "qpos_mean": np.asarray(runtime.q_mean, dtype=np.float32),
        "qpos_std": np.asarray(runtime.q_std, dtype=np.float32),
        "action_mean": np.asarray(runtime.a_mean, dtype=np.float32),
        "action_std": np.asarray(runtime.a_std, dtype=np.float32),
    }
    for name, observed in expected.items():
        source = np.asarray(normalization[name], dtype=np.float32)
        if source.shape != (ACTION_DIM,) or not np.array_equal(source, observed):
            raise RuntimeError(f"B-core runtime normalization differs at {name}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "action_min": list(map(float, normalization["action_min"])),
        "action_max": list(map(float, normalization["action_max"])),
        "action_std": list(map(float, normalization["action_std"])),
        "state_clipping": False,
        "action_clipping": False,
        "image_normalization": "frozen_DINOv3_preprocessing_from_B-core_runtime",
    }


def time_warp_plan(reference: np.ndarray, current: np.ndarray, scale: float) -> np.ndarray:
    """Warp an absolute 7-D plan without clipping or range redefinition."""

    reference = np.asarray(reference, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    if reference.shape != (ACTION_HORIZON, ACTION_DIM) or current.shape != (ACTION_DIM,):
        raise ValueError("BiCoord CARE time warp expects [100,7] and [7]")
    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError("BiCoord CARE time warp received a non-finite action")
    knots = np.concatenate((current[None], reference), axis=0)
    result = np.empty_like(reference)
    for index in range(ACTION_HORIZON):
        tau = min((index + 1) * float(scale), float(ACTION_HORIZON))
        lower = int(math.floor(tau))
        upper = min(lower + 1, ACTION_HORIZON)
        fraction = tau - lower
        result[index, :6] = (
            (1.0 - fraction) * knots[lower, :6]
            + fraction * knots[upper, :6]
        )
        # The continuous native gripper target is sampled, never thresholded.
        result[index, 6] = knots[lower, 6]
    return result


def candidate_plan(
    candidate: int,
    reference: np.ndarray,
    base: np.ndarray,
    current_qpos: np.ndarray,
) -> np.ndarray:
    """Return one of the six frozen upstream CARE candidate transforms."""

    reference = np.asarray(reference, dtype=np.float32)
    base = np.asarray(base, dtype=np.float32)
    current_qpos = np.asarray(current_qpos, dtype=np.float32)
    expected = (ACTION_HORIZON, ACTION_DIM)
    if reference.shape != expected or base.shape != expected or current_qpos.shape != (ACTION_DIM,):
        raise ValueError("CARE candidate inputs differ from the native BiCoord contract")
    if candidate == 0:
        value = reference.copy()
    elif candidate == 1:
        value = base.copy()
    elif candidate == 2:
        value = np.concatenate((current_qpos[None], reference[:-1]), axis=0)
    elif candidate == 3:
        value = time_warp_plan(reference, current_qpos, 0.75)
    elif candidate == 4:
        value = time_warp_plan(reference, current_qpos, 1.25)
    elif candidate == 5:
        value = reference.copy()
        value[:, 6] = current_qpos[6]
    else:
        raise ValueError(f"unknown CARE candidate: {candidate}")
    if value.shape != expected or not np.isfinite(value).all():
        raise ValueError("CARE candidate is non-finite or shape-invalid")
    return value.astype(np.float32, copy=False)


def _candidate_set(context: Any, focal_arm: int) -> np.ndarray:
    rows = [
        candidate_plan(
            candidate,
            context.reference_plan[focal_arm],
            context.base_plan[focal_arm],
            context.current_qpos[focal_arm],
        )
        for candidate in range(CANDIDATES)
    ]
    result = np.stack(rows)
    if result.shape != (CANDIDATES, ACTION_HORIZON, ACTION_DIM):
        raise AssertionError("CARE candidate stack differs")
    return result


def _branch_seed(snapshot_id: str, repeat: int) -> int:
    payload = f"BiCoord|CARE|physical-branch-v1|{snapshot_id}|{repeat}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _seed_rng(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _success_progress(env: Any) -> tuple[bool, float]:
    try:
        checked = bool(env.check_success())
    except Exception:
        checked = False
    success = bool(getattr(env, "eval_success", False) or checked)
    progress = float(getattr(env, "stage_eval_score", float(success)))
    if not math.isfinite(progress):
        raise RuntimeError("BiCoord task progress is non-finite")
    return success, float(np.clip(progress, 0.0, 1.0))


def _qpos(observation: Mapping[str, Any]) -> np.ndarray:
    value = np.stack(
        [project_local_observation(observation, arm)["state"] for arm in (0, 1)]
    ).astype(np.float32)
    if value.shape != (2, ACTION_DIM) or not np.isfinite(value).all():
        raise RuntimeError("BiCoord physical observation has invalid qpos")
    return value


def _base_env(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    while hasattr(current, "env") and id(current) not in seen:
        seen.add(id(current))
        current = current.env
    return getattr(current, "unwrapped", current)


def _actor_name(actor: Any, index: int) -> str:
    getter = getattr(actor, "get_name", None)
    name = getter() if callable(getter) else getattr(actor, "name", "")
    return f"{index}:{name or ''}"


def _drop_baseline(env: Any) -> dict[str, float]:
    scene = getattr(_base_env(env), "scene", None)
    if scene is None or not callable(getattr(scene, "get_all_actors", None)):
        raise RuntimeError("BiCoord scene actors are unavailable for safety labels")
    excluded = ("ground", "table", "wall", "light", "camera", "target", "visual")
    result: dict[str, float] = {}
    for index, actor in enumerate(scene.get_all_actors()):
        name = _actor_name(actor, index)
        pose = actor.get_pose()
        z = float(np.asarray(pose.p, dtype=np.float64)[2])
        if math.isfinite(z) and z > 0.20 and not any(token in name.lower() for token in excluded):
            result[name] = z
    return result


def _contact_names(contact: Any) -> tuple[str, str] | None:
    bodies = getattr(contact, "bodies", None)
    if bodies is None or len(bodies) != 2:
        return None
    names = []
    for body in bodies:
        entity = getattr(body, "entity", body)
        names.append(str(getattr(entity, "name", "")).lower())
    return names[0], names[1]


def _contact_impulse(contact: Any) -> float:
    total = 0.0
    for point in getattr(contact, "points", ()):
        impulse = getattr(point, "impulse", None)
        if impulse is not None:
            total += float(np.linalg.norm(np.asarray(impulse, dtype=np.float64)))
    return total


def _physical_safety(env: Any, baseline: Mapping[str, float]) -> dict[str, Any]:
    scene = getattr(_base_env(env), "scene", None)
    if scene is None:
        raise RuntimeError("BiCoord scene is unavailable for safety labels")
    actors = {_actor_name(actor, index): actor for index, actor in enumerate(scene.get_all_actors())}
    dropped: list[str] = []
    for name, initial_z in baseline.items():
        actor = actors.get(name)
        if actor is None:
            raise RuntimeError(f"actor identity drift during branch: {name}")
        z = float(np.asarray(actor.get_pose().p, dtype=np.float64)[2])
        if not math.isfinite(z) or z < max(0.05, float(initial_z) - 0.35):
            dropped.append(name)
    robot_contacts: list[dict[str, Any]] = []
    getter = getattr(scene, "get_contacts", None)
    if not callable(getter):
        raise RuntimeError("BiCoord scene contacts are unavailable for safety labels")
    for contact in getter():
        names = _contact_names(contact)
        if names is None:
            continue
        left, right = names
        cross_arm = ("left" in left and "right" in right) or ("right" in left and "left" in right)
        impulse = _contact_impulse(contact)
        if cross_arm and impulse > 1e-4:
            robot_contacts.append({"bodies": [left, right], "impulse_l2_sum": impulse})
    return {
        "dropped_actors": dropped,
        "robot_robot_contacts": robot_contacts,
        "drop": bool(dropped),
        "robot_collision": bool(robot_contacts),
        "hard_safety_violation": bool(dropped or robot_contacts),
    }


def _step_metrics(
    env: Any,
    observation: Mapping[str, Any],
    previous_qpos: np.ndarray,
    actions: np.ndarray,
    baseline: Mapping[str, float],
    step: int,
) -> dict[str, Any]:
    current = _qpos(observation)
    success, progress = _success_progress(env)
    safety = _physical_safety(env, baseline)
    movement = np.linalg.norm(current[:, :6] - previous_qpos[:, :6], axis=1)
    if not np.isfinite(actions).all() or not np.isfinite(movement).all():
        safety["hard_safety_violation"] = True
    return {
        "branch_step": int(step),
        "progress": progress,
        "success": success,
        "hard_safety_violation": bool(safety["hard_safety_violation"]),
        "collision_or_drop": bool(safety["hard_safety_violation"]),
        "robot_conflict": bool(safety["robot_collision"]),
        "duplicate_work": False,
        "active": [bool(value >= 0.02) for value in movement],
        "all_joint_changes_below_0_02": bool(np.all(movement < 0.02)),
        "qpos": current.tolist(),
        "safety": safety,
    }


def _deadlock_mask(rows: Sequence[Mapping[str, Any]], start_progress: float) -> list[bool]:
    stagnant: list[bool] = []
    previous = float(start_progress)
    for row in rows:
        progress = float(row["progress"])
        stagnant.append(
            abs(progress - previous) <= 1e-4
            and bool(row["all_joint_changes_below_0_02"])
            and not bool(row["success"])
        )
        previous = progress
    result = [False] * len(rows)
    first = 0
    while first < len(rows):
        if not stagnant[first]:
            first += 1
            continue
        last = first
        while last < len(rows) and stagnant[last]:
            last += 1
        if last - first >= 8:
            result[first:last] = [True] * (last - first)
        first = last
    return result


def outcome_at_horizon(
    rows: Sequence[Mapping[str, Any]], start_progress: float, horizon: int
) -> dict[str, Any]:
    """Build the registered bounded physical utility vector."""

    observed = list(rows[: int(horizon)])
    if len(observed) != int(horizon):
        raise ValueError("physical CARE branch did not reach the requested horizon")
    deadlock = _deadlock_mask(observed, start_progress)
    active = np.asarray([row["active"] for row in observed], dtype=np.float64)
    active_fraction = active.mean(0)
    first_success = next(
        (index + 1 for index, row in enumerate(observed) if bool(row["success"])),
        None,
    )
    vector = np.asarray(
        (
            float(np.clip(float(observed[-1]["progress"]) - float(start_progress), -1, 1)),
            float(first_success is not None),
            -float(np.mean([row["collision_or_drop"] for row in observed])),
            -float(np.mean([row["robot_conflict"] for row in observed])),
            -float(np.mean([row["duplicate_work"] for row in observed])),
            -float(np.mean(deadlock)),
            -float(active_fraction.max() - active_fraction.min()),
            -float((first_success if first_success is not None else horizon) / horizon),
        ),
        dtype=np.float64,
    )
    utility = float(np.dot(ORDINARY_WEIGHTS, vector))
    return {
        "requested_steps": int(horizon),
        "observed_steps": len(observed),
        "bounded_utility_vector": vector.tolist(),
        "utility_main": utility,
        "hard_safety_violation": bool(any(row["hard_safety_violation"] for row in observed)),
        "first_success_step": first_success,
        "final_progress": float(observed[-1]["progress"]),
        "active_fraction": active_fraction.tolist(),
        "physical_simulator_outcome": True,
    }


def _restore_runtime_and_env(
    env: Any,
    runtime: Any,
    simulator_state: Mapping[str, Any],
    runtime_state: Mapping[str, Any],
    *,
    branch_seed: int | None = None,
) -> Mapping[str, Any]:
    restore_state(env, simulator_state)
    runtime.restore_state(runtime_state)
    if branch_seed is not None:
        _seed_rng(branch_seed)
    return env.get_obs()


def _reference_probe(
    env: Any,
    runtime: Any,
    task: str,
    simulator_state: Mapping[str, Any],
    runtime_state: Mapping[str, Any],
) -> dict[str, Any]:
    def rollout(restored_env: Any) -> Mapping[str, Any]:
        runtime.restore_state(runtime_state)
        observation = restored_env.get_obs()
        actions: list[np.ndarray] = []
        qpos: list[np.ndarray] = []
        progress: list[float] = []
        success: list[bool] = []
        for _ in range(2):
            context = runtime.act_with_context(observation, task, belief_enabled=True, commit=False)
            actual = context.reference_plan[:, 0].copy()
            runtime.record_executed_actions(actual)
            restored_env.take_action(actual.reshape(-1))
            observation = restored_env.get_obs()
            actions.append(actual)
            qpos.append(_qpos(observation))
            ok, score = _success_progress(restored_env)
            progress.append(score)
            success.append(ok)
        return {
            "actions": np.stack(actions),
            "qpos": np.stack(qpos),
            "progress": np.asarray(progress, dtype=np.float64),
            "success": np.asarray(success, dtype=bool),
        }

    report = restore_probe(
        env,
        simulator_state,
        rollout,
        repeats=2,
        tolerance=SNAPSHOT_TOLERANCE,
    )
    if report.get("passed") is not True:
        raise RuntimeError(f"BiCoord snapshot restore probe failed: {report}")
    _restore_runtime_and_env(env, runtime, simulator_state, runtime_state)
    return report


def _run_branch(
    *,
    env: Any,
    runtime: Any,
    task: str,
    focal_arm: int,
    candidate: int,
    candidate_chunks: np.ndarray,
    regime: str,
    repeat: int,
    snapshot_id: str,
    simulator_state: Mapping[str, Any],
    runtime_state: Mapping[str, Any],
    baseline: Mapping[str, float],
    peer_replay: Sequence[np.ndarray] | None,
    start_progress: float,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    if regime not in {"reactive", "replay"}:
        raise ValueError("CARE branch regime must be reactive or replay")
    seed = _branch_seed(snapshot_id, repeat)
    observation = _restore_runtime_and_env(
        env, runtime, simulator_state, runtime_state, branch_seed=seed
    )
    executed: list[np.ndarray] = []
    metrics: list[dict[str, Any]] = []
    replay_max_abs = 0.0
    for step in range(MAX_BRANCH_STEPS):
        previous_qpos = _qpos(observation)
        context = runtime.act_with_context(observation, task, belief_enabled=True, commit=False)
        actual = context.reference_plan[:, 0].copy()
        if step < INTERVENTION_STEPS:
            actual[focal_arm] = candidate_chunks[candidate, step]
        if regime == "replay":
            if peer_replay is None or step >= len(peer_replay):
                raise RuntimeError("CARE replay branch lacks candidate-0 peer support")
            peer = 1 - focal_arm
            replayed = np.asarray(peer_replay[step], dtype=np.float32)
            if replayed.shape != (ACTION_DIM,) or not np.isfinite(replayed).all():
                raise RuntimeError("CARE replay peer action is invalid")
            actual[peer] = replayed
            replay_max_abs = max(replay_max_abs, float(np.max(np.abs(actual[peer] - replayed))))
        if actual.shape != (2, ACTION_DIM) or not np.isfinite(actual).all():
            raise RuntimeError("CARE physical branch action is invalid")
        # Commit exactly the command sent to the native absolute controller.
        runtime.record_executed_actions(actual)
        env.take_action(actual.reshape(-1))
        observation = env.get_obs()
        executed.append(actual.copy())
        metrics.append(_step_metrics(env, observation, previous_qpos, actual, baseline, step))
    outcomes = {
        str(horizon): outcome_at_horizon(metrics, start_progress, horizon)
        for horizon in HORIZONS
    }
    row = {
        "candidate_id": int(candidate),
        "regime": regime,
        "repeat_id": int(repeat),
        "branch_seed": int(seed),
        "status": "VALID",
        "physical_simulator_outcome": True,
        "simulator_steps": MAX_BRANCH_STEPS,
        "intervention_steps": INTERVENTION_STEPS,
        "candidate_transform_clipped": False,
        "action_clipped": False,
        "peer_action_source": "reactive_policy" if regime == "reactive" else "candidate0_reactive_replay_log",
        "peer_policy_output_used": regime == "reactive",
        "focal_policy_output_used": True,
        "replay_peer_action_max_abs_error": replay_max_abs,
        "outcomes": outcomes,
        "executed_actions": [value.tolist() for value in executed],
        "metrics": metrics,
    }
    peer = 1 - focal_arm
    return row, [value[peer].copy() for value in executed]


def _targets(branches: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keyed = {
        (int(row["candidate_id"]), str(row["regime"]), int(row["repeat_id"])): row
        for row in branches
    }
    expected = {
        (candidate, regime, repeat)
        for candidate in range(CANDIDATES)
        for regime in ("reactive", "replay")
        for repeat in (0, 1)
    }
    if set(keyed) != expected or len(branches) != BRANCHES_PER_FAMILY:
        raise RuntimeError("CARE family does not contain 24 unique physical branches")
    targets = np.zeros((len(HORIZONS), CANDIDATES, 2, 3), dtype=np.float32)
    safety = np.zeros((len(HORIZONS), CANDIDATES, 2), dtype=np.float32)
    usable = np.ones(len(HORIZONS), dtype=bool)
    for hi, horizon in enumerate(HORIZONS):
        key = str(horizon)
        for repeat in (0, 1):
            u0_reactive = float(keyed[(0, "reactive", repeat)]["outcomes"][key]["utility_main"])
            u0_replay = float(keyed[(0, "replay", repeat)]["outcomes"][key]["utility_main"])
            for candidate in range(CANDIDATES):
                reactive = keyed[(candidate, "reactive", repeat)]["outcomes"][key]
                replay = keyed[(candidate, "replay", repeat)]["outcomes"][key]
                direct = float(replay["utility_main"]) - u0_replay
                total = float(reactive["utility_main"]) - u0_reactive
                targets[hi, candidate, repeat] = (direct, total - direct, total)
                safety[hi, candidate, repeat] = float(reactive["hard_safety_violation"])
    if not np.array_equal(targets[:, 0], np.zeros_like(targets[:, 0])):
        raise AssertionError("CARE reference advantage is not exactly zero")
    return targets, safety, usable


def _family_spec(task: str, local: int, families_per_task: int, seeds: Sequence[int]) -> dict[str, int]:
    if len(seeds) < 1:
        raise ValueError("CARE family construction requires expert-valid seeds")
    # Frozen before any branch outcome: shallow anchors avoid terminal states,
    # while expert-valid seeds and both focal arms cover scene variation.
    task_index = TASKS.index(task)
    anchor = 2 + ((local * 7 + task_index * 3) % 24)
    if anchor + MAX_BRANCH_STEPS >= VALIDATION_MAX_STEPS[task]:
        raise AssertionError("CARE anchor exceeds the task-specific horizon")
    return {
        "family_id": task_index * families_per_task + local,
        "seed": int(seeds[local % len(seeds)]),
        "anchor_step": anchor,
        "focal_arm": local % 2,
    }


def _collect_family(
    *,
    args: argparse.Namespace,
    runtime: Any,
    checkpoint: Path,
    checkpoint_sha256: str,
    normalization: Mapping[str, Any],
    task: str,
    spec: Mapping[str, int],
    output_root: Path,
    smoke: bool | None = None,
) -> dict[str, Any]:
    family_id = int(spec["family_id"])
    if smoke is None:
        smoke = "branch_smoke" in output_root.parts
    seed = int(spec["seed"])
    anchor_step = int(spec["anchor_step"])
    focal_arm = int(spec["focal_arm"])
    snapshot_id = f"{task}:seed={seed}:anchor={anchor_step}:arm={focal_arm}:family={family_id}"
    npz_path = output_root / f"family_{family_id:06d}.npz"
    json_path = output_root / f"family_{family_id:06d}.json"
    if npz_path.exists() or json_path.exists():
        if not (npz_path.is_file() and json_path.is_file()):
            raise RuntimeError(f"partial CARE family exists: {snapshot_id}")
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "PASSED"
            and existing.get("snapshot_id") == snapshot_id
            and existing.get("checkpoint_sha256") == checkpoint_sha256
            and existing.get("physical_simulator_outcomes") is True
            and existing.get("npz_sha256") == sha256_file(npz_path)
        ):
            return {
                "family": {
                    "family_id": existing.get("family_id"),
                    "snapshot_id": existing.get("snapshot_id"),
                    "task": existing.get("task"),
                    "status": existing.get("status"),
                },
                "npz": str(npz_path.resolve()),
                "npz_sha256": sha256_file(npz_path),
                "manifest": str(json_path.resolve()),
                "manifest_sha256": sha256_file(json_path),
                "reused": True,
            }
        raise RuntimeError(f"refusing to overwrite inconsistent CARE family: {snapshot_id}")

    env = _make_env(args.benchmark_repo, task, seed)
    started = time.perf_counter()
    try:
        runtime.reset(task)
        observation = env.get_obs()
        for step in range(anchor_step):
            context = runtime.act_with_context(observation, task, belief_enabled=True, commit=False)
            action = context.reference_plan[:, 0].copy()
            runtime.record_executed_actions(action)
            env.take_action(action.reshape(-1))
            observation = env.get_obs()
            success, _progress = _success_progress(env)
            if success:
                raise RuntimeError(f"B-core reached a terminal state before frozen anchor {snapshot_id}")
        _success, start_progress = _success_progress(env)
        simulator_state = capture_state(env)
        runtime_state = runtime.snapshot_state()
        simulator_state_hash = state_sha256(simulator_state)
        runtime_state_hash = state_sha256(runtime_state)
        probe = _reference_probe(env, runtime, task, simulator_state, runtime_state)
        observation = _restore_runtime_and_env(env, runtime, simulator_state, runtime_state)
        preview = runtime.act_with_context(observation, task, belief_enabled=True, commit=False)
        if preview.memory.shape != (2, BICOORD_CARE_MEMORY_TOKENS, BICOORD_CARE_MEMORY_WIDTH):
            raise RuntimeError("B-core exposed decoded hidden instead of belief/event CARE memory")
        if preview.memory_mask.shape != (2, BICOORD_CARE_MEMORY_TOKENS):
            raise RuntimeError("B-core CARE memory mask differs")
        memory = preview.memory[focal_arm].astype(np.float32, copy=True)
        memory_mask = preview.memory_mask[focal_arm].astype(bool, copy=True)
        candidates = _candidate_set(preview, focal_arm)
        action_min = np.asarray(normalization["action_min"], dtype=np.float32)
        action_max = np.asarray(normalization["action_max"], dtype=np.float32)
        out_of_source_range = int(
            np.count_nonzero((candidates < action_min[None, None]) | (candidates > action_max[None, None]))
        )
        baseline = _drop_baseline(env)
        branches: list[dict[str, Any]] = []
        fidelity: list[dict[str, Any]] = []
        for repeat in (0, 1):
            reference_reactive, peer_log = _run_branch(
                env=env,
                runtime=runtime,
                task=task,
                focal_arm=focal_arm,
                candidate=0,
                candidate_chunks=candidates,
                regime="reactive",
                repeat=repeat,
                snapshot_id=snapshot_id,
                simulator_state=simulator_state,
                runtime_state=runtime_state,
                baseline=baseline,
                peer_replay=None,
                start_progress=start_progress,
            )
            branches.append(reference_reactive)
            reference_replay, _ = _run_branch(
                env=env,
                runtime=runtime,
                task=task,
                focal_arm=focal_arm,
                candidate=0,
                candidate_chunks=candidates,
                regime="replay",
                repeat=repeat,
                snapshot_id=snapshot_id,
                simulator_state=simulator_state,
                runtime_state=runtime_state,
                baseline=baseline,
                peer_replay=peer_log,
                start_progress=start_progress,
            )
            branches.append(reference_replay)
            difference = max(
                abs(
                    float(reference_reactive["outcomes"][str(horizon)]["utility_main"])
                    - float(reference_replay["outcomes"][str(horizon)]["utility_main"])
                )
                for horizon in HORIZONS
            )
            fidelity.append({"repeat_id": repeat, "utility_max_abs_error": difference})
            if difference > SNAPSHOT_TOLERANCE:
                raise RuntimeError(f"candidate-0 reactive/replay fidelity failed: {difference}")
            for candidate in range(1, CANDIDATES):
                reactive, _ = _run_branch(
                    env=env,
                    runtime=runtime,
                    task=task,
                    focal_arm=focal_arm,
                    candidate=candidate,
                    candidate_chunks=candidates,
                    regime="reactive",
                    repeat=repeat,
                    snapshot_id=snapshot_id,
                    simulator_state=simulator_state,
                    runtime_state=runtime_state,
                    baseline=baseline,
                    peer_replay=None,
                    start_progress=start_progress,
                )
                branches.append(reactive)
            for candidate in range(1, CANDIDATES):
                replay, _ = _run_branch(
                    env=env,
                    runtime=runtime,
                    task=task,
                    focal_arm=focal_arm,
                    candidate=candidate,
                    candidate_chunks=candidates,
                    regime="replay",
                    repeat=repeat,
                    snapshot_id=snapshot_id,
                    simulator_state=simulator_state,
                    runtime_state=runtime_state,
                    baseline=baseline,
                    peer_replay=peer_log,
                    start_progress=start_progress,
                )
                branches.append(replay)
        targets, hard_safety, usable = _targets(branches)
        _atomic_npz(
            npz_path,
            memory=memory,
            memory_mask=memory_mask,
            candidates=candidates,
            targets=targets,
            hard_safety=hard_safety,
            usable=usable,
            task_id=np.asarray(TASKS.index(task), dtype=np.int64),
            snapshot_id=np.asarray(snapshot_id),
        )
        family = {
            "schema": BRANCH_SCHEMA,
            "status": "PASSED",
            "family_id": family_id,
            "snapshot_id": snapshot_id,
            "task": task,
            "task_id": TASKS.index(task),
            "seed": seed,
            "seed_source": "policy_independent_expert_valid_manifest",
            "anchor_step": anchor_step,
            "focal_arm": focal_arm,
            "provider_policy": "B-core/TUNE",
            "smoke": bool(smoke),
            "provider_policy_family": "PredictiveTeamBeliefPolicy",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "branches_per_family": BRANCHES_PER_FAMILY,
            "candidate_count": CANDIDATES,
            "regimes": ["reactive", "replay"],
            "repeats": 2,
            "horizons": list(HORIZONS),
            "future_offsets_steps": list(BICOORD_FUTURE_OFFSETS_STEPS),
            "intervention_steps": INTERVENTION_STEPS,
            "physical_simulator_outcomes": True,
            "offline_demonstration_error_used": False,
            "pseudo_labels_used": False,
            "strict_lag_one": True,
            "care_memory_tokens": BICOORD_CARE_MEMORY_TOKENS,
            "care_memory_width": BICOORD_CARE_MEMORY_WIDTH,
            "care_memory_semantics": BICOORD_CARE_MEMORY_SEMANTICS,
            "simulator_state_sha256": simulator_state_hash,
            "runtime_state_sha256": runtime_state_hash,
            "restore_probe": probe,
            "reference_reactive_replay_fidelity": fidelity,
            "action_clipping": False,
            "candidate_transform_clipping": False,
            "candidate_values_outside_source_population_range": out_of_source_range,
            "normalization_receipt_sha256": normalization["sha256"],
            "branches": branches,
            "wall_seconds": time.perf_counter() - started,
            "npz": str(npz_path.resolve()),
            "npz_sha256": sha256_file(npz_path),
        }
        atomic_json(json_path, family)
        return {
            "family": {
                "family_id": family.get("family_id"),
                "snapshot_id": family.get("snapshot_id"),
                "task": family.get("task"),
                "status": family.get("status"),
            },
            "npz": str(npz_path.resolve()),
            "npz_sha256": sha256_file(npz_path),
            "manifest": str(json_path.resolve()),
            "manifest_sha256": sha256_file(json_path),
            "reused": False,
        }
    finally:
        _close_env(env)


def run(args: argparse.Namespace) -> dict[str, Any]:
    assert_common_paths(args, need_dataset=True)
    smoke = args.operation == "smoke"
    explicit_smoke = bool(getattr(args, "smoke", False))
    if explicit_smoke and not smoke:
        raise ValueError("formal branch collection cannot be marked smoke")
    require_stage_result(
        args.run,
        "bcore_smoke_closed_loop" if smoke else "bcore_validation20",
        config_sha256=args.config_sha256,
    )
    expected_families = 1 if smoke else 30
    requested_families = getattr(args, "families_per_task", None)
    families_per_task = expected_families if requested_families is None else int(requested_families)
    branches_per_family = int(getattr(args, "branches_per_family", BRANCHES_PER_FAMILY))
    if families_per_task != expected_families or branches_per_family != BRANCHES_PER_FAMILY:
        raise ValueError(
            "BiCoord CARE branch protocol is frozen to "
            f"{expected_families} families/task and 24 physical branches/family"
        )
    rank = int(getattr(args, "rank", 0))
    world = int(getattr(args, "world_size", 1))
    if not 0 <= rank < world or world not in {1, 4}:
        raise ValueError("BiCoord physical branch shard must use one or four workers")
    checkpoint = _checkpoint(args)
    checkpoint_sha = sha256_file(checkpoint)
    runtime = _runtime(checkpoint, args)
    normalization = _normalization_audit(args, runtime)
    seed_count = 1 if smoke else 20
    seed_rows = {task: _official_seeds(args, task, count=seed_count) for task in TASKS}
    output_root = args.run / "artifacts" / ("branch_smoke" if smoke else "branches") / f"rank_{rank}"
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for task in TASKS:
        for local in range(families_per_task):
            spec = _family_spec(task, local, families_per_task, seed_rows[task])
            if int(spec["family_id"]) % world != rank:
                continue
            record = _collect_family(
                args=args,
                runtime=runtime,
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha,
                normalization=normalization,
                task=task,
                spec=spec,
                output_root=output_root,
                smoke=smoke,
            )
            records.append(record)
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "family_id": spec["family_id"],
                        "task": task,
                        "completed": len(records),
                        "physical_simulator_outcomes": True,
                        "reused": record["reused"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    expected = sum(
        1
        for task in TASKS
        for local in range(families_per_task)
        if int(_family_spec(task, local, families_per_task, seed_rows[task])["family_id"]) % world == rank
    )
    if len(records) != expected:
        raise RuntimeError(f"physical branch shard coverage differs: {len(records)} != {expected}")
    manifest = output_root / "manifest.json"
    atomic_json(
        manifest,
        {
            "schema": SHARD_SCHEMA,
            "status": "PASSED",
            "rank": rank,
            "world_size": world,
            "families_per_task": families_per_task,
            "families": len(records),
            "smoke": smoke,
            "branches_per_family": BRANCHES_PER_FAMILY,
            "provider_policy": "B-core/TUNE",
            "provider_policy_family": "PredictiveTeamBeliefPolicy",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "physical_simulator_outcomes": True,
            "offline_demonstration_error_used": False,
            "care_memory_tokens": BICOORD_CARE_MEMORY_TOKENS,
            "care_memory_width": BICOORD_CARE_MEMORY_WIDTH,
            "care_memory_semantics": BICOORD_CARE_MEMORY_SEMANTICS,
            "normalization": normalization,
            "records": records,
        },
    )
    stage = "branch_smoke" if smoke else "branch_collection"
    return publish_result(
        args,
        stage=stage,
        include_model_contract=True,
        artifacts=[artifact(manifest, kind="branch_manifest")],
        rank=rank,
        world_size=world,
        families=len(records),
        branches_per_family=BRANCHES_PER_FAMILY,
        provider_policy="B-core/TUNE",
        physical_simulator_outcomes=True,
        offline_demonstration_error_used=False,
        restore_tolerance=SNAPSHOT_TOLERANCE,
        care_memory_tokens=BICOORD_CARE_MEMORY_TOKENS,
        care_memory_width=BICOORD_CARE_MEMORY_WIDTH,
        care_memory_semantics=BICOORD_CARE_MEMORY_SEMANTICS,
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=checkpoint_sha,
    )


def main(argv: list[str] | None = None) -> int:
    parser = common_parser(__doc__, ("smoke", "formal"))
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--families-per-task", type=int)
    parser.add_argument("--branches-per-family", type=int, default=BRANCHES_PER_FAMILY)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BRANCHES_PER_FAMILY",
    "BRANCH_SCHEMA",
    "HORIZONS",
    "candidate_plan",
    "outcome_at_horizon",
    "time_warp_plan",
]
