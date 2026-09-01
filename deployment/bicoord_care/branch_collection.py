"""Collect physical same-snapshot CARE branch families on BiCoord.

Every label in this module comes from a SAPIEN rollout.  A family freezes the
reference action prefix and B-core local-history state, deterministically
rebuilds a fresh official seeded simulator for every sibling, and then
executes the registered ``6 candidates x 2 response regimes x 2 repeats``
design.  Offline demonstration error is deliberately absent:
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
    restore_state,
    state_sha256,
)
from .branch_fidelity import FIDELITY_SCHEMA, FIDELITY_STEPS, FIDELITY_TOLERANCE
from .config import (
    ACTION_DIM,
    ACTION_HORIZON,
    BRANCH_SEED_BUCKET,
    BRANCH_SEEDS_PER_TASK,
    TASKS,
    VALIDATION_MAX_STEPS,
)
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


BRANCH_SCHEMA = "before-we-act.bicoord.care-physical-branch-family/2"
SHARD_SCHEMA = "before-we-act.bicoord.care-physical-branch-shard/1"
HORIZONS = (8, 16, 32, 64)
CANDIDATES = 6
BRANCHES_PER_FAMILY = CANDIDATES * 2 * 2
INTERVENTION_STEPS = 1
MAX_BRANCH_STEPS = max(HORIZONS)
if MAX_BRANCH_STEPS != FIDELITY_STEPS:
    raise RuntimeError("BiCoord fidelity receipt horizon differs from branch protocol")

# These fields are discrete parts of the upstream CARE physical-branch
# contract.  Utility is deliberately *not* a substitute for checking them:
# weighted/thresholded utility can remain identical while a simulator or
# wrapper has silently changed a success, activity, or safety label.
_FIDELITY_DISCRETE_KEYS = (
    "branch_step",
    "success",
    "active",
    "all_joint_changes_below_0_02",
    "hard_safety_violation",
    "collision_or_drop",
    "robot_conflict",
    "duplicate_work",
)
_FIDELITY_SAFETY_KEYS = (
    "drop",
    "robot_collision",
    "hard_safety_violation",
    "dropped_actor_names",
    "robot_contact_bodies",
)
_FIDELITY_OUTCOME_KEYS = (
    "requested_steps",
    "observed_steps",
    "hard_safety_violation",
    "first_success_step",
    "physical_simulator_outcome",
)
_FIDELITY_BRANCH_EXPECTED = {
    "candidate_id": 0,
    "status": "VALID",
    "physical_simulator_outcome": True,
    "simulator_steps": MAX_BRANCH_STEPS,
    "intervention_steps": INTERVENTION_STEPS,
    "candidate_transform_clipped": False,
    "action_clipped": False,
    "focal_policy_output_used": True,
}


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
    # Match the upstream CARE/BiCoord utility contract: ``active`` is based
    # on the command that was sent at this control tick versus the qpos at the
    # beginning of the tick.  Using the post-step qpos here is subtly wrong:
    # it folds simulator integration/rendering noise into a discrete 0.02
    # threshold and can make an identical reactive/replay command sequence
    # receive different utility labels.  The native upstream implementation
    # measures the six physical joint coordinates for this gate; the seventh
    # coordinate is the continuous gripper drive target and is not folded into
    # the inactivity threshold or otherwise reparameterized.
    command = np.asarray(actions, dtype=np.float32)
    if command.shape != (2, ACTION_DIM) or not np.isfinite(command).all():
        raise RuntimeError("BiCoord physical action metrics require finite [2,7] actions")
    movement = np.linalg.norm(command[:, :6] - previous_qpos[:, :6], axis=1)
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
    # The first observed tick has no previous *row* to compare against.  The
    # upstream CARE collector therefore starts with ``None`` rather than the
    # snapshot's scalar progress; otherwise a zero-progress first tick is
    # incorrectly counted toward the eight-step deadlock run.
    previous: float | None = None
    for row in rows:
        progress = float(row["progress"])
        stagnant.append(
            previous is not None
            and abs(progress - previous) <= 1e-4
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
    # SAPIEN's public PhysX pack/unpack omits the contact solver warm-start
    # manifold.  On contact-rich BiCoord scenes that hidden state changes the
    # very next integration even though every exposed pose/qpos/cache matches.
    # A branch-capable environment may therefore provide a deterministic
    # seed+prefix rebuild hook.  It creates a fresh official environment and
    # replays the frozen B-core prefix, restoring the complete solver history
    # without pretending that the opaque PhysX cache was serialized.
    rebuild = getattr(env, "_bicoord_rebuild_at_anchor", None)
    if callable(rebuild):
        rebuild()
    else:
        restore_state(env, simulator_state)
    runtime.restore_state(runtime_state)
    if branch_seed is not None:
        _seed_rng(branch_seed)
    return env.get_obs()


class _RebuildableBranchEnv:
    """Forward an environment while replacing it for every branch restore."""

    def __init__(self, factory: Any) -> None:
        self._factory = factory
        self._current: Any | None = None

    def _bicoord_rebuild_at_anchor(self) -> Any:
        if self._current is not None:
            _close_env(self._current)
        self._current = self._factory()
        return self._current

    def close(self) -> None:
        if self._current is not None:
            _close_env(self._current)
            self._current = None

    def close_env(self) -> None:
        self.close()

    def __getattr__(self, name: str) -> Any:
        if self._current is None:
            raise RuntimeError("branch environment has not been rebuilt")
        return getattr(self._current, name)


def _replay_anchor_environment(
    args: argparse.Namespace,
    task: str,
    seed: int,
    prefix_actions: Sequence[np.ndarray],
) -> Any:
    """Construct the official task and replay its reference prefix exactly."""

    # setup_demo and task assets consume process RNG in a few released task
    # modules.  Seed before construction so every rebuild starts from the same
    # official stream, then replay the exact native controller commands.
    _seed_rng(seed)
    env = _make_env(args.benchmark_repo, task, seed)
    try:
        observation = env.get_obs()
        for action in prefix_actions:
            value = np.asarray(action, dtype=np.float32)
            if value.shape != (2, ACTION_DIM) or not np.isfinite(value).all():
                raise RuntimeError("frozen B-core prefix action is invalid")
            env.take_action(value.reshape(-1))
            observation = env.get_obs()
        return env
    except BaseException:
        _close_env(env)
        raise


def _replay_reference_probe(
    *,
    args: argparse.Namespace,
    task: str,
    seed: int,
    prefix_actions: Sequence[np.ndarray],
    runtime: Any,
    runtime_state: Mapping[str, Any],
    expected_anchor_state_sha256: str,
) -> dict[str, Any]:
    """Prove deterministic post-anchor rollouts using fresh official envs."""

    rows: list[Mapping[str, Any]] = []
    anchor_hashes: list[str] = []
    for _ in range(2):
        env = _replay_anchor_environment(args, task, seed, prefix_actions)
        try:
            anchor_hashes.append(state_sha256(capture_state(env)))
            runtime.restore_state(runtime_state)
            observation = env.get_obs()
            actions: list[np.ndarray] = []
            qpos: list[np.ndarray] = []
            progress: list[float] = []
            success: list[bool] = []
            for _step in range(2):
                context = runtime.act_with_context(observation, task, belief_enabled=True, commit=False)
                actual = context.reference_plan[:, 0].copy()
                runtime.record_executed_actions(actual)
                env.take_action(actual.reshape(-1))
                observation = env.get_obs()
                actions.append(actual)
                qpos.append(_qpos(observation))
                ok, score = _success_progress(env)
                progress.append(score)
                success.append(ok)
            rows.append({
                "actions": np.stack(actions),
                "qpos": np.stack(qpos),
                "progress": np.asarray(progress, dtype=np.float64),
                "success": np.asarray(success, dtype=bool),
            })
        finally:
            _close_env(env)
    from .bicoord_snapshot import max_abs

    error = max((max_abs(rows[0], row) for row in rows[1:]), default=math.inf)
    anchor_match = bool(
        len(anchor_hashes) == 2
        and all(value == expected_anchor_state_sha256 for value in anchor_hashes)
    )
    return {
        "schema": "before-we-act.bicoord.seed-replay-probe/1",
        "repeats": 2,
        "max_abs_error": float(error),
        "tolerance": SNAPSHOT_TOLERANCE,
        "passed": bool(error <= SNAPSHOT_TOLERANCE and anchor_match),
        "restore_mode": "official_seed_plus_reference_prefix_replay",
        "expected_anchor_state_sha256": expected_anchor_state_sha256,
        "rebuilt_anchor_state_sha256": anchor_hashes,
        "rebuilt_anchor_state_exact_match": anchor_match,
    }


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
    env_factory: Any | None = None,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    if regime not in {"reactive", "replay"}:
        raise ValueError("CARE branch regime must be reactive or replay")
    seed = _branch_seed(snapshot_id, repeat)
    branch_env = env_factory() if env_factory is not None else env
    try:
        observation = _restore_runtime_and_env(
            branch_env, runtime, simulator_state, runtime_state, branch_seed=seed
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
            branch_env.take_action(actual.reshape(-1))
            observation = branch_env.get_obs()
            executed.append(actual.copy())
            metrics.append(_step_metrics(branch_env, observation, previous_qpos, actual, baseline, step))
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
    finally:
        if env_factory is not None:
            _close_env(branch_env)


def _fidelity_diagnostic(
    reactive: Mapping[str, Any], replay: Mapping[str, Any]
) -> dict[str, Any]:
    """Summarize continuous and discrete differences before a fidelity fail.

    This is failure evidence only; it is never consumed as CARE supervision.
    Keeping both levels is important because the utility vector contains
    thresholded labels (active/deadlock) which can amplify a tiny physical or
    policy numerical difference into a visible utility delta.
    """

    def _array(row: Mapping[str, Any], key: str) -> np.ndarray:
        return np.asarray(row.get(key, []), dtype=np.float64)

    reactive_actions = _array(reactive, "executed_actions")
    replay_actions = _array(replay, "executed_actions")
    reactive_metrics = reactive.get("metrics", [])
    replay_metrics = replay.get("metrics", [])
    if not isinstance(reactive_metrics, (list, tuple)):
        reactive_metrics = []
    if not isinstance(replay_metrics, (list, tuple)):
        replay_metrics = []
    reactive_qpos = np.asarray(
        [item.get("qpos", []) if isinstance(item, Mapping) else [] for item in reactive_metrics],
        dtype=np.float64,
    )
    replay_qpos = np.asarray(
        [item.get("qpos", []) if isinstance(item, Mapping) else [] for item in replay_metrics],
        dtype=np.float64,
    )
    reactive_progress = np.asarray(
        [item.get("progress", np.nan) if isinstance(item, Mapping) else np.nan for item in reactive_metrics],
        dtype=np.float64,
    )
    replay_progress = np.asarray(
        [item.get("progress", np.nan) if isinstance(item, Mapping) else np.nan for item in replay_metrics],
        dtype=np.float64,
    )

    def _max_abs(first: np.ndarray, second: np.ndarray) -> float:
        if first.shape != second.shape or not first.size:
            return float("inf") if first.shape != second.shape else 0.0
        return float(np.max(np.abs(first - second)))

    # Keep the old, named fields for consumers that already inspect them, but
    # also publish one complete discrete-label audit.  ``zip`` alone is not
    # sufficient here: a truncated trace must fail closed instead of looking
    # equal over its common prefix.
    active_diff: list[int] = []
    stagnant_diff: list[int] = []
    success_diff: list[int] = []
    discrete_diff: dict[str, list[int]] = {
        key: [] for key in _FIDELITY_DISCRETE_KEYS
    }
    safety_diff: dict[str, list[int]] = {key: [] for key in _FIDELITY_SAFETY_KEYS}
    metric_length_equal = len(reactive_metrics) == len(replay_metrics)
    metric_count = max(len(reactive_metrics), len(replay_metrics))
    missing = object()
    for index in range(metric_count):
        left = reactive_metrics[index] if index < len(reactive_metrics) else missing
        right = replay_metrics[index] if index < len(replay_metrics) else missing
        if left is missing or right is missing:
            for key in _FIDELITY_DISCRETE_KEYS:
                discrete_diff[key].append(index)
            for key in _FIDELITY_SAFETY_KEYS:
                safety_diff[key].append(index)
            active_diff.append(index)
            stagnant_diff.append(index)
            success_diff.append(index)
            continue
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            for key in _FIDELITY_DISCRETE_KEYS:
                discrete_diff[key].append(index)
            for key in _FIDELITY_SAFETY_KEYS:
                safety_diff[key].append(index)
            active_diff.append(index)
            stagnant_diff.append(index)
            success_diff.append(index)
            continue
        for key in _FIDELITY_DISCRETE_KEYS:
            if key not in left or key not in right or left.get(key) != right.get(key):
                discrete_diff[key].append(index)
        if (
            "active" not in left
            or "active" not in right
            or left.get("active") != right.get("active")
        ):
            active_diff.append(index)
        if (
            "all_joint_changes_below_0_02" not in left
            or "all_joint_changes_below_0_02" not in right
            or left.get("all_joint_changes_below_0_02")
            != right.get("all_joint_changes_below_0_02")
        ):
            stagnant_diff.append(index)
        if (
            "success" not in left
            or "success" not in right
            or left.get("success") != right.get("success")
        ):
            success_diff.append(index)
        left_safety = left.get("safety", {})
        right_safety = right.get("safety", {})
        if not isinstance(left_safety, Mapping):
            left_safety = {}
        if not isinstance(right_safety, Mapping):
            right_safety = {}
        # Contact impulse magnitudes are continuous and are covered by the
        # physical-state probe.  For the branch receipt, compare the discrete
        # contact/drop identities as well as their boolean safety labels.
        left_contacts = tuple(
            sorted(
                tuple(sorted(str(body) for body in contact.get("bodies", ())))
                for contact in left_safety.get("robot_robot_contacts", ())
                if isinstance(contact, Mapping)
            )
        )
        right_contacts = tuple(
            sorted(
                tuple(sorted(str(body) for body in contact.get("bodies", ())))
                for contact in right_safety.get("robot_robot_contacts", ())
                if isinstance(contact, Mapping)
            )
        )
        safety_values = {
            "drop": left_safety.get("drop") == right_safety.get("drop")
            if "drop" in left_safety and "drop" in right_safety
            else False,
            "robot_collision": left_safety.get("robot_collision")
            == right_safety.get("robot_collision")
            if "robot_collision" in left_safety and "robot_collision" in right_safety
            else False,
            "hard_safety_violation": left_safety.get("hard_safety_violation")
            == right_safety.get("hard_safety_violation")
            if "hard_safety_violation" in left_safety
            and "hard_safety_violation" in right_safety
            else False,
            "dropped_actor_names": tuple(
                sorted(str(name) for name in left_safety.get("dropped_actors", ()))
            )
            == tuple(sorted(str(name) for name in right_safety.get("dropped_actors", ())))
            if "dropped_actors" in left_safety and "dropped_actors" in right_safety
            else False,
            "robot_contact_bodies": left_contacts == right_contacts
            if "robot_robot_contacts" in left_safety
            and "robot_robot_contacts" in right_safety
            else False,
        }
        for key, equal in safety_values.items():
            if not equal:
                safety_diff[key].append(index)
    utility_diff: dict[str, dict[str, Any]] = {}
    outcome_discrete_diff: dict[str, list[str]] = {}
    for horizon in HORIZONS:
        key = str(horizon)
        reactive_outcomes = reactive.get("outcomes", {})
        replay_outcomes = replay.get("outcomes", {})
        if not isinstance(reactive_outcomes, Mapping):
            reactive_outcomes = {}
        if not isinstance(replay_outcomes, Mapping):
            replay_outcomes = {}
        left = reactive_outcomes.get(key, {})
        right = replay_outcomes.get(key, {})
        if not isinstance(left, Mapping):
            left = {}
        if not isinstance(right, Mapping):
            right = {}
        left_vector = np.asarray(left.get("bounded_utility_vector", []), dtype=np.float64)
        right_vector = np.asarray(right.get("bounded_utility_vector", []), dtype=np.float64)
        utility_diff[key] = {
            "utility_reactive": float(left.get("utility_main", np.nan)),
            "utility_replay": float(right.get("utility_main", np.nan)),
            "utility_abs_error": abs(
                float(left.get("utility_main", np.nan))
                - float(right.get("utility_main", np.nan))
            ),
            "bounded_vector_abs_error": _max_abs(left_vector, right_vector),
            "bounded_vector_reactive": left_vector.tolist(),
            "bounded_vector_replay": right_vector.tolist(),
        }
        outcome_discrete_diff[key] = [
            field
            for field in _FIDELITY_OUTCOME_KEYS
            if field not in left or field not in right or left.get(field) != right.get(field)
        ]
    action_per_step = (
        np.max(np.abs(reactive_actions - replay_actions), axis=(1, 2))
        if reactive_actions.shape == replay_actions.shape
        and reactive_actions.ndim == 3
        else np.asarray([], dtype=np.float64)
    )
    qpos_per_step = (
        np.max(np.abs(reactive_qpos - replay_qpos), axis=(1, 2))
        if reactive_qpos.shape == replay_qpos.shape and reactive_qpos.ndim == 3
        else np.asarray([], dtype=np.float64)
    )
    progress_per_step = (
        np.abs(reactive_progress - replay_progress)
        if reactive_progress.shape == replay_progress.shape
        and reactive_progress.ndim == 1
        else np.asarray([], dtype=np.float64)
    )
    action_shape_equal = reactive_actions.shape == replay_actions.shape
    qpos_shape_equal = reactive_qpos.shape == replay_qpos.shape
    trajectory_complete = bool(
        metric_length_equal
        and len(reactive_metrics) == MAX_BRANCH_STEPS
        and len(replay_metrics) == MAX_BRANCH_STEPS
        and reactive_actions.shape == (MAX_BRANCH_STEPS, 2, ACTION_DIM)
        and replay_actions.shape == (MAX_BRANCH_STEPS, 2, ACTION_DIM)
        and reactive_qpos.shape == (MAX_BRANCH_STEPS, 2, ACTION_DIM)
        and replay_qpos.shape == (MAX_BRANCH_STEPS, 2, ACTION_DIM)
        and reactive_progress.shape == (MAX_BRANCH_STEPS,)
        and replay_progress.shape == (MAX_BRANCH_STEPS,)
    )
    action_first_difference: int | None = None
    if (
        action_shape_equal
        and reactive_actions.ndim == 3
        and reactive_actions.shape[0] == replay_actions.shape[0]
    ):
        for index, (left, right) in enumerate(
            zip(reactive_actions, replay_actions)
        ):
            if not np.array_equal(left, right):
                action_first_difference = index
                break
    branch_contract_difference_fields = [
        key
        for key, expected in _FIDELITY_BRANCH_EXPECTED.items()
        if reactive.get(key) != expected or replay.get(key) != expected
    ]
    if (
        reactive.get("repeat_id") not in (0, 1)
        or reactive.get("repeat_id") != replay.get("repeat_id")
    ):
        branch_contract_difference_fields.append("repeat_id")
    if (
        not isinstance(reactive.get("branch_seed"), int)
        or reactive.get("branch_seed") != replay.get("branch_seed")
    ):
        branch_contract_difference_fields.append("branch_seed")
    return {
        "schema": "before-we-act.bicoord.care-fidelity-diagnostic/1",
        "reactive_status": reactive.get("status"),
        "replay_status": replay.get("status"),
        "executed_action_shape": list(reactive_actions.shape),
        "replay_executed_action_shape": list(replay_actions.shape),
        "executed_action_shape_equal": action_shape_equal,
        "executed_action_max_abs_error": _max_abs(reactive_actions, replay_actions),
        "executed_action_max_abs_error_by_step": action_per_step.tolist(),
        "executed_action_first_difference": action_first_difference,
        "qpos_shape": list(reactive_qpos.shape),
        "replay_qpos_shape": list(replay_qpos.shape),
        "qpos_shape_equal": qpos_shape_equal,
        "qpos_max_abs_error": _max_abs(reactive_qpos, replay_qpos),
        "qpos_max_abs_error_by_step": qpos_per_step.tolist(),
        "progress_max_abs_error": _max_abs(reactive_progress, replay_progress),
        "progress_max_abs_error_by_step": progress_per_step.tolist(),
        "metric_length_equal": metric_length_equal,
        "reactive_metric_count": len(reactive_metrics),
        "replay_metric_count": len(replay_metrics),
        "trajectory_complete": trajectory_complete,
        "branch_contract_difference_fields": branch_contract_difference_fields,
        "active_label_difference_steps": active_diff,
        "all_joint_changes_label_difference_steps": stagnant_diff,
        "success_label_difference_steps": success_diff,
        "discrete_label_difference_steps": discrete_diff,
        "safety_label_difference_steps": safety_diff,
        "outcome_discrete_difference_horizons": outcome_discrete_diff,
        "utility_by_horizon": utility_diff,
    }


def _fidelity_summary(
    reactive: Mapping[str, Any], replay: Mapping[str, Any]
) -> dict[str, Any]:
    """Build a strict continuous and discrete reactive/replay fidelity gate."""

    diagnostic = _fidelity_diagnostic(reactive, replay)
    utility_error = max(
        float(row["utility_abs_error"])
        for row in diagnostic["utility_by_horizon"].values()
    )
    bounded_utility_error = max(
        float(row["bounded_vector_abs_error"])
        for row in diagnostic["utility_by_horizon"].values()
    )
    outcome_discrete_equal = not any(
        diagnostic["outcome_discrete_difference_horizons"].values()
    )
    discrete_equal = (
        diagnostic["metric_length_equal"]
        and diagnostic["trajectory_complete"]
        and not diagnostic["branch_contract_difference_fields"]
        and not any(diagnostic["discrete_label_difference_steps"].values())
        and not any(diagnostic["safety_label_difference_steps"].values())
        and outcome_discrete_equal
    )
    passed = bool(
        utility_error <= SNAPSHOT_TOLERANCE
        and bounded_utility_error <= SNAPSHOT_TOLERANCE
        and float(diagnostic["executed_action_max_abs_error"]) <= SNAPSHOT_TOLERANCE
        and float(diagnostic["qpos_max_abs_error"]) <= SNAPSHOT_TOLERANCE
        and float(diagnostic["progress_max_abs_error"]) <= SNAPSHOT_TOLERANCE
        and discrete_equal
    )
    return {
        "schema": FIDELITY_SCHEMA,
        "tolerance": FIDELITY_TOLERANCE,
        "repeat_id": int(reactive.get("repeat_id", -1)),
        "utility_max_abs_error": utility_error,
        "bounded_utility_max_abs_error": bounded_utility_error,
        "executed_action_max_abs_error": float(
            diagnostic["executed_action_max_abs_error"]
        ),
        "executed_action_max_abs_error_by_step": diagnostic[
            "executed_action_max_abs_error_by_step"
        ],
        "qpos_max_abs_error": float(diagnostic["qpos_max_abs_error"]),
        "qpos_max_abs_error_by_step": diagnostic["qpos_max_abs_error_by_step"],
        "progress_max_abs_error": float(diagnostic["progress_max_abs_error"]),
        "progress_max_abs_error_by_step": diagnostic[
            "progress_max_abs_error_by_step"
        ],
        "metric_length_equal": bool(diagnostic["metric_length_equal"]),
        "trajectory_complete": bool(diagnostic["trajectory_complete"]),
        "branch_contract_equal": not bool(
            diagnostic["branch_contract_difference_fields"]
        ),
        "branch_contract_difference_fields": diagnostic[
            "branch_contract_difference_fields"
        ],
        "active_labels_equal": not bool(diagnostic["active_label_difference_steps"]),
        "active_label_difference_steps": diagnostic[
            "active_label_difference_steps"
        ],
        "stagnant_labels_equal": not bool(
            diagnostic["all_joint_changes_label_difference_steps"]
        ),
        "all_joint_changes_label_difference_steps": diagnostic[
            "all_joint_changes_label_difference_steps"
        ],
        "success_labels_equal": not bool(
            diagnostic["success_label_difference_steps"]
        ),
        "success_label_difference_steps": diagnostic[
            "success_label_difference_steps"
        ],
        "discrete_labels_equal": bool(discrete_equal),
        "safety_labels_equal": not any(
            diagnostic["safety_label_difference_steps"].values()
        ),
        "outcome_discrete_labels_equal": bool(outcome_discrete_equal),
        "discrete_label_difference_steps": diagnostic[
            "discrete_label_difference_steps"
        ],
        "safety_label_difference_steps": diagnostic[
            "safety_label_difference_steps"
        ],
        "outcome_discrete_difference_horizons": diagnostic[
            "outcome_discrete_difference_horizons"
        ],
        "passed": passed,
    }


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
    if len(seeds) != families_per_task:
        raise ValueError(
            "CARE family construction requires one unique expert-valid seed "
            f"per family ({len(seeds)} != {families_per_task})"
        )
    if len(set(int(seed) for seed in seeds)) != len(seeds):
        raise ValueError("CARE family construction requires unique expert-valid seeds")
    if not 0 <= local < families_per_task:
        raise ValueError("CARE family index is outside seed coverage")
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


def _official_branch_seeds(
    args: argparse.Namespace, task: str, *, count: int
) -> list[int]:
    """Load the disjoint branch seed manifest, never Validation20 seeds."""

    dependency = require_stage_result(
        args.run, "branch_seed_discovery", config_sha256=args.config_sha256
    )
    if dependency.get("stage") != "branch_seed_discovery":
        raise RuntimeError("branch seed discovery result identity differs")
    if dependency.get("seed_bucket") != BRANCH_SEED_BUCKET:
        raise RuntimeError("branch seed discovery bucket differs")
    if dependency.get("episodes_per_task") != BRANCH_SEEDS_PER_TASK:
        raise RuntimeError("branch seed discovery coverage differs")
    path = Path(str(dependency.get("seed_manifest", "")))
    digest = dependency.get("seed_manifest_sha256")
    if not path.is_file() or not isinstance(digest, str) or sha256_file(path) != digest:
        raise RuntimeError("branch seed discovery manifest is missing or changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != "before-we-act.bicoord.expert-seed-manifest/1"
        or value.get("status") != "PASSED"
        or value.get("stage") != "branch_seed_discovery"
        or value.get("policy_independent") is not True
        or value.get("seed_bucket") != BRANCH_SEED_BUCKET
        or value.get("episodes_per_task") != BRANCH_SEEDS_PER_TASK
    ):
        raise RuntimeError("branch seed manifest is not the frozen branch contract")
    rows = value.get("valid_seeds", {}).get(task)
    if (
        not isinstance(rows, list)
        or len(rows) != count
        or len(set(rows)) != len(rows)
    ):
        raise RuntimeError(f"branch expert-valid seed coverage is incomplete for {task}")
    return [int(seed) for seed in rows]


def _collect_family(
    *,
    args: argparse.Namespace,
    runtime: Any,
    checkpoint: Path,
    checkpoint_sha256: str,
    normalization: Mapping[str, Any],
    seed_manifest: str,
    seed_manifest_sha256: str,
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
        # Physical branches are not resumable artifacts.  A worker may only
        # publish a family into an empty run namespace; even a byte-valid
        # family from an earlier attempt cannot prove that its failed sibling
        # wave used the same executable/revision/runtime state.  The
        # supervisor enforces this at stage scope, and the worker independently
        # fails closed here for direct/manual invocations.
        raise RuntimeError(
            f"CARE family output already exists; use a completely new run: {snapshot_id}"
        )

    _seed_rng(seed)
    env = _make_env(args.benchmark_repo, task, seed)
    # The environment constructor applies the run-local compatibility overlay
    # for the two released metadata defects.  Persist a detached copy in the
    # family receipt so formal counterfactual data cannot silently come from a
    # different asset contract than validation.
    asset_overlay = getattr(env, "_bicoord_asset_overlay", None)
    if isinstance(asset_overlay, Mapping):
        asset_overlay = deepcopy(dict(asset_overlay))
    else:
        asset_overlay = None
    started = time.perf_counter()
    try:
        runtime.reset(task)
        observation = env.get_obs()
        prefix_actions: list[np.ndarray] = []
        for step in range(anchor_step):
            context = runtime.act_with_context(observation, task, belief_enabled=True, commit=False)
            action = context.reference_plan[:, 0].copy()
            runtime.record_executed_actions(action)
            env.take_action(action.reshape(-1))
            observation = env.get_obs()
            prefix_actions.append(action.copy())
            success, _progress = _success_progress(env)
            if success:
                raise RuntimeError(f"B-core reached a terminal state before frozen anchor {snapshot_id}")
        _success, start_progress = _success_progress(env)
        simulator_state = capture_state(env)
        runtime_state = runtime.snapshot_state()
        simulator_state_hash = state_sha256(simulator_state)
        runtime_state_hash = state_sha256(runtime_state)
        probe = _replay_reference_probe(
            args=args,
            task=task,
            seed=seed,
            prefix_actions=prefix_actions,
            runtime=runtime,
            runtime_state=runtime_state,
            expected_anchor_state_sha256=simulator_state_hash,
        )
        if probe.get("passed") is not True:
            raise RuntimeError(f"BiCoord seed+prefix replay probe failed: {probe}")
        # The original official environment is still exactly at the anchor;
        # only the pure policy runtime was used by the replay probe.
        runtime.restore_state(runtime_state)
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
        def branch_env_factory() -> _RebuildableBranchEnv:
            return _RebuildableBranchEnv(
                lambda: _replay_anchor_environment(
                    args, task, seed, prefix_actions
                )
            )

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
                env_factory=branch_env_factory,
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
                env_factory=branch_env_factory,
            )
            branches.append(reference_replay)
            fidelity_row = _fidelity_summary(reference_reactive, reference_replay)
            fidelity.append(fidelity_row)
            if fidelity_row["passed"] is not True:
                diagnostic = _fidelity_diagnostic(reference_reactive, reference_replay)
                diagnostic["repeat_id"] = int(repeat)
                diagnostic["snapshot_id"] = snapshot_id
                diagnostic_path = output_root / f"family_{family_id:06d}.fidelity_diagnostic.json"
                atomic_json(diagnostic_path, diagnostic)
                raise RuntimeError(
                    f"candidate-0 reactive/replay fidelity failed: {fidelity_row}; "
                    f"diagnostic={diagnostic_path}"
                )
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
                    env_factory=branch_env_factory,
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
                    env_factory=branch_env_factory,
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
            "seed_manifest": seed_manifest,
            "seed_manifest_sha256": seed_manifest_sha256,
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
            "simulator_restore_mode": "official_seed_plus_reference_prefix_replay",
            "reference_prefix_steps": len(prefix_actions),
            "reference_prefix_sha256": state_sha256(prefix_actions),
            "reference_reactive_replay_fidelity": fidelity,
            "action_clipping": False,
            "candidate_transform_clipping": False,
            "candidate_values_outside_source_population_range": out_of_source_range,
            "normalization_receipt_sha256": normalization["sha256"],
            "asset_overlay": asset_overlay,
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
    seed_count = 1 if smoke else BRANCH_SEEDS_PER_TASK
    seed_rows = {
        task: (
            _official_seeds(args, task, count=seed_count)
            if smoke
            else _official_branch_seeds(args, task, count=seed_count)
        )
        for task in TASKS
    }
    seed_stage = "seed_discovery_smoke" if smoke else "branch_seed_discovery"
    seed_dependency = require_stage_result(
        args.run, seed_stage, config_sha256=args.config_sha256
    )
    seed_manifest_path = Path(str(seed_dependency.get("seed_manifest", ""))).resolve()
    seed_manifest_sha = seed_dependency.get("seed_manifest_sha256")
    if not seed_manifest_path.is_file() or sha256_file(seed_manifest_path) != seed_manifest_sha:
        raise RuntimeError("branch seed manifest artifact/hash differs")
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
                seed_manifest=str(seed_manifest_path),
                seed_manifest_sha256=str(seed_manifest_sha),
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
            "seed_manifest": str(seed_manifest_path),
            "seed_manifest_sha256": str(seed_manifest_sha),
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
        seed_manifest=str(seed_manifest_path),
        seed_manifest_sha256=str(seed_manifest_sha),
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
