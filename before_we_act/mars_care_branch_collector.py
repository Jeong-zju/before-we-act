"""MARS state-restore and outcome adapter for official CARE branch families."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from before_we_act.mars_action_contract import (
    canonicalize_action,
    validate_action_space_bounds,
)
from before_we_act.care_branch_collector import (
    OUTCOME_HORIZONS,
    atomic_json,
    atomic_npz,
    canonicalize_policy_plans,
    clone_tree,
    numeric_tree_max_abs,
    outcome_at_horizon,
    sha256_file,
)
from before_we_act.care_behavior_candidates import BehaviorCandidateConfig
from before_we_act.care_candidate_family import (
    BEHAVIOR_FAMILY,
    CANDIDATE_FAMILIES,
    FIXED_FAMILY,
    build_candidate,
    candidate_count,
    family_manifest,
    validate_candidate_for_family,
)
from before_we_act.mars_care_runtime import (
    MarsCARERuntime,
    append_action,
    current_qpos,
    environment,
    load_reference,
    local_observation_tree,
    new_runtime,
    policy_plan,
    privileged_task_metrics,
    scalar,
    sliced_runtime,
)
from deployment.mars_care.common import TASK_BY_NAME


FORMAT_VERSION = (
    "before-we-act.care-mars-branch-family/4-separate-rerender-diagnostic"
)
# MARS acts on 7 joints plus a gripper over a 100-step chunk.
ACTION_HORIZON = 100
ACTION_DIM = 8
# candidates x {reactive, replay} x 2 matched repeats
BRANCH_COUNT = 24


def branch_count(family: str) -> int:
    return candidate_count(family) * 2 * 2
INTERVENTION_STEP_CHOICES = (1, 4, 8, 16)


@dataclass
class MarsSimulatorSnapshot:
    state: Any
    elapsed_steps: torch.Tensor
    wrapper_elapsed_steps: Any
    observation: Any
    runtime: MarsCARERuntime
    python_rng: object
    numpy_rng: tuple[Any, ...]
    torch_rng: torch.Tensor
    cuda_rng: list[torch.Tensor]
    episode_rng: tuple[Any, ...] | None
    start_metrics: dict[str, Any]


def capture_snapshot(
    env: Any,
    observation: Any,
    runtime: MarsCARERuntime,
    start_metrics: Mapping[str, Any],
) -> MarsSimulatorSnapshot:
    base_env = getattr(env, "base_env", env.unwrapped)
    episode_rng = None
    if getattr(base_env, "_episode_rng", None) is not None:
        episode_rng = deepcopy(base_env._episode_rng.get_state())
    return MarsSimulatorSnapshot(
        state=clone_tree(base_env.get_state_dict()),
        elapsed_steps=base_env.elapsed_steps.clone(),
        wrapper_elapsed_steps=clone_tree(getattr(env, "_elapsed_steps", None)),
        observation=clone_tree(observation),
        runtime=deepcopy(runtime),
        python_rng=random.getstate(),
        numpy_rng=deepcopy(np.random.get_state()),
        torch_rng=torch.get_rng_state().clone(),
        cuda_rng=[value.clone() for value in torch.cuda.get_rng_state_all()],
        episode_rng=episode_rng,
        start_metrics=deepcopy(dict(start_metrics)),
    )


def branch_seed(snapshot_id: str, repeat_id: int) -> int:
    value = hashlib.sha256(
        f"before-we-act/care-mars-branch/v1|{snapshot_id}|{repeat_id}".encode()
    ).digest()
    return int.from_bytes(value[:4], "big")


def _seed_branch(base_env: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if getattr(base_env, "_episode_rng", None) is not None:
        state = np.random.RandomState(seed % 2**32).get_state()
        base_env._episode_rng.set_state(state)
        if getattr(base_env, "_batched_episode_rng", None) is not None:
            base_env._batched_episode_rng[0].set_state(state)


def restore_snapshot(
    env: Any, snapshot: MarsSimulatorSnapshot, seed: int
) -> tuple[Any, MarsCARERuntime, float, float]:
    base_env = getattr(env, "base_env", env.unwrapped)
    base_env.set_state_dict(clone_tree(snapshot.state))
    base_env._elapsed_steps = snapshot.elapsed_steps.clone()
    if snapshot.wrapper_elapsed_steps is not None and hasattr(env, "_elapsed_steps"):
        env._elapsed_steps = clone_tree(snapshot.wrapper_elapsed_steps)
    random.setstate(snapshot.python_rng)
    np.random.set_state(snapshot.numpy_rng)
    torch.set_rng_state(snapshot.torch_rng)
    torch.cuda.set_rng_state_all(snapshot.cuda_rng)
    if snapshot.episode_rng is not None and getattr(base_env, "_episode_rng", None) is not None:
        base_env._episode_rng.set_state(deepcopy(snapshot.episode_rng))
        if getattr(base_env, "_batched_episode_rng", None) is not None:
            base_env._batched_episode_rng[0].set_state(deepcopy(snapshot.episode_rng))
    rerendered = base_env.get_obs()
    rerender_error = numeric_tree_max_abs(
        local_observation_tree(rerendered, snapshot.runtime.arms),
        local_observation_tree(snapshot.observation, snapshot.runtime.arms),
    )
    restored = clone_tree(snapshot.observation)
    # Branch execution deliberately consumes the captured observation, as in
    # the upstream RoboFactory CARE collector.  The state-restore admission
    # quantity is therefore proprioceptive parity; a fresh Vulkan render can
    # differ by a few integer pixel values and is retained only as a separate
    # diagnostic.  Conflating the two makes deterministic physical restores
    # fail because of renderer rasterization noise that no branch consumes.
    qpos_error = numeric_tree_max_abs(
        local_observation_tree(rerendered, snapshot.runtime.arms)["agent"],
        local_observation_tree(restored, snapshot.runtime.arms)["agent"],
    )
    _seed_branch(base_env, seed)
    return restored, deepcopy(snapshot.runtime), float(qpos_error), float(rerender_error)


def _canonical_plans(
    reference: Mapping[str, np.ndarray],
    base: Mapping[str, np.ndarray],
    env: Any,
    arms: Sequence[int],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    for arm in arms:
        validate_action_space_bounds(env.action_space.spaces[f"panda-{arm}"])
    result = canonicalize_policy_plans(reference, base, env.action_space.spaces, arms)
    return result[0], result[1]


def _canonical_candidate(plan: np.ndarray, action_space: Any) -> np.ndarray:
    """Clip transformed candidates exactly as the MARS controller does."""
    value = np.asarray(plan, dtype=np.float32)
    if value.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(value).all():
        return value
    validate_action_space_bounds(action_space)
    return canonicalize_action(value)


def intervention_action(
    reference_action: np.ndarray,
    candidate_plan: np.ndarray,
    *,
    branch_step: int,
    intervention_steps: int,
) -> np.ndarray:
    """Choose the focal action under the frozen duration contract.

    The candidate chunk is generated once from the restored snapshot.  Its
    first ``intervention_steps`` rows are executed open loop; after that the
    focal arm returns to the freshly recomputed reference policy.  Duration 1
    is bit-equivalent to the original MARS collector.
    """

    if int(intervention_steps) not in INTERVENTION_STEP_CHOICES:
        raise ValueError("MARS CARE intervention steps must be one of 1/4/8/16")
    step = int(branch_step)
    if step < 0:
        raise ValueError("MARS CARE branch step must be non-negative")
    reference = np.asarray(reference_action, dtype=np.float32)
    candidate = np.asarray(candidate_plan, dtype=np.float32)
    if reference.shape != (ACTION_DIM,) or candidate.shape != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError("MARS CARE intervention action shape differs")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("MARS CARE intervention action must be finite")
    return (candidate[step] if step < int(intervention_steps) else reference).copy()


def branch_action(
    candidate_id: int,
    reference_action: np.ndarray,
    candidate_plan_value: np.ndarray,
    *,
    branch_step: int,
    intervention_steps: int,
) -> np.ndarray:
    """Keep candidate zero exact-reference; duration affects only alternatives."""

    if int(candidate_id) == 0:
        reference = np.asarray(reference_action, dtype=np.float32)
        if reference.shape != (8,) or not np.isfinite(reference).all():
            raise ValueError("MARS CARE reference action must be finite width 8")
        return reference.copy()
    return intervention_action(
        reference_action,
        candidate_plan_value,
        branch_step=branch_step,
        intervention_steps=intervention_steps,
    )


def run_branch(
    *,
    env: Any,
    snapshot: MarsSimulatorSnapshot,
    snapshot_id: str,
    model: Any,
    stats: Mapping[str, torch.Tensor],
    task: str,
    focal_arm: int,
    candidate_id: int,
    regime: str,
    repeat_id: int,
    teammate_reference_actions: Sequence[Mapping[str, np.ndarray]] | None,
    device: torch.device,
    horizon: int = 64,
    intervention_steps: int = 1,
    candidate_family: str = FIXED_FAMILY,
    candidate_config: BehaviorCandidateConfig | None = None,
) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
    if regime not in {"reactive", "replay"}:
        raise ValueError("CARE response regime must be reactive or replay")
    if int(intervention_steps) not in INTERVENTION_STEP_CHOICES:
        raise ValueError("MARS CARE intervention steps must be one of 1/4/8/16")
    if int(intervention_steps) > int(horizon):
        raise ValueError("MARS CARE intervention exceeds branch horizon")
    seed = branch_seed(snapshot_id, repeat_id)
    observation, full_runtime, restore_error, rerender_error = restore_snapshot(
        env, snapshot, seed
    )
    runtime = full_runtime if regime == "reactive" else sliced_runtime(full_runtime, (focal_arm,))
    base_env = getattr(env, "base_env", env.unwrapped)
    rows: list[dict[str, Any]] = []
    executed: list[dict[str, np.ndarray]] = []
    candidate_valid = True
    failures: list[str] = []
    status = "VALID"
    replay_error = 0.0
    reference_error = 0.0
    intervention_plan: np.ndarray | None = None
    started = time.perf_counter()
    for branch_step in range(horizon):
        before = current_qpos(observation, full_runtime.arms)
        reference, base, physical_qpos, _memory, _mask, diagnostics = policy_plan(
            model, stats, observation, runtime, task, device
        )
        reference, base = _canonical_plans(reference, base, env, runtime.arms)
        action = {key: value[0].copy() for key, value in reference.items()}
        if regime == "replay":
            if teammate_reference_actions is None or branch_step >= len(teammate_reference_actions):
                status = "INVALID_REPLAY_LOG"
                break
            for arm in full_runtime.arms:
                if arm == focal_arm:
                    continue
                key = f"panda-{arm}"
                replayed = np.asarray(teammate_reference_actions[branch_step][key], dtype=np.float32)
                action[key] = replayed.copy()
                replay_error = max(
                    replay_error,
                    float(np.max(np.abs(action[key] - replayed))),
                )
        if branch_step == 0:
            key = f"panda-{focal_arm}"
            focal_index = runtime.arms.index(focal_arm)
            current_grip = (
                float(full_runtime.last_action[key][7])
                if full_runtime.last_action is not None
                else float(reference[key][0, 7])
            )
            transformed = build_candidate(
                candidate_family,
                candidate_id,
                reference=reference[key],
                base=base[key],
                current_qpos=physical_qpos[focal_index],
                current_grip=current_grip,
                config=candidate_config,
            )
            transformed = _canonical_candidate(transformed, env.action_space.spaces[key])
            valid, step_failures = validate_candidate_for_family(
                candidate_family,
                candidate_id,
                transformed,
                reference=reference[key],
                base=base[key],
                current_qpos=physical_qpos[focal_index],
                current_grip=current_grip,
                action_space=env.action_space.spaces[key],
                config=candidate_config,
            )
            if valid:
                intervention_plan = transformed
            else:
                candidate_valid = False
                failures.extend(step_failures)
                status = "INVALID_CANDIDATE"
        key = f"panda-{focal_arm}"
        if intervention_plan is not None:
            action[key] = branch_action(
                candidate_id,
                reference[key][0],
                intervention_plan,
                branch_step=branch_step,
                intervention_steps=intervention_steps,
            )
        if candidate_id == 0:
            reference_error = max(
                reference_error,
                float(np.max(np.abs(action[key] - reference[key][0]))),
            )
        append_action(
            runtime,
            {f"panda-{arm}": action[f"panda-{arm}"] for arm in runtime.arms},
            stats,
        )
        try:
            observation, _reward, terminated, truncated, info = env.step(action)
        except Exception as error:  # pragma: no cover - simulator fatal path
            status = f"SIMULATOR_FATAL:{type(error).__name__}"
            break
        metrics = privileged_task_metrics(base_env, task, action, before)
        metrics["branch_step"] = branch_step
        metrics["belief_diagnostics"] = diagnostics
        rows.append(metrics)
        executed.append({key: np.asarray(value).copy() for key, value in action.items()})
        success = scalar(info.get("success", False))
        if success or scalar(terminated) or scalar(truncated):
            status = "SUCCESS_TERMINATION" if success else "PREMATURE_TERMINATION"
            break
    outcomes = {
        str(value): outcome_at_horizon(
            rows, float(snapshot.start_metrics["progress"]), value
        )
        for value in OUTCOME_HORIZONS
        if rows
    }
    return (
        {
            "candidate_id": candidate_id,
            "regime": regime,
            "repeat_id": repeat_id,
            "branch_seed": seed,
            "status": status,
            "candidate_valid": candidate_valid,
            "candidate_failures": failures,
            "restore_observation_max_abs_error": restore_error,
            "restore_rerender_diagnostic_max_abs_error": rerender_error,
            "restore_observation_source": "captured_snapshot",
            "candidate0_reference_action_max_abs_error": reference_error,
            "replay_teammate_action_max_abs_error": replay_error,
            "steps": len(rows),
            "execution_horizon_steps": horizon,
            "intervention_steps_requested": int(intervention_steps),
            "intervention_steps_applied": (
                min(len(rows), int(intervention_steps))
                if intervention_plan is not None and candidate_id != 0
                else 0
            ),
            "reference_candidate_fail_closed_all_steps": candidate_id == 0,
            "wall_seconds": time.perf_counter() - started,
            "outcomes": outcomes,
            "executed_actions": [
                {key: np.asarray(value).tolist() for key, value in row.items()}
                for row in executed
            ],
        },
        executed,
    )


def collect_family(
    *,
    family: Mapping[str, Any],
    env: Any,
    model: Any,
    stats: Mapping[str, torch.Tensor],
    checkpoint: Path,
    device: torch.device,
    output_root: Path,
    checkpoint_identity_sha256: str | None = None,
    intervention_steps: int = 1,
    candidate_family: str = FIXED_FAMILY,
    candidate_config: BehaviorCandidateConfig | None = None,
) -> dict[str, Any]:
    if int(intervention_steps) not in INTERVENTION_STEP_CHOICES:
        raise ValueError("MARS CARE intervention steps must be one of 1/4/8/16")
    task = str(family["task"])
    arms = tuple(range(TASK_BY_NAME[task].arms))
    focal = int(family["focal_agent"])
    if focal not in arms:
        raise ValueError("MARS CARE focal arm is invalid")
    runtime = new_runtime(arms)
    seed = int(family["episode_seed"])
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    observation, _info = env.reset(seed=seed)
    base_env = getattr(env, "base_env", env.unwrapped)
    empty_action = {
        f"panda-{arm}": current_qpos(observation, arms)[f"panda-{arm}"][:8]
        for arm in arms
    }
    start_metrics = privileged_task_metrics(
        base_env, task, empty_action, current_qpos(observation, arms)
    )
    for _ in range(int(family["anchor_step"])):
        reference, base, _qpos, _memory, _mask, _diagnostics = policy_plan(
            model, stats, observation, runtime, task, device
        )
        reference, _base = _canonical_plans(reference, base, env, arms)
        action = {key: value[0].copy() for key, value in reference.items()}
        append_action(runtime, action, stats)
        observation, _reward, terminated, truncated, info = env.step(action)
        start_metrics = privileged_task_metrics(
            base_env, task, action, current_qpos(observation, arms)
        )
        if scalar(info.get("success", False)) or scalar(terminated) or scalar(truncated):
            raise RuntimeError(
                f"MARS CARE episode ended before anchor {family['snapshot_id']}"
            )
    snapshot = capture_snapshot(env, observation, runtime, start_metrics)
    preview_runtime = deepcopy(runtime)
    reference, base, physical_qpos, memory, memory_mask, diagnostics = policy_plan(
        model, stats, observation, preview_runtime, task, device
    )
    reference, base = _canonical_plans(reference, base, env, arms)
    focal_index = arms.index(focal)
    focal_key = f"panda-{focal}"
    current_grip = (
        float(runtime.last_action[focal_key][7])
        if runtime.last_action is not None
        else float(reference[focal_key][0, 7])
    )
    candidates: list[dict[str, Any]] = []
    for candidate_id in range(candidate_count(candidate_family)):
        plan = build_candidate(
            candidate_family,
            candidate_id,
            reference=reference[focal_key],
            base=base[focal_key],
            current_qpos=physical_qpos[focal_index],
            current_grip=current_grip,
            config=candidate_config,
        )
        plan = _canonical_candidate(plan, env.action_space.spaces[focal_key])
        valid, failures = validate_candidate_for_family(
            candidate_family,
            candidate_id,
            plan,
            reference=reference[focal_key],
            base=base[focal_key],
            current_qpos=physical_qpos[focal_index],
            current_grip=current_grip,
            action_space=env.action_space.spaces[focal_key],
            config=candidate_config,
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "valid": valid,
                "failures": failures,
                "plan": plan,
            }
        )
    if not all(row["valid"] for row in candidates):
        details = {
            int(row["candidate_id"]): list(row["failures"])
            for row in candidates
            if not row["valid"]
        }
        raise RuntimeError(
            f"illegal MARS CARE candidate family: {family['snapshot_id']}; "
            f"failures={details}"
        )
    branches: list[dict[str, Any]] = []
    started = time.perf_counter()
    for repeat_id in (0, 1):
        reference_reactive, teammate_log = run_branch(
            env=env,
            snapshot=snapshot,
            snapshot_id=str(family["snapshot_id"]),
            model=model,
            stats=stats,
            task=task,
            focal_arm=focal,
            candidate_id=0,
            regime="reactive",
            repeat_id=repeat_id,
            teammate_reference_actions=None,
            device=device,
            intervention_steps=intervention_steps,
            candidate_family=candidate_family,
            candidate_config=candidate_config,
        )
        branches.append(reference_reactive)
        reference_replay, _ = run_branch(
            env=env,
            snapshot=snapshot,
            snapshot_id=str(family["snapshot_id"]),
            model=model,
            stats=stats,
            task=task,
            focal_arm=focal,
            candidate_id=0,
            regime="replay",
            repeat_id=repeat_id,
            teammate_reference_actions=teammate_log,
            device=device,
            intervention_steps=intervention_steps,
            candidate_family=candidate_family,
            candidate_config=candidate_config,
        )
        branches.append(reference_replay)
        for candidate_id in range(1, candidate_count(candidate_family)):
            reactive, _ = run_branch(
                env=env,
                snapshot=snapshot,
                snapshot_id=str(family["snapshot_id"]),
                model=model,
                stats=stats,
                task=task,
                focal_arm=focal,
                candidate_id=candidate_id,
                regime="reactive",
                repeat_id=repeat_id,
                teammate_reference_actions=None,
                device=device,
                intervention_steps=intervention_steps,
                candidate_family=candidate_family,
                candidate_config=candidate_config,
            )
            branches.append(reactive)
        for candidate_id in range(1, candidate_count(candidate_family)):
            replay, _ = run_branch(
                env=env,
                snapshot=snapshot,
                snapshot_id=str(family["snapshot_id"]),
                model=model,
                stats=stats,
                task=task,
                focal_arm=focal,
                candidate_id=candidate_id,
                regime="replay",
                repeat_id=repeat_id,
                teammate_reference_actions=teammate_log,
                device=device,
                intervention_steps=intervention_steps,
                candidate_family=candidate_family,
                candidate_config=candidate_config,
            )
            branches.append(replay)
    if len(branches) != branch_count(candidate_family):
        raise AssertionError(len(branches))
    snapshot_id = str(family["snapshot_id"])
    npz_path = output_root / task / f"{snapshot_id}.npz"
    json_path = output_root / task / f"{snapshot_id}.json"
    atomic_npz(
        npz_path,
        memory=memory[focal_index].float().cpu().numpy().astype(np.float16),
        memory_mask=memory_mask[focal_index].cpu().numpy(),
        candidate_chunks=np.stack([row["plan"] for row in candidates]),
        focal_agent=np.asarray([focal], dtype=np.int64),
    )
    result = {
        "format_version": FORMAT_VERSION,
        "snapshot_id": snapshot_id,
        "task": task,
        "episode_seed": seed,
        "anchor_step": int(family["anchor_step"]),
        "focal_agent": focal,
        "sampling_stratum": str(family["sampling_stratum"]),
        "scenario_group_id": str(family["scenario_group_id"]),
        "checkpoint": str(checkpoint.resolve()),
        # A metadata-only migration wrapper may be used to load a legacy
        # checkpoint.  Keep the immutable source checkpoint as the corpus
        # identity so already collected families remain reusable.
        "checkpoint_sha256": checkpoint_identity_sha256 or sha256_file(checkpoint),
        "runtime_contract": {
            "shared_weights": True,
            "strict_local": True,
            "per_robot_independent_inputs": True,
            "image_layout": "native HWC 240x320 to CHW at model boundary",
            "action_encoding": "absolute_pd_joint_pos",
        },
        "candidate_legality": [
            {key: value for key, value in row.items() if key != "plan"}
            for row in candidates
        ],
        "snapshot_metrics": start_metrics,
        "prebranch_diagnostics": diagnostics,
        "branches": branches,
        "branch_count": len(branches),
        "intervention_steps": int(intervention_steps),
        **family_manifest(candidate_family, candidate_config),
        "wall_seconds": time.perf_counter() - started,
        "npz_path": str(npz_path.resolve()),
    }
    atomic_json(json_path, result)
    result["json_sha256"] = sha256_file(json_path)
    result["npz_sha256"] = sha256_file(npz_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--robofactory-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--task", choices=tuple(TASK_BY_NAME))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--checkpoint-identity-sha256",
        help="optional immutable source-checkpoint identity for a metadata wrapper",
    )
    parser.add_argument(
        "--intervention-steps",
        type=int,
        choices=INTERVENTION_STEP_CHOICES,
        default=1,
        help="fixed candidate prefix; default 1 preserves the main protocol",
    )
    parser.add_argument(
        "--candidate-family",
        choices=CANDIDATE_FAMILIES,
        default=FIXED_FAMILY,
        help=(
            "fixed keeps the archived first-step transforms; behavior uses the "
            "wait/retreat/grasp-timing family held across the commitment window"
        ),
    )
    parser.add_argument(
        "--render-device",
        help="physical SAPIEN Vulkan device, e.g. cuda:1; defaults to --device",
    )
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    candidate_config = None
    if args.candidate_family == BEHAVIOR_FAMILY:
        prefix = int(args.intervention_steps)
        if prefix < 4:
            raise ValueError(
                "behavior candidates need a commitment window of at least four "
                f"steps to be distinguishable; got --intervention-steps {prefix}"
            )
        candidate_config = BehaviorCandidateConfig(
            action_horizon=ACTION_HORIZON,
            action_dim=ACTION_DIM,
            intervention_steps=prefix,
            wait_steps=max(1, prefix // 2),
            grip_shift_steps=max(1, prefix // 2),
        )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("format_version") != "before-we-act.care-mars-family-manifest/1":
        raise ValueError("wrong MARS CARE family manifest")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model, stats, _config = load_reference(args.checkpoint, device)
    checkpoint_identity_sha256 = (
        str(args.checkpoint_identity_sha256).lower()
        if args.checkpoint_identity_sha256
        else sha256_file(args.checkpoint)
    )
    if len(checkpoint_identity_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in checkpoint_identity_sha256
    ):
        raise ValueError("checkpoint identity must be a lowercase SHA-256 digest")
    families = [
        row
        for index, row in enumerate(manifest["families"])
        if index % args.shard_count == args.shard_index
        and (args.task is None or row["task"] == args.task)
    ]
    if args.limit:
        families = families[: args.limit]
    for task in sorted({str(row["task"]) for row in families}):
        env = environment(task, args.robofactory_root, args.render_device or args.device)
        try:
            for family in (row for row in families if row["task"] == task):
                output = args.output_root / task / f"{family['snapshot_id']}.json"
                if output.is_file():
                    existing = json.loads(output.read_text())
                    if (
                        existing.get("format_version") == FORMAT_VERSION
                        and existing.get("branch_count") == BRANCH_COUNT
                        and existing.get("checkpoint_sha256") == checkpoint_identity_sha256
                        and int(existing.get("intervention_steps", -1)) == args.intervention_steps
                        and str(existing.get("candidate_family", FIXED_FAMILY))
                        == args.candidate_family
                    ):
                        print(json.dumps({"snapshot_id": family["snapshot_id"], "reused": True}), flush=True)
                        continue
                    raise RuntimeError(f"refusing to overwrite {output}")
                result = collect_family(
                    family=family,
                    env=env,
                    model=model,
                    stats=stats,
                    checkpoint=args.checkpoint,
                    checkpoint_identity_sha256=checkpoint_identity_sha256,
                    device=device,
                    output_root=args.output_root,
                    intervention_steps=args.intervention_steps,
                    candidate_family=args.candidate_family,
                    candidate_config=candidate_config,
                )
                print(
                    json.dumps(
                        {
                            "snapshot_id": result["snapshot_id"],
                            "task": result["task"],
                            "branch_count": result["branch_count"],
                            "wall_seconds": result["wall_seconds"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        finally:
            env.close()


if __name__ == "__main__":
    main()
