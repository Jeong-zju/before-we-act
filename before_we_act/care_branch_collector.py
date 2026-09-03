"""同状态 CARE 分叉采集器。

本模块实现 CARE 的动作候选、状态恢复和 reactive/replay 双分叉。
资源试跑产生的数据不得用于 CARE 训练。
"""
from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
import torch

from before_we_act.task_manifest import get_task

from before_we_act.deployment_safety import (
    DeploymentProgressWatchdog,
    ResidualSafetyConfig,
)
from before_we_act.ssc_v7_oracle_labels import (
    TASK_STAGE_NAMES,
    build_oracle_label,
    initial_automaton_state,
)
from before_we_act.temporal_history_data import HISTORY_STEPS
from scripts.before_we_act.audit_ssc_v7_m2 import (
    extract_snapshot as extract_oracle_snapshot,
    scalar_bool,
)


FORMAT_VERSION = "before-we-act.care-branch-family/1"
CONTRACT_STAGE = "CARE-BRANCH-COLLECTION"
FORMAL_FORMAT_VERSION = "before-we-act.care-formal-branch-family/1"
FORMAL_CONTRACT_STAGE = "CARE-FORMAL-COLLECTION"
FORMAL_MANIFEST_STAGE = "CARE-BRANCHES-FORMAL"
GATE_FIRST_FORMAT_VERSION = "before-we-act.care-gate-first-branch-family/1"
GATE_FIRST_CONTRACT_STAGE = "CARE-GATE-FIRST-COLLECTION"
GATE_FIRST_MANIFEST_STAGE = "CARE-GATE-FIRST-BRANCHES"
COMPACT_FORMAT_VERSION = "before-we-act.care-compact-branch-family/1"
COMPACT_CONTRACT_STAGE = "CARE-COMPACT-COLLECTION"
COMPACT_MANIFEST_STAGE = "CARE-COMPACT-BRANCHES"
COMMON_SUPPORT_FORMAT_VERSION = "before-we-act.care-common-support-branch-family/1"
COMMON_SUPPORT_CONTRACT_STAGE = "CARE-COMMON-SUPPORT-COLLECTION"
COMMON_SUPPORT_MANIFEST_STAGE = "CARE-COMMON-SUPPORT-BRANCHES"
OUTCOME_HORIZONS = (8, 16, 32, 64)
MAIN_WEIGHTS = np.asarray((0.30, 0.30, 0.12, 0.08, 0.06, 0.06, 0.03, 0.05))


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def clone_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: clone_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.copy()
    if torch.is_tensor(value):
        return value.clone()
    return deepcopy(value)


def numpy_value(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def numeric_tree_max_abs(first: Any, second: Any) -> float:
    """比较两个嵌套数值树；结构不一致返回正无穷。"""

    if isinstance(first, Mapping) or isinstance(second, Mapping):
        if not isinstance(first, Mapping) or not isinstance(second, Mapping):
            return math.inf
        if set(first) != set(second):
            return math.inf
        return max(
            (numeric_tree_max_abs(first[key], second[key]) for key in first),
            default=0.0,
        )
    if isinstance(first, (tuple, list)) or isinstance(second, (tuple, list)):
        if not isinstance(first, (tuple, list)) or not isinstance(
            second, (tuple, list)
        ):
            return math.inf
        if len(first) != len(second):
            return math.inf
        return max(
            (numeric_tree_max_abs(left, right) for left, right in zip(first, second)),
            default=0.0,
        )
    left = numpy_value(first)
    right = numpy_value(second)
    if left.shape != right.shape:
        return math.inf
    if left.dtype.kind in "OUS" or right.dtype.kind in "OUS":
        return 0.0 if np.array_equal(left, right) else math.inf
    if left.size == 0:
        return 0.0
    return float(
        np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))
    )


def deployment_observation_tree(
    observation: Mapping[str, Any], arms: Sequence[int]
) -> dict[str, Any]:
    """只保留部署策略实际读取的关节状态和相机图像。"""

    sensors = observation["sensor_data"]
    selected_sensors = {
        "head_camera_global": {
            "rgb": clone_tree(sensors["head_camera_global"]["rgb"])
        }
    }
    for arm in arms:
        key = f"head_camera_agent{int(arm)}"
        selected_sensors[key] = {"rgb": clone_tree(sensors[key]["rgb"])}
    return {
        "agent": {
            f"panda-{int(arm)}": {
                "qpos": clone_tree(observation["agent"][f"panda-{int(arm)}"]["qpos"])
            }
            for arm in arms
        },
        "sensor_data": selected_sensors,
    }


class ConsolidatedChunkEnsembler:
    """把所有仍有效的原始 chunk 合成为一条 100 步物理计划。"""

    def __init__(self, arms: Sequence[int], *, horizon: int = 100, decay: float = 0.01):
        self.arms = tuple(int(arm) for arm in arms)
        self.horizon = int(horizon)
        self.decay = float(decay)
        self.histories: dict[int, list[tuple[int, np.ndarray]]] = {
            arm: [] for arm in self.arms
        }

    def append_and_plan(
        self, step: int, chunks: np.ndarray
    ) -> dict[str, np.ndarray]:
        if chunks.shape[:2] != (len(self.arms), self.horizon):
            raise ValueError("CARE chunk shape differs from the frozen 100-step contract")
        plans: dict[str, np.ndarray] = {}
        for local_index, arm in enumerate(self.arms):
            history = self.histories[arm]
            history.append((int(step), np.asarray(chunks[local_index]).copy()))
            history[:] = [
                item for item in history if int(step) - item[0] < len(item[1])
            ]
            rows: list[np.ndarray] = []
            for offset in range(self.horizon):
                absolute = int(step) + offset
                candidates = [
                    chunk[absolute - start]
                    for start, chunk in history
                    if 0 <= absolute - start < len(chunk)
                ]
                if not candidates:
                    raise RuntimeError("newest CARE chunk did not cover its own horizon")
                candidate_array = np.asarray(candidates)
                weights = np.exp(
                    -self.decay
                    * np.arange(len(candidates) - 1, -1, -1, dtype=np.float64)
                )
                weights /= weights.sum()
                rows.append(np.sum(candidate_array * weights[:, None], axis=0))
            plans[f"panda-{arm}"] = np.asarray(rows, dtype=np.float32)
        return plans


@dataclass
class PolicyRuntime:
    arms: tuple[int, ...]
    history: Any
    reference_ensembler: ConsolidatedChunkEnsembler
    base_ensembler: ConsolidatedChunkEnsembler
    watchdogs: dict[int, DeploymentProgressWatchdog]
    oracle_memory: dict[str, Any]
    step: int = 0
    last_action: dict[str, np.ndarray] | None = None


@dataclass
class SimulatorSnapshot:
    state: Any
    elapsed_steps: torch.Tensor
    wrapper_elapsed_steps: Any
    observation: Any
    runtime: PolicyRuntime
    python_rng: object
    numpy_rng: tuple[Any, ...]
    torch_rng: torch.Tensor
    cuda_rng: list[torch.Tensor]
    episode_rng: tuple[Any, ...] | None
    label: dict[str, Any]


def new_runtime(
    arms: Sequence[int], safety: ResidualSafetyConfig, task: str
) -> PolicyRuntime:
    from before_we_act.evaluate_temporal_history_policy import EpisodeHistory

    canonical = tuple(int(arm) for arm in arms)
    return PolicyRuntime(
        arms=canonical,
        history=EpisodeHistory(canonical),
        reference_ensembler=ConsolidatedChunkEnsembler(canonical),
        base_ensembler=ConsolidatedChunkEnsembler(canonical),
        watchdogs={arm: DeploymentProgressWatchdog(safety) for arm in canonical},
        oracle_memory=initial_automaton_state(task),
    )


def sliced_runtime(runtime: PolicyRuntime, arms: Sequence[int]) -> PolicyRuntime:
    from before_we_act.evaluate_temporal_history_policy import EpisodeHistory

    selected = tuple(int(arm) for arm in arms)
    if not set(selected).issubset(runtime.arms):
        raise ValueError("requested CARE runtime slice contains an unknown arm")
    result = PolicyRuntime(
        arms=selected,
        history=EpisodeHistory(selected),
        reference_ensembler=ConsolidatedChunkEnsembler(selected),
        base_ensembler=ConsolidatedChunkEnsembler(selected),
        watchdogs={arm: deepcopy(runtime.watchdogs[arm]) for arm in selected},
        oracle_memory=deepcopy(runtime.oracle_memory),
        step=int(runtime.step),
        last_action=(
            None
            if runtime.last_action is None
            else {
                f"panda-{arm}": np.asarray(
                    runtime.last_action[f"panda-{arm}"]
                ).copy()
                for arm in selected
            }
        ),
    )
    for arm in selected:
        result.history.visual[arm] = deepcopy(runtime.history.visual[arm])
        result.history.qpos[arm] = deepcopy(runtime.history.qpos[arm])
        result.history.actions[arm] = deepcopy(runtime.history.actions[arm])
        result.reference_ensembler.histories[arm] = deepcopy(
            runtime.reference_ensembler.histories[arm]
        )
        result.base_ensembler.histories[arm] = deepcopy(
            runtime.base_ensembler.histories[arm]
        )
    return result


def capture_snapshot(
    env: Any,
    observation: Any,
    runtime: PolicyRuntime,
    label: Mapping[str, Any],
) -> SimulatorSnapshot:
    base_env = getattr(env, "base_env", env.unwrapped)
    episode_rng = None
    if getattr(base_env, "_episode_rng", None) is not None:
        episode_rng = deepcopy(base_env._episode_rng.get_state())
    wrapper_elapsed = getattr(env, "_elapsed_steps", None)
    return SimulatorSnapshot(
        state=clone_tree(base_env.get_state_dict()),
        elapsed_steps=base_env.elapsed_steps.clone(),
        wrapper_elapsed_steps=clone_tree(wrapper_elapsed),
        observation=clone_tree(observation),
        runtime=deepcopy(runtime),
        python_rng=random.getstate(),
        numpy_rng=deepcopy(np.random.get_state()),
        torch_rng=torch.get_rng_state().clone(),
        cuda_rng=[value.clone() for value in torch.cuda.get_rng_state_all()],
        episode_rng=episode_rng,
        label=deepcopy(dict(label)),
    )


def sha256_tree(value: Any) -> str:
    """为仿真状态树生成包含键、形状、类型和值的稳定哈希。"""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, Mapping):
            digest.update(b"mapping{")
            for key in sorted(item, key=str):
                digest.update(str(key).encode("utf-8"))
                digest.update(b"=")
                update(item[key])
            digest.update(b"}")
            return
        if isinstance(item, (tuple, list)):
            digest.update(f"sequence[{len(item)}]".encode("ascii"))
            for child in item:
                update(child)
            return
        array = np.ascontiguousarray(numpy_value(item))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())

    update(value)
    return digest.hexdigest()


def seed_branch_rng(base_env: Any, seed: int) -> None:
    value = int(seed)
    random.seed(value)
    np.random.seed(value % 2**32)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    if getattr(base_env, "_episode_rng", None) is not None:
        state = np.random.RandomState(value % 2**32).get_state()
        base_env._episode_rng.set_state(state)
        if getattr(base_env, "_batched_episode_rng", None) is not None:
            base_env._batched_episode_rng[0].set_state(state)


def restore_snapshot(
    env: Any, snapshot: SimulatorSnapshot, *, branch_seed: int
) -> tuple[Any, PolicyRuntime, float, float]:
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
        deployment_observation_tree(rerendered, snapshot.runtime.arms),
        deployment_observation_tree(snapshot.observation, snapshot.runtime.arms),
    )
    restored = clone_tree(snapshot.observation)
    qpos_error = numeric_tree_max_abs(
        restored["agent"], rerendered["agent"]
    )
    seed_branch_rng(base_env, branch_seed)
    return restored, deepcopy(snapshot.runtime), qpos_error, rerender_error


def branch_seed(snapshot_id: str, repeat_id: int) -> int:
    message = (
        f"{CONTRACT_STAGE}|branch-v1|{snapshot_id}|{int(repeat_id)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "big")


def arm_is_inactive(action: np.ndarray, qpos: np.ndarray, threshold: float) -> bool:
    return float(np.linalg.norm(action[:7] - qpos[:7])) < float(threshold)


def normalized_actions(
    action: Mapping[str, np.ndarray], arms: Sequence[int], stats: Mapping[str, torch.Tensor]
) -> dict[int, torch.Tensor]:
    mean = stats["a_mean"].detach().float().cpu().numpy()
    std = stats["a_std"].detach().float().cpu().numpy()
    return {
        int(arm): torch.from_numpy(
            ((np.asarray(action[f"panda-{arm}"], dtype=np.float32) - mean) / std)
        ).float()
        for arm in arms
    }


def canonicalize_plan(
    plan: np.ndarray, action_space: Any
) -> tuple[np.ndarray, dict[str, Any]]:
    """显式复现控制器的动作边界映射，得到真正送入物理控制器的计划。"""

    value = np.asarray(plan, dtype=np.float32)
    if value.shape != (100, 8) or not np.isfinite(value).all():
        raise ValueError("CARE policy plan must be finite with shape [100,8]")
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    canonical = np.clip(value, low[None], high[None]).astype(np.float32)
    difference = np.abs(canonical.astype(np.float64) - value.astype(np.float64))
    return canonical, {
        "changed_values": int(np.count_nonzero(difference)),
        "max_abs_change": float(np.max(difference)),
    }


def canonicalize_policy_plans(
    reference: Mapping[str, np.ndarray],
    base: Mapping[str, np.ndarray],
    action_spaces: Mapping[str, Any],
    arms: Sequence[int],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    canonical_reference: dict[str, np.ndarray] = {}
    canonical_base: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    for arm in arms:
        key = f"panda-{arm}"
        canonical_reference[key], reference_diagnostics = canonicalize_plan(
            reference[key], action_spaces[key]
        )
        canonical_base[key], base_diagnostics = canonicalize_plan(
            base[key], action_spaces[key]
        )
        diagnostics[key] = {
            "reference": reference_diagnostics,
            "belief_off": base_diagnostics,
        }
    return canonical_reference, canonical_base, diagnostics


def _policy_plan_one_or_batch(
    model: Any,
    stats: Mapping[str, torch.Tensor],
    observation: Any,
    runtime: PolicyRuntime,
    task: str,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    from before_we_act.evaluate_predictive_team_belief import predict_team_belief

    chunks, base_chunks, normalized_qpos, diagnostics = predict_team_belief(
        model,
        stats,
        observation,
        runtime.arms,
        runtime.history,
        task,
        device,
    )
    reference = runtime.reference_ensembler.append_and_plan(runtime.step, chunks)
    base = runtime.base_ensembler.append_and_plan(runtime.step, base_chunks)
    qpos = (
        normalized_qpos * stats["q_std"] + stats["q_mean"]
    ).detach().float().cpu().numpy()
    reasons: dict[str, str] = {}
    for local_index, arm in enumerate(runtime.arms):
        key = f"panda-{arm}"
        use_base, reason = runtime.watchdogs[arm].choose_base(
            candidate_inactive=arm_is_inactive(
                reference[key][0], qpos[local_index],
                runtime.watchdogs[arm].config.progress_inactivity_l2,
            ),
            base_inactive=arm_is_inactive(
                base[key][0], qpos[local_index],
                runtime.watchdogs[arm].config.progress_inactivity_l2,
            ),
        )
        if use_base:
            reference[key] = base[key].copy()
        reasons[key] = reason
    diagnostics = dict(diagnostics)
    diagnostics["watchdog"] = reasons
    return reference, base, qpos, diagnostics


def _merge_runtime_arm(
    target: PolicyRuntime, source: PolicyRuntime, arm: int
) -> None:
    """把一次单机器人推理产生的状态写回完整运行时。"""

    target.history.visual[arm] = deepcopy(source.history.visual[arm])
    target.history.qpos[arm] = deepcopy(source.history.qpos[arm])
    target.history.actions[arm] = deepcopy(source.history.actions[arm])
    target.reference_ensembler.histories[arm] = deepcopy(
        source.reference_ensembler.histories[arm]
    )
    target.base_ensembler.histories[arm] = deepcopy(
        source.base_ensembler.histories[arm]
    )
    target.watchdogs[arm] = deepcopy(source.watchdogs[arm])


def policy_plan(
    model: Any,
    stats: Mapping[str, torch.Tensor],
    observation: Any,
    runtime: PolicyRuntime,
    task: str,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """按部署合同逐台机器人独立运行同一策略。"""

    if len(runtime.arms) == 1:
        return _policy_plan_one_or_batch(
            model, stats, observation, runtime, task, device
        )

    references: dict[str, np.ndarray] = {}
    bases: dict[str, np.ndarray] = {}
    qpos_rows: list[np.ndarray] = []
    diagnostic_rows: list[Mapping[str, Any]] = []
    watchdog: dict[str, str] = {}
    for arm in runtime.arms:
        per_robot = sliced_runtime(runtime, (arm,))
        reference, base, qpos, diagnostics = _policy_plan_one_or_batch(
            model, stats, observation, per_robot, task, device
        )
        key = f"panda-{arm}"
        references[key] = reference[key]
        bases[key] = base[key]
        qpos_rows.append(qpos[0])
        diagnostic_rows.append(diagnostics)
        watchdog.update(diagnostics.get("watchdog", {}))
        _merge_runtime_arm(runtime, per_robot, arm)

    combined = {
        key: float(np.mean([float(row[key]) for row in diagnostic_rows]))
        for key in ("gate", "residual_norm", "reliability", "sigma")
    }
    combined["events"] = int(
        sum(int(row["events"]) for row in diagnostic_rows)
    )
    combined["watchdog"] = watchdog
    combined["per_robot_independent_inference"] = True
    return references, bases, np.stack(qpos_rows), combined


def append_executed_action(
    runtime: PolicyRuntime,
    action: Mapping[str, np.ndarray],
    stats: Mapping[str, torch.Tensor],
) -> None:
    runtime.history.append_action(normalized_actions(action, runtime.arms, stats))
    runtime.last_action = {
        f"panda-{arm}": np.asarray(action[f"panda-{arm}"], dtype=np.float32).copy()
        for arm in runtime.arms
    }
    runtime.step += 1


def time_warp_plan(reference: np.ndarray, current_qpos: np.ndarray, scale: float, current_grip: float) -> np.ndarray:
    if reference.shape != (100, 8):
        raise ValueError("CARE time warp expects a [100,8] reference")
    arm_knots = np.concatenate((current_qpos[:7][None], reference[:, :7]), axis=0)
    grip_knots = np.concatenate((np.asarray([current_grip]), reference[:, 7]))
    result = np.empty_like(reference, dtype=np.float32)
    for index in range(100):
        tau = min((index + 1) * float(scale), 100.0)
        lower = int(math.floor(tau))
        upper = min(lower + 1, 100)
        fraction = tau - lower
        result[index, :7] = (
            (1.0 - fraction) * arm_knots[lower] + fraction * arm_knots[upper]
        )
        result[index, 7] = grip_knots[lower]
    return result


def candidate_plan(
    candidate_id: int,
    reference: np.ndarray,
    base: np.ndarray,
    current_qpos: np.ndarray,
    current_grip: float,
) -> np.ndarray:
    if candidate_id == 0:
        return reference.copy()
    if candidate_id == 1:
        return base.copy()
    if candidate_id == 2:
        hold = np.concatenate((current_qpos[:7], np.asarray([current_grip])))
        return np.concatenate((hold[None], reference[:-1]), axis=0).astype(np.float32)
    if candidate_id == 3:
        return time_warp_plan(reference, current_qpos, 0.75, current_grip)
    if candidate_id == 4:
        return time_warp_plan(reference, current_qpos, 1.25, current_grip)
    if candidate_id == 5:
        result = reference.copy()
        result[:, 7] = current_grip
        return result
    raise ValueError(f"unknown CARE candidate {candidate_id}")


def validate_candidate(
    candidate_id: int,
    plan: np.ndarray,
    reference: np.ndarray,
    base: np.ndarray,
    current_qpos: np.ndarray,
    current_grip: float,
    action_space: Any,
) -> tuple[bool, list[str]]:
    """检查一步干预真正会执行的动作；计划尾部只做有限值和边界检查。"""

    failures: list[str] = []
    if plan.shape != (100, 8) or not np.isfinite(plan).all():
        return False, ["shape_or_finite"]
    if np.any(plan < action_space.low[None] - 1e-6) or np.any(
        plan > action_space.high[None] + 1e-6
    ):
        failures.append("action_domain")
    if candidate_id in {2, 3, 4, 5}:
        reference_arm = np.concatenate(
            (current_qpos[:7][None], reference[:, :7]), axis=0
        )
        reference_rate = np.max(
            np.abs(np.diff(reference_arm, axis=0)), axis=0
        )
        candidate_rate = np.abs(plan[0, :7] - current_qpos[:7])
        if np.any(candidate_rate > 1.25 * reference_rate + 1e-5):
            failures.append("joint_rate_envelope")
    if candidate_id == 1:
        allowed_grip = base[:, 7]
    elif candidate_id == 5:
        allowed_grip = np.asarray([current_grip])
    else:
        allowed_grip = np.concatenate(
            (np.asarray([current_grip]), reference[:, 7])
        )
    if any(np.min(np.abs(allowed_grip - value)) > 1e-6 for value in plan[:, 7]):
        failures.append("gripper_domain")
    return not failures, failures


def candidate_is_active(branch_step: int, intervention_steps: int) -> bool:
    """Whether a receding-horizon candidate transform is active at this step."""

    if int(intervention_steps) < 1:
        raise ValueError("CARE intervention must contain at least one control step")
    return 0 <= int(branch_step) < int(intervention_steps)


def task_progress(task: str, label: Mapping[str, Any]) -> float:
    names = TASK_STAGE_NAMES[task]
    index = names.index(str(label["stage_id"]))
    return float(
        np.clip((index + float(label["within_stage_progress"])) / len(names), 0.0, 1.0)
    )


def step_metrics(
    task: str,
    label: Mapping[str, Any],
    action: Mapping[str, np.ndarray],
    qpos_before: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    risk = label["collision_drop_contention_risk"]
    custody = label["grasp_contact_custody_state"]
    contested = bool(risk["contested_objects"])
    missing_custody = any(
        value["current_custodian"] is None and bool(value["controller_agents"])
        for value in custody.values()
    )
    conflict = bool(risk["robot_proximity_risk"] or contested or missing_custody)

    object_roles: dict[str, list[str]] = {}
    for row in label["agent_object_role_slots"]:
        roles = [str(value) for value in row["roles"] if value != "none"]
        object_roles.setdefault(str(row["object"]), []).extend(roles)
    duplicate = any(len(roles) >= 2 for roles in object_roles.values())
    if task in {"lift_barrier", "camera_alignment", "take_photo"}:
        duplicate = False
    if str(label["stage_id"]).startswith("handoff"):
        duplicate = False

    active: list[bool] = []
    contribution = {
        int(row["agent_slot"]): bool(row["active"])
        for row in label["per_agent_contribution"]
    }
    joint_changes: list[float] = []
    for slot, key in enumerate(sorted(action, key=lambda value: int(value.split("-")[-1]))):
        change = float(
            np.linalg.norm(np.asarray(action[key])[:7] - np.asarray(qpos_before[key])[:7])
        )
        joint_changes.append(change)
        active.append(change >= 0.02 or contribution.get(slot, False))
    return {
        "progress": task_progress(task, label),
        "success": bool(label["task_complete"]),
        "collision_or_drop": bool(risk["robot_collision"] or risk["dropped_objects"]),
        "robot_conflict": conflict,
        "duplicate_work": duplicate,
        "active": active,
        "all_joint_changes_below_0_02": all(value < 0.02 for value in joint_changes),
        "stage_id": str(label["stage_id"]),
        "factorized_predicates": clone_tree(label["factorized_predicates"]),
    }


def deadlock_mask(rows: Sequence[Mapping[str, Any]]) -> list[bool]:
    stagnant = []
    previous = None
    for row in rows:
        progress = float(row["progress"])
        stagnant.append(
            previous is not None
            and abs(progress - previous) <= 1e-4
            and bool(row["all_joint_changes_below_0_02"])
            and not bool(row.get("success", False))
        )
        previous = progress
    result = [False] * len(rows)
    start = 0
    while start < len(stagnant):
        if not stagnant[start]:
            start += 1
            continue
        stop = start
        while stop < len(stagnant) and stagnant[stop]:
            stop += 1
        if stop - start >= 8:
            result[start:stop] = [True] * (stop - start)
        start = stop
    return result


def outcome_at_horizon(
    rows: Sequence[Mapping[str, Any]], start_progress: float, horizon: int
) -> dict[str, Any]:
    observed = list(rows[:horizon])
    if not observed:
        raise ValueError("CARE branch outcome has no steps")
    selected = list(observed)
    if len(selected) < horizon and bool(selected[-1]["success"]):
        terminal = deepcopy(selected[-1])
        terminal["collision_or_drop"] = False
        terminal["robot_conflict"] = False
        terminal["duplicate_work"] = False
        terminal["active"] = [False] * len(terminal["active"])
        terminal["all_joint_changes_below_0_02"] = True
        selected.extend(deepcopy(terminal) for _ in range(horizon - len(selected)))
    count = len(observed)
    deadlock = deadlock_mask(selected)
    active = np.asarray([row["active"] for row in selected], dtype=np.float64)
    active_fraction = active.mean(0)
    first_success = next(
        (index + 1 for index, row in enumerate(selected) if row["success"]), None
    )
    vector = np.asarray(
        (
            float(np.clip(float(selected[-1]["progress"]) - start_progress, -1.0, 1.0)),
            float(first_success is not None),
            -float(np.mean([row["collision_or_drop"] for row in selected])),
            -float(np.mean([row["robot_conflict"] for row in selected])),
            -float(np.mean([row["duplicate_work"] for row in selected])),
            -float(np.mean(deadlock)),
            -float(active_fraction.max() - active_fraction.min()),
            -float((first_success if first_success is not None else horizon) / horizon),
        ),
        dtype=np.float64,
    )
    hard_safety = bool(any(row["collision_or_drop"] for row in selected))
    utility = -3.0 * float(hard_safety) + float(np.dot(MAIN_WEIGHTS, vector))
    return {
        "requested_steps": int(horizon),
        "observed_steps": count,
        "bounded_utility_vector": vector.tolist(),
        "utility_main": utility,
        "hard_safety_violation": hard_safety,
        "first_success_step": first_success,
        "final_stage_id": str(selected[-1]["stage_id"]),
        "final_factorized_predicates": clone_tree(
            selected[-1]["factorized_predicates"]
        ),
        "active_fraction": active_fraction.tolist(),
    }


def current_qpos_map(observation: Any, arms: Sequence[int]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for arm in arms:
        value = numpy_value(observation["agent"][f"panda-{arm}"]["qpos"])
        result[f"panda-{arm}"] = (
            value[0].copy() if value.ndim == 2 else value.copy()
        )
    return result


def update_oracle(
    base_env: Any,
    task: str,
    success: bool,
    runtime: PolicyRuntime,
) -> Mapping[str, Any]:
    privileged = extract_oracle_snapshot(base_env, task, success)
    label = build_oracle_label(privileged, runtime.oracle_memory)
    runtime.oracle_memory = deepcopy(label["causal_automaton_state"])
    return label


def run_branch(
    *,
    env: Any,
    snapshot: SimulatorSnapshot,
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
) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
    if regime not in {"reactive", "replay"}:
        raise ValueError("CARE response regime must be reactive or replay")
    seed = branch_seed(snapshot_id, repeat_id)
    (
        observation,
        full_runtime,
        restore_observation_error,
        restore_rerender_error,
    ) = restore_snapshot(
        env,
        snapshot,
        branch_seed=seed,
    )
    runtime = (
        full_runtime
        if regime == "reactive"
        else sliced_runtime(full_runtime, (focal_arm,))
    )
    base_env = getattr(env, "base_env", env.unwrapped)
    start_progress = task_progress(task, snapshot.label)
    rows: list[dict[str, Any]] = []
    executed_rows: list[dict[str, np.ndarray]] = []
    replay_action_max_abs = 0.0
    policy_evaluations = {"focal": 0, "teammates": 0}
    candidate_valid = True
    candidate_failures: list[str] = []
    intervention_steps_applied = 0
    reference_action_error = 0.0
    reference_canonicalization_max_abs = 0.0
    base_canonicalization_max_abs = 0.0
    status = "VALID"
    started = time.perf_counter()
    for branch_step in range(horizon):
        qpos_before = current_qpos_map(observation, full_runtime.arms)
        reference, base, physical_qpos, diagnostics = policy_plan(
            model, stats, observation, runtime, task, device
        )
        reference, base, canonicalization = canonicalize_policy_plans(
            reference,
            base,
            env.action_space.spaces,
            runtime.arms,
        )
        reference_canonicalization_max_abs = max(
            reference_canonicalization_max_abs,
            max(
                float(row["reference"]["max_abs_change"])
                for row in canonicalization.values()
            ),
        )
        base_canonicalization_max_abs = max(
            base_canonicalization_max_abs,
            max(
                float(row["belief_off"]["max_abs_change"])
                for row in canonicalization.values()
            ),
        )
        diagnostics = dict(diagnostics)
        diagnostics["physical_action_canonicalization"] = canonicalization
        policy_evaluations["focal"] += 1
        if regime == "reactive":
            policy_evaluations["teammates"] += len(runtime.arms) - 1
        action = {key: value[0].copy() for key, value in reference.items()}
        if regime == "replay":
            if teammate_reference_actions is None or branch_step >= len(
                teammate_reference_actions
            ):
                status = "INVALID_REPLAY_LOG"
                break
            for arm in full_runtime.arms:
                if arm == focal_arm:
                    continue
                key = f"panda-{arm}"
                replayed = np.asarray(
                    teammate_reference_actions[branch_step][key], dtype=np.float32
                )
                action[key] = replayed.copy()
                replay_action_max_abs = max(
                    replay_action_max_abs,
                    float(
                        np.max(
                            np.abs(
                                action[key]
                                - np.asarray(teammate_reference_actions[branch_step][key])
                            )
                        )
                    ),
                )
        if candidate_is_active(branch_step, intervention_steps):
            focal_index = runtime.arms.index(focal_arm)
            key = f"panda-{focal_arm}"
            current_grip = (
                float(full_runtime.last_action[key][7])
                if full_runtime.last_action is not None
                else float(reference[key][0, 7])
            )
            transformed = candidate_plan(
                candidate_id,
                reference[key],
                base[key],
                physical_qpos[focal_index],
                current_grip,
            )
            step_valid, step_failures = validate_candidate(
                candidate_id,
                transformed,
                reference[key],
                base[key],
                physical_qpos[focal_index],
                current_grip,
                env.action_space.spaces[key],
            )
            if step_valid:
                action[key] = transformed[0].copy()
                intervention_steps_applied += 1
            else:
                candidate_valid = False
                candidate_failures.extend(
                    f"step_{branch_step}:{failure}" for failure in step_failures
                )
                action[key] = reference[key][0].copy()
                status = "INVALID_CANDIDATE"
            if candidate_id == 0:
                reference_action_error = max(
                    reference_action_error,
                    float(np.max(np.abs(reference[key][0] - action[key]))),
                )
        append_executed_action(
            runtime,
            {f"panda-{arm}": action[f"panda-{arm}"] for arm in runtime.arms},
            stats,
        )
        try:
            observation, _reward, terminated, truncated, info = env.step(action)
        except Exception as error:  # pragma: no cover - simulator-specific fatal path
            status = f"SIMULATOR_FATAL:{type(error).__name__}"
            break
        success = scalar_bool(info["success"])
        label = update_oracle(base_env, task, success, runtime)
        row = step_metrics(task, label, action, qpos_before)
        row["branch_step"] = branch_step
        row["watchdog"] = diagnostics.get("watchdog", {})
        rows.append(row)
        executed_rows.append({key: np.asarray(value).copy() for key, value in action.items()})
        if scalar_bool(terminated) or scalar_bool(truncated):
            if success:
                if status == "VALID":
                    status = "SUCCESS_TERMINATION"
            elif branch_step + 1 < horizon:
                status = "PREMATURE_TERMINATION"
            break
    outcomes = {
        str(value): outcome_at_horizon(rows, start_progress, value)
        for value in OUTCOME_HORIZONS
        if rows and value <= horizon
    }
    return (
        {
            "candidate_id": int(candidate_id),
            "regime": regime,
            "repeat_id": int(repeat_id),
            "branch_seed": int(seed),
            "status": status,
            "candidate_valid": candidate_valid,
            "candidate_failures": candidate_failures,
            "restore_observation_max_abs_error": restore_observation_error,
            "restore_rerender_diagnostic_max_abs_error": restore_rerender_error,
            "restore_observation_source": "captured_snapshot",
            "candidate0_reference_action_max_abs_error": reference_action_error,
            "reference_canonicalization_max_abs_change": (
                reference_canonicalization_max_abs
            ),
            "belief_off_canonicalization_max_abs_change": (
                base_canonicalization_max_abs
            ),
            "replay_teammate_action_max_abs_error": replay_action_max_abs,
            "policy_evaluations": policy_evaluations,
            "steps": len(rows),
            "execution_horizon_steps": int(horizon),
            "intervention_steps_requested": int(intervention_steps),
            "intervention_steps_applied": int(intervention_steps_applied),
            "intervention_protocol": (
                "recompute the CARE reference policy from the current observation and reapply the frozen "
                "candidate transform at every active control step"
            ),
            "wall_seconds": time.perf_counter() - started,
            "outcomes": outcomes,
            "executed_actions": [
                {key: np.asarray(value).tolist() for key, value in row.items()}
                for row in executed_rows
            ],
        },
        executed_rows,
    )


def freeze_common_replay_support(
    reference_branch: dict[str, Any],
    teammate_actions: Sequence[Mapping[str, np.ndarray]],
    *,
    maximum_horizon: int = 64,
) -> int:
    """Freeze the replay-supported prefix before any sibling branch is run."""

    support = int(reference_branch["steps"])
    if support != len(teammate_actions):
        raise RuntimeError("reference branch steps and teammate replay log disagree")
    if not 0 <= support <= int(maximum_horizon):
        raise RuntimeError("reference teammate replay support is outside the branch horizon")
    supported = [value for value in OUTCOME_HORIZONS if value <= support]
    reference_branch["outcomes"] = {
        str(value): reference_branch["outcomes"][str(value)]
        for value in supported
        if str(value) in reference_branch["outcomes"]
    }
    reference_branch["common_replay_support_steps"] = support
    reference_branch["supported_outcome_horizons"] = supported
    reference_branch["unsupported_outcome_horizons"] = [
        value for value in OUTCOME_HORIZONS if value > support
    ]
    return support


def annotate_common_replay_support(branch: dict[str, Any], support: int) -> None:
    supported = [value for value in OUTCOME_HORIZONS if value <= int(support)]
    branch["common_replay_support_steps"] = int(support)
    branch["supported_outcome_horizons"] = supported
    branch["unsupported_outcome_horizons"] = [
        value for value in OUTCOME_HORIZONS if value > int(support)
    ]


def reference_probe(
    *,
    env: Any,
    snapshot: SimulatorSnapshot,
    model: Any,
    stats: Mapping[str, torch.Tensor],
    task: str,
    device: torch.device,
    steps: int = 10,
) -> dict[str, Any]:
    repeats = []
    for repeat in range(2):
        observation, runtime, restore_error, rerender_error = restore_snapshot(
            env,
            snapshot,
            branch_seed=branch_seed(f"{task}|restore-probe", 0),
        )
        rows = []
        for _ in range(steps):
            reference, base, _qpos, _diagnostics = policy_plan(
                model, stats, observation, runtime, task, device
            )
            reference, _base, _canonicalization = canonicalize_policy_plans(
                reference,
                base,
                env.action_space.spaces,
                runtime.arms,
            )
            action = {key: value[0].copy() for key, value in reference.items()}
            append_executed_action(runtime, action, stats)
            observation, reward, terminated, truncated, info = env.step(action)
            rows.append(
                {
                    "action": clone_tree(action),
                    "qpos": current_qpos_map(observation, runtime.arms),
                    "observation": deployment_observation_tree(
                        observation, runtime.arms
                    ),
                    "reward": clone_tree(reward),
                    "terminated": scalar_bool(terminated),
                    "truncated": scalar_bool(truncated),
                    "success": scalar_bool(info["success"]),
                }
            )
        repeats.append(
            {
                "restore_error": restore_error,
                "rerender_error": rerender_error,
                "rows": rows,
            }
        )
    action_error = qpos_error = observation_error = reward_error = 0.0
    terminal_exact = True
    for left, right in zip(repeats[0]["rows"], repeats[1]["rows"]):
        action_error = max(action_error, numeric_tree_max_abs(left["action"], right["action"]))
        qpos_error = max(qpos_error, numeric_tree_max_abs(left["qpos"], right["qpos"]))
        observation_error = max(
            observation_error,
            numeric_tree_max_abs(left["observation"], right["observation"]),
        )
        reward_error = max(reward_error, numeric_tree_max_abs(left["reward"], right["reward"]))
        terminal_exact &= all(
            left[key] == right[key] for key in ("terminated", "truncated", "success")
        )
    tolerance = 1e-6
    return {
        "steps": steps,
        "restore_observation_max_abs_error": max(
            float(repeats[0]["restore_error"]), float(repeats[1]["restore_error"])
        ),
        "restore_rerender_diagnostic_max_abs_error": max(
            float(repeats[0]["rerender_error"]),
            float(repeats[1]["rerender_error"]),
        ),
        "restore_observation_source": "captured_snapshot",
        "reference_action_max_abs_error": action_error,
        "qpos_max_abs_error": qpos_error,
        "observation_max_abs_error": observation_error,
        "reward_max_abs_error": reward_error,
        "terminal_and_success_exact": terminal_exact,
        "tolerance": tolerance,
        "passed": bool(
            max(
                repeats[0]["restore_error"],
                repeats[1]["restore_error"],
                action_error,
                qpos_error,
                observation_error,
                reward_error,
            )
            <= tolerance
            and terminal_exact
        ),
    }


def history_arrays(
    runtime: PolicyRuntime, current_qpos_normalized: np.ndarray
) -> dict[str, np.ndarray]:
    count = len(runtime.arms)
    visual = np.zeros((count, HISTORY_STEPS, 2, 768), dtype=np.float32)
    qpos = np.zeros((count, HISTORY_STEPS, 9), dtype=np.float32)
    action = np.zeros((count, HISTORY_STEPS, 8), dtype=np.float32)
    history_mask = np.zeros((count, HISTORY_STEPS), dtype=bool)
    action_mask = np.zeros((count, HISTORY_STEPS), dtype=bool)
    for index, arm in enumerate(runtime.arms):
        visual_values = list(runtime.history.visual[arm])
        qpos_values = list(runtime.history.qpos[arm])
        if visual_values:
            first = HISTORY_STEPS - 1 - len(visual_values)
            visual[index, first:-1] = np.stack([numpy_value(value) for value in visual_values])
            qpos[index, first:-1] = np.stack([numpy_value(value) for value in qpos_values])
            history_mask[index, first:-1] = True
        qpos[index, -1] = current_qpos_normalized[index]
        history_mask[index, -1] = True
        action_values = list(runtime.history.actions[arm])
        if action_values:
            first = HISTORY_STEPS - len(action_values)
            action[index, first:] = np.stack([numpy_value(value) for value in action_values])
            action_mask[index, first:] = True
    return {
        "history_visual_raw": visual,
        "history_qpos_normalized": qpos,
        "history_action_normalized": action,
        "history_mask": history_mask,
        "action_history_mask": action_mask,
    }


def make_env(task: str, robofactory_root: Path) -> Any:
    import robofactory  # noqa: F401

    specification = get_task(task)
    return gym.make(
        specification["env_id"],
        config=str(robofactory_root / specification["config"]),
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="sensors",
        reward_mode="dense",
        sim_backend="cpu",
        sensor_configs=dict(shader_pack="default", width=640, height=480),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
    )


def collect_family(
    *,
    family: Mapping[str, Any],
    env: Any,
    model: Any,
    stats: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    checkpoint: Path,
    checkpoint_sha256: str,
    contract_sha256: str,
    device: torch.device,
    output_root: Path,
    format_version: str = FORMAT_VERSION,
    stage_id: str = "CARE-BRANCHES-PILOT",
    resource_only: bool = True,
    common_replay_support: bool = False,
) -> dict[str, Any]:
    task = str(family["task"])
    specification = get_task(task)
    arms = tuple(int(value) for value in specification["agents"])
    focal = int(family["focal_agent"])
    if focal not in arms:
        raise ValueError("pilot manifest selected an invalid focal robot")
    safety = ResidualSafetyConfig.from_mapping(config.get("residual_safety"))
    runtime = new_runtime(arms, safety, task)
    random.seed(int(family["episode_seed"]))
    np.random.seed(int(family["episode_seed"]) % 2**32)
    torch.manual_seed(int(family["episode_seed"]))
    torch.cuda.manual_seed_all(int(family["episode_seed"]))
    observation, info = env.reset(seed=int(family["episode_seed"]))
    base_env = getattr(env, "base_env", env.unwrapped)
    label = update_oracle(base_env, task, scalar_bool(info["success"]), runtime)
    for _ in range(int(family["anchor_step"])):
        reference, base, _qpos, _diagnostics = policy_plan(
            model, stats, observation, runtime, task, device
        )
        reference, _base, _canonicalization = canonicalize_policy_plans(
            reference,
            base,
            env.action_space.spaces,
            runtime.arms,
        )
        action = {key: value[0].copy() for key, value in reference.items()}
        append_executed_action(runtime, action, stats)
        observation, _reward, terminated, truncated, info = env.step(action)
        label = update_oracle(base_env, task, scalar_bool(info["success"]), runtime)
        if scalar_bool(terminated) or scalar_bool(truncated):
            raise RuntimeError("episode ended before the preregistered pilot snapshot")
    snapshot = capture_snapshot(env, observation, runtime, label)
    preview_runtime = deepcopy(runtime)
    preview_reference, preview_base, preview_qpos, diagnostics = policy_plan(
        model, stats, observation, preview_runtime, task, device
    )
    preview_reference, preview_base, preview_canonicalization = (
        canonicalize_policy_plans(
            preview_reference,
            preview_base,
            env.action_space.spaces,
            preview_runtime.arms,
        )
    )
    diagnostics = dict(diagnostics)
    diagnostics["physical_action_canonicalization"] = preview_canonicalization
    focal_index = arms.index(focal)
    focal_key = f"panda-{focal}"
    current_grip = float(runtime.last_action[focal_key][7])
    candidates = []
    for candidate_id in range(6):
        plan = candidate_plan(
            candidate_id,
            preview_reference[focal_key],
            preview_base[focal_key],
            preview_qpos[focal_index],
            current_grip,
        )
        valid, failures = validate_candidate(
            candidate_id,
            plan,
            preview_reference[focal_key],
            preview_base[focal_key],
            preview_qpos[focal_index],
            current_grip,
            env.action_space.spaces[focal_key],
        )
        candidates.append(
            {"candidate_id": candidate_id, "valid": valid, "failures": failures, "plan": plan}
        )
    snapshot_id = str(family["snapshot_id"])
    started = time.perf_counter()
    probe = reference_probe(
        env=env,
        snapshot=snapshot,
        model=model,
        stats=stats,
        task=task,
        device=device,
    )
    branches: list[dict[str, Any]] = []
    repeat_support = []
    for repeat_id in range(2):
        reference_reactive, teammate_log = run_branch(
            env=env,
            snapshot=snapshot,
            snapshot_id=snapshot_id,
            model=model,
            stats=stats,
            task=task,
            focal_arm=focal,
            candidate_id=0,
            regime="reactive",
            repeat_id=repeat_id,
            teammate_reference_actions=None,
            device=device,
        )
        if common_replay_support:
            support = freeze_common_replay_support(
                reference_reactive, teammate_log
            )
            repeat_support.append(
                {
                    "repeat_id": repeat_id,
                    "common_replay_support_steps": support,
                    "supported_outcome_horizons": [
                        value for value in OUTCOME_HORIZONS if value <= support
                    ],
                    "unsupported_outcome_horizons": [
                        value for value in OUTCOME_HORIZONS if value > support
                    ],
                }
            )
        else:
            support = 64
        branches.append(reference_reactive)
        reference_replay, _ = run_branch(
            env=env,
            snapshot=snapshot,
            snapshot_id=snapshot_id,
            model=model,
            stats=stats,
            task=task,
            focal_arm=focal,
            candidate_id=0,
            regime="replay",
            repeat_id=repeat_id,
            teammate_reference_actions=teammate_log,
            device=device,
            horizon=support,
        )
        if common_replay_support:
            annotate_common_replay_support(reference_replay, support)
        branches.append(reference_replay)
        for candidate_id in range(1, 6):
            reactive, _ = run_branch(
                env=env,
                snapshot=snapshot,
                snapshot_id=snapshot_id,
                model=model,
                stats=stats,
                task=task,
                focal_arm=focal,
                candidate_id=candidate_id,
                regime="reactive",
                repeat_id=repeat_id,
                teammate_reference_actions=None,
                device=device,
                horizon=support,
            )
            if common_replay_support:
                annotate_common_replay_support(reactive, support)
            branches.append(reactive)
        for candidate_id in range(1, 6):
            replay, _ = run_branch(
                env=env,
                snapshot=snapshot,
                snapshot_id=snapshot_id,
                model=model,
                stats=stats,
                task=task,
                focal_arm=focal,
                candidate_id=candidate_id,
                regime="replay",
                repeat_id=repeat_id,
                teammate_reference_actions=teammate_log,
                device=device,
                horizon=support,
            )
            if common_replay_support:
                annotate_common_replay_support(replay, support)
            branches.append(replay)
    fidelity = []
    for repeat_id in range(2):
        reactive = next(
            row for row in branches
            if row["repeat_id"] == repeat_id and row["candidate_id"] == 0 and row["regime"] == "reactive"
        )
        replay = next(
            row for row in branches
            if row["repeat_id"] == repeat_id and row["candidate_id"] == 0 and row["regime"] == "replay"
        )
        common = sorted(set(reactive["outcomes"]) & set(replay["outcomes"]), key=int)
        maximum = max(
            (
                abs(
                    float(reactive["outcomes"][key]["utility_main"])
                    - float(replay["outcomes"][key]["utility_main"])
                )
                for key in common
            ),
            default=math.inf,
        )
        fidelity.append({"repeat_id": repeat_id, "utility_max_abs_error": maximum})
    legal_arrays = history_arrays(
        preview_runtime,
        ((torch.as_tensor(preview_qpos, device=device) - stats["q_mean"]) / stats["q_std"])
        .detach().float().cpu().numpy(),
    )
    legal_arrays["candidate_chunks"] = np.stack([row["plan"] for row in candidates])
    legal_arrays["focal_agent"] = np.asarray([focal], dtype=np.int64)
    npz_path = output_root / task / f"{snapshot_id}.npz"
    json_path = output_root / task / f"{snapshot_id}.json"
    atomic_npz(npz_path, **legal_arrays)
    result = {
        "format_version": format_version,
        "stage_id": stage_id,
        "resource_only": resource_only,
        "forbidden_uses": (
            ["CARE training", "Gate A", "Gate B", "candidate revision"]
            if resource_only
            else ["CARE training before formal Gate A and Gate B pass"]
        ),
        "snapshot_id": snapshot_id,
        "task": task,
        "episode_seed": int(family["episode_seed"]),
        "anchor_step": int(family["anchor_step"]),
        "focal_agent": focal,
        "split": family.get("split"),
        "sampling_stratum": family.get("sampling_stratum"),
        "scenario_group_id": family.get("scenario_group_id"),
        "source_scan_id": family.get("source_scan_id"),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "contract_sha256": contract_sha256,
        "snapshot_state_sha256": sha256_tree(snapshot.state),
        "legal_history_artifact": str(npz_path.resolve()),
        "candidate_legality": [
            {key: value for key, value in row.items() if key != "plan"}
            for row in candidates
        ],
        "snapshot_label": {
            "stage_id": label["stage_id"],
            "within_stage_progress": label["within_stage_progress"],
            "factorized_predicates": label["factorized_predicates"],
        },
        "prebranch_diagnostics": diagnostics,
        "restore_probe": probe,
        "reference_reactive_replay_fidelity": fidelity,
        "branches": branches,
        "branch_count": len(branches),
        "wall_seconds": time.perf_counter() - started,
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    if common_replay_support:
        result["common_replay_support_protocol"] = (
            "candidate-0 reactive log length freezes the sibling execution prefix"
        )
        result["repeat_common_replay_support"] = repeat_support
    atomic_json(json_path, result)
    result["json_sha256"] = sha256_file(json_path)
    result["npz_sha256"] = sha256_file(npz_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", default="")
    parser.add_argument("--snapshot-id", default="")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--robofactory-root", type=Path, default=Path("/workspace/RoboFactory"))
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    from before_we_act.evaluate_predictive_team_belief import load_team_belief

    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    manifest_stage = str(manifest.get("stage_id"))
    common_replay_support = False
    if manifest_stage == "CARE-BRANCHES-PILOT":
        expected_status = "FROZEN_BEFORE_OUTCOME"
        expected_contract_stage = CONTRACT_STAGE
        output_format = FORMAT_VERSION
        resource_only = True
    elif manifest_stage == FORMAL_MANIFEST_STAGE:
        expected_status = "FROZEN_BEFORE_FORMAL_BRANCH_OUTCOME"
        expected_contract_stage = FORMAL_CONTRACT_STAGE
        output_format = FORMAL_FORMAT_VERSION
        resource_only = False
    elif manifest_stage == GATE_FIRST_MANIFEST_STAGE:
        expected_status = "FROZEN_BEFORE_GATE_BRANCH_OUTCOME"
        expected_contract_stage = GATE_FIRST_CONTRACT_STAGE
        output_format = GATE_FIRST_FORMAT_VERSION
        resource_only = False
    elif manifest_stage == COMPACT_MANIFEST_STAGE:
        expected_status = "FROZEN_BEFORE_GATE_BRANCH_OUTCOME"
        expected_contract_stage = COMPACT_CONTRACT_STAGE
        output_format = COMPACT_FORMAT_VERSION
        resource_only = False
    elif manifest_stage == COMMON_SUPPORT_MANIFEST_STAGE:
        expected_status = "FROZEN_BEFORE_GATE_BRANCH_OUTCOME"
        expected_contract_stage = COMMON_SUPPORT_CONTRACT_STAGE
        output_format = COMMON_SUPPORT_FORMAT_VERSION
        resource_only = False
        common_replay_support = True
    else:
        raise RuntimeError("unsupported CARE branch manifest stage")
    if manifest.get("status") != expected_status:
        raise RuntimeError("CARE branch manifest is not frozen")
    if contract.get("stage_id") != expected_contract_stage:
        raise RuntimeError("wrong CARE contract stage")
    parent = contract.get("parent_contract", {})
    parent_path = Path(str(parent.get("path", "")))
    if not parent_path.is_absolute():
        parent_path = Path.cwd() / parent_path
    if not parent_path.is_file() or sha256_file(parent_path) != parent.get("sha256"):
        raise RuntimeError("CARE parent contract is missing or drifted")
    executed_intervention_steps = (
        contract.get("candidate_family_revision", {})
        .get("legality", {})
        .get("executed_intervention_steps")
    )
    if executed_intervention_steps is None:
        executed_intervention_steps = contract.get(
            "candidate_and_branch_contract", {}
        ).get("executed_intervention_steps")
    if executed_intervention_steps != 1:
        raise RuntimeError("CARE one-step legality contract drifted")
    contract_sha = sha256_file(args.contract)
    checkpoint_sha = sha256_file(args.checkpoint)
    if contract_sha != manifest["contract_sha256"]:
        raise RuntimeError("CARE contract hash drifted")
    if checkpoint_sha != contract["reference_policy"]["checkpoint_sha256"]:
        raise RuntimeError("CARE reference deployment checkpoint hash drifted")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    task_families = [
        row
        for index, row in enumerate(manifest["families"])
        if index % args.shard_count == args.shard_index
        and (not args.task or row["task"] == args.task)
    ]
    if args.snapshot_id:
        task_families = [
            row for row in task_families if row["snapshot_id"] == args.snapshot_id
        ]
    if not task_families:
        raise ValueError("CARE manifest contains no requested family")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(min(12, os.cpu_count() or 12))
    model, stats, config = load_team_belief(str(args.checkpoint), device)
    for task in sorted({str(row["task"]) for row in task_families}):
        env = make_env(task, args.robofactory_root)
        try:
            for family in (row for row in task_families if row["task"] == task):
                output = args.output_root / task / f"{family['snapshot_id']}.json"
                if output.is_file():
                    existing = json.loads(output.read_text(encoding="utf-8"))
                    if (
                        existing.get("format_version") == output_format
                        and existing.get("checkpoint_sha256") == checkpoint_sha
                        and existing.get("contract_sha256") == contract_sha
                        and existing.get("branch_count") == 24
                    ):
                        print(json.dumps({"snapshot_id": family["snapshot_id"], "reused": True}), flush=True)
                        continue
                    raise RuntimeError(f"refusing to overwrite inconsistent family: {output}")
                result = collect_family(
                    family=family,
                    env=env,
                    model=model,
                    stats=stats,
                    config=config,
                    checkpoint=args.checkpoint,
                    checkpoint_sha256=checkpoint_sha,
                    contract_sha256=contract_sha,
                    device=device,
                    output_root=args.output_root,
                    format_version=output_format,
                    stage_id=manifest_stage,
                    resource_only=resource_only,
                    common_replay_support=common_replay_support,
                )
                print(
                    json.dumps(
                        {
                            "snapshot_id": result["snapshot_id"],
                            "task": result["task"],
                            "branch_count": result["branch_count"],
                            "legacy_restore_probe_passed": result["restore_probe"]["passed"],
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
