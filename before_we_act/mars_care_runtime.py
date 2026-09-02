"""MARS-Control boundary for the unchanged RoboFactory CARE policy stack.

This module owns benchmark-specific observation extraction, temporal runtime
state, physical action decoding, and privileged task progress.  It does not
define a policy, candidate family, scorer, or loss.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import itertools
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from before_we_act.mars_action_contract import canonicalize_action
from before_we_act.care_branch_collector import ConsolidatedChunkEnsembler
from before_we_act.evaluate_mars_predictive_team_belief import load_policy
from before_we_act.evaluate_mars_temporal_policy import LocalHistory
from deployment.mars_care.common import TASK_BY_NAME, local_observation, make_env


MARS_TASKS = tuple(TASK_BY_NAME)


def as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def scalar(value: Any) -> bool:
    return bool(as_numpy(value).all())


@dataclass
class MarsCARERuntime:
    arms: tuple[int, ...]
    history: LocalHistory
    reference_ensembler: ConsolidatedChunkEnsembler
    base_ensembler: ConsolidatedChunkEnsembler
    step: int = 0
    last_action: dict[str, np.ndarray] | None = None


def new_runtime(arms: Sequence[int]) -> MarsCARERuntime:
    selected = tuple(int(arm) for arm in arms)
    return MarsCARERuntime(
        arms=selected,
        history=LocalHistory(selected),
        reference_ensembler=ConsolidatedChunkEnsembler(selected),
        base_ensembler=ConsolidatedChunkEnsembler(selected),
    )


def sliced_runtime(runtime: MarsCARERuntime, arms: Sequence[int]) -> MarsCARERuntime:
    selected = tuple(int(arm) for arm in arms)
    if not set(selected).issubset(runtime.arms):
        raise ValueError("MARS CARE runtime slice contains an unknown arm")
    result = new_runtime(selected)
    result.step = int(runtime.step)
    result.last_action = (
        None
        if runtime.last_action is None
        else {
            f"panda-{arm}": np.asarray(runtime.last_action[f"panda-{arm}"]).copy()
            for arm in selected
        }
    )
    for arm in selected:
        result.history.visual[arm] = deepcopy(runtime.history.visual[arm])
        result.history.qpos[arm] = deepcopy(runtime.history.qpos[arm])
        result.history.action[arm] = deepcopy(runtime.history.action[arm])
        result.reference_ensembler.histories[arm] = deepcopy(
            runtime.reference_ensembler.histories[arm]
        )
        result.base_ensembler.histories[arm] = deepcopy(
            runtime.base_ensembler.histories[arm]
        )
    return result


@torch.inference_mode()
def policy_plan(
    model: Any,
    stats: Mapping[str, torch.Tensor],
    observation: Mapping[str, Any],
    runtime: MarsCARERuntime,
    task: str,
    device: torch.device,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    torch.Tensor,
    torch.Tensor,
    dict[str, float],
]:
    """Run shared weights on a batch of strictly arm-local inputs.

    Batch rows never attend to each other.  Batching is only a GPU scheduling
    optimization and is equivalent to deploying one copy per robot.
    """

    images: list[np.ndarray] = []
    qposes: list[np.ndarray] = []
    for arm in runtime.arms:
        image, qpos = local_observation(dict(observation), arm)
        if image.shape != (240, 320, 3):
            raise ValueError(f"MARS CARE RGB drift: {image.shape}")
        images.append(image)
        qposes.append(qpos.reshape(-1))
    qpos = torch.as_tensor(np.stack(qposes), device=device).float()
    qnorm = (qpos - stats["q_mean"]) / stats["q_std"]
    rgb = (
        torch.as_tensor(np.stack(images), device=device)
        .permute(0, 3, 1, 2)
        .float()
        .div_(255)
    )
    temporal = runtime.history.batch(qnorm, task, device, True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output = model(rgb, rgb, **temporal)
    runtime.history.append_observation(output.current_visual_raw, qnorm)
    reference_chunks = (
        output.prediction * stats["a_std"] + stats["a_mean"]
    ).float().cpu().numpy()
    base_chunks = (
        output.base_prediction * stats["a_std"] + stats["a_mean"]
    ).float().cpu().numpy()
    reference = runtime.reference_ensembler.append_and_plan(runtime.step, reference_chunks)
    base = runtime.base_ensembler.append_and_plan(runtime.step, base_chunks)
    memory = torch.cat((output.belief.mu, output.belief.event_memory), dim=1).float()
    memory_mask = torch.cat(
        (
            torch.ones(
                output.belief.mu.shape[:2],
                dtype=torch.bool,
                device=output.belief.mu.device,
            ),
            output.belief.event_mask,
        ),
        dim=1,
    )
    diagnostics = {
        "gate": float(output.residual_gate.float().mean()),
        "residual_norm": float(output.belief_residual.float().norm(dim=-1).mean()),
        "reliability": float(output.belief.reliability.float().mean()),
        "sigma": float(output.belief.sigma.float().mean()),
        "events": float(output.belief.event_mask.sum()),
    }
    return reference, base, qpos.float().cpu().numpy(), memory, memory_mask, diagnostics


def append_action(
    runtime: MarsCARERuntime,
    action: Mapping[str, np.ndarray],
    stats: Mapping[str, torch.Tensor],
) -> None:
    canonical = {
        int(arm): canonicalize_action(action[f"panda-{arm}"])
        for arm in runtime.arms
    }
    normalized = {
        arm: (
            torch.as_tensor(canonical[arm], device=stats["a_mean"].device)
            - stats["a_mean"]
        )
        / stats["a_std"]
        for arm in runtime.arms
    }
    runtime.history.append_action(normalized)
    runtime.last_action = {
        f"panda-{arm}": canonical[arm].copy()
        for arm in runtime.arms
    }
    runtime.step += 1


def current_qpos(observation: Mapping[str, Any], arms: Sequence[int]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for arm in arms:
        value = as_numpy(observation["agent"][f"panda-{arm}"]["qpos"])
        result[f"panda-{arm}"] = value[0].copy() if value.ndim == 2 else value.copy()
    return result


def local_observation_tree(
    observation: Mapping[str, Any], arms: Sequence[int]
) -> dict[str, Any]:
    return {
        "agent": {
            f"panda-{arm}": {
                "qpos": current_qpos(observation, (arm,))[f"panda-{arm}"]
            }
            for arm in arms
        },
        "sensor_data": {
            f"head_camera_agent{arm}": {
                "rgb": local_observation(dict(observation), arm)[0].copy()
            }
            for arm in arms
        },
    }


def _position(value: Any) -> np.ndarray:
    result = as_numpy(value)
    if result.ndim > 1:
        result = result[0]
    return result.astype(np.float64, copy=False)


def _tcp_positions(base_env: Any) -> list[np.ndarray]:
    return [_position(agent.tcp.pose.p) for agent in base_env.agent.agents]


def _proximity_conflict(base_env: Any) -> bool:
    rows = _tcp_positions(base_env)
    return any(np.linalg.norm(left - right) < 0.06 for left, right in itertools.combinations(rows, 2))


def _active_rows(
    action: Mapping[str, np.ndarray], qpos_before: Mapping[str, np.ndarray]
) -> tuple[list[bool], list[float]]:
    changes = [
        float(np.linalg.norm(np.asarray(action[key])[:7] - np.asarray(qpos_before[key])[:7]))
        for key in sorted(action, key=lambda value: int(value.split("-")[-1]))
    ]
    return [value >= 0.02 for value in changes], changes


def privileged_task_metrics(
    base_env: Any,
    task: str,
    action: Mapping[str, np.ndarray],
    qpos_before: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Map official MARS task state into CARE's bounded outcome interface.

    These predicates are training-label only.  No value from this function is
    accepted by the deployed reference policy or CARE scorer.
    """

    info = base_env.evaluate()
    success = scalar(info.get("success", False))
    predicates: dict[str, Any]
    hard_safety = False
    duplicate = False
    if task == "place_cube_in_cup":
        cube = _position(base_env.cube.pose.p)
        cup = _position(base_env.cup.pose.p)
        horizontal = float(np.linalg.norm(cube[:2] - cup[:2]))
        horizontal_score = float(np.clip(1.0 - horizontal / 0.95, 0.0, 1.0))
        vertical_score = float(np.clip((cube[2] - cup[2] + 0.05) / 0.15, 0.0, 1.0))
        rotation = scalar(info.get("valid_rotation", True))
        progress = 1.0 if success else 0.55 * horizontal_score + 0.30 * vertical_score + 0.15 * float(rotation)
        hard_safety = cube[2] < -0.02 or not rotation
        stage = "success" if success else ("aligned" if horizontal < 0.08 else "approach")
        predicates = {
            "horizontal_distance": horizontal,
            "vertical_position": float(cube[2]),
            "cup_position": float(cup[2]),
            "valid_rotation": rotation,
            "success": success,
        }
    elif task == "strike_cube_hard":
        cube = _position(base_env.cube.pose.p)
        goal = _position(base_env.goal_region.pose.p)
        matrix = np.asarray(base_env.annotation_data["hammer"]["functional_matrix"][0], dtype=np.float64).copy()
        scale = np.asarray(base_env.annotation_data["hammer"]["scale"], dtype=np.float64).reshape(-1)
        matrix[:3, 3] *= float(scale[0])
        hammer_matrix = as_numpy(base_env.hammer.pose.to_transformation_matrix())
        if hammer_matrix.ndim == 3:
            hammer_matrix = hammer_matrix[0]
        functional = (hammer_matrix @ matrix)[:3, 3]
        hammer_distance = float(np.linalg.norm(cube - functional))
        goal_distance = float(np.linalg.norm(cube - goal))
        contact_score = float(np.clip(1.0 - hammer_distance / 0.70, 0.0, 1.0))
        goal_score = float(np.clip(1.0 - goal_distance / 1.20, 0.0, 1.0))
        progress = 1.0 if success else 0.45 * contact_score + 0.55 * goal_score
        hard_safety = cube[2] < -0.02 or _position(base_env.hammer.pose.p)[2] < -0.05
        stage = "success" if success else ("struck" if hammer_distance < 0.05 else "approach")
        predicates = {
            "hammer_functional_point_distance": hammer_distance,
            "cube_goal_distance": goal_distance,
            "success": success,
        }
    elif task == "three_robots_place_shoes":
        left = _position(base_env.shoe_left.pose.p)
        right = _position(base_env.shoe_right.pose.p)
        box = _position(base_env.shoe_box.pose.p)
        left_distance = float(np.linalg.norm(left[:2] - box[:2]))
        right_distance = float(np.linalg.norm(right[:2] - box[:2]))
        left_in = scalar(info["shoe_left_in_box"])
        right_in = scalar(info["shoe_right_in_box"])
        lid = scalar(info["lid_on_box"])
        left_grasped = scalar(info["is_shoe_left_grasped"])
        right_grasped = scalar(info["is_shoe_right_grasped"])
        approach = 0.5 * float(np.clip(1.0 - left_distance / 1.0, 0.0, 1.0)) + 0.5 * float(np.clip(1.0 - right_distance / 1.0, 0.0, 1.0))
        progress = 1.0 if success else 0.20 * approach + 0.25 * float(left_in) + 0.25 * float(right_in) + 0.20 * float(lid) + 0.10 * float(not left_grasped and not right_grasped)
        hard_safety = left[2] < -0.03 or right[2] < -0.03
        stage = "success" if success else ("box" if left_in or right_in else "approach")
        predicates = {
            "shoe_left_in_box": left_in,
            "shoe_right_in_box": right_in,
            "lid_on_box": lid,
            "is_shoe_left_grasped": left_grasped,
            "is_shoe_right_grasped": right_grasped,
            "success": success,
        }
    elif task == "four_robots_stack_cube":
        cube_a = _position(base_env.cubeA.pose.p)
        cube_b = _position(base_env.cubeB.pose.p)
        goal = _position(base_env.goal_region.pose.p)
        offset = cube_b - cube_a
        xy_distance = float(np.linalg.norm(offset[:2]))
        z_error = float(abs(offset[2] - 0.04))
        goal_distance = float(np.linalg.norm(cube_b[:2] - goal[:2]))
        stack_score = 0.5 * float(np.clip(1.0 - xy_distance / 0.40, 0.0, 1.0)) + 0.5 * float(np.clip(1.0 - z_error / 0.30, 0.0, 1.0))
        goal_score = float(np.clip(1.0 - goal_distance / 1.20, 0.0, 1.0))
        on_stack = scalar(info["is_cubeA_on_cubeB"])
        placed = scalar(info["cubeB_placed"])
        grasp_flags = [
            scalar(info["is_cubeA_grasped_1"]),
            scalar(info["is_cubeB_grasped_1"]),
            scalar(info["is_cubeA_grasped_2"]),
            scalar(info["is_cubeB_grasped_2"]),
        ]
        progress = 1.0 if success else 0.30 * stack_score + 0.20 * goal_score + 0.25 * float(on_stack) + 0.15 * float(placed) + 0.10 * float(any(grasp_flags))
        hard_safety = cube_a[2] < -0.03 or cube_b[2] < -0.03
        duplicate = sum(grasp_flags[::2]) > 1 or sum(grasp_flags[1::2]) > 1
        stage = "success" if success else ("stacked" if on_stack else "approach")
        predicates = {
            "cube_stack_xy_distance": xy_distance,
            "cube_stack_z_error": z_error,
            "cube_goal_distance": goal_distance,
            # RoboFactory's official evaluate() key is is_cubeA_on_cubeB.
            # Keep the same orientation here; the old reversed recorder label
            # made video/telemetry interpretation needlessly ambiguous.
            "is_cubeA_on_cubeB": on_stack,
            "cubeB_placed": placed,
            "grasp_flags": grasp_flags,
            "success": success,
        }
    else:
        raise KeyError(f"unknown MARS CARE task {task}")
    active, changes = _active_rows(action, qpos_before)
    return {
        "progress": float(np.clip(progress, 0.0, 1.0)),
        "success": success,
        "collision_or_drop": bool(hard_safety),
        "robot_conflict": _proximity_conflict(base_env),
        "duplicate_work": bool(duplicate),
        "active": active,
        "all_joint_changes_below_0_02": all(value < 0.02 for value in changes),
        "stage_id": stage,
        "factorized_predicates": predicates,
    }


def load_reference(checkpoint: Path, device: torch.device):
    return load_policy(checkpoint, device)


def environment(task: str, root: Path, render_device: str | None = None):
    return make_env(TASK_BY_NAME[task], root, render_device)


__all__ = [
    "MARS_TASKS",
    "MarsCARERuntime",
    "append_action",
    "current_qpos",
    "environment",
    "load_reference",
    "local_observation_tree",
    "new_runtime",
    "policy_plan",
    "privileged_task_metrics",
    "scalar",
    "sliced_runtime",
]
