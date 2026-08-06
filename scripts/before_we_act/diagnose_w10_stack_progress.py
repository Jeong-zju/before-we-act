#!/usr/bin/env python3
"""Measure where W10 (or a forced W10 role) fails ThreeRobotsStackCube.

This is a diagnostic only: simulator state is used for metrics and is never
passed to the policy.  Policy inference is exactly the frozen W10 image/qpos
path, with an optional fixed ARCA role used to measure candidate-bank headroom.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import gymnasium as gym
import numpy as np
import torch

import robofactory  # noqa: F401

from stereo_core.evaluate_no_wrist_pair import (
    TemporalChunkEnsembler,
    denormalize_action_chunks,
    load_model,
    prepare_no_wrist_batch,
    reset_reproducibly,
)
from stereo_core.two_three_task_manifest import get_task


MILESTONES = (
    "is_cubeA_grasped",
    "is_cubeB_grasped",
    "is_cubeC_grasped",
    # RoboFactory currently exposes these two historical key names.  They
    # mean B-on-A and C-on-B respectively (see task evaluate()).
    "is_cubeA_on_cubeB",
    "is_cubeC_on_cubeA",
    "cubeB_placed",
    "success",
)


def scalar_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return bool(array.astype(bool).all())


def vector(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1, 3)[0]


def scene_metrics(env) -> dict[str, float]:
    raw = env.unwrapped
    a = vector(raw.cubeA.pose.p)
    b = vector(raw.cubeB.pose.p)
    c = vector(raw.cubeC.pose.p)
    goal = vector(raw.goal_region.pose.p)
    return {
        "cubeA_goal_xy": float(np.linalg.norm(a[:2] - goal[:2])),
        "cubeB_goal_xy": float(np.linalg.norm(b[:2] - goal[:2])),
        "cubeC_goal_xy": float(np.linalg.norm(c[:2] - goal[:2])),
        "cubeBA_xy": float(np.linalg.norm((b - a)[:2])),
        "cubeCB_xy": float(np.linalg.norm((c - b)[:2])),
        "cubeBA_z_error": float(abs((b - a)[2] - 0.04)),
        "cubeCB_z_error": float(abs((c - b)[2] - 0.04)),
    }


@torch.no_grad()
def predict(model, stats, observation, arms, device, role: int | None):
    global_rgb, local_rgb, qpos = prepare_no_wrist_batch(
        observation, arms, stats, device
    )
    with torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        if role is None:
            chunks = model(global_rgb, local_rgb, qpos)[0]
        else:
            context = model.encode_context(global_rgb, local_rgb, qpos)
            gates = context.observation.new_zeros(
                (len(arms), model.horizon, model.roles_n)
            )
            gates[..., role] = 1
            chunks = model.decode_with_gates(context, gates)
    return denormalize_action_chunks(chunks, stats).float().cpu().numpy()


def update_first(first: dict[str, int | None], info, step: int) -> None:
    for key in MILESTONES:
        if first[key] is None and key in info and scalar_bool(info[key]):
            first[key] = step


def diagnostic_row(
    model,
    stats,
    env,
    seed: int,
    arms,
    device,
    role: int | None,
    max_steps: int,
) -> dict[str, Any]:
    observation, _ = reset_reproducibly(env, seed)
    initial = scene_metrics(env)
    minimum = dict(initial)
    first: dict[str, int | None] = {key: None for key in MILESTONES}
    ever = {key: False for key in MILESTONES}
    true_steps = {key: 0 for key in MILESTONES}
    ensembler = TemporalChunkEnsembler(arms)
    action_l2: list[list[float]] = [[] for _ in arms]
    action_max_abs: list[list[float]] = [[] for _ in arms]
    gripper: list[list[float]] = [[] for _ in arms]
    info: dict[str, Any] = {}
    success = False

    for step in range(max_steps):
        chunks = predict(model, stats, observation, arms, device, role)
        action = ensembler.append_and_select(step, chunks)
        for local_index, arm in enumerate(arms):
            value = np.asarray(action[f"panda-{arm}"], dtype=np.float64)
            action_l2[local_index].append(float(np.linalg.norm(value[:7])))
            action_max_abs[local_index].append(float(np.max(np.abs(value[:7]))))
            gripper[local_index].append(float(value[7]))
        observation, _, terminated, truncated, info = env.step(action)
        update_first(first, info, step + 1)
        for key in MILESTONES:
            flag = key in info and scalar_bool(info[key])
            ever[key] = ever[key] or flag
            true_steps[key] += int(flag)
        current = scene_metrics(env)
        for key, value in current.items():
            minimum[key] = min(minimum[key], value)
        success = scalar_bool(info.get("success", False))
        if success or scalar_bool(terminated) or scalar_bool(truncated):
            break

    final = scene_metrics(env)
    per_agent = []
    for local_index, arm in enumerate(arms):
        grip = np.asarray(gripper[local_index])
        sign_changes = int(np.count_nonzero(np.diff(np.signbit(grip))))
        per_agent.append(
            {
                "arm": int(arm),
                "joint_action_l2_mean": float(np.mean(action_l2[local_index])),
                "joint_action_max_abs": float(np.max(action_max_abs[local_index])),
                "gripper_mean": float(np.mean(grip)),
                "gripper_min": float(np.min(grip)),
                "gripper_max": float(np.max(grip)),
                "gripper_sign_changes": sign_changes,
            }
        )
    return {
        "seed": int(seed),
        "mode": "native" if role is None else f"forced_role_{role}",
        "success": success,
        "steps": step + 1,
        "milestone_ever": ever,
        "milestone_first_step": first,
        "milestone_true_steps": true_steps,
        "scene_initial": initial,
        "scene_minimum": minimum,
        "scene_final": final,
        "per_agent_action": per_agent,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--role", type=int, choices=range(4))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    seed_path = Path(args.seed_file).resolve(strict=True)
    seed_bytes = seed_path.read_bytes()
    seeds = [int(value) for value in json.loads(seed_bytes)["seeds"]]
    if args.episodes < 1 or args.episodes > len(seeds):
        raise ValueError("episodes exceed the seed manifest")
    seeds = seeds[: args.episodes]
    device = torch.device(args.device)
    model, stats, config = load_model(args.checkpoint, device)
    specification = get_task("three_robots_stack_cube")
    arms = specification["agents"]
    env = gym.make(
        specification["env_id"],
        config=f"/workspace/RoboFactory/{specification['config']}",
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="sensors",
        reward_mode="dense",
        sim_backend="cpu",
        sensor_configs=dict(
            shader_pack="default", width=640, height=480
        ),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
    )
    rows = []
    try:
        for seed in seeds:
            row = diagnostic_row(
                model,
                stats,
                env,
                seed,
                arms,
                device,
                args.role,
                args.max_steps,
            )
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        env.close()

    payload = {
        "schema_version": 1,
        "diagnostic": "W10 ThreeRobotsStackCube causal stage progress",
        "checkpoint": str(Path(args.checkpoint).resolve(strict=True)),
        "checkpoint_policy_input": config.get("policy_input"),
        "mode": "native" if args.role is None else f"forced_role_{args.role}",
        "episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "milestone_episode_counts": {
            key: sum(row["milestone_ever"][key] for row in rows)
            for key in MILESTONES
        },
        "seed_protocol": {
            "source": str(seed_path),
            "sha256": hashlib.sha256(seed_bytes).hexdigest(),
        },
        "privileged_state_policy_input": False,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload | {"rows": "saved"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
