"""Collect decentralized FE-PC-WAM episodes.

Unlike the legacy collector, this module writes an initial observation and then
records every transition as ``o_t, a_t, o_{t+1}``.  Simulator truth is passed to
``LocalObservationSimulator`` to create noisy/occluded measurements and is
otherwise stored only below the privileged HDF5 group.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from tqdm import trange

from data.local_observation import (
    LocalObservationSimulator,
    LocalObservationSpec,
    PrivilegedAgentState,
    SensorSimulationConfig,
    ego_relative_pose,
    stack_packets,
)
from data.policies import ScriptedPolicy
from data.schema import failure_to_id, phase_to_id
from data.schema import (
    LOCAL_FORCE_UNITS,
    SCHEMA_VERSION,
    STRICT_LOCAL_CONTACT_SEMANTICS,
    STRICT_LOCAL_FORCE_SEMANTICS,
    STRICT_LOCAL_SENSOR_PROVENANCE,
    Episode,
    save_episode,
)
from data.research_v2 import MatchedBranchGroup
from envs.two_robot_carry_env import CarryEnvConfig, TwoRobotCarryNarrowPassageEnv


def collect_one_episode(
    env: TwoRobotCarryNarrowPassageEnv,
    policy: ScriptedPolicy,
    seed: int,
    *,
    randomize: bool = True,
    sensor_config: SensorSimulationConfig | None = None,
    collect_matched_branches: bool = False,
) -> tuple[Episode, Dict[str, Any], LocalObservationSpec]:
    """Collect one episode with strict transition alignment."""

    spec = LocalObservationSpec(joint_dim=0, force_dim=1)
    sensor = LocalObservationSimulator(
        spec=spec,
        config=sensor_config or SensorSimulationConfig(control_dt=env.cfg.control_dt),
        seed=seed,
    )
    sensor.reset(seed=seed)
    obs = env.reset(seed=seed, randomize=randomize)

    packets: Dict[int, list] = {0: [], 1: []}
    actions: Dict[int, list] = {0: [], 1: []}
    privileged_obs: Dict[str, list] = {
        "time": [],
        "progress": [],
        "robot_pose_world": [],
        "object_pose_world": [],
        "object_pose_ego": [],
        "teammate_pose_ego": [],
        "base_twist_ego": [],
        "global_state": [],
        "private_event_truth": [],
        "private_event_cue_agents": [],
        "private_event_valid_agents": [],
        "next_gate_context_agents": [],
    }
    privileged_tr: Dict[str, list] = {
        "environment_action_world": [],
        "reward": [],
        "done": [],
        "success": [],
        "failure": [],
        "failure_reason": [],
        "phase": [],
        "progress": [],
        "force_proxy": [],
        "contact": [],
        "grasp": [],
        "private_event_type": [],
        "private_event_informed_agent": [],
        "private_event_maneuver": [],
        "private_event_error": [],
        "branch_valid": [],
        "branch_plan_pair": [],
        "branch_action": [],
        "branch_return": [],
        "branch_success": [],
        "branch_constraint_violation": [],
    }

    base_twists = np.zeros((2, 3), dtype=np.float32)
    local_grasps = np.zeros(2, dtype=np.bool_)
    _append_observation(
        packets,
        privileged_obs,
        sensor,
        spec,
        obs,
        obs["metrics"],
        env,
        base_twists,
        local_grasps,
    )

    done = False
    final_info = obs["metrics"]
    branched_events: set[int] = set()
    matched_branch_groups: list[MatchedBranchGroup] = []
    while not done:
        event_index = int(obs["metrics"].get("private_event_index", -1))
        event_branch = bool(obs["metrics"].get("private_event_active", False)) and (
            event_index not in branched_events
        )
        scheduled_v2_branch = collect_matched_branches and len(matched_branch_groups) < 3 and len(
            actions[0]
        ) in (32, 64, 96)
        should_branch = event_branch or scheduled_v2_branch
        if should_branch and collect_matched_branches:
            matched = _matched_counterfactual_branch_group(
                env,
                sensor,
                spec,
                decision_t=len(actions[0]),
                group_id=len(matched_branch_groups),
            )
            matched_branch_groups.append(matched)
            branch = {
                "valid": matched.valid_mask.any(axis=-1).astype(np.float32),
                "plan_pair": matched.plan_pairs,
                "action": matched.actions,
                "return": matched.reward.sum(axis=-1),
                "success": matched.success,
                "constraint_violation": matched.constraint,
            }
        else:
            if should_branch:
                branch = _counterfactual_branches(env)
            elif collect_matched_branches:
                branch = _empty_research_v2_branches()
            else:
                branch = _empty_branches()
        if event_branch:
            branched_events.add(event_index)
        policy_out = policy(env)
        joint_action = np.asarray(policy_out.action, dtype=np.float32).reshape(2, 4)
        # The simulator happens to accept world-axis planar commands.   plan
        # latents and real-robot commands use each sender's ego/base frame, so
        # convert before writing the deployable action stream.
        ego_action = _ego_action_commands(joint_action, obs)
        next_obs, reward, done, final_info = env.step(joint_action.reshape(-1))

        actions[0].append(ego_action[0].copy())
        actions[1].append(ego_action[1].copy())
        privileged_tr["environment_action_world"].append(joint_action.reshape(-1).copy())
        privileged_tr["reward"].append([float(reward)])
        privileged_tr["done"].append([float(done)])
        privileged_tr["success"].append([float(final_info.get("success", False))])
        privileged_tr["failure"].append([float(final_info.get("failure", False))])
        privileged_tr["failure_reason"].append(
            [failure_to_id(str(final_info.get("failure_reason", "none")))]
        )
        privileged_tr["phase"].append([phase_to_id(policy_out.phase)])
        privileged_tr["progress"].append([float(final_info.get("progress", 0.0))])
        privileged_tr["force_proxy"].append([float(final_info.get("force_proxy", 0.0))])
        privileged_tr["contact"].append([float(final_info.get("ncon", 0) > 0)])
        privileged_tr["grasp"].append([float(final_info.get("grasped", False))])
        privileged_tr["private_event_type"].append(
            [int(final_info.get("private_event_type", -1))]
        )
        privileged_tr["private_event_informed_agent"].append(
            [int(final_info.get("private_event_informed_agent", -1))]
        )
        privileged_tr["private_event_maneuver"].append(
            [int(final_info.get("private_event_maneuver", 0))]
        )
        privileged_tr["private_event_error"].append(
            [float(final_info.get("private_event_error_steps", 0) > 0)]
        )
        privileged_tr["branch_valid"].append(branch["valid"])
        privileged_tr["branch_plan_pair"].append(branch["plan_pair"])
        privileged_tr["branch_action"].append(branch["action"])
        privileged_tr["branch_return"].append(branch["return"])
        privileged_tr["branch_success"].append(branch["success"])
        privileged_tr["branch_constraint_violation"].append(
            branch["constraint_violation"]
        )

        base_twists = _base_twists_from_action(env, joint_action, next_obs)
        local_grasps = joint_action[:, 3] > 0.5
        _append_observation(
            packets,
            privileged_obs,
            sensor,
            spec,
            next_obs,
            final_info,
            env,
            base_twists,
            local_grasps,
        )
        obs = next_obs

    episode = Episode(
        local_observations={agent_id: stack_packets(values, spec) for agent_id, values in packets.items()},
        actions={agent_id: np.asarray(values, dtype=np.float32) for agent_id, values in actions.items()},
        privileged_observations={
            name: np.asarray(values, dtype=np.float32) for name, values in privileged_obs.items()
        },
        privileged_transitions={
            name: np.asarray(
                values,
                dtype=np.int32
                if name in {
                    "failure_reason",
                    "phase",
                    "private_event_type",
                    "private_event_informed_agent",
                    "private_event_maneuver",
                }
                else np.float32,
            )
            for name, values in privileged_tr.items()
        },
        metadata={
            "seed": int(seed),
            "scenario": env.cfg.scenario,
            "success": bool(final_info.get("success", False)),
            "failure": bool(final_info.get("failure", False)),
            "failure_reason": str(final_info.get("failure_reason", "none")),
            "control_dt": float(env.cfg.control_dt),
            "action_coordinate_frame": "sender ego/base frame",
            "local_packet_source": "simulated noisy/occluded local sensors",
            "local_contact_semantics": STRICT_LOCAL_CONTACT_SEMANTICS,
            "local_force_semantics": STRICT_LOCAL_FORCE_SEMANTICS,
            "local_force_units": LOCAL_FORCE_UNITS,
            "local_sensor_provenance": STRICT_LOCAL_SENSOR_PROVENANCE,
            "local_force_scale_newtons": float(
                env.cfg.local_force_scale_newtons
            ),
            "rgb_available_on_real_platform": True,
            "rgb_camera_names": [],
            "rgb_calibration_reference": "",
        },
        research_v2_branch_groups=matched_branch_groups,
    )
    return episode, dict(episode.metadata), spec


_BRANCH_PLAN_PAIRS = np.asarray(
    [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0), (-1, 0)],
    dtype=np.float32,
)
RESEARCH_V2_BRANCH_ACTION_PROGRAM = "joint_4d_anchors_smooth_random_v1"


def _empty_branches() -> dict[str, np.ndarray]:
    count = len(_BRANCH_PLAN_PAIRS)
    return {
        "valid": np.zeros(count, dtype=np.float32),
        "plan_pair": _BRANCH_PLAN_PAIRS.copy(),
        "action": np.zeros((count, 2, 16, 4), dtype=np.float32),
        "return": np.zeros(count, dtype=np.float32),
        "success": np.zeros(count, dtype=np.float32),
        "constraint_violation": np.zeros(count, dtype=np.float32),
    }


def _empty_research_v2_branches() -> dict[str, np.ndarray]:
    count = len(_BRANCH_PLAN_PAIRS) + 2
    return {
        "valid": np.zeros(count, dtype=np.float32),
        "plan_pair": np.zeros((count, 2), dtype=np.float32),
        "action": np.zeros((count, 2, 16, 4), dtype=np.float32),
        "return": np.zeros(count, dtype=np.float32),
        "success": np.zeros(count, dtype=np.float32),
        "constraint_violation": np.zeros(count, dtype=np.float32),
    }


def _counterfactual_branches(
    env: TwoRobotCarryNarrowPassageEnv,
    *,
    horizon: int = 16,
) -> dict[str, np.ndarray]:
    """Evaluate six action-plan pairs from one identical simulator snapshot."""

    snapshot = env.snapshot()
    returns = np.zeros(len(_BRANCH_PLAN_PAIRS), dtype=np.float32)
    successes = np.zeros_like(returns)
    constraints = np.zeros_like(returns)
    branch_actions = np.zeros(
        (len(_BRANCH_PLAN_PAIRS), 2, horizon, 4), dtype=np.float32
    )
    try:
        for branch_index, plan_pair in enumerate(_BRANCH_PLAN_PAIRS):
            env.restore(snapshot)
            done = False
            info = env.get_obs()["metrics"]
            for branch_step in range(horizon):
                action = env.scripted_action().copy()
                # A plan pair is an ego-local maneuver choice, represented here
                # by a bounded lateral command for each agent.
                action[0] = 0.45 * float(plan_pair[0])
                action[4] = 0.45 * float(plan_pair[1])
                branch_actions[branch_index, :, branch_step] = _ego_action_commands(
                    action.reshape(2, 4), env.get_obs()
                )
                _, reward, done, info = env.step(action)
                returns[branch_index] += float(reward)
                constraints[branch_index] = max(
                    constraints[branch_index],
                    float(
                        info.get("force_violation", False)
                        or info.get("collision", False)
                        or info.get("private_event_error_steps", 0) > 0
                    ),
                )
                if done:
                    break
            successes[branch_index] = float(info.get("success", False))
    finally:
        env.restore(snapshot)
    return {
        "valid": np.ones(len(_BRANCH_PLAN_PAIRS), dtype=np.float32),
        "plan_pair": _BRANCH_PLAN_PAIRS.copy(),
        "action": branch_actions,
        "return": returns,
        "success": successes,
        "constraint_violation": constraints,
    }


def _matched_counterfactual_branch_group(
    env: TwoRobotCarryNarrowPassageEnv,
    sensor: LocalObservationSimulator,
    spec: LocalObservationSpec,
    *,
    decision_t: int,
    group_id: int,
    horizon: int = 16,
) -> MatchedBranchGroup:
    """Collect action-matched future local observations from one snapshot."""

    env_snapshot = env.snapshot()
    sensor_snapshot = sensor.snapshot()
    plan_pairs = _research_v2_plan_pairs(env, decision_t, group_id)
    action_delta, grip_override = _research_v2_action_programs(
        env,
        plan_pairs,
        decision_t=decision_t,
        group_id=group_id,
        horizon=horizon,
    )
    count = len(plan_pairs)
    actions = np.zeros((count, 2, horizon, 4), dtype=np.float32)
    valid = np.zeros((count, horizon), dtype=np.bool_)
    reward_values = np.zeros((count, horizon), dtype=np.float32)
    progress = np.zeros((count, horizon), dtype=np.float32)
    contact = np.zeros((count, horizon, 2), dtype=np.float32)
    force = np.zeros((count, horizon, 2), dtype=np.float32)
    success = np.zeros(count, dtype=np.float32)
    constraint = np.zeros(count, dtype=np.float32)
    terminal = np.zeros(count, dtype=np.float32)
    future_local = {
        agent_id: {
            name: np.zeros((count, horizon, *shape), dtype=np.float32)
            for name, shape in spec.field_shapes().items()
        }
        for agent_id in (0, 1)
    }
    try:
        for branch_index, _plan_pair in enumerate(plan_pairs):
            env.restore(env_snapshot)
            sensor.restore(sensor_snapshot)
            done = False
            info = env.get_obs()["metrics"]
            for step in range(horizon):
                observation_before = env.get_obs()
                action_matrix = env.scripted_action().copy().reshape(2, 4)
                action_matrix[:, :3] += action_delta[branch_index, step, :, :3]
                override = grip_override[branch_index, step]
                action_matrix[:, 3] = np.where(
                    np.isfinite(override), override, action_matrix[:, 3]
                )
                action_matrix = np.clip(action_matrix, -1.0, 1.0)
                action = action_matrix.reshape(-1)
                actions[branch_index, :, step] = _ego_action_commands(
                    action_matrix, observation_before
                )
                observation_after, reward, done, info = env.step(action)
                valid[branch_index, step] = True
                reward_values[branch_index, step] = float(reward)
                progress[branch_index, step] = float(info.get("progress", 0.0))
                local_contact = np.asarray(info.get("local_contact_agents", np.zeros(2)), dtype=np.float32)
                local_force = np.asarray(info.get("local_force_agents", np.zeros(2)), dtype=np.float32)
                contact[branch_index, step] = local_contact
                force[branch_index, step] = local_force
                constraint[branch_index] = max(
                    constraint[branch_index],
                    float(
                        info.get("force_violation", False)
                        or info.get("collision", False)
                        or info.get("private_event_error_steps", 0) > 0
                    ),
                )
                packet_lists = {0: [], 1: []}
                privileged_lists = _empty_privileged_observation_lists()
                base_twists = _base_twists_from_action(env, action_matrix, observation_after)
                _append_observation(
                    packet_lists,
                    privileged_lists,
                    sensor,
                    spec,
                    observation_after,
                    info,
                    env,
                    base_twists,
                    action_matrix[:, 3] > 0.5,
                )
                for agent_id in (0, 1):
                    mapping = stack_packets(packet_lists[agent_id], spec)
                    for name in spec.field_shapes():
                        future_local[agent_id][name][branch_index, step] = mapping[name][0]
                if done:
                    terminal[branch_index] = 1.0
                    break
            success[branch_index] = float(info.get("success", False))
    finally:
        env.restore(env_snapshot)
        sensor.restore(sensor_snapshot)
    return MatchedBranchGroup(
        group_id=group_id,
        decision_t=decision_t,
        plan_pairs=plan_pairs,
        actions=actions,
        valid_mask=valid,
        future_local_observations=future_local,
        reward=reward_values,
        progress=progress,
        contact=contact,
        force=force,
        success=success,
        constraint=constraint,
        terminal=terminal,
    )


def _research_v2_plan_pairs(
    env: TwoRobotCarryNarrowPassageEnv, decision_t: int, group_id: int
) -> np.ndarray:
    """Six anchors plus outcome-independent random and state-adaptive pairs."""

    rng = np.random.default_rng(int(env.cfg.seed) + 9973 * decision_t + group_id)
    random_pair = rng.uniform(-1.0, 1.0, size=(1, 2)).astype(np.float32)
    object_x = float(env._object_pose_xy_yaw()[0])
    correction = float(np.clip(-2.0 * object_x, -1.0, 1.0))
    adaptive_pair = np.asarray([[correction, correction]], dtype=np.float32)
    return np.concatenate((_BRANCH_PLAN_PAIRS, random_pair, adaptive_pair), axis=0)


def _research_v2_action_programs(
    env: TwoRobotCarryNarrowPassageEnv,
    plan_pairs: np.ndarray,
    *,
    decision_t: int,
    group_id: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic, outcome-independent joint 4-D branch programs.

    ``plan_pairs`` remains the compact target-only maneuver label required by
    the on-disk schema.  The forced actions are intentionally richer than that
    label: anchors perturb lateral/longitudinal/yaw commands over time, one
    anchor contains a short grasp dropout, one branch is a seeded smooth random
    joint program, and the final branch uses only the decision-time object pose
    for a state-adaptive correction.  Candidate 4 is an unmodified scripted
    control.  No future outcome is inspected while constructing the programs.
    """

    pairs = np.asarray(plan_pairs, dtype=np.float32)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("Research-v2 plan pairs must have shape [N,2]")
    if horizon <= 0:
        raise ValueError("Research-v2 branch horizon must be positive")
    count = int(pairs.shape[0])
    delta = np.zeros((count, horizon, 2, 4), dtype=np.float32)
    grip_override = np.full((count, horizon, 2), np.nan, dtype=np.float32)
    phase = np.linspace(0.0, 1.0, horizon, dtype=np.float32)
    envelope = 0.55 + 0.45 * np.sin(np.pi * phase)
    oscillation = np.sin(2.0 * np.pi * phase)

    # The first six candidates are fixed anchors.  Index 4 deliberately stays
    # at zero and therefore reproduces the scripted policy from the snapshot.
    for branch_index in range(min(len(_BRANCH_PLAN_PAIRS), count)):
        if branch_index == 4:
            continue
        pair = pairs[branch_index]
        pair_mean = float(pair.mean())
        pair_difference = float(0.5 * (pair[0] - pair[1]))
        delta[branch_index, :, :, 0] = (
            0.34 * envelope[:, None] * pair[None, :]
        )
        delta[branch_index, :, 0, 1] = (
            0.16 * pair_mean * envelope + 0.12 * pair_difference * oscillation
        )
        delta[branch_index, :, 1, 1] = (
            0.16 * pair_mean * envelope - 0.12 * pair_difference * oscillation
        )
        delta[branch_index, :, 0, 2] = (
            0.30 * float(pair[0]) * oscillation + 0.16 * pair_difference * envelope
        )
        delta[branch_index, :, 1, 2] = (
            0.30 * float(pair[1]) * oscillation - 0.16 * pair_difference * envelope
        )

    # A short one-agent grip dropout supplies the fourth action dimension while
    # retaining valid pre- and post-pulse dynamics in the same branch.
    if count > 5:
        pulse_start = max(1, horizon // 3)
        pulse_stop = min(horizon, pulse_start + max(2, horizon // 4))
        grip_override[5, pulse_start:pulse_stop, 0] = 0.0

    seed = int(env.cfg.seed) + 9973 * int(decision_t) + 101 * int(group_id)
    if count > 6:
        rng = np.random.default_rng(seed + 53)
        noise = rng.normal(0.0, 1.0, size=(horizon, 2, 3)).astype(np.float32)
        # A short causal smoothing filter produces trajectories instead of
        # independent action jitter, closer to tokenizer-generated plans.
        for step in range(1, horizon):
            noise[step] = 0.68 * noise[step - 1] + 0.32 * noise[step]
        noise /= np.maximum(noise.std(axis=0, keepdims=True), 1e-5)
        delta[6, :, :, :3] = noise * np.asarray([0.30, 0.24, 0.36], dtype=np.float32)
        random_agent = int(rng.integers(0, 2))
        random_start = int(rng.integers(max(1, horizon // 4), max(2, 3 * horizon // 4)))
        random_stop = min(horizon, random_start + max(2, horizon // 5))
        grip_override[6, random_start:random_stop, random_agent] = 0.0

    if count > 7:
        object_x, _object_y, object_yaw = (
            float(value) for value in env._object_pose_xy_yaw()
        )
        correction = float(np.clip(-2.0 * object_x, -1.0, 1.0))
        yaw_correction = float(np.clip(-1.5 * object_yaw, -1.0, 1.0))
        delta[7, :, :, 0] = 0.34 * correction * envelope[:, None]
        delta[7, :, :, 1] = 0.12 * envelope[:, None]
        delta[7, :, 0, 2] = 0.30 * yaw_correction * envelope + 0.10 * oscillation
        delta[7, :, 1, 2] = 0.30 * yaw_correction * envelope - 0.10 * oscillation

    return delta, grip_override


def _empty_privileged_observation_lists() -> Dict[str, list]:
    return {
        "time": [],
        "progress": [],
        "robot_pose_world": [],
        "object_pose_world": [],
        "object_pose_ego": [],
        "teammate_pose_ego": [],
        "base_twist_ego": [],
        "global_state": [],
        "private_event_truth": [],
        "private_event_cue_agents": [],
        "private_event_valid_agents": [],
        "next_gate_context_agents": [],
    }


def _append_observation(
    packets: Dict[int, list],
    privileged: Dict[str, list],
    sensor: LocalObservationSimulator,
    spec: LocalObservationSpec,
    obs: dict,
    info: dict,
    env: TwoRobotCarryNarrowPassageEnv,
    base_twists: np.ndarray,
    local_grasps: np.ndarray,
) -> None:
    global_state = np.asarray(obs["global_state"], dtype=np.float32)
    robot_poses = np.stack([global_state[0:3], global_state[3:6]], axis=0)
    object_pose = np.asarray(obs["object"], dtype=np.float32).reshape(3)
    goal_world = np.asarray([0.0, env.cfg.goal_y, 0.0], dtype=np.float32)
    local_contacts = np.asarray(
        info.get("local_contact_agents", obs["metrics"].get("local_contact_agents")),
        dtype=np.float32,
    )
    if local_contacts.shape != (2,):
        raise ValueError(" collection requires two per-agent local contact flags")
    local_forces = np.asarray(
        info.get("local_force_agents", obs["metrics"].get("local_force_agents")),
        dtype=np.float32,
    )
    if local_forces.shape != (2,):
        raise ValueError(" collection requires two per-agent local force values")
    event_cues = np.asarray(info.get("private_event_cue_agents", np.zeros((2, 3))), dtype=np.float32)
    event_valid = np.asarray(info.get("private_event_valid_agents", np.zeros(2)), dtype=np.float32)
    event_age = np.asarray(info.get("private_event_age_agents", np.zeros(2)), dtype=np.float32)
    gate_context = np.asarray(info.get("next_gate_context_agents", np.zeros((2, 3))), dtype=np.float32)
    if event_cues.shape != (2, 3) or gate_context.shape != (2, 3):
        raise ValueError("private event cue/context must be per-agent")

    object_rel_truth = []
    teammate_rel_truth = []
    for agent_id in (0, 1):
        truth = PrivilegedAgentState(
            ego_pose_world=robot_poses[agent_id],
            object_pose_world=object_pose,
            task_goal_world=goal_world,
            base_twist=np.asarray(base_twists[agent_id], dtype=np.float32),
            joint_position=np.zeros(spec.joint_dim, dtype=np.float32),
            joint_velocity=np.zeros(spec.joint_dim, dtype=np.float32),
            joint_torque=np.zeros(spec.joint_dim, dtype=np.float32),
            local_force=np.asarray([local_forces[agent_id]], dtype=np.float32),
            contact=bool(local_contacts[agent_id] > 0.5),
            grasp=bool(local_grasps[agent_id]),
            private_event_cue=event_cues[agent_id],
            private_event_valid=bool(event_valid[agent_id] > 0.5),
            private_event_age=float(event_age[agent_id]),
            next_gate_context=gate_context[agent_id],
        )
        # Object visibility can later be supplied by the RGB perception stack.
        # For now stochastic dropout/noise is owned by the sensor simulator.
        packets[agent_id].append(sensor.observe(agent_id, truth, object_visible=True))
        object_rel_truth.append(ego_relative_pose(object_pose, robot_poses[agent_id]))
        teammate_rel_truth.append(
            ego_relative_pose(robot_poses[1 - agent_id], robot_poses[agent_id])
        )

    privileged["time"].append([float(env.step_count) * float(env.cfg.control_dt)])
    privileged["progress"].append([float(info.get("progress", 0.0))])
    privileged["robot_pose_world"].append(robot_poses)
    privileged["object_pose_world"].append(object_pose)
    privileged["object_pose_ego"].append(np.stack(object_rel_truth, axis=0))
    # Exact teammate state is a label/evaluation target only.  It is never
    # copied into either deployable agent stream.
    privileged["teammate_pose_ego"].append(np.stack(teammate_rel_truth, axis=0))
    privileged["base_twist_ego"].append(np.asarray(base_twists, dtype=np.float32))
    privileged["global_state"].append(global_state)
    privileged["private_event_truth"].append(
        np.asarray(
            [
                info.get("private_event_index", -1),
                info.get("private_event_type", -1),
                info.get("private_event_informed_agent", -1),
                info.get("private_event_maneuver", 0),
            ],
            dtype=np.float32,
        )
    )
    privileged["private_event_cue_agents"].append(event_cues)
    privileged["private_event_valid_agents"].append(event_valid[:, None])
    privileged["next_gate_context_agents"].append(gate_context)


def _base_twists_from_action(
    env: TwoRobotCarryNarrowPassageEnv,
    action: np.ndarray,
    obs: dict,
) -> np.ndarray:
    """Convert the simulator's world-axis velocity command to ego twist."""

    state = np.asarray(obs["global_state"], dtype=np.float32)
    result = np.zeros((2, 3), dtype=np.float32)
    for agent_id in (0, 1):
        yaw = float(state[2] if agent_id == 0 else state[5])
        world_v = np.asarray(action[agent_id, :2], dtype=np.float32) * float(env.cfg.max_action_v)
        c, s = np.cos(yaw), np.sin(yaw)
        result[agent_id, 0] = c * world_v[0] + s * world_v[1]
        result[agent_id, 1] = -s * world_v[0] + c * world_v[1]
        result[agent_id, 2] = float(action[agent_id, 2]) * float(env.cfg.max_action_w)
    return result


def _ego_action_commands(action: np.ndarray, obs: dict) -> np.ndarray:
    """Rotate the simulator's world-axis commands into each sender's frame."""

    action = np.asarray(action, dtype=np.float32).reshape(2, 4)
    state = np.asarray(obs["global_state"], dtype=np.float32)
    result = action.copy()
    for agent_id in (0, 1):
        yaw = float(state[2] if agent_id == 0 else state[5])
        c, s = np.cos(yaw), np.sin(yaw)
        world_x, world_y = action[agent_id, 0], action[agent_id, 1]
        result[agent_id, 0] = c * world_x + s * world_y
        result[agent_id, 1] = -s * world_x + c * world_y
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scripted", "noisy", "recovery"], required=True)
    parser.add_argument("--num_episodes", type=int, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument("--randomize", type=int, default=1)
    parser.add_argument("--scenario", type=str, default="nominal")
    parser.add_argument("--object_dropout_prob", type=float, default=0.05)
    parser.add_argument("--object_position_std", type=float, default=0.025)
    parser.add_argument("--object_yaw_std", type=float, default=0.035)
    parser.add_argument("--rgb_camera_names", nargs="*", default=[])
    parser.add_argument("--rgb_calibration_reference", type=str, default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = TwoRobotCarryNarrowPassageEnv(CarryEnvConfig(scenario=args.scenario))
    sensor_config = SensorSimulationConfig(
        control_dt=env.cfg.control_dt,
        object_dropout_prob=args.object_dropout_prob,
        object_position_std=args.object_position_std,
        object_yaw_std=args.object_yaw_std,
    )

    summaries = []
    for episode_index in trange(args.num_episodes):
        seed = args.seed_start + episode_index
        policy = ScriptedPolicy(
            noise_std=args.noise_std,
            seed=seed,
            mode=args.mode,
        )
        episode, metadata, spec = collect_one_episode(
            env,
            policy,
            seed,
            randomize=bool(args.randomize),
            sensor_config=sensor_config,
        )
        episode.metadata.update(
            {
                "episode_index": episode_index,
                "mode": args.mode,
                "policy_noise_std": args.noise_std,
                "rgb_camera_names": list(args.rgb_camera_names),
                "rgb_calibration_reference": args.rgb_calibration_reference,
            }
        )
        save_episode(out_dir / f"episode_{episode_index:06d}.hdf5", episode, spec)
        summaries.append(metadata)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "num_episodes": len(summaries),
        "success_rate": float(np.mean([entry["success"] for entry in summaries])) if summaries else 0.0,
        "out_dir": str(out_dir),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
