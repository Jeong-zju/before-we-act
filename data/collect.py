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
from envs.two_robot_carry_env import CarryEnvConfig, TwoRobotCarryNarrowPassageEnv


def collect_one_episode(
    env: TwoRobotCarryNarrowPassageEnv,
    policy: ScriptedPolicy,
    seed: int,
    *,
    randomize: bool = True,
    sensor_config: SensorSimulationConfig | None = None,
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
    while not done:
        event_index = int(obs["metrics"].get("private_event_index", -1))
        should_branch = bool(obs["metrics"].get("private_event_active", False)) and (
            event_index not in branched_events
        )
        branch = _counterfactual_branches(env) if should_branch else _empty_branches()
        if should_branch:
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
    )
    return episode, dict(episode.metadata), spec


_BRANCH_PLAN_PAIRS = np.asarray(
    [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0), (-1, 0)],
    dtype=np.float32,
)


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
