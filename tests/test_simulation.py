"""Simulation bridge tests."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from data.local_observation import (
    LocalObservationSpec,
    SensorSimulationConfig,
    ego_relative_pose,
)
from data.simulation import (
    SimulationAdapter,
    ego_action_to_world,
    world_action_to_ego,
)
from envs.two_robot_carry_env import CarryEnvConfig, TwoRobotCarryNarrowPassageEnv


def _noise_free_sensor(control_dt: float = 0.05) -> SensorSimulationConfig:
    return SensorSimulationConfig(
        control_dt=control_dt,
        base_twist_std=0.0,
        joint_position_std=0.0,
        joint_velocity_std=0.0,
        joint_torque_std=0.0,
        force_std=0.0,
        object_position_std=0.0,
        object_yaw_std=0.0,
        object_dropout_prob=0.0,
    )


@pytest.mark.parametrize("shape", [(8,), (2, 4)])
def test_action_coordinate_roundtrip_preserves_shape_and_values(shape):
    world = np.asarray(
        [[0.8, -0.3, 0.2, 1.0], [-0.4, 0.7, -0.6, 0.0]],
        dtype=np.float32,
    ).reshape(shape)
    yaws = np.asarray([np.pi / 2.0, -0.73], dtype=np.float32)

    ego = world_action_to_ego(world, yaws)
    recovered = ego_action_to_world(ego, yaws)

    assert ego.shape == shape
    assert recovered.shape == shape
    np.testing.assert_allclose(recovered, world, atol=1e-6)


def test_action_coordinate_transform_accepts_environment_observation():
    obs = {
        "global_state": np.asarray(
            [0.0, 0.0, np.pi / 2.0, 0.0, 0.0, -np.pi / 2.0],
            dtype=np.float32,
        )
    }
    world = np.asarray(
        [[1.0, 0.0, 0.2, 1.0], [1.0, 0.0, -0.2, 0.0]],
        dtype=np.float32,
    )
    ego = world_action_to_ego(world, obs)

    np.testing.assert_allclose(ego[0], [0.0, -1.0, 0.2, 1.0], atol=1e-6)
    np.testing.assert_allclose(ego[1], [0.0, 1.0, -0.2, 0.0], atol=1e-6)
    np.testing.assert_allclose(ego_action_to_world(ego, obs), world, atol=1e-6)


def test_adapter_builds_current_strict_packet_and_physical_base_twist():
    env = TwoRobotCarryNarrowPassageEnv(CarryEnvConfig())
    obs = env.reset(seed=3, randomize=False)
    spec = LocalObservationSpec(joint_dim=0, force_dim=1)
    adapter = SimulationAdapter(
        env,
        sensor_config=_noise_free_sensor(env.cfg.control_dt),
        spec=spec,
    )
    adapter.reset(seed=3)
    previous_world_action = np.asarray(
        [[1.0, 0.0, 0.25, 1.0], [0.0, -0.5, -0.5, 0.0]],
        dtype=np.float32,
    )

    packets = adapter.packets(obs, obs["metrics"], previous_world_action)

    assert isinstance(packets, tuple) and len(packets) == 2
    for packet in packets:
        packet.validate(spec)
        assert set(packet.as_mapping()) == set(spec.field_shapes())
        assert all("teammate" not in name and "global" not in name for name in packet.as_mapping())

    np.testing.assert_allclose(
        adapter.base_twists,
        [[env.cfg.max_action_v, 0.0, 0.25 * env.cfg.max_action_w],
         [0.0, -0.5 * env.cfg.max_action_v, -0.5 * env.cfg.max_action_w]],
        atol=1e-6,
    )
    np.testing.assert_allclose(packets[0].base_twist, adapter.base_twists[0], atol=1e-6)
    np.testing.assert_allclose(packets[1].base_twist, adapter.base_twists[1], atol=1e-6)
    assert packets[0].grasp.tolist() == [1.0]
    assert packets[1].grasp.tolist() == [0.0]

    state = np.asarray(obs["global_state"], dtype=np.float32)
    np.testing.assert_allclose(
        packets[0].object_estimate.pose,
        ego_relative_pose(obs["object"], state[0:3]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        packets[1].object_estimate.pose,
        ego_relative_pose(obs["object"], state[3:6]),
        atol=1e-6,
    )


def test_packet_firewall_ignores_teammate_and_unrelated_global_truth():
    env = TwoRobotCarryNarrowPassageEnv(CarryEnvConfig())
    obs = env.reset(seed=9, randomize=False)
    info = obs["metrics"]
    adapter = SimulationAdapter(
        env,
        sensor_config=_noise_free_sensor(env.cfg.control_dt),
    )
    adapter.reset(seed=101)
    original = adapter.packets(obs, info)[0].as_mapping()

    changed_obs = copy.deepcopy(obs)
    changed_info = copy.deepcopy(info)
    # Agent 1's private pose/vector and agent 0's legacy teammate slots must
    # never influence agent 0's deployable packet.
    changed_obs["global_state"][3:6] = [91.0, -72.0, 1.7]
    changed_obs["global_state"][9:] = 1234.0
    changed_obs["robot_1"][:] = -999.0
    changed_obs["robot_0"][6:9] = [444.0, 555.0, -2.0]
    changed_info["rel_target_pose_agents"] = np.full((2, 3), 888.0, dtype=np.float32)
    changed_info["local_obs_agents"] = np.full((2, 17), 777.0, dtype=np.float32)
    changed_info["local_contact_agents"][1] = 1.0
    # Keep the peer sensor reading contract-valid: ``packets`` returns both
    # agents, so an invalid peer reading must reject the joint call rather
    # than be silently accepted.  A legal but maximally different value still
    # proves that the peer channel cannot affect agent 0's packet.
    changed_info["local_force_agents"][1] = 1.0

    adapter.reset(seed=101)
    changed = adapter.packets(changed_obs, changed_info)[0].as_mapping()

    assert set(original) == set(changed)
    for name in original:
        np.testing.assert_array_equal(changed[name], original[name], err_msg=name)


def test_adapter_keeps_peer_only_contact_out_of_ego_packet():
    env = TwoRobotCarryNarrowPassageEnv(CarryEnvConfig())
    obs = env.reset(seed=10, randomize=False)
    info = copy.deepcopy(obs["metrics"])
    info["local_contact_agents"] = np.asarray([0.0, 1.0], dtype=np.float32)
    info["local_force_agents"] = np.asarray([0.25, 0.75], dtype=np.float32)
    adapter = SimulationAdapter(
        env,
        sensor_config=_noise_free_sensor(env.cfg.control_dt),
    )

    packets = adapter.packets(obs, info)

    assert packets[0].contact.tolist() == [0.0]
    assert packets[1].contact.tolist() == [1.0]
    assert packets[0].local_force.tolist() == [0.25]
    assert packets[1].local_force.tolist() == [0.75]


def test_sensor_reset_replays_packets_deterministically_and_clears_twist():
    env = TwoRobotCarryNarrowPassageEnv(CarryEnvConfig())
    obs = env.reset(seed=12, randomize=False)
    config = SensorSimulationConfig(
        control_dt=env.cfg.control_dt,
        base_twist_std=0.02,
        force_std=0.03,
        object_position_std=0.04,
        object_yaw_std=0.05,
        object_dropout_prob=0.25,
    )
    adapter = SimulationAdapter(env, sensor_config=config)
    action = np.asarray([0.2, 0.3, 0.1, 1.0, -0.4, 0.5, -0.2, 1.0], dtype=np.float32)

    adapter.reset(seed=77)
    first = [packet.as_mapping() for packet in adapter.packets(obs, obs["metrics"], action)]
    assert np.any(adapter.base_twists != 0.0)

    adapter.reset(seed=77)
    np.testing.assert_array_equal(adapter.base_twists, np.zeros((2, 3), dtype=np.float32))
    replay = [packet.as_mapping() for packet in adapter.packets(obs, obs["metrics"], action)]

    for first_agent, replay_agent in zip(first, replay):
        for name in first_agent:
            np.testing.assert_array_equal(replay_agent[name], first_agent[name], err_msg=name)


def test_environment_exogenous_rng_tape_stays_aligned_after_trajectory_divergence():
    cfg = CarryEnvConfig(scenario="occlusion")
    inside = TwoRobotCarryNarrowPassageEnv(copy.deepcopy(cfg))
    outside = TwoRobotCarryNarrowPassageEnv(copy.deepcopy(cfg))
    inside.reset(seed=123, randomize=False)
    outside.reset(seed=123, randomize=False)

    # Only one trajectory is in the shared passage.  Both calls must still
    # consume the same three exogenous random draws.
    inside.data.qpos[inside.robot_a_qpos_addr + 1] = 1.0
    inside.data.qpos[inside.robot_b_qpos_addr + 1] = 1.0
    inside._set_object_pose(0.0, 1.0, inside.cfg.object_z, 0.0)
    outside.data.qpos[outside.robot_a_qpos_addr + 1] = -1.2
    outside.data.qpos[outside.robot_b_qpos_addr + 1] = -1.2
    outside._set_object_pose(0.0, -0.95, outside.cfg.object_z, 0.0)

    inside._update_scenario_state()
    outside._update_scenario_state()

    assert inside.rng.random() == outside.rng.random()


def test_adapter_rejects_non_current_local_observation_shape():
    env = TwoRobotCarryNarrowPassageEnv(CarryEnvConfig())
    with pytest.raises(ValueError, match="joint_dim=0, force_dim=1"):
        SimulationAdapter(env, spec=LocalObservationSpec(joint_dim=1, force_dim=1))
