"""Online/offline parity tests."""

from __future__ import annotations

import h5py
import numpy as np

from data.collect import collect_one_episode
from data.decentralized_dataset import DecentralizedTransitionDataset
from data.local_observation import LocalHistoryBuffer
from data.policies import ScriptedPolicy
from data.schema import read_local_observations, save_episode
from envs.two_robot_carry_env import CarryEnvConfig, TwoRobotCarryNarrowPassageEnv


def test_online_ring_buffer_matches_offline_dataset_history(tmp_path):
    env = TwoRobotCarryNarrowPassageEnv(CarryEnvConfig(episode_len=5))
    episode, _, spec = collect_one_episode(
        env,
        ScriptedPolicy(seed=11, mode="scripted"),
        seed=11,
        randomize=False,
    )
    path = tmp_path / "episode_000000.hdf5"
    save_episode(path, episode, spec)

    dataset = DecentralizedTransitionDataset(tmp_path, history=4, horizon=1)
    # Index order is t-major then ego-major: index 4 is (decision_t=2, ego=0).
    offline = dataset[4]

    online = LocalHistoryBuffer(spec, action_dim=4, history=4)
    with h5py.File(path, "r") as file:
        for observation_t in range(3):
            mapping = read_local_observations(file, 0, spec, observation_t)
            previous_action = (
                None
                if observation_t == 0
                else file["transitions/actions/agent_0"][observation_t - 1]
            )
            online.append_mapping(mapping, previous_action=previous_action)

    arrays = online.as_arrays()
    for key in (
        "model_observation_history",
        "prev_action_history",
        "local_history",
        "history_mask",
        "padding_mask",
        "prev_action_valid_mask",
        "object_observation_history",
        "object_valid_history",
        "object_confidence_history",
        "object_age_history",
    ):
        np.testing.assert_array_equal(arrays[key], offline[key].numpy(), err_msg=key)


def test_online_buffer_never_adds_current_candidate_action():
    from data.local_observation import (
        LocalObservationPacket,
        LocalObservationSpec,
        PoseEstimate,
    )

    spec = LocalObservationSpec()
    packet = LocalObservationPacket(
        base_twist=np.zeros(3, dtype=np.float32),
        joint_position=np.zeros(0, dtype=np.float32),
        joint_velocity=np.zeros(0, dtype=np.float32),
        joint_torque=np.zeros(0, dtype=np.float32),
        local_force=np.zeros(1, dtype=np.float32),
        contact=np.zeros(1, dtype=np.float32),
        grasp=np.zeros(1, dtype=np.float32),
        object_estimate=PoseEstimate(
            pose=np.zeros(3, dtype=np.float32),
            valid=np.zeros(1, dtype=np.float32),
            confidence=np.zeros(1, dtype=np.float32),
            age=np.ones(1, dtype=np.float32),
        ),
        task_goal=np.zeros(3, dtype=np.float32),
    )
    buffer = LocalHistoryBuffer(spec, action_dim=4, history=3)
    buffer.append(packet, previous_action=None)
    arrays = buffer.as_arrays()
    np.testing.assert_array_equal(arrays["prev_action_history"], 0.0)
    assert arrays["prev_action_valid_mask"].tolist() == [False, False, False]
