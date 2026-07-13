"""Data contract tests."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from data.decentralized_dataset import DecentralizedTransitionDataset
from data.local_observation import (
    LocalObservationSimulator,
    LocalObservationSpec,
    PrivilegedAgentState,
    SensorSimulationConfig,
    ego_relative_pose,
)
from data.schema import (
    LOCAL_FORCE_UNITS,
    SCHEMA_VERSION,
    STRICT_LOCAL_CONTACT_SEMANTICS,
    STRICT_LOCAL_FORCE_SEMANTICS,
    STRICT_LOCAL_SENSOR_PROVENANCE,
    TRANSITION_SEMANTICS,
    Episode,
    save_episode,
    validate_episode,
)


def test_ego_relative_pose_rotates_translation_into_robot_frame():
    ego = np.asarray([1.0, 2.0, np.pi / 2.0], dtype=np.float32)
    target = np.asarray([2.0, 2.0, np.pi], dtype=np.float32)
    relative = ego_relative_pose(target, ego)
    np.testing.assert_allclose(relative, [0.0, -1.0, np.pi / 2.0], atol=1e-6)


def test_deployable_packet_has_no_teammate_state_and_occlusion_does_not_leak_truth():
    spec = LocalObservationSpec()
    config = SensorSimulationConfig(
        control_dt=0.1,
        base_twist_std=0.0,
        force_std=0.0,
        object_position_std=0.0,
        object_yaw_std=0.0,
        object_dropout_prob=0.0,
    )
    simulator = LocalObservationSimulator(spec, config, seed=3)
    visible_truth = PrivilegedAgentState(
        ego_pose_world=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        object_pose_world=np.asarray([1.0, 2.0, 0.2], dtype=np.float32),
        task_goal_world=np.asarray([0.0, 4.0, 0.0], dtype=np.float32),
        local_force=np.asarray([0.4], dtype=np.float32),
    )
    visible = simulator.observe(0, visible_truth, object_visible=True)

    hidden_truth = PrivilegedAgentState(
        ego_pose_world=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        object_pose_world=np.asarray([99.0, -77.0, -2.0], dtype=np.float32),
        task_goal_world=np.asarray([0.0, 4.0, 0.0], dtype=np.float32),
        local_force=np.asarray([0.4], dtype=np.float32),
    )
    hidden = simulator.observe(0, hidden_truth, object_visible=False)

    assert spec.flat_dim == 23
    assert all("teammate" not in name for name in spec.field_shapes())
    assert all("teammate" not in name for name in spec.feature_names())
    np.testing.assert_allclose(hidden.object_estimate.pose, visible.object_estimate.pose)
    assert hidden.object_estimate.valid.item() == 0.0
    assert hidden.object_estimate.age.item() == pytest.approx(0.1)
    assert hidden.object_estimate.confidence.item() < visible.object_estimate.confidence.item()


def test_schema_rejects_non_transition_aligned_or_extra_deployable_data(tmp_path):
    spec = LocalObservationSpec()
    episode = _make_episode(spec, transitions=4)
    episode.actions[0] = np.zeros((5, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="same action shape"):
        validate_episode(episode, spec)

    episode = _make_episode(spec, transitions=4)
    episode.local_observations[0]["estimates/teammate/pose"] = np.zeros(
        (5, 3), dtype=np.float32
    )
    with pytest.raises(ValueError, match="extra=.*teammate"):
        save_episode(tmp_path / "episode_000000.hdf5", episode, spec)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("local/force", -0.01, r"local/force must lie in \[0, 1\]"),
        ("local/contact", 2.0, "binary flags"),
        ("local/grasp", -1.0, "binary flags"),
        ("estimates/object/valid", 0.5, "binary flags"),
        ("estimates/object/confidence", 1.01, r"must lie in \[0, 1\]"),
        ("estimates/object/age", -0.1, "cannot be negative"),
    ],
)
def test_strict_schema_rejects_out_of_range_deployable_values(
    field, bad_value, message
):
    spec = LocalObservationSpec()
    episode = _make_episode(spec, transitions=2)
    episode.local_observations[0][field][0, 0] = bad_value

    with pytest.raises(ValueError, match=message):
        validate_episode(episode, spec)


def test_strict_schema_requires_explicit_sensor_provenance(tmp_path):
    spec = LocalObservationSpec()
    episode = _make_episode(spec, transitions=2)
    episode.metadata.pop("local_sensor_provenance")

    with pytest.raises(ValueError, match="local_sensor_provenance"):
        save_episode(tmp_path / "episode_000000.hdf5", episode, spec)


def test_dataset_previous_action_padding_targets_and_ego_isolation(tmp_path):
    spec = LocalObservationSpec()
    episode = _make_episode(spec, transitions=6)
    path = tmp_path / "episode_000000.hdf5"
    save_episode(path, episode, spec)

    with h5py.File(path, "r") as file:
        assert file.attrs["schema_version"] == SCHEMA_VERSION
        assert file.attrs["transition_semantics"] == TRANSITION_SEMANTICS
        assert not bool(
            file["schema/local_observation"].attrs["explicit_teammate_state_allowed"]
        )
        assert "raw_sensors/agent_0/rgb" in file
        assert file["observations/agent_0/deployable/self/base_twist"].shape[0] == 7
        assert file["transitions/actions/agent_0"].shape[0] == 6

    dataset = DecentralizedTransitionDataset(
        tmp_path, history=4, horizon=2, stride=1
    )
    # Index order is (t=0, ego=0), (t=0, ego=1), ..., so index 4 is t=2, ego=0.
    sample = dataset[4]
    assert sample["ego_id"].item() == 0
    assert sample["decision_t"].item() == 2
    assert dataset.observation_dim == 23
    assert dataset.local_history_dim == 21
    assert all("object" not in name for name in dataset.input_feature_names)
    assert sample["padding_mask"].tolist() == [True, False, False, False]
    assert sample["history_mask"].tolist() == [False, True, True, True]
    assert sample["history_valid_mask"].tolist() == [False, True, True, True]
    assert sample["prev_action_valid_mask"].tolist() == [False, False, True, True]

    # History rows align as (o_0, no action), (o_1, a_0), (o_2, a_1).
    np.testing.assert_allclose(sample["prev_action_history"][1].numpy(), 0.0)
    np.testing.assert_allclose(sample["prev_action_history"][2].numpy(), 0.1)
    np.testing.assert_allclose(sample["prev_action_history"][3].numpy(), 0.2)
    assert not np.any(np.isclose(sample["prev_action_history"].numpy(), 0.3))
    np.testing.assert_allclose(sample["ego_future_action"][0].numpy(), 0.3)
    assert sample["flat_observation_history"].shape == (4, 23)
    assert sample["model_history"].shape == (4, 21)
    np.testing.assert_allclose(sample["local_history"], sample["model_history"])
    np.testing.assert_allclose(sample["object_observation"].numpy(), [20.0, 0.0, 0.0])
    np.testing.assert_allclose(sample["future_object_observation"][0].numpy(), [30.0, 0.0, 0.0])
    assert sample["ego_future_observation"].shape == (2, 23)
    assert sample["future_model_observation"].shape == (2, 17)
    np.testing.assert_allclose(
        sample["future_model_observation"][0, :3].numpy(), [3.0, 3.0, 3.0]
    )
    assert not np.any(np.isclose(sample["future_model_observation"].numpy(), 30.0))
    np.testing.assert_allclose(
        sample["target_local_force"].numpy(), [0.5, 2.0 / 3.0], atol=1e-6
    )
    np.testing.assert_allclose(sample["target_local_contact"].numpy(), [1.0, 0.0])
    np.testing.assert_allclose(sample["target_progress"].numpy(), [0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(sample["target_object_pose_world"][0].numpy(), [3.0, 0.0, 0.0])
    np.testing.assert_allclose(sample["target_object_pose_ego"][0].numpy(), [0.0, 0.0, 0.0])
    assert DecentralizedTransitionDataset.INPUT_KEYS == {
        "ego_id",
        "local_history",
        "history_mask",
        "object_observation_history",
        "object_valid_history",
        "object_confidence_history",
        "object_age_history",
    }

    ego_input_before = sample["local_history"].clone()
    with h5py.File(path, "r+") as file:
        teammate_private = file[
            "observations/agent_1/deployable/self/base_twist"
        ]
        teammate_private[:] = 123456.0
    ego_input_after = dataset[4]["local_history"]
    np.testing.assert_array_equal(ego_input_before.numpy(), ego_input_after.numpy())


def _make_episode(spec: LocalObservationSpec, transitions: int) -> Episode:
    observations = transitions + 1
    local = {}
    for agent_id in (0, 1):
        fields = {
            name: np.zeros((observations, *shape), dtype=np.float32)
            for name, shape in spec.field_shapes().items()
        }
        fields["self/base_twist"][:] = np.arange(observations, dtype=np.float32)[:, None]
        fields["self/base_twist"][:] += 100.0 * agent_id
        fields["estimates/object/valid"][:] = 1.0
        fields["estimates/object/confidence"][:] = 0.9
        fields["estimates/object/pose"][:, 0] = 10.0 * np.arange(observations)
        fields["local/force"][:, 0] = np.linspace(
            0.0, 1.0, observations, dtype=np.float32
        )
        fields["local/contact"][:, 0] = np.arange(observations) % 2
        fields["task/goal"][:, 1] = 3.0
        local[agent_id] = fields

    actions = {
        0: np.stack(
            [np.full(4, 0.1 * (t + 1), dtype=np.float32) for t in range(transitions)]
        ),
        1: np.stack(
            [np.full(4, -0.1 * (t + 1), dtype=np.float32) for t in range(transitions)]
        ),
    }
    robot_pose_world = np.zeros((observations, 2, 3), dtype=np.float32)
    robot_pose_world[:, 0, 0] = np.arange(observations)
    robot_pose_world[:, 1, 0] = np.arange(observations) + 1.0
    object_pose_world = np.zeros((observations, 3), dtype=np.float32)
    object_pose_world[:, 0] = np.arange(observations)
    privileged_observations = {
        "time": np.arange(observations, dtype=np.float32)[:, None],
        "robot_pose_world": robot_pose_world,
        "object_pose_world": object_pose_world,
        "object_pose_ego": np.zeros((observations, 2, 3), dtype=np.float32),
        "teammate_pose_ego": np.zeros((observations, 2, 3), dtype=np.float32),
        "base_twist_ego": np.zeros((observations, 2, 3), dtype=np.float32),
        "global_state": np.zeros((observations, 12), dtype=np.float32),
    }
    privileged_transitions = {
        "reward": np.arange(transitions, dtype=np.float32)[:, None],
        "done": np.zeros((transitions, 1), dtype=np.float32),
        "success": np.zeros((transitions, 1), dtype=np.float32),
        "failure": np.zeros((transitions, 1), dtype=np.float32),
        "failure_reason": np.zeros((transitions, 1), dtype=np.int32),
        "phase": np.zeros((transitions, 1), dtype=np.int32),
        "progress": (0.1 * np.arange(transitions, dtype=np.float32))[:, None],
        "force_proxy": np.zeros((transitions, 1), dtype=np.float32),
        "contact": np.zeros((transitions, 1), dtype=np.float32),
        "grasp": np.zeros((transitions, 1), dtype=np.float32),
    }
    return Episode(
        local_observations=local,
        actions=actions,
        privileged_observations=privileged_observations,
        privileged_transitions=privileged_transitions,
        metadata={
            "rgb_camera_names": ["front", "wrist"],
            "local_contact_semantics": STRICT_LOCAL_CONTACT_SEMANTICS,
            "local_force_semantics": STRICT_LOCAL_FORCE_SEMANTICS,
            "local_force_units": LOCAL_FORCE_UNITS,
            "local_sensor_provenance": STRICT_LOCAL_SENSOR_PROVENANCE,
            "local_force_scale_newtons": 1000.0,
        },
    )
