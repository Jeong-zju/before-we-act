from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from data.exporters import HDF5TrajectoryExporter, LeRobotTrajectoryExporter
from data.robofactory import ROBOFACTORY_SCHEMA_VERSION, RoboFactoryDataset
from train.progress import progress_detail


def test_robofactory_hdf5_conversion_aligns_and_renames_multi_agent_data(
    tmp_path: Path,
) -> None:
    source_path = _write_robofactory_source(tmp_path / "LiftBarrier-rf.h5")
    output = tmp_path / "converted"

    with RoboFactoryDataset(source_path) as source:
        assert [agent.source_name for agent in source.layout.agents] == [
            "panda-0",
            "panda-1",
        ]
        assert source.layout.state_size == 8
        assert source.layout.action_size == 4
        assert {
            camera.source_name: camera.target_name for camera in source.layout.cameras
        } == {
            "head_camera_agent0": "agent_0",
            "head_camera_global": "global",
        }
        schema = source.build_schema()
        manifest = source.convert(
            (HDF5TrajectoryExporter(output, schema),),
            fps=20,
            schema=schema,
            success_only=True,
        )

    assert [episode["source_episode_id"] for episode in manifest["episodes"]] == [9]
    assert manifest["task"] == "Lift Barrier"
    assert manifest["field_mapping"]["centralized_action"] == [
        {"source": "actions/panda-0", "target": "action", "slice": [0, 2]},
        {"source": "actions/panda-1", "target": "action", "slice": [2, 4]},
    ]

    episode_path = output / "episode_000000.hdf5"
    with h5py.File(episode_path, "r") as episode:
        assert episode.attrs["schema_profile"] == "robofactory"
        assert episode.attrs["schema_version"] == ROBOFACTORY_SCHEMA_VERSION
        assert episode.attrs["seed"] == 109
        assert episode.attrs["num_steps"] == 3
        assert episode["data/observation/state"].shape == (3, 8)
        assert episode["data/action"].shape == (3, 4)
        np.testing.assert_allclose(
            episode["data/observation/state"][0],
            [100.0, 101.0, 110.0, 111.0, 200.0, 201.0, 210.0, 211.0],
        )
        np.testing.assert_allclose(
            episode["data/next_observation/state"][0],
            [102.0, 103.0, 112.0, 113.0, 202.0, 203.0, 212.0, 213.0],
        )
        np.testing.assert_allclose(
            episode["data/action"][0], [1000.0, 1001.0, 2000.0, 2001.0]
        )
        np.testing.assert_allclose(
            episode["data/agents/panda_0/action"][0], [1000.0, 1001.0]
        )
        assert episode["data/observation/images/agent_0"].shape == (3, 2, 3, 3)
        assert episode["data/observation/images/global"][0, 0, 0, 0] == 90
        assert episode[
            "data/observation/camera_calibration/agent_0/intrinsic_cv"
        ].shape == (3, 3, 3)
        assert episode["data/next/success"][:].tolist() == [False, False, True]
        assert episode["data/next/done"][:].tolist() == [False, False, True]
        metadata = json.loads(episode.attrs["episode_metadata_json"])
        assert metadata["source_episode_id"] == 9
        assert metadata["agent_name_map"] == {
            "panda-0": "panda_0",
            "panda-1": "panda_1",
        }


def test_robofactory_lerobot_conversion_uses_canonical_features_and_finalizes(
    tmp_path: Path,
) -> None:
    source_path = _write_robofactory_source(tmp_path / "LiftBarrier-rf.h5")
    created: dict[str, Any] = {}
    fake = _FakeLeRobotDataset()
    progress_values: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> _FakeLeRobotDataset:
        created.update(kwargs)
        return fake

    with RoboFactoryDataset(source_path) as source:
        assert source.conversion_totals(max_episodes=1) == (1, 3)
        schema = source.build_schema(include_calibration=False)
        exporter = LeRobotTrajectoryExporter(
            tmp_path / "lerobot",
            schema,
            repo_id="local/robofactory-test",
            fps=20,
            dataset_factory=factory,
        )
        manifest = source.convert(
            (exporter,),
            fps=20,
            schema=schema,
            task="Lift the barrier together",
            max_episodes=1,
            progress=progress_values.append,
        )

    assert manifest["episodes"][0]["source_episode_id"] == 7
    assert created["robot_type"] == "two_robot_carry"
    assert created["features"]["observation.state"] == {
        "dtype": "float32",
        "shape": (8,),
        "names": [f"observation.state_{index}" for index in range(8)],
    }
    assert created["features"]["action"]["shape"] == (4,)
    assert created["features"]["observation.images.agent_0"] == {
        "dtype": "video",
        "shape": (2, 3, 3),
        "names": ["height", "width", "channel"],
    }
    assert "task" not in created["features"]
    assert len(fake.frames) == 3
    assert fake.frames[0]["task"] == "Lift the barrier together"
    np.testing.assert_allclose(
        fake.frames[0]["action"], [0.0, 1.0, 1000.0, 1001.0]
    )
    assert fake.saved_episodes == 1
    assert fake.finalized
    assert progress_values == [
        {
            "source_episode": 7,
            "episode": 1,
            "episodes": 1,
            "frame": frame,
            "frames": 3,
        }
        for frame in (1, 2, 3)
    ]
    assert progress_detail(progress_values[-1]) == "src 7 episode 1/1 frame 3/3"


def test_robofactory_conversion_rejects_observation_action_misalignment(
    tmp_path: Path,
) -> None:
    source_path = _write_robofactory_source(
        tmp_path / "bad.h5", malformed_qpos=True
    )
    with pytest.raises(ValueError, match=r"must have T\+1"):
        RoboFactoryDataset(source_path)


def _write_robofactory_source(
    path: Path,
    *,
    malformed_qpos: bool = False,
) -> Path:
    steps = 3
    with h5py.File(path, "w") as file:
        for source_id, episode_offset in ((7, 0), (9, 100)):
            trajectory = file.create_group(f"traj_{source_id}")
            actions = trajectory.create_group("actions")
            observations = trajectory.create_group("obs")
            agents = observations.create_group("agent")
            sensor_data = observations.create_group("sensor_data")
            sensor_param = observations.create_group("sensor_param")
            for agent_index in range(2):
                source_name = f"panda-{agent_index}"
                agent_offset = episode_offset + agent_index * 100
                actions.create_dataset(
                    source_name,
                    data=(
                        np.arange(steps * 2, dtype=np.float64).reshape(steps, 2)
                        + agent_offset * 10
                    ),
                )
                agent = agents.create_group(source_name)
                qpos_rows = steps if malformed_qpos and source_id == 7 else steps + 1
                agent.create_dataset(
                    "qpos",
                    data=(
                        np.arange(qpos_rows * 2, dtype=np.float32).reshape(
                            qpos_rows, 2
                        )
                        + agent_offset
                    ),
                )
                agent.create_dataset(
                    "qvel",
                    data=(
                        np.arange((steps + 1) * 2, dtype=np.float32).reshape(
                            steps + 1, 2
                        )
                        + agent_offset
                        + 10
                    ),
                )
            for camera_name, value in (
                ("head_camera_agent0", source_id),
                ("head_camera_global", source_id * 10),
            ):
                camera = sensor_data.create_group(camera_name)
                camera.create_dataset(
                    "rgb",
                    data=np.full((steps + 1, 2, 3, 3), value, dtype=np.uint8),
                )
                parameters = sensor_param.create_group(camera_name)
                parameters.create_dataset(
                    "intrinsic_cv",
                    data=np.repeat(
                        np.eye(3, dtype=np.float32)[None], steps + 1, axis=0
                    ),
                )
                parameters.create_dataset(
                    "extrinsic_cv",
                    data=np.zeros((steps + 1, 3, 4), dtype=np.float32),
                )
                parameters.create_dataset(
                    "cam2world_gl",
                    data=np.repeat(
                        np.eye(4, dtype=np.float32)[None], steps + 1, axis=0
                    ),
                )
            trajectory.create_dataset(
                "rewards", data=np.arange(steps, dtype=np.float32)
            )
            trajectory.create_dataset(
                "terminated", data=[False, False, True]
            )
            trajectory.create_dataset("truncated", data=[False, False, False])
            trajectory.create_dataset(
                "success",
                data=[False, False, source_id == 9],
            )

    sidecar = {
        "env_info": {
            "env_id": "LiftBarrier-rf",
            "env_kwargs": {"control_mode": "pd_joint_pos"},
        },
        "episodes": [
            {
                "episode_id": 7,
                "episode_seed": 107,
                "elapsed_steps": steps,
                "success": False,
            },
            {
                "episode_id": 9,
                "episode_seed": 109,
                "elapsed_steps": steps,
                "success": True,
            },
        ],
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    return path


class _FakeLeRobotDataset:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.saved_episodes = 0
        self.finalized = False

    def add_frame(self, frame: dict[str, Any]) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        self.saved_episodes += 1

    def finalize(self) -> None:
        self.finalized = True
