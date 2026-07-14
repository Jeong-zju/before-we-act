from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pytest
import torch

from data.exporters import (
    ExportObserver,
    HDF5TrajectoryExporter,
    LeRobotTrajectoryExporter,
)
from data.trajectory import parse_field_assignment, schema_profile
from envs.runtime import (
    CallablePolicy,
    RenderRequest,
    RunnerConfig,
    SimulationRunner,
    SimulationTransition,
)
from envs.video import StreamingVideoObserver
from models import WorldActionModel, WorldActionModelConfig, WorldModelInputs

ROOT = Path(__file__).resolve().parents[1]


class _FakeEnvironment:
    control_dt = 0.1

    def __init__(self) -> None:
        self.value = 0

    def reset(self, seed=None, randomize=True):
        del seed, randomize
        self.value = 0
        return self._observation()

    def step(self, action):
        self.value += 1
        done = self.value >= 2
        info = {
            "success": done,
            "failure": False,
            "failure_reason": "none",
            "progress": self.value / 2,
            "force_proxy": 0.25,
        }
        return self._observation(), float(np.asarray(action).sum()), done, info

    def render(self, *, camera, width, height):
        del camera
        return np.full((height, width, 3), self.value, dtype=np.uint8)

    def close(self):
        pass

    def _observation(self):
        state = np.asarray([self.value, self.value + 1], dtype=np.float32)
        return {
            "robot_0": state,
            "robot_1": -state,
            "object": np.asarray([self.value, 0.0, 0.0], dtype=np.float32),
            "global_state": state,
            "metrics": {
                "success": self.value >= 2,
                "failure": False,
                "failure_reason": "none",
            },
        }


def test_module_dependency_direction_is_enforced():
    _assert_no_package_import("models", forbidden={"data", "envs"})
    _assert_no_package_import("envs", forbidden={"data", "models"})


def test_world_action_model_consumes_only_explicit_tensor_inputs():
    model = WorldActionModel(
        WorldActionModelConfig(state_dim=3, action_dim=2, hidden_dim=8, hidden_layers=2)
    )
    inputs = WorldModelInputs(
        state=torch.zeros(4, 3),
        action=torch.ones(4, 2),
    )

    output = model(inputs)

    assert output.next_state.shape == (4, 3)
    assert output.reward.shape == (4, 1)
    assert output.done_logit.shape == (4, 1)
    with pytest.raises(ValueError, match="action must end"):
        model(WorldModelInputs(state=torch.zeros(4, 3), action=torch.zeros(4, 3)))


def test_realtime_runner_streams_aligned_transitions():
    env = _FakeEnvironment()
    now = [0.0]
    sleeps: list[float] = []

    def sleep(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    observer = _RecordingObserver()
    runner = SimulationRunner(
        env,
        CallablePolicy(
            lambda observation: np.asarray([observation["global_state"][0]])
        ),
        RunnerConfig(
            realtime=True,
            render=(RenderRequest("head", "fixed", width=8, height=6),),
        ),
        clock=lambda: now[0],
        sleeper=sleep,
    )
    summary = runner.run_episode(seed=7, observers=(observer,))

    assert summary.steps == 2
    assert sleeps == [0.1, 0.1]
    assert [step.frame_index for step in observer.transitions] == [0, 1]
    assert [step.timestamp for step in observer.transitions] == [0.0, 0.1]
    assert observer.transitions[0].observation["global_state"][0] == 0
    assert observer.transitions[0].next_observation["global_state"][0] == 1
    assert observer.transitions[0].images["head"].shape == (6, 8, 3)
    assert observer.transitions[0].next_images["head"][0, 0, 0] == 1


def test_hdf5_export_is_incremental_customizable_and_dtype_stable(tmp_path):
    env = _FakeEnvironment()
    schema = schema_profile("wam").with_overrides(
        add=(parse_field_assignment("diagnostics.force=info.force_proxy::float32"),),
        drop=("observation.object",),
    )
    exporter = HDF5TrajectoryExporter(tmp_path, schema)
    observer = ExportObserver((exporter,), fps=10.0)
    runner = SimulationRunner(
        env,
        CallablePolicy(lambda observation: np.ones(1, dtype=np.float32)),
    )
    summary = runner.run_episode(seed=3, observers=(observer,))
    observer.close()

    assert summary.steps == 2
    path = tmp_path / "episode_000000.hdf5"
    assert path.exists()
    assert not path.with_suffix(".partial.hdf5").exists()
    with h5py.File(path, "r") as file:
        assert file.attrs["num_steps"] == 2
        assert file["data/reward"].dtype == np.dtype("float32")
        assert file["data/diagnostics/force"].dtype == np.dtype("float32")
        assert file["data/observation/global_state"].shape == (2, 2)
        assert "data/observation/object" not in file
        assert file["data/task"].asstr()[0].startswith("carry the object")


def test_streaming_video_preserves_transition_frame_count(tmp_path):
    path = tmp_path / "rollout.mp4"
    video = StreamingVideoObserver(path, stream="head", fps=10.0)
    runner = SimulationRunner(
        _FakeEnvironment(),
        CallablePolicy(lambda observation: np.ones(1, dtype=np.float32)),
        RunnerConfig(render=(RenderRequest("head", "fixed", width=8, height=6),)),
    )

    summary = runner.run_episode(observers=(video,))
    video.close()

    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == summary.steps == 2
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 8
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 6
    finally:
        capture.release()


def test_lerobot_export_uses_official_streaming_writer_contract(tmp_path):
    created: dict[str, Any] = {}
    fake = _FakeLeRobotDataset()

    def factory(**kwargs):
        created.update(kwargs)
        return fake

    schema = schema_profile("vla", cameras=("head",))
    exporter = LeRobotTrajectoryExporter(
        tmp_path,
        schema,
        repo_id="local/test",
        fps=10,
        dataset_factory=factory,
    )
    observer = ExportObserver((exporter,), fps=10.0)
    runner = SimulationRunner(
        _FakeEnvironment(),
        CallablePolicy(lambda observation: np.ones(1, dtype=np.float32)),
        RunnerConfig(
            render=(RenderRequest("head", "fixed", width=8, height=6),),
            task="test instruction",
        ),
    )
    runner.run_episode(observers=(observer,))
    observer.close()

    assert created["streaming_encoding"] is True
    assert created["features"]["observation.images.head"] == {
        "dtype": "video",
        "shape": (6, 8, 3),
        "names": ["height", "width", "channel"],
    }
    assert created["features"]["observation.state"]["dtype"] == "float32"
    assert "timestamp" not in created["features"]
    assert len(fake.frames) == 2
    assert fake.frames[0]["task"] == "test instruction"
    assert fake.saved_episodes == 1
    assert fake.finalized


def _assert_no_package_import(package: str, *, forbidden: set[str]) -> None:
    for path in (ROOT / package).glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            roots: set[str] = set()
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".", 1)[0])
            overlap = roots & forbidden
            assert not overlap, f"{path.relative_to(ROOT)} imports forbidden {overlap}"


class _RecordingObserver:
    def __init__(self) -> None:
        self.transitions: list[SimulationTransition] = []

    def on_episode_start(self, **kwargs):
        self.start = kwargs

    def on_transition(self, transition):
        self.transitions.append(transition)

    def on_episode_end(self, summary):
        self.summary = summary


class _FakeLeRobotDataset:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.saved_episodes = 0
        self.finalized = False

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved_episodes += 1

    def finalize(self):
        self.finalized = True
