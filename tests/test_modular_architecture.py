from __future__ import annotations

import ast
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import cv2
import h5py
import mujoco
import numpy as np
import pytest
import torch
import yaml

from data.exporters import (
    ExportObserver,
    HDF5TrajectoryExporter,
    LeRobotTrajectoryExporter,
)
from data.trajectory import parse_field_assignment, schema_profile
from envs.annotations import (
    annotate_cooperative_stop_frame,
    update_cooperative_stop_viewer_labels,
)
from envs.runtime import (
    CallablePolicy,
    RenderRequest,
    RunnerConfig,
    SimulationRunner,
    SimulationTransition,
)
from envs.two_robot_carry_env import (
    CooperativeStopEnvConfig,
    TwoRobotCooperativeStopEnv,
)
from envs.video import StreamingVideoObserver
from models import (
    OneStepMLPWorldModel,
    OneStepMLPWorldModelConfig,
    WorldModelInputs,
)
from models.wam import RWMARConfig, WorldModelRolloutInputs, WorldModelSequenceInputs

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
            "response_progress": self.value / 2,
            "coordination_error": 0.25,
        }
        return self._observation(), float(np.asarray(action).sum()), done, info

    def render(self, *, camera, width, height):
        del camera
        return np.full((height, width, 3), self.value, dtype=np.uint8)

    def close(self):
        pass

    def _observation(self):
        state = np.asarray([self.value, self.value + 1], dtype=np.float32)
        image = np.full((6, 8, 3), self.value, dtype=np.uint8)
        return {
            "robot_0": {"state": state, "image": image},
            "robot_1": {"state": -state, "image": image},
            "proprioception": np.concatenate([state, -state]),
            "privileged_state": {
                "state": state,
                "object_pose": np.asarray([self.value, 0.0, 0.0], dtype=np.float32),
            },
        }


def test_module_dependency_direction_is_enforced():
    _assert_no_package_import("models", forbidden={"data", "envs"})
    _assert_no_package_import("envs", forbidden={"data", "models"})


def test_one_step_mlp_world_model_consumes_only_explicit_tensor_inputs():
    model = OneStepMLPWorldModel(
        OneStepMLPWorldModelConfig(
            state_dim=3, action_dim=2, hidden_dim=8, hidden_layers=2
        )
    )
    inputs = WorldModelInputs(
        state=torch.zeros(4, 3),
        action=torch.ones(4, 2),
    )

    output = model(inputs)

    assert output.next_state.shape == (4, 3)
    assert output.reward.shape == (4, 1)
    assert output.done_logit.shape == (4, 1)
    assert output.success_logit.shape == (4, 1)
    assert output.failure_logit.shape == (4, 1)
    with pytest.raises(ValueError, match="action must end"):
        model(WorldModelInputs(state=torch.zeros(4, 3), action=torch.zeros(4, 3)))


def test_phase1_wam_contracts_are_reserved_without_environment_dependencies():
    config = RWMARConfig()
    history = WorldModelSequenceInputs(
        states=torch.zeros(4, config.history_horizon, config.state_dim),
        past_actions=torch.zeros(
            4, config.history_horizon - 1, config.action_dim
        ),
        valid_mask=torch.ones(4, config.history_horizon, dtype=torch.bool),
    )
    rollout = WorldModelRolloutInputs(
        history=history,
        candidate_actions=torch.zeros(
            4, config.train_forecast_horizon, config.action_dim
        ),
        num_particles=8,
    )

    assert rollout.history.states.shape == (4, 32, 22)
    assert rollout.candidate_actions.shape == (4, 16, 8)
    with pytest.raises(ValueError, match="num_particles"):
        WorldModelRolloutInputs(
            history=history,
            candidate_actions=rollout.candidate_actions,
            num_particles=0,
        )


def test_phase1_task_config_matches_rwm_ar_contract():
    payload = yaml.safe_load(
        (ROOT / "configs/wam/cooperative_stop_v1.yaml").read_text(encoding="utf-8")
    )
    data = payload["data"]
    model = payload["model"]
    config = RWMARConfig(
        state_dim=data["state_dim"],
        action_dim=data["action_dim"],
        history_horizon=data["history_horizon"],
        train_forecast_horizon=data["train_forecast_horizon"],
        planning_horizon=data["planning_horizon"],
        encoder_hidden_dim=model["encoder_hidden_dim"],
        gru_hidden_dim=model["gru_hidden_dim"],
        gru_layers=model["gru_layers"],
        dropout=model["dropout"],
        min_log_std=model["min_log_std"],
        max_log_std=model["max_log_std"],
    )

    assert payload["phase"] == "phase1_rwm_ar"
    assert config == RWMARConfig()
    assert payload["evaluation"]["open_loop_horizons"] == [1, 5, 10, 20, 40]
    assert payload["checkpoint"]["format_version"] == "wam.rwm_ar/1"


def test_realtime_runner_streams_aligned_transitions():
    env = _FakeEnvironment()
    now = [0.0]
    sleeps: list[float] = []
    policy_observations: list[Mapping[str, Any]] = []

    def sleep(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    observer = _RecordingObserver()

    def act(observation):
        policy_observations.append(observation)
        return np.asarray([observation["proprioception"][0]])

    runner = SimulationRunner(
        env,
        CallablePolicy(act),
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
    assert observer.transitions[0].observation["proprioception"][0] == 0
    assert observer.transitions[0].next_observation["proprioception"][0] == 1
    assert all("privileged_state" not in item for item in policy_observations)
    assert "privileged_state" in observer.transitions[0].observation
    assert observer.transitions[0].images["head"].shape == (6, 8, 3)
    assert observer.transitions[0].next_images["head"][0, 0, 0] == 1


def test_hdf5_export_is_incremental_customizable_and_dtype_stable(tmp_path):
    env = _FakeEnvironment()
    schema = schema_profile("wam").with_overrides(
        add=(
            parse_field_assignment(
                "diagnostics.coordination=info.coordination_error::float32"
            ),
        ),
        drop=("observation.object",),
    )
    exporter = HDF5TrajectoryExporter(tmp_path, schema, stream_videos=True)
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
        assert file["data/diagnostics/coordination"].dtype == np.dtype("float32")
        assert file["data/observation/privileged_state"].shape == (2, 2)
        assert "data/observation/object" not in file
        assert file["data/task"].asstr()[0].startswith("carry the object")
    for stream in ("robot_0", "robot_1"):
        video_path = tmp_path / "videos" / "episode_000000" / f"{stream}.mp4"
        capture = cv2.VideoCapture(str(video_path))
        try:
            assert capture.isOpened()
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
            ok, frame = capture.read()
            assert ok
            assert float(np.std(frame)) < 1.0
        finally:
            capture.release()


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


def test_rollout_render_annotation_is_opt_in_and_receives_aligned_info():
    calls: list[int] = []

    def annotator(frame: np.ndarray, info: Mapping[str, Any]) -> np.ndarray:
        calls.append(int(info.get("step_count", 0)))
        result = frame.copy()
        result[0, 0] = 255
        return result

    observer = _RecordingObserver()
    runner = SimulationRunner(
        _FakeEnvironment(),
        CallablePolicy(lambda observation: np.ones(1, dtype=np.float32)),
        RunnerConfig(
            render=(
                RenderRequest("raw", "fixed", width=8, height=6),
                RenderRequest(
                    "annotated",
                    "fixed",
                    width=8,
                    height=6,
                    annotator=annotator,
                ),
            )
        ),
    )
    runner.run_episode(observers=(observer,))

    first = observer.transitions[0]
    assert first.images["raw"][0, 0, 0] == 0
    assert np.all(first.images["annotated"][0, 0] == 255)
    assert calls == [0, 0, 0]


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


def test_real_environment_runs_seeded_cooperative_stop_task():
    env = TwoRobotCooperativeStopEnv(
        CooperativeStopEnvConfig(
            include_camera_images=False,
            agent_camera_width=16,
            agent_camera_height=12,
        )
    )
    try:
        observation, info = env.reset(seed=0, randomize=False)
        assert env.geometry.task_center.tolist() == pytest.approx([0.0, 0.85])
        assert observation["proprioception"].shape == (22,)
        assert observation["robot_0"]["base_pose"].shape == (3,)
        assert observation["robot_0"]["base_velocity"].shape == (3,)
        assert observation["robot_0"]["gripper"].shape == (2,)
        assert observation["robot_0"]["base_effort"].shape == (3,)
        assert observation["robot_0"]["image"].shape == (12, 16, 3)
        assert observation["privileged_state"]["state"].shape == (34,)
        assert observation["privileged_state"]["braking_event"].shape == (10,)
        assert "object_pose" not in observation["robot_0"]
        assert info["geometry_source"] == "mujoco_xml"
        assert info["scenario"] == "standard"
        assert info["braking_agent"] in {0, 1}
        assert 2.0 <= info["brake_start_time"] <= 5.0

        next_observation, _, terminated, truncated, _ = env.step(env.scripted_action())
        assert not terminated
        assert not truncated
        assert next_observation["robot_0"]["base_effort"][1] > 0.0

        while not (terminated or truncated):
            next_observation, _, terminated, truncated, info = env.step(
                env.scripted_action()
            )
        assert terminated
        assert not truncated
        assert info["success"]
        assert info["pre_brake_motion_valid"]
        assert info["response_started"]
        assert info["follower_brake_steps"] >= env.cfg.min_gradual_brake_steps
        assert info["stop_hold_steps"] >= env.cfg.stop_hold_steps
        assert info["both_stopped"]
    finally:
        env.close()


def test_human_annotations_label_viewer_but_do_not_mutate_raw_frame():
    env = TwoRobotCooperativeStopEnv(
        CooperativeStopEnvConfig(include_camera_images=False)
    )
    try:
        observation, info = env.reset(seed=0, randomize=False)
        raw = np.full((180, 320, 3), 127, dtype=np.uint8)
        annotated = annotate_cooperative_stop_frame(raw, info)
        assert np.all(raw == 127)
        assert not np.array_equal(annotated, raw)
        assert np.array_equal(annotated[-20:], raw[-20:])

        class FakeViewer:
            def __init__(self):
                self.user_scn = mujoco.MjvScene(env.model, maxgeom=8)

            @staticmethod
            def lock():
                return nullcontext()

        viewer = FakeViewer()
        update_cooperative_stop_viewer_labels(viewer, observation, info)
        assert viewer.user_scn.ngeom == 2
        labels = [viewer.user_scn.geoms[index].label for index in range(2)]
        assert "BRAKE ROBOT" in labels[info["braking_agent"]]
        assert f"{info['brake_start_time']:.2f}s" in labels[info["braking_agent"]]
        assert labels[info["responding_agent"]] == "RESPONDER"
        for agent_id in range(2):
            position = viewer.user_scn.geoms[agent_id].pos
            assert position[:2] == pytest.approx(
                observation[f"robot_{agent_id}"]["base_pose"][:2]
            )
    finally:
        env.close()


def test_cooperative_stop_task_has_only_standard_scenario_and_seeded_event():
    with pytest.raises(ValueError, match="only 'standard'"):
        TwoRobotCooperativeStopEnv(
            CooperativeStopEnvConfig(
                scenario="private_gates", include_camera_images=False
            )
        )

    env = TwoRobotCooperativeStopEnv(
        CooperativeStopEnvConfig(include_camera_images=False)
    )
    try:
        events = []
        for seed in range(8):
            _, info = env.reset(seed=seed, randomize=False)
            events.append((info["braking_agent"], info["brake_start_step"]))
        assert {agent for agent, _ in events} == {0, 1}
        assert len({step for _, step in events}) > 1
    finally:
        env.close()


def test_stationary_policy_cannot_satisfy_cooperative_stop_success():
    env = TwoRobotCooperativeStopEnv(
        CooperativeStopEnvConfig(
            include_camera_images=False,
            brake_start_time_min=0.1,
            brake_start_time_max=0.1,
            max_response_time=0.5,
            episode_len=40,
        )
    )
    try:
        env.reset(seed=0, randomize=False)
        action = np.asarray([0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32)
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(action)
        assert terminated
        assert not truncated
        assert not info["success"]
        assert not info["pre_brake_motion_valid"]
        assert info["failure_reason"] == "response_timeout"
    finally:
        env.close()


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
