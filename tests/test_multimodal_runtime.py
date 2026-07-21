from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pytest

from envs.runtime import RenderRequest, RunnerConfig, SimulationRunner

CAMERAS = ("fixed", "robot_0_camera", "robot_1_camera")


class _MultirateEnvironment:
    control_dt = 0.05
    action_dim = 2

    def __init__(self, calibration_mutation: str | None = None) -> None:
        self.step_count = 0
        self.calibration_mutation = calibration_mutation
        self.calibration_calls: list[tuple[str, int]] = []

    def reset(self, seed=None, randomize=True):
        del seed, randomize
        self.step_count = 0
        return self._observation(), {"step_count": 0}

    def step(self, action):
        commanded = np.asarray(action, dtype=np.float32)
        self.step_count += 1
        done = self.step_count >= 4
        info = {
            "step_count": self.step_count,
            "success": done,
            "failure": False,
            "failure_reason": "none",
            "executed_action": 0.5 * commanded,
        }
        return self._observation(), 0.0, done, False, info

    def render(self, *, camera, width, height):
        camera_index = CAMERAS.index(camera)
        return np.full(
            (height, width, 3), self.step_count + 20 * camera_index, dtype=np.uint8
        )

    def camera_calibration(self, *, camera, width, height):
        self.calibration_calls.append((camera, self.step_count))
        if self.calibration_mutation == "not_mapping":
            return None
        camera_index = CAMERAS.index(camera)
        intrinsics = np.asarray(
            [
                [100.0 + camera_index, 0.0, width / 2.0],
                [0.0, 100.0 + camera_index, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        extrinsics = np.eye(4, dtype=np.float32)
        if camera == "robot_0_camera":
            extrinsics[0, 3] = float(self.step_count)
        elif camera == "robot_1_camera":
            extrinsics[1, 3] = -float(self.step_count)
        resolution: np.ndarray = np.asarray([height, width], dtype=np.int64)
        if self.calibration_mutation == "missing":
            return {"intrinsics": intrinsics, "extrinsics": extrinsics}
        if self.calibration_mutation == "shape":
            intrinsics = intrinsics[:2]
        elif self.calibration_mutation == "nonfinite":
            extrinsics[0, 0] = np.nan
        elif self.calibration_mutation == "resolution_dtype":
            resolution = resolution.astype(np.float32)
        elif self.calibration_mutation == "resolution_value":
            resolution = np.asarray([width, height], dtype=np.int64)
        return {
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "resolution": resolution,
        }

    def close(self):
        pass

    def _observation(self) -> dict[str, Any]:
        return {
            "proprioception": np.asarray(
                [self.step_count, self.step_count + 1], dtype=np.float32
            ),
            "robot_0": {"image": np.full((2, 2, 3), 255, dtype=np.uint8)},
            "privileged_state": {"cue": 1},
        }


class _RecordingPolicy:
    def __init__(self) -> None:
        self.observations: list[Mapping[str, Any]] = []

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        self.observations.append(observation)
        return np.ones(2, dtype=np.float32)


def test_multirate_render_timestamps_and_policy_view_are_explicit():
    policy = _RecordingPolicy()
    transitions = []
    environment = _MultirateEnvironment()

    class Observer:
        def on_episode_start(self, **kwargs):
            del kwargs

        def on_transition(self, transition):
            transitions.append(transition)

        def on_episode_end(self, summary):
            del summary

    runner = SimulationRunner(
        environment,
        policy,
        RunnerConfig(
            render=tuple(
                RenderRequest(camera, camera, width=8, height=6, fps=10.0)
                for camera in CAMERAS
            ),
            policy_observation_keys=("proprioception",),
            expose_rendered_images_to_policy=True,
            policy_image_streams=("fixed",),
            expose_task_to_policy=True,
            task_id="visual_event_stop",
            task="react to the visual event",
            policy_action_history=3,
        ),
    )

    runner.run_episode(observers=(Observer(),))

    assert [item.image_frame_indices["fixed"] for item in transitions] == [
        0,
        0,
        1,
        1,
    ]
    assert [item.next_image_frame_indices["fixed"] for item in transitions] == [
        0,
        1,
        1,
        2,
    ]
    assert [item.image_timestamps["fixed"] for item in transitions] == pytest.approx(
        [0.0, 0.0, 0.1, 0.1]
    )
    assert [
        item.timestamp - item.image_timestamps["fixed"] for item in transitions
    ] == pytest.approx([0.0, 0.05, 0.0, 0.05])
    assert all(
        item.image_timestamps == item.image_state_timestamps
        for item in transitions
    )
    assert environment.calibration_calls == [
        (camera, step_count)
        for step_count in (0, 2, 4)
        for camera in CAMERAS
    ]
    for transition in transitions:
        assert tuple(transition.camera_intrinsics) == CAMERAS
        assert tuple(transition.next_camera_intrinsics) == CAMERAS
        assert tuple(transition.camera_extrinsics) == CAMERAS
        assert tuple(transition.next_camera_extrinsics) == CAMERAS
        assert all(
            np.array_equal(resolution, np.asarray([6, 8], dtype=np.int64))
            for resolution in transition.camera_resolutions.values()
        )
        assert all(
            np.array_equal(resolution, np.asarray([6, 8], dtype=np.int64))
            for resolution in transition.next_camera_resolutions.values()
        )
    assert [
        transition.camera_extrinsics["fixed"][0, 3]
        for transition in transitions
    ] == [0.0, 0.0, 0.0, 0.0]
    assert [
        transition.camera_extrinsics["robot_0_camera"][0, 3]
        for transition in transitions
    ] == [0.0, 0.0, 2.0, 2.0]
    assert [
        transition.next_camera_extrinsics["robot_0_camera"][0, 3]
        for transition in transitions
    ] == [0.0, 2.0, 2.0, 4.0]
    assert [
        transition.camera_extrinsics["robot_1_camera"][1, 3]
        for transition in transitions
    ] == [0.0, 0.0, -2.0, -2.0]
    for transition in transitions:
        for camera in CAMERAS:
            if (
                transition.image_frame_indices[camera]
                == transition.next_image_frame_indices[camera]
            ):
                assert np.array_equal(
                    transition.camera_intrinsics[camera],
                    transition.next_camera_intrinsics[camera],
                )
                assert np.array_equal(
                    transition.camera_extrinsics[camera],
                    transition.next_camera_extrinsics[camera],
                )
    expected_keys = {
        "proprioception",
        "images",
        "image_timestamps",
        "image_frame_indices",
        "task",
        "past_executed_actions",
    }
    assert all(set(observation) == expected_keys for observation in policy.observations)
    assert [
        observation["past_executed_actions"].shape[0]
        for observation in policy.observations
    ] == [0, 1, 2, 3]
    assert np.all(policy.observations[1]["past_executed_actions"] == 0.5)
    assert all("robot_0" not in observation for observation in policy.observations)
    assert all(
        "privileged_state" not in observation for observation in policy.observations
    )


def test_policy_rgb_requires_an_explicit_unannotated_stream():
    request = RenderRequest(
        "fixed",
        "fixed",
        width=8,
        height=6,
        annotator=lambda frame, info: frame,
    )
    with pytest.raises(ValueError, match="policy_image_streams"):
        RunnerConfig(
            render=(request,),
            expose_rendered_images_to_policy=True,
        )
    with pytest.raises(ValueError, match="annotated streams"):
        RunnerConfig(
            render=(request,),
            expose_rendered_images_to_policy=True,
            policy_image_streams=("fixed",),
        )


def test_policy_allowlist_rejects_privileged_state():
    with pytest.raises(ValueError, match="privileged_state"):
        RunnerConfig(policy_observation_keys=("privileged_state",))


@pytest.mark.parametrize(
    ("mutation", "exception", "message"),
    (
        ("not_mapping", TypeError, "must return a mapping"),
        ("missing", KeyError, "missing fields"),
        ("shape", ValueError, "finite \\[3,3\\]"),
        ("nonfinite", ValueError, "finite \\[4,4\\]"),
        ("resolution_dtype", TypeError, "must contain integers"),
        ("resolution_value", ValueError, "resolution must be"),
    ),
)
def test_render_calibration_is_validated_fail_closed(
    mutation: str, exception: type[Exception], message: str
) -> None:
    runner = SimulationRunner(
        _MultirateEnvironment(calibration_mutation=mutation),
        _RecordingPolicy(),
        RunnerConfig(
            render=(RenderRequest("fixed", "fixed", width=8, height=6),),
        ),
    )
    with pytest.raises(exception, match=message):
        runner.run_episode()


def test_render_requires_camera_calibration_contract() -> None:
    environment = _MultirateEnvironment()
    environment.camera_calibration = None  # type: ignore[method-assign]
    runner = SimulationRunner(
        environment,
        _RecordingPolicy(),
        RunnerConfig(
            render=(RenderRequest("fixed", "fixed", width=8, height=6),),
        ),
    )
    with pytest.raises(TypeError, match="must implement camera_calibration"):
        runner.run_episode()
