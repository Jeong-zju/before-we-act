from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import cv2
import numpy as np
import pytest

from envs.runtime import RenderRequest, RunnerConfig, SimulationRunner
from envs.video import StreamingVideoObserver
from scripts.rollout_multimodal_wam import (
    POLICY_STREAM,
    VIDEO_STREAM,
    _aggregate_records,
    _episode_pairs,
    _execution_evidence,
    _finalize_video_evidence,
    _prepare_output_directory,
    _probe_video,
    _validate_settings,
    _video_path,
)


class _FourStepEnvironment:
    control_dt = 0.05

    def __init__(self) -> None:
        self.step_count = 0

    def reset(
        self, seed: int | None = None, randomize: bool = True
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del seed, randomize
        self.step_count = 0
        return self._observation(), {"success": False}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        assert action.shape == (1,)
        self.step_count += 1
        terminated = self.step_count == 4
        info = {"success": terminated, "executed_action": action.copy()}
        return self._observation(), float(terminated), terminated, False, info

    def render(self, *, camera: str, width: int, height: int) -> np.ndarray:
        assert camera == "fixed"
        value = min(255, 20 + self.step_count * 50)
        return np.full((height, width, 3), value, dtype=np.uint8)

    def camera_calibration(
        self, *, camera: str, width: int, height: int
    ) -> dict[str, np.ndarray]:
        assert camera == "fixed"
        return {
            "intrinsics": np.eye(3, dtype=np.float32),
            "extrinsics": np.eye(4, dtype=np.float32),
            "resolution": np.asarray([height, width], dtype=np.int64),
        }

    def close(self) -> None:
        pass

    def _observation(self) -> dict[str, np.ndarray]:
        return {"proprioception": np.asarray([self.step_count], dtype=np.float32)}


class _CapturingPolicy:
    def __init__(self) -> None:
        self.image_keys: list[tuple[str, ...]] = []
        self.frame_indices: list[int] = []

    def reset(self) -> None:
        self.image_keys.clear()
        self.frame_indices.clear()

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        images = observation["images"]
        indices = observation["image_frame_indices"]
        assert isinstance(images, Mapping)
        assert isinstance(indices, Mapping)
        self.image_keys.append(tuple(sorted(str(key) for key in images)))
        self.frame_indices.append(int(indices[POLICY_STREAM]))
        assert np.asarray(images[POLICY_STREAM]).shape == (96, 96, 3)
        return np.zeros(1, dtype=np.float32)


def _settings(**overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "train_seeds": (101,),
        "tasks": ("visual_event_stop",),
        "cue_variants": (0, 1),
        "physical_seed_count": 2,
        "max_steps": 4,
        "video_width": 64,
        "video_height": 48,
        "torch_threads": 1,
        "video_episodes_per_task": 2,
        "video_codec": "mp4v",
        "control_hz": 20.0,
        "visual_hz": 10.0,
    }
    result.update(overrides)
    return result


def test_rollout_matrix_and_settings_are_fail_closed() -> None:
    assert _episode_pairs(
        physical_seed_start=710100,
        physical_seed_count=2,
        cue_variants=(0, 1),
    ) == ((710100, 0), (710100, 1), (710101, 0), (710101, 1))

    _validate_settings(_settings())
    for invalid in (
        {"train_seeds": (101, 101)},
        {"tasks": ("not_a_task",)},
        {"cue_variants": (0, 0)},
        {"cue_variants": (2,)},
        {"physical_seed_count": 0},
        {"video_width": 0},
        {"video_episodes_per_task": 5},
        {"video_codec": "bad"},
        {"visual_hz": 21.0},
    ):
        with pytest.raises(ValueError):
            _validate_settings(_settings(**invalid))


def test_rollout_dual_stream_isolated_and_terminal_mp4(tmp_path: Path) -> None:
    env = _FourStepEnvironment()
    policy = _CapturingPolicy()
    path = tmp_path / "rollout.mp4"
    runner = SimulationRunner(
        env,
        policy,
        RunnerConfig(
            max_steps=4,
            render=(
                RenderRequest(
                    POLICY_STREAM,
                    "fixed",
                    width=96,
                    height=96,
                    fps=10.0,
                ),
                RenderRequest(VIDEO_STREAM, "fixed", width=64, height=48),
            ),
            policy_observation_keys=("proprioception",),
            expose_rendered_images_to_policy=True,
            policy_image_streams=(POLICY_STREAM,),
        ),
    )
    with StreamingVideoObserver(
        path,
        stream=VIDEO_STREAM,
        fps=20.0,
        frame_getter=lambda transition: transition.next_images[VIDEO_STREAM],
    ) as video:
        summary = runner.run_episode(observers=(video,))
        frames_written = video.frames_written

    assert summary.steps == frames_written == 4
    assert policy.image_keys == [(POLICY_STREAM,)] * 4
    assert policy.frame_indices == [0, 0, 1, 1]
    probe = _probe_video(path)
    assert probe == {
        "opened": True,
        "container_frames": 4,
        "decoded_frames": 4,
        "width": 64,
        "height": 48,
        "fps": 20.0,
    }
    capture = cv2.VideoCapture(str(path))
    last_frame = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            last_frame = frame
    finally:
        capture.release()
    assert last_frame is not None
    assert float(np.mean(last_frame)) > 190.0

    record = {
        "train_seed": 101,
        "task_id": "visual_event_stop",
        "physical_seed": 710100,
        "cue_id": 0,
        "episode_seed": 1420200,
        "success": True,
        "failure": False,
        "failure_reason": "",
        "steps": 4,
        "action_source": "m1_state_vision_future",
        "fallback_used": False,
        "privileged_observation_seen": False,
        "actions_finite_and_bounded": True,
    }
    with pytest.raises(RuntimeError, match="video contract"):
        _finalize_video_evidence(
            path,
            record=record,
            frames_written=3,
            settings=_settings(),
        )
    assert not path.with_suffix(".json").exists()
    evidence = _finalize_video_evidence(
        path,
        record=record,
        frames_written=frames_written,
        settings=_settings(),
    )
    assert evidence["verified"] is True
    assert evidence["terminal_frame_included"] is True
    assert evidence["video_stream_exposed_to_policy"] is False
    assert all(evidence["checks"].values())
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["video_sha256"] == evidence["video_sha256"]


def test_rollout_execution_contract_rejects_leakage_and_fallback() -> None:
    policy = SimpleNamespace(
        actions_finite_and_bounded=True,
        replan_events=2,
        cold_replan_events=2,
        warm_replan_events=0,
    )
    diagnostics = {
        "action_source": "m1_state_vision_future",
        "fallback_used": False,
        "privileged_state_seen": False,
        "presented_observation_paths": ["images.fixed", "proprioception"],
        "consumed_observation_paths": ["images.fixed", "proprioception"],
    }
    result = _execution_evidence(policy, diagnostics)
    assert all(result["execution_checks"].values())
    assert result["cold_replan_events"] == 2

    mutations = (
        {"action_source": "legacy_joint_wam_direct"},
        {"fallback_used": True},
        {"privileged_state_seen": True},
        {"consumed_observation_paths": ["images.rollout", "images.fixed"]},
        {"consumed_observation_paths": ["proprioception"]},
    )
    for mutation in mutations:
        invalid = dict(diagnostics, **mutation)
        with pytest.raises(RuntimeError, match="execution contract"):
            _execution_evidence(policy, invalid)

    policy.actions_finite_and_bounded = False
    with pytest.raises(RuntimeError, match="execution contract"):
        _execution_evidence(policy, diagnostics)


def test_rollout_aggregation_paths_and_stale_output(tmp_path: Path) -> None:
    records = [
        {
            "train_seed": train_seed,
            "task_id": task_id,
            "physical_seed": 710100 + offset,
            "cue_id": cue_id,
            "success": success,
            "steps": 4,
            "total_reward": float(success),
        }
        for train_seed, task_id, offset, cue_id, success in (
            (101, "visual_event_stop", 0, 0, True),
            (101, "visual_event_stop", 0, 1, False),
            (202, "visual_target_select", 1, 0, True),
            (202, "visual_target_select", 1, 1, True),
        )
    ]
    result = _aggregate_records(records)
    assert result["overall"]["successes"] == 3
    assert result["overall"]["episodes"] == 4
    assert result["overall"]["success_rate"] == 0.75
    assert result["by_task"]["visual_event_stop"]["success_rate"] == 0.5
    assert result["by_train_seed"]["202"]["success_rate"] == 1.0
    assert result["by_cue"]["1"]["success_rate"] == 0.5
    with pytest.raises(ValueError, match="duplicate"):
        _aggregate_records([records[0], records[0]])

    expected = (
        tmp_path
        / "videos"
        / "train_seed_101"
        / "visual_event_stop"
        / "physical_seed_710100_cue_0.mp4"
    )
    assert _video_path(
        tmp_path / "videos",
        train_seed=101,
        task_id="visual_event_stop",
        physical_seed=710100,
        cue_id=0,
    ) == expected

    output = tmp_path / "output"
    _prepare_output_directory(output)
    _prepare_output_directory(output)
    (output / "stale.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="stale"):
        _prepare_output_directory(output)
