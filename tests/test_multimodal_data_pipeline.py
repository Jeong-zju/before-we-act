from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import h5py
import numpy as np
import pytest
import torch

from data.exporters.base import EpisodeMetadata
from data.exporters.hdf5 import HDF5TrajectoryExporter
from data.trajectory import MULTIMODAL_WAM_SCHEMA_VERSION, schema_profile
from envs.runtime import RolloutSummary
from train.multimodal_trajectory_dataset import MultimodalSequenceDataset

CAMERAS = ("fixed", "robot_0_camera", "robot_1_camera")


def test_multimodal_schema_is_exact_versioned_and_privilege_free() -> None:
    schema = schema_profile("wam_multimodal")
    alias = schema_profile("multimodal_wam")
    names = [field.name for field in schema.fields]

    assert schema.version == MULTIMODAL_WAM_SCHEMA_VERSION
    assert schema.profile == alias.profile == "wam_multimodal"
    assert schema.fields == alias.fields
    assert names == [
        "timestamp",
        "frame_index",
        "episode_index",
        "seed",
        "task.text",
        "task.id",
        "observation.state",
        "observation.images.fixed",
        "observation.image_timestamp.fixed",
        "observation.image_state_timestamp.fixed",
        "observation.image_frame_index.fixed",
        "action.commanded",
        "action.executed",
        "next_observation.state",
        "next_observation.images.fixed",
        "next_observation.image_timestamp.fixed",
        "next_observation.image_state_timestamp.fixed",
        "next_observation.image_frame_index.fixed",
        "camera.intrinsics.fixed",
        "camera.extrinsics.fixed",
        "camera.resolution.fixed",
        "next_camera.intrinsics.fixed",
        "next_camera.extrinsics.fixed",
        "next_camera.resolution.fixed",
        "event.visual_signal_active",
        "event.visual_signal_onset_step",
        "event.visual_signal_kind",
        "event.rendered_cue_variant",
        "reward",
        "terminated",
        "truncated",
        "done",
        "success",
        "failure",
        "failure_reason",
        "schema_version",
        "behavior_id",
        "environment_config",
        "randomization_config",
    ]
    assert not any("privileged" in field.source for field in schema.fields)
    assert not any(field.source.startswith("metadata.camera") for field in schema.fields)
    assert {
        field.name: field.source
        for field in schema.fields
        if field.name.startswith("event.")
    } == {
        "event.visual_signal_active": "info.visual_signal_active",
        "event.visual_signal_onset_step": "info.visual_signal_onset_step",
        "event.visual_signal_kind": "info.visual_signal_kind",
        "event.rendered_cue_variant": "info.rendered_cue_variant",
    }
    assert {field.name for field in schema.fields if field.is_image} == {
        "observation.images.fixed",
        "next_observation.images.fixed",
    }
    with pytest.raises(ValueError, match="unique"):
        schema_profile("wam_multimodal", cameras=("fixed", "fixed"))


def test_hdf5_roundtrip_and_loader_preserve_20hz_state_10hz_rgb_causality(
    tmp_path: Path,
) -> None:
    root = tmp_path / "episodes"
    first = _write_episode(root, episode_index=0, seed=11, state_offset=0.0)
    second = _write_episode(root, episode_index=1, seed=12, state_offset=100.0)

    with h5py.File(first, "r") as file:
        assert file.attrs["schema_version"] == MULTIMODAL_WAM_SCHEMA_VERSION
        assert file.attrs["camera_order_json"] == (
            '["fixed","robot_0_camera","robot_1_camera"]'
        )
        assert file["data/action/commanded"].shape == (4, 2)
        assert file["data/action/executed"].shape == (4, 2)
        assert file["data/observation/images/fixed"].dtype == np.dtype("uint8")
        assert file["data/next_observation/images/fixed"][1, 0, 0, 0] == 60
        assert file["data/observation/image_frame_index/fixed"][:].tolist() == [
            0,
            0,
            1,
            1,
        ]
        assert file["data/next_observation/image_frame_index/fixed"][:].tolist() == [
            0,
            1,
            1,
            2,
        ]
        assert file["data/event/visual_signal_active"][:].tolist() == [
            False,
            False,
            True,
            True,
        ]
        assert file["data/event/visual_signal_onset_step"][:].tolist() == [
            2,
            2,
            2,
            2,
        ]
        assert file["data/event/visual_signal_kind"].asstr()[:].tolist() == [
            "braking_light",
        ] * 4
        assert file["data/event/rendered_cue_variant"][:].tolist() == [7] * 4
        assert np.array_equal(
            file["data/camera/extrinsics/robot_0_camera"][0],
            file["data/camera/extrinsics/robot_0_camera"][1],
        )
        assert np.array_equal(
            file["data/next_camera/extrinsics/robot_0_camera"][0],
            file["data/camera/extrinsics/robot_0_camera"][0],
        )
        assert np.array_equal(
            file["data/next_camera/extrinsics/robot_0_camera"][1],
            file["data/camera/extrinsics/robot_0_camera"][2],
        )

    dataset = MultimodalSequenceDataset(
        paths=(first, second),
        history_horizon=3,
        forecast_horizon=3,
        state_dim=3,
        action_dim=2,
        camera_order=CAMERAS,
        hdf5_cache_size=1,
    )
    try:
        item = dataset[2]
        assert item["valid_mask"].tolist() == [True, True, True]
        assert item["images"].shape == (3, 3, 3, 6, 8)
        assert item["images"].dtype == torch.uint8
        assert item["image_frame_indices"][:, 0].tolist() == [0, 0, 1]
        assert item["image_age_seconds"][:, 0].tolist() == pytest.approx(
            [0.0, 0.05, 0.0]
        )
        assert item["target_image_frame_indices"][:, 0].tolist() == [1, 2, -1]
        assert item["target_image_age_seconds"][:2, 0].tolist() == pytest.approx(
            [0.05, 0.0]
        )
        assert item["forecast_mask"].tolist() == [True, True, False]
        assert item["target_image_valid_mask"][:, 0].tolist() == [True, True, False]
        assert item["past_actions"][-1].tolist() == pytest.approx([0.5, -0.5])
        assert item["past_commanded_actions"][-1].tolist() == pytest.approx(
            [1.0, -1.0]
        )
        assert item["task_text"] == "respond to the visual signal"
        assert item["task_id"] == "visual_event_stop"
        assert item["camera_resolution"].shape == (3, 3, 2)
        assert item["camera_resolution"].tolist() == [
            [[6, 8], [6, 8], [6, 8]],
            [[6, 8], [6, 8], [6, 8]],
            [[6, 8], [6, 8], [6, 8]],
        ]
        assert item["camera_extrinsics"][:, 0, 0, 3].tolist() == [0.0, 0.0, 0.0]
        assert item["camera_extrinsics"][:, 1, 0, 3].tolist() == [0.0, 0.0, 2.0]
        assert item["camera_extrinsics"][:, 2, 1, 3].tolist() == [0.0, 0.0, -2.0]
        assert item["target_camera_extrinsics"][:, 1, 0, 3].tolist() == [
            2.0,
            4.0,
            0.0,
        ]
        assert item["target_camera_resolution"].shape == (3, 3, 2)
        assert item["target_camera_resolution"][2].count_nonzero().item() == 0

        boundary = dataset[4]
        assert boundary["episode_index"].item() == 1
        assert boundary["episode_seed"].item() == 12
        assert boundary["valid_mask"].tolist() == [False, False, True]
        assert boundary["states"][-1, 0].item() == pytest.approx(100.0)
        assert boundary["states"][:-1].count_nonzero().item() == 0
        assert boundary["image_valid_mask"][:, 0].tolist() == [False, False, True]
        assert boundary["image_frame_indices"][:, 0].tolist() == [-1, -1, 0]
        assert boundary["camera_resolution"][:2].count_nonzero().item() == 0
        assert boundary["camera_resolution"][-1].tolist() == [
            [6, 8],
            [6, 8],
            [6, 8],
        ]
    finally:
        dataset.close()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("schema", "schema version"),
        ("camera_order", "camera order"),
        ("shape", "matching uint8 HWC RGB"),
        ("future", "leaks the future"),
        ("frame_reference", "frame references disagree"),
        ("next_calibration_shape", "intrinsics must be"),
        ("calibration_hold", "calibration sample-hold"),
        ("next_calibration_nonfinite", "calibration contains NaN/Inf"),
    ),
)
def test_multimodal_loader_rejects_bad_contracts(
    tmp_path: Path, mutation: str, message: str
) -> None:
    path = _write_episode(tmp_path / mutation, episode_index=0, seed=3)
    with h5py.File(path, "r+") as file:
        if mutation == "schema":
            file.attrs["schema_version"] = "wam.multimodal/bad"
        elif mutation == "camera_order":
            file.attrs["camera_order_json"] = '["other"]'
        elif mutation == "shape":
            del file["data/next_observation/images/fixed"]
            file["data/next_observation/images"].create_dataset(
                "fixed", data=np.zeros((4, 7, 8, 3), dtype=np.uint8)
            )
        elif mutation == "future":
            file["data/observation/image_timestamp/fixed"][:2] = 0.01
        elif mutation == "frame_reference":
            file["data/next_observation/image_frame_index/fixed"][0] = 1
        elif mutation == "next_calibration_shape":
            del file["data/next_camera/intrinsics/fixed"]
            file["data/next_camera/intrinsics"].create_dataset(
                "fixed", data=np.zeros((4, 2, 3), dtype=np.float32)
            )
        elif mutation == "calibration_hold":
            file["data/camera/extrinsics/robot_0_camera"][1, 0, 3] = 999.0
        elif mutation == "next_calibration_nonfinite":
            file["data/next_camera/extrinsics/robot_1_camera"][0, 0, 0] = np.nan
        else:  # pragma: no cover - parameter exhaustiveness.
            raise AssertionError(mutation)

    with pytest.raises((TypeError, ValueError), match=message):
        MultimodalSequenceDataset(
            paths=(path,),
            history_horizon=2,
            forecast_horizon=2,
            camera_order=CAMERAS,
        )


def test_multimodal_loader_rejects_stale_frames_and_sensor_skew(
    tmp_path: Path,
) -> None:
    stale = _write_episode(tmp_path / "stale", episode_index=0, seed=3)
    with h5py.File(stale, "r+") as file:
        file["data/observation/image_timestamp/fixed"][2:] = 0.0
    with pytest.raises(ValueError, match="frame age exceeds"):
        MultimodalSequenceDataset(
            paths=(stale,),
            camera_order=CAMERAS,
            max_frame_age_seconds=0.1,
        )

    skewed = _write_episode(tmp_path / "skewed", episode_index=0, seed=4)
    with h5py.File(skewed, "r+") as file:
        file["data/observation/image_timestamp/fixed"][2] = 0.074
    with pytest.raises(ValueError, match="sensor/state skew exceeds"):
        MultimodalSequenceDataset(
            paths=(skewed,),
            camera_order=CAMERAS,
            max_sensor_state_skew_seconds=0.025,
        )


def test_hdf5_export_refuses_overwrite_and_streams_only_current_rgb(
    tmp_path: Path,
) -> None:
    root = tmp_path / "episodes"
    path = _write_episode(
        root,
        episode_index=0,
        seed=5,
        stream_videos=True,
    )
    assert path.exists()
    video_dir = root / "videos" / "episode_000000"
    assert sorted(item.name for item in video_dir.glob("*.mp4")) == [
        "fixed.mp4",
        "robot_0_camera.mp4",
        "robot_1_camera.mp4",
    ]
    capture = cv2.VideoCapture(str(video_dir / "fixed.mp4"))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    finally:
        capture.release()

    exporter = HDF5TrajectoryExporter(
        root, schema_profile("wam_multimodal", cameras=CAMERAS)
    )
    metadata = _episode_metadata(episode_index=0, seed=5)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        exporter.start_episode(metadata)
    exporter.close()


def _write_episode(
    root: Path,
    *,
    episode_index: int,
    seed: int,
    state_offset: float = 0.0,
    stream_videos: bool = False,
) -> Path:
    schema = schema_profile("wam_multimodal", cameras=CAMERAS)
    exporter = HDF5TrajectoryExporter(root, schema, stream_videos=stream_videos)
    metadata = _episode_metadata(episode_index=episode_index, seed=seed)
    exporter.start_episode(metadata)
    total_reward = 0.0
    for step in range(4):
        transition = _transition(
            episode_index=episode_index,
            seed=seed,
            step=step,
            state_offset=state_offset,
        )
        exporter.write_transition(transition)
        total_reward += transition.reward
    exporter.end_episode(
        RolloutSummary(
            episode_index=episode_index,
            seed=seed,
            steps=4,
            total_reward=total_reward,
            terminated=True,
            truncated=False,
            elapsed_wall_seconds=0.01,
            final_info={"success": True},
        )
    )
    exporter.close()
    return root / f"episode_{episode_index:06d}.hdf5"


def _episode_metadata(*, episode_index: int, seed: int) -> EpisodeMetadata:
    return EpisodeMetadata(
        episode_index=episode_index,
        seed=seed,
        task="respond to the visual signal",
        fps=20.0,
        initial_observation={"proprioception": np.zeros(3, dtype=np.float32)},
        initial_info={},
    )


def _transition(
    *, episode_index: int, seed: int, step: int, state_offset: float
) -> SimpleNamespace:
    timestamp = step * 0.05
    current_frame_index = step // 2
    next_frame_index = (step + 1) // 2
    current_image_timestamp = current_frame_index * 0.1
    next_image_timestamp = next_frame_index * 0.1
    commanded = np.asarray([float(step), -float(step)], dtype=np.float32)
    executed = commanded * 0.5
    current_state = np.asarray(
        [state_offset + step, 1.0, -1.0], dtype=np.float32
    )
    next_state = current_state.copy()
    next_state[0] += 1.0
    metadata = {
        "schema_version": MULTIMODAL_WAM_SCHEMA_VERSION,
        "seed": seed,
        "task_id": "visual_event_stop",
        "behavior_id": "scripted_visual_oracle_v1",
        "environment_config": "{}",
        "randomization_config": "{}",
    }
    current_calibration = {
        camera: _camera_calibration(camera, current_frame_index)
        for camera in CAMERAS
    }
    next_calibration = {
        camera: _camera_calibration(camera, next_frame_index) for camera in CAMERAS
    }
    done = step == 3
    return SimpleNamespace(
        episode_index=episode_index,
        frame_index=step,
        timestamp=timestamp,
        observation={"proprioception": current_state},
        action=commanded,
        next_observation={"proprioception": next_state},
        reward=float(step),
        terminated=done,
        truncated=False,
        done=done,
        info={
            "executed_action": executed,
            "success": done,
            "failure": False,
            "failure_reason": "none",
            "visual_signal_active": step >= 2,
            "visual_signal_onset_step": 2,
            "visual_signal_kind": "braking_light",
            "rendered_cue_variant": 7,
        },
        task="respond to the visual signal",
        images={
            camera: _image(current_frame_index, camera_index)
            for camera_index, camera in enumerate(CAMERAS)
        },
        next_images={
            camera: _image(next_frame_index, camera_index)
            for camera_index, camera in enumerate(CAMERAS)
        },
        image_timestamps={camera: current_image_timestamp for camera in CAMERAS},
        next_image_timestamps={camera: next_image_timestamp for camera in CAMERAS},
        image_state_timestamps={
            camera: current_image_timestamp for camera in CAMERAS
        },
        next_image_state_timestamps={
            camera: next_image_timestamp for camera in CAMERAS
        },
        image_frame_indices={camera: current_frame_index for camera in CAMERAS},
        next_image_frame_indices={camera: next_frame_index for camera in CAMERAS},
        camera_intrinsics={
            camera: calibration[0]
            for camera, calibration in current_calibration.items()
        },
        next_camera_intrinsics={
            camera: calibration[0]
            for camera, calibration in next_calibration.items()
        },
        camera_extrinsics={
            camera: calibration[1]
            for camera, calibration in current_calibration.items()
        },
        next_camera_extrinsics={
            camera: calibration[1]
            for camera, calibration in next_calibration.items()
        },
        camera_resolutions={
            camera: calibration[2]
            for camera, calibration in current_calibration.items()
        },
        next_camera_resolutions={
            camera: calibration[2]
            for camera, calibration in next_calibration.items()
        },
        metadata=metadata,
    )


def _camera_calibration(
    camera: str, frame_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_index = CAMERAS.index(camera)
    intrinsics = np.asarray(
        [
            [100.0 + camera_index, 0.0, 4.0],
            [0.0, 100.0 + camera_index, 3.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    extrinsics = np.eye(4, dtype=np.float32)
    if camera == "robot_0_camera":
        extrinsics[0, 3] = float(2 * frame_index)
    elif camera == "robot_1_camera":
        extrinsics[1, 3] = -float(2 * frame_index)
    resolution = np.asarray([6, 8], dtype=np.int64)
    return intrinsics, extrinsics, resolution


def _image(frame_index: int, camera_index: int = 0) -> np.ndarray:
    value = frame_index * 60 + camera_index * 10
    return np.full((6, 8, 3), value, dtype=np.uint8)
