"""MuJoCo camera calibration helpers shared by simulation environments."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


_MUJOCO_TO_OPENCV_OPTICAL = np.diag([1.0, -1.0, -1.0])


def camera_calibration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    camera: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Return pinhole intrinsics and an OpenCV camera pose in world.

    MuJoCo cameras look along local ``-Z`` with ``+Y`` up.  OpenCV optical
    coordinates look along ``+Z`` with ``+Y`` down, hence the two-axis flip in
    the returned camera-to-world rotation.
    """

    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("camera calibration dimensions must be positive")
    camera_id = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, str(camera))
    )
    if camera_id < 0:
        raise ValueError(f"MuJoCo model has no camera {camera!r}")

    mujoco.mj_forward(model, data)
    fovy_degrees = float(model.cam_fovy[camera_id])
    if not np.isfinite(fovy_degrees) or not 0.0 < fovy_degrees < 180.0:
        raise ValueError(f"camera {camera!r} has invalid vertical field of view")
    focal = 0.5 * float(height) / np.tan(np.deg2rad(fovy_degrees) * 0.5)
    intrinsics = np.asarray(
        [
            [focal, 0.0, 0.5 * (width - 1)],
            [0.0, focal, 0.5 * (height - 1)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    mujoco_rotation = np.asarray(data.cam_xmat[camera_id], dtype=np.float64).reshape(
        3, 3
    )
    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[:3, :3] = (
        mujoco_rotation @ _MUJOCO_TO_OPENCV_OPTICAL
    ).astype(np.float32)
    extrinsics[:3, 3] = np.asarray(
        data.cam_xpos[camera_id], dtype=np.float32
    )
    parent_body_id = int(model.cam_bodyid[camera_id])
    parent_body_name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, parent_body_id
    )
    return {
        "camera_name": str(camera),
        "camera_id": camera_id,
        "parent_body_id": parent_body_id,
        "parent_body_name": str(parent_body_name or "world"),
        "fovy_degrees": fovy_degrees,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "resolution": np.asarray([height, width], dtype=np.int64),
        "model": "pinhole",
        "convention": "opencv_optical_camera_pose_in_world",
        "optical_frame": "opencv",
        "renderer_backend": "mujoco.Renderer",
    }


__all__ = ["camera_calibration"]
