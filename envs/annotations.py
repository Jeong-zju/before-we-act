"""Human-facing annotations kept outside raw simulation observations."""

from __future__ import annotations

from typing import Any, Mapping

import cv2
import mujoco
import numpy as np

from envs.runtime import Observation


def annotate_cooperative_stop_frame(
    frame: np.ndarray, info: Mapping[str, Any]
) -> np.ndarray:
    """Add braking-event status to a rollout RGB frame.

    The input is copied before drawing so raw renderer output can still be
    consumed by dataset exporters without text contamination.
    """

    annotated = np.asarray(frame, dtype=np.uint8).copy()
    if annotated.ndim != 3 or annotated.shape[2] != 3:
        raise ValueError("annotation input must have shape [height,width,3]")

    braking_agent = int(info.get("braking_agent", -1))
    responding_agent = int(info.get("responding_agent", -1))
    brake_time = float(info.get("brake_start_time", -1.0))
    event_active = bool(info.get("brake_event_active", False))
    step_count = int(info.get("step_count", 0))
    start_step = int(info.get("brake_start_step", -1))
    control_dt = float(
        info.get(
            "control_dt",
            brake_time / start_step if start_step > 0 and brake_time >= 0.0 else 0.0,
        )
    )
    current_time = step_count * control_dt

    if event_active:
        event_elapsed = max(0.0, current_time - brake_time)
        event_status = f"BRAKING ACTIVE | elapsed {event_elapsed:.2f}s"
    else:
        countdown = max(0.0, brake_time - current_time)
        event_status = f"BRAKE START IN {countdown:.2f}s"

    braking_speed = float(info.get("braking_agent_speed", 0.0))
    responding_speed = float(info.get("responding_agent_speed", 0.0))
    if not event_active:
        response_status = (
            "CRUISING" if bool(info.get("has_grasped", False)) else "PREPARING"
        )
    elif bool(info.get("both_stopped", False)):
        response_status = "BOTH STOPPED"
    elif bool(info.get("response_started", False)):
        response_status = "RESPONDING"
    else:
        response_status = "WAITING"

    lines = (
        f"BRAKE ROBOT: robot_{braking_agent} @ {brake_time:.2f}s",
        event_status,
        (
            f"RESPONDER: robot_{responding_agent} | {response_status} | "
            f"v={responding_speed:.2f} | brake v={braking_speed:.2f} m/s"
        ),
    )
    _draw_text_panel(annotated, lines)
    return annotated


def update_cooperative_stop_viewer_labels(
    viewer: Any,
    observation: Observation,
    info: Mapping[str, Any],
) -> None:
    """Attach 3D braking-role labels to the two robots in a passive viewer."""

    scene = viewer.user_scn
    if scene is None:
        return

    braking_agent = int(info.get("braking_agent", -1))
    brake_time = float(info.get("brake_start_time", -1.0))
    event_active = bool(info.get("brake_event_active", False))
    with viewer.lock():
        scene.ngeom = 0
        for agent_id in range(2):
            robot = observation[f"robot_{agent_id}"]
            base_pose = np.asarray(robot["base_pose"], dtype=np.float64)
            if agent_id == braking_agent:
                phase = "since" if event_active else "starts"
                label = f"BRAKE ROBOT | {phase} {brake_time:.2f}s"
                rgba = np.asarray([1.0, 0.25, 0.10, 1.0], dtype=np.float32)
            else:
                label = "RESPONDER"
                rgba = np.asarray([0.15, 0.55, 1.0, 1.0], dtype=np.float32)
            _append_viewer_label(
                scene,
                position=np.asarray([base_pose[0], base_pose[1], 0.42]),
                label=label,
                rgba=rgba,
            )


def _append_viewer_label(
    scene: mujoco.MjvScene,
    *,
    position: np.ndarray,
    label: str,
    rgba: np.ndarray,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo viewer user scene has no free geometry slots")
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LABEL,
        np.zeros(3, dtype=np.float64),
        position,
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba,
    )
    geom.label = label[:99]
    scene.ngeom += 1


def _draw_text_panel(frame: np.ndarray, lines: tuple[str, ...]) -> None:
    height, width = frame.shape[:2]
    font_scale = max(0.38, min(0.65, width / 1000.0))
    thickness = max(1, int(round(font_scale * 2.0)))
    line_height = max(16, int(round(27 * font_scale / 0.55)))
    margin = max(6, int(round(width * 0.012)))
    panel_height = margin * 2 + line_height * len(lines)
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (width - 1, min(height - 1, panel_height)),
        (12, 12, 12),
        thickness=-1,
    )
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0.0, dst=frame)
    for index, line in enumerate(lines):
        y = margin + (index + 1) * line_height - 4
        if y >= height:
            break
        cv2.putText(
            frame,
            line,
            (margin, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (245, 245, 245),
            thickness,
            cv2.LINE_AA,
        )


__all__ = [
    "annotate_cooperative_stop_frame",
    "update_cooperative_stop_viewer_labels",
]
