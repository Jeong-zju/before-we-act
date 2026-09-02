"""旁路记录 MARS CARE 闭环的逐步视频和 telemetry。

The recorder is deliberately observer-only.  It receives already-created
policy inputs/actions and privileged metrics from the evaluator, but is never
called by the policy and never changes an action.  Every transition is flushed
to disk before the next one is simulated, so a simulator or process failure
leaves an auditable prefix instead of losing the episode.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


FORMAT_VERSION = "before-we-act.care-mars-rollout-telemetry/1"
VIDEO_FORMAT_VERSION = "before-we-act.care-mars-rollout-video/1"


def _json_value(value: Any) -> Any:
    """Convert nested numeric values to JSON, representing non-finite as null."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return _json_value(value.detach().cpu().numpy())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if value is None or isinstance(value, (str, bytes)):
        return value.decode() if isinstance(value, bytes) else value
    return str(value)


def _as_rgb(value: Any) -> np.ndarray:
    frame = np.asarray(value)
    if frame.ndim == 4:
        if frame.shape[0] != 1:
            raise ValueError(f"MARS recorder expects one RGB frame, got {frame.shape}")
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[-1] < 3:
        raise ValueError(f"MARS recorder RGB shape differs: {frame.shape}")
    frame = frame[..., :3]
    if not np.issubdtype(frame.dtype, np.number) or not np.isfinite(frame).all():
        raise ValueError("MARS recorder RGB must be finite numeric data")
    return np.clip(frame, 0, 255).astype(np.uint8, copy=False)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def derive_events(
    task: str,
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """Return transition events from privileged observer metrics.

    Events are diagnostic labels only.  The hammer ``strike_proxy`` and cup
    ``alignment_proxy`` are explicitly marked as proxies because those tasks
    do not expose an official contact/alignment predicate in the current
    environment.  Shoes and stack use the official predicates exposed by
    ``evaluate()``.
    """

    before = previous or {}
    old_pred = before.get("factorized_predicates", {})
    new_pred = current.get("factorized_predicates", {})

    def changed(key: str) -> bool:
        return bool(new_pred.get(key, False)) and not bool(old_pred.get(key, False))

    events: dict[str, bool] = {
        "stage_changed": before.get("stage_id") not in (None, current.get("stage_id")),
        "success": bool(current.get("success", False)) and not bool(before.get("success", False)),
        "collision_or_drop": bool(current.get("collision_or_drop", False))
        and not bool(before.get("collision_or_drop", False)),
        "robot_conflict": bool(current.get("robot_conflict", False))
        and not bool(before.get("robot_conflict", False)),
    }
    events["progress_increase"] = float(current.get("progress", 0.0)) > float(
        before.get("progress", 0.0)
    ) + 1e-6

    if task == "place_cube_in_cup":
        old_distance = float(old_pred.get("horizontal_distance", np.inf))
        new_distance = float(new_pred.get("horizontal_distance", np.inf))
        events.update(
            {
                "alignment_proxy": new_distance < 0.08 and old_distance >= 0.08,
                "valid_rotation": changed("valid_rotation"),
            }
        )
    elif task == "strike_cube_hard":
        old_distance = float(old_pred.get("hammer_functional_point_distance", np.inf))
        new_distance = float(new_pred.get("hammer_functional_point_distance", np.inf))
        events.update(
            {
                "strike_proxy": new_distance < 0.05 and old_distance >= 0.05,
                "strike_proxy_is_not_contact": True,
            }
        )
    elif task == "three_robots_place_shoes":
        events.update(
            {
                "left_grasp": changed("is_shoe_left_grasped"),
                "right_grasp": changed("is_shoe_right_grasped"),
                "left_in_box": changed("shoe_left_in_box"),
                "right_in_box": changed("shoe_right_in_box"),
                "lid_on_box": changed("lid_on_box"),
            }
        )
    elif task == "four_robots_stack_cube":
        # The canonical name is is_cubeA_on_cubeB; older reports used the
        # reversed spelling and are normalized in mars_care_runtime.py.
        events.update(
            {
                "cubeA_on_cubeB": changed("is_cubeA_on_cubeB"),
                "cubeB_placed": changed("cubeB_placed"),
            }
        )
        flags = new_pred.get("grasp_flags", ())
        old_flags = old_pred.get("grasp_flags", ())
        events["grasp_transition"] = any(
            bool(new) and not bool(old)
            for new, old in zip(flags, old_flags)
        )
    return events


@dataclass
class _VideoWriter:
    path: Path
    fps: float
    codec: str
    writer: cv2.VideoWriter | None = None
    shape: tuple[int, int, int] | None = None
    frames: int = 0

    def write(self, frame: np.ndarray) -> None:
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            self.writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*self.codec), self.fps, (width, height)
            )
            if not self.writer.isOpened():
                raise RuntimeError(f"failed to open MARS recorder video: {self.path}")
            self.shape = tuple(frame.shape)
        if tuple(frame.shape) != self.shape:
            raise ValueError(f"MARS recorder frame shape changed: {self.shape} -> {frame.shape}")
        assert self.writer is not None
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.frames += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class MarsCARERolloutRecorder:
    """Incremental per-episode recorder for MARS CARE diagnostics."""

    def __init__(
        self,
        root: str | Path,
        *,
        task: str,
        seed: int,
        arms: Sequence[int],
        fps: float = 20.0,
        codec: str = "mp4v",
        record_candidate_plans: bool = True,
    ) -> None:
        if fps <= 0 or len(codec) != 4:
            raise ValueError("MARS recorder fps/codec is invalid")
        self.root = Path(root)
        self.task = str(task)
        self.seed = int(seed)
        self.arms = tuple(int(arm) for arm in arms)
        self.fps = float(fps)
        self.codec = codec
        self.record_candidate_plans = bool(record_candidate_plans)
        self.telemetry_path = self.root / "telemetry.jsonl"
        self.summary_path = self.root / "episode.json"
        self.array_root = self.root / "arrays"
        self.video_root = self.root / "videos"
        self._videos = {
            arm: _VideoWriter(self.video_root / f"panda-{arm}.mp4", self.fps, codec)
            for arm in self.arms
        }
        self._stream = None
        self._previous_metrics: Mapping[str, Any] | None = None
        self._steps = 0
        self._started = False
        self._closed = False
        self._metadata: dict[str, Any] = {}
        self._trace = hashlib.sha256()

    def start(self, observation: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None) -> None:
        if self._started:
            raise RuntimeError("MARS recorder episode already started")
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"refusing to mix stale recorder output: {self.root}")
        self.root.mkdir(parents=True, exist_ok=False)
        self.array_root.mkdir()
        self.video_root.mkdir()
        self._stream = self.telemetry_path.open("a", encoding="utf-8")
        self._metadata = {
            "format_version": FORMAT_VERSION,
            "video_format_version": VIDEO_FORMAT_VERSION,
            "task": self.task,
            "seed": self.seed,
            "arms": list(self.arms),
            "fps": self.fps,
            "codec": self.codec,
            "observer_only": True,
            "privileged_state_returned_to_policy": False,
            "metadata": dict(metadata or {}),
        }
        self._started = True
        self._write_line({"type": "episode_start", **self._metadata})
        # Save the initial local RGB frame as step 0.  The corresponding
        # action is recorded by record_step, so video and telemetry are aligned.
        del observation

    def _write_line(self, value: Mapping[str, Any]) -> None:
        if self._stream is None:
            raise RuntimeError("MARS recorder is not started")
        self._stream.write(json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True) + "\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())

    @staticmethod
    def _first_actions(plans: Mapping[str, np.ndarray]) -> dict[str, Any]:
        return {key: np.asarray(value)[0] for key, value in plans.items()}

    def record_step(
        self,
        *,
        step: int,
        observation_before: Mapping[str, Any],
        observation_after: Mapping[str, Any],
        qpos_before: Mapping[str, np.ndarray],
        qpos_after: Mapping[str, np.ndarray],
        qpos_normalized: Mapping[str, np.ndarray],
        reference_plans: Mapping[str, np.ndarray],
        base_plans: Mapping[str, np.ndarray],
        candidates: Sequence[np.ndarray],
        candidate_legality: Sequence[Sequence[bool]],
        selected: Sequence[int],
        masked_lower: Any,
        best_lower: Sequence[float],
        reason_names: Sequence[str],
        illegal: Any,
        learned_unsafe: Any,
        assembly: Mapping[str, Any],
        action_before_canonicalize: Mapping[str, np.ndarray],
        action_applied: Mapping[str, np.ndarray],
        action_bounds: Mapping[str, Mapping[str, np.ndarray]],
        diagnostics: Mapping[str, Any],
        physical: Mapping[str, Any],
        info: Mapping[str, Any],
    ) -> None:
        if not self._started or self._closed:
            raise RuntimeError("MARS recorder is not active")
        if int(step) != self._steps:
            raise ValueError(f"MARS recorder step is not contiguous: {step} != {self._steps}")
        frames: dict[str, int] = {}
        for arm in self.arms:
            key = f"panda-{arm}"
            frame = _as_rgb(observation_before["sensor_data"][f"head_camera_agent{arm}"]["rgb"])
            self._videos[arm].write(frame)
            frames[key] = self._videos[arm].frames

        arrays: dict[str, np.ndarray] = {}
        for arm in self.arms:
            key = f"panda-{arm}"
            arrays[f"qpos_before_{arm}"] = np.asarray(qpos_before[key], dtype=np.float32)
            arrays[f"qpos_after_{arm}"] = np.asarray(qpos_after[key], dtype=np.float32)
            arrays[f"qpos_normalized_{arm}"] = np.asarray(qpos_normalized[key], dtype=np.float32)
            arrays[f"reference_plan_{arm}"] = np.asarray(reference_plans[key], dtype=np.float32)
            arrays[f"base_plan_{arm}"] = np.asarray(base_plans[key], dtype=np.float32)
            arrays[f"action_before_canonicalize_{arm}"] = np.asarray(action_before_canonicalize[key], dtype=np.float32)
            arrays[f"action_applied_{arm}"] = np.asarray(action_applied[key], dtype=np.float32)
        if self.record_candidate_plans:
            for arm, value in zip(self.arms, candidates):
                arrays[f"candidates_{arm}"] = np.asarray(value, dtype=np.float32)
        array_path = self.array_root / f"step_{int(step):06d}.npz"
        _atomic_npz(array_path, **arrays)

        clipping = {}
        grippers = {}
        for arm in self.arms:
            key = f"panda-{arm}"
            before = np.asarray(action_before_canonicalize[key], dtype=np.float32)
            applied = np.asarray(action_applied[key], dtype=np.float32)
            bounds = action_bounds[key]
            clipping[key] = {
                "changed_by_canonicalize": bool(not np.array_equal(before, applied)),
                "elements_clipped": int(np.count_nonzero(before != applied)),
                "low": np.asarray(bounds["low"], dtype=np.float32),
                "high": np.asarray(bounds["high"], dtype=np.float32),
            }
            grippers[key] = {
                "reference": float(np.asarray(reference_plans[key])[0, 7]),
                "applied": float(applied[7]),
                "before": float(np.asarray(qpos_before[key]).reshape(-1)[-1]),
            }
            self._trace.update(applied.tobytes())

        events = derive_events(self.task, self._previous_metrics, physical)
        row = {
            "type": "step",
            "step": int(step),
            "frames": frames,
            "array_path": str(array_path.relative_to(self.root)),
            "qpos_before": qpos_before,
            "qpos_after": qpos_after,
            "qpos_normalized": qpos_normalized,
            "reference_first_action": self._first_actions(reference_plans),
            "base_first_action": self._first_actions(base_plans),
            "action_before_canonicalize": action_before_canonicalize,
            "action_applied": action_applied,
            "gripper": grippers,
            "action_bounds": clipping,
            "candidate_first_actions": [np.asarray(value)[:, 0] for value in candidates],
            "candidate_legality": candidate_legality,
            "selected_candidate": selected,
            "masked_lower": masked_lower,
            "best_lower": best_lower,
            "selection_reason": reason_names,
            "illegal_mask": illegal,
            "learned_unsafe_mask": learned_unsafe,
            "assembly": assembly,
            "care_diagnostics": diagnostics,
            "privileged_metrics": physical,
            "events": events,
            "simulator_info": info,
        }
        self._write_line(row)
        self._previous_metrics = dict(physical)
        self._steps += 1
        del observation_after

    def finish(
        self,
        *,
        success: bool,
        final_observation: Mapping[str, Any] | None,
        final_info: Mapping[str, Any] | None,
        final_physical: Mapping[str, Any] | None,
        status: str = "complete",
    ) -> dict[str, Any]:
        if not self._started or self._closed:
            raise RuntimeError("MARS recorder is not active")
        for writer in self._videos.values():
            writer.close()
        video_streams = {}
        for arm, writer in self._videos.items():
            if writer.frames and (not writer.path.is_file() or writer.path.stat().st_size == 0):
                raise RuntimeError(f"MARS recorder video was not finalized: {writer.path}")
            video_streams[f"panda-{arm}"] = {
                "path": str(writer.path.relative_to(self.root)),
                "frames": writer.frames,
                "verified_shape": writer.shape,
                "bytes": writer.path.stat().st_size if writer.path.is_file() else 0,
                "sha256": _sha256(writer.path) if writer.path.is_file() else None,
            }
        summary = {
            **self._metadata,
            "status": str(status),
            "success": bool(success),
            "steps": self._steps,
            "action_trace_sha256": self._trace.hexdigest(),
            "final_info": dict(final_info or {}),
            "final_physical_metrics": dict(final_physical or {}),
            "video_streams": video_streams,
        }
        self._write_line({"type": "episode_end", **summary})
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        self._closed = True
        _atomic_json(self.summary_path, summary)
        del final_observation
        return summary

    def abort(self, *, error: BaseException) -> None:
        """Close writers while retaining the already flushed telemetry prefix."""

        if not self._started or self._closed:
            return
        try:
            self._write_line(
                {
                    "type": "episode_abort",
                    "steps": self._steps,
                    "error": f"{type(error).__name__}: {error}",
                    "action_trace_sha256_prefix": self._trace.hexdigest(),
                }
            )
        finally:
            for writer in self._videos.values():
                writer.close()
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            self._closed = True


__all__ = [
    "FORMAT_VERSION",
    "MarsCARERolloutRecorder",
    "VIDEO_FORMAT_VERSION",
    "derive_events",
]
