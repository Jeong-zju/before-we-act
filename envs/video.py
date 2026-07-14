"""Incremental MP4 output for live environment rollouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from envs.runtime import Observation, RolloutSummary, SimulationTransition


class StreamingVideoObserver:
    """Encode one RGB stream as frames arrive; no episode is buffered in RAM."""

    def __init__(
        self,
        path: str | Path,
        *,
        stream: str,
        fps: float,
        codec: str = "mp4v",
        frame_getter: Callable[[SimulationTransition], np.ndarray] | None = None,
    ) -> None:
        if fps <= 0.0:
            raise ValueError("video fps must be positive")
        if len(codec) != 4:
            raise ValueError("OpenCV codec must contain four characters")
        self.path = Path(path)
        self.stream = stream
        self.fps = float(fps)
        self.codec = codec
        self.frame_getter = frame_getter
        self._writer: cv2.VideoWriter | None = None
        self._shape: tuple[int, int, int] | None = None
        self.frames_written = 0

    def on_episode_start(
        self,
        *,
        episode_index: int,
        seed: int | None,
        observation: Observation,
        info: Mapping[str, Any],
        task: str,
    ) -> None:
        del episode_index, seed, observation, info, task

    def on_transition(self, transition: SimulationTransition) -> None:
        if self.frame_getter is None:
            if self.stream not in transition.images:
                raise KeyError(f"rollout does not contain video stream {self.stream!r}")
            value = transition.images[self.stream]
        else:
            value = self.frame_getter(transition)
        frame = np.asarray(value, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("video frames must have shape [height,width,3]")
        if self._writer is None:
            self._open(frame)
        if frame.shape != self._shape:
            raise ValueError(
                f"video frame shape changed from {self._shape} to {frame.shape}"
            )
        assert self._writer is not None
        self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.frames_written += 1

    def on_episode_end(self, summary: RolloutSummary) -> None:
        del summary

    def _open(self, frame: np.ndarray) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*self.codec),
            self.fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"failed to open video writer for {self.path}")
        self._writer = writer
        self._shape = tuple(frame.shape)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "StreamingVideoObserver":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = ["StreamingVideoObserver"]
