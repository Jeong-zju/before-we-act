"""Exporter protocol and environment-to-dataset observer adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from envs.runtime import Observation, RolloutSummary, SimulationTransition


@dataclass(frozen=True)
class EpisodeMetadata:
    episode_index: int
    seed: int | None
    task: str
    fps: float
    initial_observation: Observation
    initial_info: Mapping[str, Any]


@runtime_checkable
class TrajectoryExporter(Protocol):
    """Incremental dataset writer; implementations must not buffer episodes."""

    def start_episode(self, metadata: EpisodeMetadata) -> None: ...

    def write_transition(self, transition: SimulationTransition) -> None: ...

    def end_episode(self, summary: RolloutSummary) -> None: ...

    def close(self) -> None: ...


class ExportObserver:
    """Fan one environment rollout into one or more independent exporters."""

    def __init__(
        self,
        exporters: Sequence[TrajectoryExporter],
        *,
        fps: float,
    ) -> None:
        if fps <= 0.0:
            raise ValueError("dataset fps must be positive")
        self.exporters = tuple(exporters)
        self.fps = float(fps)

    def on_episode_start(
        self,
        *,
        episode_index: int,
        seed: int | None,
        observation: Observation,
        info: Mapping[str, Any],
        task: str,
    ) -> None:
        metadata = EpisodeMetadata(
            episode_index=episode_index,
            seed=seed,
            task=task,
            fps=self.fps,
            initial_observation=observation,
            initial_info=info,
        )
        for exporter in self.exporters:
            exporter.start_episode(metadata)

    def on_transition(self, transition: SimulationTransition) -> None:
        for exporter in self.exporters:
            exporter.write_transition(transition)

    def on_episode_end(self, summary: RolloutSummary) -> None:
        for exporter in self.exporters:
            exporter.end_episode(summary)

    def close(self) -> None:
        errors: list[BaseException] = []
        for exporter in reversed(self.exporters):
            try:
                exporter.close()
            except BaseException as exc:  # preserve every backend cleanup attempt.
                errors.append(exc)
        if errors:
            raise RuntimeError(
                f"{len(errors)} dataset exporter(s) failed to close"
            ) from errors[0]

    def __enter__(self) -> "ExportObserver":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = ["EpisodeMetadata", "ExportObserver", "TrajectoryExporter"]
