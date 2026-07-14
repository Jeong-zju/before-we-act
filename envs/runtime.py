"""Environment-only rollout contracts and a real-time simulation runner.

The runner is deliberately model- and dataset-agnostic.  A policy only sees
the observation passed to ``act`` and observers only receive immutable rollout
records.  Dataset exporters can implement :class:`RolloutObserver` without the
environment importing them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

Observation = Mapping[str, Any]


@dataclass(frozen=True)
class RenderRequest:
    """One named RGB stream requested from an environment."""

    name: str
    camera: str
    width: int = 640
    height: int = 360

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("render stream name cannot be empty")
        if not self.camera:
            raise ValueError("camera name cannot be empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("render dimensions must be positive")


@dataclass(frozen=True)
class RunnerConfig:
    """Execution controls shared by batch and real-time simulation."""

    realtime: bool = False
    max_steps: int | None = None
    render: tuple[RenderRequest, ...] = ()
    task: str = "carry the object through the passage to the goal"

    def __post_init__(self) -> None:
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")


@dataclass(frozen=True)
class SimulationTransition:
    """One strictly aligned ``observation, action, next_observation`` record."""

    episode_index: int
    frame_index: int
    timestamp: float
    observation: Observation
    action: np.ndarray
    next_observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
    task: str
    images: Mapping[str, np.ndarray] = field(default_factory=dict)
    next_images: Mapping[str, np.ndarray] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return bool(self.terminated or self.truncated)


@dataclass(frozen=True)
class RolloutSummary:
    episode_index: int
    seed: int | None
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    elapsed_wall_seconds: float
    final_info: Mapping[str, Any]


@runtime_checkable
class SimulationEnvironment(Protocol):
    """Minimal environment surface consumed by :class:`SimulationRunner`."""

    @property
    def control_dt(self) -> float: ...

    def reset(
        self, seed: int | None = None, randomize: bool = True
    ) -> Observation | tuple[Observation, Mapping[str, Any]]: ...

    def step(
        self, action: np.ndarray
    ) -> (
        tuple[Observation, float, bool, Mapping[str, Any]]
        | tuple[Observation, float, bool, bool, Mapping[str, Any]]
    ): ...

    def render(self, *, camera: str, width: int, height: int) -> np.ndarray: ...

    def close(self) -> None: ...


@runtime_checkable
class Policy(Protocol):
    """A policy receives only the current observation."""

    def act(self, observation: Observation) -> np.ndarray: ...


@runtime_checkable
class RolloutObserver(Protocol):
    """Optional streaming consumer for rollouts (viewer, video, dataset, etc.)."""

    def on_episode_start(
        self,
        *,
        episode_index: int,
        seed: int | None,
        observation: Observation,
        info: Mapping[str, Any],
        task: str,
    ) -> None: ...

    def on_transition(self, transition: SimulationTransition) -> None: ...

    def on_episode_end(self, summary: RolloutSummary) -> None: ...


class CallablePolicy:
    """Adapt a callable to the explicit :class:`Policy` contract."""

    def __init__(self, function: Callable[[Observation], np.ndarray]) -> None:
        self.function = function

    def act(self, observation: Observation) -> np.ndarray:
        return np.asarray(self.function(observation))


class SimulationRunner:
    """Run an environment with optional wall-clock pacing and frame streams."""

    def __init__(
        self,
        env: SimulationEnvironment,
        policy: Policy,
        config: RunnerConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.env = env
        self.policy = policy
        self.config = config or RunnerConfig()
        self._clock = clock
        self._sleep = sleeper
        control_dt = float(env.control_dt)
        if not np.isfinite(control_dt) or control_dt <= 0.0:
            raise ValueError("environment control_dt must be finite and positive")
        self.control_dt = control_dt

    def run_episode(
        self,
        *,
        seed: int | None = None,
        episode_index: int = 0,
        randomize: bool = True,
        observers: Sequence[RolloutObserver] = (),
    ) -> RolloutSummary:
        observation, reset_info = self._reset(seed=seed, randomize=randomize)
        for observer in observers:
            observer.on_episode_start(
                episode_index=episode_index,
                seed=seed,
                observation=observation,
                info=reset_info,
                task=self.config.task,
            )

        images = self._render()
        start = self._clock()
        frame_index = 0
        total_reward = 0.0
        terminated = False
        truncated = False
        final_info: Mapping[str, Any] = reset_info

        while not (terminated or truncated):
            action = np.asarray(self.policy.act(observation), dtype=np.float32)
            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = self._step(action)
            if (
                not terminated
                and not truncated
                and self.config.max_steps is not None
                and frame_index + 1 >= self.config.max_steps
            ):
                truncated = True
                info = dict(info, runner_truncated=True)
            next_images = self._render()
            transition = SimulationTransition(
                episode_index=episode_index,
                frame_index=frame_index,
                timestamp=frame_index * self.control_dt,
                observation=observation,
                action=action.copy(),
                next_observation=next_observation,
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=info,
                task=self.config.task,
                images=images,
                next_images=next_images,
            )
            for observer in observers:
                observer.on_transition(transition)

            total_reward += float(reward)
            frame_index += 1
            observation = next_observation
            images = next_images
            final_info = info
            self._pace(start=start, completed_steps=frame_index)

        summary = RolloutSummary(
            episode_index=episode_index,
            seed=seed,
            steps=frame_index,
            total_reward=total_reward,
            terminated=terminated,
            truncated=truncated,
            elapsed_wall_seconds=max(0.0, self._clock() - start),
            final_info=final_info,
        )
        for observer in observers:
            observer.on_episode_end(summary)
        return summary

    def _reset(
        self, *, seed: int | None, randomize: bool
    ) -> tuple[Observation, Mapping[str, Any]]:
        result = self.env.reset(seed=seed, randomize=randomize)
        if isinstance(result, tuple):
            observation, info = result
        else:
            observation = result
            info = observation.get("metrics", {})
        return observation, info

    def _step(
        self, action: np.ndarray
    ) -> tuple[Observation, float, bool, bool, Mapping[str, Any]]:
        result = self.env.step(action)
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
        elif len(result) == 4:
            observation, reward, done, info = result
            timeout = str(info.get("failure_reason", "")) == "timeout"
            truncated = bool(done and timeout)
            terminated = bool(done and not timeout)
        else:  # pragma: no cover - defensive guard for third-party adapters.
            raise ValueError("environment step must return a 4- or 5-tuple")
        return (
            observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def _render(self) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        for request in self.config.render:
            frame = np.asarray(
                self.env.render(
                    camera=request.camera,
                    width=request.width,
                    height=request.height,
                ),
                dtype=np.uint8,
            )
            expected = (request.height, request.width, 3)
            if frame.shape != expected:
                raise ValueError(
                    f"camera {request.camera!r} returned {frame.shape}, expected {expected}"
                )
            frames[request.name] = frame.copy()
        return frames

    def _pace(self, *, start: float, completed_steps: int) -> None:
        if not self.config.realtime:
            return
        deadline = start + completed_steps * self.control_dt
        delay = deadline - self._clock()
        if delay > 0.0:
            self._sleep(delay)


__all__ = [
    "CallablePolicy",
    "Observation",
    "Policy",
    "RenderRequest",
    "RolloutObserver",
    "RolloutSummary",
    "RunnerConfig",
    "SimulationEnvironment",
    "SimulationRunner",
    "SimulationTransition",
]
