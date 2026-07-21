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
FrameAnnotator = Callable[[np.ndarray, Mapping[str, Any]], np.ndarray]


@dataclass(frozen=True)
class RenderRequest:
    """One named RGB stream requested from an environment."""

    name: str
    camera: str
    width: int = 640
    height: int = 360
    annotator: FrameAnnotator | None = None
    fps: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("render stream name cannot be empty")
        if not self.camera:
            raise ValueError("camera name cannot be empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("render dimensions must be positive")
        if self.fps is not None and (
            not np.isfinite(float(self.fps)) or float(self.fps) <= 0.0
        ):
            raise ValueError("render fps must be finite and positive when provided")


@dataclass(frozen=True)
class RunnerConfig:
    """Execution controls shared by batch and real-time simulation."""

    realtime: bool = False
    max_steps: int | None = None
    render: tuple[RenderRequest, ...] = ()
    expose_privileged_state_to_policy: bool = False
    policy_observation_keys: tuple[str, ...] | None = None
    expose_rendered_images_to_policy: bool = False
    policy_image_streams: tuple[str, ...] = ()
    expose_task_to_policy: bool = False
    task_id: str = "cooperative_stop"
    policy_action_history: int = 0
    task: str = (
        "carry the object together; when one robot slows to a stop, "
        "the other robot should gradually slow and stop"
    )

    def __post_init__(self) -> None:
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")
        if self.policy_action_history < 0:
            raise ValueError("policy_action_history cannot be negative")
        if not self.task_id:
            raise ValueError("task_id cannot be empty")
        if self.policy_observation_keys is not None:
            if len(set(self.policy_observation_keys)) != len(
                self.policy_observation_keys
            ):
                raise ValueError("policy_observation_keys must be unique")
            if "privileged_state" in self.policy_observation_keys:
                raise ValueError(
                    "privileged_state cannot appear in the policy observation allowlist"
                )
        render_names = tuple(request.name for request in self.render)
        if len(set(render_names)) != len(render_names):
            raise ValueError("render stream names must be unique")
        if self.expose_rendered_images_to_policy:
            if not self.policy_image_streams:
                raise ValueError(
                    "policy_image_streams must explicitly select raw policy RGB"
                )
            unknown = sorted(set(self.policy_image_streams) - set(render_names))
            if unknown:
                raise ValueError(f"unknown policy image streams: {unknown}")
            annotated = sorted(
                request.name
                for request in self.render
                if request.name in self.policy_image_streams
                and request.annotator is not None
            )
            if annotated:
                raise ValueError(
                    f"annotated streams cannot be exposed to policy: {annotated}"
                )


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
    image_timestamps: Mapping[str, float] = field(default_factory=dict)
    next_image_timestamps: Mapping[str, float] = field(default_factory=dict)
    image_state_timestamps: Mapping[str, float] = field(default_factory=dict)
    next_image_state_timestamps: Mapping[str, float] = field(default_factory=dict)
    image_frame_indices: Mapping[str, int] = field(default_factory=dict)
    next_image_frame_indices: Mapping[str, int] = field(default_factory=dict)
    camera_intrinsics: Mapping[str, np.ndarray] = field(default_factory=dict)
    next_camera_intrinsics: Mapping[str, np.ndarray] = field(default_factory=dict)
    camera_extrinsics: Mapping[str, np.ndarray] = field(default_factory=dict)
    next_camera_extrinsics: Mapping[str, np.ndarray] = field(default_factory=dict)
    camera_resolutions: Mapping[str, np.ndarray] = field(default_factory=dict)
    next_camera_resolutions: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

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


@dataclass(frozen=True)
class _RenderBatch:
    frames: Mapping[str, np.ndarray]
    timestamps: Mapping[str, float]
    state_timestamps: Mapping[str, float]
    frame_indices: Mapping[str, int]
    intrinsics: Mapping[str, np.ndarray]
    extrinsics: Mapping[str, np.ndarray]
    resolutions: Mapping[str, np.ndarray]


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

    def camera_calibration(
        self, *, camera: str, width: int, height: int
    ) -> Mapping[str, Any]: ...

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
        self._render_frames: dict[str, np.ndarray] = {}
        self._render_timestamps: dict[str, float] = {}
        self._render_state_timestamps: dict[str, float] = {}
        self._render_frame_indices: dict[str, int] = {}
        self._render_intrinsics: dict[str, np.ndarray] = {}
        self._render_extrinsics: dict[str, np.ndarray] = {}
        self._render_resolutions: dict[str, np.ndarray] = {}

    def run_episode(
        self,
        *,
        seed: int | None = None,
        episode_index: int = 0,
        randomize: bool = True,
        observers: Sequence[RolloutObserver] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> RolloutSummary:
        episode_metadata = dict(metadata or {})
        observation, reset_info = self._reset(seed=seed, randomize=randomize)
        reset_policy = getattr(self.policy, "reset", None)
        if callable(reset_policy):
            reset_policy()
        for observer in observers:
            observer.on_episode_start(
                episode_index=episode_index,
                seed=seed,
                observation=observation,
                info=reset_info,
                task=self.config.task,
            )

        self._reset_render_state()
        rendered = self._render(reset_info, simulation_time=0.0)
        action_history: list[np.ndarray] = []
        start = self._clock()
        frame_index = 0
        total_reward = 0.0
        terminated = False
        truncated = False
        final_info: Mapping[str, Any] = reset_info

        while not (terminated or truncated):
            policy_observation = self._policy_observation(
                observation,
                rendered=rendered,
                action_history=action_history,
            )
            action = np.asarray(self.policy.act(policy_observation), dtype=np.float32)
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
            next_rendered = self._render(
                info,
                simulation_time=(frame_index + 1) * self.control_dt,
            )
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
                images=rendered.frames,
                next_images=next_rendered.frames,
                image_timestamps=rendered.timestamps,
                next_image_timestamps=next_rendered.timestamps,
                image_state_timestamps=rendered.state_timestamps,
                next_image_state_timestamps=next_rendered.state_timestamps,
                image_frame_indices=rendered.frame_indices,
                next_image_frame_indices=next_rendered.frame_indices,
                camera_intrinsics=rendered.intrinsics,
                next_camera_intrinsics=next_rendered.intrinsics,
                camera_extrinsics=rendered.extrinsics,
                next_camera_extrinsics=next_rendered.extrinsics,
                camera_resolutions=rendered.resolutions,
                next_camera_resolutions=next_rendered.resolutions,
                metadata=episode_metadata,
            )
            for observer in observers:
                observer.on_transition(transition)

            total_reward += float(reward)
            frame_index += 1
            observation = next_observation
            rendered = next_rendered
            executed_action = np.asarray(
                info.get("executed_action", action), dtype=np.float32
            ).reshape(-1)
            if executed_action.shape != action.reshape(-1).shape:
                raise ValueError(
                    "info['executed_action'] must match the commanded action shape"
                )
            action_history.append(executed_action.copy())
            if self.config.policy_action_history:
                del action_history[: -self.config.policy_action_history]
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

    def _policy_observation(
        self,
        observation: Observation,
        *,
        rendered: _RenderBatch,
        action_history: Sequence[np.ndarray],
    ) -> Observation:
        """Build an explicit policy view and keep simulator truth fail-closed."""

        if self.config.policy_observation_keys is None:
            result = dict(observation)
        else:
            missing = [
                key
                for key in self.config.policy_observation_keys
                if key not in observation
            ]
            if missing:
                raise KeyError(f"policy observation allowlist keys are missing: {missing}")
            result = {
                key: observation[key] for key in self.config.policy_observation_keys
            }
        if not self.config.expose_privileged_state_to_policy:
            result.pop("privileged_state", None)
        if self.config.expose_rendered_images_to_policy:
            streams = self.config.policy_image_streams
            result["images"] = {
                name: rendered.frames[name].copy() for name in streams
            }
            result["image_timestamps"] = {
                name: float(rendered.timestamps[name]) for name in streams
            }
            result["image_frame_indices"] = {
                name: int(rendered.frame_indices[name]) for name in streams
            }
        if self.config.expose_task_to_policy:
            result["task"] = {"id": self.config.task_id, "text": self.config.task}
        if self.config.policy_action_history:
            action_dim = int(getattr(self.env, "action_dim", 0))
            if action_dim <= 0:
                raise ValueError(
                    "policy action history requires environment.action_dim"
                )
            if action_history:
                history = np.stack(action_history[-self.config.policy_action_history :])
            else:
                history = np.zeros((0, action_dim), dtype=np.float32)
            result["past_executed_actions"] = np.asarray(
                history, dtype=np.float32
            ).copy()
        return result

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

    def _reset_render_state(self) -> None:
        self._render_frames.clear()
        self._render_timestamps.clear()
        self._render_state_timestamps.clear()
        self._render_frame_indices.clear()
        self._render_intrinsics.clear()
        self._render_extrinsics.clear()
        self._render_resolutions.clear()

    def _render(
        self, info: Mapping[str, Any], *, simulation_time: float
    ) -> _RenderBatch:
        for request in self.config.render:
            last_timestamp = self._render_timestamps.get(request.name)
            period = None if request.fps is None else 1.0 / float(request.fps)
            capture = last_timestamp is None or period is None
            if last_timestamp is not None and period is not None:
                capture = simulation_time - last_timestamp >= period - 1e-12
            if not capture:
                continue
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
            if request.annotator is not None:
                frame = np.asarray(request.annotator(frame, info), dtype=np.uint8)
                if frame.shape != expected:
                    raise ValueError(
                        f"annotator for {request.name!r} returned {frame.shape}, "
                        f"expected {expected}"
                    )
            intrinsics, extrinsics, resolution = self._camera_calibration(request)
            self._render_frames[request.name] = frame.copy()
            self._render_timestamps[request.name] = float(simulation_time)
            self._render_state_timestamps[request.name] = float(simulation_time)
            self._render_frame_indices[request.name] = (
                self._render_frame_indices.get(request.name, -1) + 1
            )
            self._render_intrinsics[request.name] = intrinsics
            self._render_extrinsics[request.name] = extrinsics
            self._render_resolutions[request.name] = resolution
        return _RenderBatch(
            frames={key: value.copy() for key, value in self._render_frames.items()},
            timestamps=dict(self._render_timestamps),
            state_timestamps=dict(self._render_state_timestamps),
            frame_indices=dict(self._render_frame_indices),
            intrinsics={
                key: value.copy() for key, value in self._render_intrinsics.items()
            },
            extrinsics={
                key: value.copy() for key, value in self._render_extrinsics.items()
            },
            resolutions={
                key: value.copy() for key, value in self._render_resolutions.items()
            },
        )

    def _camera_calibration(
        self, request: RenderRequest
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        method = getattr(self.env, "camera_calibration", None)
        if not callable(method):
            raise TypeError(
                "rendered environments must implement camera_calibration"
            )
        raw = method(
            camera=request.camera,
            width=request.width,
            height=request.height,
        )
        if not isinstance(raw, Mapping):
            raise TypeError("camera_calibration must return a mapping")
        missing = [
            name
            for name in ("intrinsics", "extrinsics", "resolution")
            if name not in raw
        ]
        if missing:
            raise KeyError(f"camera_calibration is missing fields: {missing}")
        intrinsics = np.asarray(raw["intrinsics"], dtype=np.float32)
        extrinsics = np.asarray(raw["extrinsics"], dtype=np.float32)
        raw_resolution = np.asarray(raw["resolution"])
        if raw_resolution.dtype.kind not in {"i", "u"}:
            raise TypeError("camera calibration resolution must contain integers")
        resolution = raw_resolution.astype(np.int64, copy=False)
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            raise ValueError(
                f"camera {request.camera!r} intrinsics must be finite [3,3]"
            )
        if extrinsics.shape != (4, 4) or not np.isfinite(extrinsics).all():
            raise ValueError(
                f"camera {request.camera!r} extrinsics must be finite [4,4]"
            )
        expected_resolution = np.asarray(
            [request.height, request.width], dtype=np.int64
        )
        if resolution.shape != (2,) or not np.array_equal(
            resolution, expected_resolution
        ):
            raise ValueError(
                f"camera {request.camera!r} calibration resolution must be "
                f"[height,width]={expected_resolution.tolist()}"
            )
        return intrinsics.copy(), extrinsics.copy(), resolution.copy()

    def _pace(self, *, start: float, completed_steps: int) -> None:
        if not self.config.realtime:
            return
        deadline = start + completed_steps * self.control_dt
        delay = deadline - self._clock()
        if delay > 0.0:
            self._sleep(delay)


__all__ = [
    "CallablePolicy",
    "FrameAnnotator",
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
