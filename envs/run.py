"""Standalone CLI for batch or real-time simulation."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from envs.annotations import (
    annotate_cooperative_stop_frame,
    update_cooperative_stop_viewer_labels,
)
from envs.runtime import (
    CallablePolicy,
    RenderRequest,
    RolloutSummary,
    RunnerConfig,
    SimulationRunner,
    SimulationTransition,
)
from envs.two_robot_carry_env import (
    DEFAULT_TASK_INSTRUCTION,
    CooperativeStopEnvConfig,
    TwoRobotCooperativeStopEnv,
)
from envs.video import StreamingVideoObserver


class _PassiveViewerObserver:
    def __init__(self, viewer: Any) -> None:
        self.viewer = viewer

    def on_episode_start(
        self,
        *,
        observation: Mapping[str, Any],
        info: Mapping[str, Any],
        **_: Any,
    ) -> None:
        update_cooperative_stop_viewer_labels(self.viewer, observation, info)
        self.viewer.sync()

    def on_transition(self, transition: SimulationTransition) -> None:
        update_cooperative_stop_viewer_labels(
            self.viewer, transition.next_observation, transition.info
        )
        self.viewer.sync()

    def on_episode_end(self, summary: RolloutSummary) -> None:
        del summary
        self.viewer.sync()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the two-robot cooperative stopping environment without model "
            "or dataset modules."
        )
    )
    parser.add_argument("--scenario", choices=("standard",), default="standard")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-randomize", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--camera", action="append", default=[])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK_INSTRUCTION,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    cameras = list(args.camera)
    if args.video is not None and not cameras:
        cameras = ["fixed"]
    render = tuple(
        RenderRequest(
            name=camera,
            camera=camera,
            width=args.width,
            height=args.height,
            annotator=annotate_cooperative_stop_frame,
        )
        for camera in cameras
    )
    env = TwoRobotCooperativeStopEnv(CooperativeStopEnvConfig(scenario=args.scenario))
    runner = SimulationRunner(
        env,
        CallablePolicy(lambda observation: env.scripted_action()),
        RunnerConfig(
            realtime=args.realtime,
            max_steps=args.max_steps,
            render=render,
            task=args.task,
        ),
    )

    summaries: list[dict[str, Any]] = []
    try:
        with ExitStack() as stack:
            observers: list[Any] = []
            if args.video is not None:
                video = stack.enter_context(
                    StreamingVideoObserver(
                        args.video,
                        stream=cameras[0],
                        fps=1.0 / env.control_dt,
                    )
                )
                observers.append(video)
            if args.viewer:
                import mujoco.viewer

                viewer = stack.enter_context(
                    mujoco.viewer.launch_passive(env.model, env.data)
                )
                observers.append(_PassiveViewerObserver(viewer))
            for episode_index in range(args.episodes):
                summary = runner.run_episode(
                    seed=args.seed + episode_index,
                    episode_index=episode_index,
                    randomize=not args.no_randomize,
                    observers=observers,
                )
                summaries.append(_summary_json(summary))
    finally:
        env.close()

    print(json.dumps({"episodes": summaries}, indent=2, sort_keys=True))
    return 0


def _summary_json(summary: RolloutSummary) -> dict[str, Any]:
    result = asdict(summary)
    result["final_info"] = {
        key: _json_value(value) for key, value in summary.final_info.items()
    }
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
