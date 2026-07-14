"""Collect trajectories through the decoupled environment/export contracts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.exporters import (  # noqa: E402
    ExportObserver,
    HDF5TrajectoryExporter,
    LeRobotTrajectoryExporter,
)
from data.trajectory import (  # noqa: E402
    load_schema_json,
    parse_field_assignment,
    schema_profile,
)
from envs.runtime import (  # noqa: E402
    CallablePolicy,
    RenderRequest,
    RunnerConfig,
    SimulationRunner,
)
from envs.two_robot_carry_env import (  # noqa: E402
    CarryEnvConfig,
    TwoRobotCarryNarrowPassageEnv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect once and stream the same rollout to HDF5 and/or LeRobot."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--format",
        action="append",
        choices=("hdf5", "lerobot"),
        required=True,
        help="Repeat to export the same rollout to multiple formats.",
    )
    parser.add_argument(
        "--profile",
        choices=("vla", "wam", "robocasa", "rmbench"),
        default="vla",
    )
    parser.add_argument("--schema-json", type=Path, default=None)
    parser.add_argument("--field", action="append", default=[])
    parser.add_argument("--drop-field", action="append", default=[])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", default="nominal")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--no-randomize", action="store_true")
    parser.add_argument("--camera", action="append", default=[])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--stream-video", action="store_true")
    parser.add_argument("--video-codec", default="mp4v")
    parser.add_argument("--repo-id", default="local/wam-modular")
    parser.add_argument("--robot-type", default="two_robot_carry")
    parser.add_argument(
        "--task",
        default="carry the object through the passage to the goal",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.schema_json is not None:
        schema = load_schema_json(args.schema_json)
    else:
        schema = schema_profile(args.profile, cameras=args.camera)
    schema = schema.with_overrides(
        add=tuple(parse_field_assignment(field) for field in args.field),
        drop=args.drop_field,
    )

    env = TwoRobotCarryNarrowPassageEnv(CarryEnvConfig(scenario=args.scenario))
    fps = 1.0 / env.control_dt
    exporters: list[Any] = []
    for format_name in dict.fromkeys(args.format):
        if format_name == "hdf5":
            exporters.append(
                HDF5TrajectoryExporter(
                    args.out_dir / "hdf5",
                    schema,
                    stream_videos=args.stream_video,
                    video_codec=args.video_codec,
                )
            )
        elif format_name == "lerobot":
            exporters.append(
                LeRobotTrajectoryExporter(
                    args.out_dir / "lerobot",
                    schema,
                    repo_id=args.repo_id,
                    fps=fps,
                    robot_type=args.robot_type,
                    use_videos=bool(args.camera),
                    streaming_encoding=args.stream_video,
                )
            )

    render = tuple(
        RenderRequest(name=camera, camera=camera, width=args.width, height=args.height)
        for camera in args.camera
    )
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
    observer = ExportObserver(exporters, fps=fps)
    try:
        for episode_index in range(args.episodes):
            summary = runner.run_episode(
                seed=args.seed + episode_index,
                episode_index=episode_index,
                randomize=not args.no_randomize,
                observers=(observer,),
            )
            payload = asdict(summary)
            payload["final_info"] = {
                "success": bool(summary.final_info.get("success", False)),
                "failure": bool(summary.final_info.get("failure", False)),
                "failure_reason": str(summary.final_info.get("failure_reason", "none")),
            }
            summaries.append(payload)
    finally:
        try:
            observer.close()
        finally:
            env.close()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "formats": list(dict.fromkeys(args.format)),
        "schema_profile": schema.profile,
        "fields": [asdict(field) for field in schema.fields],
        "fps": fps,
        "episodes": summaries,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
