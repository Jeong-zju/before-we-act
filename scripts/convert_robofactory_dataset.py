"""Convert RoboFactory/ManiSkill HDF5 into WAM HDF5 and/or LeRobot v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.exporters import HDF5TrajectoryExporter, LeRobotTrajectoryExporter  # noqa: E402
from data.robofactory import RoboFactoryDataset  # noqa: E402
from train.progress import TrainingProgress  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a RoboFactory ManiSkill .h5/.json pair into one-episode-per-file "
            "WAM HDF5, LeRobotDataset v3, or both."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="RoboFactory .h5 file")
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Optional ManiSkill sidecar; defaults to INPUT with a .json suffix.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("robofactory", "m1-scratch"),
        default="robofactory",
        help=(
            "Output schema. 'm1-scratch' writes current/next RGB, frame IDs, "
            "commanded action, and an explicitly declared executed-action source."
        ),
    )
    parser.add_argument(
        "--format",
        action="append",
        choices=("hdf5", "lerobot"),
        required=True,
        help="Repeat to produce both formats from one streaming pass.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Control frequency recorded in the target dataset (default: 20).",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Natural-language task; defaults to the normalized source env_id.",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help="Stable task identifier; defaults to a snake_case source env_id.",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help=(
            "Export one camera by source or normalized name, for example 'global'. "
            "Repeat for multiple cameras; defaults to all source cameras."
        ),
    )
    parser.add_argument(
        "--executed-action-source",
        choices=("command-echo",),
        default=None,
        help=(
            "How action.executed is produced. 'command-echo' writes an exact copy "
            "of action.commanded and records that no independent actuator feedback exists."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Convert at most this many episodes after filtering.",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Keep episodes marked successful by the JSON sidecar or success label.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Omit RGB streams (useful for proprioceptive training or quick checks).",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Omit camera intrinsic/extrinsic matrices.",
    )
    parser.add_argument(
        "--canonical-only",
        action="store_true",
        help="Omit duplicate per-agent qpos/qvel/action fields; keep concatenated state/action.",
    )
    parser.add_argument(
        "--compression",
        choices=("gzip", "lzf", "none"),
        default="gzip",
        help="Compression for array-valued WAM HDF5 fields.",
    )
    parser.add_argument("--repo-id", default="local/robofactory")
    parser.add_argument("--robot-type", default="robofactory_multi_agent")
    parser.add_argument(
        "--lerobot-images",
        choices=("video", "image"),
        default="video",
        help="Store LeRobot RGB streams as MP4 video or individual images.",
    )
    parser.add_argument(
        "--no-streaming-encoding",
        action="store_true",
        help="Disable LeRobot's streaming video encoder.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Rich progress bars for CI or redirected logs.",
    )
    parser.add_argument(
        "--progress-refresh-hz",
        type=float,
        default=4.0,
        help="Maximum Rich display refresh rate (default: 4 Hz).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    formats = tuple(dict.fromkeys(args.format))
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.episodes is not None and args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.progress_refresh_hz <= 0:
        raise ValueError("--progress-refresh-hz must be positive")
    if args.profile == "m1-scratch":
        if formats != ("hdf5",):
            raise ValueError("--profile m1-scratch currently supports --format hdf5 only")
        if args.no_images:
            raise ValueError("--profile m1-scratch requires RGB; omit --no-images")
        if args.executed_action_source != "command-echo":
            raise ValueError(
                "--profile m1-scratch requires the explicit option "
                "--executed-action-source command-echo"
            )
    elif args.executed_action_source is not None:
        raise ValueError(
            "--executed-action-source is only valid with --profile m1-scratch"
        )
    _require_empty_targets(args.out_dir, formats)

    with TrainingProgress(
        enabled=not args.no_progress,
        total_stages=2,
        refresh_per_second=args.progress_refresh_hz,
    ) as progress:
        with RoboFactoryDataset(
            args.input,
            metadata_path=args.metadata_json,
        ) as source:
            episode_count, transition_count = source.conversion_totals(
                max_episodes=args.episodes,
                success_only=args.success_only,
            )
            conversion_progress = progress.add_phase(
                "convert RoboFactory dataset",
                transition_count,
                show_loss_chart=False,
            )
            schema = source.build_schema(
                profile=args.profile,
                cameras=args.camera,
                include_images=not args.no_images,
                include_calibration=not args.no_calibration,
                include_agent_fields=not args.canonical_only,
            )
            exporters: list[Any] = []
            if "hdf5" in formats:
                exporters.append(
                    HDF5TrajectoryExporter(
                        args.out_dir / "hdf5",
                        schema,
                        compression=(
                            None if args.compression == "none" else args.compression
                        ),
                    )
                )
            if "lerobot" in formats:
                exporters.append(
                    LeRobotTrajectoryExporter(
                        args.out_dir / "lerobot",
                        schema,
                        repo_id=args.repo_id,
                        fps=args.fps,
                        robot_type=args.robot_type,
                        use_videos=(
                            args.lerobot_images == "video" and not args.no_images
                        ),
                        streaming_encoding=(
                            not args.no_streaming_encoding
                            and args.lerobot_images == "video"
                            and not args.no_images
                        ),
                    )
                )
            manifest = source.convert(
                exporters,
                fps=args.fps,
                schema=schema,
                task=args.task,
                task_id=args.task_id,
                executed_action_source=args.executed_action_source,
                max_episodes=args.episodes,
                success_only=args.success_only,
                progress=conversion_progress.advance,
            )
            conversion_progress.finish(
                f"{episode_count} episodes, {transition_count} frames"
            )

        manifest_progress = progress.add_phase(
            "write conversion manifest", 1, show_loss_chart=False
        )
        manifest["formats"] = list(formats)
        manifest["outputs"] = {
            name: str((args.out_dir / name).resolve()) for name in formats
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = args.out_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        manifest_progress.advance({"batch": 1})
        manifest_progress.finish(str(manifest_path))
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "formats": list(formats),
                "episodes": len(manifest["episodes"]),
                "state_size": manifest["layout"]["state_size"],
                "action_size": manifest["layout"]["action_size"],
                "camera_names": manifest["field_mapping"]["camera_names"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _require_empty_targets(out_dir: Path, formats: tuple[str, ...]) -> None:
    if out_dir.exists() and not out_dir.is_dir():
        raise FileExistsError(f"output root is not a directory: {out_dir}")
    for name in formats:
        target = out_dir / name
        if target.exists():
            raise FileExistsError(f"output target already exists: {target}")
    manifest = out_dir / "manifest.json"
    if manifest.exists():
        raise FileExistsError(f"refusing to replace existing manifest: {manifest}")


if __name__ == "__main__":
    raise SystemExit(main())
