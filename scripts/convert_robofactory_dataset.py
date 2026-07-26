"""Convert RoboFactory/ManiSkill HDF5 into WAM HDF5 and/or LeRobot v3."""

from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.exporters import HDF5TrajectoryExporter, LeRobotTrajectoryExporter  # noqa: E402
from data.robofactory import RoboFactoryDataset  # noqa: E402
from scripts.select_robofactory_collection_workers import (  # noqa: E402
    available_memory_bytes,
    effective_cpu_count,
)
from train.progress import TrainingProgress  # noqa: E402

_PARALLEL_STATE: dict[str, Any] | None = None


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
    parser.add_argument(
        "--num-workers",
        default="1",
        help=(
            "Episode conversion processes. Use 'auto' for cgroup-aware CPU/RAM "
            "selection. Parallel conversion currently supports HDF5 only."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Maximum workers selected by --num-workers auto (default: 16).",
    )
    parser.add_argument(
        "--worker-memory-mib",
        type=int,
        default=1536,
        help="Estimated resident memory per conversion worker (default: 1536 MiB).",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.75,
        help="Fraction of currently available memory usable by workers.",
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
    if args.max_workers <= 0 or args.worker_memory_mib <= 0:
        raise ValueError("--max-workers and --worker-memory-mib must be positive")
    if not 0.1 <= args.memory_fraction <= 0.95:
        raise ValueError("--memory-fraction must lie in [0.1, 0.95]")
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
            worker_count = _resolve_worker_count(
                args.num_workers,
                episodes=episode_count,
                max_workers=args.max_workers,
                worker_memory_mib=args.worker_memory_mib,
                memory_fraction=args.memory_fraction,
            )
            if worker_count > 1 and formats != ("hdf5",):
                raise ValueError("parallel conversion currently supports HDF5 only")
            if worker_count > 1:
                _limit_worker_threads()
            conversion_progress = progress.add_phase(
                "convert RoboFactory dataset",
                episode_count if worker_count > 1 else transition_count,
                show_loss_chart=False,
            )
            schema = source.build_schema(
                profile=args.profile,
                cameras=args.camera,
                include_images=not args.no_images,
                include_calibration=not args.no_calibration,
                include_agent_fields=not args.canonical_only,
            )
            if worker_count > 1:
                manifest = _convert_hdf5_parallel(
                    args=args,
                    schema=schema,
                    episode_count=episode_count,
                    workers=worker_count,
                    progress=conversion_progress.advance,
                )
            else:
                manifest = _convert_sequential(
                    args=args,
                    formats=formats,
                    source=source,
                    schema=schema,
                    progress=conversion_progress.advance,
                )
            conversion_progress.finish(
                f"{episode_count} episodes, {transition_count} frames"
            )

        manifest_progress = progress.add_phase(
            "write conversion manifest", 1, show_loss_chart=False
        )
        manifest["conversion_execution"] = {
            "num_workers": worker_count,
            "parallel_unit": "episode" if worker_count > 1 else None,
            "multiprocessing_context": "spawn" if worker_count > 1 else None,
        }
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
                "num_workers": worker_count,
                "state_size": manifest["layout"]["state_size"],
                "action_size": manifest["layout"]["action_size"],
                "camera_names": manifest["field_mapping"]["camera_names"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _convert_sequential(
    *,
    args: argparse.Namespace,
    formats: tuple[str, ...],
    source: RoboFactoryDataset,
    schema: Any,
    progress: Any,
) -> dict[str, Any]:
    exporters: list[Any] = []
    if "hdf5" in formats:
        exporters.append(
            HDF5TrajectoryExporter(
                args.out_dir / "hdf5",
                schema,
                compression=None if args.compression == "none" else args.compression,
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
                use_videos=args.lerobot_images == "video" and not args.no_images,
                streaming_encoding=(
                    not args.no_streaming_encoding
                    and args.lerobot_images == "video"
                    and not args.no_images
                ),
            )
        )
    return source.convert(
        exporters,
        fps=args.fps,
        schema=schema,
        task=args.task,
        task_id=args.task_id,
        executed_action_source=args.executed_action_source,
        max_episodes=args.episodes,
        success_only=args.success_only,
        progress=progress,
    )


def _resolve_worker_count(
    requested: str,
    *,
    episodes: int,
    max_workers: int,
    worker_memory_mib: int,
    memory_fraction: float,
) -> int:
    cpus = effective_cpu_count()
    memory = available_memory_bytes()
    cpu_limit = max(1, cpus - min(2, max(0, cpus - 1)))
    memory_limit = max(
        1,
        int(memory * memory_fraction) // (worker_memory_mib * 1024**2),
    )
    automatic_limit = min(episodes, max_workers, cpu_limit, memory_limit)
    normalized = requested.strip().lower()
    if normalized == "auto":
        workers = automatic_limit
        mode = "auto"
    else:
        try:
            workers = int(normalized)
        except ValueError as exc:
            raise ValueError("--num-workers must be 'auto' or an integer") from exc
        if not 1 <= workers <= episodes:
            raise ValueError("--num-workers must lie in [1, selected episodes]")
        mode = "manual"
    warning = (
        " WARNING: manual value exceeds the automatic safety limit."
        if mode == "manual" and workers > automatic_limit
        else ""
    )
    print(
        "[conversion-workers] "
        f"mode={mode} cpus={cpus} available_memory_gib={memory / 1024**3:.1f} "
        f"cpu_limit={cpu_limit} memory_limit={memory_limit} cap={max_workers} "
        f"workers={workers}.{warning}",
        file=sys.stderr,
        flush=True,
    )
    return workers


def _limit_worker_threads() -> None:
    threads = os.environ.get("M2_CONVERSION_THREADS_PER_WORKER", "1")
    if not threads.isdigit() or int(threads) <= 0:
        raise ValueError("M2_CONVERSION_THREADS_PER_WORKER must be positive")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = threads


def _parallel_worker_init(config: Mapping[str, Any]) -> None:
    global _PARALLEL_STATE
    source = RoboFactoryDataset(
        config["input"], metadata_path=config["metadata_json"]
    )
    schema = source.build_schema(
        profile=config["profile"],
        cameras=config["cameras"],
        include_images=config["include_images"],
        include_calibration=config["include_calibration"],
        include_agent_fields=config["include_agent_fields"],
    )
    _PARALLEL_STATE = {"config": dict(config), "source": source, "schema": schema}
    atexit.register(source.close)


def _parallel_convert_episode(output_index: int) -> dict[str, Any]:
    if _PARALLEL_STATE is None:
        raise RuntimeError("parallel converter worker is not initialized")
    config = _PARALLEL_STATE["config"]
    source = _PARALLEL_STATE["source"]
    schema = _PARALLEL_STATE["schema"]
    exporter = HDF5TrajectoryExporter(
        Path(config["out_dir"]) / "hdf5",
        schema,
        compression=config["compression"],
    )
    return source.convert(
        (exporter,),
        fps=config["fps"],
        schema=schema,
        task=config["task"],
        task_id=config["task_id"],
        executed_action_source=config["executed_action_source"],
        max_episodes=config["episodes"],
        success_only=config["success_only"],
        episode_indices=(output_index,),
        output_indices=(output_index,),
        compute_source_hashes=False,
    )


def _convert_hdf5_parallel(
    *,
    args: argparse.Namespace,
    schema: Any,
    episode_count: int,
    workers: int,
    progress: Any,
) -> dict[str, Any]:
    metadata_path = args.metadata_json or args.input.with_suffix(".json")
    config = {
        "input": str(args.input),
        "metadata_json": str(metadata_path),
        "out_dir": str(args.out_dir),
        "profile": args.profile,
        "cameras": tuple(args.camera) if args.camera is not None else None,
        "include_images": not args.no_images,
        "include_calibration": not args.no_calibration,
        "include_agent_fields": not args.canonical_only,
        "compression": None if args.compression == "none" else args.compression,
        "fps": args.fps,
        "task": args.task,
        "task_id": args.task_id,
        "executed_action_source": args.executed_action_source,
        "episodes": args.episodes,
        "success_only": args.success_only,
    }
    started = time.monotonic()
    results: dict[int, dict[str, Any]] = {}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_parallel_worker_init,
        initargs=(config,),
    ) as executor:
        futures = {
            executor.submit(_parallel_convert_episode, index): index
            for index in range(episode_count)
        }
        for future in as_completed(futures):
            index = futures[future]
            manifest = future.result()
            summaries = manifest.get("episodes", [])
            if len(summaries) != 1 or summaries[0]["episode_index"] != index:
                raise RuntimeError(f"worker returned invalid episode summary for {index}")
            results[index] = manifest
            summary = summaries[0]
            progress(
                {
                    "source_episode": summary["source_episode_id"],
                    "episode": len(results),
                    "episodes": episode_count,
                    "frame": summary["steps"],
                    "frames": summary["steps"],
                }
            )
    return _merge_parallel_manifests(
        results,
        episode_count=episode_count,
        source_path=args.input,
        metadata_path=metadata_path,
        output_dir=args.out_dir / "hdf5",
        elapsed_wall_seconds=time.monotonic() - started,
        expected_schema_version=schema.version,
    )


def _merge_parallel_manifests(
    results: Mapping[int, dict[str, Any]],
    *,
    episode_count: int,
    source_path: Path,
    metadata_path: Path,
    output_dir: Path,
    elapsed_wall_seconds: float,
    expected_schema_version: str,
) -> dict[str, Any]:
    if set(results) != set(range(episode_count)):
        raise RuntimeError("parallel conversion did not return every episode")
    manifest = results[0]
    contract_keys = (
        "format_version",
        "schema_profile",
        "schema_version",
        "source",
        "task",
        "task_id",
        "fps",
        "transition_semantics",
        "data_semantics",
        "field_mapping",
        "layout",
        "fields",
        "filters",
    )
    if manifest["schema_version"] != expected_schema_version:
        raise RuntimeError("parallel worker schema version mismatch")
    for index in range(1, episode_count):
        candidate = results[index]
        if any(candidate[key] != manifest[key] for key in contract_keys):
            raise RuntimeError(f"parallel worker {index} produced a different contract")
    manifest["episodes"] = [
        results[index]["episodes"][0] for index in range(episode_count)
    ]
    manifest["elapsed_wall_seconds"] = elapsed_wall_seconds
    manifest["source"]["hdf5_sha256"] = _sha256_file(source_path)
    manifest["source"]["metadata_json_sha256"] = (
        _sha256_file(metadata_path) if metadata_path.is_file() else None
    )
    expected_files = {
        f"episode_{index:06d}.hdf5" for index in range(episode_count)
    }
    actual_files = {path.name for path in output_dir.glob("episode_*.hdf5")}
    if actual_files != expected_files:
        raise RuntimeError("parallel conversion output file set is incomplete")
    return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
