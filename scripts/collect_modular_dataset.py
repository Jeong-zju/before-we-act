"""Collect trajectories through the decoupled environment/export contracts."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeRemainingColumn,
    )
except ImportError:  # Rich is optional; collection must remain headless-safe.
    Console = None
    Progress = None

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
    DEFAULT_TASK_INSTRUCTION,
    CooperativeStopEnvConfig,
    TwoRobotCooperativeStopEnv,
)
from policies.collection import CooperativeStopCollectionPolicy  # noqa: E402


class CollectionProgressObserver:
    """Render dataset collection progress without entering the data path."""

    def __init__(
        self,
        total_episodes: int,
        *,
        enabled: bool = True,
        refresh_per_second: float = 4.0,
        step_interval: int = 5,
    ) -> None:
        self.total_episodes = int(total_episodes)
        self.completed_episodes = 0
        self.successes = 0
        self.failures = 0
        self.current_step = 0
        self.step_interval = int(step_interval)
        self._last_phase = ""
        self._progress: Any | None = None
        self._task_id: Any | None = None

        if refresh_per_second <= 0:
            raise ValueError("refresh_per_second must be positive")
        if self.step_interval <= 0:
            raise ValueError("step_interval must be positive")

        if not enabled:
            return
        if Progress is None or Console is None:
            print(
                "Progress display unavailable: install 'rich' to enable it.",
                file=sys.stderr,
            )
            return

        self._progress = Progress(
            SpinnerColumn(style="bold cyan"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None, complete_style="cyan"),
            MofNCompleteColumn(),
            TextColumn("[white]{task.fields[step_text]}"),
            TextColumn("[green]{task.fields[success_text]}"),
            TextColumn("[red]{task.fields[failure_text]}"),
            TextColumn("[magenta]{task.fields[phase_text]}"),
            TimeRemainingColumn(),
            console=Console(stderr=True),
            expand=True,
            refresh_per_second=float(refresh_per_second),
        )
        self._task_id = self._progress.add_task(
            "collect",
            total=self.total_episodes,
            step_text="s=0000",
            success_text="ok=00000",
            failure_text="bad=00000",
            phase_text=_fit_progress_text("initializing", 16),
        )

    def start(self) -> None:
        if self._progress is not None:
            self._progress.start()

    def stop(self) -> None:
        if self._progress is not None:
            self._progress.stop()

    def on_episode_start(
        self,
        *,
        episode_index: int,
        seed: int | None,
        observation: Any,
        info: Any,
        task: str,
    ) -> None:
        del episode_index, seed, observation, task
        self.current_step = 0
        braking_agent = int(info.get("braking_agent", -1))
        brake_time = float(info.get("brake_start_time", -1.0))
        phase = f"brake r{braking_agent} @ {brake_time:.2f}s"
        self._last_phase = phase
        self._update(
            step=0,
            phase=phase,
        )

    def on_transition(self, transition: Any) -> None:
        self.current_step = int(transition.frame_index) + 1
        phase = str(transition.info.get("task_phase", "running")).replace("_", " ")
        if self.current_step % self.step_interval != 0 and phase == self._last_phase:
            return
        self._last_phase = phase
        self._update(step=self.current_step, phase=phase)

    def on_episode_end(self, summary: Any) -> None:
        success = bool(summary.final_info.get("success", False))
        self.successes += int(success)
        self.failures += int(not success)
        self.completed_episodes += 1
        failure_reason = str(summary.final_info.get("failure_reason", "none"))
        phase = "success" if success else f"failed: {failure_reason}"
        self._update(
            advance=1,
            step=int(summary.steps),
            successes=self.successes,
            failures=self.failures,
            phase=phase,
        )

    def finalizing(self) -> None:
        self._update(phase="closing exporters")

    def _update(self, *, advance: int = 0, **fields: Any) -> None:
        if self._progress is not None and self._task_id is not None:
            stable_fields: dict[str, Any] = {}
            if "step" in fields:
                stable_fields["step_text"] = f"s={int(fields['step']):04d}"
            if "successes" in fields:
                stable_fields["success_text"] = (
                    f"ok={int(fields['successes']):05d}"
                )
            if "failures" in fields:
                stable_fields["failure_text"] = (
                    f"bad={int(fields['failures']):05d}"
                )
            if "phase" in fields:
                stable_fields["phase_text"] = _fit_progress_text(
                    str(fields["phase"]), 16
                )
            self._progress.update(
                self._task_id,
                advance=advance,
                **stable_fields,
            )


def _fit_progress_text(value: str, width: int) -> str:
    """Keep dynamic fields at a fixed width so the Rich table does not reflow."""

    text = " ".join(str(value).split())
    if len(text) > width:
        text = f"{text[: width - 1]}…"
    return text.ljust(width)


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
        choices=("vla", "wam", "wam_proprio", "robocasa", "rmbench"),
        default="vla",
    )
    parser.add_argument("--schema-json", type=Path, default=None)
    parser.add_argument("--field", action="append", default=[])
    parser.add_argument("--drop-field", action="append", default=[])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--behavior-profile",
        choices=CooperativeStopCollectionPolicy.PROFILES,
        default="scripted_oracle_v1",
        help="Offline data behavior distribution; mixed_proprio is the full WAM mixture.",
    )
    parser.add_argument(
        "--mixture-seed",
        type=int,
        default=20260714,
        help="Seed for behavior scheduling and perturbation parameters.",
    )
    parser.add_argument("--scenario", choices=("standard",), default="standard")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--no-randomize", action="store_true")
    parser.add_argument("--camera", action="append", default=[])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--stream-video", action="store_true")
    parser.add_argument("--video-codec", default="mp4v")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the Rich collection progress display.",
    )
    parser.add_argument(
        "--progress-refresh-hz",
        type=float,
        default=4.0,
        help="Maximum Rich display refresh rate (default: 4 Hz).",
    )
    parser.add_argument(
        "--progress-step-interval",
        type=int,
        default=5,
        help="Update the displayed step every N transitions (default: 5).",
    )
    parser.add_argument("--repo-id", default="local/wam-modular")
    parser.add_argument("--robot-type", default="two_robot_cooperative_stop")
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK_INSTRUCTION,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.progress_refresh_hz <= 0:
        raise ValueError("progress_refresh_hz must be positive")
    if args.progress_step_interval <= 0:
        raise ValueError("progress_step_interval must be positive")
    if args.schema_json is not None:
        schema = load_schema_json(args.schema_json)
    else:
        schema = schema_profile(args.profile, cameras=args.camera)
    schema = schema.with_overrides(
        add=tuple(parse_field_assignment(field) for field in args.field),
        drop=args.drop_field,
    )

    proprio_only = schema.profile == "wam_proprio"
    if proprio_only and any(name != "hdf5" for name in args.format):
        raise ValueError("wam_proprio currently uses the HDF5 backend only")
    if proprio_only and (args.camera or args.stream_video):
        raise ValueError("wam_proprio collection does not render or encode images")
    if args.behavior_profile == "mixed_proprio" and not proprio_only:
        raise ValueError("mixed_proprio is restricted to the wam_proprio profile")

    env_config = CooperativeStopEnvConfig(
        scenario=args.scenario,
        include_camera_images=not proprio_only,
    )
    env = TwoRobotCooperativeStopEnv(env_config)
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
                    use_videos=any(field.is_image for field in schema.fields),
                    streaming_encoding=args.stream_video,
                )
            )

    render = tuple(
        RenderRequest(name=camera, camera=camera, width=args.width, height=args.height)
        for camera in args.camera
    )
    collection_policy = CooperativeStopCollectionPolicy(
        env,
        profile=args.behavior_profile,
        mixture_seed=args.mixture_seed,
    )
    runner = SimulationRunner(
        env,
        CallablePolicy(collection_policy.act),
        RunnerConfig(
            realtime=args.realtime,
            max_steps=args.max_steps,
            render=render,
            task=args.task,
        ),
    )
    summaries: list[dict[str, Any]] = []
    observer = ExportObserver(exporters, fps=fps)
    progress = CollectionProgressObserver(
        args.episodes,
        enabled=not args.no_progress,
        refresh_per_second=args.progress_refresh_hz,
        step_interval=args.progress_step_interval,
    )
    progress.start()
    try:
        for episode_index in range(args.episodes):
            episode_seed = args.seed + episode_index
            behavior = collection_policy.configure_episode(
                episode_index=episode_index,
                episode_seed=episode_seed,
            )
            episode_metadata = {
                "schema_version": schema.version,
                "behavior_id": behavior.behavior_id,
                "perturbation_config": json.dumps(
                    behavior.perturbation_config, sort_keys=True
                ),
                "environment_config": json.dumps(
                    asdict(env_config), sort_keys=True
                ),
                "randomization_config": json.dumps(
                    {
                        "enabled": not args.no_randomize,
                        "seed": episode_seed,
                    },
                    sort_keys=True,
                ),
            }
            summary = runner.run_episode(
                seed=episode_seed,
                episode_index=episode_index,
                randomize=not args.no_randomize,
                observers=(observer, progress),
                metadata=episode_metadata,
            )
            payload = asdict(summary)
            payload["behavior_id"] = behavior.behavior_id
            payload["perturbation_config"] = behavior.perturbation_config
            payload["final_info"] = {
                "success": bool(summary.final_info.get("success", False)),
                "failure": bool(summary.final_info.get("failure", False)),
                "failure_reason": str(summary.final_info.get("failure_reason", "none")),
                "braking_agent": int(summary.final_info.get("braking_agent", -1)),
                "brake_start_step": int(summary.final_info.get("brake_start_step", -1)),
                "response_delay_steps": int(
                    summary.final_info.get("response_delay_steps", -1)
                ),
            }
            summaries.append(payload)
        progress.finalizing()
    finally:
        try:
            observer.close()
        finally:
            try:
                env.close()
            finally:
                progress.stop()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "formats": list(dict.fromkeys(args.format)),
        "schema_profile": schema.profile,
        "schema_version": schema.version,
        "behavior_profile": args.behavior_profile,
        "mixture_seed": args.mixture_seed,
        "behavior_counts": dict(
            sorted(Counter(item["behavior_id"] for item in summaries).items())
        ),
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
