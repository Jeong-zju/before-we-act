"""Run diagnostic Phase M1 closed-loop rollouts and record verified MP4s."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

# Headless MuJoCo rendering is the default for reproducible rollout capture.
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.runtime import RenderRequest, RunnerConfig, SimulationRunner  # noqa: E402
from envs.video import StreamingVideoObserver  # noqa: E402
from envs.visual_required_env import (  # noqa: E402
    VISUAL_REQUIRED_TASKS,
    VISUAL_REQUIRED_TASK_TEXTS,
    VisualRequiredEnv,
    VisualRequiredEnvConfig,
)
from eval.m1_acceptance import PRIMARY_VARIANT  # noqa: E402
from eval.m1_vision_contract import (  # noqa: E402
    validate_loaded_checkpoint_vision,
    validate_training_summary_vision,
)
from policies.multimodal_joint_wam import MultimodalJointWAMPolicy  # noqa: E402
from scripts.evaluate_multimodal_wam import (  # noqa: E402
    CLEAN,
    _InterventionPolicy,
    _policy_config,
    _warm_policy_runtime,
)
from train.m1_checkpointing import (  # noqa: E402
    checkpoint_tree_sha256,
    load_m1_checkpoint,
)
from train.progress import TrainingProgress  # noqa: E402


FORMAT_VERSION = "wam.multimodal.m1.rollout/1"
VIDEO_FORMAT_VERSION = "wam.multimodal.m1.rollout_video/1"
DEFAULT_CONFIG = ROOT / "configs/wam_multimodal/m1_latent_wam_dinov3.yaml"
POLICY_STREAM = "fixed"
VIDEO_STREAM = "rollout"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--training-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=16)
    parser.add_argument(
        "--train-seeds",
        nargs="+",
        type=int,
        help="default: all formal training seeds from the M1 config",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=VISUAL_REQUIRED_TASKS,
        help="default: all visual-required tasks",
    )
    parser.add_argument(
        "--physical-seeds",
        type=int,
        default=10,
        help="number of physical seeds per task; each is run for every cue",
    )
    parser.add_argument(
        "--physical-seed-start",
        type=int,
        help=(
            "default: first seed after the canonical M1 formal range, so a "
            "diagnostic run does not silently reuse formal seeds"
        ),
    )
    parser.add_argument(
        "--cue-variants",
        nargs="+",
        type=int,
        choices=(0, 1),
        default=[0, 1],
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--video-episodes-per-task",
        type=int,
        default=2,
        help=(
            "number of the first physical-seed/cue rollouts recorded for every "
            "train-seed/task pair; 0 disables MP4 output"
        ),
    )
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=360)
    parser.add_argument("--video-codec", default="mp4v")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = _load_yaml(config_path)
    settings = _resolve_settings(args, config)
    _validate_settings(settings)
    device = _device(str(args.device))
    _configure_torch_threads(int(args.torch_threads))

    training_summary_path = settings["training_summary"]
    training_summary = _read_json(training_summary_path)
    _validate_training_summary(
        training_summary,
        config,
        config_path=config_path,
        train_seeds=settings["train_seeds"],
    )
    checkpoint_evidence = _checkpoint_evidence(
        settings["checkpoint_root"],
        settings["train_seeds"],
        training_summary,
    )

    output = settings["output"]
    _prepare_output_directory(output)
    records_path = output / "rollout_episodes.jsonl"
    report_path = output / "rollout_summary.json"
    video_root = output / "videos"

    records: list[dict[str, Any]] = []
    video_evidence: list[dict[str, Any]] = []
    episode_pairs = _episode_pairs(
        physical_seed_start=settings["physical_seed_start"],
        physical_seed_count=settings["physical_seed_count"],
        cue_variants=settings["cue_variants"],
    )
    recorded_pairs = frozenset(episode_pairs[: settings["video_episodes_per_task"]])
    episodes_per_checkpoint = len(settings["tasks"]) * len(episode_pairs)

    with TrainingProgress(
        enabled=not args.no_progress,
        total_stages=len(settings["train_seeds"]),
    ) as progress:
        for train_seed in settings["train_seeds"]:
            phase = progress.add_phase(
                f"M1 rollout seed-{train_seed}",
                episodes_per_checkpoint,
                show_loss_chart=False,
            )
            checkpoint = Path(checkpoint_evidence[str(train_seed)]["path"])
            model, flow, legacy_world, legacy_flow, metadata = load_m1_checkpoint(
                checkpoint,
                device=device,
                expected_schema_version=str(config["data"]["schema_version"]),
            )
            _validate_checkpoint_metadata(
                metadata,
                config=config,
                model=model,
                train_seed=train_seed,
            )
            base_policy = MultimodalJointWAMPolicy(
                model,
                flow,
                legacy_world,
                legacy_flow,
                _policy_config(config),
                device=device,
            )
            policy = _InterventionPolicy(base_policy, CLEAN)
            for task_id in settings["tasks"]:
                env = VisualRequiredEnv(
                    VisualRequiredEnvConfig(
                        task_id=task_id,
                        control_dt=1.0 / settings["control_hz"],
                        episode_len=settings["max_steps"],
                        image_width=settings["policy_image_size"],
                        image_height=settings["policy_image_size"],
                        render_cue_mode="truth",
                    )
                )
                try:
                    _warm_policy_runtime(
                        env,
                        policy,
                        task_id=task_id,
                        consumes_state=True,
                        consumes_vision=True,
                        seed=2 * int(episode_pairs[0][0]),
                        image_size=settings["policy_image_size"],
                    )
                    plain_runner = _runner(
                        env,
                        policy,
                        config=config,
                        task_id=task_id,
                        settings=settings,
                        record_video=False,
                    )
                    video_runner = _runner(
                        env,
                        policy,
                        config=config,
                        task_id=task_id,
                        settings=settings,
                        record_video=True,
                    )
                    for physical_seed, cue_id in episode_pairs:
                        should_record = (physical_seed, cue_id) in recorded_pairs
                        video_path = (
                            _video_path(
                                video_root,
                                train_seed=train_seed,
                                task_id=task_id,
                                physical_seed=physical_seed,
                                cue_id=cue_id,
                            )
                            if should_record
                            else None
                        )
                        episode_index = len(records)
                        episode_seed = 2 * physical_seed + cue_id
                        observer: StreamingVideoObserver | None = None
                        with ExitStack() as stack:
                            observers: tuple[StreamingVideoObserver, ...] = ()
                            if video_path is not None:
                                observer = stack.enter_context(
                                    StreamingVideoObserver(
                                        video_path,
                                        stream=VIDEO_STREAM,
                                        fps=settings["control_hz"],
                                        codec=settings["video_codec"],
                                        frame_getter=(
                                            lambda transition: transition.next_images[
                                                VIDEO_STREAM
                                            ]
                                        ),
                                    )
                                )
                                observers = (observer,)
                            summary = (
                                video_runner if should_record else plain_runner
                            ).run_episode(
                                seed=episode_seed,
                                episode_index=episode_index,
                                randomize=True,
                                observers=observers,
                            )
                        diagnostics = dict(policy.last_diagnostics)
                        execution = _execution_evidence(policy, diagnostics)
                        record = {
                            "train_seed": int(train_seed),
                            "task_id": task_id,
                            "physical_seed": int(physical_seed),
                            "cue_id": int(cue_id),
                            "episode_seed": int(episode_seed),
                            "success": bool(summary.final_info.get("success", False)),
                            "failure": bool(summary.final_info.get("failure", False)),
                            "failure_reason": str(
                                summary.final_info.get("failure_reason", "")
                            ),
                            "steps": int(summary.steps),
                            "total_reward": float(summary.total_reward),
                            "model_variant": PRIMARY_VARIANT,
                            "condition": CLEAN,
                            **execution,
                            "video_path": (
                                str(video_path.resolve())
                                if video_path is not None
                                else None
                            ),
                        }
                        if observer is not None and video_path is not None:
                            evidence = _finalize_video_evidence(
                                video_path,
                                record=record,
                                frames_written=int(observer.frames_written),
                                settings=settings,
                            )
                            video_evidence.append(evidence)
                            record["video_sidecar"] = evidence["sidecar_path"]
                            record["video_sha256"] = evidence["video_sha256"]
                        else:
                            record["video_sidecar"] = None
                            record["video_sha256"] = None
                        records.append(record)
                        _write_jsonl(records_path, records)
                        phase.advance(
                            {
                                "episode": len(records),
                                "task": task_id,
                                "success": int(record["success"]),
                            }
                        )
                finally:
                    env.close()
            checkpoint_records = [
                item for item in records if item["train_seed"] == train_seed
            ]
            successes = sum(int(item["success"]) for item in checkpoint_records)
            phase.finish(
                f"success {successes}/{len(checkpoint_records)} "
                f"({successes / len(checkpoint_records):.1%})"
            )

    report = {
        "format_version": FORMAT_VERSION,
        "formal_protocol": False,
        "diagnostic_only": True,
        "completed": True,
        "execution_contract_passed": True,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "training_summary": str(training_summary_path),
        "training_summary_sha256": _sha256(training_summary_path),
        "checkpoint_evidence": checkpoint_evidence,
        "protocol": {
            "model_variant": PRIMARY_VARIANT,
            "condition": CLEAN,
            "train_seeds": list(settings["train_seeds"]),
            "tasks": list(settings["tasks"]),
            "physical_seed_start": settings["physical_seed_start"],
            "physical_seed_count": settings["physical_seed_count"],
            "cue_variants": list(settings["cue_variants"]),
            "episode_seed_mapping": "2 * physical_seed + cue_id",
            "randomize": True,
            "max_steps": settings["max_steps"],
            "control_hz": settings["control_hz"],
            "policy_visual_hz": settings["visual_hz"],
            "policy_stream": {
                "name": POLICY_STREAM,
                "camera": "fixed",
                "width": settings["policy_image_size"],
                "height": settings["policy_image_size"],
                "exposed_to_policy": True,
            },
            "video_stream": {
                "name": VIDEO_STREAM,
                "camera": "fixed",
                "width": settings["video_width"],
                "height": settings["video_height"],
                "fps": settings["control_hz"],
                "exposed_to_policy": False,
                "frame_alignment": "post_action_next_image_includes_terminal",
            },
        },
        "aggregation": _aggregate_records(records),
        "episode_records": str(records_path.resolve()),
        "episode_records_sha256": _sha256(records_path),
        "videos": {
            "requested_per_train_seed_task": settings["video_episodes_per_task"],
            "recorded": len(video_evidence),
            "all_verified": all(item["verified"] for item in video_evidence),
            "directory": str(video_root.resolve()),
            "evidence": video_evidence,
        },
        "limitations": [
            "This is a diagnostic rollout and does not replace canonical M1 acceptance.",
            "MP4 is lossy visual evidence; JSONL success records remain canonical for this run.",
            "The recorded frames are simulator-camera observations, not decoded future-head predictions.",
        ],
    }
    _write_json(report_path, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "episodes": report["aggregation"]["overall"]["episodes"],
                "success_rate": report["aggregation"]["overall"]["success_rate"],
                "videos": report["videos"]["recorded"],
                "execution_contract_passed": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _resolve_settings(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> dict[str, Any]:
    evaluation = _mapping(config, "evaluation")
    training = _mapping(config, "training")
    train_seeds = tuple(int(value) for value in (args.train_seeds or training["seeds"]))
    tasks = tuple(str(value) for value in (args.tasks or VISUAL_REQUIRED_TASKS))
    cue_variants = tuple(int(value) for value in args.cue_variants)
    formal_start = int(evaluation["physical_seed_start"])
    formal_count = int(evaluation["formal_physical_seeds_per_task"])
    return {
        "checkpoint_root": (
            args.checkpoint_root or ROOT / str(training["checkpoint_root"])
        ).resolve(),
        "training_summary": (
            args.training_summary
            or ROOT / str(training["report_root"]) / "training_summary.json"
        ).resolve(),
        "output": args.output_dir.resolve(),
        "train_seeds": train_seeds,
        "tasks": tasks,
        "physical_seed_count": int(args.physical_seeds),
        "physical_seed_start": int(
            args.physical_seed_start
            if args.physical_seed_start is not None
            else formal_start + formal_count
        ),
        "cue_variants": cue_variants,
        "max_steps": int(
            args.max_steps if args.max_steps is not None else evaluation["max_steps"]
        ),
        "control_hz": float(evaluation["control_hz"]),
        "visual_hz": float(evaluation["visual_refresh_hz"]),
        "policy_image_size": int(_mapping(config, "model")["vision_input_size"]),
        "video_episodes_per_task": int(args.video_episodes_per_task),
        "video_width": int(args.video_width),
        "video_height": int(args.video_height),
        "video_codec": str(args.video_codec),
        "torch_threads": int(args.torch_threads),
    }


def _validate_settings(settings: Mapping[str, Any]) -> None:
    for key in ("train_seeds", "tasks", "cue_variants"):
        values = tuple(settings[key])
        if not values:
            raise ValueError(f"{key} cannot be empty")
        if len(set(values)) != len(values):
            raise ValueError(f"{key} cannot contain duplicates")
    if not set(settings["tasks"]).issubset(set(VISUAL_REQUIRED_TASKS)):
        raise ValueError("rollout tasks must be visual-required tasks")
    if not set(settings["cue_variants"]).issubset({0, 1}):
        raise ValueError("cue variants must be 0 or 1")
    positive = (
        "physical_seed_count",
        "max_steps",
        "video_width",
        "video_height",
        "torch_threads",
    )
    if any(int(settings[key]) <= 0 for key in positive):
        raise ValueError(f"{positive} must all be positive")
    video_count = int(settings["video_episodes_per_task"])
    total_per_task = int(settings["physical_seed_count"]) * len(
        settings["cue_variants"]
    )
    if video_count < 0 or video_count > total_per_task:
        raise ValueError(
            "video-episodes-per-task must be in [0, physical_seeds * cue_variants]"
        )
    if len(str(settings["video_codec"])) != 4:
        raise ValueError("video codec must contain four characters")
    control_hz = float(settings["control_hz"])
    visual_hz = float(settings["visual_hz"])
    if not np.isfinite(control_hz) or not np.isfinite(visual_hz):
        raise ValueError("rollout rates must be finite")
    if control_hz <= 0.0 or visual_hz <= 0.0 or visual_hz > control_hz:
        raise ValueError("rollout rates must satisfy 0 < visual_hz <= control_hz")


def _validate_training_summary(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    config_path: Path,
    train_seeds: Sequence[int],
) -> None:
    if summary.get("format_version") != "wam.multimodal.m1.training/1":
        raise ValueError("unsupported M1 training summary")
    if summary.get("formal_protocol") is not True or summary.get("passed") is not True:
        raise ValueError("M1 rollout requires a passed formal training summary")
    if summary.get("config_sha256") != _sha256(config_path):
        raise ValueError("M1 rollout config differs from the training summary")
    declared_config = Path(str(summary.get("config", "")))
    declared_config = (
        declared_config if declared_config.is_absolute() else ROOT / declared_config
    ).resolve()
    if declared_config != config_path:
        raise ValueError("M1 rollout config path differs from the training summary")
    validate_training_summary_vision(summary, config, project_root=ROOT)
    available = {int(value) for value in summary.get("train_seeds", ())}
    missing = sorted(set(int(value) for value in train_seeds) - available)
    if missing:
        raise ValueError(f"training summary does not contain train seeds {missing}")


def _checkpoint_evidence(
    checkpoint_root: Path,
    train_seeds: Sequence[int],
    summary: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    expected_hashes = _mapping(_mapping(summary, "checkpoint_sha256"), PRIMARY_VARIANT)
    strict_reloads = _mapping(_mapping(summary, "strict_reload"), PRIMARY_VARIANT)
    result: dict[str, dict[str, Any]] = {}
    for train_seed in train_seeds:
        checkpoint = checkpoint_root / PRIMARY_VARIANT / f"seed_{train_seed}"
        observed = checkpoint_tree_sha256(checkpoint)
        expected = str(expected_hashes.get(str(train_seed), ""))
        strict = strict_reloads.get(str(train_seed))
        if observed != expected:
            raise ValueError(f"checkpoint tree mismatch for train seed {train_seed}")
        if not isinstance(strict, Mapping) or strict.get("passed") is not True:
            raise ValueError(
                f"checkpoint strict reload is not passed for seed {train_seed}"
            )
        if float(strict.get("max_abs_diff", float("inf"))) != 0.0:
            raise ValueError(f"checkpoint strict reload differs for seed {train_seed}")
        result[str(train_seed)] = {
            "path": str(checkpoint.resolve()),
            "tree_sha256": observed,
            "strict_reload": dict(strict),
        }
    return result


def _validate_checkpoint_metadata(
    metadata: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    model: Any,
    train_seed: int,
) -> None:
    schema = _mapping(metadata, "schema")
    if schema.get("model_variant") != PRIMARY_VARIANT:
        raise ValueError(
            "rollout checkpoint is not the primary state_vision_future model"
        )
    if int(schema.get("train_seed", -1)) != int(train_seed):
        raise ValueError(
            "rollout checkpoint train seed differs from the requested seed"
        )
    validate_loaded_checkpoint_vision(config, model, metadata)


def _runner(
    env: VisualRequiredEnv,
    policy: _InterventionPolicy,
    *,
    config: Mapping[str, Any],
    task_id: str,
    settings: Mapping[str, Any],
    record_video: bool,
) -> SimulationRunner:
    render = [
        RenderRequest(
            POLICY_STREAM,
            "fixed",
            width=int(settings["policy_image_size"]),
            height=int(settings["policy_image_size"]),
            fps=float(settings["visual_hz"]),
        )
    ]
    if record_video:
        render.append(
            RenderRequest(
                VIDEO_STREAM,
                "fixed",
                width=int(settings["video_width"]),
                height=int(settings["video_height"]),
            )
        )
    return SimulationRunner(
        env,
        policy,
        RunnerConfig(
            max_steps=int(settings["max_steps"]),
            render=tuple(render),
            policy_observation_keys=("proprioception",),
            expose_privileged_state_to_policy=False,
            expose_rendered_images_to_policy=True,
            policy_image_streams=(POLICY_STREAM,),
            expose_task_to_policy=True,
            task_id=task_id,
            task=VISUAL_REQUIRED_TASK_TEXTS[task_id],
            policy_action_history=int(_mapping(config, "data")["state_history"] - 1),
        ),
    )


def _execution_evidence(
    policy: _InterventionPolicy,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    source = str(diagnostics.get("action_source", ""))
    expected_source = f"m1_{PRIMARY_VARIANT}"
    fallback = bool(diagnostics.get("fallback_used", False))
    privileged = bool(diagnostics.get("privileged_state_seen", False))
    presented = tuple(
        str(value) for value in diagnostics.get("presented_observation_paths", ())
    )
    consumed = tuple(
        str(value) for value in diagnostics.get("consumed_observation_paths", ())
    )
    video_leakage = any(
        value == f"images.{VIDEO_STREAM}"
        or value.startswith(f"images.{VIDEO_STREAM}.")
        or value == f"image_frame_indices.{VIDEO_STREAM}"
        for value in (*presented, *consumed)
    )
    checks = {
        "action_source": source == expected_source,
        "no_fallback": not fallback,
        "no_privileged_observation": not privileged,
        "actions_finite_and_bounded": bool(policy.actions_finite_and_bounded),
        "policy_consumed_fixed_rgb": f"images.{POLICY_STREAM}" in consumed,
        "video_stream_not_presented_or_consumed": not video_leakage,
    }
    if not all(checks.values()):
        raise RuntimeError(f"M1 rollout execution contract failed: {checks}")
    return {
        "action_source": source,
        "fallback_used": fallback,
        "privileged_observation_seen": privileged,
        "actions_finite_and_bounded": bool(policy.actions_finite_and_bounded),
        "presented_observation_paths": list(presented),
        "consumed_observation_paths": list(consumed),
        "replan_events": int(policy.replan_events),
        "cold_replan_events": int(policy.cold_replan_events),
        "warm_replan_events": int(policy.warm_replan_events),
        "execution_checks": checks,
    }


def _episode_pairs(
    *,
    physical_seed_start: int,
    physical_seed_count: int,
    cue_variants: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    pairs = tuple(
        (physical_seed, int(cue_id))
        for physical_seed in range(
            int(physical_seed_start),
            int(physical_seed_start) + int(physical_seed_count),
        )
        for cue_id in cue_variants
    )
    if len(set(pairs)) != len(pairs):
        raise ValueError("rollout physical-seed/cue matrix contains duplicates")
    return pairs


def _video_path(
    root: Path,
    *,
    train_seed: int,
    task_id: str,
    physical_seed: int,
    cue_id: int,
) -> Path:
    return (
        root
        / f"train_seed_{train_seed}"
        / task_id
        / f"physical_seed_{physical_seed}_cue_{cue_id}.mp4"
    )


def _finalize_video_evidence(
    path: Path,
    *,
    record: Mapping[str, Any],
    frames_written: int,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    probe = _probe_video(path)
    expected_steps = int(record["steps"])
    expected_width = int(settings["video_width"])
    expected_height = int(settings["video_height"])
    expected_fps = float(settings["control_hz"])
    checks = {
        "writer_frames_match_steps": int(frames_written) == expected_steps,
        "decoded_frames_match_steps": int(probe["decoded_frames"]) == expected_steps,
        "container_frames_match_steps": int(probe["container_frames"])
        == expected_steps,
        "width_matches": int(probe["width"]) == expected_width,
        "height_matches": int(probe["height"]) == expected_height,
        "fps_matches": abs(float(probe["fps"]) - expected_fps) <= 0.05,
        "nonempty": path.stat().st_size > 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"encoded M1 rollout video contract failed: {checks}")
    sidecar_path = path.with_suffix(".json")
    sidecar = {
        "format_version": VIDEO_FORMAT_VERSION,
        "verified": True,
        "train_seed": int(record["train_seed"]),
        "task_id": str(record["task_id"]),
        "physical_seed": int(record["physical_seed"]),
        "cue_id": int(record["cue_id"]),
        "episode_seed": int(record["episode_seed"]),
        "success": bool(record["success"]),
        "failure": bool(record["failure"]),
        "failure_reason": str(record["failure_reason"]),
        "steps": expected_steps,
        "action_source": str(record["action_source"]),
        "fallback_used": bool(record["fallback_used"]),
        "privileged_observation_seen": bool(record["privileged_observation_seen"]),
        "actions_finite_and_bounded": bool(record["actions_finite_and_bounded"]),
        "terminal_frame_included": True,
        "frame_alignment": "transition.next_images.rollout",
        "camera": "fixed",
        "fps": expected_fps,
        "width": expected_width,
        "height": expected_height,
        "policy_stream_exposed": POLICY_STREAM,
        "video_stream_exposed_to_policy": False,
        "frames_written": int(frames_written),
        "video_path": str(path.resolve()),
        "sidecar_path": str(sidecar_path.resolve()),
        "video_bytes": path.stat().st_size,
        "video_sha256": _sha256(path),
        "video_probe": probe,
        "checks": checks,
    }
    _write_json(sidecar_path, sidecar)
    return sidecar


def _probe_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    try:
        opened = bool(capture.isOpened())
        container_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
        fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
        decoded = 0
        while opened:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.shape[:2] != (height, width):
                raise RuntimeError(f"decoded rollout frame shape is invalid: {path}")
            decoded += 1
    finally:
        capture.release()
    if not opened or container_frames <= 0 or decoded <= 0:
        raise RuntimeError(f"encoded M1 rollout video is unreadable: {path}")
    return {
        "opened": opened,
        "container_frames": container_frames,
        "decoded_frames": decoded,
        "width": width,
        "height": height,
        "fps": fps,
    }


def _aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate empty M1 rollout records")
    identities = [
        (
            int(item["train_seed"]),
            str(item["task_id"]),
            int(item["physical_seed"]),
            int(item["cue_id"]),
        )
        for item in records
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("M1 rollout records contain duplicate identities")

    def summary(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        successes = sum(int(bool(item["success"])) for item in values)
        episodes = len(values)
        return {
            "successes": successes,
            "episodes": episodes,
            "success_rate": successes / episodes,
            "mean_steps": float(np.mean([int(item["steps"]) for item in values])),
            "mean_return": float(
                np.mean([float(item["total_reward"]) for item in values])
            ),
        }

    groups: dict[str, dict[Any, list[Mapping[str, Any]]]] = {
        "by_task": defaultdict(list),
        "by_train_seed": defaultdict(list),
        "by_cue": defaultdict(list),
    }
    for item in records:
        groups["by_task"][str(item["task_id"])].append(item)
        groups["by_train_seed"][int(item["train_seed"])].append(item)
        groups["by_cue"][int(item["cue_id"])].append(item)
    return {
        "overall": summary(records),
        "by_task": {
            str(key): summary(values)
            for key, values in sorted(groups["by_task"].items())
        },
        "by_train_seed": {
            str(key): summary(values)
            for key, values in sorted(groups["by_train_seed"].items())
        },
        "by_cue": {
            str(key): summary(values)
            for key, values in sorted(groups["by_cue"].items())
        },
    }


def _prepare_output_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"rollout output is not a directory: {path}")
        stale = sorted(
            str(item.relative_to(path)) for item in path.rglob("*") if item.is_file()
        )
        if stale:
            raise FileExistsError(
                f"refusing to mix stale M1 rollout evidence in {path}: {stale[:5]}"
            )
    path.mkdir(parents=True, exist_ok=True)


def _configure_torch_threads(count: int) -> None:
    if count <= 0:
        raise ValueError("torch-threads must be positive")
    torch.set_num_threads(int(count))
    torch.set_num_interop_threads(1)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return device


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"missing mapping {key!r}")
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("M1 config must contain a mapping")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
