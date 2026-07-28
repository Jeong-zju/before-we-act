#!/usr/bin/env python3
"""Serve any native RoboFactory task to a Phase M2 inference process."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robofactory_rpc import (  # noqa: E402
    configure_socket,
    extract_robofactory_multiview_observation,
    receive_message,
    scalar_bool,
    scalar_float,
    send_message,
    split_robofactory_action,
    wilson_interval,
)


TASKS = {
    "CameraAlignment-rf": ("camera_alignment", "Align the cameras together", "camera_alignment.yaml"),
    "LiftBarrier-rf": ("lift_barrier", "Lift the barrier together", "lift_barrier.yaml"),
    "LongPipelineDelivery-rf": (
        "long_pipeline_delivery",
        "Deliver the long pipeline together",
        "long_pipeline_delivery.yaml",
    ),
    "PassShoe-rf": ("pass_shoe", "Pass the shoe between robots", "pass_shoe.yaml"),
    "PickMeat-rf": ("pick_meat", "Pick the meat together", "pick_meat.yaml"),
    "PlaceFood-rf": ("place_food", "Place the food together", "place_food.yaml"),
    "StackCube-rf": ("stack_cube", "Stack the cubes together", "stack_cube.yaml"),
    "StrikeCube-rf": ("strike_cube", "Strike the cube together", "strike_cube.yaml"),
    "TakePhoto-rf": ("take_photo", "Take a photo together", "take_photo.yaml"),
    "ThreeRobotsStackCube-rf": (
        "three_robots_stack_cube",
        "Stack the cubes with three robots",
        "three_robots_stack_cube.yaml",
    ),
    "TwoRobotsStackCube-rf": (
        "two_robots_stack_cube",
        "Stack the cubes with two robots",
        "two_robots_stack_cube.yaml",
    ),
}

# These limits mirror the native RoboFactory registrations.  Validation is
# task-specific so long-horizon tasks are not truncated by the 500-step limit
# used by LiftBarrier and several shorter tasks.
TASK_MAX_EPISODE_STEPS = {
    "CameraAlignment-rf": 1500,
    "LiftBarrier-rf": 500,
    "LongPipelineDelivery-rf": 1500,
    "PassShoe-rf": 500,
    "PickMeat-rf": 500,
    "PlaceFood-rf": 500,
    "StackCube-rf": 200,
    "StrikeCube-rf": 350,
    "TakePhoto-rf": 1500,
    "ThreeRobotsStackCube-rf": 800,
    "TwoRobotsStackCube-rf": 500,
}
if set(TASK_MAX_EPISODE_STEPS) != set(TASKS):  # pragma: no cover - invariant.
    raise RuntimeError("M2 task and episode-limit registries differ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robofactory-root", type=Path, default=WORKSPACE / "RoboFactory")
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--scene", choices=("table", "robocasa"), default="table")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8872)
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--sim-backend", choices=("cpu", "gpu", "auto"), default="cpu")
    parser.add_argument("--shader", choices=("default", "rt-fast", "rt"), default="default")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--future-path", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--socket-timeout", type=float, default=600.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    robofactory_root = args.robofactory_root.expanduser().resolve(strict=True)
    task_id, task_text, config_name = TASKS[args.task]
    config_path = (
        robofactory_root / "robofactory/configs" / args.scene / config_name
    ).resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    _prepare_output(output_dir)
    videos = output_dir / "videos"
    if not args.no_video:
        videos.mkdir(parents=True)
    episode_path = output_dir / "rollout_episodes.jsonl"
    summary_path = output_dir / "rollout_summary.json"
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    listener: socket.socket | None = None
    connection: socket.socket | None = None
    env: Any = None
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    fatal_error: dict[str, str] | None = None
    client: dict[str, Any] | None = None
    contract: dict[str, Any] | None = None
    try:
        env = _make_environment(
            robofactory_root=robofactory_root,
            config_path=config_path,
            environment_id=args.task,
            sim_backend=args.sim_backend,
            shader=args.shader,
            videos=videos,
            video_fps=args.video_fps,
            record_video=not args.no_video,
        )
        first_observation, _ = _seeded_reset(env, seeds[0])
        raw_agents = first_observation.get("agent")
        if not isinstance(raw_agents, Mapping):
            raise RuntimeError("RoboFactory reset lacks an agent mapping")
        agent_order = tuple(sorted(map(str, raw_agents), key=_natural_key))
        camera_sources = _select_native_cameras(
            first_observation,
            agent_order=agent_order,
        )
        first_state, first_images = extract_robofactory_multiview_observation(
            first_observation,
            agent_order=agent_order,
            camera_names=camera_sources,
        )
        contract = {
            "environment_id": args.task,
            "scene": args.scene,
            "task_id": task_id,
            "task_text": task_text,
            "agent_order": list(agent_order),
            "agent_count": len(agent_order),
            "state_dim": int(first_state.size),
            "action_dim": 8 * len(agent_order),
            "camera_order": list(camera_sources),
            "camera_sources": dict(camera_sources),
            "camera_shapes": {
                camera: list(image.shape)
                for camera, image in first_images.items()
            },
            "control_mode": "pd_joint_pos",
            "success_source": "info.success",
            "future_path": bool(args.future_path),
        }
        listener = _listen(args.host, args.port, timeout=args.socket_timeout)
        print(
            f"[environment] {args.task} ready on {args.host}:{args.port}; "
            "waiting for M2 inference…",
            flush=True,
        )
        connection, peer = listener.accept()
        configure_socket(connection, timeout_seconds=args.socket_timeout)
        print(f"[environment] inference connected from {peer}", flush=True)
        send_message(
            connection,
            {
                "type": "hello",
                "role": "robofactory_m2_environment",
                "contract": contract,
                "episodes": args.episodes,
                "seeds": seeds,
                "video_enabled": not args.no_video,
            },
        )
        ready, arrays = receive_message(connection)
        _raise_peer_error(ready)
        if arrays or ready.get("type") != "ready":
            raise RuntimeError("M2 peer did not return a valid ready handshake")
        if ready.get("accepted_contract") != contract:
            raise RuntimeError("M2 peer altered the environment contract")
        client = _validate_client(ready.get("client"), contract=contract)
        pending = first_observation
        for episode_index, seed in enumerate(seeds):
            if episode_index:
                pending, _ = _seeded_reset(env, seed)
            state, images = extract_robofactory_multiview_observation(
                pending,
                agent_order=agent_order,
                camera_names=camera_sources,
            )
            success = terminated = truncated = False
            latencies: list[float] = []
            round_trips: list[float] = []
            action_sources: set[str] = set()
            episode_return = 0.0
            step = 0
            episode_started = time.perf_counter()
            while step < args.max_steps:
                request_id = f"{episode_index}:{step}"
                request_started = time.perf_counter()
                send_message(
                    connection,
                    {
                        "type": "observation",
                        "request_id": request_id,
                        "episode_index": episode_index,
                        "seed": seed,
                        "step": step,
                        "reset": step == 0,
                        "task": {"id": task_id, "text": task_text},
                        "image_frame_index": step,
                    },
                    {
                        "proprioception": state,
                        **{
                            _rgb_array_name(camera): image
                            for camera, image in images.items()
                        },
                    },
                )
                message, action_arrays = receive_message(connection)
                _raise_peer_error(message)
                round_trips.append((time.perf_counter() - request_started) * 1000.0)
                if (
                    message.get("type") != "action"
                    or message.get("request_id") != request_id
                    or int(message.get("episode_index", -1)) != episode_index
                    or int(message.get("step", -1)) != step
                    or set(action_arrays) != {"action"}
                ):
                    raise RuntimeError("M2 action response identity is invalid")
                action = action_arrays["action"]
                if (
                    action.shape != (contract["action_dim"],)
                    or action.dtype != np.float32
                    or not np.isfinite(action).all()
                ):
                    raise RuntimeError("M2 response action violates the task contract")
                diagnostics = message.get("diagnostics")
                if not isinstance(diagnostics, Mapping):
                    raise RuntimeError("M2 response lacks diagnostics")
                source = str(diagnostics.get("action_source", ""))
                expected_source = str(client["policy"]["action_source"])
                if (
                    source != expected_source
                    or diagnostics.get("fallback_used") is not False
                    or diagnostics.get("direct_model_action") is not True
                ):
                    raise RuntimeError("M2 response did not prove direct model control")
                action_sources.add(source)
                latency = float(message.get("inference_latency_ms", math.nan))
                if not math.isfinite(latency) or latency < 0.0:
                    raise RuntimeError("M2 response latency is invalid")
                latencies.append(latency)
                pending, raw_reward, raw_terminated, raw_truncated, info = env.step(
                    split_robofactory_action(action, agent_order=agent_order)
                )
                episode_return += scalar_float(raw_reward, name="reward")
                step += 1
                terminated = scalar_bool(raw_terminated, name="terminated")
                truncated = scalar_bool(raw_truncated, name="truncated")
                if not isinstance(info, Mapping) or "success" not in info:
                    raise RuntimeError("RoboFactory info lacks success")
                success = success or scalar_bool(info["success"], name="info.success")
                if success or terminated or truncated or step >= args.max_steps:
                    break
                state, images = extract_robofactory_multiview_observation(
                    pending,
                    agent_order=agent_order,
                    camera_names=camera_sources,
                )
            video = _flush_video(
                env,
                videos=videos,
                episode_index=episode_index,
                seed=seed,
                success=success,
                enabled=not args.no_video,
            )
            result = {
                "format_version": "wam.robofactory.m2.rollout_episode/1",
                "episode_index": episode_index,
                "seed": seed,
                "task_id": task_id,
                "success": success,
                "episode_return": episode_return,
                "steps": step,
                "terminated": terminated,
                "truncated": truncated,
                "video": video,
                "action_sources": sorted(action_sources),
                "inference_latency_ms": _latency(latencies),
                "rpc_round_trip_latency_ms": _latency(round_trips),
                "duration_seconds": time.perf_counter() - episode_started,
            }
            results.append(result)
            _append_jsonl(episode_path, result)
            send_message(
                connection,
                {
                    "type": "episode_result",
                    **result,
                    "successes_so_far": sum(value["success"] for value in results),
                },
            )
            print(
                f"[environment] {args.task} {episode_index + 1}/{args.episodes} "
                f"seed={seed} success={success} steps={step}",
                flush=True,
            )
        summary = _summary(
            args=args,
            contract=contract,
            client=client,
            results=results,
            fatal_error=None,
            elapsed=time.perf_counter() - started,
        )
        _write_json(summary_path, summary)
        send_message(connection, {"type": "summary", "summary": summary})
        print(f"[environment] summary: {summary_path}", flush=True)
        return 0
    except BaseException as exc:
        fatal_error = {"type": type(exc).__name__, "message": str(exc)}
        if connection is not None:
            try:
                send_message(connection, {"type": "error", "fatal": True, "error": fatal_error})
            except BaseException:
                pass
        summary = _summary(
            args=args,
            contract=contract,
            client=client,
            results=results,
            fatal_error=fatal_error,
            elapsed=time.perf_counter() - started,
        )
        _write_json(summary_path, summary)
        raise
    finally:
        if env is not None:
            env.close()
        if connection is not None:
            connection.close()
        if listener is not None:
            listener.close()


def _make_environment(
    *,
    robofactory_root: Path,
    config_path: Path,
    environment_id: str,
    sim_backend: str,
    shader: str,
    videos: Path,
    video_fps: int,
    record_video: bool,
) -> Any:
    if str(robofactory_root) not in sys.path:
        sys.path.insert(0, str(robofactory_root))
    try:
        import gymnasium as gym
        import robofactory  # noqa: F401
        from robofactory.utils.wrappers.record import RecordEpisodeMA
    except ImportError as exc:
        raise RuntimeError("run the M2 environment server in RoboFactory Python 3.9") from exc
    env = gym.make(
        environment_id,
        config=str(config_path),
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="sensors",
        reward_mode="dense",
        sensor_configs={"shader_pack": shader},
        human_render_camera_configs={"shader_pack": shader},
        viewer_camera_configs={"shader_pack": shader},
        num_envs=1,
        sim_backend=sim_backend,
        parallel_in_single_scene=False,
    )
    if not record_video:
        return env
    return RecordEpisodeMA(
        env,
        output_dir=str(videos),
        save_trajectory=False,
        save_video=True,
        info_on_video=False,
        save_on_reset=False,
        max_steps_per_video=None,
        clean_on_close=False,
        record_reward=False,
        record_env_state=False,
        record_observation=False,
        video_fps=video_fps,
        avoid_overwriting_video=True,
        source_type="m2_closed_loop",
        source_desc="Phase M2 RoboFactory-only direct closed-loop rollout",
    )


def _validate_client(raw: Any, *, contract: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise RuntimeError("M2 ready.client must be an object")
    value = dict(raw)
    checkpoint_format = value.get("checkpoint_format")
    supported_sources = {
        "wam.robofactory.m2.checkpoint/5": {
            "m2_block_causal_fast_path",
            "m2_block_causal_future_path",
        },
        "wam.robofactory.static_rgb_act_moe.checkpoint/1": {
            "static_rgb_dino_act_moe",
            "static_rgb_dino_act_dense",
        },
    }
    if checkpoint_format not in supported_sources:
        raise RuntimeError("client did not load a supported direct-policy checkpoint")
    if contract["task_id"] not in value.get("task_vocabulary", []):
        raise RuntimeError("M2 checkpoint does not contain the requested task")
    if value.get("future_path") is not contract["future_path"]:
        raise RuntimeError("M2 client/environment future-path modes differ")
    policy = value.get("policy")
    if (
        not isinstance(policy, Mapping)
        or policy.get("action_source") not in supported_sources[checkpoint_format]
    ):
        raise RuntimeError("client did not declare a supported direct action source")
    return value


def _summary(
    *,
    args: argparse.Namespace,
    contract: Mapping[str, Any] | None,
    client: Mapping[str, Any] | None,
    results: Sequence[Mapping[str, Any]],
    fatal_error: Mapping[str, str] | None,
    elapsed: float,
) -> dict[str, Any]:
    successes = sum(bool(value["success"]) for value in results)
    episodes = len(results)
    interval = wilson_interval(successes, episodes) if episodes else (0.0, 0.0)
    returns = [float(value["episode_return"]) for value in results]
    completed = fatal_error is None and episodes == args.episodes
    expected_source = (
        None
        if client is None
        else client.get("policy", {}).get("action_source")
    )
    direct = (
        completed
        and isinstance(expected_source, str)
        and all(
            value.get("action_sources") == [expected_source]
            for value in results
        )
    )
    return {
        "format_version": "wam.robofactory.m2.rollout_summary/2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_source_policy": "robofactory_only_no_custom_scenes",
        "completed": completed,
        "fatal_error": fatal_error,
        "task": args.task,
        "scene": args.scene,
        "future_path": bool(args.future_path),
        "episodes_requested": args.episodes,
        "episodes_completed": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else 0.0,
        "success_rate_wilson_95": list(interval),
        "episode_return": (
            {
                "mean": float(np.mean(returns)),
                "p50": float(np.percentile(returns, 50)),
                "min": float(np.min(returns)),
                "max": float(np.max(returns)),
            }
            if returns
            else None
        ),
        "direct_model_action_coverage": 1.0 if direct else 0.0,
        "engineering_smoke_passed": completed and direct,
        "closed_loop_smoke_passed": completed and direct and successes > 0,
        "formal_benchmark": {
            "reportable": completed and direct and args.episodes >= 100,
            "minimum_episodes": 100,
        },
        "contract": contract,
        "client": client,
        "episodes": list(results),
        "elapsed_seconds": elapsed,
        "output_dir": str(args.output_dir.expanduser().resolve()),
    }


def _validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.port <= 65535 or args.episodes <= 0:
        raise ValueError("invalid M2 port/episode count")
    if args.seed_start < 0:
        raise ValueError("invalid M2 seed_start")
    task_limit = TASK_MAX_EPISODE_STEPS.get(str(args.task))
    if task_limit is None:
        raise ValueError(f"unknown M2 task {args.task!r}")
    if not 1 <= args.max_steps <= task_limit:
        raise ValueError(
            f"--max-steps for {args.task} must be in [1,{task_limit}]"
        )
    if args.video_fps <= 0 or args.socket_timeout <= 0.0:
        raise ValueError("video_fps/socket_timeout must be positive")
    if not args.allow_remote and args.host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("refusing unauthenticated non-loopback bind")


def _prepare_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.rglob("*"))):
        raise FileExistsError(f"M2 rollout output must be a fresh directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _listen(host: str, port: int, *, timeout: float) -> socket.socket:
    listener = socket.socket(
        socket.AF_INET6 if ":" in host else socket.AF_INET,
        socket.SOCK_STREAM,
    )
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(1)
    listener.settimeout(timeout)
    return listener


def _raise_peer_error(message: Mapping[str, Any]) -> None:
    if message.get("type") == "error":
        raise RuntimeError(f"M2 inference failed: {message.get('error')}")


def _flush_video(
    env: Any,
    *,
    videos: Path,
    episode_index: int,
    seed: int,
    success: bool,
    enabled: bool,
) -> str | None:
    if not enabled:
        return None
    status = "success" if success else "failure"
    name = f"episode_{episode_index:04d}_seed_{seed}_{status}"
    target = videos / f"{name}.mp4"
    if target.exists():
        raise FileExistsError(target)
    env.flush_video(name=name, verbose=False)
    if not target.is_file() or target.stat().st_size <= 0:
        raise RuntimeError(f"RoboFactory did not write {target}")
    return str(target.relative_to(videos.parent))


def _latency(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise RuntimeError("M2 latency evidence is empty/non-finite")
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _seeded_reset(env: Any, seed: int) -> Any:
    return env.reset(seed=seed)


def _select_native_cameras(
    observation: Mapping[str, Any],
    *,
    agent_order: Sequence[str],
) -> dict[str, str]:
    sensors = observation.get("sensor_data")
    if not isinstance(sensors, Mapping):
        raise RuntimeError("RoboFactory reset lacks sensor_data")
    available = set(map(str, sensors))
    selected: dict[str, str] = {}
    for candidate in ("head_camera_global", "head_camera"):
        if candidate in available:
            selected["global"] = candidate
            break
    if "global" not in selected:
        raise RuntimeError(
            "RoboFactory task lacks the canonical workspace RGB sensor; "
            f"available={sorted(available)}"
        )
    for index, _agent in enumerate(agent_order):
        source = f"head_camera_agent{index}"
        if source not in available:
            raise RuntimeError(
                "RoboFactory task lacks an agent RGB sensor required by M2; "
                f"missing={source!r}, available={sorted(available)}"
            )
        selected[f"agent_{index}"] = source
    return selected


def _rgb_array_name(camera: str) -> str:
    if not re.fullmatch(r"[a-z0-9_]+", camera):
        raise ValueError(f"invalid logical camera name {camera!r}")
    return f"rgb_{camera}"


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(item) if item.isdigit() else item for item in re.split(r"(\d+)", value))


if __name__ == "__main__":
    raise SystemExit(main())
