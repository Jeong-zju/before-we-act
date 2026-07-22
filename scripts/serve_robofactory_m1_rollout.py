"""Serve RoboFactory LiftBarrier episodes to an external M1 inference process.

Run this file with the RoboFactory Python 3.9 environment.  Model code is not
imported here; observations/actions cross a loopback TCP socket using the
lossless protocol in :mod:`robofactory_rpc`.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from typing import Any, Optional, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robofactory_rpc import (  # noqa: E402
    FORMAL_LIFTBARRIER_M1_CONFIG_SHA256,
    PROTOCOL_VERSION,
    configure_socket,
    extract_liftbarrier_observation,
    receive_message,
    scalar_bool,
    send_message,
    split_liftbarrier_action,
    wilson_interval,
)


SUMMARY_FORMAT = "wam.robofactory.closed_loop_summary/1"
EPISODE_FORMAT = "wam.robofactory.closed_loop_episode/1"
FORMAL_BENCHMARK_PROTOCOL = "robofactory.lift_barrier.m1.seed1000_n100/1"
FORMAL_LIFTBARRIER_ENV_CONFIG_SHA256 = (
    "2bddda0c45b10c2fedabefa6e4617f394499ee2f73724f8e9bc2074a9f3f443a"
)
FORMAL_LIFTBARRIER_RF_SOURCE_SHA256 = {
    "robofactory/tasks/lift_barrier.py": (
        "8eeb128123c9828547a306b2577b10dc33281dcb08c21ba352b7a53d22ab9abe"
    ),
    "robofactory/utils/scenes/scene_builder.py": (
        "9d0dfa6e4359917812059627a09ea5e23a6dac197d86d6f1c1455969ed6ec693"
    ),
    "robofactory/utils/wrappers/record.py": (
        "e7070da6d96588f1a9dc81cf1f125ce59641269da6088395df193595822623d3"
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run LiftBarrier in the RoboFactory environment, stream observations "
            "to M1, record MP4s, and report closed-loop success rate."
        )
    )
    parser.add_argument(
        "--robofactory-root",
        type=Path,
        default=WORKSPACE / "RoboFactory",
        help="RoboFactory repository root used by this Python environment.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="LiftBarrier YAML; defaults inside --robofactory-root.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Permit a non-loopback bind. The protocol has no authentication.",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--sim-backend", choices=("cpu", "gpu", "auto"), default="cpu")
    parser.add_argument("--shader", choices=("default", "rt-fast", "rt"), default="default")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable MP4 recording (intended only for debugging).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/robofactory_m1_closed_loop",
    )
    parser.add_argument(
        "--socket-timeout",
        type=float,
        default=600.0,
        help="Maximum seconds to wait for one peer message.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    robofactory_root = args.robofactory_root.expanduser().resolve(strict=True)
    config_path = (
        args.config.expanduser().resolve(strict=True)
        if args.config is not None
        else (
            robofactory_root / "robofactory/configs/table/lift_barrier.yaml"
        ).resolve(strict=True)
    )
    output_dir = args.output_dir.expanduser().resolve()
    _prepare_output_directory(output_dir)
    video_dir = output_dir / "videos"
    if not args.no_video:
        video_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "rollout_episodes.jsonl"
    summary_path = output_dir / "rollout_summary.json"
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))

    listener: Optional[socket.socket] = None
    connection: Optional[socket.socket] = None
    env: Any = None
    results: list[dict[str, Any]] = []
    client_metadata: Optional[dict[str, Any]] = None
    robofactory_provenance: Optional[dict[str, Any]] = None
    environment_contract: Optional[dict[str, Any]] = None
    action_min = np.full(16, np.inf, dtype=np.float32)
    action_max = np.full(16, -np.inf, dtype=np.float32)
    started = time.perf_counter()
    started_at = _utc_now()
    current_episode: Optional[Tuple[int, int]] = None
    fatal_error: Optional[dict[str, str]] = None
    completed = False

    try:
        robofactory_provenance = _robofactory_provenance(
            robofactory_root,
            config_path=config_path,
        )
        listener = _listen(args.host, args.port)
        print(
            f"[environment] RPC listening on {args.host}:{args.port} "
            f"({PROTOCOL_VERSION})",
            flush=True,
        )
        print("[environment] Initializing RoboFactory renderer and LiftBarrier…", flush=True)
        env = _make_environment(
            robofactory_root=robofactory_root,
            config_path=config_path,
            sim_backend=args.sim_backend,
            shader=args.shader,
            video_dir=video_dir,
            video_fps=args.video_fps,
            record_video=not args.no_video,
        )
        pending_observation, _ = _seeded_reset(env, seeds[0])
        first_state, first_rgb = extract_liftbarrier_observation(pending_observation)
        control_hz, environment_max_steps = _inspect_environment_limits(env)
        contract = _environment_contract(
            first_state,
            first_rgb,
            args.max_steps,
            control_hz=control_hz,
            environment_max_steps=environment_max_steps,
        )
        environment_contract = contract
        print(
            "[environment] Simulator ready; waiting for the M1 inference client…",
            flush=True,
        )
        connection, peer = listener.accept()
        configure_socket(connection, timeout_seconds=args.socket_timeout)
        print(f"[environment] Inference client connected from {peer}", flush=True)
        send_message(
            connection,
            {
                "type": "hello",
                "role": "robofactory_environment",
                "contract": contract,
                "episodes": args.episodes,
                "seeds": seeds,
                "video_enabled": not args.no_video,
            },
        )
        ready, ready_arrays = receive_message(connection)
        _raise_peer_error(ready, peer="inference")
        if ready_arrays or ready.get("type") != "ready":
            raise RuntimeError("inference peer did not send a valid ready handshake")
        if ready.get("accepted_contract") != contract:
            raise RuntimeError("inference peer rejected or altered the environment contract")
        client_metadata = _validate_client_metadata(ready.get("client"))

        for episode_index, seed in enumerate(seeds):
            current_episode = (episode_index, seed)
            if episode_index == 0:
                observation = pending_observation
                state, rgb = first_state, first_rgb
            else:
                observation, _ = _seeded_reset(env, seed)
                state, rgb = extract_liftbarrier_observation(observation)
            episode_started = time.perf_counter()
            success = False
            terminated = False
            truncated = False
            step = 0
            inference_latencies: list[float] = []
            round_trip_latencies: list[float] = []
            episode_action_min = np.full(16, np.inf, dtype=np.float32)
            episode_action_max = np.full(16, -np.inf, dtype=np.float32)
            stop_reason = "max_steps"

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
                        "task": {
                            "id": "lift_barrier",
                            "text": "Lift the barrier together",
                        },
                        "image_frame_index": step,
                    },
                    {"proprioception": state, "rgb_global": rgb},
                )
                action_message, action_arrays = receive_message(connection)
                _raise_peer_error(action_message, peer="inference")
                round_trip_latencies.append(
                    (time.perf_counter() - request_started) * 1000.0
                )
                _validate_action_message(
                    action_message,
                    request_id=request_id,
                    episode_index=episode_index,
                    step=step,
                    action_codec_sha256=str(
                        client_metadata["action_codec_sha256"]
                    ),
                )
                if set(action_arrays) != {"action"}:
                    raise RuntimeError("inference response must contain exactly one action")
                action = action_arrays["action"]
                if (
                    action.shape != (16,)
                    or action.dtype != np.float32
                    or not np.isfinite(action).all()
                ):
                    raise RuntimeError("inference action must be finite float32[16]")
                split_action = split_liftbarrier_action(action)
                latency = float(action_message.get("inference_latency_ms", -1.0))
                if not np.isfinite(latency) or latency < 0.0:
                    raise RuntimeError("inference response has invalid latency metadata")
                inference_latencies.append(latency)
                action_min = np.minimum(action_min, action)
                action_max = np.maximum(action_max, action)
                episode_action_min = np.minimum(episode_action_min, action)
                episode_action_max = np.maximum(episode_action_max, action)

                observation, _, raw_terminated, raw_truncated, info = env.step(split_action)
                step += 1
                terminated = scalar_bool(raw_terminated, name="terminated")
                truncated = scalar_bool(raw_truncated, name="truncated")
                now_success = _success_from_info(info)
                success = success or now_success
                if success:
                    stop_reason = "success"
                elif terminated:
                    stop_reason = "terminated"
                elif truncated:
                    stop_reason = "truncated"
                if success or terminated or truncated or step >= args.max_steps:
                    break
                state, rgb = extract_liftbarrier_observation(observation)

            video_path = _flush_episode_video(
                env,
                video_dir=video_dir,
                episode_index=episode_index,
                seed=seed,
                success=success,
                enabled=not args.no_video,
            )
            episode_result = {
                "format_version": EPISODE_FORMAT,
                "episode_index": episode_index,
                "seed": seed,
                "success": success,
                "steps": step,
                "terminated": terminated,
                "truncated": truncated,
                "stop_reason": stop_reason,
                "video": video_path,
                "duration_seconds": time.perf_counter() - episode_started,
                "inference_latency_ms": _latency_summary(inference_latencies),
                "rpc_round_trip_latency_ms": _latency_summary(round_trip_latencies),
                "action_min": episode_action_min.astype(float).tolist(),
                "action_max": episode_action_max.astype(float).tolist(),
            }
            results.append(episode_result)
            _append_jsonl(episodes_path, episode_result)
            send_message(
                connection,
                {
                    "type": "episode_result",
                    **episode_result,
                    "successes_so_far": sum(item["success"] for item in results),
                    "success_rate_so_far": sum(item["success"] for item in results)
                    / len(results),
                },
            )
            print(
                f"[environment] episode={episode_index + 1}/{args.episodes} "
                f"seed={seed} success={success} steps={step} "
                f"running_rate={sum(item['success'] for item in results) / len(results):.2%}",
                flush=True,
            )
            current_episode = None

        completed = True
        summary = _build_summary(
            args=args,
            config_path=config_path,
            output_dir=output_dir,
            results=results,
            client_metadata=client_metadata,
            robofactory_provenance=robofactory_provenance,
            environment_contract=environment_contract,
            action_min=action_min,
            action_max=action_max,
            started_at=started_at,
            elapsed_seconds=time.perf_counter() - started,
            completed=True,
            fatal_error=None,
        )
        _write_json_atomic(summary_path, summary)
        try:
            send_message(connection, {"type": "summary", "summary": summary})
        except OSError as exc:
            print(
                "[environment] warning: complete summary is safely on disk, but "
                f"the client notification failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
        print(
            f"[environment] complete: {summary['successes']}/{summary['episodes_completed']} "
            f"= {summary['success_rate']:.2%}",
            flush=True,
        )
        print(
            "[environment] formal benchmark reportable: "
            f"{summary['formal_benchmark']['reportable']}",
            flush=True,
        )
        print(f"[environment] summary: {summary_path}", flush=True)
        return 0
    except BaseException as exc:
        fatal_error = {"type": type(exc).__name__, "message": str(exc)}
        if connection is not None:
            try:
                send_message(
                    connection,
                    {"type": "error", "error": fatal_error, "fatal": True},
                )
            except BaseException:
                pass
        if env is not None and current_episode is not None and not args.no_video:
            try:
                _flush_aborted_video(
                    env,
                    video_dir=video_dir,
                    episode_index=current_episode[0],
                    seed=current_episode[1],
                )
            except BaseException:
                pass
        summary = _build_summary(
            args=args,
            config_path=config_path,
            output_dir=output_dir,
            results=results,
            client_metadata=client_metadata,
            robofactory_provenance=robofactory_provenance,
            environment_contract=environment_contract,
            action_min=action_min,
            action_max=action_max,
            started_at=started_at,
            elapsed_seconds=time.perf_counter() - started,
            completed=False,
            fatal_error=fatal_error,
        )
        _write_json_atomic(summary_path, summary)
        raise
    finally:
        if env is not None:
            if not completed:
                try:
                    env.flush_video(save=False)
                except BaseException:
                    pass
            env.close()
        if connection is not None:
            connection.close()
        if listener is not None:
            listener.close()


def _validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in [1,65535]")
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if not 1 <= args.max_steps <= 500:
        raise ValueError("--max-steps must be in [1,500]")
    if args.seed_start < 0:
        raise ValueError("--seed-start must be non-negative")
    if args.seed_start + args.episodes - 1 > np.iinfo(np.uint32).max:
        raise ValueError("requested seed schedule exceeds NumPy's uint32 seed range")
    if (
        args.video_fps <= 0
        or not math.isfinite(args.socket_timeout)
        or args.socket_timeout <= 0.0
    ):
        raise ValueError("--video-fps and --socket-timeout must be positive")
    if not args.allow_remote and args.host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError(
            "refusing unauthenticated non-loopback bind; pass --allow-remote explicitly"
        )


def _prepare_output_directory(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"rollout output is not a directory: {path}")
    if path.exists() and any(value.is_file() for value in path.rglob("*")):
        raise FileExistsError(
            f"rollout output already contains files: {path}; choose a fresh directory"
        )
    path.mkdir(parents=True, exist_ok=True)


def _listen(host: str, port: int) -> socket.socket:
    listener = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(1)
    return listener


def _make_environment(
    *,
    robofactory_root: Path,
    config_path: Path,
    sim_backend: str,
    shader: str,
    video_dir: Path,
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
        raise RuntimeError(
            "RoboFactory/ManiSkill imports failed; run this server in the "
            "RoboFactory Python 3.9 environment"
        ) from exc

    env = gym.make(
        "LiftBarrier-rf",
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
        output_dir=str(video_dir),
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
        source_type="m1_closed_loop",
        source_desc="scratch M1 closed-loop policy evaluation",
    )


def _inspect_environment_limits(env: Any) -> Tuple[float, int]:
    environment_max_steps = _wrapper_episode_limit(env)
    base_env = env.unwrapped
    control_hz = float(base_env.control_freq)
    if environment_max_steps != 500:
        raise RuntimeError(
            f"LiftBarrier TimeLimit drifted: expected 500, got {environment_max_steps}"
        )
    if control_hz != 20.0:
        raise RuntimeError(
            f"LiftBarrier control frequency drifted: expected 20, got {control_hz}"
        )
    return control_hz, int(environment_max_steps)


def _wrapper_episode_limit(env: Any) -> int:
    """Read TimeLimit evidence without deprecated wrapper attribute forwarding."""

    current = env
    visited: set[int] = set()
    observed: list[Tuple[str, str, int]] = []
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        namespace = vars(current)
        for field in ("max_episode_steps", "_max_episode_steps"):
            if field not in namespace or namespace[field] is None:
                continue
            raw_value = namespace[field]
            if isinstance(raw_value, bool) or not isinstance(
                raw_value, (int, np.integer)
            ):
                raise RuntimeError(
                    f"environment wrapper {type(current).__name__}.{field} "
                    f"is not an integer: {raw_value!r}"
                )
            observed.append((type(current).__name__, field, int(raw_value)))
        current = namespace.get("env")
    if not observed:
        raise RuntimeError("RoboFactory wrapper chain has no episode TimeLimit")
    unique = {value for _, _, value in observed}
    if len(unique) != 1:
        raise RuntimeError(f"RoboFactory wrapper TimeLimits disagree: {observed}")
    return unique.pop()


def _environment_contract(
    state: np.ndarray,
    rgb: np.ndarray,
    max_steps: int,
    *,
    control_hz: float = 20.0,
    environment_max_steps: int = 500,
) -> dict[str, Any]:
    if (
        state.shape != (36,)
        or state.dtype != np.float32
        or not np.isfinite(state).all()
        or rgb.shape != (240, 320, 3)
        or rgb.dtype != np.uint8
    ):
        raise RuntimeError("RoboFactory reset observation violates the M1 contract")
    return {
        "environment_id": "LiftBarrier-rf",
        "task_id": "lift_barrier",
        "task_text": "Lift the barrier together",
        "state_dim": 36,
        "state_order": [
            "panda-0.qpos[9]",
            "panda-0.qvel[9]",
            "panda-1.qpos[9]",
            "panda-1.qvel[9]",
        ],
        "action_dim": 16,
        "agent_order": ["panda-0", "panda-1"],
        "control_mode": "pd_joint_pos",
        "control_hz": float(control_hz),
        "camera_order": ["global"],
        "camera_source": "head_camera_global",
        "camera_shape": list(rgb.shape),
        "camera_dtype": str(rgb.dtype),
        "rgb_encoding": "raw_lossless",
        "environment_max_episode_steps": int(environment_max_steps),
        "rollout_max_steps": int(max_steps),
        "success_source": "info.success",
    }


def _validate_client_metadata(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise RuntimeError("ready.client must be an object")
    value = dict(raw)
    required = {
        "checkpoint",
        "checkpoint_tree_sha256",
        "checkpoint_format",
        "config_sha256",
        "train_seed",
        "vision_identity",
        "vision_runtime",
        "action_codec_sha256",
        "action_anchor_mode",
        "device",
        "provenance",
        "policy",
    }
    if set(value) != required:
        raise RuntimeError(
            f"ready.client fields differ from the required contract: {sorted(value)}"
        )
    if value["checkpoint_format"] != "wam.multimodal.m1.scratch_checkpoint/1":
        raise RuntimeError("client checkpoint is not a scratch M1 checkpoint")
    if value["config_sha256"] != FORMAL_LIFTBARRIER_M1_CONFIG_SHA256:
        raise RuntimeError("client config is not the formal LiftBarrier M1 config")
    if value["action_anchor_mode"] != "none":
        raise RuntimeError("client checkpoint unexpectedly uses an action anchor")
    for key in ("checkpoint_tree_sha256", "action_codec_sha256"):
        digest = value.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError(f"client {key} is not a lowercase SHA-256")
    if not isinstance(value.get("train_seed"), int) or value["train_seed"] < 0:
        raise RuntimeError("client checkpoint train_seed is invalid")
    if not isinstance(value.get("checkpoint"), str) or not value["checkpoint"]:
        raise RuntimeError("client checkpoint path is invalid")
    if not isinstance(value.get("device"), str) or not value["device"]:
        raise RuntimeError("client device identity is invalid")
    _validate_client_provenance(value.get("provenance"))
    vision = value.get("vision_identity")
    expected_vision = {
        "family": "FrozenDINOv3Encoder",
        "output_dim": 1024,
        "frozen": True,
    }
    if not isinstance(vision, Mapping) or any(
        vision.get(key) != expected
        for key, expected in expected_vision.items()
    ):
        raise RuntimeError("client did not prove a frozen visual encoder")
    for key in ("artifact_sha256", "config_sha256"):
        digest = vision.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError(f"client vision {key} is invalid")
    vision_runtime = value.get("vision_runtime")
    expected_vision_runtime = {
        "encoder_name": "dinov3_vitl16_lvd",
        "model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "revision": "dd0a398fa8e84f2a37179332f6c561d20276300b",
        "expected_config_sha256": (
            "ce962b0c8ca4f2deb48c6fdfd6035257e3769f1d4d9154c92aba51991e46e290"
        ),
        "expected_weights_sha256": (
            "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
        ),
        "preprocess_id": "dinov3_imagenet_rgb_resize_square_antialias_v1",
        "input_size": 256,
        "inference_batch_size": 16,
        "frozen": True,
    }
    if vision_runtime != expected_vision_runtime:
        raise RuntimeError("client DINO runtime contract is invalid")
    policy = value.get("policy")
    expected_policy = {
        "camera_order": ["global"],
        "visual_history_frames": 2,
        "action_horizon": 8,
        "execution_steps": 2,
        "solver_steps": 4,
        "solver": "euler",
        "normalized_action_clip": 10.0,
        "replan_on_new_image": False,
        "warm_start": True,
        "cold_start_history": "masked_zero_padding_no_action/1",
    }
    if not isinstance(policy, Mapping) or any(
        policy.get(key) != expected
        for key, expected in expected_policy.items()
    ):
        raise RuntimeError("client policy runtime contract is invalid")
    return value


def _validate_client_provenance(raw: Any) -> None:
    if not isinstance(raw, Mapping) or set(raw) != {
        "repository_root",
        "source_sha256",
        "git",
        "runtime",
    }:
        raise RuntimeError("client WAM provenance is incomplete")
    if not isinstance(raw.get("repository_root"), str) or not raw["repository_root"]:
        raise RuntimeError("client WAM repository identity is invalid")
    sources = raw.get("source_sha256")
    required_sources = {
        "robofactory_rpc.py",
        "scripts/run_robofactory_m1_inference.py",
        "models/wam/action_codec.py",
        "models/wam/config.py",
        "models/wam/heads.py",
        "models/wam/stateful_action_flow.py",
        "models/wam_multimodal/latent_wam.py",
        "models/wam_multimodal/latent_world_head.py",
        "models/wam_multimodal/token_resampler.py",
        "models/wam_multimodal/vision_encoder.py",
        "policies/scratch_m1.py",
        "train/m1_scratch_builder.py",
        "train/m1_scratch_checkpointing.py",
    }
    if not isinstance(sources, Mapping) or set(sources) != required_sources:
        raise RuntimeError("client WAM source hashes are missing")
    for name, digest in sources.items():
        if not isinstance(name, str) or not _is_sha256(digest):
            raise RuntimeError("client WAM source hashes are invalid")
    git = raw.get("git")
    if not isinstance(git, Mapping) or not isinstance(git.get("available"), bool):
        raise RuntimeError("client WAM Git provenance is invalid")
    if git["available"] and (
        not isinstance(git.get("commit"), str)
        or len(git["commit"]) != 40
        or not _is_sha256(git.get("tracked_diff_sha256"))
        or not isinstance(git.get("dirty"), bool)
        or not isinstance(git.get("status_porcelain"), list)
    ):
        raise RuntimeError("client WAM Git provenance is incomplete")
    runtime = raw.get("runtime")
    required_runtime = {
        "python",
        "numpy",
        "torch",
        "torch_cuda",
        "cuda_available",
        "cuda_device_name",
        "transformers",
        "safetensors",
        "pyyaml",
    }
    if not isinstance(runtime, Mapping) or set(runtime) != required_runtime:
        raise RuntimeError("client WAM runtime provenance is invalid")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _raise_peer_error(message: Mapping[str, Any], *, peer: str) -> None:
    if message.get("type") == "error":
        error = message.get("error")
        raise RuntimeError(f"{peer} failed: {error}")


def _validate_action_message(
    message: Mapping[str, Any],
    *,
    request_id: str,
    episode_index: int,
    step: int,
    action_codec_sha256: str,
) -> None:
    expected = {
        "type": "action",
        "request_id": request_id,
        "episode_index": episode_index,
        "step": step,
    }
    mismatched = {
        key: {"expected": value, "observed": message.get(key)}
        for key, value in expected.items()
        if message.get(key) != value
    }
    if mismatched:
        raise RuntimeError(f"stale/out-of-order action response: {mismatched}")
    diagnostics = message.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise RuntimeError("action response lacks policy diagnostics")
    expected_diagnostics = {
        "action_source": "m1_scratch_latent_flow",
        "initialization_mode": "scratch",
        "action_anchor_mode": "none",
        "legacy_bypass_used": False,
        "fallback_used": False,
        "privileged_state_seen": False,
        "action_dim": 16,
        "action_codec_sha256": action_codec_sha256,
    }
    diagnostic_mismatches = {
        key: {"expected": value, "observed": diagnostics.get(key)}
        for key, value in expected_diagnostics.items()
        if diagnostics.get(key) != value
    }
    if diagnostic_mismatches:
        raise RuntimeError(
            f"M1 policy diagnostics violate the rollout contract: "
            f"{diagnostic_mismatches}"
        )


def _success_from_info(info: Any) -> bool:
    if not isinstance(info, Mapping) or "success" not in info:
        raise RuntimeError("RoboFactory step info lacks the success label")
    return scalar_bool(info["success"], name="info.success")


def _flush_episode_video(
    env: Any,
    *,
    video_dir: Path,
    episode_index: int,
    seed: int,
    success: bool,
    enabled: bool,
) -> Optional[str]:
    if not enabled:
        return None
    status = "success" if success else "failure"
    name = f"episode_{episode_index:04d}_seed_{seed}_{status}"
    target = video_dir / f"{name}.mp4"
    if target.exists():
        raise FileExistsError(f"refusing to replace rollout video: {target}")
    env.flush_video(name=name, verbose=False)
    if not target.is_file() or target.stat().st_size <= 0:
        raise RuntimeError(f"RoboFactory did not produce the expected MP4: {target}")
    return str(target.relative_to(video_dir.parent))


def _flush_aborted_video(
    env: Any,
    *,
    video_dir: Path,
    episode_index: int,
    seed: int,
) -> None:
    if not getattr(env, "render_images", None):
        return
    name = f"episode_{episode_index:04d}_seed_{seed}_aborted"
    target = video_dir / f"{name}.mp4"
    if target.exists():
        raise FileExistsError(f"refusing to replace aborted rollout video: {target}")
    env.flush_video(name=name, verbose=False, ignore_empty_transition=False)


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("latency evidence is empty or non-finite")
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _robofactory_provenance(
    robofactory_root: Path,
    *,
    config_path: Path,
) -> dict[str, Any]:
    source_paths = (
        "robofactory/tasks/lift_barrier.py",
        "robofactory/utils/scenes/scene_builder.py",
        "robofactory/utils/wrappers/record.py",
    )
    sources = {
        name: _sha256(robofactory_root / name)
        for name in source_paths
    }
    return {
        "repository_root": str(robofactory_root),
        "config_sha256": _sha256(config_path),
        "source_sha256": sources,
        "git": _git_provenance(robofactory_root),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": _package_version("torch"),
            "gymnasium": _package_version("gymnasium"),
            "mani_skill": _package_version("mani-skill"),
            "sapien": _package_version("sapien"),
        },
    }


def _git_provenance(repository: Path) -> dict[str, Any]:
    try:
        commit = _git_output(repository, "rev-parse", "HEAD").decode().strip()
        status = _git_output(
            repository,
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
        ).decode("utf-8", errors="replace").splitlines()
        diff = _git_output(repository, "diff", "--binary", "HEAD")
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": True,
        "commit": commit,
        "dirty": bool(status),
        "status_porcelain": status,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _git_output(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _package_version(distribution: str) -> Optional[str]:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _build_summary(
    *,
    args: argparse.Namespace,
    config_path: Path,
    output_dir: Path,
    results: Sequence[Mapping[str, Any]],
    client_metadata: Optional[Mapping[str, Any]],
    robofactory_provenance: Optional[Mapping[str, Any]],
    environment_contract: Optional[Mapping[str, Any]],
    action_min: np.ndarray,
    action_max: np.ndarray,
    started_at: str,
    elapsed_seconds: float,
    completed: bool,
    fatal_error: Optional[Mapping[str, str]],
) -> dict[str, Any]:
    count = len(results)
    successes = sum(bool(value["success"]) for value in results)
    interval = wilson_interval(successes, count) if count else None
    has_actions = bool(np.isfinite(action_min).all() and np.isfinite(action_max).all())
    formal_benchmark = _formal_benchmark_status(
        args=args,
        results=results,
        completed=completed,
        fatal_error=fatal_error,
        client_metadata=client_metadata,
        robofactory_provenance=robofactory_provenance,
        environment_contract=environment_contract,
    )
    return {
        "format_version": SUMMARY_FORMAT,
        "protocol_version": PROTOCOL_VERSION,
        "completed": completed,
        "fatal_error": dict(fatal_error) if fatal_error is not None else None,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "elapsed_seconds": elapsed_seconds,
        "environment": {
            "id": "LiftBarrier-rf",
            "config": str(config_path),
            "sim_backend": args.sim_backend,
            "shader": args.shader,
            "control_mode": "pd_joint_pos",
            "control_hz": (
                environment_contract.get("control_hz")
                if environment_contract is not None
                else None
            ),
            "environment_max_episode_steps": (
                environment_contract.get("environment_max_episode_steps")
                if environment_contract is not None
                else None
            ),
            "rollout_max_steps": (
                environment_contract.get("rollout_max_steps")
                if environment_contract is not None
                else args.max_steps
            ),
            "contract": (
                dict(environment_contract)
                if environment_contract is not None
                else None
            ),
            "provenance": (
                dict(robofactory_provenance)
                if robofactory_provenance is not None
                else None
            ),
        },
        "benchmark_protocol": {
            "episodes_requested": args.episodes,
            "seed_start": args.seed_start,
            "seeds_requested": list(
                range(args.seed_start, args.seed_start + args.episodes)
            ),
            "stop_on_success": True,
            "success_aggregation": "episode_any_info.success",
            "seed_application": "numpy_global_then_env.reset(seed)/1",
            "video_enabled": not args.no_video,
            "video_fps": args.video_fps if not args.no_video else None,
        },
        "formal_benchmark": formal_benchmark,
        "client": dict(client_metadata) if client_metadata is not None else None,
        "limitations": [
            {
                "id": "reset_history_distribution_gap",
                "detail": (
                    "At reset the online policy has one RGB/state observation and no "
                    "previous action; missing history is mask-aware zero padding. The "
                    "training windows did not include this exact single-frame reset case."
                ),
                "policy_behavior": "masked_zero_padding_no_action/1",
            }
        ],
        "episodes_completed": count,
        "successes": successes,
        "success_rate": successes / count if count else None,
        "success_rate_wilson_95": list(interval) if interval is not None else None,
        "mean_episode_steps": (
            float(np.mean([int(value["steps"]) for value in results]))
            if count
            else None
        ),
        "observed_action_min": action_min.astype(float).tolist() if has_actions else None,
        "observed_action_max": action_max.astype(float).tolist() if has_actions else None,
        "output_dir": str(output_dir),
        "episodes_file": "rollout_episodes.jsonl",
        "episodes": [dict(value) for value in results],
    }


def _formal_benchmark_status(
    *,
    args: argparse.Namespace,
    results: Sequence[Mapping[str, Any]],
    completed: bool,
    fatal_error: Optional[Mapping[str, str]],
    client_metadata: Optional[Mapping[str, Any]],
    robofactory_provenance: Optional[Mapping[str, Any]],
    environment_contract: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_seeds = list(range(1000, 1100))
    violations: list[str] = []
    checks = (
        (args.episodes == 100, "episodes_must_equal_100"),
        (args.seed_start == 1000, "seed_start_must_equal_1000"),
        (args.max_steps == 500, "max_steps_must_equal_500"),
        (args.sim_backend == "cpu", "sim_backend_must_equal_cpu"),
        (args.shader == "default", "shader_must_equal_default"),
        (not args.no_video, "video_must_be_enabled"),
        (args.video_fps == 20, "video_fps_must_equal_20"),
        (completed, "run_must_be_completed"),
        (fatal_error is None, "fatal_error_must_be_null"),
        (len(results) == 100, "episodes_completed_must_equal_100"),
        (client_metadata is not None, "validated_client_metadata_is_required"),
        (
            robofactory_provenance is not None,
            "robofactory_provenance_is_required",
        ),
        (
            environment_contract is not None,
            "validated_environment_contract_is_required",
        ),
    )
    violations.extend(reason for passed, reason in checks if not passed)
    observed_seeds = [value.get("seed") for value in results]
    if observed_seeds != expected_seeds:
        violations.append("completed_seed_schedule_must_equal_1000_through_1099")
    if len(results) != 100 or any(
        not isinstance(value.get("video"), str)
        or not str(value["video"]).endswith(".mp4")
        for value in results
    ):
        violations.append("every_episode_must_have_a_verified_mp4")
    observed_env_config_sha256 = (
        robofactory_provenance.get("config_sha256")
        if robofactory_provenance is not None
        else None
    )
    if observed_env_config_sha256 != FORMAL_LIFTBARRIER_ENV_CONFIG_SHA256:
        violations.append("environment_config_sha256_mismatch")
    observed_sources = (
        robofactory_provenance.get("source_sha256")
        if robofactory_provenance is not None
        else None
    )
    if observed_sources != FORMAL_LIFTBARRIER_RF_SOURCE_SHA256:
        violations.append("robofactory_source_sha256_mismatch")
    if environment_contract is None or (
        environment_contract.get("control_hz") != 20.0
        or environment_contract.get("environment_max_episode_steps") != 500
        or environment_contract.get("rollout_max_steps") != 500
    ):
        violations.append("observed_environment_limits_mismatch")
    return {
        "protocol_id": FORMAL_BENCHMARK_PROTOCOL,
        "reportable": not violations,
        "violations": violations,
        "requirements": {
            "episodes": 100,
            "seeds": expected_seeds,
            "max_steps": 500,
            "sim_backend": "cpu",
            "shader": "default",
            "video_enabled": True,
            "video_fps": 20,
            "environment_config_sha256": FORMAL_LIFTBARRIER_ENV_CONFIG_SHA256,
            "robofactory_source_sha256": FORMAL_LIFTBARRIER_RF_SOURCE_SHA256,
        },
    }


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _seeded_reset(env: Any, seed: int) -> Any:
    """Seed both RoboFactory's global NumPy randomizer and ManiSkill reset."""

    np.random.seed(int(seed))
    return env.reset(seed=int(seed))


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
