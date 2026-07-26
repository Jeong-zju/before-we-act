#!/usr/bin/env python3
"""Load a strict M2 checkpoint and answer RoboFactory rollout requests."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import socket
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.wam_multimodal import FrozenDINOv3Config, FrozenDINOv3Encoder  # noqa: E402
from policies.robofactory_m2 import (  # noqa: E402
    RoboFactoryM2Policy,
    RoboFactoryM2PolicyConfig,
)
from robofactory_rpc import configure_socket, receive_message, send_message  # noqa: E402
from train.m2_checkpointing import (  # noqa: E402
    load_m2_checkpoint,
    m2_checkpoint_tree_sha256,
)
from train.m2_training import RGBStatisticsVisionEncoder  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/wam_multimodal/m2_causal_wam.yaml",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8872)
    parser.add_argument("--connect-timeout", type=float, default=600.0)
    parser.add_argument("--socket-timeout", type=float, default=600.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in [1,65535]")
    if args.connect_timeout <= 0.0 or args.socket_timeout <= 0.0:
        raise ValueError("M2 inference timeouts must be positive")
    config = _load_yaml(args.config.expanduser().resolve(strict=True))
    device = _device(args.device)
    precision = _precision(args.precision, device=device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model, runtime, schema = load_m2_checkpoint(
        args.checkpoint.expanduser().resolve(strict=True), device=device
    )
    vision = _build_vision(config, schema=schema).to(device).eval()
    connection: socket.socket | None = None
    active_request: Mapping[str, Any] | None = None
    try:
        connection = _connect(args.host, args.port, timeout=args.connect_timeout)
        configure_socket(connection, timeout_seconds=args.socket_timeout)
        hello, arrays = receive_message(connection)
        _raise_peer_error(hello)
        if arrays or hello.get("type") != "hello":
            raise RuntimeError("RoboFactory M2 server did not send a valid hello")
        contract = _mapping(hello, "contract")
        data_config = _mapping(config, "data")
        expected_image_shape = (
            (
                int(data_config["image_height"]),
                int(data_config["image_width"]),
                3,
            )
            if "image_height" in data_config and "image_width" in data_config
            else None
        )
        task = _validate_contract(
            contract,
            runtime=runtime,
            expected_image_shape=expected_image_shape,
        )
        generation = _checkpoint_action_generation(config, schema=schema)
        policy_config = RoboFactoryM2PolicyConfig(
            camera_order=tuple(contract["camera_order"]),
            execution_steps=int(generation["execution_steps"]),
            solver_steps=int(generation["solver_steps"]),
            solver=str(generation["solver"]),
            normalized_action_clip=float(generation["normalized_action_clip"]),
            warm_start=bool(generation["warm_start"]),
            future_path=bool(contract["future_path"]),
        )
        policy = RoboFactoryM2Policy(
            model,
            vision,
            runtime,
            policy_config,
            device=device,
        )
        client = {
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "checkpoint_tree_sha256": m2_checkpoint_tree_sha256(args.checkpoint),
            "checkpoint_format": schema["format_version"],
            "task_vocabulary": list(schema["task_vocabulary"]),
            "trainable_parameters": int(schema["trainable_parameters"]),
            "vision_identity": dict(schema["vision_identity"]),
            "device": str(device),
            "precision": precision,
            "future_path": bool(contract["future_path"]),
            "policy": {
                "execution_steps": policy_config.execution_steps,
                "solver_steps": policy_config.solver_steps,
                "solver": policy_config.solver,
                "normalized_action_clip": policy_config.normalized_action_clip,
                "warm_start": policy_config.warm_start,
                "action_source": (
                    "m2_block_causal_future_path"
                    if policy_config.future_path
                    else "m2_block_causal_fast_path"
                ),
            },
        }
        send_message(
            connection,
            {"type": "ready", "accepted_contract": dict(contract), "client": client},
        )
        expected_episode = 0
        expected_step = 0
        while True:
            message, values = receive_message(connection)
            message_type = message.get("type")
            if message_type == "observation":
                active_request = message
                episode = int(message.get("episode_index", -1))
                step = int(message.get("step", -1))
                if episode != expected_episode or step != expected_step:
                    raise RuntimeError("M2 observation schedule is out of order")
                if bool(message.get("reset")):
                    if step != 0:
                        raise RuntimeError("M2 reset observation must be step zero")
                    policy.reset()
                camera_order = tuple(map(str, contract["camera_order"]))
                expected_arrays = {
                    "proprioception",
                    *{_rgb_array_name(camera) for camera in camera_order},
                }
                if set(values) != expected_arrays:
                    raise RuntimeError("M2 observation arrays differ from the contract")
                task_message = _mapping(message, "task")
                if (
                    str(task_message.get("id")) != contract["task_id"]
                    or str(task_message.get("text")) != contract["task_text"]
                ):
                    raise RuntimeError("M2 task condition drifted during rollout")
                observation = {
                    "task": {
                        "id": str(task_message["id"]),
                        "text": str(task_message["text"]),
                    },
                    "proprioception": values["proprioception"],
                    "images": {
                        camera: values[_rgb_array_name(camera)]
                        for camera in camera_order
                    },
                    "image_frame_indices": {
                        camera: int(message["image_frame_index"])
                        for camera in camera_order
                    },
                }
                inference_started = time.perf_counter()
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=precision == "bf16",
                ):
                    action = policy.act(observation)
                latency = (time.perf_counter() - inference_started) * 1000.0
                send_message(
                    connection,
                    {
                        "type": "action",
                        "request_id": message["request_id"],
                        "episode_index": episode,
                        "step": step,
                        "inference_latency_ms": latency,
                        "diagnostics": _plain(policy.last_diagnostics),
                    },
                    {"action": action},
                )
                expected_step += 1
                active_request = None
            elif message_type == "episode_result":
                if values or int(message.get("episode_index", -1)) != expected_episode:
                    raise RuntimeError("M2 episode result identity is invalid")
                print(
                    f"[inference] task={task['task_id']} episode={expected_episode + 1}/"
                    f"{hello['episodes']} success={message['success']} steps={message['steps']}",
                    flush=True,
                )
                expected_episode += 1
                expected_step = 0
            elif message_type == "summary":
                if values:
                    raise RuntimeError("M2 summary unexpectedly contains arrays")
                summary = _mapping(message, "summary")
                if summary.get("completed") is not True:
                    raise RuntimeError("M2 server returned an incomplete summary")
                print(
                    f"[inference] complete {summary['successes']}/"
                    f"{summary['episodes_completed']} = {summary['success_rate']:.2%}",
                    flush=True,
                )
                print(f"[inference] outputs: {summary['output_dir']}", flush=True)
                return 0
            elif message_type == "error":
                raise RuntimeError(f"M2 environment failed: {message.get('error')}")
            else:
                raise RuntimeError(f"unexpected M2 message type {message_type!r}")
    except BaseException as exc:
        if connection is not None:
            try:
                send_message(
                    connection,
                    {
                        "type": "error",
                        "fatal": True,
                        "request_id": (
                            None if active_request is None else active_request.get("request_id")
                        ),
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    },
                )
            except BaseException:
                pass
        raise
    finally:
        if connection is not None:
            connection.close()


def _build_vision(config: Mapping[str, Any], *, schema: Mapping[str, Any]) -> torch.nn.Module:
    identity = _mapping(schema, "vision_identity")
    family = str(identity.get("family"))
    if family == "rgb_statistics_smoke":
        if schema.get("training", {}).get("smoke") is not True:
            raise RuntimeError("RGB statistics vision is allowed only in smoke checkpoints")
        return RGBStatisticsVisionEncoder(int(identity["output_dim"]))
    if family != "FrozenDINOv3Encoder":
        raise ValueError(f"unsupported M2 vision family {family!r}")
    vision = _mapping(config, "vision")
    rectangular = "input_height" in vision or "input_width" in vision
    encoder = FrozenDINOv3Encoder(
        FrozenDINOv3Config(
            encoder_name=str(vision["encoder_name"]),
            model_id=str(vision["model_id"]),
            revision=str(vision["revision"]),
            config_path=(ROOT / str(vision["config_path"])).resolve(strict=True),
            weights_path=(ROOT / str(vision["weights_path"])).resolve(strict=True),
            expected_config_sha256=str(vision["expected_config_sha256"]),
            expected_weights_sha256=str(vision["expected_weights_sha256"]),
            preprocess_id=str(vision["preprocess_id"]),
            input_size=(
                None
                if rectangular
                else int(vision.get("input_size", 256))
            ),
            input_height=(
                int(vision["input_height"]) if rectangular else None
            ),
            input_width=(
                int(vision["input_width"]) if rectangular else None
            ),
            inference_batch_size=int(vision.get("inference_batch_size", 64)),
        )
    )
    if (
        encoder.artifact_sha256 != identity.get("artifact_sha256")
        or encoder.config_sha256 != identity.get("config_sha256")
        or encoder.output_dim != int(identity.get("output_dim", -1))
        or encoder.config.image_height != int(identity.get("input_height", -1))
        or encoder.config.image_width != int(identity.get("input_width", -1))
    ):
        raise ValueError("runtime DINO identity differs from the M2 checkpoint")
    return encoder


def _validate_contract(
    contract: Mapping[str, Any],
    *,
    runtime: Sequence[Mapping[str, Any]],
    expected_image_shape: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    by_task = {str(value["task_id"]): dict(value) for value in runtime}
    task_id = str(contract.get("task_id", ""))
    if task_id not in by_task:
        raise ValueError(f"checkpoint does not support RoboFactory task {task_id!r}")
    task = by_task[task_id]
    codec = _mapping(task, "action_codec")
    metadata = _mapping(codec, "metadata")
    expected = {
        "task_text": str(task["task_text"]),
        "state_dim": int(task["state_dim"]),
        "action_dim": int(task["action_dim"]),
        "agent_count": int(task["agent_count"]),
        "agent_order": list(metadata["agent_order"]),
        "camera_order": list(task["camera_order"]),
        "control_mode": "pd_joint_pos",
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"RoboFactory M2 contract mismatch for {key}")
    camera_order = expected["camera_order"]
    camera_sources = contract.get("camera_sources")
    camera_shapes = contract.get("camera_shapes")
    if not isinstance(camera_sources, Mapping) or not isinstance(
        camera_shapes, Mapping
    ):
        raise ValueError("RoboFactory M2 multi-camera contract is invalid")
    # JSON metadata is deliberately serialized with sort_keys=True, so mapping
    # insertion order is not a wire-level property.  The explicit camera_order
    # field is authoritative; nested mappings only need the same logical keys.
    expected_cameras = set(camera_order)
    if (
        set(camera_sources) != expected_cameras
        or set(camera_shapes) != expected_cameras
    ):
        raise ValueError("RoboFactory M2 multi-camera contract keys are invalid")
    if any(
        not isinstance(camera_sources[camera], str)
        or not camera_sources[camera]
        or not isinstance(camera_shapes[camera], list)
        or len(camera_shapes[camera]) != 3
        or any(
            not isinstance(dimension, int) or dimension <= 0
            for dimension in camera_shapes[camera]
        )
        or camera_shapes[camera][-1] != 3
        for camera in camera_order
    ):
        raise ValueError("RoboFactory M2 multi-camera contract values are invalid")
    if expected_image_shape is not None and any(
        tuple(camera_shapes[camera]) != expected_image_shape
        for camera in camera_order
    ):
        raise ValueError(
            "RoboFactory M2 native camera resolution differs from the "
            f"training contract {expected_image_shape}"
        )
    return task


def _rgb_array_name(camera: str) -> str:
    if not camera or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
        for character in camera
    ):
        raise ValueError(f"invalid logical camera name {camera!r}")
    return f"rgb_{camera}"


def _checkpoint_action_generation(
    config: Mapping[str, Any], *, schema: Mapping[str, Any]
) -> dict[str, Any]:
    configured = dict(_mapping(config, "action_generation"))
    checkpoint = dict(_mapping(schema, "action_generation"))
    keys = (
        "solver_steps",
        "solver",
        "normalized_action_clip",
        "execution_steps",
        "warm_start",
    )
    normalized_config = {
        "solver_steps": int(configured.get("solver_steps", 0)),
        "solver": str(configured.get("solver", "")),
        "normalized_action_clip": float(
            configured.get("normalized_action_clip", 0.0)
        ),
        "execution_steps": int(configured.get("execution_steps", 0)),
        "warm_start": configured.get("warm_start"),
    }
    normalized_checkpoint = {
        "solver_steps": int(checkpoint.get("solver_steps", 0)),
        "solver": str(checkpoint.get("solver", "")),
        "normalized_action_clip": float(
            checkpoint.get("normalized_action_clip", 0.0)
        ),
        "execution_steps": int(checkpoint.get("execution_steps", 0)),
        "warm_start": checkpoint.get("warm_start"),
    }
    if set(configured) != set(keys) or normalized_config != normalized_checkpoint:
        raise ValueError(
            "runtime action_generation differs from the checkpoint training contract"
        )
    return normalized_checkpoint


def _connect(host: str, port: int, *, timeout: float) -> socket.socket:
    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            return socket.create_connection((host, port), timeout=min(5.0, timeout))
        except OSError as exc:
            last_error = exc
            time.sleep(0.25)
    raise TimeoutError(f"could not connect to M2 server: {last_error}")


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    return device


def _precision(value: str, *, device: torch.device) -> str:
    if value == "auto":
        return (
            "bf16"
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else "fp32"
        )
    if value == "bf16" and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("M2 BF16 inference requires a CUDA GPU with native BF16")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("M2 config root must be an object")
    return value


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    selected = value.get(key)
    if not isinstance(selected, Mapping):
        raise ValueError(f"M2 field {key!r} must be an object")
    return selected


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("M2 diagnostics contain NaN/Inf")
    return value


def _raise_peer_error(message: Mapping[str, Any]) -> None:
    if message.get("type") == "error":
        raise RuntimeError(f"M2 peer failed: {message.get('error')}")


if __name__ == "__main__":
    raise SystemExit(main())
