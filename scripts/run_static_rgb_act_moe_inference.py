#!/usr/bin/env python3
"""Serve static-camera DINO+ACT+MoE actions to the RoboFactory M2 RPC loop."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
import socket
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.static_rgb_act import (  # noqa: E402
    StaticRGBMoEACT,
    StaticRGBMoEACTConfig,
    build_chunk_aggregator,
)
from models.wam import AffineActionCodec, AffineActionCodecConfig  # noqa: E402
from models.wam_multimodal import (  # noqa: E402
    FrozenDINOv3Config,
    FrozenDINOv3Encoder,
)
from robofactory_rpc import configure_socket, receive_message, send_message  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/static_act/lpd_static_dino_act_moe.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8872)
    parser.add_argument("--connect-timeout", type=float, default=600.0)
    parser.add_argument("--socket-timeout", type=float, default=600.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("static RGB ACT inference requires exactly one visible GPU")
    device = torch.device(args.device)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    config_path = args.config.expanduser().resolve(strict=True)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if saved.get("format_version") != (
        "wam.robofactory.static_rgb_act_moe.checkpoint/1"
    ):
        raise ValueError("checkpoint is not static RGB ACT+MoE")
    model_config = StaticRGBMoEACTConfig.from_dict(
        _mapping(saved, "model_config")
    )
    model = StaticRGBMoEACT(model_config).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    raw_config = _load_yaml(config_path)
    configured_model = StaticRGBMoEACTConfig.from_dict(
        _mapping(raw_config, "model")
    )
    if configured_model != model_config:
        raise ValueError("runtime config model does not match checkpoint model")
    vision = _vision(raw_config).to(device).eval()
    runtime = {
        str(value["task_id"]): dict(value)
        for value in saved["task_runtime"]
    }
    inference = _mapping(raw_config, "inference")
    aggregator = build_chunk_aggregator(
        mode=str(inference.get("chunk_aggregation", "temporal_ensemble")),
        horizon=model.config.horizon,
        decay=float(inference.get("temporal_ensemble_decay", 0.01)),
    )
    action_source = (
        "static_rgb_dino_act_moe"
        if model.config.decoder_kind == "sparse_moe"
        else "static_rgb_dino_act_dense"
    )
    connection: socket.socket | None = None
    try:
        connection = _connect(
            args.host, args.port, timeout=args.connect_timeout
        )
        configure_socket(connection, timeout_seconds=args.socket_timeout)
        hello, arrays = receive_message(connection)
        if arrays or hello.get("type") != "hello":
            raise RuntimeError("RoboFactory server did not send a valid hello")
        contract = _mapping(hello, "contract")
        task_id = str(contract.get("task_id", ""))
        if task_id not in runtime:
            raise ValueError(f"checkpoint does not contain task {task_id!r}")
        if bool(contract.get("future_path")):
            raise ValueError("static RGB ACT does not expose the WAM future path")
        task = runtime[task_id]
        agent_count = int(contract["agent_count"])
        if (
            agent_count != int(task["agent_count"])
            or int(contract["state_dim"]) != int(task["state_dim"])
            or int(contract["action_dim"]) != int(task["action_dim"])
        ):
            raise ValueError("environment and checkpoint task dimensions differ")
        expected_cameras = ("global",) + tuple(
            f"agent_{index}" for index in range(agent_count)
        )
        if tuple(contract["camera_order"]) != expected_cameras:
            raise ValueError("static RGB ACT requires existing global+agent RGB order")
        codec = AffineActionCodec(
            AffineActionCodecConfig.from_dict(task["action_codec"])
        ).to(device)
        state_mean = torch.tensor(task["state_mean"], device=device)
        state_std = torch.tensor(task["state_std"], device=device)
        action_mean = torch.tensor(task["action_mean"], device=device)
        action_std = torch.tensor(task["action_std"], device=device)
        client = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_format": saved["format_version"],
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "task_vocabulary": list(runtime),
            "train_seed": int(_mapping(saved, "training")["seed"]),
            "source": dict(_mapping(saved, "source")),
            "future_path": False,
            "device": str(device),
            "policy": {
                "action_source": action_source,
                "horizon": model.config.horizon,
                "decoder_kind": model.config.decoder_kind,
                "experts": (
                    model.config.experts
                    if model.config.decoder_kind == "sparse_moe"
                    else None
                ),
                "dense_ffn_dim": (
                    model.config.dense_ffn_dim
                    if model.config.decoder_kind == "dense"
                    else None
                ),
                "chunk_aggregation": aggregator.mode,
                "temporal_ensemble_decay": aggregator.decay,
                "camera_protocol": (
                    "existing world-fixed per-agent RGB; no wrist; no depth"
                ),
            },
        }
        send_message(
            connection,
            {
                "type": "ready",
                "accepted_contract": dict(contract),
                "client": client,
            },
        )
        expected_episode = 0
        expected_step = 0
        while True:
            message, values = receive_message(connection)
            message_type = message.get("type")
            if message_type == "observation":
                episode = int(message.get("episode_index", -1))
                step = int(message.get("step", -1))
                if episode != expected_episode or step != expected_step:
                    raise RuntimeError("static RGB ACT observation schedule drifted")
                if bool(message.get("reset")):
                    aggregator.reset()
                started = time.perf_counter()
                state = torch.as_tensor(
                    values["proprioception"], device=device, dtype=torch.float32
                )
                normalized_state = (
                    (state - state_mean[: state.numel()])
                    / state_std[: state.numel()]
                ).reshape(agent_count, 18)
                images = torch.stack(
                    [
                        torch.as_tensor(
                            values[_rgb_array_name(f"agent_{index}")],
                            device=device,
                        ).permute(2, 0, 1)
                        for index in range(agent_count)
                    ]
                )
                with torch.inference_mode():
                    vision_tokens = vision(images).spatial_tokens
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        chunks = model(
                            vision_tokens,
                            normalized_state,
                        )[0]
                aggregator.push(chunks.float())
                normalized_action = aggregator.current().flatten()
                canonical = (
                    normalized_action * action_std[: normalized_action.numel()]
                    + action_mean[: normalized_action.numel()]
                ).clamp(-1.0, 1.0)
                raw_action = codec.decode(canonical, clip=True)
                if not isinstance(raw_action, Tensor):
                    raise TypeError("action codec returned a non-tensor")
                action = (
                    raw_action.float().cpu().numpy().astype(np.float32, copy=False)
                )
                aggregator.advance()
                send_message(
                    connection,
                    {
                        "type": "action",
                        "request_id": message["request_id"],
                        "episode_index": episode,
                        "step": step,
                        "inference_latency_ms": (
                            time.perf_counter() - started
                        )
                        * 1000.0,
                        "diagnostics": {
                            "action_source": action_source,
                            "fallback_used": False,
                            "direct_model_action": True,
                            "task_id": task_id,
                            "horizon": model.config.horizon,
                            "decoder_kind": model.config.decoder_kind,
                            "chunk_aggregation": aggregator.mode,
                            "wrist_camera": False,
                            "depth": False,
                        },
                    },
                    {"action": action},
                )
                expected_step += 1
            elif message_type == "episode_result":
                if int(message.get("episode_index", -1)) != expected_episode:
                    raise RuntimeError("episode result identity drifted")
                expected_episode += 1
                expected_step = 0
            elif message_type == "summary":
                summary = _mapping(message, "summary")
                if summary.get("completed") is not True:
                    raise RuntimeError("RoboFactory returned an incomplete summary")
                print(
                    f"complete {summary['successes']}/"
                    f"{summary['episodes_completed']} = "
                    f"{summary['success_rate']:.2%}",
                    flush=True,
                )
                return 0
            elif message_type == "error":
                raise RuntimeError(f"RoboFactory failed: {message.get('error')}")
            else:
                raise RuntimeError(f"unexpected RPC message {message_type!r}")
    finally:
        if connection is not None:
            connection.close()


def _vision(config: Mapping[str, Any]) -> FrozenDINOv3Encoder:
    value = _mapping(config, "vision")
    return FrozenDINOv3Encoder(
        FrozenDINOv3Config(
            encoder_name=str(value["encoder_name"]),
            model_id=str(value["model_id"]),
            revision=str(value["revision"]),
            config_path=(ROOT / str(value["config_path"])).resolve(strict=True),
            weights_path=(ROOT / str(value["weights_path"])).resolve(strict=True),
            expected_config_sha256=str(value["expected_config_sha256"]),
            expected_weights_sha256=str(value["expected_weights_sha256"]),
            preprocess_id=str(value["preprocess_id"]),
            input_size=None,
            input_height=int(value["input_height"]),
            input_width=int(value["input_width"]),
            inference_batch_size=int(value.get("inference_batch_size", 2)),
        )
    )


def _rgb_array_name(camera: str) -> str:
    if not camera or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
        for character in camera
    ):
        raise ValueError(f"invalid logical camera name {camera!r}")
    return f"rgb_{camera}"


def _connect(host: str, port: int, *, timeout: float) -> socket.socket:
    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            return socket.create_connection((host, port), timeout=min(timeout, 5.0))
        except OSError as exc:
            last_error = exc
            time.sleep(0.25)
    raise TimeoutError(f"could not connect to RoboFactory: {last_error}")


def _mapping(value: Mapping[str, Any], key: str | None = None) -> Mapping[str, Any]:
    result: Any = value if key is None else value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"{key or 'value'} must be a mapping")
    return result


def _load_yaml(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
