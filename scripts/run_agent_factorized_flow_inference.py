#!/usr/bin/env python3
"""Serve S1-R1 per-agent cold Rectified Flow to the RoboFactory RPC loop."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import socket
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.static_rgb_act import (  # noqa: E402
    StaticRGBMoEACTConfig,
    build_chunk_aggregator,
)
from models.wam import AffineActionCodec, AffineActionCodecConfig  # noqa: E402
from models.wam_multimodal import AgentFactorizedFlowWAM  # noqa: E402
from robofactory_rpc import (  # noqa: E402
    configure_socket,
    receive_message,
    send_message,
)
from scripts.run_static_rgb_act_moe_inference import (  # noqa: E402
    _connect,
    _load_yaml,
    _mapping,
    _rgb_array_name,
    _sha256,
    _vision,
)


CHECKPOINT_FORMAT = "wam.robofactory.agent_factorized_flow.checkpoint/1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/wam_flow/s1_r1_f1_flow_cold.yaml",
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
        raise RuntimeError("S1-R1 Flow inference requires one visible GPU")
    device = torch.device(args.device)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    config_path = args.config.expanduser().resolve(strict=True)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if saved.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError("checkpoint is not S1-R1 AgentFactorizedFlow")
    method = _mapping(saved, "method")
    if (
        method.get("action_generator") != "rectified_flow_cold"
        or method.get("future_path") is not False
        or method.get("active_agent_loss_weighting") is not False
    ):
        raise ValueError("checkpoint violates the frozen S1-R1 F1 method contract")
    model_config = StaticRGBMoEACTConfig.from_dict(
        _mapping(saved, "model_config")
    )
    model = AgentFactorizedFlowWAM(model_config).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    raw_config = _load_yaml(config_path)
    configured_model = StaticRGBMoEACTConfig.from_dict(
        _mapping(raw_config, "model")
    )
    if configured_model != model_config:
        raise ValueError("runtime config model does not match checkpoint model")
    generation = _validated_generation(
        _mapping(raw_config, "generation"),
        _mapping(saved, "generation"),
    )
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
    noise_seed = int(inference.get("flow_noise_seed", 101))
    action_source = "agent_factorized_rectified_flow_cold"
    connection: socket.socket | None = None
    try:
        connection = _connect(
            args.host,
            args.port,
            timeout=args.connect_timeout,
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
            raise ValueError("S1-R1 F1 does not expose a future path")
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
            raise ValueError("S1-R1 Flow requires existing global+agent RGB order")
        codec = AffineActionCodec(
            AffineActionCodecConfig.from_dict(task["action_codec"])
        ).to(device)
        state_mean = torch.tensor(task["state_mean"], device=device)
        state_std = torch.tensor(task["state_std"], device=device)
        action_mean = torch.tensor(task["action_mean"], device=device)
        action_std = torch.tensor(task["action_std"], device=device)
        send_message(
            connection,
            {
                "type": "ready",
                "accepted_contract": dict(contract),
                "client": {
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
                        "action_generator": "rectified_flow_cold",
                        "source_distribution": "standard_normal",
                        "horizon": model.config.horizon,
                        "decoder_kind": model.config.decoder_kind,
                        "solver_steps": generation["solver_steps"],
                        "solver": generation["solver"],
                        "chunk_aggregation": aggregator.mode,
                        "temporal_ensemble_decay": aggregator.decay,
                        "flow_noise_seed": noise_seed,
                        "camera_protocol": (
                            "existing world-fixed per-agent RGB; no wrist; no depth"
                        ),
                    },
                },
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
                    raise RuntimeError("S1-R1 Flow observation schedule drifted")
                if bool(message.get("reset")):
                    aggregator.reset()
                started = time.perf_counter()
                state = torch.as_tensor(
                    values["proprioception"],
                    device=device,
                    dtype=torch.float32,
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
                generator = torch.Generator(device=device)
                generator.manual_seed(
                    noise_seed + episode * 1_000_003 + step
                )
                with torch.inference_mode():
                    vision_tokens = vision(images).spatial_tokens
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        chunks = model.generate_actions(
                            vision_tokens,
                            normalized_state,
                            solver_steps=generation["solver_steps"],
                            solver=generation["solver"],
                            normalized_clip=generation["normalized_action_clip"],
                            generator=generator,
                        )
                aggregator.push(chunks.float())
                normalized_action = aggregator.current().flatten()
                canonical = (
                    normalized_action
                    * action_std[: normalized_action.numel()]
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
                            "action_generator": "rectified_flow_cold",
                            "source_distribution": "standard_normal",
                            "solver_steps": generation["solver_steps"],
                            "solver": generation["solver"],
                            "flow_noise_seed": (
                                noise_seed + episode * 1_000_003 + step
                            ),
                            "future_path": False,
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


def _validated_generation(
    configured: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        "source_distribution": "standard_normal",
        "solver_steps": 4,
        "solver": "euler",
        "normalized_action_clip": 10.0,
    }
    observed = {
        "source_distribution": str(configured.get("source_distribution", "")),
        "solver_steps": int(configured.get("solver_steps", 0)),
        "solver": str(configured.get("solver", "")),
        "normalized_action_clip": float(
            configured.get("normalized_action_clip", 0.0)
        ),
    }
    saved = {
        "source_distribution": str(checkpoint.get("source_distribution", "")),
        "solver_steps": int(checkpoint.get("solver_steps", 0)),
        "solver": str(checkpoint.get("solver", "")),
        "normalized_action_clip": float(
            checkpoint.get("normalized_action_clip", 0.0)
        ),
    }
    if observed != expected or saved != expected:
        raise ValueError("runtime/checkpoint generation differs from S1-R1 F1")
    return observed


if __name__ == "__main__":
    raise SystemExit(main())
