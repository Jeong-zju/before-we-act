#!/usr/bin/env python3
"""Serve an S3-R6 frozen-parent Flow with optional predicted-future injection."""

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
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.static_rgb_act import build_chunk_aggregator  # noqa: E402
from models.wam import AffineActionCodec, AffineActionCodecConfig  # noqa: E402
from robofactory_rpc import configure_socket, receive_message, send_message  # noqa: E402
from scripts.run_static_rgb_act_moe_inference import (  # noqa: E402
    _connect,
    _load_yaml,
    _mapping,
    _rgb_array_name,
    _sha256,
    _vision,
)
from scripts.s3_r6_model_io import build_s3_r6_model  # noqa: E402
from scripts.train_s3_r6_world_action_flow import CHECKPOINT_FORMAT  # noqa: E402
from train.s2_future_prediction import (  # noqa: E402
    load_s2_artifact,
    project_dino_grid,
)
from train.s3_model_registry import validate_s3_r6_candidate  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8872)
    parser.add_argument("--connect-timeout", type=float, default=600.0)
    parser.add_argument("--socket-timeout", type=float, default=600.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("S3-R6 inference requires exactly one visible GPU")
    device = torch.device(args.device)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    config_path = args.config.expanduser().resolve(strict=True)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(saved, Mapping) or saved.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError("checkpoint is not an S3-R6 world-action Flow")
    raw = _load_yaml(config_path)
    identity = validate_s3_r6_candidate(_mapping(raw, "round"))
    method = _mapping(saved, "method")
    if tuple(method.get(name) for name in (
        "micro_round", "candidate_id", "model_kind", "future_scope", "injection"
    )) != (identity[0], identity[1], identity[2], identity[3], identity[4]):
        raise ValueError("runtime config and S3 checkpoint candidate identities differ")
    model, parent_identity = build_s3_r6_model(
        raw,
        device=device,
        future_scope=identity[3],
        injection=identity[4],
    )
    if dict(_mapping(saved, "parent_identity")) != parent_identity:
        raise ValueError("S3 runtime parent identities differ from checkpoint")
    if identity[4]:
        model.load_adapter_state_dict(_mapping(saved, "adapter"))
    model.eval()
    vision = _vision(raw).to(device).eval()
    artifact = load_s2_artifact(
        (ROOT / str(_mapping(raw, "artifacts")["pca_statistics"])).resolve(
            strict=True
        ),
        device=device,
    )
    generation = _validated_generation(
        _mapping(raw, "generation"), _mapping(saved, "generation")
    )
    runtime = {str(value["task_id"]): dict(value) for value in saved["task_runtime"]}
    inference = _mapping(raw, "inference")
    aggregator = build_chunk_aggregator(
        mode=str(inference.get("chunk_aggregation", "temporal_ensemble")),
        horizon=model.base_flow.config.horizon,
        decay=float(inference.get("temporal_ensemble_decay", 0.01)),
    )
    noise_seed = int(inference.get("flow_noise_seed", 606))
    action_source = _action_source(identity[0], identity[1], identity[4])
    connection: socket.socket | None = None
    try:
        connection = _connect(args.host, args.port, timeout=args.connect_timeout)
        configure_socket(connection, timeout_seconds=args.socket_timeout)
        hello, arrays = receive_message(connection)
        if arrays or hello.get("type") != "hello":
            raise RuntimeError("RoboFactory server did not send a valid hello")
        contract = _mapping(hello, "contract")
        task_id = str(contract.get("task_id", ""))
        if task_id not in runtime:
            raise ValueError(f"checkpoint does not contain task {task_id!r}")
        if bool(contract.get("future_path")):
            raise ValueError("S3 uses predicted latent futures, not environment future input")
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
            raise ValueError("S3-R6 requires the canonical global+agent RGB order")
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
                        "action_generator": "rectified_flow_cold_gated_residual",
                        "future_scope": identity[3],
                        "injection": identity[4],
                        "gate": float(model.adapter.bounded_gate().detach())
                        if identity[4]
                        else 0.0,
                        "candidate_action_contract": (
                            "clean_endpoint_each_solver_evaluation"
                        ),
                        "solver_steps": generation["solver_steps"],
                        "solver": generation["solver"],
                        "chunk_aggregation": aggregator.mode,
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
                    raise RuntimeError("S3-R6 observation schedule drifted")
                if bool(message.get("reset")):
                    aggregator.reset()
                started = time.perf_counter()
                state = torch.as_tensor(
                    values["proprioception"], device=device, dtype=torch.float32
                )
                normalized = (
                    (state - state_mean[: state.numel()])
                    / state_std[: state.numel()]
                ).reshape(agent_count, 18)
                padded_state = torch.zeros(1, 4, 18, device=device)
                padded_state[0, :agent_count] = normalized
                agent_images = torch.stack(
                    [
                        torch.as_tensor(
                            values[_rgb_array_name(f"agent_{index}")], device=device
                        ).permute(2, 0, 1)
                        for index in range(agent_count)
                    ]
                )
                global_image = torch.as_tensor(
                    values[_rgb_array_name("global")], device=device
                ).permute(2, 0, 1)[None]
                valid = torch.zeros(1, 4, dtype=torch.bool, device=device)
                valid[0, :agent_count] = True
                with torch.inference_mode():
                    context = _visual_context(
                        vision,
                        artifact,
                        agent_images,
                        global_image,
                        agent_count=agent_count,
                        grid_height=int(_mapping(raw, "pca")["grid_height"]),
                        grid_width=int(_mapping(raw, "pca")["grid_width"]),
                    )
                    generator = torch.Generator(device=device)
                    generator.manual_seed(noise_seed + episode * 1_000_003 + step)
                    initial = torch.randn(
                        1,
                        4,
                        model.base_flow.config.horizon,
                        model.base_flow.config.action_dim,
                        device=device,
                        generator=generator,
                    )
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        chunks = model.integrate_actions(
                            context["raw_local"],
                            padded_state,
                            context["local_visual"],
                            context["shared_visual"],
                            valid,
                            initial_actions=initial,
                            solver_steps=generation["solver_steps"],
                            solver=generation["solver"],
                            normalized_clip=generation["normalized_action_clip"],
                        )[0, :agent_count]
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
                action = raw_action.float().cpu().numpy().astype(np.float32, copy=False)
                aggregator.advance()
                send_message(
                    connection,
                    {
                        "type": "action",
                        "request_id": message["request_id"],
                        "episode_index": episode,
                        "step": step,
                        "inference_latency_ms": (time.perf_counter() - started) * 1000,
                        "diagnostics": {
                            "action_source": action_source,
                            "action_generator": "rectified_flow_cold_gated_residual",
                            "future_scope": identity[3],
                            "injection": identity[4],
                            "gate": float(model.adapter.bounded_gate().detach())
                            if identity[4]
                            else 0.0,
                            "fallback_used": False,
                            "direct_model_action": True,
                            "task_id": task_id,
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
                    f"complete {summary['successes']}/{summary['episodes_completed']} "
                    f"= {summary['success_rate']:.2%}",
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


def _visual_context(
    vision: torch.nn.Module,
    artifact: Mapping[str, Any],
    agent_images: Tensor,
    global_image: Tensor,
    *,
    agent_count: int,
    grid_height: int,
    grid_width: int,
) -> dict[str, Tensor]:
    raw = vision(agent_images).spatial_tokens.float()
    patch_rows = int(vision.config.image_height) // int(vision.patch_size)
    patch_columns = int(vision.config.image_width) // int(vision.patch_size)
    grid = F.adaptive_avg_pool2d(
        raw.transpose(1, 2).reshape(
            raw.shape[0], raw.shape[-1], patch_rows, patch_columns
        ),
        (grid_height, grid_width),
    ).flatten(2).transpose(1, 2)
    local = project_dino_grid(grid, artifact)
    shared_grid = vision.forward_spatial_grid(
        global_image, grid_height=grid_height, grid_width=grid_width
    ).spatial_tokens.float()
    shared = project_dino_grid(shared_grid, artifact)
    raw_padded = raw.new_zeros(1, 4, raw.shape[1], raw.shape[2])
    local_padded = local.new_zeros(1, 4, local.shape[1], local.shape[2])
    raw_padded[0, :agent_count] = raw
    local_padded[0, :agent_count] = local
    return {
        "raw_local": raw_padded,
        "local_visual": local_padded,
        "shared_visual": shared,
    }


def _validated_generation(
    configured: Mapping[str, Any], saved: Mapping[str, Any]
) -> dict[str, Any]:
    observed = {
        "source_distribution": str(configured.get("source_distribution", "")),
        "solver_steps": int(configured.get("solver_steps", 0)),
        "solver": str(configured.get("solver", "")),
        "normalized_action_clip": float(configured.get("normalized_action_clip", 0)),
    }
    expected = {
        "source_distribution": "standard_normal",
        "solver_steps": 4,
        "solver": "euler",
        "normalized_action_clip": 10.0,
    }
    saved_value = {
        key: (
            float(saved.get(key, 0))
            if key == "normalized_action_clip"
            else int(saved.get(key, 0))
            if key == "solver_steps"
            else str(saved.get(key, ""))
        )
        for key in expected
    }
    if observed != expected or saved_value != expected:
        raise ValueError("S3 generation differs from the frozen four-step Euler contract")
    return observed


def _action_source(micro_round: str, candidate_id: str, injection: bool) -> str:
    return (
        f"s3_{micro_round.lower()}_protected_"
        f"{'gated_residual' if injection else 'offpath'}_flow_{candidate_id.lower()}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
