#!/usr/bin/env python3
"""Serve an accepted-format S4-R7 policy under a versioned intervention."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import os
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

from models.static_rgb_act import build_chunk_aggregator  # noqa: E402
from models.wam import AffineActionCodec, AffineActionCodecConfig  # noqa: E402
from models.wam_multimodal import PredictedFutureLatents  # noqa: E402
from robofactory_rpc import configure_socket, receive_message, send_message  # noqa: E402
from scripts.run_s3_r6_world_action_flow_inference import (  # noqa: E402
    _validated_generation,
    _visual_context,
)
from scripts.run_static_rgb_act_moe_inference import (  # noqa: E402
    _connect,
    _load_yaml,
    _mapping,
    _rgb_array_name,
    _sha256,
    _vision,
)
from scripts.s4_r7_model_io import build_s4_r7_model  # noqa: E402
from scripts.s4_r8_model_io import build_s4_r8_model  # noqa: E402
from scripts.train_static_rgb_act_moe import _append_jsonl  # noqa: E402
from train.s2_future_prediction import load_s2_artifact  # noqa: E402
from train.s4_model_registry import (  # noqa: E402
    validate_s4_r7_candidate,
    validate_s4_r8_candidate,
)


INTERVENTIONS = (
    "legacy_reference",
    "normal",
    "world_evidence_gate_zero",
    "all_world_gates_zero",
    "shuffle_all",
    "shuffle_own",
    "shuffle_peer",
    "shuffle_shared",
)
EVIDENCE_BANK_FORMAT = "wam.robofactory.s4_r7.predicted_future_donor_bank/1"
R7_CHECKPOINT_FORMAT = "wam.robofactory.s4_r7.world_utility.checkpoint/1"
R8_CHECKPOINT_FORMAT = "wam.robofactory.s4_r8.horizon_causal.checkpoint/1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8872)
    parser.add_argument("--connect-timeout", type=float, default=600.0)
    parser.add_argument("--socket-timeout", type=float, default=600.0)
    parser.add_argument("--intervention", choices=INTERVENTIONS)
    parser.add_argument("--evidence-bank-dir", type=Path)
    parser.add_argument("--progress-log", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("S4-R7 inference requires exactly one visible GPU")
    device = torch.device(args.device)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    config_path = args.config.expanduser().resolve(strict=True)
    checkpoint_sha256 = _sha256(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(saved, Mapping) or saved.get("format_version") not in {
        R7_CHECKPOINT_FORMAT,
        R8_CHECKPOINT_FORMAT,
    }:
        raise ValueError("checkpoint is not a registered S4-R7/R8 world Flow")
    raw = _load_yaml(config_path)
    round_id = str(_mapping(raw, "round").get("round_id", ""))
    if round_id == "s4-r7":
        candidate_id, model_kind, utility_weight = validate_s4_r7_candidate(raw)
        expected_format = R7_CHECKPOINT_FORMAT
        model_builder = build_s4_r7_model
        environment_prefix = "S4_R7"
        action_source = (
            "s4_r7_token_preserving_world_flow"
            if candidate_id == "P0"
            else "s4_r7_world_utility_coupled_flow"
        )
    elif round_id == "s4-r8":
        candidate_id, model_kind, _ = validate_s4_r8_candidate(raw)
        utility_weight = float(_mapping(raw, "training")["utility_coupling_weight"])
        expected_format = R8_CHECKPOINT_FORMAT
        model_builder = build_s4_r8_model
        environment_prefix = "S4_R8"
        action_source = (
            "s4_r8_horizon_prefix_mean_world_flow"
            if candidate_id == "P0"
            else "s4_r8_causal_prefix_attention_world_flow"
        )
    else:
        raise ValueError(f"unsupported S4 runtime round: {round_id!r}")
    method = _mapping(saved, "method")
    if (
        saved.get("format_version") != expected_format
        or method.get("round_id") != round_id
        or method.get("candidate_id") != candidate_id
        or method.get("model_kind") != model_kind
        or float(method.get("utility_coupling_weight", -1.0)) != utility_weight
    ):
        raise ValueError("runtime config and S4-R7 checkpoint identities differ")
    model, legacy_reference, parent_identity = model_builder(raw, device=device)
    if dict(_mapping(saved, "parent_identity")) != parent_identity:
        raise ValueError("S4-R7 runtime ancestor identities differ from checkpoint")
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    intervention = args.intervention or os.environ.get(
        f"{environment_prefix}_INTERVENTION",
        str(_mapping(raw, "inference").get("world_intervention", "normal")),
    )
    if intervention not in INTERVENTIONS:
        raise ValueError(f"unsupported S4-R7 intervention: {intervention}")
    if intervention == "legacy_reference":
        legacy_reference = legacy_reference.to(device).eval()

    vision = _vision(raw).to(device).eval()
    artifact = load_s2_artifact(
        (ROOT / str(_mapping(raw, "artifacts")["pca_statistics"])).resolve(strict=True),
        device=device,
    )
    generation = _validated_generation(
        _mapping(raw, "generation"), _mapping(saved, "generation")
    )
    runtime = {str(value["task_id"]): dict(value) for value in saved["task_runtime"]}
    inference = _mapping(raw, "inference")
    aggregator = build_chunk_aggregator(
        mode=str(inference.get("chunk_aggregation", "temporal_ensemble")),
        horizon=model.active_parent.base_flow.config.horizon,
        decay=float(inference.get("temporal_ensemble_decay", 0.01)),
    )
    noise_seed = int(inference.get("flow_noise_seed", 606))
    progress_log = (
        args.progress_log.expanduser().resolve()
        if args.progress_log is not None
        else Path(os.environ[f"{environment_prefix}_ROLLOUT_PROGRESS"])
        .expanduser()
        .resolve()
        if os.environ.get(f"{environment_prefix}_ROLLOUT_PROGRESS")
        else None
    )
    bank_root = (
        args.evidence_bank_dir.expanduser().resolve()
        if args.evidence_bank_dir is not None
        else Path(os.environ[f"{environment_prefix}_EVIDENCE_BANK_DIR"])
        .expanduser()
        .resolve()
        if os.environ.get(f"{environment_prefix}_EVIDENCE_BANK_DIR")
        else None
    )
    if intervention.startswith("shuffle_") and bank_root is None:
        raise ValueError("shuffle interventions require a normal predicted-future bank")

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
            raise ValueError(
                "S4 uses predicted futures, never environment future input"
            )
        task = runtime[task_id]
        agent_count = int(contract["agent_count"])
        if (
            agent_count != int(task["agent_count"])
            or int(contract["state_dim"]) != int(task["state_dim"])
            or int(contract["action_dim"]) != int(task["action_dim"])
        ):
            raise ValueError("environment and S4 checkpoint task dimensions differ")
        expected_cameras = ("global",) + tuple(
            f"agent_{index}" for index in range(agent_count)
        )
        if tuple(contract["camera_order"]) != expected_cameras:
            raise ValueError("S4-R7 requires canonical global+agent RGB cameras")
        codec = AffineActionCodec(
            AffineActionCodecConfig.from_dict(task["action_codec"])
        ).to(device)
        state_mean = torch.tensor(task["state_mean"], device=device)
        state_std = torch.tensor(task["state_std"], device=device)
        action_mean = torch.tensor(task["action_mean"], device=device)
        action_std = torch.tensor(task["action_std"], device=device)
        bank = PredictedFutureDonorBank(
            bank_root,
            task_id=task_id,
            checkpoint_sha256=checkpoint_sha256,
            intervention=intervention,
            device=device,
        )
        send_message(
            connection,
            {
                "type": "ready",
                "accepted_contract": dict(contract),
                "client": {
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha256,
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
                        "action_generator": "rectified_flow_cold_token_preserving_world_evidence",
                        "model_kind": model_kind,
                        "utility_coupling_weight": utility_weight,
                        "world_intervention": intervention,
                        "evidence_shuffle_protocol": (
                            "within_task_different_episode_predicted_future"
                            if intervention.startswith("shuffle_")
                            else None
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
                    raise RuntimeError("S4-R7 observation schedule drifted")
                if bool(message.get("reset")):
                    aggregator.reset()
                    bank.begin_episode(episode)
                started = time.perf_counter()
                state = torch.as_tensor(
                    values["proprioception"], device=device, dtype=torch.float32
                )
                normalized = (
                    (state - state_mean[: state.numel()]) / state_std[: state.numel()]
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
                        model.active_parent.base_flow.config.horizon,
                        model.active_parent.base_flow.config.action_dim,
                        device=device,
                        generator=generator,
                    )
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        if intervention == "legacy_reference":
                            chunks = legacy_reference.integrate_actions(
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
                        else:
                            chunks, captured = _integrate_s4(
                                model,
                                context,
                                padded_state,
                                valid,
                                initial,
                                generation,
                                intervention=intervention,
                                donor_bank=bank,
                                episode=episode,
                                environment_step=step,
                            )
                            chunks = chunks[0, :agent_count]
                            bank.capture_step(episode, captured)
                aggregator.push(chunks.float())
                normalized_action = aggregator.current().flatten()
                canonical = (
                    normalized_action * action_std[: normalized_action.numel()]
                    + action_mean[: normalized_action.numel()]
                ).clamp(-1.0, 1.0)
                raw_action = codec.decode(canonical, clip=True)
                if not isinstance(raw_action, Tensor):
                    raise TypeError("action codec returned a non-tensor")
                action = raw_action.float().cpu().numpy().astype(np.float32, copy=False)
                aggregator.advance()
                latency = (time.perf_counter() - started) * 1000
                send_message(
                    connection,
                    {
                        "type": "action",
                        "request_id": message["request_id"],
                        "episode_index": episode,
                        "step": step,
                        "inference_latency_ms": latency,
                        "diagnostics": {
                            "action_source": action_source,
                            "model_kind": model_kind,
                            "world_intervention": intervention,
                            "fallback_used": intervention == "legacy_reference",
                            "direct_model_action": True,
                            "task_id": task_id,
                        },
                    },
                    {"action": action},
                )
                _progress(
                    progress_log,
                    condition=intervention,
                    task=task_id,
                    episode=episode + 1,
                    step=step + 1,
                    latency_ms=latency,
                )
                expected_step += 1
            elif message_type == "episode_result":
                if int(message.get("episode_index", -1)) != expected_episode:
                    raise RuntimeError("S4-R7 episode result identity drifted")
                bank.end_episode(expected_episode)
                expected_episode += 1
                expected_step = 0
            elif message_type == "summary":
                summary = _mapping(message, "summary")
                if summary.get("completed") is not True:
                    raise RuntimeError("RoboFactory S4-R7 rollout was incomplete")
                bank.finish()
                break
            elif message_type == "error":
                raise RuntimeError(f"RoboFactory rollout error: {message.get('error')}")
            else:
                raise RuntimeError(f"unexpected RoboFactory message: {message_type!r}")
    finally:
        if connection is not None:
            connection.close()
    return 0


def _integrate_s4(
    model: torch.nn.Module,
    context: Mapping[str, Tensor],
    state: Tensor,
    valid: Tensor,
    initial: Tensor,
    generation: Mapping[str, Any],
    *,
    intervention: str,
    donor_bank: "PredictedFutureDonorBank",
    episode: int,
    environment_step: int,
) -> tuple[Tensor, list[PredictedFutureLatents]]:
    current = initial.clone()
    steps = int(generation["solver_steps"])
    dt = 1.0 / steps
    captured: list[PredictedFutureLatents] = []
    for solver_step in range(steps):
        tau = torch.full(
            (current.shape[0],),
            solver_step * dt,
            device=current.device,
            dtype=current.dtype,
        )
        transform = donor_bank.intervention(
            episode, environment_step, solver_step, intervention
        )
        velocity, diagnostics = model.velocity(
            context["raw_local"],
            state,
            context["local_visual"],
            context["shared_visual"],
            current,
            tau,
            valid,
            force_world_evidence_gate_zero=intervention == "world_evidence_gate_zero",
            force_all_world_gates_zero=intervention == "all_world_gates_zero",
            future_intervention=transform,
        )
        predicted = diagnostics.get("predicted_futures")
        if intervention == "normal" and isinstance(predicted, PredictedFutureLatents):
            captured.append(_future_to_cpu(predicted))
        current = (current + dt * velocity).clamp(
            -float(generation["normalized_action_clip"]),
            float(generation["normalized_action_clip"]),
        )
    return current, captured


class PredictedFutureDonorBank:
    def __init__(
        self,
        root: Path | None,
        *,
        task_id: str,
        checkpoint_sha256: str,
        intervention: str,
        device: torch.device,
    ) -> None:
        self.root = root / task_id if root is not None else None
        self.task_id = task_id
        self.checkpoint_sha256 = checkpoint_sha256
        self.mode = intervention
        self.device = device
        self.current_episode = -1
        self.current_steps: list[list[PredictedFutureLatents]] = []
        self.donors: dict[int, list[list[PredictedFutureLatents]]] = {}
        if intervention.startswith("shuffle_"):
            assert self.root is not None
            for episode in (0, 1):
                path = self.root / f"episode_{episode:03d}.pt"
                payload = torch.load(
                    path.resolve(strict=True), map_location="cpu", weights_only=False
                )
                if (
                    not isinstance(payload, Mapping)
                    or payload.get("format_version") != EVIDENCE_BANK_FORMAT
                    or payload.get("task_id") != task_id
                    or payload.get("checkpoint_sha256") != checkpoint_sha256
                    or int(payload.get("episode_index", -1)) != episode
                ):
                    raise ValueError(f"invalid predicted-future donor bank: {path}")
                self.donors[episode] = payload["steps"]

    def begin_episode(self, episode: int) -> None:
        self.current_episode = episode
        self.current_steps = []

    def capture_step(self, episode: int, futures: list[PredictedFutureLatents]) -> None:
        if self.mode == "normal" and self.root is not None and episode in {0, 1}:
            if len(futures) != 4:
                raise RuntimeError("normal donor capture requires all four Euler reads")
            self.current_steps.append(futures)

    def end_episode(self, episode: int) -> None:
        if self.mode != "normal" or self.root is None or episode not in {0, 1}:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"episode_{episode:03d}.pt"
        payload = {
            "format_version": EVIDENCE_BANK_FORMAT,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task_id": self.task_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "condition": "normal",
            "episode_index": episode,
            "steps": self.current_steps,
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def intervention(
        self,
        episode: int,
        environment_step: int,
        solver_step: int,
        intervention: str,
    ):
        if not intervention.startswith("shuffle_"):
            return None
        donor_episode = 1 if episode == 0 else 0
        steps = self.donors[donor_episode]
        if not steps:
            raise RuntimeError("predicted-future donor episode contains no steps")
        donor = steps[environment_step % len(steps)][solver_step]

        def transform(current: PredictedFutureLatents) -> PredictedFutureLatents:
            value = _future_to_device(donor, current.own_state)
            replace_own = intervention in {"shuffle_all", "shuffle_own"}
            replace_peer = intervention in {"shuffle_all", "shuffle_peer"}
            replace_shared = intervention in {"shuffle_all", "shuffle_shared"}
            return PredictedFutureLatents(
                own_state=value.own_state if replace_own else current.own_state,
                own_visual=value.own_visual if replace_own else current.own_visual,
                peer_state=value.peer_state if replace_peer else current.peer_state,
                peer_visual=value.peer_visual if replace_peer else current.peer_visual,
                shared_visual=value.shared_visual
                if replace_shared
                else current.shared_visual,
            )

        return transform

    def finish(self) -> None:
        if self.mode.startswith("shuffle_") and set(self.donors) != {0, 1}:
            raise RuntimeError("shuffle rollout did not retain both donor episodes")


def _future_to_cpu(value: PredictedFutureLatents) -> PredictedFutureLatents:
    return PredictedFutureLatents(
        **{
            name: getattr(value, name).detach().to(device="cpu", dtype=torch.bfloat16)
            for name in (
                "own_state",
                "own_visual",
                "peer_state",
                "peer_visual",
                "shared_visual",
            )
        }
    )


def _future_to_device(
    value: PredictedFutureLatents, reference: Tensor
) -> PredictedFutureLatents:
    return PredictedFutureLatents(
        **{
            name: getattr(value, name).to(
                device=reference.device, dtype=reference.dtype, non_blocking=True
            )
            for name in (
                "own_state",
                "own_visual",
                "peer_state",
                "peer_visual",
                "shared_visual",
            )
        }
    )


def _progress(
    path: Path | None,
    *,
    condition: str,
    task: str,
    episode: int,
    step: int,
    latency_ms: float,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _append_jsonl(
        path,
        {
            "event": "rollout_step",
            "program": "run_s4_r7_world_utility_inference.py",
            "condition": condition,
            "task": task,
            "episode": episode,
            "episodes_total": 20,
            "step": step,
            "inference_latency_ms": latency_ms,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
