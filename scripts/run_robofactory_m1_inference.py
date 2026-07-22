"""Load the formal scratch M1 checkpoint and serve actions to RoboFactory."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robofactory_rpc import (  # noqa: E402
    FORMAL_LIFTBARRIER_M1_CONFIG_SHA256,
    configure_socket,
    receive_message,
    send_message,
)
from models.wam import ActionChunkConfig  # noqa: E402
from models.wam_multimodal import (  # noqa: E402
    FrozenDINOv3Config,
    FrozenDINOv3Encoder,
)
from policies.scratch_m1 import ScratchM1Policy, ScratchM1PolicyConfig  # noqa: E402
from train.m1_scratch_checkpointing import (  # noqa: E402
    load_scratch_m1_checkpoint,
    scratch_checkpoint_tree_sha256,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly load a frozen-DINO scratch M1 checkpoint and answer "
            "RoboFactory closed-loop observation requests."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/m1_liftbarrier_scratch_seed101",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/wam_multimodal/m1_liftbarrier_scratch.yaml",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=600.0,
        help="Total seconds to retry connecting to the environment server.",
    )
    parser.add_argument(
        "--socket-timeout",
        type=float,
        default=600.0,
        help="Maximum seconds to wait for one environment message.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    config_path = args.config.expanduser().resolve(strict=True)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    config_sha256 = _sha256(config_path)
    if config_sha256 != FORMAL_LIFTBARRIER_M1_CONFIG_SHA256:
        raise ValueError(
            "rollout config SHA-256 differs from the formal LiftBarrier M1 "
            f"training config: {config_sha256}"
        )
    config = _load_yaml(config_path)
    device = _device(args.device)
    _configure_compute(config, device=device)

    print(
        f"[inference] Loading and verifying frozen DINOv3 on {device}…",
        flush=True,
    )
    vision_encoder = _build_vision_encoder(_mapping(config, "vision"))
    print(f"[inference] Strictly loading checkpoint {checkpoint_path}…", flush=True)
    bundle, metadata = load_scratch_m1_checkpoint(
        checkpoint_path,
        vision_encoder=vision_encoder,
        device=device,
    )
    policy_config = _policy_config(config)
    policy = ScratchM1Policy.from_bundle(
        bundle,
        policy_config,
        device=device,
    )
    checkpoint_tree = scratch_checkpoint_tree_sha256(checkpoint_path)
    schema = metadata["schema"]
    client_metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_tree_sha256": checkpoint_tree,
        "checkpoint_format": schema["format_version"],
        "config_sha256": config_sha256,
        "train_seed": int(schema["train_seed"]),
        "vision_identity": schema["vision_identity"],
        "vision_runtime": _vision_runtime_identity(_mapping(config, "vision")),
        "action_codec_sha256": schema["action_codec_sha256"],
        "action_anchor_mode": schema["action_anchor_mode"],
        "device": str(device),
        "provenance": _wam_provenance(device=device),
        "policy": {
            "camera_order": list(policy_config.camera_order),
            "visual_history_frames": policy_config.visual_history_frames,
            "action_horizon": policy_config.action_chunk.horizon,
            "execution_steps": policy_config.action_chunk.execution_steps,
            "solver_steps": policy_config.action_chunk.solver_steps,
            "solver": policy_config.solver,
            "normalized_action_clip": policy_config.normalized_action_clip,
            "replan_on_new_image": policy_config.replan_on_new_image,
            "warm_start": policy_config.replan_warm_start_enabled,
            "cold_start_history": "masked_zero_padding_no_action/1",
        },
    }

    connection: socket.socket | None = None
    active_request: Mapping[str, Any] | None = None
    try:
        print(
            f"[inference] Connecting to RoboFactory at {args.host}:{args.port}…",
            flush=True,
        )
        connection = _connect_with_retry(
            args.host,
            args.port,
            timeout_seconds=args.connect_timeout,
        )
        configure_socket(connection, timeout_seconds=args.socket_timeout)
        hello, hello_arrays = receive_message(connection)
        _raise_peer_error(hello, peer="environment")
        if hello_arrays or hello.get("type") != "hello":
            raise RuntimeError("environment peer did not send a valid hello handshake")
        _validate_hello_schedule(hello)
        contract = _mapping(hello, "contract")
        _validate_environment_contract(contract, bundle=bundle, config=config)
        send_message(
            connection,
            {
                "type": "ready",
                "accepted_contract": dict(contract),
                "client": client_metadata,
            },
        )
        print(
            f"[inference] Ready: {hello['episodes']} episodes, "
            f"seeds {hello['seeds'][0]}..{hello['seeds'][-1]}",
            flush=True,
        )

        current_episode: int | None = None
        expected_step = 0
        completed_episodes = 0
        while True:
            message, arrays = receive_message(connection)
            message_type = message.get("type")
            if message_type == "observation":
                active_request = message
                episode_index, step = _validate_observation_message(
                    message,
                    arrays,
                    current_episode=current_episode,
                    expected_episode=completed_episodes,
                    expected_step=expected_step,
                )
                if bool(message["reset"]):
                    policy.reset()
                    current_episode = episode_index
                    expected_step = 0
                task = _mapping(message, "task")
                observation = {
                    "task": {"id": str(task["id"]), "text": str(task["text"])},
                    "proprioception": arrays["proprioception"],
                    "images": {"global": arrays["rgb_global"]},
                    "image_frame_indices": {
                        "global": int(message["image_frame_index"])
                    },
                }
                inference_started = time.perf_counter()
                action = policy.act(observation)
                inference_latency_ms = (time.perf_counter() - inference_started) * 1000.0
                diagnostics = _json_diagnostics(policy.last_diagnostics)
                send_message(
                    connection,
                    {
                        "type": "action",
                        "request_id": message["request_id"],
                        "episode_index": episode_index,
                        "step": step,
                        "inference_latency_ms": inference_latency_ms,
                        "diagnostics": diagnostics,
                    },
                    {"action": action},
                )
                expected_step = step + 1
                active_request = None
            elif message_type == "episode_result":
                if arrays:
                    raise RuntimeError("episode_result unexpectedly contains arrays")
                if current_episode is None or int(message["episode_index"]) != current_episode:
                    raise RuntimeError("received an episode result for the wrong episode")
                if int(message.get("steps", -1)) != expected_step:
                    raise RuntimeError("episode result step count differs from issued actions")
                completed_episodes += 1
                print(
                    f"[inference] episode={completed_episodes}/{hello['episodes']} "
                    f"seed={message['seed']} success={message['success']} "
                    f"steps={message['steps']} "
                    f"running_rate={float(message['success_rate_so_far']):.2%}",
                    flush=True,
                )
                current_episode = None
                expected_step = 0
            elif message_type == "summary":
                if arrays:
                    raise RuntimeError("summary unexpectedly contains arrays")
                summary = _mapping(message, "summary")
                if summary.get("completed") is not True:
                    raise RuntimeError("environment returned an incomplete rollout summary")
                if int(summary.get("episodes_completed", -1)) != int(hello["episodes"]):
                    raise RuntimeError("environment summary has an incomplete episode count")
                print(
                    f"[inference] complete: {summary['successes']}/"
                    f"{summary['episodes_completed']} = "
                    f"{float(summary['success_rate']):.2%}",
                    flush=True,
                )
                formal = _mapping(summary, "formal_benchmark")
                print(
                    "[inference] formal benchmark reportable: "
                    f"{formal.get('reportable')}",
                    flush=True,
                )
                print(f"[inference] outputs: {summary['output_dir']}", flush=True)
                return 0
            elif message_type == "error":
                raise RuntimeError(f"environment failed: {message.get('error')}")
            else:
                raise RuntimeError(f"unexpected environment message type {message_type!r}")
    except BaseException as exc:
        if connection is not None:
            try:
                send_message(
                    connection,
                    {
                        "type": "error",
                        "fatal": True,
                        "request_id": (
                            active_request.get("request_id")
                            if active_request is not None
                            else None
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


def _validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in [1,65535]")
    if (
        not math.isfinite(args.connect_timeout)
        or args.connect_timeout <= 0.0
        or not math.isfinite(args.socket_timeout)
        or args.socket_timeout <= 0.0
    ):
        raise ValueError("connection/socket timeouts must be positive")


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("M1 config root must be a mapping")
    if value.get("format_version") != "wam.multimodal.m1.scratch_config/1":
        raise ValueError("unsupported scratch M1 config format")
    return value


def _build_vision_encoder(config: Mapping[str, Any]) -> FrozenDINOv3Encoder:
    if config.get("frozen") is not True:
        raise ValueError("closed-loop M1 requires a frozen DINOv3 encoder")
    return FrozenDINOv3Encoder(
        FrozenDINOv3Config(
            encoder_name=str(config["encoder_name"]),
            model_id=str(config["model_id"]),
            revision=str(config["revision"]),
            config_path=_root_path(config["config_path"]),
            weights_path=_root_path(config["weights_path"]),
            expected_config_sha256=str(config["expected_config_sha256"]),
            expected_weights_sha256=str(config["expected_weights_sha256"]),
            preprocess_id=str(config["preprocess_id"]),
            input_size=int(config["input_size"]),
            inference_batch_size=int(config["inference_batch_size"]),
        )
    )


def _vision_runtime_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "encoder_name": str(config["encoder_name"]),
        "model_id": str(config["model_id"]),
        "revision": str(config["revision"]),
        "expected_config_sha256": str(config["expected_config_sha256"]),
        "expected_weights_sha256": str(config["expected_weights_sha256"]),
        "preprocess_id": str(config["preprocess_id"]),
        "input_size": int(config["input_size"]),
        "inference_batch_size": int(config["inference_batch_size"]),
        "frozen": config.get("frozen") is True,
    }


def _policy_config(
    config: Mapping[str, Any],
) -> ScratchM1PolicyConfig:
    data = _mapping(config, "data")
    flow = _mapping(config, "flow_objective")
    build = _mapping(config, "build")
    action_flow = _mapping(build, "action_flow")
    return ScratchM1PolicyConfig(
        action_chunk=ActionChunkConfig(
            action_dim=int(data["action_dim"]),
            horizon=int(action_flow["horizon"]),
            execution_steps=int(flow["execution_steps"]),
            solver_steps=int(flow["solver_steps"]),
        ),
        camera_order=tuple(str(value) for value in data["camera_order"]),
        visual_history_frames=int(data["visual_history_frames"]),
        solver=str(flow["solver"]),
        normalized_action_clip=float(flow["normalized_action_clip"]),
        replan_on_new_image=False,
        replan_warm_start_enabled=True,
    )


def _configure_compute(config: Mapping[str, Any], *, device: torch.device) -> None:
    training = _mapping(config, "training")
    torch.set_float32_matmul_precision(
        str(training.get("torch_float32_matmul_precision", "high"))
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(training.get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(training.get("allow_tf32", True))
        torch.backends.cudnn.benchmark = bool(training.get("cudnn_benchmark", True))


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def _connect_with_retry(host: str, port: int, *, timeout_seconds: float) -> socket.socket:
    deadline = time.monotonic() + timeout_seconds
    last_error: OSError | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError(
                f"could not connect to RoboFactory at {host}:{port}"
            ) from last_error
        try:
            return socket.create_connection((host, port), timeout=min(2.0, remaining))
        except OSError as exc:
            last_error = exc
            time.sleep(min(0.5, max(remaining, 0.0)))


def _validate_hello_schedule(hello: Mapping[str, Any]) -> None:
    episodes = hello.get("episodes")
    seeds = hello.get("seeds")
    if not isinstance(episodes, int) or episodes <= 0:
        raise RuntimeError("environment hello has an invalid episode count")
    if (
        not isinstance(seeds, list)
        or len(seeds) != episodes
        or any(not isinstance(seed, int) or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise RuntimeError("environment hello has an invalid/duplicate seed schedule")


def _validate_environment_contract(
    contract: Mapping[str, Any],
    *,
    bundle: Any,
    config: Mapping[str, Any],
) -> None:
    data = _mapping(config, "data")
    expected = {
        "environment_id": "LiftBarrier-rf",
        "task_id": "lift_barrier",
        "task_text": "Lift the barrier together",
        "state_dim": bundle.model.world_model.config.state_dim,
        "state_order": [
            "panda-0.qpos[9]",
            "panda-0.qvel[9]",
            "panda-1.qpos[9]",
            "panda-1.qvel[9]",
        ],
        "action_dim": bundle.action_flow.config.action_dim,
        "agent_order": ["panda-0", "panda-1"],
        "control_mode": "pd_joint_pos",
        "control_hz": float(data["control_hz"]),
        "camera_order": list(data["camera_order"]),
        "camera_source": "head_camera_global",
        "camera_shape": [240, 320, 3],
        "camera_dtype": "uint8",
        "rgb_encoding": "raw_lossless",
        "success_source": "info.success",
    }
    mismatched = {
        key: {"expected": value, "observed": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatched:
        raise RuntimeError(f"RoboFactory/M1 online contract mismatch: {mismatched}")
    environment_max = int(contract.get("environment_max_episode_steps", 0))
    rollout_max = int(contract.get("rollout_max_steps", 0))
    if environment_max != 500 or not 1 <= rollout_max <= environment_max:
        raise RuntimeError("environment/rollout episode limits are inconsistent")


def _validate_observation_message(
    message: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    current_episode: int | None,
    expected_episode: int,
    expected_step: int,
) -> tuple[int, int]:
    if set(arrays) != {"proprioception", "rgb_global"}:
        raise RuntimeError("observation must contain proprioception and rgb_global")
    state = arrays["proprioception"]
    rgb = arrays["rgb_global"]
    if state.shape != (36,) or state.dtype != np.float32 or not np.isfinite(state).all():
        raise RuntimeError("online proprioception must be finite float32[36]")
    if rgb.shape != (240, 320, 3) or rgb.dtype != np.uint8:
        raise RuntimeError("online global RGB must be uint8[240,320,3]")
    episode_index = int(message.get("episode_index", -1))
    step = int(message.get("step", -1))
    request_id = message.get("request_id")
    reset = message.get("reset")
    if request_id != f"{episode_index}:{step}":
        raise RuntimeError("observation request_id does not match episode/step")
    if int(message.get("image_frame_index", -1)) != step:
        raise RuntimeError("online image frame index must equal the control step")
    task = message.get("task")
    if not isinstance(task, Mapping) or task.get("id") != "lift_barrier":
        raise RuntimeError("observation task identity is invalid")
    if reset is True:
        if (
            step != 0
            or current_episode is not None
            or episode_index != expected_episode
        ):
            raise RuntimeError("episode reset is stale or out of order")
    elif reset is False:
        if current_episode is None or episode_index != current_episode:
            raise RuntimeError("observation belongs to the wrong active episode")
        if step != expected_step:
            raise RuntimeError(
                f"non-idempotent policy request is out of order: "
                f"expected step {expected_step}, received {step}"
            )
    else:
        raise RuntimeError("observation reset flag must be boolean")
    return episode_index, step


def _json_diagnostics(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("policy diagnostics are not finite JSON data") from exc


def _raise_peer_error(message: Mapping[str, Any], *, peer: str) -> None:
    if message.get("type") == "error":
        raise RuntimeError(f"{peer} failed: {message.get('error')}")


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ValueError(f"field {key!r} must be a mapping")
    return item


def _root_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve(strict=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wam_provenance(*, device: torch.device) -> dict[str, Any]:
    source_paths = (
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
    )
    return {
        "repository_root": str(ROOT),
        "source_sha256": {name: _sha256(ROOT / name) for name in source_paths},
        "git": _git_provenance(ROOT),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "transformers": _package_version("transformers"),
            "safetensors": _package_version("safetensors"),
            "pyyaml": _package_version("PyYAML"),
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
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
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


def _package_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
