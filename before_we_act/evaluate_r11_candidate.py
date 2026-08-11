"""Closed-loop evaluator for a branch-local R11 world-action candidate."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import torch

import robofactory  # noqa: F401

from before_we_act.r11_data import SIX_TASKS
from before_we_act.train_r11_candidate import (
    atomic_json,
    load_checkpoint_model_state,
    sha256_file,
)
from stereo_core.two_three_task_manifest import get_task


INFERENCE_MODES = ("normal", "prediction_off", "prediction_shuffled")


def reset_reproducibly(env, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return env.reset(seed=seed)


def load_candidate(
    checkpoint_path: Path,
    *,
    expected_sha256: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, torch.Tensor], dict, str]:
    observed_sha256 = sha256_file(checkpoint_path)
    if observed_sha256 != expected_sha256:
        raise ValueError("candidate checkpoint SHA256 differs from the launch receipt")
    saved = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    if saved.get("format_version") != "before-we-act.r11.checkpoint/1":
        raise ValueError("unsupported R11 checkpoint format")
    config = saved["config"]
    if saved.get("candidate") != config.get("candidate") or saved.get(
        "model_name"
    ) != config.get("model"):
        raise ValueError("checkpoint candidate/model identity differs")
    from before_we_act.r11_registry import build_r11_model

    project_root = Path(__file__).resolve().parents[1]
    model = build_r11_model(
        config["model"], saved["provenance"]["config_path"], project_root
    )
    load_checkpoint_model_state(model, saved)
    model.to(device).eval()
    stats = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in saved["stats"].items()
    }
    task_texts = saved["provenance"]["dataset_projection"]["task_texts"]
    return model, stats, task_texts, observed_sha256


def prepare_batch(
    observation: Mapping[str, Any],
    arms: tuple[int, ...],
    *,
    stats: Mapping[str, torch.Tensor],
    task: str,
    task_text: str,
    device: torch.device,
) -> dict[str, Any]:
    sensors = observation["sensor_data"]
    global_image = np.asarray(sensors["head_camera_global"]["rgb"])
    global_image = global_image[0] if global_image.ndim == 4 else global_image
    current, qposes = [], []
    for arm in arms:
        local_key = f"head_camera_agent{arm}"
        if local_key in sensors:
            local_image = np.asarray(sensors[local_key]["rgb"])
            local_image = local_image[0] if local_image.ndim == 4 else local_image
        elif task == "place_food":
            local_image = global_image
        else:
            raise KeyError(f"live observation is missing required camera {local_key}")
        qpos = np.asarray(observation["agent"][f"panda-{arm}"]["qpos"])
        qpos = qpos[0] if qpos.ndim == 2 else qpos
        if global_image.shape != (480, 640, 3) or local_image.shape != (480, 640, 3):
            raise ValueError(
                f"strict 640x480 RGB required, got {global_image.shape}/{local_image.shape}"
            )
        if qpos.shape != (9,):
            raise ValueError(f"strict 9D qpos required, got {qpos.shape}")
        current.append(np.stack((global_image, local_image)))
        qposes.append(qpos.astype(np.float32, copy=False))
    current_rgb = torch.as_tensor(np.stack(current)).permute(0, 1, 4, 2, 3)
    qpos = torch.as_tensor(np.stack(qposes), dtype=torch.float32, device=device)
    qpos = (qpos - stats["q_mean"]) / stats["q_std"]
    return {
        "current_rgb": current_rgb.contiguous().to(device=device, non_blocking=True),
        "qpos": qpos,
        "task": [task] * len(arms),
        "task_text": [task_text] * len(arms),
        "agent": torch.as_tensor(arms, dtype=torch.long, device=device),
        "objective_slot": torch.arange(len(arms), device=device),
    }


@torch.no_grad()
def predict_chunk(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    stats: Mapping[str, torch.Tensor],
    *,
    mode: str,
    device: torch.device,
) -> tuple[np.ndarray, int, float]:
    if mode not in INFERENCE_MODES:
        raise ValueError(mode)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with autocast:
        output = model(batch, mode=mode)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000
    action = output.get("action")
    if not isinstance(action, torch.Tensor) or action.shape[-2:] != (100, 8):
        raise ValueError("candidate did not return the common [B,100,8] action chunk")
    action = action.float() * stats["a_std"] + stats["a_mean"]
    if not torch.isfinite(action).all():
        raise FloatingPointError("candidate returned non-finite physical action")
    cadence = int(output.get("execution_cadence", 100))
    if not 1 <= cadence <= 100:
        raise ValueError(f"invalid candidate execution cadence: {cadence}")
    return action.cpu().numpy(), cadence, elapsed_ms


def _recover_rows(
    log_path: Path,
    *,
    task: str,
    mode: str,
    checkpoint_sha256: str,
    requested: list[int],
) -> list[dict[str, Any]]:
    if not log_path.is_file():
        return []
    allowed = set(requested)
    recovered: dict[int, dict[str, Any]] = {}
    for line in log_path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            row.get("event") == "episode"
            and row.get("task") == task
            and row.get("mode") == mode
            and row.get("checkpoint_sha256") == checkpoint_sha256
            and row.get("seed") in allowed
            and isinstance(row.get("success"), bool)
            and isinstance(row.get("steps"), int)
            and row.get("invalid_actions") == 0
            and row.get("fallback_calls") == 0
        ):
            recovered[int(row["seed"])] = row
    return [recovered[seed] for seed in requested if seed in recovered]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--task", choices=SIX_TASKS, required=True)
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--mode", choices=INFERENCE_MODES, default="normal")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--robofactory-root", default="/workspace/RoboFactory")
    parser.add_argument("--resume-log", default="")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def evaluate(
    args: argparse.Namespace,
    *,
    loaded: tuple[torch.nn.Module, dict[str, torch.Tensor], dict, str] | None = None,
) -> dict[str, Any]:
    if args.episodes < 1 or args.max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    device = torch.device(args.device)
    model, stats, task_texts, checkpoint_sha256 = (
        load_candidate(
            checkpoint,
            expected_sha256=args.checkpoint_sha256,
            device=device,
        )
        if loaded is None
        else loaded
    )
    if checkpoint_sha256 != args.checkpoint_sha256:
        raise ValueError("loaded candidate identity differs from requested checkpoint")
    seed_path = Path(args.seed_file).resolve(strict=True)
    seed_raw = seed_path.read_bytes()
    seed_manifest = json.loads(seed_raw)
    seeds = [int(value) for value in seed_manifest["seeds"]]
    if args.episodes > len(seeds):
        raise ValueError("requested episodes exceed frozen seed manifest")
    requested = seeds[: args.episodes]
    resume_log = Path(args.resume_log).resolve() if args.resume_log else None
    recovered = (
        _recover_rows(
            resume_log,
            task=args.task,
            mode=args.mode,
            checkpoint_sha256=checkpoint_sha256,
            requested=requested,
        )
        if resume_log
        else []
    )
    completed = {row["seed"] for row in recovered}
    specification = get_task(args.task)
    arms = tuple(specification["agents"])
    env = gym.make(
        specification["env_id"],
        config=str(Path(args.robofactory_root) / specification["config"]),
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="sensors",
        reward_mode="dense",
        sim_backend="cpu",
        sensor_configs=dict(shader_pack="default", width=640, height=480),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
    )
    rows = list(recovered)
    try:
        for seed in requested:
            if seed in completed:
                continue
            reset_episode = getattr(model, "reset_episode", None)
            if callable(reset_episode):
                reset_episode()
            observation, _ = reset_reproducibly(env, seed)
            chunk = None
            cadence = None
            plan_step = 0
            latencies = []
            success = False
            for step in range(args.max_steps):
                if chunk is None or plan_step >= cadence:
                    batch = prepare_batch(
                        observation,
                        arms,
                        stats=stats,
                        task=args.task,
                        task_text=task_texts[args.task],
                        device=device,
                    )
                    chunk, cadence, latency = predict_chunk(
                        model, batch, stats, mode=args.mode, device=device
                    )
                    if chunk.shape != (len(arms), 100, 8):
                        raise ValueError("candidate action batch does not match live agents")
                    latencies.append(latency)
                    plan_step = 0
                action = {
                    f"panda-{arm}": chunk[index, plan_step]
                    for index, arm in enumerate(arms)
                }
                if not all(np.isfinite(value).all() for value in action.values()):
                    raise FloatingPointError("illegal non-finite action before env.step")
                observation, _, terminated, truncated, info = env.step(action)
                plan_step += 1
                success = bool(np.asarray(info.get("success", False)).all())
                if bool(np.asarray(terminated).all()) or bool(np.asarray(truncated).all()):
                    break
            row = {
                "event": "episode",
                "task": args.task,
                "mode": args.mode,
                "seed": seed,
                "success": success,
                "steps": step + 1,
                "plans": len(latencies),
                "execution_cadence": cadence,
                "latencies_ms": latencies,
                "latency_ms_p50": float(np.percentile(latencies, 50)),
                "latency_ms_p95": float(np.percentile(latencies, 95)),
                "invalid_actions": 0,
                "fallback_calls": 0,
                "checkpoint_sha256": checkpoint_sha256,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        env.close()

    rows.sort(key=lambda row: requested.index(row["seed"]))
    if len(rows) != args.episodes or [row["seed"] for row in rows] != requested:
        raise RuntimeError("closed-loop result is incomplete or seed order drifted")
    all_latencies = [
        latency for row in rows for latency in row.get("latencies_ms", [])
    ]
    if not all_latencies:
        raise RuntimeError("closed-loop result contains no measured model calls")
    result = {
        "format_version": "before-we-act.r11.closed_loop/1",
        "status": "PASSED",
        "candidate": model.provenance["model"],
        "task": args.task,
        "mode": args.mode,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "seed_file": str(seed_path),
        "seed_file_sha256": hashlib.sha256(seed_raw).hexdigest(),
        "selection_method": seed_manifest.get("selection_method"),
        "episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / len(rows),
        "max_steps": args.max_steps,
        "execution_cadence": rows[0]["execution_cadence"],
        "latency_ms_p50": float(np.percentile(all_latencies, 50)),
        "latency_ms_p95": float(np.percentile(all_latencies, 95)),
        "invalid_actions": sum(row["invalid_actions"] for row in rows),
        "fallback_calls": sum(row["fallback_calls"] for row in rows),
        "peak_gpu_memory_gb": (
            round(torch.cuda.max_memory_allocated() / 2**30, 3)
            if device.type == "cuda"
            else 0.0
        ),
        "rows": rows,
        "completed_at_epoch": time.time(),
    }
    atomic_json(Path(args.output).resolve(), result)
    print(json.dumps(result | {"rows": "saved"}, sort_keys=True), flush=True)
    return result


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
