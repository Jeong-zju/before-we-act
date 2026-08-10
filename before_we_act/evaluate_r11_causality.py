"""Frozen offline future/action causal probes for one R11 checkpoint."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from before_we_act.r11_data import (
    ExactSixTaskAccumulationSampler,
    R11EpisodeDataset,
    SIX_TASKS,
    load_r11_episodes,
)
from before_we_act.train_r11_candidate import (
    atomic_json,
    capture_rng,
    move_batch,
    restore_rng,
    sha256_file,
    training_contract,
)


PROBE_SEED = 20260811
PROBE_UPDATES = 4
SAMPLES_PER_TASK = 32


def _load_model(checkpoint: Path, expected_sha256: str, device: torch.device):
    observed = sha256_file(checkpoint)
    if observed != expected_sha256:
        raise ValueError("causal probe checkpoint SHA256 differs")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if saved.get("format_version") != "before-we-act.r11.checkpoint/1":
        raise ValueError("unsupported R11 checkpoint format")
    config = saved["config"]
    from before_we_act.r11_registry import build_r11_model

    project_root = Path(__file__).resolve().parents[1]
    model = build_r11_model(
        config["model"], saved["provenance"]["config_path"], project_root
    )
    model.load_state_dict(saved["model"], strict=True)
    model.to(device).eval()
    return model, saved, config, observed


def _reset_model(model: torch.nn.Module) -> None:
    reset = getattr(model, "reset_episode", None)
    if callable(reset):
        reset()


def _masked_representation_sse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[float, float]:
    if prediction.shape != target.shape:
        raise ValueError(
            f"causal representation shape drift: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    error = (prediction.float() - target.float()).square()
    if mask is None:
        return float(error.sum().cpu()), float(error.numel())
    mask = mask.to(device=error.device, dtype=error.dtype)
    if prediction.shape[: mask.ndim] != mask.shape:
        raise ValueError("future representation mask does not match prediction prefix")
    expanded = mask.reshape(*mask.shape, *([1] * (prediction.ndim - mask.ndim)))
    elements_per_mask = math.prod(prediction.shape[mask.ndim:])
    return float((error * expanded).sum().cpu()), float(mask.sum().cpu()) * elements_per_mask


def _action_sse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[float, float]:
    if prediction.shape != target.shape or prediction.shape[-2:] != (100, 8):
        raise ValueError("action causal probe requires matching [B,100,8] tensors")
    per = (prediction.float() - target.float()).square()
    expanded = mask.to(per.device, per.dtype).unsqueeze(-1)
    return float((per * expanded).sum().cpu()), float(expanded.sum().cpu()) * 8


def _probe_batch(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, tuple[float, float]]:
    def autocast_context():
        return (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
    rng = capture_rng()
    _reset_model(model)
    with torch.no_grad(), autocast_context():
        normal_future = model.causal_probe(
            batch, action_condition_mode="normal"
        )
    restore_rng(rng)
    _reset_model(model)
    with torch.no_grad(), autocast_context():
        shuffled_future = model.causal_probe(
            batch, action_condition_mode="action_shuffled"
        )
    for key in ("future_prediction", "future_target", "persistence_prediction"):
        if not isinstance(normal_future.get(key), torch.Tensor):
            raise TypeError(f"candidate causal_probe is missing tensor {key}")
    mask = normal_future.get("future_mask")
    if mask is not None and not isinstance(mask, torch.Tensor):
        raise TypeError("future_mask must be a tensor or None")
    values = {
        "future_normal": _masked_representation_sse(
            normal_future["future_prediction"], normal_future["future_target"], mask
        ),
        "future_persistence": _masked_representation_sse(
            normal_future["persistence_prediction"], normal_future["future_target"], mask
        ),
        "future_action_shuffled": _masked_representation_sse(
            shuffled_future["future_prediction"], normal_future["future_target"], mask
        ),
    }

    # The deployment action path must never see demonstration actions or
    # future supervision. They remain outside the model call as metric targets.
    deployment_keys = {
        "current_rgb", "qpos", "task", "task_text", "agent", "objective_slot"
    }
    deployment_batch = {
        key: value for key, value in batch.items() if key in deployment_keys
    }
    action_outputs = {}
    action_rng = capture_rng()
    for mode in ("normal", "prediction_off", "prediction_shuffled"):
        restore_rng(action_rng)
        _reset_model(model)
        with torch.no_grad(), autocast_context():
            output = model(deployment_batch, mode=mode)
        action = output.get("action")
        if not isinstance(action, torch.Tensor):
            raise TypeError(f"{mode} did not return action tensor")
        action_outputs[mode] = _action_sse(
            action, batch["action"], batch["action_mask"]
        )
    values.update({f"action_{key}": value for key, value in action_outputs.items()})
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--enforce", action="store_true")
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    device = torch.device(args.device)
    model, saved, config, checkpoint_sha256 = _load_model(
        checkpoint, args.checkpoint_sha256, device
    )
    if not callable(getattr(model, "causal_probe", None)):
        raise TypeError("candidate does not implement the required causal_probe path")
    episodes = load_r11_episodes(args.manifests, split="validation")
    dataset = R11EpisodeDataset(episodes, saved["stats"])
    contract = training_contract(config)
    sampler = ExactSixTaskAccumulationSampler(
        episodes,
        updates=PROBE_UPDATES,
        seed=PROBE_SEED,
        micro_batch_size=contract["micro_batch_size"],
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    totals: dict[str, dict[str, list[float]]] = {
        task: defaultdict(lambda: [0.0, 0.0]) for task in SIX_TASKS
    }
    sample_counts = defaultdict(int)
    started = time.time()
    for micro_index, raw_batch in enumerate(loader):
        batch = move_batch(raw_batch, device)
        within = micro_index % sampler.accumulation_steps
        batch["objective_slot"] = torch.arange(
            within * contract["micro_batch_size"],
            (within + 1) * contract["micro_batch_size"],
            device=device,
        )
        tasks = list(raw_batch["task"])
        if len(set(tasks)) != 1:
            # The exact sampler is globally shuffled. Attribute mixed batches by
            # re-running one sample at a time; this is also the fit-safe path for
            # heterogeneous latent shapes in third-party adapters.
            for index, task in enumerate(tasks):
                single = {
                    key: (
                        value[index : index + 1]
                        if isinstance(value, torch.Tensor)
                        else type(value)((value[index],))
                        if isinstance(value, (list, tuple))
                        else value
                    )
                    for key, value in batch.items()
                }
                single["objective_slot"] = torch.zeros(1, dtype=torch.long, device=device)
                one = _probe_batch(model, single, device=device)
                for name, (sse, count) in one.items():
                    totals[task][name][0] += sse
                    totals[task][name][1] += count
                sample_counts[task] += 1
        else:
            task = tasks[0]
            values = _probe_batch(model, batch, device=device)
            for name, (sse, count) in values.items():
                totals[task][name][0] += sse
                totals[task][name][1] += count
            sample_counts[task] += len(tasks)
    if dict(sample_counts) != {task: SAMPLES_PER_TASK for task in SIX_TASKS}:
        raise RuntimeError(f"causal probe sample counts drifted: {dict(sample_counts)}")

    task_rows = {}
    for task in SIX_TASKS:
        mse = {
            name: sse / max(count, 1.0)
            for name, (sse, count) in totals[task].items()
        }
        normal_future = mse["future_normal"]
        persistence = mse["future_persistence"]
        shuffled_future = mse["future_action_shuffled"]
        action_normal = math.sqrt(mse["action_normal"])
        action_off = math.sqrt(mse["action_prediction_off"])
        action_shuffled = math.sqrt(mse["action_prediction_shuffled"])
        task_rows[task] = {
            "samples": sample_counts[task],
            "future_error": normal_future,
            "persistence_error": persistence,
            "prediction_gain": 1.0 - normal_future / max(persistence, 1e-12),
            "action_shuffled_future_error": shuffled_future,
            "action_shuffle_degradation": shuffled_future / max(normal_future, 1e-12) - 1.0,
            "action_nrmse": action_normal,
            "prediction_off_action_nrmse": action_off,
            "prediction_shuffled_action_nrmse": action_shuffled,
            "prediction_off_action_degradation": action_off / max(action_normal, 1e-12) - 1.0,
            "prediction_shuffled_action_degradation": action_shuffled / max(action_normal, 1e-12) - 1.0,
        }
    macro_gain = sum(row["prediction_gain"] for row in task_rows.values()) / len(SIX_TASKS)
    persistence_improved = sum(row["prediction_gain"] > 0 for row in task_rows.values())
    action_shuffle_tasks = sum(
        row["action_shuffle_degradation"] >= 0.05 for row in task_rows.values()
    )
    off_degradation = sum(
        row["prediction_off_action_degradation"] for row in task_rows.values()
    ) / len(SIX_TASKS)
    shuffled_degradation = sum(
        row["prediction_shuffled_action_degradation"] for row in task_rows.values()
    ) / len(SIX_TASKS)
    checks = [
        {
            "id": "future_vs_persistence",
            "passed": macro_gain >= 0.05 and persistence_improved >= 4,
            "macro_prediction_gain": macro_gain,
            "tasks_improved": persistence_improved,
            "threshold": {"macro": 0.05, "tasks": 4},
        },
        {
            "id": "action_shuffle_to_future",
            "passed": action_shuffle_tasks >= 4,
            "tasks_degraded_at_least_5pct": action_shuffle_tasks,
            "threshold_tasks": 4,
        },
        {
            "id": "prediction_to_action_offline",
            "passed": max(off_degradation, shuffled_degradation) >= 0.02,
            "prediction_off_macro_degradation": off_degradation,
            "prediction_shuffled_macro_degradation": shuffled_degradation,
            "threshold": 0.02,
            "hard_task_validation5_alternative_pending": True,
        },
    ]
    result = {
        "format_version": "before-we-act.r11.causal_acceptance/1",
        "status": "PASSED" if all(check["passed"] for check in checks) else "FAILED",
        "candidate": config["candidate"],
        "model": config["model"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "probe_seed": PROBE_SEED,
        "probe_updates": PROBE_UPDATES,
        "samples_per_task": SAMPLES_PER_TASK,
        "split": "validation",
        "task_text_held_fixed": True,
        "macro_prediction_gain": macro_gain,
        "persistence_tasks_improved": persistence_improved,
        "action_shuffle_tasks_passed": action_shuffle_tasks,
        "prediction_off_action_degradation": off_degradation,
        "prediction_shuffled_action_degradation": shuffled_degradation,
        "checks": checks,
        "tasks": task_rows,
        "duration_seconds": time.time() - started,
        "completed_at_epoch": time.time(),
    }
    atomic_json(Path(args.output).resolve(), result)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.enforce and result["status"] != "PASSED":
        raise SystemExit(10)
    return result


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
