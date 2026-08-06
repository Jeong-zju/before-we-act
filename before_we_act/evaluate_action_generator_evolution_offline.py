"""Full 22,475-timestep held-out evaluation for an R12-E1 specialist."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from before_we_act.action_generator.evolution import (
    TaskConditionedActionGenerator,
    load_r12_evolution_config,
)
from before_we_act.data.full_episode_windows import (
    FULL_EPISODE_PROTOCOL,
    FullEpisodeActionWindows,
    SequentialFullEpisodeSampler,
)
from before_we_act.data.raw_team_windows import TASKS
from before_we_act.team_belief.base import PredictiveBeliefModel, load_r11_config


EXPECTED_VALIDATION = {
    "lift_barrier": 1015,
    "camera_alignment": 1457,
    "three_robots_stack_cube": 6138,
    "long_pipeline_delivery": 10981,
    "take_photo": 2884,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--belief-config", required=True)
    parser.add_argument("--belief-checkpoint", required=True)
    parser.add_argument("--full-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--heartbeat", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("R12-E1 offline batch/workers differ")
    device = torch.device(args.device)
    config_path = Path(args.config).resolve(strict=True)
    config = load_r12_evolution_config(config_path)
    checkpoint_path = Path(args.checkpoint).resolve(strict=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("round") != "R12-E1"
        or checkpoint.get("candidate_id") != config.candidate_id
        or int(checkpoint.get("update", -1)) != int(config.training["updates"])
        or not checkpoint.get("core_free_runtime")
    ):
        raise ValueError("R12-E1 offline checkpoint identity differs")
    model = TaskConditionedActionGenerator(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    belief_config = load_r11_config(args.belief_config)
    belief_path = Path(args.belief_checkpoint).resolve(strict=True)
    if sha256(belief_path) != str(config.raw["belief_checkpoint_sha256"]):
        raise ValueError("R12-E1 offline W11 identity differs")
    belief_saved = torch.load(belief_path, map_location="cpu", weights_only=False)
    belief = PredictiveBeliefModel(belief_config).to(device)
    belief.load_state_dict(belief_saved["model"], strict=True)
    belief.eval()

    index_path = Path(args.full_index).resolve(strict=True)
    index = json.loads(index_path.read_text())
    if (
        index.get("schema_version") != 1
        or index.get("round") != "R12-R4"
        or index.get("protocol_variant") != FULL_EPISODE_PROTOCOL
        or index.get("step_counts", {}).get("validation") != EXPECTED_VALIDATION
    ):
        raise ValueError("R12-E1 full validation index differs")
    stats = {
        key: torch.as_tensor(checkpoint["stats"][key], dtype=torch.float32)
        for key in ("a_mean", "a_std")
    }
    dataset = FullEpisodeActionWindows(
        index["episodes"], stats, split="validation", cache_episodes=8
    )
    if len(dataset) != sum(EXPECTED_VALIDATION.values()):
        raise ValueError("R12-E1 validation is not full-timestep")
    loader = DataLoader(
        dataset,
        batch_sampler=SequentialFullEpisodeSampler(dataset, args.batch_size),
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    heartbeat = Path(args.heartbeat).resolve() if args.heartbeat else None
    totals = {
        task: {
            "rows": 0,
            "first_squared_error": 0.0,
            "first_elements": 0,
            "full_squared_error": 0.0,
            "full_elements": 0,
        }
        for task in TASKS
    }
    observed, finite = 0, True
    started = time.monotonic()
    for batch_index, cpu_batch in enumerate(loader):
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in cpu_batch.items()
        }
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            belief_state = belief(batch)["belief"]
            generator = torch.Generator(device=device).manual_seed(
                args.seed + 1_000_003 * batch_index
            )
            noise = torch.randn(
                (len(batch["joint_actions"]), 100, 32),
                generator=generator,
                device=device,
            )
            proposals = model.sample(
                belief_state,
                spatial_tokens=batch["spatial_tokens"],
                spatial_view_mask=batch["spatial_view_mask"],
                task_index=batch["task_index"],
                noise=noise,
            )
        predicted = proposals.actions[:, 0].permute(0, 2, 1, 3).float()
        target = batch["joint_actions"].float()
        active = batch["agent_mask"][:, None, :, None]
        valid = active & batch["action_step_mask"][:, :, None, None]
        finite = finite and bool(
            torch.isfinite(predicted[valid.expand_as(predicted)]).all()
        )
        error = (predicted - target).square()
        for task_index, task in enumerate(TASKS):
            selected = batch["task_index"].eq(task_index)
            if not bool(selected.any()):
                continue
            selected_error = error[selected]
            selected_active = active[selected]
            selected_valid = valid[selected]
            first_mask = selected_active[:, 0].expand_as(selected_error[:, 0])
            full_mask = selected_valid.expand_as(selected_error)
            row = totals[task]
            row["rows"] += int(selected.sum())
            row["first_squared_error"] += float(
                selected_error[:, 0][first_mask].sum().cpu()
            )
            row["first_elements"] += int(first_mask.sum())
            row["full_squared_error"] += float(
                selected_error[full_mask].sum().cpu()
            )
            row["full_elements"] += int(full_mask.sum())
        observed += len(target)
        if heartbeat and (batch_index == 0 or batch_index % 50 == 0):
            elapsed = time.monotonic() - started
            atomic_json(
                heartbeat,
                {
                    "producer": "evaluate_action_generator_evolution_offline",
                    "candidate": config.candidate_id,
                    "pid": os.getpid(),
                    "rows": observed,
                    "total_rows": len(dataset),
                    "eta_seconds": (len(dataset) - observed)
                    * elapsed
                    / max(observed, 1),
                    "updated_at": now(),
                },
            )
    per_task = {
        task: {
            "rows": row["rows"],
            "first_step_normalized_mse": row["first_squared_error"]
            / row["first_elements"],
            "full_chunk_normalized_mse": row["full_squared_error"]
            / row["full_elements"],
        }
        for task, row in totals.items()
    }
    first_squared = sum(row["first_squared_error"] for row in totals.values())
    first_elements = sum(row["first_elements"] for row in totals.values())
    full_squared = sum(row["full_squared_error"] for row in totals.values())
    full_elements = sum(row["full_elements"] for row in totals.values())
    result = {
        "schema_version": 1,
        "round": "R12-E1",
        "candidate_id": config.candidate_id,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "update": int(checkpoint["update"]),
        "full_index": str(index_path),
        "full_index_sha256": sha256(index_path),
        "validation_rows": observed,
        "expected_validation_rows": sum(EXPECTED_VALIDATION.values()),
        "all_outputs_finite": finite,
        "first_step_normalized_mse": first_squared / first_elements,
        "full_chunk_normalized_mse": full_squared / full_elements,
        "per_task": per_task,
        "observation": "native 480x640 RGB primary; W11 and task ID supplemental",
        "elapsed_seconds": time.monotonic() - started,
        "created_at": now(),
    }
    if not finite or observed != sum(EXPECTED_VALIDATION.values()):
        raise ValueError("R12-E1 full validation failed completeness/finite checks")
    atomic_json(Path(args.output), result)
    if heartbeat:
        atomic_json(
            heartbeat,
            {
                "producer": "evaluate_action_generator_evolution_offline",
                "candidate": config.candidate_id,
                "state": "PASSED",
                "rows": observed,
                "updated_at": now(),
            },
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
