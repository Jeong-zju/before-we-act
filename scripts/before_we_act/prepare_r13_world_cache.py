#!/usr/bin/env python3
"""Materialize the one shared, target-separated W11+W12 cache for R13."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time

import torch
from torch.utils.data._utils.collate import default_collate

from before_we_act.action_generator.evolution import (
    TaskConditionedActionGenerator,
    load_r12_evolution_config,
)
from before_we_act.data.full_episode_windows import FULL_EPISODE_PROTOCOL, TASKS
from before_we_act.data.world_windows import R13SourceWindows
from before_we_act.team_belief.base import PredictiveBeliefModel, load_r11_config


TRAIN_ROWS = 4096
VALIDATION_ROWS = 1024
SEED = 20260806


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def balanced_requests(dataset: R13SourceWindows, count: int, seed: int):
    rng = random.Random(seed)
    buckets = {}
    for task_index in range(len(TASKS)):
        values = list(dataset.requests_by_task[task_index])
        rng.shuffle(values)
        buckets[task_index] = values
    offsets = [0] * len(TASKS)
    result = []
    for row in range(count):
        task = row % len(TASKS)
        values = buckets[task]
        result.append(values[offsets[task] % len(values)])
        offsets[task] += 1
    rng.shuffle(result)
    return result


def project_spatial(
    tokens: torch.Tensor, view_mask: torch.Tensor, projection: torch.Tensor
) -> torch.Tensor:
    # tokens [...,view,patch,768], mask [...,view]
    weights = view_mask[..., :, None, None].to(tokens.dtype)
    pooled = (tokens * weights).sum(dim=(-3, -2))
    denominator = weights.sum(dim=(-3, -2)).clamp_min(1) * tokens.shape[-2]
    pooled = pooled / denominator
    return pooled @ projection


@torch.inference_mode()
def materialize_split(
    dataset: R13SourceWindows,
    requests,
    *,
    belief: PredictiveBeliefModel,
    action: TaskConditionedActionGenerator,
    projection: torch.Tensor,
    device: torch.device,
    batch_size: int,
    heartbeat: Path | None,
    split: str,
) -> dict[str, torch.Tensor]:
    rows: dict[str, list[torch.Tensor]] = {}
    started = time.monotonic()
    for start in range(0, len(requests), batch_size):
        cpu = default_collate([dataset[request] for request in requests[start : start + batch_size]])
        batch = {key: value.to(device, non_blocking=True) for key, value in cpu.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            belief_output = belief(batch)["belief"]
            proposal = action.sample(
                belief_output,
                spatial_tokens=batch["spatial_tokens"],
                spatial_view_mask=batch["spatial_view_mask"],
                task_index=batch["task_index"],
            )
        current_latent = project_spatial(
            batch["current_spatial_tokens"].float(),
            batch["current_spatial_view_mask"],
            projection,
        )
        future_latent = project_spatial(
            batch["future_spatial_tokens"].float(),
            batch["future_spatial_view_mask"],
            projection,
        )
        future_qpos_delta = batch["future_qpos"].float() - batch["current_qpos"].float()[:, None]
        future_qpos_delta *= batch["agent_mask"][:, None, :, None].to(future_qpos_delta.dtype)
        values = {
            "belief_tokens": belief_output.tokens,
            "belief_agent_tokens": belief_output.agent_tokens,
            "belief_consensus": belief_output.consensus_token,
            "belief_uncertainty": belief_output.uncertainty,
            "agent_mask": belief_output.agent_mask,
            "candidate_actions": proposal.actions,
            "candidate_valid_mask": proposal.valid_mask,
            "current_latent": current_latent,
            "future_latent": future_latent[:, :, None],
            "future_qpos_delta": future_qpos_delta,
            "future_progress": batch["future_progress"],
            "future_failure": torch.zeros_like(batch["future_progress"]),
            "horizon_mask": batch["horizon_mask"],
            "task_index": batch["task_index"],
        }
        for key, value in values.items():
            value = value.detach().cpu()
            if value.is_floating_point():
                value = value.to(torch.float16)
            rows.setdefault(key, []).append(value)
        complete = min(start + batch_size, len(requests))
        if heartbeat:
            atomic_json(
                heartbeat,
                {
                    "producer": "prepare_r13_world_cache",
                    "pid": os.getpid(),
                    "stage": "PREPARING",
                    "split": split,
                    "rows": complete,
                    "total_rows": len(requests),
                    "elapsed_seconds": time.monotonic() - started,
                    "updated_at": now(),
                },
            )
        if complete % max(batch_size * 10, 1) == 0 or complete == len(requests):
            print(json.dumps({"split": split, "rows": complete, "total": len(requests)}), flush=True)
    return {key: torch.cat(value, dim=0) for key, value in rows.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--belief-config", required=True)
    parser.add_argument("--belief-checkpoint", required=True)
    parser.add_argument("--belief-sha256", required=True)
    parser.add_argument("--action-config", required=True)
    parser.add_argument("--action-checkpoint", required=True)
    parser.add_argument("--action-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--heartbeat")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        existing = torch.load(output, map_location="cpu", weights_only=False)
        if (
            existing.get("round") == "R13"
            and existing.get("metadata", {}).get("belief_checkpoint_sha256") == args.belief_sha256
            and existing.get("metadata", {}).get("action_checkpoint_sha256") == args.action_sha256
        ):
            print(json.dumps({"cache_reused": str(output), "sha256": sha256(output)}))
            return
        raise FileExistsError(f"refusing to overwrite a different cache: {output}")
    if args.batch_size < 1:
        raise ValueError("R13 cache batch size must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("R13 cache preparation requires CUDA")
    belief_path = Path(args.belief_checkpoint).resolve(strict=True)
    action_path = Path(args.action_checkpoint).resolve(strict=True)
    if sha256(belief_path) != args.belief_sha256 or sha256(action_path) != args.action_sha256:
        raise ValueError("R13 frozen W11/W12 checkpoint digest differs")
    belief_saved = torch.load(belief_path, map_location="cpu", weights_only=False)
    belief_config = load_r11_config(args.belief_config)
    belief = PredictiveBeliefModel(belief_config).to(device)
    belief.load_state_dict(belief_saved["model"], strict=True)
    belief.eval()
    action_saved = torch.load(action_path, map_location="cpu", weights_only=False)
    action_config = load_r12_evolution_config(args.action_config)
    action = TaskConditionedActionGenerator(action_config).to(device)
    action.load_state_dict(action_saved["model"], strict=True)
    action.eval()
    if int(action_saved.get("update", -1)) != int(action_config.training["updates"]):
        raise ValueError("R13 W12 checkpoint is not the frozen completed action generator")
    index_path = Path(args.index).resolve(strict=True)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("round") != "R12-R4" or index.get("protocol_variant") != FULL_EPISODE_PROTOCOL:
        raise ValueError("R13 source index differs from frozen R12 full-episode cache")
    stats = {key: torch.as_tensor(action_saved["stats"][key]) for key in ("a_mean", "a_std")}
    train_dataset = R13SourceWindows(index["episodes"], stats, split="train", cache_episodes=8)
    validation_dataset = R13SourceWindows(
        index["episodes"], stats, split="validation", cache_episodes=8
    )
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    random_projection = torch.randn((768, 96), generator=generator)
    projection, _ = torch.linalg.qr(random_projection, mode="reduced")
    projection = projection.to(device)
    projection_digest = hashlib.sha256(projection.cpu().numpy().tobytes()).hexdigest()
    heartbeat = Path(args.heartbeat).resolve() if args.heartbeat else None
    train = materialize_split(
        train_dataset,
        balanced_requests(train_dataset, TRAIN_ROWS, SEED),
        belief=belief,
        action=action,
        projection=projection,
        device=device,
        batch_size=args.batch_size,
        heartbeat=heartbeat,
        split="train",
    )
    validation = materialize_split(
        validation_dataset,
        balanced_requests(validation_dataset, VALIDATION_ROWS, SEED + 1),
        belief=belief,
        action=action,
        projection=projection,
        device=device,
        batch_size=args.batch_size,
        heartbeat=heartbeat,
        split="validation",
    )
    payload = {
        "schema_version": 1,
        "round": "R13",
        "metadata": {
            "created_at": now(),
            "seed": SEED,
            "train_rows": TRAIN_ROWS,
            "validation_rows": VALIDATION_ROWS,
            "tasks": list(TASKS),
            "source_index": str(index_path),
            "source_index_sha256": sha256(index_path),
            "belief_checkpoint": str(belief_path),
            "belief_checkpoint_sha256": args.belief_sha256,
            "action_checkpoint": str(action_path),
            "action_checkpoint_sha256": args.action_sha256,
            "action_units": "W12 normalized ActionProposalBatch before denormalization",
            "projection": "seeded_orthonormal_768_to_96",
            "projection_sha256": projection_digest,
            "prediction_horizons": [1, 5, 15],
            "future_targets_are_model_inputs": False,
            "failure_labels": "success_demonstrations_only_all_zero_no_AUROC_claim",
            "routing_note": (
                "Stack rows are deployed W12 specialist output; protected-task rows are "
                "off-path W12 specialist counterfactuals. Frozen deployed action equality "
                "is audited separately and world outputs never enter action selection."
            ),
        },
        "train": train,
        "validation": validation,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    if heartbeat:
        atomic_json(
            heartbeat,
            {
                "producer": "prepare_r13_world_cache",
                "pid": os.getpid(),
                "stage": "PREPARING",
                "status": "complete",
                "updated_at": now(),
            },
        )
    print(json.dumps({"cache": str(output), "sha256": sha256(output)}))


if __name__ == "__main__":
    main()
