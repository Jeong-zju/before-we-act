#!/usr/bin/env python3
"""Build a resumable four-GPU DINOv3 spatial cache aligned to R12 rows."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from before_we_act.data.raw_team_windows import TASKS, manifest_receipt
from before_we_act.spatial_observation import (
    R12SpatialObservationEncoder,
    locked_r12_spatial_observation,
)
from scripts.before_we_act.prepare_r12_action_cache import choose_examples, patch_means


PROTOCOL = "current_dinov3_vitb16_4x4_per_fixed_view_v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifests(data_root: Path) -> dict:
    return {
        task: json.loads(
            (data_root / task / "training_manifest.json").read_text(encoding="utf-8")
        )
        for task in TASKS
    }


def split_examples(manifests: dict, action_metadata: dict, split: str):
    count = int(action_metadata[f"{split}_interior_per_episode"])
    return choose_examples(manifests, split, count, int(action_metadata["seed"]))


def grouped_rows(examples):
    groups = {}
    for row_index, (task, episode, current) in enumerate(examples):
        key = (task, episode["hdf5_path"])
        group = groups.setdefault(
            key,
            {"task": task, "episode": episode, "rows": []},
        )
        group["rows"].append((row_index, int(current)))
    return list(groups.values())


@torch.inference_mode()
def encode_group(
    encoder: R12SpatialObservationEncoder,
    data_root: Path,
    group: dict,
    device: torch.device,
    image_batch_size: int,
):
    task = group["task"]
    episode = group["episode"]
    rows = group["rows"]
    path = data_root / task / episode["hdf5_path"]
    count = len(rows)
    features = torch.zeros((count, 5, 16, 768), dtype=torch.float16)
    current_visual = torch.zeros((count, 16, 15), dtype=torch.float16)
    view_mask = torch.zeros((count, 5), dtype=torch.bool)
    requests = []
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        agents = sorted(data["observation/agents"].keys())
        views = ["global"] + [f"agent_{index}" for index in range(len(agents))]
        for local_row, (_global_row, current) in enumerate(rows):
            for view_index, view in enumerate(views):
                image = np.asarray(data[f"observation/images/{view}"][current])
                if tuple(image.shape) != (480, 640, 3):
                    raise ValueError(f"unexpected fixed-view RGB shape at {path}:{current}")
                current_visual[
                    local_row, :, view_index * 3 : (view_index + 1) * 3
                ] = torch.from_numpy(patch_means(image)).to(torch.float16)
                view_mask[local_row, view_index] = True
                requests.append((local_row, view_index, image))
        for start in range(0, len(requests), image_batch_size):
            batch = requests[start : start + image_batch_size]
            images = torch.from_numpy(
                np.stack([item[2].transpose(2, 0, 1) for item in batch])
            ).to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                tokens = encoder.encoder.forward_spatial_grid(
                    images, grid_height=4, grid_width=4
                ).spatial_tokens
            tokens = tokens.to(device="cpu", dtype=torch.float16)
            for offset, (local_row, view_index, _image) in enumerate(batch):
                features[local_row, view_index] = tokens[offset]
    row_indices = torch.tensor([row for row, _current in rows], dtype=torch.long)
    progress = torch.tensor(
        [current / max(int(episode["steps"]) - 1, 1) for _row, current in rows],
        dtype=torch.float32,
    )
    task_index = torch.full((count,), TASKS.index(task), dtype=torch.long)
    return {
        "row_indices": row_indices,
        "spatial_tokens": features,
        "spatial_view_mask": view_mask,
        "current_visual": current_visual,
        "progress": progress,
        "task_index": task_index,
    }


def run_shard(args) -> None:
    if not (0 <= args.rank < args.world_size):
        raise ValueError("rank must be in [0, world_size)")
    output = args.shard_dir / f"rank_{args.rank:02d}.pt"
    if output.is_file():
        saved = torch.load(output, map_location="cpu", weights_only=False)
        if (
            saved.get("protocol_variant") != PROTOCOL
            or saved.get("rank") != args.rank
            or saved.get("world_size") != args.world_size
        ):
            raise ValueError("existing R12-R3 spatial shard identity differs")
        print(json.dumps({"reused": str(output), "rank": args.rank}))
        return
    data_root = args.data_root.resolve(strict=True)
    action = torch.load(args.action_cache, map_location="cpu", weights_only=False)
    metadata = action["metadata"]
    manifests = load_manifests(data_root)
    if metadata.get("manifest_sha256") != manifest_receipt(data_root):
        raise ValueError("R12 action cache and raw manifests differ")
    device = torch.device(args.device)
    encoder = R12SpatialObservationEncoder(
        locked_r12_spatial_observation(),
        args.vision_artifact,
        inference_batch_size=args.image_batch_size,
    ).to(device).eval()
    result = {}
    completed_groups = 0
    total_groups = 0
    for split in ("train", "validation"):
        examples = split_examples(manifests, metadata, split)
        groups = grouped_rows(examples)
        assigned = [
            group for index, group in enumerate(groups) if index % args.world_size == args.rank
        ]
        total_groups += len(assigned)
        chunks = []
        for group in assigned:
            chunks.append(
                encode_group(
                    encoder,
                    data_root,
                    group,
                    device,
                    args.image_batch_size,
                )
            )
            completed_groups += 1
            atomic_json(
                args.heartbeat,
                {
                    "producer": "prepare_r12_spatial_cache",
                    "rank": args.rank,
                    "split": split,
                    "completed_groups": completed_groups,
                    "updated_at": now(),
                },
            )
            atomic_json(
                args.state,
                {
                    "state": "PREPARING",
                    "stage": "dinov3_spatial_shard",
                    "rank": args.rank,
                    "completed_groups": completed_groups,
                    "total_groups": total_groups,
                    "updated_at": now(),
                },
            )
        result[split] = {
            key: torch.cat([chunk[key] for chunk in chunks], dim=0)
            for key in chunks[0]
        }
    payload = {
        "schema_version": 1,
        "round": "R12-R3",
        "protocol_variant": PROTOCOL,
        "rank": args.rank,
        "world_size": args.world_size,
        "observation": locked_r12_spatial_observation(),
        **result,
    }
    args.shard_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    atomic_json(
        args.state,
        {
            "state": "PASSED",
            "stage": "spatial_shard_complete",
            "rank": args.rank,
            "output": str(output),
            "sha256": sha256(output),
            "updated_at": now(),
        },
    )
    atomic_json(args.heartbeat, {"rank": args.rank, "complete": True, "updated_at": now()})


def run_consolidate(args) -> None:
    if args.output.is_file():
        saved = torch.load(args.output, map_location="cpu", weights_only=False)
        if saved.get("metadata", {}).get("protocol_variant") != PROTOCOL:
            raise ValueError("existing R12-R3 spatial cache identity differs")
        print(json.dumps({"reused": str(args.output), "sha256": sha256(args.output)}))
        return
    shard_paths = [args.shard_dir / f"rank_{rank:02d}.pt" for rank in range(args.world_size)]
    while not all(path.is_file() for path in shard_paths):
        atomic_json(
            args.heartbeat,
            {
                "producer": "prepare_r12_spatial_cache_consolidate",
                "ready_shards": sum(path.is_file() for path in shard_paths),
                "total_shards": len(shard_paths),
                "updated_at": now(),
            },
        )
        time.sleep(10)
    action_path = args.action_cache.resolve(strict=True)
    action = torch.load(action_path, map_location="cpu", weights_only=False)
    output_splits = {}
    for split in ("train", "validation"):
        size = len(action[split]["task_index"])
        values = {
            "spatial_tokens": torch.empty((size, 5, 16, 768), dtype=torch.float16),
            "spatial_view_mask": torch.empty((size, 5), dtype=torch.bool),
            "progress": torch.empty((size,), dtype=torch.float32),
            "task_index": torch.empty((size,), dtype=torch.long),
        }
        visuals = torch.empty((size, 16, 15), dtype=torch.float16)
        seen = torch.zeros(size, dtype=torch.bool)
        for path in shard_paths:
            shard = torch.load(path, map_location="cpu", weights_only=False)
            if (
                shard.get("protocol_variant") != PROTOCOL
                or shard.get("world_size") != args.world_size
            ):
                raise ValueError(f"spatial shard identity differs: {path}")
            rows = shard[split]["row_indices"]
            if bool(seen.index_select(0, rows).any()):
                raise ValueError("duplicate rows across R12-R3 spatial shards")
            seen[rows] = True
            for key in values:
                values[key].index_copy_(0, rows, shard[split][key])
            visuals.index_copy_(0, rows, shard[split]["current_visual"])
        if not bool(seen.all()):
            raise ValueError("R12-R3 spatial shards do not cover every action row")
        if not torch.equal(visuals, action[split]["visual"][:, -1]):
            raise ValueError("raw RGB spatial cache is not row-aligned to action cache")
        expected_mask = action[split]["view_mask"][:, -1].bool()
        if not torch.equal(values["spatial_view_mask"], expected_mask):
            raise ValueError("R12-R3 view masks are not row-aligned")
        if not torch.equal(values["task_index"], action[split]["task_index"]):
            raise ValueError("R12-R3 task buckets are not row-aligned")
        output_splits[split] = values
        atomic_json(
            args.heartbeat,
            {"producer": "prepare_r12_spatial_cache_consolidate", "split": split, "updated_at": now()},
        )
    payload = {
        "schema_version": 1,
        "round": "R12-R3",
        "metadata": {
            "created_at": now(),
            "protocol_variant": PROTOCOL,
            "action_cache": str(action_path),
            "action_cache_sha256": sha256(action_path),
            "row_alignment": "exact_recomputed_current_patch_means_and_view_mask",
            "observation": locked_r12_spatial_observation(),
            "legal_inputs": "current fixed-view raw RGB only",
            "forbidden_inputs": "future RGB, task/robot ID, commanded action, simulator state, W10 hidden state",
            "train_windows": len(output_splits["train"]["task_index"]),
            "validation_windows": len(output_splits["validation"]["task_index"]),
            "world_size": args.world_size,
            "shard_sha256": {path.name: sha256(path) for path in shard_paths},
        },
        **output_splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    atomic_json(
        args.state,
        {
            "state": "PASSED",
            "stage": "spatial_cache_complete",
            "output": str(args.output),
            "sha256": sha256(args.output),
            "updated_at": now(),
        },
    )
    atomic_json(args.heartbeat, {"complete": True, "updated_at": now()})
    print(json.dumps(json.loads(args.state.read_text()), sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("shard", "consolidate"), required=True)
    parser.add_argument("--action-cache", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--vision-artifact", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.world_size < 1 or args.image_batch_size < 1:
        raise ValueError("world size and image batch size must be positive")
    if args.mode == "shard":
        if args.data_root is None or args.vision_artifact is None:
            raise ValueError("shard mode requires data root and vision artifact")
        run_shard(args)
    else:
        if args.output is None:
            raise ValueError("consolidate mode requires output")
        run_consolidate(args)


if __name__ == "__main__":
    main()
