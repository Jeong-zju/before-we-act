#!/usr/bin/env python3
"""Build non-overwriting full-timestep features from native 480x640 RGB."""
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

from before_we_act.data.full_episode_windows import FULL_EPISODE_PROTOCOL
from before_we_act.data.raw_team_windows import TASKS, manifest_receipt
from before_we_act.spatial_observation import (
    R12SpatialObservationEncoder,
    locked_r12_full_episode_observation,
)


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
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_hdf5_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    with h5py.File(temporary, "w") as handle:
        handle.attrs["schema_version"] = int(payload["schema_version"])
        handle.attrs["round"] = str(payload["round"])
        handle.attrs["metadata_json"] = json.dumps(
            payload["metadata"], sort_keys=True
        )
        for key, value in payload.items():
            if key in ("schema_version", "round", "metadata"):
                continue
            array = value.detach().cpu().numpy()
            handle.create_dataset(
                key,
                data=array,
                chunks=(True if array.ndim > 1 else None),
            )
        handle.flush()
    os.replace(temporary, path)


def patch_means(image: np.ndarray, grid: int = 4) -> np.ndarray:
    if tuple(image.shape) != (480, 640, 3):
        raise ValueError(f"fixed RGB shape differs: {tuple(image.shape)}")
    value = image.reshape(grid, 480 // grid, grid, 640 // grid, 3)
    return (
        value.mean(axis=(1, 3), dtype=np.float32).reshape(grid * grid, 3)
        / 127.5
        - 1.0
    )


def load_episode_rows(data_root: Path) -> list[dict]:
    rows = []
    for task in TASKS:
        manifest_path = data_root / task / "training_manifest.json"
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
        for episode in manifest["episodes"]:
            if episode["split"] not in ("train", "validation"):
                continue
            rows.append(
                {
                    "task": task,
                    "split": episode["split"],
                    "seed": int(episode["seed"]),
                    "steps": int(episode["steps"]),
                    "hdf5_path": str(data_root / task / episode["hdf5_path"]),
                    "hdf5_sha256": str(episode["hdf5_sha256"]),
                    "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                    "episode_index": int(episode["episode_index"]),
                }
            )
    return rows


@torch.inference_mode()
def encode_episode(args, encoder, row: dict, output: Path) -> dict:
    path = Path(row["hdf5_path"]).resolve(strict=True)
    steps = int(row["steps"])
    spatial = torch.zeros((steps, 5, 48, 768), dtype=torch.float16)
    visual = torch.zeros((steps, 16, 15), dtype=torch.float16)
    view_mask = torch.zeros((steps, 5), dtype=torch.bool)
    qpos = torch.zeros((steps, 4, 9), dtype=torch.float32)
    executed = torch.zeros((steps, 4, 8), dtype=torch.float32)
    commanded = torch.zeros((steps, 4, 8), dtype=torch.float32)
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        agents = sorted(data["observation/agents"].keys())
        if not 1 <= len(agents) <= 4:
            raise ValueError(f"unsupported agent count at {path}")
        views = ["global"] + [f"agent_{index}" for index in range(len(agents))]
        agent_mask = torch.arange(4) < len(agents)
        for agent_index, agent in enumerate(agents):
            qpos[:, agent_index] = torch.from_numpy(
                np.asarray(data[f"observation/agents/{agent}/qpos"][:steps])
            )
            executed[:, agent_index] = torch.from_numpy(
                np.asarray(data[f"action/agents/{agent}/executed"][:steps])
            )
            commanded[:, agent_index] = torch.from_numpy(
                np.asarray(data[f"action/agents/{agent}/commanded"][:steps])
            )
        for start in range(0, steps, args.frame_batch_size):
            end = min(steps, start + args.frame_batch_size)
            requests = []
            for frame in range(start, end):
                for view_index, view in enumerate(views):
                    image = np.asarray(data[f"observation/images/{view}"][frame])
                    if tuple(image.shape) != (480, 640, 3):
                        raise ValueError(f"native RGB shape differs at {path}:{frame}")
                    visual[frame, :, view_index * 3 : (view_index + 1) * 3] = (
                        torch.from_numpy(patch_means(image)).to(torch.float16)
                    )
                    view_mask[frame, view_index] = True
                    requests.append((frame, view_index, image))
            images = torch.from_numpy(
                np.stack([item[2].transpose(2, 0, 1) for item in requests])
            ).to(args.device, non_blocking=True)
            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=str(args.device).startswith("cuda"),
            ):
                # The encoder first sees all 480x640 pixels and emits its native
                # 30x40 patch grid.  forward_spatial_grid pools only afterwards.
                tokens = encoder.encoder.forward_spatial_grid(
                    images, grid_height=6, grid_width=8
                ).spatial_tokens
            tokens = tokens.cpu().to(torch.float16)
            for offset, (frame, view_index, _image) in enumerate(requests):
                spatial[frame, view_index] = tokens[offset]
            atomic_json(
                args.heartbeat,
                {
                    "producer": "prepare_r12_native_rgb_full_cache",
                    "rank": args.rank,
                    "task": row["task"],
                    "split": row["split"],
                    "episode_index": row["episode_index"],
                    "frame": end,
                    "episode_steps": steps,
                    "updated_at": now(),
                },
            )
    payload = {
        "schema_version": 1,
        "round": "R12-R4",
        "metadata": {
            "created_at": now(),
            "protocol_variant": FULL_EPISODE_PROTOCOL,
            "task": row["task"],
            "split": row["split"],
            "seed": row["seed"],
            "steps": steps,
            "episode_index": row["episode_index"],
            "source_hdf5": str(path),
            "hdf5_sha256": row["hdf5_sha256"],
            "manifest_sha256": row["manifest_sha256"],
            "observation": locked_r12_full_episode_observation(),
            "legal_inputs": "native 480x640 current fixed-view RGB plus causal qpos/executed-action history",
            "forbidden_inputs": "future RGB, task/robot ID at policy input, simulator state, W10 hidden state",
        },
        "visual": visual,
        "view_mask": view_mask,
        "qpos": qpos,
        "executed_actions": executed,
        "commanded_actions": commanded,
        "agent_mask": agent_mask,
        "spatial_tokens": spatial,
        "spatial_view_mask": view_mask.clone(),
    }
    atomic_hdf5_save(output, payload)
    return {
        "path": str(output.resolve()),
        "sha256": sha256(output),
        "size_bytes": output.stat().st_size,
        **{
            key: row[key]
            for key in (
                "task",
                "split",
                "seed",
                "steps",
                "hdf5_sha256",
                "episode_index",
            )
        },
    }


def existing_record(output: Path, row: dict) -> dict:
    with h5py.File(output, "r") as saved:
        metadata = json.loads(str(saved.attrs["metadata_json"]))
    if (
        metadata.get("protocol_variant") != FULL_EPISODE_PROTOCOL
        or metadata.get("hdf5_sha256") != row["hdf5_sha256"]
        or int(metadata.get("steps", -1)) != int(row["steps"])
    ):
        raise ValueError(f"existing full-episode output differs: {output}")
    return {
        "path": str(output.resolve()),
        "sha256": sha256(output),
        "size_bytes": output.stat().st_size,
        **{
            key: row[key]
            for key in (
                "task",
                "split",
                "seed",
                "steps",
                "hdf5_sha256",
                "episode_index",
            )
        },
    }


def run_shard(args) -> None:
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be in [0, world-size)")
    data_root = args.data_root.resolve(strict=True)
    rows = load_episode_rows(data_root)
    assigned = [row for index, row in enumerate(rows) if index % args.world_size == args.rank]
    receipt_path = args.output_root / f"rank_{args.rank:02d}_index.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt.get("protocol_variant") != FULL_EPISODE_PROTOCOL
            or receipt.get("rank") != args.rank
            or receipt.get("world_size") != args.world_size
            or not all(Path(row["path"]).is_file() for row in receipt.get("episodes", []))
        ):
            raise ValueError("existing full-episode rank receipt differs")
        print(json.dumps({"reused": str(receipt_path), "episodes": len(receipt["episodes"])}))
        return
    observation = locked_r12_full_episode_observation()
    encoder = R12SpatialObservationEncoder(
        observation,
        args.vision_artifact,
        inference_batch_size=args.image_batch_size,
    ).to(args.device).eval()
    completed = []
    for index, row in enumerate(assigned, 1):
        output = (
            args.output_root
            / "episodes"
            / row["split"]
            / row["task"]
            / f"episode_{row['episode_index']:06d}_seed_{row['seed']}.hdf5"
        )
        record = existing_record(output, row) if output.exists() else encode_episode(
            args, encoder, row, output
        )
        completed.append(record)
        atomic_json(
            args.state,
            {
                "state": "PREPARING",
                "stage": "native_rgb_post_dino_feature_cache",
                "rank": args.rank,
                "completed_episodes": index,
                "total_episodes": len(assigned),
                "output_bytes": sum(item["size_bytes"] for item in completed),
                "updated_at": now(),
            },
        )
    receipt = {
        "schema_version": 1,
        "round": "R12-R4",
        "protocol_variant": FULL_EPISODE_PROTOCOL,
        "rank": args.rank,
        "world_size": args.world_size,
        "manifest_receipt": manifest_receipt(data_root),
        "observation": observation,
        "episodes": completed,
        "created_at": now(),
    }
    atomic_json(receipt_path, receipt)
    atomic_json(
        args.state,
        {
            "state": "PASSED",
            "stage": "native_rgb_post_dino_shard_complete",
            "receipt": str(receipt_path),
            "updated_at": now(),
        },
    )
    print(json.dumps({"receipt": str(receipt_path), "episodes": len(completed)}))


def run_index(args) -> None:
    receipts = [
        args.output_root / f"rank_{rank:02d}_index.json"
        for rank in range(args.world_size)
    ]
    while not all(path.is_file() for path in receipts):
        atomic_json(
            args.heartbeat,
            {
                "producer": "prepare_r12_native_rgb_full_index",
                "ready": sum(path.is_file() for path in receipts),
                "total": len(receipts),
                "updated_at": now(),
            },
        )
        time.sleep(10)
    payloads = [json.loads(path.read_text()) for path in receipts]
    if any(
        payload.get("protocol_variant") != FULL_EPISODE_PROTOCOL
        or payload.get("world_size") != args.world_size
        for payload in payloads
    ):
        raise ValueError("full-episode rank receipt identity differs")
    episodes = [row for payload in payloads for row in payload["episodes"]]
    identities = [
        (row["task"], row["split"], row["seed"], row["episode_index"])
        for row in episodes
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate episode across full-episode rank receipts")
    counts = {
        split: {
            task: sum(
                row["steps"]
                for row in episodes
                if row["split"] == split and row["task"] == task
            )
            for task in TASKS
        }
        for split in ("train", "validation")
    }
    index = {
        "schema_version": 1,
        "round": "R12-R4",
        "protocol_variant": FULL_EPISODE_PROTOCOL,
        "world_size": args.world_size,
        "observation": locked_r12_full_episode_observation(),
        "cache_semantics": "native_480x640_rgb_encoded_before_post_dino_6x8_pooling",
        "episodes": sorted(
            episodes,
            key=lambda row: (row["split"], row["task"], row["episode_index"]),
        ),
        "step_counts": counts,
        "rank_receipt_sha256": {path.name: sha256(path) for path in receipts},
        "created_at": now(),
    }
    atomic_json(args.index, index)
    atomic_json(
        args.state,
        {
            "state": "PASSED",
            "stage": "native_rgb_full_index_complete",
            "index": str(args.index),
            "episodes": len(episodes),
            "step_counts": counts,
            "updated_at": now(),
        },
    )
    print(json.dumps(json.loads(args.state.read_text()), sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("shard", "index"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--vision-artifact", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=1)
    parser.add_argument("--image-batch-size", type=int, default=1)
    parser.add_argument("--index", type=Path)
    args = parser.parse_args()
    if args.world_size < 1 or args.frame_batch_size < 1 or args.image_batch_size < 1:
        raise ValueError("cache parallelism/batches must be positive")
    if args.mode == "shard":
        if args.data_root is None or args.vision_artifact is None:
            raise ValueError("shard mode requires data root and vision artifact")
        run_shard(args)
    else:
        if args.index is None:
            raise ValueError("index mode requires --index")
        run_index(args)


if __name__ == "__main__":
    main()
