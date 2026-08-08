#!/usr/bin/env python3
"""Build the R13N six-task feature index without duplicating four valid caches."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from before_we_act.data.full_episode_windows import FULL_EPISODE_PROTOCOL
from before_we_act.r13n import (
    FULL_CACHE_PROTOCOL,
    TASKS,
    TASK_SPECS,
    observation_contract,
    sha256,
    validate_manifest,
)
from before_we_act.spatial_observation import R12SpatialObservationEncoder


NEW_TASKS = ("pass_shoe", "place_food")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_hdf5(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with h5py.File(temporary, "w") as handle:
        handle.attrs["schema_version"] = payload["schema_version"]
        handle.attrs["round"] = payload["round"]
        handle.attrs["metadata_json"] = json.dumps(payload["metadata"], sort_keys=True)
        for key, value in payload.items():
            if key in {"schema_version", "round", "metadata"}:
                continue
            array = value.detach().cpu().numpy()
            handle.create_dataset(key, data=array, chunks=(True if array.ndim > 1 else None))
        handle.flush()
    os.replace(temporary, path)


def patch_means(image: np.ndarray, grid: int = 4) -> np.ndarray:
    if tuple(image.shape) != (480, 640, 3):
        raise ValueError(f"R13N fixed RGB shape differs: {tuple(image.shape)}")
    value = image.reshape(grid, 480 // grid, grid, 640 // grid, 3)
    return value.mean(axis=(1, 3), dtype=np.float32).reshape(16, 3) / 127.5 - 1.0


def source_rows(data_root: Path) -> list[dict]:
    rows: list[dict] = []
    for task in NEW_TASKS:
        receipt = validate_manifest(data_root, task, require_files=True)
        path = data_root / task / "training_manifest.json"
        payload = json.loads(path.read_text())
        camera_order = tuple(payload["vision"]["camera_order"])
        for episode in payload["episodes"]:
            if episode["split"] not in {"train", "validation"}:
                continue
            rows.append(
                {
                    "task": task,
                    "split": episode["split"],
                    "seed": int(episode["seed"]),
                    "steps": int(episode["steps"]),
                    "episode_index": int(episode["episode_index"]),
                    "hdf5_path": str(data_root / task / episode["hdf5_path"]),
                    "hdf5_sha256": str(episode["hdf5_sha256"]),
                    "manifest_sha256": receipt["manifest_sha256"],
                    "camera_order": camera_order,
                    "agents": int(TASK_SPECS[task]["agents"]),
                }
            )
    return rows


@torch.inference_mode()
def encode_episode(args, encoder, row: dict, output: Path) -> dict:
    source = Path(row["hdf5_path"]).resolve(strict=True)
    observed_source_sha = sha256(source)
    if observed_source_sha != row["hdf5_sha256"]:
        raise ValueError(f"source HDF5 hash differs: {source}")
    steps = int(row["steps"])
    spatial = torch.zeros((steps, 5, 48, 768), dtype=torch.float16)
    visual = torch.zeros((steps, 16, 15), dtype=torch.float16)
    view_mask = torch.zeros((steps, 5), dtype=torch.bool)
    qpos = torch.zeros((steps, 4, 9), dtype=torch.float32)
    executed = torch.zeros((steps, 4, 8), dtype=torch.float32)
    commanded = torch.zeros((steps, 4, 8), dtype=torch.float32)
    camera_order = tuple(row["camera_order"])
    with h5py.File(source, "r") as handle:
        data = handle["data"]
        agents = sorted(data["observation/agents"].keys())
        if len(agents) != int(row["agents"]):
            raise ValueError(f"R13N agent count differs: {source}")
        agent_mask = torch.arange(4) < len(agents)
        for index, agent in enumerate(agents):
            qpos[:, index] = torch.from_numpy(np.asarray(data[f"observation/agents/{agent}/qpos"][:steps]))
            executed[:, index] = torch.from_numpy(np.asarray(data[f"action/agents/{agent}/executed"][:steps]))
            commanded[:, index] = torch.from_numpy(np.asarray(data[f"action/agents/{agent}/commanded"][:steps]))
        for start in range(0, steps, args.frame_batch_size):
            end = min(steps, start + args.frame_batch_size)
            requests: list[tuple[int, int, np.ndarray]] = []
            for frame in range(start, end):
                for view_index, view in enumerate(camera_order):
                    image = np.asarray(data[f"observation/images/{view}"][frame])
                    if tuple(image.shape) != (480, 640, 3):
                        raise ValueError(f"native RGB shape differs: {source}:{frame}:{view}")
                    visual[frame, :, view_index * 3 : (view_index + 1) * 3] = torch.from_numpy(patch_means(image)).to(torch.float16)
                    view_mask[frame, view_index] = True
                    requests.append((frame, view_index, image))
            images = torch.from_numpy(np.stack([row_[2].transpose(2, 0, 1) for row_ in requests])).to(args.device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(args.device).startswith("cuda")):
                encoded = encoder.encoder.forward_spatial_grid(images, grid_height=6, grid_width=8).spatial_tokens
            encoded = encoded.cpu().to(torch.float16)
            for offset, (frame, view_index, _image) in enumerate(requests):
                spatial[frame, view_index] = encoded[offset]
            atomic_json(
                args.heartbeat,
                {
                    "producer": "prepare_r13n_full_cache",
                    "rank": args.rank,
                    "task": row["task"],
                    "episode_index": row["episode_index"],
                    "frame": end,
                    "episode_steps": steps,
                    "updated_at_epoch": time.time(),
                },
            )
    metadata = {
        "created_at_epoch": time.time(),
        "protocol_variant": FULL_CACHE_PROTOCOL,
        "task": row["task"],
        "split": row["split"],
        "seed": row["seed"],
        "steps": steps,
        "episode_index": row["episode_index"],
        "source_hdf5": str(source),
        "hdf5_sha256": row["hdf5_sha256"],
        "manifest_sha256": row["manifest_sha256"],
        "camera_order": list(camera_order),
        "observation": observation_contract(),
        "legal_inputs": "manifest-selected current fixed-view RGB plus causal qpos/executed-action history",
        "forbidden_inputs": "future RGB, simulator state, expert action at inference",
    }
    atomic_hdf5(
        output,
        {
            "schema_version": 1,
            "round": "R13N",
            "metadata": metadata,
            "visual": visual,
            "view_mask": view_mask,
            "qpos": qpos,
            "executed_actions": executed,
            "commanded_actions": commanded,
            "agent_mask": agent_mask,
            "spatial_tokens": spatial,
            "spatial_view_mask": view_mask.clone(),
        },
    )
    return {
        "path": str(output.resolve()),
        "sha256": sha256(output),
        "size_bytes": output.stat().st_size,
        "task": row["task"],
        "split": row["split"],
        "seed": row["seed"],
        "steps": steps,
        "hdf5_sha256": row["hdf5_sha256"],
        "episode_index": row["episode_index"],
        "cache_round": "R13N",
        "cache_protocol": FULL_CACHE_PROTOCOL,
    }


def existing_record(output: Path, row: dict) -> dict:
    with h5py.File(output, "r") as handle:
        metadata = json.loads(str(handle.attrs["metadata_json"]))
    if (
        metadata.get("protocol_variant") != FULL_CACHE_PROTOCOL
        or metadata.get("hdf5_sha256") != row["hdf5_sha256"]
        or int(metadata.get("steps", -1)) != int(row["steps"])
    ):
        raise ValueError(f"existing R13N feature shard differs: {output}")
    return {
        "path": str(output.resolve()),
        "sha256": sha256(output),
        "size_bytes": output.stat().st_size,
        "task": row["task"], "split": row["split"], "seed": row["seed"],
        "steps": row["steps"], "hdf5_sha256": row["hdf5_sha256"],
        "episode_index": row["episode_index"], "cache_round": "R13N",
        "cache_protocol": FULL_CACHE_PROTOCOL,
    }


def run_shard(args) -> None:
    rows = source_rows(args.data_root.resolve(strict=True))
    assigned = [row for index, row in enumerate(rows) if index % args.world_size == args.rank]
    receipt_path = args.output_root / f"rank_{args.rank:02d}_index.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("protocol_variant") != FULL_CACHE_PROTOCOL or not all(Path(row["path"]).is_file() for row in receipt.get("episodes", [])):
            raise ValueError("existing R13N rank receipt differs")
        print(json.dumps({"reused": str(receipt_path), "episodes": len(receipt["episodes"])}))
        return
    encoder = R12SpatialObservationEncoder(observation_contract(), args.vision_artifact, inference_batch_size=args.image_batch_size).to(args.device).eval()
    completed = []
    atomic_json(args.state, {"status":"PREPARING","rank":args.rank,"completed_episodes":0,"total_episodes":len(assigned),"updated_at_epoch":time.time()})
    for number, row in enumerate(assigned, 1):
        output = args.output_root / "episodes" / row["split"] / row["task"] / f"episode_{row['episode_index']:06d}_seed_{row['seed']}.hdf5"
        record = existing_record(output, row) if output.exists() else encode_episode(args, encoder, row, output)
        completed.append(record)
        atomic_json(args.state, {"status":"PREPARING","rank":args.rank,"completed_episodes":number,"total_episodes":len(assigned),"output_bytes":sum(item["size_bytes"] for item in completed),"updated_at_epoch":time.time()})
    atomic_json(receipt_path, {"schema_version":1,"round":"R13N","protocol_variant":FULL_CACHE_PROTOCOL,"rank":args.rank,"world_size":args.world_size,"episodes":completed,"created_at_epoch":time.time()})
    atomic_json(args.state, {"status":"PASSED","rank":args.rank,"completed_episodes":len(completed),"total_episodes":len(assigned),"receipt":str(receipt_path),"updated_at_epoch":time.time()})
    print(json.dumps({"receipt": str(receipt_path), "episodes": len(completed)}, sort_keys=True))


def aggregate_action_stats(data_root: Path) -> dict[str, list[float]]:
    count = 0
    total = np.zeros(8, dtype=np.float64)
    square = np.zeros(8, dtype=np.float64)
    for task in TASKS:
        spec = TASK_SPECS[task]
        values = np.load(data_root / task / "normalization.npz")
        agents = int(spec["agents"])
        mean = np.asarray(values["action_mean"], dtype=np.float64).reshape(agents, 8)
        std = np.asarray(values["action_std"], dtype=np.float64).reshape(agents, 8)
        steps = int(spec["train_steps"])
        count += steps * agents
        total += mean.sum(axis=0) * steps
        square += (np.square(std) + np.square(mean)).sum(axis=0) * steps
    mean = total / count
    variance = np.maximum(square / count - np.square(mean), 1e-8)
    return {"a_mean": mean.astype(np.float32).tolist(), "a_std": np.sqrt(variance).astype(np.float32).tolist(), "active_agent_rows": count}


def run_index(args) -> None:
    receipts = [args.output_root / f"rank_{rank:02d}_index.json" for rank in range(args.world_size)]
    while not all(path.is_file() for path in receipts):
        atomic_json(args.heartbeat, {"producer":"prepare_r13n_index","ready":sum(path.is_file() for path in receipts),"total":len(receipts),"updated_at_epoch":time.time()})
        time.sleep(10)
    old = json.loads(args.reuse_index.resolve(strict=True).read_text())
    if old.get("round") != "R12-R4" or old.get("protocol_variant") != FULL_EPISODE_PROTOCOL:
        raise ValueError("R13N reusable four-task index identity differs")
    reused = []
    for source in old["episodes"]:
        if source["task"] not in TASKS[:4]:
            continue
        row = dict(source)
        row.update(cache_round="R12-R4", cache_protocol=FULL_EPISODE_PROTOCOL)
        if not Path(row["path"]).is_file():
            raise ValueError(f"reusable R13N cache shard missing: {row['path']}")
        reused.append(row)
    generated = [row for path in receipts for row in json.loads(path.read_text())["episodes"]]
    episodes = reused + generated
    identities = [(row["task"], row["split"], int(row["episode_index"])) for row in episodes]
    if len(identities) != len(set(identities)) or len(episodes) != len(TASKS) * 135:
        raise ValueError("R13N combined episode coverage differs")
    counts = {
        split: {task: sum(int(row["steps"]) for row in episodes if row["split"] == split and row["task"] == task) for task in TASKS}
        for split in ("train", "validation")
    }
    if counts["train"] != {task:int(TASK_SPECS[task]["train_steps"]) for task in TASKS}:
        raise ValueError("R13N combined train step counts differ")
    manifests = {task:validate_manifest(args.data_root, task, require_files=True) for task in TASKS}
    index = {
        "schema_version":1,"round":"R13N","protocol_variant":FULL_CACHE_PROTOCOL,
        "cache_semantics":"native_480x640_rgb_encoded_before_post_dino_6x8_pooling",
        "tasks":list(TASKS),"observation":observation_contract(),
        "episodes":sorted(episodes,key=lambda row:(row["split"],row["task"],row["episode_index"])),
        "step_counts":counts,"stats":aggregate_action_stats(args.data_root),
        "manifest_receipts":manifests,"reused_index":str(args.reuse_index.resolve()),
        "reused_index_sha256":sha256(args.reuse_index),
        "rank_receipt_sha256":{path.name:sha256(path) for path in receipts},
        "created_at_epoch":time.time(),
    }
    atomic_json(args.index, index)
    atomic_json(args.state, {"status":"PASSED","stage":"six_task_index_complete","index":str(args.index),"episodes":len(episodes),"step_counts":counts,"updated_at_epoch":time.time()})
    print(json.dumps({"index":str(args.index),"episodes":len(episodes),"step_counts":counts},sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("shard", "index"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--vision-artifact", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=1)
    parser.add_argument("--image-batch-size", type=int, default=5)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--reuse-index", type=Path)
    args = parser.parse_args()
    if args.world_size < 1 or args.frame_batch_size < 1 or args.image_batch_size < 1:
        raise ValueError("R13N cache parallelism/batches must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "shard":
        if args.vision_artifact is None:
            raise ValueError("R13N shard mode requires --vision-artifact")
        run_shard(args)
    else:
        if args.index is None or args.reuse_index is None:
            raise ValueError("R13N index mode requires --index and --reuse-index")
        run_index(args)


if __name__ == "__main__":
    main()
