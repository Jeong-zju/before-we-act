#!/usr/bin/env python3
"""Build resumable frozen-DINO pooled history features for Step-2's 720 episodes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from before_we_act.step2_temporal_data import (  # noqa: E402
    SIX_TASKS,
    load_step2_episodes,
    sha256_file,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifests", nargs="+", required=True)
    result.add_argument("--dino-model", required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--device", default="cuda")
    return result


def cache_is_valid(path: Path, episode) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["source_hdf5_sha256"].item()) != episode.hdf5_sha256:
                return False
            expected = {"view_global", *(f"view_agent_{arm}" for arm in episode.arms)}
            actual = {name for name in data.files if name.startswith("view_")}
            if actual != expected:
                return False
            return all(
                data[name].shape == (episode.length, 768)
                and data[name].dtype == np.float16
                for name in expected
            )
    except (KeyError, OSError, ValueError):
        return False


def main() -> None:
    args = parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("cache batch size must be positive")
    if not Path(args.dino_model).is_dir():
        raise FileNotFoundError(args.dino_model)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world_size > 1:
        # Bind the rank to its CUDA device before NCCL is created.  Otherwise
        # the final cache barrier can guess a device and deadlock.
        torch.distributed.init_process_group(backend="nccl")
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(args.dino_model)
    model = AutoModel.from_pretrained(args.dino_model).to(device).eval()
    model.requires_grad_(False)
    mean = torch.tensor(processor.image_mean, device=device).view(1, -1, 1, 1)
    std = torch.tensor(processor.image_std, device=device).view(1, -1, 1, 1)
    first_patch = 1 + int(getattr(model.config, "num_register_tokens", 0))
    episodes = load_step2_episodes(args.manifests)
    assigned = episodes[rank::world_size]
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / f"rank_{rank:02d}_state.json"
    started = time.time()
    completed = 0
    encoded_images = 0

    @torch.no_grad()
    def encode(dataset, length: int) -> np.ndarray:
        nonlocal encoded_images
        pieces: list[np.ndarray] = []
        for first in range(0, length, args.batch_size):
            array = np.asarray(
                dataset[first : min(first + args.batch_size, length)],
                dtype=np.uint8,
            )
            value = torch.from_numpy(array).permute(0, 3, 1, 2).to(
                device, non_blocking=True
            )
            value = value.float().div_(255)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                raw = model(pixel_values=(value - mean) / std).last_hidden_state
                tokens = raw[:, first_patch:]
                if tokens.shape[1:] != (1200, 768):
                    raise ValueError(f"unexpected DINO token grid: {tokens.shape}")
                pooled = tokens.mean(1)
            pieces.append(pooled.float().cpu().numpy().astype(np.float16))
            encoded_images += len(array)
        return np.concatenate(pieces, axis=0)

    for episode in assigned:
        target = args.output / episode.task / f"{episode.hdf5_sha256}.npz"
        if cache_is_valid(target, episode):
            completed += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        values: dict[str, np.ndarray] = {}
        with h5py.File(episode.path, "r") as source:
            images = source["data/observation/images"]
            encoded_sources: dict[str, np.ndarray] = {}
            for key in ("global", *(f"agent_{arm}" for arm in episode.arms)):
                source_key = key
                if source_key not in images:
                    if episode.task != "place_food" or not key.startswith("agent_"):
                        raise KeyError(f"missing {key} in {episode.path}")
                    source_key = "global"
                shape = images[source_key].shape
                if shape[0] < episode.length or shape[1:] != (480, 640, 3):
                    raise ValueError(
                        "cache source cannot cover selected original 640x480 RGB "
                        f"transitions: {episode.path}/{source_key}, "
                        f"frames={shape[0]}, selected={episode.length}, shape={shape[1:]}"
                    )
                if source_key not in encoded_sources:
                    # Manifests deliberately stop at first success, while the
                    # original HDF5 can retain post-success frames. Cache only
                    # the selected legal prefix and encode reused views once.
                    encoded_sources[source_key] = encode(
                        images[source_key], episode.length
                    )
                values[f"view_{key}"] = encoded_sources[source_key]
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                source_hdf5_sha256=np.asarray(episode.hdf5_sha256),
                dino_model=np.asarray(str(Path(args.dino_model).resolve())),
                **values,
            )
        os.replace(temporary, target)
        completed += 1
        atomic_json(
            state_path,
            {
                "format_version": "before-we-act.step2.visual_cache_rank/1",
                "status": "RUNNING",
                "rank": rank,
                "world_size": world_size,
                "completed_episodes": completed,
                "assigned_episodes": len(assigned),
                "encoded_images": encoded_images,
                "elapsed_seconds": time.time() - started,
                "updated_at_utc": utc_now(),
            },
        )
    atomic_json(
        state_path,
        {
            "format_version": "before-we-act.step2.visual_cache_rank/1",
            "status": "PASSED",
            "rank": rank,
            "world_size": world_size,
            "completed_episodes": completed,
            "assigned_episodes": len(assigned),
            "encoded_images": encoded_images,
            "elapsed_seconds": time.time() - started,
            "completed_at_utc": utc_now(),
        },
    )
    if world_size > 1:
        torch.distributed.barrier()
    if rank == 0:
        rows = []
        per_task = {task: 0 for task in SIX_TASKS}
        for episode in episodes:
            path = args.output / episode.task / f"{episode.hdf5_sha256}.npz"
            if not cache_is_valid(path, episode):
                raise RuntimeError(f"visual cache did not complete: {path}")
            per_task[episode.task] += 1
            rows.append(
                {
                    "task": episode.task,
                    "episode_index": episode.episode_index,
                    "source_hdf5_sha256": episode.hdf5_sha256,
                    "cache_path": str(path),
                    "cache_sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
        atomic_json(
            args.output / "cache_receipt.json",
            {
                "format_version": "before-we-act.step2.visual_cache/1",
                "status": "PASSED",
                "source": "original 720 trajectory RGB at 640x480",
                "transform": "frozen DINOv3 ViT-B/16 mean of 30x40 patch grid",
                "feature_dtype": "float16",
                "feature_width": 768,
                "dino_model": str(Path(args.dino_model).resolve()),
                "episodes": len(rows),
                "episodes_per_task": per_task,
                "files": rows,
                "completed_at_utc": utc_now(),
            },
        )
        print("STEP2_VISUAL_CACHE_PASSED", flush=True)
    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
