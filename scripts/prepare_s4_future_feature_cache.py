#!/usr/bin/env python3
"""Precompute one two-GPU shared DINO-PCA cache for S4 future targets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback
from typing import Any

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_static_rgb_act_moe import (  # noqa: E402
    _load_yaml,
    _mapping,
    _vision,
)
from train.s2_future_prediction import (  # noqa: E402
    load_s2_artifact,
    project_dino_grid,
)
from train.s4_future_feature_cache import (  # noqa: E402
    CACHE_FORMAT,
    CAMERAS,
    S4ProjectedFutureFeatureCache,
    file_sha256,
)
from train.shared_hdf5_receipt import validate_shared_hdf5_receipt  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--batch-rows", type=int, default=24)
    return parser


def _dataset_index(
    manifests: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], int]:
    episodes: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, str]] = []
    offset = 0
    for task_index, manifest in enumerate(manifests):
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"manifest root must be a mapping: {manifest}")
        task_id = str(_mapping(raw, "task")["id"])
        vision = _mapping(raw, "vision")
        cameras = tuple(str(value) for value in vision["camera_order"])
        if not cameras or cameras != CAMERAS[: len(cameras)]:
            raise ValueError(f"non-canonical camera order in {manifest}")
        next_prefix = str(vision["next_prefix"])
        raw_episodes = raw.get("episodes")
        if not isinstance(raw_episodes, list) or len(raw_episodes) != 150:
            raise ValueError(f"S4 cache requires 150 episodes in {manifest}")
        manifest_rows.append(
            {
                "task_id": task_id,
                "path": str(manifest),
                "sha256": file_sha256(manifest),
            }
        )
        for value in raw_episodes:
            if not isinstance(value, dict):
                raise ValueError(f"manifest episode must be a mapping: {manifest}")
            episode = value
            split = str(episode.get("split", ""))
            if split not in {"train", "validation", "test"}:
                raise ValueError(f"unsupported episode split {split!r} in {manifest}")
            steps = int(episode["steps"])
            hdf5_path = (manifest.parent / str(episode["hdf5_path"])).resolve(
                strict=True
            )
            episodes.append(
                {
                    "task_id": task_id,
                    "task_index": task_index,
                    "episode_index": int(episode["episode_index"]),
                    "hdf5_path": str(hdf5_path),
                    "hdf5_sha256": str(episode["hdf5_sha256"]),
                    "steps": steps,
                    "offset": offset,
                    "cameras": list(cameras),
                    "next_prefix": next_prefix,
                    "split": split,
                }
            )
            offset += steps
    if len(episodes) != 750:
        raise ValueError("S4 cache requires exactly 750 all-split episodes")
    return episodes, manifest_rows, offset


def _worker(
    worker_index: int,
    gpu: str,
    config_raw: str,
    features_raw: str,
    episodes: list[dict[str, Any]],
    batch_rows: int,
    error_raw: str,
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        device = torch.device("cuda:0")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(f"cache worker {worker_index} lacks one visible GPU")
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        config_path = Path(config_raw).resolve(strict=True)
        raw = _load_yaml(config_path)
        vision = _vision(raw).to(device).eval()
        artifact_path = (
            ROOT / str(_mapping(raw, "artifacts")["pca_statistics"])
        ).resolve(strict=True)
        artifact = load_s2_artifact(artifact_path, device=device)
        features = np.load(features_raw, mmap_mode="r+", allow_pickle=False)
        total = len(episodes)
        for position, episode in enumerate(episodes, start=1):
            cameras = tuple(str(value) for value in episode["cameras"])
            camera_indices = [CAMERAS.index(camera) for camera in cameras]
            steps = int(episode["steps"])
            offset = int(episode["offset"])
            with h5py.File(str(episode["hdf5_path"]), "r") as file:
                datasets = [
                    file[f"{episode['next_prefix']}/{camera}"] for camera in cameras
                ]
                if any(dataset.shape[0] < steps for dataset in datasets):
                    raise ValueError("future RGB rows are shorter than manifest steps")
                for start in range(0, steps, batch_rows):
                    stop = min(start + batch_rows, steps)
                    per_camera = [
                        np.asarray(dataset[start:stop], dtype=np.uint8)
                        for dataset in datasets
                    ]
                    images = np.stack(per_camera, axis=1)
                    images = np.ascontiguousarray(
                        images.transpose(0, 1, 4, 2, 3).reshape(
                            -1, 3, images.shape[2], images.shape[3]
                        )
                    )
                    encoded = vision.forward_spatial_grid(
                        torch.from_numpy(images).to(device=device, non_blocking=True),
                        grid_height=2,
                        grid_width=2,
                    ).spatial_tokens.float()
                    projected = project_dino_grid(encoded, artifact).reshape(
                        stop - start, len(cameras), 4, 256
                    )
                    projected_cpu = projected.cpu().numpy()
                    for local_camera, canonical_camera in enumerate(camera_indices):
                        features[
                            offset + start : offset + stop, canonical_camera
                        ] = projected_cpu[:, local_camera]
            features.flush()
            if position == 1 or position % 10 == 0 or position == total:
                print(
                    json.dumps(
                        {
                            "event": "future_cache_progress",
                            "worker": worker_index,
                            "gpu": gpu,
                            "episode": position,
                            "episodes": total,
                            "task_id": episode["task_id"],
                            "episode_index": episode["episode_index"],
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        features.flush()
    except BaseException:
        Path(error_raw).write_text(traceback.format_exc(), encoding="utf-8")
        raise


def _validate_existing(
    root: Path,
    *,
    manifests: list[Path],
    pca_sha256: str,
    vision_sha256: str,
) -> str:
    digest = (root / "features.npy.sha256").read_text().strip()
    if file_sha256(root / "features.npy") != digest:
        raise ValueError("existing S4 future cache binary SHA256 changed")
    S4ProjectedFutureFeatureCache(
        root,
        manifests=manifests,
        expected_features_sha256=digest,
        expected_pca_sha256=pca_sha256,
        expected_vision_weights_sha256=vision_sha256,
    )
    return digest


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_rows <= 0:
        raise ValueError("--batch-rows must be positive")
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if len(gpus) != 2 or len(set(gpus)) != 2:
        raise ValueError("S4 shared future cache requires exactly two GPU ids")
    config_path = args.config.expanduser().resolve(strict=True)
    raw = _load_yaml(config_path)
    data = _mapping(raw, "data")
    manifests = [
        (ROOT / str(value)).resolve(strict=True)
        for value in data["manifests"]  # type: ignore[index]
    ]
    parent = _mapping(raw, "parent")
    validate_shared_hdf5_receipt(
        args.receipt,
        manifests,
        expected_proof_sha256=str(parent["expected_legacy_r6l_policy_sha256"]),
        expected_receipt_sha256=args.receipt_sha256,
    )
    artifact_path = (
        ROOT / str(_mapping(raw, "artifacts")["pca_statistics"])
    ).resolve(strict=True)
    pca_sha256 = file_sha256(artifact_path)
    if pca_sha256 != str(parent["expected_pca_sha256"]):
        raise ValueError("S4 future cache PCA artifact differs from config")
    vision = _mapping(raw, "vision")
    vision_path = (ROOT / str(vision["weights_path"])).resolve(strict=True)
    vision_sha256 = file_sha256(vision_path)
    if vision_sha256 != str(vision["expected_weights_sha256"]):
        raise ValueError("S4 future cache DINO weights differ from config")
    output = args.output_root.expanduser().resolve()
    if output.exists():
        digest = _validate_existing(
            output,
            manifests=manifests,
            pca_sha256=pca_sha256,
            vision_sha256=vision_sha256,
        )
        print(
            json.dumps(
                {"cache": str(output), "features_sha256": digest, "reused": True},
                sort_keys=True,
            )
        )
        return 0

    episodes, manifest_rows, total_rows = _dataset_index(manifests)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        features_path = temporary / "features.npy"
        features = np.lib.format.open_memmap(
            features_path,
            mode="w+",
            dtype=np.float32,
            shape=(total_rows, 5, 4, 256),
        )
        features[:] = 0.0
        features.flush()
        del features
        shards = [episodes[index::2] for index in range(2)]
        context = mp.get_context("spawn")
        processes: list[mp.Process] = []
        errors: list[Path] = []
        for worker_index, (gpu, shard) in enumerate(zip(gpus, shards, strict=True)):
            error = temporary / f"worker_{worker_index}.error.txt"
            process = context.Process(
                target=_worker,
                args=(
                    worker_index,
                    gpu,
                    str(config_path),
                    str(features_path),
                    shard,
                    args.batch_rows,
                    str(error),
                ),
            )
            process.start()
            processes.append(process)
            errors.append(error)
        for process in processes:
            process.join()
        failed = [process for process in processes if process.exitcode != 0]
        if failed:
            detail = "\n".join(
                error.read_text(encoding="utf-8")
                for error in errors
                if error.is_file()
            )
            raise RuntimeError(f"S4 future cache worker failed:\n{detail}")
        features_sha256 = file_sha256(features_path)
        metadata = {
            "format_version": CACHE_FORMAT,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_config_sha256": file_sha256(config_path),
            "pca_artifact_sha256": pca_sha256,
            "vision_weights_sha256": vision_sha256,
            "vision_config_sha256": str(vision["expected_config_sha256"]),
            "preprocess_id": str(vision["preprocess_id"]),
            "inference_batch_size": int(vision["inference_batch_size"]),
            "grid": [2, 2],
            "latent_dim": 256,
            "dtype": "float32",
            "shape": [total_rows, 5, 4, 256],
            "features_size_bytes": features_path.stat().st_size,
            "features_sha256": features_sha256,
            "manifests": manifest_rows,
            "episodes": episodes,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "features.npy.sha256").write_text(
            features_sha256 + "\n", encoding="utf-8"
        )
        (temporary / "cache.ready").write_text(
            f"S4 projected future cache ready at {metadata['created_at']}\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    _validate_existing(
        output,
        manifests=manifests,
        pca_sha256=pca_sha256,
        vision_sha256=vision_sha256,
    )
    print(
        json.dumps(
            {
                "cache": str(output),
                "features_sha256": features_sha256,
                "rows": total_rows,
                "episodes": len(episodes),
                "reused": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
