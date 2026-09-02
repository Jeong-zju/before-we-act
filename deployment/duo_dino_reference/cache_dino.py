"""Build a resumable frozen-DINO cache for DuoBench B0-H.

The cache is deliberately episode-addressed.  It stores pooled 768-wide patch
features for the shared head and both wrist cameras; the dataset chooses only
the focal arm's wrist feature.  ``--smoke`` uses a deterministic, clearly
labelled projection and is useful for checking the I/O contract without
downloading a multi-gigabyte foundation model.  Formal training rejects such
a receipt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch

from .data import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DuoTemporalEpisode,
    TASKS,
    _load_task_arrays,
    load_duo_episodes,
    resize_rgb_batch,
)
from .preprocessing import (
    DINO_NORMALIZATION_ID,
    IMAGE_PREPROCESS_ID,
    validate_dino_model_contract,
    validate_dino_processor_contract,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _cache_path(root: Path, episode: DuoTemporalEpisode) -> Path:
    return root / episode.task / f"{episode.cache_key}.npz"


def _valid_cache(path: Path, episode: DuoTemporalEpisode) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as source:
            if str(source["source_identity"].item()) != episode.source_identity:
                return False
            if str(source["image_preprocess_id"].item()) != IMAGE_PREPROCESS_ID:
                return False
            if str(source["dino_normalization_id"].item()) not in {
                DINO_NORMALIZATION_ID,
                "smoke_projection_not_dino",
            }:
                return False
            strict = source.get("strict_dino_contract")
            # Formal caches carry an explicit opt-in bit.  Smoke projection
            # files intentionally carry ``False`` and are never admissible to
            # formal training; accepting a missing field would allow an old
            # cache to be relabelled by a new receipt.
            if strict is None:
                return False
            strict_value = bool(np.asarray(strict).item())
            normalization = str(source["dino_normalization_id"].item())
            if strict_value != (normalization == DINO_NORMALIZATION_ID):
                return False
            if {key for key in source.files if key.startswith("view_")} != {
                "view_head",
                "view_wrist_0",
                "view_wrist_1",
            }:
                return False
            return all(
                source[key].shape == (episode.length, 768)
                and source[key].dtype == np.float16
                and np.isfinite(source[key]).all()
                for key in ("view_head", "view_wrist_0", "view_wrist_1")
            )
    except (OSError, KeyError, ValueError):
        return False


def _smoke_encoder(images: np.ndarray) -> np.ndarray:
    """Deterministic 768-D fallback for contract smoke tests only."""

    # Average a fixed 8x8 RGB grid (192 values), then tile with a deterministic
    # sinusoidal projection.  It is intentionally not accepted by formal train.
    value = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255)
    pooled = torch.nn.functional.adaptive_avg_pool2d(value, (8, 8)).flatten(1)
    columns = torch.arange(768, dtype=pooled.dtype).view(1, -1)
    weights = torch.sin(columns * 0.017 + torch.arange(192).view(-1, 1) * 0.013)
    return (pooled @ weights / 192.0).numpy().astype(np.float16)


def _make_dino_encoder(
    model_name: str,
    device: torch.device,
    image_height: int,
    image_width: int,
    batch_size: int,
) -> Callable[[np.ndarray], np.ndarray]:
    from transformers import AutoImageProcessor, AutoModel

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    processor = AutoImageProcessor.from_pretrained(model_name, token=token)
    processor_contract = validate_dino_processor_contract(processor)
    model = AutoModel.from_pretrained(model_name, token=token)
    model_contract = validate_dino_model_contract(model)
    model = model.to(device).eval()
    model.requires_grad_(False)
    mean = torch.tensor(processor.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, device=device).view(1, 3, 1, 1)
    expected_tokens = (image_height // 16) * (image_width // 16)
    first_patch = 1 + int(getattr(model.config, "num_register_tokens", 0))

    @torch.inference_mode()
    def encode(images: np.ndarray) -> np.ndarray:
        pieces: list[np.ndarray] = []
        for first in range(0, len(images), batch_size):
            raw = resize_rgb_batch(
                images[first : first + batch_size], image_height, image_width
            )
            value = raw.to(device, non_blocking=True).float().div_(255)
            # DINOv3 is frozen; autocast is used only on CUDA where bfloat16 is
            # available.  CPU smoke/formal diagnostics remain numerically clear.
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = model(pixel_values=(value - mean) / std).last_hidden_state
            else:
                hidden = model(pixel_values=(value - mean) / std).last_hidden_state
            tokens = hidden[:, first_patch:]
            if tuple(tokens.shape[1:]) != (expected_tokens, 768):
                raise ValueError(
                    "DINOv3 token contract differs: expected "
                    f"{(expected_tokens, 768)}, got {tuple(tokens.shape[1:])}"
                )
            pieces.append(tokens.mean(1).float().cpu().numpy().astype(np.float16))
        return np.concatenate(pieces, axis=0) if pieces else np.empty((0, 768), np.float16)

    # The callable carries immutable provenance for the cache receipt.  Keeping
    # it attached avoids a second processor/model load and ensures the exact
    # object used for encoding is the one recorded in metadata.
    encode.contract = {**processor_contract, **model_contract}  # type: ignore[attr-defined]
    return encode


def build_cache(
    prepared_data: Path,
    output: Path,
    *,
    dino_model: str,
    image_height: int = DEFAULT_IMAGE_HEIGHT,
    image_width: int = DEFAULT_IMAGE_WIDTH,
    batch_size: int = 32,
    device: str = "cuda",
    smoke: bool = False,
) -> dict:
    if image_height % 16 or image_width % 16 or image_height <= 0 or image_width <= 0:
        raise ValueError("DINO dimensions must be positive multiples of 16")
    episodes = load_duo_episodes(prepared_data, require_formal=not smoke)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and 48 % world:
        raise ValueError("Duo B0-H effective batch 48 must divide WORLD_SIZE")
    dev = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}" if device == "cuda" else device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    if world > 1:
        torch.distributed.init_process_group("nccl")
    encoder = _smoke_encoder if smoke else _make_dino_encoder(
        dino_model, dev, image_height, image_width, batch_size
    )
    encoder_contract = (
        {
            "image_preprocess_id": IMAGE_PREPROCESS_ID,
            "dino_normalization_id": "smoke_projection_not_dino",
            "encoder_contract": "smoke_only",
        }
        if smoke
        else getattr(encoder, "contract")
    )
    task_arrays = {
        task: _load_task_arrays(Path(prepared_data), task)
        for task in TASKS
        if any(item.task == task for item in episodes[rank::world])
    }
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    completed = 0
    assigned = episodes[rank::world]
    for episode in assigned:
        target = _cache_path(output, episode)
        if _valid_cache(target, episode):
            completed += 1
            continue
        arrays = task_arrays[episode.task]
        values = {}
        for key, source_name in (
            ("view_head", "head"),
            ("view_wrist_0", "left"),
            ("view_wrist_1", "right"),
        ):
            values[key] = encoder(
                np.asarray(arrays[source_name][episode.start : episode.end])
            )
        if any(value.shape != (episode.length, 768) for value in values.values()):
            raise ValueError(f"cache encoder returned wrong episode shape: {episode}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                source_identity=np.asarray(episode.source_identity),
                dino_model=np.asarray("smoke_projection" if smoke else str(Path(dino_model).resolve())),
                image_height=np.asarray(image_height),
                image_width=np.asarray(image_width),
                image_preprocess_id=np.asarray(IMAGE_PREPROCESS_ID),
                dino_normalization_id=np.asarray(encoder_contract["dino_normalization_id"]),
                strict_dino_contract=np.asarray(not smoke),
                **values,
            )
        os.replace(temporary, target)
        completed += 1
        if completed == 1 or completed % 10 == 0:
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "completed_episodes": completed,
                        "assigned_episodes": len(assigned),
                        "elapsed_seconds": time.time() - started,
                    }
                ),
                flush=True,
            )
    if world > 1:
        torch.distributed.barrier(device_ids=[int(dev.index or 0)])
    if rank == 0:
        files = []
        for episode in episodes:
            path = _cache_path(output, episode)
            if not _valid_cache(path, episode):
                raise RuntimeError(f"Duo DINO cache incomplete: {path}")
            files.append(
                {
                    "task": episode.task,
                    "episode_id": episode.episode_id,
                    "cache_path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        receipt = {
            "schema": "before-we-act.duobench.dino-cache/1",
            "status": "SMOKE" if smoke else "PASSED",
            "encoder": "deterministic_smoke_projection" if smoke else "dinov3_vitb16_frozen",
            "dino_model": str(Path(dino_model).resolve()) if not smoke else "smoke_projection",
            "image_height": image_height,
            "image_width": image_width,
            "patch_size": 16,
            "feature_width": 768,
            "episodes": len(files),
            "episodes_per_task": {task: sum(item["task"] == task for item in files) for task in TASKS},
            "image_preprocess_id": encoder_contract["image_preprocess_id"],
            "dino_normalization_id": encoder_contract["dino_normalization_id"],
            "strict_dino_contract": not smoke,
            "dino_processor_contract_sha256": encoder_contract.get("dino_processor_contract_sha256"),
            "dino_model_contract_sha256": encoder_contract.get("dino_model_contract_sha256"),
            "dino_image_mean": encoder_contract.get("dino_image_mean"),
            "dino_image_std": encoder_contract.get("dino_image_std"),
            "dino_rescale_factor": encoder_contract.get("dino_rescale_factor"),
            "dino_processor_resize": encoder_contract.get("dino_processor_resize"),
            "dino_processor_resample": encoder_contract.get("dino_processor_resample"),
            "files": files,
            "created_at_utc": _now(),
        }
        _atomic_json(output / "cache_receipt.json", receipt)
        print(json.dumps(receipt), flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()
    return receipt if rank == 0 else {"status": "rank_complete", "rank": rank}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--dino-model", default="/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    build_cache(
        args.prepared_data,
        args.output,
        dino_model=args.dino_model,
        image_height=args.image_height,
        image_width=args.image_width,
        batch_size=args.batch_size,
        device=args.device,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
