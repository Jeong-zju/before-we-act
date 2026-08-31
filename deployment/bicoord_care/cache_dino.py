"""Create a resumable frozen DINOv3 ViT-B/16 feature cache for BiCoord."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import torch

from .config import (
    DATASET_REVISION,
    DINO_HIDDEN_SIZE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    TASKS,
    TOTAL_EPISODES,
)
from .data import discover_bicoord_episodes
from .hdf5_data import BiCoordHDF5Reader
from .preprocessing import (
    DINO_NORMALIZATION_ID,
    IMAGE_PREPROCESS_ID,
    decode_bicoord_jpeg_rgb,
    resize_rgb_batch,
    validate_dino_model_contract,
    validate_dino_processor_contract,
)
from .stage_common import artifact, assert_common_paths, atomic_json, canonical_sha256, common_parser, publish_result, sha256_file


def _dino_source_manifest(model_path: Path) -> dict[str, Any]:
    """Hash the local upstream DINO artifact, including every weight shard."""

    root = model_path.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"DINO model must be a pinned local directory: {root}")
    names = {
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    }
    files = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file()
            and (
                path.name in names
                or path.name.startswith("model-") and path.suffix == ".safetensors"
                or path.name.startswith("pytorch_model-") and path.suffix == ".bin"
            )
        ),
        key=lambda path: path.name,
    )
    if not any(
        path.name.endswith((".safetensors", ".bin")) for path in files
    ) or not (root / "config.json").is_file() or not (root / "preprocessor_config.json").is_file():
        raise RuntimeError(f"DINO directory lacks config/processor/weight artifacts: {root}")
    rows = [
        {"name": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
    ]
    value: dict[str, Any] = {"root": str(root), "files": rows}
    value["sha256"] = canonical_sha256(value)
    return value


def _real_encoder(model_path: Path, device: torch.device) -> Callable[[np.ndarray], np.ndarray]:
    try:
        from transformers import AutoImageProcessor, AutoModel
    except Exception as error:
        raise RuntimeError("transformers is required for formal DINO cache") from error
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        processor = AutoImageProcessor.from_pretrained(str(model_path), token=token)
        model = AutoModel.from_pretrained(str(model_path), token=token)
    except TypeError as error:
        # Transformers <4.40 used use_auth_token.  Only retry the known token
        # keyword incompatibility; model/config failures remain fatal.
        if token is None or "token" not in str(error):
            raise
        processor = AutoImageProcessor.from_pretrained(
            str(model_path), use_auth_token=token
        )
        model = AutoModel.from_pretrained(str(model_path), use_auth_token=token)
    processor_contract = validate_dino_processor_contract(processor)
    model_contract = validate_dino_model_contract(model)
    model = model.to(device).eval()
    model.requires_grad_(False)
    mean = torch.tensor(processor.image_mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    first_patch = 1 + int(getattr(model.config, "num_register_tokens", 0))

    @torch.inference_mode()
    def encode(images: np.ndarray) -> np.ndarray:
        raw = resize_rgb_batch(images, IMAGE_HEIGHT, IMAGE_WIDTH)
        value = raw.to(device).float().div_(255)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = model(pixel_values=(value - mean) / std).last_hidden_state
        tokens = hidden[:, first_patch:]
        if tuple(tokens.shape[1:]) != ((IMAGE_HEIGHT // 16) * (IMAGE_WIDTH // 16), DINO_HIDDEN_SIZE):
            raise ValueError(f"DINO token contract differs: {tuple(tokens.shape)}")
        return tokens.float().mean(1).cpu().numpy().astype(np.float16)

    encode.contract = {**processor_contract, **model_contract}  # type: ignore[attr-defined]
    return encode


def _cache_path(root: Path, episode) -> Path:
    return root / episode.task / f"{episode.hdf5_sha256}.npz"


def _valid_cache(path: Path, episode, *, model_sha256: str | None = None) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as source:
            if str(np.asarray(source["source_identity"]).item()) != episode.source_identity:
                return False
            if model_sha256 is not None and str(np.asarray(source["dino_source_sha256"]).item()) != model_sha256:
                return False
            if str(np.asarray(source["dino_normalization_id"]).item()) != DINO_NORMALIZATION_ID:
                return False
            if str(np.asarray(source["image_preprocess_id"]).item()) != IMAGE_PREPROCESS_ID:
                return False
            if int(np.asarray(source["image_height"]).item()) != IMAGE_HEIGHT or int(np.asarray(source["image_width"]).item()) != IMAGE_WIDTH:
                return False
            if bool(np.asarray(source["strict_dino_contract"]).item()) is not True:
                return False
            for key in ("view_head", "view_wrist_0", "view_wrist_1"):
                value = source[key]
                if value.shape != (episode.length, DINO_HIDDEN_SIZE) or value.dtype != np.float16 or not np.isfinite(value).all():
                    return False
    except Exception:
        return False
    return True


def run(args: argparse.Namespace) -> dict[str, Any]:
    assert_common_paths(args, need_dataset=True)
    smoke = bool(
        args.operation == "smoke"
        or getattr(args, "smoke", False)
        or os.environ.get("BICOORD_SMOKE") == "1"
    )
    episodes = discover_bicoord_episodes(
        args.dataset, require_formal=not smoke, verify_schema=not smoke
    )
    if smoke:
        # Smoke is allowed to reduce data volume only.  The exact same
        # upstream DINO artifact and preprocessing path remain mandatory.
        limit = int(os.environ.get("BICOORD_SMOKE_EPISODES", "1"))
        if limit < 1:
            raise ValueError("BICOORD_SMOKE_EPISODES must be positive")
        episodes = episodes[:limit]
    rank = int(getattr(args, "rank", 0)); world = int(getattr(args, "world_size", 1))
    if world < 1 or not 0 <= rank < world:
        raise ValueError(f"invalid cache shard rank/world: {rank}/{world}")
    if world > 4:
        raise ValueError("BiCoord cache is frozen to at most four shards")
    if not torch.cuda.is_available():
        raise RuntimeError("BiCoord DINO cache requires the supervisor GPU lease")
    # The supervisor assigns one physical GPU per cache worker and rewrites it
    # to logical cuda:0 through CUDA_VISIBLE_DEVICES.
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dino_source = _dino_source_manifest(args.dino_model)
    encoder = _real_encoder(args.dino_model.resolve(), device)
    # Keep the cache under the run's artifact namespace.  This is the path
    # recorded in the frozen supervisor config and consumed by B0-H.
    cache_root = args.run / "artifacts" / "dino_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    assigned = episodes[rank::world]
    started = time.time(); rows: list[dict[str, Any]] = []
    for episode in assigned:
        target = _cache_path(cache_root, episode)
        if target.exists() and not _valid_cache(target, episode, model_sha256=dino_source["sha256"]):
            raise RuntimeError(
                f"existing DINO cache is invalid or belongs to another contract; refusing overwrite: {target}"
            )
        if not target.exists():
            reader = BiCoordHDF5Reader(episode.path, task=episode.task, episode_id=episode.episode_id)
            arrays: dict[str, np.ndarray] = {}
            for key, camera in (("view_head", "head_camera"), ("view_wrist_0", "left_camera"), ("view_wrist_1", "right_camera")):
                pieces: list[np.ndarray] = []
                for first in range(0, episode.length, 32):
                    frames = np.stack(
                        [
                            decode_bicoord_jpeg_rgb(reader.frame_bytes(camera, index))
                            for index in range(first, min(episode.length, first + 32))
                        ],
                        axis=0,
                    )
                    pieces.append(encoder(frames))
                arrays[key] = np.concatenate(pieces, axis=0)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            try:
                with temporary.open("xb") as stream:
                    np.savez(
                        stream,
                        source_identity=np.asarray(episode.source_identity),
                        dino_model=np.asarray(str(args.dino_model.resolve())),
                        dino_source_sha256=np.asarray(dino_source["sha256"]),
                        image_height=np.asarray(IMAGE_HEIGHT), image_width=np.asarray(IMAGE_WIDTH),
                        image_preprocess_id=np.asarray(IMAGE_PREPROCESS_ID),
                        dino_normalization_id=np.asarray(DINO_NORMALIZATION_ID),
                        strict_dino_contract=np.asarray(True), **arrays,
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
        if not _valid_cache(target, episode, model_sha256=dino_source["sha256"]):
            raise RuntimeError(f"cache validation failed after write: {target}")
        rows.append({"task": episode.task, "episode_id": episode.episode_id, "source_identity": episode.source_identity, "path": str(target.resolve()), "sha256": sha256_file(target)})
        if len(rows) == 1 or len(rows) % 10 == 0:
            print(json.dumps({"event": "dino_cache", "rank": rank, "completed": len(rows), "assigned": len(assigned), "elapsed_seconds": time.time() - started}), flush=True)
    shard_manifest = cache_root / f"shard_{rank}.json"
    atomic_json(shard_manifest, {"schema": "before-we-act.bicoord.dino-cache-shard/1", "rank": rank, "world_size": world, "smoke": smoke, "config_sha256": args.config_sha256, "dataset_revision": DATASET_REVISION, "dino_source_sha256": dino_source["sha256"], "files": rows})
    # Rank zero waits for peer manifests instead of assuming process launch
    # order.  This also works when the supervisor uses four independent child
    # processes rather than torch.distributed.
    receipt_path = cache_root / "cache_receipt.json"
    if rank == 0:
        deadline = time.time() + float(os.environ.get("BICOORD_CACHE_WAIT_SECONDS", "3600"))
        while time.time() < deadline and not all((cache_root / f"shard_{i}.json").is_file() for i in range(world)):
            time.sleep(1.0)
        if not all((cache_root / f"shard_{i}.json").is_file() for i in range(world)):
            raise TimeoutError("timed out waiting for DINO cache shards")
        all_rows: list[dict[str, Any]] = []
        for i in range(world):
            shard = json.loads((cache_root / f"shard_{i}.json").read_text())
            if shard.get("rank") != i or shard.get("world_size") != world or shard.get("smoke") is not smoke or shard.get("config_sha256") != args.config_sha256 or shard.get("dataset_revision") != DATASET_REVISION or shard.get("dino_source_sha256") != dino_source["sha256"]:
                raise RuntimeError(f"DINO cache shard {i} belongs to a different frozen run")
            all_rows.extend(shard["files"])
        all_rows.sort(key=lambda row: (TASKS.index(row["task"]), int(row["episode_id"])))
        identities = {(ep.task, int(ep.episode_id)): ep for ep in episodes}
        seen: set[tuple[str, int]] = set()
        for row in all_rows:
            key = (str(row.get("task")), int(row.get("episode_id", -1)))
            if key in seen or key not in identities:
                raise RuntimeError(f"DINO cache receipt has duplicate/unknown episode: {key}")
            seen.add(key)
            path = Path(str(row.get("path", ""))).resolve()
            if cache_root.resolve() not in path.parents:
                raise RuntimeError(f"DINO cache path escapes cache root: {path}")
            if str(row.get("sha256")) != sha256_file(path):
                raise RuntimeError(f"DINO cache file hash changed: {path}")
            expected_episode = identities[key]
            if str(row.get("source_identity", "")) != expected_episode.source_identity:
                raise RuntimeError(f"DINO cache source identity differs: {path}")
            if not _valid_cache(path, expected_episode, model_sha256=dino_source["sha256"]):
                raise RuntimeError(f"DINO cache receipt cannot validate file: {path}")
        if len(all_rows) != len(episodes) or len(seen) != len(episodes):
            raise RuntimeError("DINO cache receipt cannot prove complete source coverage")
        receipt = {
            "schema": "before-we-act.bicoord.dino-cache/1", "status": "SMOKE" if smoke else "PASSED",
            "encoder": "dinov3_vitb16_frozen",
            "dataset_revision": DATASET_REVISION,
            "config_sha256": args.config_sha256,
            "dino_model": str(args.dino_model.resolve()),
            "dino_source": dino_source,
            "image_height": IMAGE_HEIGHT, "image_width": IMAGE_WIDTH, "patch_size": 16,
            "feature_width": DINO_HIDDEN_SIZE, "episodes": len(all_rows),
            "episodes_per_task": {task: sum(row["task"] == task for row in all_rows) for task in TASKS},
            "image_preprocess_id": IMAGE_PREPROCESS_ID,
            "dino_normalization_id": DINO_NORMALIZATION_ID,
            "strict_dino_contract": True,
            "model_contract": dict(getattr(encoder, "contract", {})),
            "files": all_rows,
        }
        atomic_json(receipt_path, receipt)
    # Nonzero workers always publish their immutable shard manifest.  A stale
    # prior global receipt may be overwritten by rank zero and must never make
    # a peer worker's artifact hash change after publication.
    evidence = receipt_path if rank == 0 else shard_manifest
    return publish_result(args, stage="dino_cache", include_model_contract=True, artifacts=[artifact(evidence, kind="dino_cache")], rank=rank, world_size=world, cache_receipt=str(receipt_path.resolve()), episodes=len(assigned), strict_dino_contract=True, dino_normalization_id=DINO_NORMALIZATION_ID)


def main(argv: list[str] | None = None) -> int:
    parser = common_parser(__doc__, ("cache-all", "smoke"))
    parser.add_argument("--rank", type=int, default=0); parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv); run(args); return 0


if __name__ == "__main__":
    raise SystemExit(main())
