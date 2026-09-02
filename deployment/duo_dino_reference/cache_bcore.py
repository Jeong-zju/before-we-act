"""Cache full frozen B0-H action contexts for DuoBench B-core training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data._utils.collate import default_collate

from before_we_act.temporal_history_policy import TemporalHistoryPolicy
from .bcore_data import BCORE_CACHE_SCHEMA, sha256_file, validate_b0h_payload
from .data import (
    ACTION_HORIZON,
    ACTION_LAG_ROWS,
    TASKS,
    DuoTemporalDataset,
    DuoTemporalRequest,
    load_duo_episodes,
)
from .preprocessing import DINO_NORMALIZATION_ID, IMAGE_PREPROCESS_ID


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, path)


def _load_model(
    checkpoint: Path, dino_model: str | None, device: torch.device
) -> tuple[TemporalHistoryPolicy, Mapping[str, object]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = validate_b0h_payload(payload)
    model_name = str(dino_model or config.get("dino_model") or "")
    if not model_name:
        raise ValueError("Duo B-core cache requires a DINO model path")
    model = TemporalHistoryPolicy(
        state_dim=int(config["state_dim"]),
        action_dim=int(config["action_dim"]),
        variant="hidden_residual",
        horizon=int(config["horizon"]),
        d_model=int(config.get("d_model", 384)),
        enc_layers=int(config.get("enc_layers", 4)),
        dec_layers=int(config.get("dec_layers", 7)),
        roles=int(config.get("roles", 4)),
        role_rank=int(config.get("role_rank", 32)),
        history_layers=int(config.get("history_layers", 2)),
        dino_model=model_name,
        image_height=int(config["image_height"]),
        image_width=int(config["image_width"]),
        strict_dino_contract=True,
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    if model.hidden_residual is None:
        raise RuntimeError("Duo B-core cache requires hidden-residual B0-H")
    return model, config


def _cached_paths(output: Path, episode: Any) -> tuple[Path, Path, Path]:
    root = output / episode.task
    return (
        root / f"{episode.cache_key}.decoded.npy",
        root / f"{episode.cache_key}.base_action.npy",
        root / f"{episode.cache_key}.complete.json",
    )


def _valid_episode_cache(output: Path, episode: Any, b0h_sha: str) -> bool:
    decoded_path, base_path, marker_path = _cached_paths(output, episode)
    if not decoded_path.is_file() or not base_path.is_file() or not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text())
        decoded = np.load(decoded_path, mmap_mode="r")
        base = np.load(base_path, mmap_mode="r")
        return bool(
            marker.get("status") == "PASSED"
            and marker.get("b0h_checkpoint_sha256") == b0h_sha
            and marker.get("source_identity") == episode.source_identity
            and decoded.shape == (episode.length - ACTION_LAG_ROWS, 2, ACTION_HORIZON, 384)
            and base.shape == (episode.length - ACTION_LAG_ROWS, 2, ACTION_HORIZON, 8)
            and decoded.dtype == np.float16
            and base.dtype == np.float16
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _worker(rank: int, world: int, values: Mapping[str, object]) -> None:
    prepared = Path(str(values["prepared_data"]))
    visual_cache = Path(str(values["visual_cache"]))
    checkpoint = Path(str(values["b0h_checkpoint"]))
    output = Path(str(values["output"]))
    batch_size = int(values["batch_size"])
    if batch_size < 1:
        raise ValueError("Duo B-core cache batch size must be positive")
    device_kind = str(values["device"])
    if device_kind == "cuda":
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
    else:
        if world != 1:
            raise ValueError("CPU B-core cache supports one worker only")
        device = torch.device(device_kind)
    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 8) // max(world, 1))))
    episodes = load_duo_episodes(prepared, require_formal=True)
    model, config = _load_model(
        checkpoint, str(values.get("dino_model") or "") or None, device
    )
    dataset = DuoTemporalDataset(
        prepared,
        episodes,
        visual_cache,
        image_height=int(config["image_height"]),
        image_width=int(config["image_width"]),
        cache_limit=8,
    )
    b0h_sha = sha256_file(checkpoint)
    assigned = list(range(rank, len(episodes), world))
    completed = 0
    samples = 0
    started = time.time()
    task_tokens: dict[str, list[float]] = {}
    for ordinal, episode_index in enumerate(assigned, start=1):
        episode = episodes[episode_index]
        decoded_path, base_path, marker_path = _cached_paths(output, episode)
        if _valid_episode_cache(output, episode, b0h_sha):
            completed += 1
            samples += (episode.length - ACTION_LAG_ROWS) * 2
            continue
        decoded = np.empty(
            (episode.length - ACTION_LAG_ROWS, 2, ACTION_HORIZON, 384), dtype=np.float16
        )
        base = np.empty(
            (episode.length - ACTION_LAG_ROWS, 2, ACTION_HORIZON, 8), dtype=np.float16
        )
        requests = [
            DuoTemporalRequest(
                episode_index,
                arm,
                time_index,
                f"cache:{episode.cache_key}:{arm}:{time_index}",
                episode.task,
            )
            for time_index in range(episode.length - ACTION_LAG_ROWS)
            for arm in (0, 1)
        ]
        for first in range(0, len(requests), batch_size):
            selected = requests[first : first + batch_size]
            batch = default_collate([dataset[request] for request in selected])
            inputs = {
                key: batch[key].to(device, non_blocking=True)
                for key in DuoTemporalDataset.MODEL_INPUT_FIELDS
            }
            inputs["global_rgb"] = inputs["global_rgb"].float().div_(255)
            inputs["local_rgb"] = inputs["local_rgb"].float().div_(255)
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                context = model._decode_action_context(**inputs, actions=None)
                prediction = model.out(context.decoded)
                history = context.history_summary.unsqueeze(1).expand(
                    -1, ACTION_HORIZON, -1
                )
                prediction = prediction + model.hidden_residual(
                    torch.cat((context.decoded, history), dim=-1)
                )
            if episode.task not in task_tokens:
                task_tokens[episode.task] = context.task_token[0].float().cpu().tolist()
            for local, request in enumerate(selected):
                decoded[request.time_index, request.arm] = (
                    context.decoded[local].float().cpu().numpy().astype(np.float16)
                )
                base[request.time_index, request.arm] = (
                    prediction[local].float().cpu().numpy().astype(np.float16)
                )
        _atomic_npy(decoded_path, decoded)
        _atomic_npy(base_path, base)
        _atomic_json(
            marker_path,
            {
                "status": "PASSED",
                "task": episode.task,
                "episode_id": episode.episode_id,
                "source_identity": episode.source_identity,
                "samples": (episode.length - ACTION_LAG_ROWS) * 2,
                "b0h_checkpoint_sha256": b0h_sha,
                "decoded_sha256": sha256_file(decoded_path),
                "base_action_sha256": sha256_file(base_path),
            },
        )
        completed += 1
        samples += (episode.length - ACTION_LAG_ROWS) * 2
        if ordinal == 1 or ordinal % 5 == 0 or ordinal == len(assigned):
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "episodes": completed,
                        "assigned": len(assigned),
                        "samples": samples,
                        "episodes_per_hour": completed
                        / max(time.time() - started, 1e-6)
                        * 3600,
                    }
                ),
                flush=True,
            )
    _atomic_json(
        output / f"rank_{rank:02d}_receipt.json",
        {
            "rank": rank,
            "world_size": world,
            "episodes": completed,
            "samples": samples,
            "task_tokens": task_tokens,
        },
    )


def build_cache(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.b0h_checkpoint, map_location="cpu", weights_only=False)
    config = validate_b0h_payload(payload)
    episodes = load_duo_episodes(args.prepared_data, require_formal=True)
    b0h_sha = sha256_file(args.b0h_checkpoint)
    requested_world = int(args.gpus)
    if args.device == "cuda":
        visible = torch.cuda.device_count()
        world = min(requested_world if requested_world > 0 else visible, visible)
        if world < 1:
            raise RuntimeError("Duo B-core cache requires at least one visible CUDA GPU")
    else:
        world = 1
    values = {
        "prepared_data": str(args.prepared_data),
        "visual_cache": str(args.visual_cache),
        "b0h_checkpoint": str(args.b0h_checkpoint),
        "dino_model": str(args.dino_model or ""),
        "output": str(args.output),
        "batch_size": args.batch_size,
        "device": args.device,
    }
    if world == 1:
        _worker(0, 1, values)
    else:
        mp.spawn(_worker, args=(world, values), nprocs=world, join=True)

    rank_rows = [
        json.loads((args.output / f"rank_{rank:02d}_receipt.json").read_text())
        for rank in range(world)
    ]
    markers = list(args.output.glob("*/*.complete.json"))
    invalid = [
        episode.cache_key
        for episode in episodes
        if not _valid_episode_cache(args.output, episode, b0h_sha)
    ]
    if len(markers) != 550 or invalid:
        raise RuntimeError(
            f"Duo B-core cache incomplete: markers={len(markers)}, invalid={invalid[:4]}"
        )
    task_tokens: dict[str, list[float]] = {}
    for row in rank_rows:
        task_tokens.update(row.get("task_tokens", {}))
    if set(task_tokens) != set(TASKS):
        raise RuntimeError(f"Duo B-core task-token coverage differs: {sorted(task_tokens)}")
    _atomic_json(args.output / "task_tokens.json", task_tokens)
    total_samples = sum((episode.length - ACTION_LAG_ROWS) * 2 for episode in episodes)
    if sum(int(row["samples"]) for row in rank_rows) != total_samples:
        raise RuntimeError("Duo B-core rank sample coverage differs")
    receipt = {
        "schema": BCORE_CACHE_SCHEMA,
        "status": "PASSED",
        "episodes": 550,
        "episodes_per_task": {task: 50 for task in TASKS},
        "samples": total_samples,
        "tasks": list(TASKS),
        "dtype": "float16",
        "decoded_shape_per_sample": [ACTION_HORIZON, 384],
        "base_action_shape_per_sample": [ACTION_HORIZON, 8],
        "base_action_semantics": "complete_TemporalHistoryPolicy_hidden_residual_prediction",
        "policy_family": "TemporalHistoryPolicy",
        "method_family": "CARE",
        "downstream_policy_family": "PredictiveTeamBeliefPolicy",
        "benchmark_adapter": "DuoBench",
        "vision_backbone": "dinov3_vitb16_frozen",
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "strict_dino_contract": True,
        "strictly_decentralized": True,
        "act_provider_allowed": False,
        "action_encoding": "absolute_joint7_binary_gripper1",
        "b0h_checkpoint": str(args.b0h_checkpoint.resolve()),
        "b0h_checkpoint_sha256": b0h_sha,
        "prepared_manifest_sha256": sha256_file(args.prepared_data / "manifest.json"),
        "visual_cache_receipt_sha256": sha256_file(
            args.visual_cache / "cache_receipt.json"
        ),
        "world_size": world,
        "rank_receipts": [f"rank_{rank:02d}_receipt.json" for rank in range(world)],
        "created_at_utc": _now(),
    }
    _atomic_json(args.output / "cache_receipt.json", receipt)
    print(json.dumps(receipt), flush=True)
    return receipt


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--b0h-checkpoint", type=Path, required=True)
    parser.add_argument("--dino-model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    build_cache(_arguments())


if __name__ == "__main__":
    main()


__all__ = ["build_cache"]
