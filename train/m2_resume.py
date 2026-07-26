"""Crash-safe, bounded-retention resume snapshots for Phase M2 training."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any, Mapping

import numpy as np
from safetensors.torch import load_file, save_file
import torch
from torch import nn


M2_RESUME_FORMAT = "wam.robofactory.m2.resume/1"


def save_m2_resume_checkpoint(
    directory: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    identity: Mapping[str, Any],
    progress: Mapping[str, Any],
    coverage_seen: torch.Tensor,
    keep_last: int = 2,
) -> dict[str, Any]:
    """Publish one complete snapshot, then prune superseded generations."""

    if keep_last <= 0:
        raise ValueError("M2 resume keep_last must be positive")
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    global_step = int(progress.get("global_step", -1))
    if global_step < 0:
        raise ValueError("M2 resume global_step cannot be negative")
    generation_name = (
        f"step_{global_step:08d}_{time.time_ns()}_{os.getpid()}"
    )
    staging = root / f".{generation_name}.tmp"
    generation = root / generation_name
    if staging.exists() or generation.exists():
        raise FileExistsError(f"M2 resume generation already exists: {generation}")
    staging.mkdir()
    try:
        core = model.module if hasattr(model, "module") else model
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in core.state_dict().items()
            },
            staging / "model.safetensors",
        )
        torch.save(optimizer.state_dict(), staging / "optimizer.pt")
        tensors = {
            "coverage_seen": coverage_seen.detach().to(
                device="cpu", dtype=torch.bool
            ).contiguous(),
            "torch_cpu_rng": torch.get_rng_state().cpu().contiguous(),
        }
        if torch.cuda.is_available():
            for index, value in enumerate(torch.cuda.get_rng_state_all()):
                tensors[f"torch_cuda_rng_{index}"] = value.cpu().contiguous()
        save_file(tensors, staging / "training_tensors.safetensors")
        files = {
            name: _sha256(staging / name)
            for name in (
                "model.safetensors",
                "optimizer.pt",
                "training_tensors.safetensors",
            )
        }
        schema = {
            "format_version": M2_RESUME_FORMAT,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "identity": dict(identity),
            "progress": dict(progress),
            "python_random_state": _jsonable_tuple(random.getstate()),
            "numpy_random_state": _numpy_random_state(),
            "files": files,
        }
        _write_json(staging / "state.json", schema)
        os.replace(staging, generation)
        _fsync_directory(root)
        state_sha256 = _sha256(generation / "state.json")
        pointer = {
            "format_version": M2_RESUME_FORMAT,
            "generation": generation_name,
            "state_sha256": state_sha256,
        }
        _write_json_atomic(root / "latest.json", pointer)
        _prune_generations(root, keep_last=keep_last)
        return {
            "directory": str(root),
            "generation": generation_name,
            "global_step": global_step,
            "state_sha256": state_sha256,
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_latest_m2_resume_checkpoint(
    directory: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_identity: Mapping[str, Any],
    device: str | torch.device,
) -> dict[str, Any] | None:
    """Strictly restore the latest fully published resume generation."""

    root = Path(directory).expanduser().resolve()
    if not root.exists():
        return None
    if not root.is_dir():
        raise ValueError(f"M2 resume path is not a directory: {root}")
    pointer_path = root / "latest.json"
    if not pointer_path.exists():
        visible = [path for path in root.iterdir() if not path.name.startswith(".")]
        if not visible:
            return None
        raise ValueError("M2 resume directory has no latest.json pointer")
    pointer = _read_json(pointer_path)
    if pointer.get("format_version") != M2_RESUME_FORMAT:
        raise ValueError("unsupported M2 resume pointer format")
    generation_name = str(pointer.get("generation", ""))
    if (
        not generation_name
        or generation_name in {".", ".."}
        or Path(generation_name).name != generation_name
    ):
        raise ValueError("M2 resume generation path is unsafe")
    generation = root / generation_name
    state_path = generation / "state.json"
    if (
        not generation.is_dir()
        or not state_path.is_file()
        or _sha256(state_path) != pointer.get("state_sha256")
    ):
        raise ValueError("M2 resume pointer does not identify a complete generation")
    schema = _read_json(state_path)
    if schema.get("format_version") != M2_RESUME_FORMAT:
        raise ValueError("unsupported M2 resume state format")
    if schema.get("identity") != dict(expected_identity):
        raise ValueError("M2 resume identity differs from this training run")
    files = schema.get("files")
    expected_files = {
        "model.safetensors",
        "optimizer.pt",
        "training_tensors.safetensors",
    }
    if not isinstance(files, dict) or set(files) != expected_files:
        raise ValueError("M2 resume file manifest is incomplete")
    for name, expected_sha256 in files.items():
        path = generation / name
        if not path.is_file() or _sha256(path) != expected_sha256:
            raise ValueError(f"M2 resume file identity mismatch: {name}")

    core = model.module if hasattr(model, "module") else model
    incompatible = core.load_state_dict(
        load_file(generation / "model.safetensors", device=str(device)),
        strict=True,
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict M2 resume model load failed: {incompatible}")
    optimizer_state = torch.load(
        generation / "optimizer.pt",
        map_location=device,
        weights_only=True,
    )
    optimizer.load_state_dict(optimizer_state)
    tensors = load_file(generation / "training_tensors.safetensors", device="cpu")
    if "coverage_seen" not in tensors or "torch_cpu_rng" not in tensors:
        raise ValueError("M2 resume training tensors are incomplete")
    torch.set_rng_state(tensors["torch_cpu_rng"])
    cuda_keys = sorted(
        (name for name in tensors if name.startswith("torch_cuda_rng_")),
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )
    if cuda_keys:
        if not torch.cuda.is_available():
            raise RuntimeError("M2 resume contains CUDA RNG state but CUDA is unavailable")
        cuda_states = [tensors[name] for name in cuda_keys]
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("M2 resume CUDA RNG device count changed")
        torch.cuda.set_rng_state_all(cuda_states)
    random.setstate(_tuple_from_json(schema.get("python_random_state")))
    _restore_numpy_random_state(schema.get("numpy_random_state"))
    progress = schema.get("progress")
    if not isinstance(progress, dict):
        raise ValueError("M2 resume progress must be an object")
    return {
        **progress,
        "coverage_seen": tensors["coverage_seen"].bool(),
        "generation": generation_name,
        "created_at": schema.get("created_at"),
    }


def _prune_generations(root: Path, *, keep_last: int) -> None:
    generations = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir()
            and path.name.startswith("step_")
            and not path.name.startswith(".")
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for obsolete in generations[keep_last:]:
        shutil.rmtree(obsolete)


def _numpy_random_state() -> dict[str, Any]:
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "name": str(name),
        "keys": keys.astype(np.uint32, copy=False).tolist(),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _restore_numpy_random_state(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("M2 resume NumPy RNG state must be an object")
    np.random.set_state(
        (
            str(value["name"]),
            np.asarray(value["keys"], dtype=np.uint32),
            int(value["position"]),
            int(value["has_gauss"]),
            float(value["cached_gaussian"]),
        )
    )


def _jsonable_tuple(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable_tuple(item) for item in value]
    return value


def _tuple_from_json(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_from_json(item) for item in value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid M2 resume JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"M2 resume JSON is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "M2_RESUME_FORMAT",
    "load_latest_m2_resume_checkpoint",
    "save_m2_resume_checkpoint",
]
