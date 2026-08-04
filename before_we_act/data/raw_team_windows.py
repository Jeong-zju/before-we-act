from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import Dataset


TASKS = (
    "lift_barrier",
    "camera_alignment",
    "three_robots_stack_cube",
    "long_pipeline_delivery",
    "take_photo",
)


class CachedTeamWindows(Dataset):
    def __init__(self, cache_path: str | Path, split: str) -> None:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported R11 observation cache")
        if split not in ("train", "validation"):
            raise ValueError("R11 cache split must be train or validation")
        self.metadata = payload["metadata"]
        self.data: Mapping[str, torch.Tensor] = payload[split]
        size = int(self.data["visual"].shape[0])
        if not size or any(int(value.shape[0]) != size for value in self.data.values()):
            raise ValueError("R11 cache tensors have inconsistent lengths")

    def __len__(self) -> int:
        return int(self.data["visual"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.data.items()}


def read_cache_metadata(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return dict(payload["metadata"])


def manifest_receipt(data_root: str | Path) -> dict[str, str]:
    import hashlib

    root = Path(data_root)
    receipt = {}
    for task in TASKS:
        path = root / task / "training_manifest.json"
        data = path.read_bytes()
        payload = json.loads(data)
        if payload["task"]["id"] != task or len(payload["episodes"]) != 150:
            raise ValueError(f"invalid five-task manifest: {path}")
        receipt[task] = hashlib.sha256(data).hexdigest()
    return receipt
