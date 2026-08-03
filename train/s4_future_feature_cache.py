"""Shared projected DINO cache for fixed S4 future-visual targets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
from torch import Tensor


CACHE_FORMAT = "wam.robofactory.s4.projected_future_feature_cache/1"
CAMERAS = ("global", "agent_0", "agent_1", "agent_2", "agent_3")
HORIZONS = (1, 25, 50, 100)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


class S4ProjectedFutureFeatureCache:
    """Read a stat/hash-bound float32 PCA grid from a shared memory map."""

    def __init__(
        self,
        root: str | Path,
        *,
        manifests: Sequence[str | Path],
        expected_features_sha256: str,
        expected_pca_sha256: str,
        expected_vision_weights_sha256: str,
    ) -> None:
        cache_root = Path(root).resolve(strict=True)
        metadata_path = cache_root / "metadata.json"
        features_path = cache_root / "features.npy"
        if not metadata_path.is_file() or not features_path.is_file():
            raise ValueError(f"incomplete S4 future feature cache: {cache_root}")
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = _mapping(raw, "future feature cache metadata")
        if metadata.get("format_version") != CACHE_FORMAT:
            raise ValueError("unsupported S4 future feature cache format")
        if (
            SHA256_PATTERN.fullmatch(expected_features_sha256) is None
            or metadata.get("features_sha256") != expected_features_sha256
            or metadata.get("pca_artifact_sha256") != expected_pca_sha256
            or metadata.get("vision_weights_sha256")
            != expected_vision_weights_sha256
        ):
            raise ValueError("S4 future feature cache artifact identity differs")
        manifest_rows = metadata.get("manifests")
        if not isinstance(manifest_rows, list):
            raise ValueError("S4 future feature cache lacks manifest identities")
        expected_manifests = {
            Path(value).resolve(strict=True): file_sha256(value) for value in manifests
        }
        observed_manifests = {
            Path(str(_mapping(row, "cache manifest").get("path", ""))).resolve(
                strict=True
            ): str(_mapping(row, "cache manifest").get("sha256", ""))
            for row in manifest_rows
        }
        if observed_manifests != expected_manifests:
            raise ValueError("S4 future feature cache manifests differ")
        shape = tuple(int(value) for value in metadata.get("shape", ()))
        if len(shape) != 4 or shape[1:] != (5, 4, 256):
            raise ValueError("S4 future feature cache shape contract differs")
        features = np.load(features_path, mmap_mode="r", allow_pickle=False)
        if features.dtype != np.float32 or tuple(features.shape) != shape:
            raise ValueError("S4 future feature cache binary shape/dtype differs")
        if features_path.stat().st_size != int(metadata.get("features_size_bytes", -1)):
            raise ValueError("S4 future feature cache binary size changed")
        episodes = metadata.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError("S4 future feature cache episode index is missing")
        episode_index: dict[tuple[int, int], tuple[int, int]] = {}
        for value in episodes:
            row = _mapping(value, "cache episode")
            key = (int(row["task_index"]), int(row["episode_index"]))
            if key in episode_index:
                raise ValueError(f"duplicate S4 future cache episode {key}")
            episode_index[key] = (int(row["offset"]), int(row["steps"]))
        if len(episode_index) != 750:
            raise ValueError("S4 future feature cache requires 750 episodes")
        self.root = cache_root
        self.metadata = dict(metadata)
        self.features = features
        self.episode_index = episode_index

    def projected_future(self, grouped: Mapping[str, Tensor]) -> Tensor:
        """Return cached projected next-view grids as ``[B,F,5,4,256]``."""

        tasks = grouped["task_index"].detach().cpu().tolist()
        episodes = grouped["episode_index"].detach().cpu().tolist()
        decisions = grouped["decision_t"].detach().cpu().tolist()
        horizons = tuple(int(value) for value in grouped["future_horizons"][0].tolist())
        if horizons != HORIZONS:
            raise ValueError("S4 future cache requires horizons [1,25,50,100]")
        rows: list[np.ndarray] = []
        for task, episode, decision in zip(tasks, episodes, decisions, strict=True):
            key = (int(task), int(episode))
            if key not in self.episode_index:
                raise ValueError(f"S4 future cache lacks episode {key}")
            offset, steps = self.episode_index[key]
            remaining = steps - int(decision)
            if remaining <= 0:
                raise ValueError("S4 future cache decision lies beyond the episode")
            selected = [
                int(decision) + min(horizon, remaining) - 1
                for horizon in horizons
            ]
            rows.append(
                np.asarray(
                    self.features[[offset + value for value in selected]],
                    dtype=np.float32,
                )
            )
        return torch.from_numpy(np.stack(rows, axis=0))

    def normalized_targets(
        self,
        grouped: Mapping[str, Tensor],
        artifact: Mapping[str, Any],
        *,
        current_local: Tensor,
        current_shared: Tensor,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        projected = self.projected_future(grouped).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        future_shared = projected[:, :, 0]
        future_local = projected[:, :, 1:5].permute(0, 2, 1, 3, 4).contiguous()
        local_valid = grouped["future_agent_visual_valid_mask"].to(
            device=device, dtype=torch.bool
        )
        shared_valid = grouped["future_shared_visual_valid_mask"].to(
            device=device, dtype=torch.bool
        )
        local_delta = future_local - current_local[:, :, None]
        local_mean = artifact["visual_delta_mean"].to(local_delta)
        local_std = artifact["visual_delta_std"].to(local_delta)
        normalized_local = (
            local_delta - local_mean[None, None, :, None]
        ) / local_std[None, None, :, None]
        normalized_local = normalized_local.masked_fill(
            ~local_valid[:, :, :, None, None], 0.0
        )
        shared_delta = future_shared - current_shared[:, None]
        shared_mean = artifact.get("shared_visual_delta_mean")
        shared_std = artifact.get("shared_visual_delta_std")
        if not isinstance(shared_mean, Tensor) or not isinstance(shared_std, Tensor):
            raise ValueError("S4 artifact lacks shared future visual statistics")
        normalized_shared = (
            shared_delta - shared_mean.to(shared_delta)[None, :, None]
        ) / shared_std.to(shared_delta)[None, :, None]
        normalized_shared = normalized_shared.masked_fill(
            ~shared_valid[:, :, None, None], 0.0
        )
        return normalized_local, normalized_shared


__all__ = [
    "CACHE_FORMAT",
    "CAMERAS",
    "HORIZONS",
    "S4ProjectedFutureFeatureCache",
    "file_sha256",
]
