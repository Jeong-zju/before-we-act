"""Shared, target-separated data contracts for R13 latent world training."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import Dataset

from .full_episode_windows import FullEpisodeActionWindows


LEGAL_INPUT_KEYS = (
    "belief_tokens",
    "belief_agent_tokens",
    "belief_consensus",
    "belief_uncertainty",
    "agent_mask",
    "candidate_actions",
    "candidate_valid_mask",
)
TARGET_KEYS = (
    "current_latent",
    "future_latent",
    "future_qpos_delta",
    "future_progress",
    "future_failure",
    "horizon_mask",
)


class CachedWorldWindows(Dataset):
    """Hash-audited R13 cache with inputs and future targets kept distinct."""

    def __init__(self, path: str | Path, split: str) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 1 or payload.get("round") != "R13":
            raise ValueError("unsupported R13 world cache")
        if split not in ("train", "validation"):
            raise ValueError("R13 cache split must be train or validation")
        self.metadata = dict(payload["metadata"])
        if self.metadata.get("future_targets_are_model_inputs") is not False:
            raise ValueError("R13 cache does not prove target/input separation")
        self.data: Mapping[str, torch.Tensor] = payload[split]
        expected = set(LEGAL_INPUT_KEYS) | set(TARGET_KEYS) | {"task_index"}
        if set(self.data) != expected:
            raise ValueError("R13 cache tensor keys differ from the frozen schema")
        size = int(self.data["belief_tokens"].shape[0])
        if not size or any(int(value.shape[0]) != size for value in self.data.values()):
            raise ValueError("R13 cache tensors have inconsistent lengths")
        if self.data["candidate_actions"].ndim != 5:
            raise ValueError("R13 cached candidates must be [N,P,A,H,D]")
        if self.data["future_latent"].ndim != 4:
            raise ValueError("R13 future latent must be [N,K,T,D]")

    def __len__(self) -> int:
        return int(self.data["belief_tokens"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.data.items()}


class R13SourceWindows(FullEpisodeActionWindows):
    """R12 causal rows plus future-only supervision kept outside model input."""

    def __init__(self, *args, prediction_horizons=(1, 5, 15), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.prediction_horizons = tuple(int(value) for value in prediction_horizons)
        if self.prediction_horizons != (1, 5, 15):
            raise ValueError("R13 future horizons differ from the frozen protocol")

    def __getitem__(self, request) -> dict[str, torch.Tensor]:
        item = super().__getitem__(request)
        episode_index, current = map(int, request)
        episode = self._load(episode_index)
        steps = int(self.episodes[episode_index]["steps"])
        indices = [min(current + value, steps - 1) for value in self.prediction_horizons]
        item.update(
            current_spatial_tokens=episode.row("spatial_tokens", current).float(),
            current_spatial_view_mask=episode.row("spatial_view_mask", current).bool(),
            current_qpos=episode.row("qpos", current).float(),
            future_spatial_tokens=episode.rows("spatial_tokens", indices).float(),
            future_spatial_view_mask=episode.rows("spatial_view_mask", indices).bool(),
            future_qpos=episode.rows("qpos", indices).float(),
            future_progress=torch.tensor(
                [index / max(steps - 1, 1) for index in indices], dtype=torch.float32
            ),
            horizon_mask=torch.tensor(
                [current + value < steps for value in self.prediction_horizons],
                dtype=torch.bool,
            ),
        )
        return item


def legal_model_inputs(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return only fields allowed by the deployed R13 forward schema."""

    missing = set(LEGAL_INPUT_KEYS) - set(batch)
    if missing:
        raise ValueError(f"R13 batch misses legal inputs: {sorted(missing)}")
    return {key: batch[key] for key in LEGAL_INPUT_KEYS}
