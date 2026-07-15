"""Episode-bootstrap utilities and diagnostics for Phase 2 RWM-U training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from models.wam import RWMUEnsemble


@dataclass(frozen=True)
class EpisodeBootstrap:
    """One episode-level draw with replacement and its fragment indices."""

    seed: int
    episode_draws: tuple[int, ...]
    sample_indices: np.ndarray

    @property
    def unique_episode_count(self) -> int:
        return len(set(self.episode_draws))

    def manifest(self) -> dict[str, Any]:
        counts = np.bincount(
            np.asarray(self.episode_draws, dtype=np.int64),
            minlength=max(self.episode_draws, default=-1) + 1,
        )
        return {
            "seed": self.seed,
            "episode_draws": list(self.episode_draws),
            "episode_draw_counts": {
                str(index): int(count)
                for index, count in enumerate(counts)
                if count > 0
            },
            "draw_count": len(self.episode_draws),
            "unique_episode_count": self.unique_episode_count,
            "fragment_count": len(self.sample_indices),
        }


def make_episode_bootstrap(dataset: Dataset, *, seed: int) -> EpisodeBootstrap:
    """Bootstrap whole episodes; fragments from an episode always move together."""

    sample_episode = _sample_episode_positions(dataset)
    if sample_episode.size != len(dataset):
        raise RuntimeError("episode-position vector does not match dataset length")
    episode_count = int(sample_episode.max()) + 1
    if episode_count < 2:
        raise ValueError("episode bootstrap requires at least two episodes")
    draws = np.random.default_rng(seed).integers(
        0, episode_count, size=episode_count, dtype=np.int64
    )
    order = np.argsort(sample_episode, kind="stable")
    counts = np.bincount(sample_episode, minlength=episode_count)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
    indices = np.concatenate(
        [order[offsets[int(draw)] : offsets[int(draw) + 1]] for draw in draws]
    )
    return EpisodeBootstrap(
        seed=int(seed),
        episode_draws=tuple(int(value) for value in draws),
        sample_indices=indices.astype(np.int64, copy=False),
    )


@torch.no_grad()
def ensemble_parameter_diversity(ensemble: RWMUEnsemble) -> dict[str, float | bool]:
    """Detect accidental cloned members before uncertainty evaluation."""

    vectors = [
        torch.cat([parameter.detach().float().cpu().reshape(-1) for parameter in member.parameters()])
        for member in ensemble.members
    ]
    distances = [
        float(torch.sqrt(torch.mean((vectors[left] - vectors[right]).square())))
        for left in range(len(vectors))
        for right in range(left + 1, len(vectors))
    ]
    minimum = min(distances, default=0.0)
    maximum = max(distances, default=0.0)
    return {
        "passed": bool(minimum > 0.0),
        "pair_count": len(distances),
        "minimum_parameter_rms_distance": minimum,
        "maximum_parameter_rms_distance": maximum,
    }


def _sample_episode_positions(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "_sample_episode"):
        return np.asarray(getattr(dataset, "_sample_episode"), dtype=np.int64)
    if hasattr(dataset, "index"):
        return np.asarray(
            [int(item.file_index) for item in getattr(dataset, "index")],
            dtype=np.int64,
        )
    raise TypeError(
        "dataset does not expose episode-safe indexing required by Phase 2 bootstrap"
    )


__all__ = [
    "EpisodeBootstrap",
    "ensemble_parameter_diversity",
    "make_episode_bootstrap",
]
