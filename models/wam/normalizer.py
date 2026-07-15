"""Serializable normalization statistics shared by baselines and RWM models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class NormalizationStats:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    delta_mean: np.ndarray
    delta_std: np.ndarray
    reward_mean: np.ndarray
    reward_std: np.ndarray

    def save(self, path: str) -> None:
        np.savez(path, **vars(self))

    def tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            name: torch.as_tensor(value, dtype=torch.float32, device=device)
            for name, value in vars(self).items()
        }


__all__ = ["NormalizationStats"]
