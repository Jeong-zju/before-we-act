"""Serializable normalization statistics shared by baselines and RWM models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

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

    def __post_init__(self) -> None:
        expected = {
            "state_mean": self.state_mean.shape,
            "state_std": self.state_mean.shape,
            "action_mean": self.action_mean.shape,
            "action_std": self.action_mean.shape,
            "delta_mean": self.state_mean.shape,
            "delta_std": self.state_mean.shape,
            "reward_mean": (1,),
            "reward_std": (1,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")
            if name.endswith("_std") and np.any(value <= 0.0):
                raise ValueError(f"{name} must be strictly positive")
            object.__setattr__(self, name, value)

    @classmethod
    def load(cls, path: str | Path) -> "NormalizationStats":
        with np.load(path, allow_pickle=False) as payload:
            missing = set(cls.__dataclass_fields__) - set(payload.files)
            if missing:
                raise ValueError(f"normalization file is missing {sorted(missing)}")
            return cls(**{name: payload[name] for name in cls.__dataclass_fields__})

    def save(self, path: str | Path) -> None:
        np.savez(path, **vars(self))

    def sha256(self) -> str:
        digest = hashlib.sha256()
        for name in sorted(vars(self)):
            value = np.ascontiguousarray(getattr(self, name), dtype=np.float32)
            digest.update(name.encode("utf-8"))
            digest.update(str(value.shape).encode("ascii"))
            digest.update(value.tobytes())
        return digest.hexdigest()

    def tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            name: torch.as_tensor(value, dtype=torch.float32, device=device)
            for name, value in vars(self).items()
        }


__all__ = ["NormalizationStats"]
