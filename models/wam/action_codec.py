"""Invertible per-dimension action codecs shared by training and deployment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, overload

import numpy as np
import torch
from torch import Tensor, nn

from models.wam.normalizer import NormalizationStats


AFFINE_ACTION_CODEC_VERSION = "wam.action_codec.affine/1"
CANONICAL_ACTION_DOMAIN = "canonical_unit_action"


@dataclass(frozen=True)
class AffineActionCodecConfig:
    """Serializable raw-controller to canonical ``[-1, 1]`` contract."""

    codec_id: str
    low: tuple[float, ...]
    high: tuple[float, ...]
    raw_domain: str
    encoded_domain: str = CANONICAL_ACTION_DOMAIN
    format_version: str = AFFINE_ACTION_CODEC_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        low = tuple(float(value) for value in self.low)
        high = tuple(float(value) for value in self.high)
        if self.format_version != AFFINE_ACTION_CODEC_VERSION:
            raise ValueError(
                f"unsupported action codec format {self.format_version!r}"
            )
        if not str(self.codec_id) or not str(self.raw_domain):
            raise ValueError("codec_id and raw_domain cannot be empty")
        if self.encoded_domain != CANONICAL_ACTION_DOMAIN:
            raise ValueError(
                f"encoded_domain must be {CANONICAL_ACTION_DOMAIN!r}"
            )
        if not low or len(low) != len(high):
            raise ValueError("action codec low/high must have equal positive length")
        low_array = np.asarray(low, dtype=np.float64)
        high_array = np.asarray(high, dtype=np.float64)
        if not np.isfinite(low_array).all() or not np.isfinite(high_array).all():
            raise ValueError("action codec bounds must be finite")
        if np.any(high_array <= low_array):
            raise ValueError("every action codec high bound must exceed low")
        metadata = _plain(self.metadata)
        if not isinstance(metadata, dict):
            raise TypeError("action codec metadata must be a mapping")
        object.__setattr__(self, "codec_id", str(self.codec_id))
        object.__setattr__(self, "raw_domain", str(self.raw_domain))
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "metadata", metadata)

    @property
    def action_dim(self) -> int:
        return len(self.low)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "codec_id": self.codec_id,
            "action_dim": self.action_dim,
            "raw_domain": self.raw_domain,
            "encoded_domain": self.encoded_domain,
            "encoded_range": [-1.0, 1.0],
            "low": list(self.low),
            "high": list(self.high),
            "metadata": _plain(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AffineActionCodecConfig":
        raw = dict(payload)
        declared_dim = raw.pop("action_dim", None)
        encoded_range = raw.pop("encoded_range", None)
        if encoded_range is not None and list(encoded_range) != [-1.0, 1.0]:
            raise ValueError("affine action codec encoded_range must be [-1,1]")
        config = cls(
            format_version=str(raw.pop("format_version")),
            codec_id=str(raw.pop("codec_id")),
            low=tuple(raw.pop("low")),
            high=tuple(raw.pop("high")),
            raw_domain=str(raw.pop("raw_domain")),
            encoded_domain=str(
                raw.pop("encoded_domain", CANONICAL_ACTION_DOMAIN)
            ),
            metadata=raw.pop("metadata", {}),
        )
        if raw:
            raise ValueError(f"unknown action codec fields: {sorted(raw)}")
        if declared_dim is not None and int(declared_dim) != config.action_dim:
            raise ValueError("action codec action_dim disagrees with its bounds")
        return config

    @classmethod
    def load(cls, path: str | Path) -> "AffineActionCodecConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("action codec JSON root must be an object")
        return cls.from_dict(payload)

    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict())).hexdigest()


class AffineActionCodec(nn.Module):
    """Map physical controller actions to/from a canonical unit cube.

    Encoding is differentiable for tensors.  Training input is validated instead
    of silently clipped; decoding clips model output to the declared canonical
    range before converting it back to physical controller units.
    """

    def __init__(self, config: AffineActionCodecConfig) -> None:
        super().__init__()
        self.config = config
        low = torch.tensor(config.low, dtype=torch.float32)
        high = torch.tensor(config.high, dtype=torch.float32)
        self.register_buffer("low", low, persistent=True)
        self.register_buffer("high", high, persistent=True)
        self.register_buffer("center", 0.5 * (low + high), persistent=True)
        self.register_buffer("half_range", 0.5 * (high - low), persistent=True)

    @property
    def action_dim(self) -> int:
        return self.config.action_dim

    @property
    def semantic_sha256(self) -> str:
        return self.config.sha256()

    def forward(self, actions: Tensor) -> Tensor:
        return self.encode(actions)

    @overload
    def encode(self, actions: Tensor, *, validate: bool = True) -> Tensor: ...

    @overload
    def encode(
        self, actions: np.ndarray, *, validate: bool = True
    ) -> np.ndarray: ...

    def encode(
        self, actions: Tensor | np.ndarray, *, validate: bool = True
    ) -> Tensor | np.ndarray:
        """Encode raw actions without clipping out-of-contract training data."""

        if isinstance(actions, Tensor):
            self._validate_tensor(actions)
            low = self.low.to(actions)
            high = self.high.to(actions)
            if validate and actions.numel() and (
                bool(torch.any(actions < low - 1e-6))
                or bool(torch.any(actions > high + 1e-6))
            ):
                raise ValueError("raw action lies outside the declared codec bounds")
            return (actions - self.center.to(actions)) / self.half_range.to(actions)
        array = np.asarray(actions)
        self._validate_numpy(array)
        low_np = np.asarray(self.config.low, dtype=array.dtype)
        high_np = np.asarray(self.config.high, dtype=array.dtype)
        if validate and array.size and (
            np.any(array < low_np - 1e-6) or np.any(array > high_np + 1e-6)
        ):
            raise ValueError("raw action lies outside the declared codec bounds")
        center = 0.5 * (low_np + high_np)
        half_range = 0.5 * (high_np - low_np)
        return np.asarray((array - center) / half_range, dtype=array.dtype)

    @overload
    def decode(self, actions: Tensor, *, clip: bool = True) -> Tensor: ...

    @overload
    def decode(self, actions: np.ndarray, *, clip: bool = True) -> np.ndarray: ...

    def decode(
        self, actions: Tensor | np.ndarray, *, clip: bool = True
    ) -> Tensor | np.ndarray:
        """Decode canonical actions to physical controller units."""

        if isinstance(actions, Tensor):
            self._validate_tensor(actions)
            canonical = actions.clamp(-1.0, 1.0) if clip else actions
            if not clip and canonical.numel() and bool(torch.any(canonical.abs() > 1.0)):
                raise ValueError("canonical action lies outside [-1,1]")
            return canonical * self.half_range.to(actions) + self.center.to(actions)
        array = np.asarray(actions)
        self._validate_numpy(array)
        if clip:
            canonical_np = np.clip(array, -1.0, 1.0)
        else:
            if array.size and np.any(np.abs(array) > 1.0):
                raise ValueError("canonical action lies outside [-1,1]")
            canonical_np = array
        center = np.asarray(self.config.low, dtype=array.dtype)
        center = 0.5 * (center + np.asarray(self.config.high, dtype=array.dtype))
        half_range = 0.5 * (
            np.asarray(self.config.high, dtype=array.dtype)
            - np.asarray(self.config.low, dtype=array.dtype)
        )
        return np.asarray(canonical_np * half_range + center, dtype=array.dtype)

    def encode_normalization(
        self, raw: NormalizationStats
    ) -> NormalizationStats:
        """Convert raw-action moments to the exact affine canonical moments."""

        if raw.action_mean.shape != (self.action_dim,):
            raise ValueError("raw normalization action dimension differs from codec")
        center = np.asarray(self.config.low, dtype=np.float32)
        center = 0.5 * (center + np.asarray(self.config.high, dtype=np.float32))
        half_range = 0.5 * (
            np.asarray(self.config.high, dtype=np.float32)
            - np.asarray(self.config.low, dtype=np.float32)
        )
        return NormalizationStats(
            state_mean=raw.state_mean.copy(),
            state_std=raw.state_std.copy(),
            action_mean=(raw.action_mean - center) / half_range,
            action_std=raw.action_std / half_range,
            delta_mean=raw.delta_mean.copy(),
            delta_std=raw.delta_std.copy(),
            reward_mean=raw.reward_mean.copy(),
            reward_std=raw.reward_std.copy(),
        )

    def _validate_tensor(self, actions: Tensor) -> None:
        if not torch.is_floating_point(actions):
            raise TypeError("actions must be floating point")
        if actions.ndim == 0 or actions.shape[-1] != self.action_dim:
            raise ValueError(f"actions must end in dimension {self.action_dim}")
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("actions contain NaN or Inf")

    def _validate_numpy(self, actions: np.ndarray) -> None:
        if not np.issubdtype(actions.dtype, np.floating):
            raise TypeError("actions must be floating point")
        if actions.ndim == 0 or actions.shape[-1] != self.action_dim:
            raise ValueError(f"actions must end in dimension {self.action_dim}")
        if not np.isfinite(actions).all():
            raise ValueError("actions contain NaN or Inf")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "AFFINE_ACTION_CODEC_VERSION",
    "CANONICAL_ACTION_DOMAIN",
    "AffineActionCodec",
    "AffineActionCodecConfig",
]
