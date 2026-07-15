"""Numerically stable transforms shared by RWM rollout and training."""

from __future__ import annotations

import torch
from torch import Tensor


def wrap_to_pi(value: Tensor) -> Tensor:
    """Wrap radians to ``[-pi, pi)`` without breaking autograd."""

    return torch.remainder(value + torch.pi, 2.0 * torch.pi) - torch.pi


def symlog(value: Tensor) -> Tensor:
    return torch.sign(value) * torch.log1p(torch.abs(value))


def symexp(value: Tensor) -> Tensor:
    return torch.sign(value) * torch.expm1(torch.abs(value))


__all__ = ["symlog", "symexp", "wrap_to_pi"]
