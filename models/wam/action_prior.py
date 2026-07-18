"""Behavior-cloning action prior on top of the frozen world-model belief."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from models.wam.config import ActionPriorConfig


@dataclass(frozen=True)
class ActionPriorOutput:
    mean: Tensor
    log_std: Tensor


class ActionPrior(nn.Module):
    """Tanh-Gaussian action distribution conditioned on a recurrent belief."""

    def __init__(self, config: ActionPriorConfig) -> None:
        super().__init__()
        self.config = config
        self.action_network = _mlp(
            config.feature_dim,
            2 * config.action_dim,
            config.hidden_dim,
            config.hidden_layers,
        )
        final = self.action_network[-1]
        assert isinstance(final, nn.Linear)
        target = 2.0 * (0.0 - config.min_log_std) / (
            config.max_log_std - config.min_log_std
        ) - 1.0
        raw = math.atanh(max(-0.999, min(0.999, target)))
        with torch.no_grad():
            final.bias[config.action_dim :].fill_(raw)

    def forward(self, features: Tensor) -> ActionPriorOutput:
        self._validate_features(features)
        mean, raw_log_std = self.action_network(features).chunk(2, dim=-1)
        log_std = self.config.min_log_std + 0.5 * (
            torch.tanh(raw_log_std) + 1.0
        ) * (self.config.max_log_std - self.config.min_log_std)
        return ActionPriorOutput(mean=mean, log_std=log_std)

    def deterministic_action(self, features: Tensor) -> Tensor:
        return torch.tanh(self(features).mean)

    def sample_action(
        self,
        features: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        output = self(features)
        noise = torch.randn(
            output.mean.shape,
            device=output.mean.device,
            dtype=output.mean.dtype,
            generator=generator,
        )
        return torch.tanh(output.mean + output.log_std.exp() * noise)

    def nll(self, features: Tensor, actions: Tensor) -> Tensor:
        """Return one bounded-action negative log likelihood per sample."""

        output = self(features)
        if actions.shape != output.mean.shape:
            raise ValueError(
                f"actions must have shape {tuple(output.mean.shape)}, "
                f"got {tuple(actions.shape)}"
            )
        clipped = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh = torch.atanh(clipped)
        gaussian = 0.5 * (
            (pre_tanh - output.mean) * torch.exp(-output.log_std)
        ).square() + output.log_std + 0.5 * math.log(2.0 * math.pi)
        log_det = torch.log(1.0 - clipped.square() + 1e-6)
        return (gaussian + log_det).sum(dim=-1)

    def _validate_features(self, features: Tensor) -> None:
        if features.ndim < 2 or features.shape[-1] != self.config.feature_dim:
            raise ValueError(
                f"features must end in {self.config.feature_dim}, "
                f"got {tuple(features.shape)}"
            )
        if not torch.is_floating_point(features):
            raise TypeError("features must be floating point")


def _mlp(input_dim: int, output_dim: int, hidden_dim: int, layers: int) -> nn.Sequential:
    modules: list[nn.Module] = []
    for index in range(layers):
        modules.extend(
            (
                nn.Linear(input_dim if index == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
        )
    modules.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*modules)


__all__ = ["ActionPrior", "ActionPriorOutput"]
