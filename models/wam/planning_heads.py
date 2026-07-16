"""Phase 3 action-prior and terminal-value heads.

The accepted Phase 2 ensemble stays frozen.  These small heads consume the
deployable recurrent belief feature exposed by one accepted ensemble member;
no environment or privileged field is part of their tensor interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from models.wam.config import WAMPlanningHeadConfig
from models.wam.rollout import symexp


@dataclass(frozen=True)
class WAMPlanningHeadOutput:
    action_mean: Tensor
    action_log_std: Tensor
    value_symlog: Tensor

    @property
    def value(self) -> Tensor:
        return symexp(self.value_symlog.float().clamp(-20.0, 20.0))


class WAMPlanningHeads(nn.Module):
    """Tanh-Gaussian behavior prior plus scalar Monte-Carlo return head."""

    def __init__(self, config: WAMPlanningHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.action_network = _mlp(
            config.feature_dim,
            2 * config.action_dim,
            config.hidden_dim,
            config.hidden_layers,
        )
        self.value_network = _mlp(
            config.feature_dim,
            1,
            config.hidden_dim,
            config.hidden_layers,
        )
        # A zero raw log-std would map to the midpoint (-2 with the default
        # bounds), making saturated grip commands produce an enormous initial
        # pre-tanh NLL.  Start at unit standard deviation; training can then
        # contract uncertainty without drowning the motion dimensions.
        final_action = self.action_network[-1]
        assert isinstance(final_action, nn.Linear)
        target = 2.0 * (0.0 - config.min_log_std) / (
            config.max_log_std - config.min_log_std
        ) - 1.0
        raw = math.atanh(max(-0.999, min(0.999, target)))
        with torch.no_grad():
            final_action.bias[config.action_dim :].fill_(raw)

    def forward(self, features: Tensor) -> WAMPlanningHeadOutput:
        self._validate_features(features)
        raw_action = self.action_network(features)
        mean, raw_log_std = raw_action.chunk(2, dim=-1)
        log_std = self.config.min_log_std + 0.5 * (
            torch.tanh(raw_log_std) + 1.0
        ) * (self.config.max_log_std - self.config.min_log_std)
        return WAMPlanningHeadOutput(
            action_mean=mean,
            action_log_std=log_std,
            value_symlog=self.value_network(features),
        )

    def deterministic_action(self, features: Tensor) -> Tensor:
        return torch.tanh(self(features).action_mean)

    def sample_action(
        self,
        features: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        output = self(features)
        noise = torch.randn(
            output.action_mean.shape,
            device=output.action_mean.device,
            dtype=output.action_mean.dtype,
            generator=generator,
        )
        return torch.tanh(
            output.action_mean + output.action_log_std.exp() * noise
        )

    def action_nll(self, features: Tensor, actions: Tensor) -> Tensor:
        """Per-sample negative log likelihood under a tanh Gaussian."""

        output = self(features)
        if actions.shape != output.action_mean.shape:
            raise ValueError(
                f"actions must have shape {tuple(output.action_mean.shape)}, "
                f"got {tuple(actions.shape)}"
            )
        clipped = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh = torch.atanh(clipped)
        inverse_std = torch.exp(-output.action_log_std)
        gaussian = 0.5 * (
            (pre_tanh - output.action_mean) * inverse_std
        ).square() + output.action_log_std + 0.5 * math.log(2.0 * math.pi)
        # Change-of-variables for a=tanh(u).  Summing dimensions yields a
        # proper sequence-independent BC objective for bounded actions.
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


__all__ = ["WAMPlanningHeadOutput", "WAMPlanningHeads"]
