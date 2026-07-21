"""Action-conditioned future visual-latent prediction head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


CANONICAL_FUTURE_HORIZONS = (1, 2, 4, 8)


@dataclass(frozen=True)
class FutureLatentHeadConfig:
    """Dimensions for predicting frozen-teacher pooled visual latents."""

    planning_feature_dim: int
    action_dim: int = 8
    visual_dim: int = 512
    latent_dim: int = 512
    action_hidden_dim: int = 512
    hidden_dim: int = 2048
    horizons: tuple[int, ...] = CANONICAL_FUTURE_HORIZONS

    def __post_init__(self) -> None:
        for name in (
            "planning_feature_dim",
            "action_dim",
            "visual_dim",
            "latent_dim",
            "action_hidden_dim",
            "hidden_dim",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        horizons = tuple(int(value) for value in self.horizons)
        if not horizons or any(value <= 0 for value in horizons):
            raise ValueError("future latent horizons must be positive")
        if tuple(sorted(set(horizons))) != horizons:
            raise ValueError("future latent horizons must be sorted and unique")
        object.__setattr__(self, "horizons", horizons)


class ActionConditionedFutureLatentHead(nn.Module):
    """Predict teacher-space pooled latents without accepting future images.

    The action sequence is encoded autoregressively and sampled at each target
    horizon.  The public signature deliberately has no target-image argument;
    frozen teacher targets are constructed by the trainer on a separate path.
    """

    def __init__(self, config: FutureLatentHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.planning_projection = nn.Sequential(
            nn.LayerNorm(config.planning_feature_dim),
            nn.Linear(config.planning_feature_dim, config.latent_dim),
        )
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(config.visual_dim),
            nn.Linear(config.visual_dim, config.latent_dim),
        )
        self.action_encoder = nn.GRU(
            input_size=config.action_dim,
            hidden_size=config.action_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.action_projection: nn.Module
        if config.action_hidden_dim == config.latent_dim:
            self.action_projection = nn.Identity()
        else:
            self.action_projection = nn.Linear(
                config.action_hidden_dim, config.latent_dim
            )
        self.horizon_embedding = nn.Embedding(len(config.horizons), config.latent_dim)
        self.prediction = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        self.register_buffer(
            "action_horizon_indices",
            torch.tensor(config.horizons, dtype=torch.int64) - 1,
            persistent=True,
        )

    @property
    def horizons(self) -> tuple[int, ...]:
        return self.config.horizons

    def forward(
        self,
        planning_features: Tensor,
        visual_tokens: Tensor,
        candidate_actions: Tensor,
    ) -> Tensor:
        if (
            planning_features.ndim != 2
            or planning_features.shape[-1] != self.config.planning_feature_dim
        ):
            raise ValueError(
                "planning_features must have shape "
                f"[B,{self.config.planning_feature_dim}]"
            )
        batch_size = planning_features.shape[0]
        if visual_tokens.ndim != 3 or visual_tokens.shape[0] != batch_size:
            raise ValueError("visual_tokens must have shape [B,N,D]")
        if visual_tokens.shape[1] <= 0 or visual_tokens.shape[2] != self.config.visual_dim:
            raise ValueError(
                f"visual_tokens must end in non-empty [{self.config.visual_dim}]"
            )
        if (
            candidate_actions.ndim != 3
            or candidate_actions.shape[0] != batch_size
            or candidate_actions.shape[2] != self.config.action_dim
        ):
            raise ValueError(
                "candidate_actions must have shape "
                f"[B,H,{self.config.action_dim}]"
            )
        if candidate_actions.shape[1] < max(self.config.horizons):
            raise ValueError(
                "candidate_actions are shorter than the largest future horizon"
            )
        if not all(
            value.device == planning_features.device
            for value in (visual_tokens, candidate_actions)
        ):
            raise TypeError("future-head inputs must share a device")
        if not all(
            value.dtype == planning_features.dtype
            for value in (visual_tokens, candidate_actions)
        ):
            raise TypeError("future-head inputs must share a dtype")

        action_features, _ = self.action_encoder(candidate_actions)
        selected_actions = action_features.index_select(
            1, self.action_horizon_indices
        )
        selected_actions = self.action_projection(selected_actions)
        planning = self.planning_projection(planning_features).unsqueeze(1)
        visual = self.visual_projection(visual_tokens.mean(dim=1)).unsqueeze(1)
        horizon_ids = torch.arange(
            len(self.config.horizons), device=planning_features.device
        )
        horizon = self.horizon_embedding(horizon_ids).unsqueeze(0)
        combined = planning + visual + selected_actions + horizon
        return self.prediction(combined)


__all__ = [
    "ActionConditionedFutureLatentHead",
    "CANONICAL_FUTURE_HORIZONS",
    "FutureLatentHeadConfig",
]
