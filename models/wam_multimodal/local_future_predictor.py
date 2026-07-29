"""Off-path local future predictor for S2-R3 action-conditioning tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LocalFuturePredictorConfig:
    max_agents: int = 4
    state_dim: int = 18
    action_dim: int = 8
    action_horizon: int = 100
    future_horizons: tuple[int, ...] = (1, 25, 50, 100)
    visual_grid_tokens: int = 4
    visual_latent_dim: int = 256
    d_model: int = 384
    ffn_dim: int = 1536
    layers: int = 3
    heads: int = 8
    dropout: float = 0.1

    def __post_init__(self) -> None:
        integer_fields = (
            self.max_agents,
            self.state_dim,
            self.action_dim,
            self.action_horizon,
            self.visual_grid_tokens,
            self.visual_latent_dim,
            self.d_model,
            self.ffn_dim,
            self.layers,
            self.heads,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("local future predictor dimensions must be positive")
        if not self.future_horizons or any(
            value <= 0 or value > self.action_horizon
            for value in self.future_horizons
        ):
            raise ValueError("future horizons must lie inside the action chunk")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "LocalFuturePredictorConfig":
        payload = dict(value)
        if "future_horizons" in payload:
            payload["future_horizons"] = tuple(
                int(item) for item in payload["future_horizons"]  # type: ignore[arg-type]
            )
        return cls(**payload)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["future_horizons"] = list(self.future_horizons)
        return value


class LocalActionConditionedFuturePredictor(nn.Module):
    """Shared local predictor; W0/W1 differ only in the action input mask."""

    def __init__(self, config: LocalFuturePredictorConfig) -> None:
        super().__init__()
        self.config = config
        self.state_projection = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(config.visual_latent_dim),
            nn.Linear(config.visual_latent_dim, config.d_model),
        )
        self.action_projection = nn.Sequential(
            nn.LayerNorm(config.action_dim),
            nn.Linear(config.action_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.action_position = nn.Parameter(
            torch.randn(1, 1, config.action_horizon, config.d_model) * 0.02
        )
        self.future_position = nn.Parameter(
            torch.randn(1, 1, len(config.future_horizons), config.d_model)
            * 0.02
        )
        self.grid_position = nn.Parameter(
            torch.randn(
                1, 1, 1, config.visual_grid_tokens, config.d_model
            )
            * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.layers,
            enable_nested_tensor=False,
        )
        self.state_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.state_dim),
        )
        self.visual_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.visual_latent_dim),
        )

    def forward(
        self,
        current_state: Tensor,
        current_visual_latent: Tensor,
        candidate_actions: Tensor,
        valid_agent_mask: Tensor,
        action_condition_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        config = self.config
        if current_state.ndim != 3 or current_state.shape[1:] != (
            config.max_agents,
            config.state_dim,
        ):
            raise ValueError("current_state must be [B,A,state_dim]")
        batch_size = current_state.shape[0]
        if current_visual_latent.shape != (
            batch_size,
            config.max_agents,
            config.visual_grid_tokens,
            config.visual_latent_dim,
        ):
            raise ValueError("current_visual_latent has an invalid shape")
        if candidate_actions.shape != (
            batch_size,
            config.max_agents,
            config.action_horizon,
            config.action_dim,
        ):
            raise ValueError("candidate_actions has an invalid shape")
        if valid_agent_mask.shape != (batch_size, config.max_agents):
            raise ValueError("valid_agent_mask must be [B,A]")
        if action_condition_mask.shape != valid_agent_mask.shape:
            raise ValueError("action_condition_mask must be [B,A]")
        if bool((action_condition_mask & ~valid_agent_mask).any()):
            raise ValueError("invalid agents cannot expose candidate actions")

        state_token = self.state_projection(current_state)
        visual_tokens = self.visual_projection(current_visual_latent)
        action_tokens = (
            self.action_projection(candidate_actions) + self.action_position
        )
        action_summary = action_tokens.mean(dim=2)
        action_summary = action_summary * action_condition_mask[..., None]
        tokens = torch.cat(
            (
                state_token[:, :, None],
                visual_tokens,
                action_summary[:, :, None],
            ),
            dim=2,
        )
        flat = tokens.reshape(
            batch_size * config.max_agents,
            tokens.shape[2],
            config.d_model,
        )
        encoded = self.context_encoder(flat)
        context = encoded.mean(dim=1).reshape(
            batch_size, config.max_agents, config.d_model
        )
        context = context * valid_agent_mask[..., None]

        future = context[:, :, None] + self.future_position
        state_delta = self.state_head(future)
        visual_query = future[:, :, :, None] + self.grid_position
        visual_delta = self.visual_head(visual_query)
        state_delta = state_delta * valid_agent_mask[:, :, None, None]
        visual_delta = visual_delta * valid_agent_mask[:, :, None, None, None]
        return state_delta, visual_delta


__all__ = [
    "LocalActionConditionedFuturePredictor",
    "LocalFuturePredictorConfig",
]
