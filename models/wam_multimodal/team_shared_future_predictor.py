"""Team/shared S2-R4 predictor initialized from the accepted R3 local path."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from models.wam_multimodal.local_future_predictor import (
    LocalActionConditionedFuturePredictor,
    LocalFuturePredictorConfig,
)


@dataclass(frozen=True)
class TeamSharedFuturePredictorConfig:
    layers: int = 2
    heads: int = 8
    ffn_dim: int = 1536
    dropout: float = 0.1
    own_residual_max: float = 0.1

    def __post_init__(self) -> None:
        if min(self.layers, self.heads, self.ffn_dim) <= 0:
            raise ValueError("team/shared transformer dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if not 0.0 < self.own_residual_max <= 1.0:
            raise ValueError("own_residual_max must be in (0,1]")

    @classmethod
    def from_dict(
        cls,
        value: dict[str, object],
    ) -> "TeamSharedFuturePredictorConfig":
        return cls(**value)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TeamSharedFuturePrediction:
    own_state: Tensor
    own_visual: Tensor
    peer_state: Tensor
    peer_visual: Tensor
    shared_visual: Tensor


class TeamSharedFuturePredictor(nn.Module):
    """Predict own, pairwise peer, and global-slot futures for each focal agent."""

    def __init__(
        self,
        local_config: LocalFuturePredictorConfig,
        team_config: TeamSharedFuturePredictorConfig,
    ) -> None:
        super().__init__()
        if local_config.d_model % team_config.heads:
            raise ValueError("local d_model must be divisible by team heads")
        self.local_config = local_config
        self.team_config = team_config
        self.local_predictor = LocalActionConditionedFuturePredictor(
            local_config
        )
        self.shared_projection = nn.Sequential(
            nn.LayerNorm(local_config.visual_latent_dim),
            nn.Linear(local_config.visual_latent_dim, local_config.d_model),
            nn.GELU(),
            nn.Linear(local_config.d_model, local_config.d_model),
        )
        self.team_agent_norm = nn.LayerNorm(local_config.d_model)
        self.slot_embedding = nn.Parameter(
            torch.randn(
                1,
                1,
                local_config.max_agents + 1,
                local_config.d_model,
            )
            * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=local_config.d_model,
            nhead=team_config.heads,
            dim_feedforward=team_config.ffn_dim,
            dropout=team_config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.team_encoder = nn.TransformerEncoder(
            layer,
            num_layers=team_config.layers,
            enable_nested_tensor=False,
        )
        self.own_state_residual = _head(
            local_config.d_model,
            local_config.state_dim,
        )
        self.own_visual_residual = _head(
            local_config.d_model,
            local_config.visual_latent_dim,
        )
        self.peer_state_head = _head(
            local_config.d_model,
            local_config.state_dim,
        )
        self.peer_visual_head = _head(
            local_config.d_model,
            local_config.visual_latent_dim,
        )
        self.shared_visual_head = _head(
            local_config.d_model,
            local_config.visual_latent_dim,
        )
        self.own_residual_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        current_state: Tensor,
        current_visual_latent: Tensor,
        shared_visual_latent: Tensor,
        candidate_actions: Tensor,
        valid_agent_mask: Tensor,
        *,
        actions_by_focal: Tensor | None = None,
    ) -> TeamSharedFuturePrediction:
        config = self.local_config
        batch_size = current_state.shape[0]
        if shared_visual_latent.shape != (
            batch_size,
            config.visual_grid_tokens,
            config.visual_latent_dim,
        ):
            raise ValueError("shared_visual_latent has an invalid shape")
        if actions_by_focal is None:
            actions_by_focal = candidate_actions[:, None].expand(
                -1,
                config.max_agents,
                -1,
                -1,
                -1,
            )
        if actions_by_focal.shape != (
            batch_size,
            config.max_agents,
            config.max_agents,
            config.action_horizon,
            config.action_dim,
        ):
            raise ValueError(
                "actions_by_focal must be [B,focal,target,H,action_dim]"
            )

        local_context = self.local_predictor.encode_context(
            current_state,
            current_visual_latent,
            candidate_actions,
            valid_agent_mask,
            valid_agent_mask,
        )
        local_state, local_visual = self.local_predictor.decode_future(
            local_context,
            valid_agent_mask,
        )

        # Peer/shared objectives must not rewrite the accepted local feature
        # path; only the own-target objective updates the R3-derived module.
        state_token = self.local_predictor.state_projection(
            current_state
        ).detach()
        visual_token = self.local_predictor.visual_projection(
            current_visual_latent
        ).mean(dim=2).detach()
        action_token = self.local_predictor.action_projection(
            actions_by_focal
        ).detach()
        action_token = (
            action_token
            + self.local_predictor.action_position[:, None]
        ).mean(dim=3)
        agent_tokens = self.team_agent_norm(
            state_token[:, None] + visual_token[:, None] + action_token
        )
        shared_token = self.shared_projection(
            shared_visual_latent
        ).mean(dim=1)
        shared_tokens = shared_token[:, None, None].expand(
            -1,
            config.max_agents,
            -1,
            -1,
        )
        tokens = torch.cat((shared_tokens, agent_tokens), dim=2)
        tokens = tokens + self.slot_embedding
        valid_sequence = torch.cat(
            (
                torch.ones(
                    batch_size,
                    1,
                    dtype=torch.bool,
                    device=valid_agent_mask.device,
                ),
                valid_agent_mask,
            ),
            dim=1,
        )
        padding = ~valid_sequence[:, None].expand(
            -1,
            config.max_agents,
            -1,
        )
        encoded = self.team_encoder(
            tokens.reshape(
                batch_size * config.max_agents,
                config.max_agents + 1,
                config.d_model,
            ),
            src_key_padding_mask=padding.reshape(
                batch_size * config.max_agents,
                config.max_agents + 1,
            ),
        ).reshape(
            batch_size,
            config.max_agents,
            config.max_agents + 1,
            config.d_model,
        )
        encoded_shared = encoded[:, :, 0]
        encoded_agents = encoded[:, :, 1:]
        focal_index = torch.arange(
            config.max_agents,
            device=current_state.device,
        )
        focal_context = encoded_agents[:, focal_index, focal_index]

        future_position = self.local_predictor.future_position
        grid_position = self.local_predictor.grid_position
        own_future = focal_context[:, :, None] + future_position
        own_gate = self.team_config.own_residual_max * torch.tanh(
            self.own_residual_gate
        )
        own_state = local_state + own_gate * self.own_state_residual(own_future)
        own_visual = local_visual + own_gate * self.own_visual_residual(
            own_future[:, :, :, None] + grid_position
        )

        pair_context = (
            encoded_agents
            + focal_context[:, :, None]
        )
        pair_future = pair_context[:, :, :, None] + future_position[:, None]
        peer_state = self.peer_state_head(pair_future)
        peer_visual = self.peer_visual_head(
            pair_future[:, :, :, :, None] + grid_position[:, None]
        )
        shared_future = encoded_shared[:, :, None] + future_position
        shared_visual = self.shared_visual_head(
            shared_future[:, :, :, None] + grid_position
        )

        focal_valid = valid_agent_mask[:, :, None, None, None]
        pair_valid = (
            valid_agent_mask[:, :, None]
            & valid_agent_mask[:, None, :]
        )
        own_state = own_state * valid_agent_mask[:, :, None, None]
        own_visual = own_visual * focal_valid
        peer_state = peer_state * pair_valid[:, :, :, None, None]
        peer_visual = peer_visual * pair_valid[:, :, :, None, None, None]
        shared_visual = shared_visual * focal_valid
        return TeamSharedFuturePrediction(
            own_state=own_state,
            own_visual=own_visual,
            peer_state=peer_state,
            peer_visual=peer_visual,
            shared_visual=shared_visual,
        )


def _head(d_model: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Linear(d_model, output_dim),
    )


__all__ = [
    "TeamSharedFuturePrediction",
    "TeamSharedFuturePredictor",
    "TeamSharedFuturePredictorConfig",
]
