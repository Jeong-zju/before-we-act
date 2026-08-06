"""Trainable query bridge from full rectangular DINO tokens to R12 actions."""
from __future__ import annotations

import torch
from torch import nn

from before_we_act.contracts import TeamBeliefState


class SpatialQueryBridge(nn.Module):
    """Compress five ordered 6x8 view grids without a gradient-blocking gate.

    The bridge follows the small learned-query pattern used by BLIP-2's
    Q-Former, adapted to causal robot observations.  It does not consume text,
    task identity, agent identity, future frames, or simulator state.
    """

    def __init__(
        self,
        *,
        spatial_dim: int = 768,
        belief_dim: int = 96,
        views: int = 5,
        grid_height: int = 6,
        grid_width: int = 8,
        queries: int = 16,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if min(
            spatial_dim,
            belief_dim,
            views,
            grid_height,
            grid_width,
            queries,
            heads,
        ) <= 0 or belief_dim % heads:
            raise ValueError("invalid R12 spatial query bridge dimensions")
        self.spatial_dim = int(spatial_dim)
        self.belief_dim = int(belief_dim)
        self.views = int(views)
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.query_count = int(queries)
        self.norm = nn.LayerNorm(spatial_dim)
        self.projection = nn.Linear(spatial_dim, belief_dim)
        self.view_embedding = nn.Parameter(torch.empty(views, belief_dim))
        self.row_embedding = nn.Parameter(torch.empty(grid_height, belief_dim))
        self.column_embedding = nn.Parameter(torch.empty(grid_width, belief_dim))
        self.queries = nn.Parameter(torch.empty(queries, belief_dim))
        self.cross_attention = nn.MultiheadAttention(
            belief_dim, heads, batch_first=True
        )
        self.query_norm = nn.LayerNorm(belief_dim)
        self.query_ffn = nn.Sequential(
            nn.Linear(belief_dim, belief_dim * 4),
            nn.GELU(),
            nn.Linear(belief_dim * 4, belief_dim),
        )
        for value in (
            self.view_embedding,
            self.row_embedding,
            self.column_embedding,
            self.queries,
        ):
            nn.init.normal_(value, std=0.02)

    def _spatial(
        self, spatial_tokens: torch.Tensor, spatial_view_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = len(spatial_tokens)
        patches = self.grid_height * self.grid_width
        expected = (batch, self.views, patches, self.spatial_dim)
        if tuple(spatial_tokens.shape) != expected:
            raise ValueError(
                f"R12 full spatial tokens {tuple(spatial_tokens.shape)} != {expected}"
            )
        if tuple(spatial_view_mask.shape) != (batch, self.views):
            raise ValueError("R12 full spatial view mask shape differs")
        mask = spatial_view_mask.bool()
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every R12 sample requires at least one legal fixed view")
        values = self.projection(self.norm(spatial_tokens))
        position = (
            self.view_embedding[:, None, :]
            + self.row_embedding[:, None, :]
            .expand(-1, self.grid_width, -1)
            .reshape(1, patches, -1)
            + self.column_embedding[None, :, :]
            .expand(self.grid_height, -1, -1)
            .reshape(1, patches, -1)
        )
        values = (values + position[None]).reshape(
            batch, self.views * patches, self.belief_dim
        )
        token_mask = mask[:, :, None].expand(-1, -1, patches).reshape(batch, -1)
        return values, token_mask

    def forward(
        self,
        belief: TeamBeliefState,
        spatial_tokens: torch.Tensor,
        spatial_view_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        belief.validate()
        values, spatial_mask = self._spatial(spatial_tokens, spatial_view_mask)
        query = self.queries[None].expand(len(values), -1, -1)
        residual, _ = self.cross_attention(
            query=query,
            key=values,
            value=values,
            key_padding_mask=~spatial_mask,
            need_weights=False,
        )
        query = query + residual
        query = query + self.query_ffn(self.query_norm(query))
        belief_tokens = torch.cat(
            [belief.tokens, belief.agent_tokens, belief.consensus_token[:, None]], dim=1
        )
        belief_mask = torch.cat(
            [
                torch.ones(
                    belief.tokens.shape[:2],
                    dtype=torch.bool,
                    device=belief.tokens.device,
                ),
                belief.agent_mask.bool(),
                torch.ones(
                    (len(belief.tokens), 1),
                    dtype=torch.bool,
                    device=belief.tokens.device,
                ),
            ],
            dim=1,
        )
        return torch.cat([belief_tokens, query], dim=1), torch.cat(
            [belief_mask, torch.ones_like(query[..., 0], dtype=torch.bool)], dim=1
        )


__all__ = ["SpatialQueryBridge"]
