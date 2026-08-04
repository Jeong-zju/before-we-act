"""Aligned local RGB-D patch fusion used by wrist-only Stereo-ACT.

This module intentionally operates on tokens only. Camera geometry is handled
by the single native RGB-D sensor; no global view, peer observation, or second
camera image can enter this fusion block.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativePositionBias2D(nn.Module):
    """Per-head B(i,j)=b_row(delta_row)+b_col(delta_col) attention bias."""

    def __init__(self, heads: int, grid_h: int, grid_w: int):
        super().__init__()
        self.heads, self.grid_h, self.grid_w = heads, grid_h, grid_w
        self.row = nn.Parameter(torch.zeros(2 * grid_h - 1, heads))
        self.col = nn.Parameter(torch.zeros(2 * grid_w - 1, heads))
        nn.init.trunc_normal_(self.row, std=0.02)
        nn.init.trunc_normal_(self.col, std=0.02)

    def forward(self) -> torch.Tensor:
        rows = torch.arange(self.grid_h, device=self.row.device)
        cols = torch.arange(self.grid_w, device=self.col.device)
        rb = self.row[rows[:, None] - rows[None, :] + self.grid_h - 1]
        cb = self.col[cols[:, None] - cols[None, :] + self.grid_w - 1]
        # [query_row, query_col, key_row, key_col, heads] -> [heads, query, key]
        return (rb[:, None, :, None, :] + cb[None, :, None, :, :]).permute(4, 0, 1, 2, 3).reshape(
            self.heads, self.grid_h * self.grid_w, self.grid_h * self.grid_w
        )


class RelativeBiasCrossAttention(nn.Module):
    """RGB-query/depth-key attention over a matched H×W grid."""

    def __init__(self, d_model: int, heads: int, grid_h: int, grid_w: int, dropout: float = 0.1):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads, self.head_dim = heads, d_model // heads
        self.q, self.k, self.v, self.out = (nn.Linear(d_model, d_model) for _ in range(4))
        self.bias = RelativePositionBias2D(heads, grid_h, grid_w)
        self.dropout = dropout

    def forward(self, rgb_query: torch.Tensor, depth_key_value: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = rgb_query.shape
        if depth_key_value.shape != (batch, tokens, dim):
            raise ValueError("cross_relbias requires identical RGB/depth token grids")
        q = self.q(rgb_query).view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(depth_key_value).view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(depth_key_value).view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        # SDPA keeps this 30×40 full-grid operation on the 5090 fused-attention
        # path where available; the bias is an additive per-head attention mask.
        mask = self.bias().to(dtype=q.dtype, device=q.device).unsqueeze(0)
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        return self.out(output.transpose(1, 2).reshape(batch, tokens, dim))


class RGBDRegionFusionBlock(nn.Module):
    """RGB self-attention, depth self-attention, then relative-biased cross fusion."""

    def __init__(self, d_model: int, heads: int, grid_h: int, grid_w: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.rgb_norm, self.depth_norm, self.cross_norm, self.ffn_norm = (nn.LayerNorm(d_model) for _ in range(4))
        self.rgb_self = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.depth_self = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.cross = RelativeBiasCrossAttention(d_model, heads, grid_h, grid_w, dropout)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_q, depth_q = self.rgb_norm(rgb + position), self.depth_norm(depth + position)
        rgb = rgb + self.dropout(self.rgb_self(rgb_q, rgb_q, rgb, need_weights=False)[0])
        depth = depth + self.dropout(self.depth_self(depth_q, depth_q, depth, need_weights=False)[0])
        rgb = rgb + self.dropout(self.cross(self.cross_norm(rgb + position), depth + position))
        return rgb + self.dropout(self.ffn(self.ffn_norm(rgb))), depth


class RGBDPatchFusion(nn.Module):
    """Two-layer (by default) 30×40 region-aligned RGB-D fusion."""

    def __init__(self, d_model: int = 384, heads: int = 8, grid_h: int = 30, grid_w: int = 40,
                 layers: int = 2, ffn_dim: int = 1536, dropout: float = 0.1):
        super().__init__()
        self.grid_h, self.grid_w = grid_h, grid_w
        self.blocks = nn.ModuleList(
            RGBDRegionFusionBlock(d_model, heads, grid_h, grid_w, ffn_dim, dropout) for _ in range(layers)
        )

    def forward(self, rgb_tokens: torch.Tensor, depth_tokens: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        expected = self.grid_h * self.grid_w
        if rgb_tokens.shape[1] != expected or depth_tokens.shape[1] != expected:
            raise ValueError(f"expected aligned {self.grid_h}x{self.grid_w} tokens, got {rgb_tokens.shape[1]} and {depth_tokens.shape[1]}")
        for block in self.blocks:
            rgb_tokens, depth_tokens = block(rgb_tokens, depth_tokens, position)
        return rgb_tokens
