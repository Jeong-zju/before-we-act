"""Perceiver-style visual token resampling for Phase M1."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PerceiverResamplerConfig:
    """Cross/self-attention resampler configuration."""

    input_dim: int = 512
    width: int = 512
    num_latents: int = 16
    num_layers: int = 3
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.0
    raw_patch_grid: int = 8
    raw_patch_hidden_dim: int = 128
    raw_shortcut_hidden_dim: int = 512
    max_visual_history: int = 4
    max_visual_cameras: int = 4

    def __post_init__(self) -> None:
        for name in (
            "input_dim",
            "width",
            "num_latents",
            "num_layers",
            "num_heads",
            "mlp_ratio",
            "raw_patch_grid",
            "raw_patch_hidden_dim",
            "raw_shortcut_hidden_dim",
            "max_visual_history",
            "max_visual_cameras",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.width % self.num_heads:
            raise ValueError("resampler width must be divisible by num_heads")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if self.width % 4:
            raise ValueError("spatial adapter width must be divisible by four")


@dataclass(frozen=True)
class VisualAdapterOutput:
    """Position-aware teacher and raw-RGB context for cross-attention."""

    context: Tensor
    context_valid_mask: Tensor
    spatial_shortcut: Tensor


def _two_dimensional_sincos(
    height: int,
    width: int,
    embedding_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return deterministic row-major 2-D sine/cosine position features."""

    if height <= 0 or width <= 0:
        raise ValueError("visual token grid dimensions must be positive")
    if embedding_dim % 4:
        raise ValueError("2-D positional embedding width must be divisible by four")
    axis_dim = embedding_dim // 2
    frequency_count = axis_dim // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(frequency_count, device=device, dtype=torch.float32)
        / max(frequency_count - 1, 1)
    )
    rows = torch.arange(height, device=device, dtype=torch.float32)
    columns = torch.arange(width, device=device, dtype=torch.float32)
    row_angles = rows[:, None] * frequencies[None, :]
    column_angles = columns[:, None] * frequencies[None, :]
    row_embedding = torch.cat((row_angles.sin(), row_angles.cos()), dim=-1)
    column_embedding = torch.cat((column_angles.sin(), column_angles.cos()), dim=-1)
    row_grid = row_embedding[:, None, :].expand(height, width, axis_dim)
    column_grid = column_embedding[None, :, :].expand(height, width, axis_dim)
    return (
        torch.cat((row_grid, column_grid), dim=-1)
        .reshape(height * width, embedding_dim)
        .to(dtype=dtype)
    )


class SpatialVisualTokenAdapter(nn.Module):
    """Preserve spatial/color cues while keeping the visual teacher frozen.

    The teacher route receives explicit 2-D coordinates instead of being
    treated as an unordered patch set.  A second generic route adaptively
    pools the unnormalised RGB image to a fixed grid and projects each local
    RGB average into the same width.  This route is deliberately a learned
    patch projection rather than a task-specific colour or cue decoder.
    """

    def __init__(self, config: PerceiverResamplerConfig) -> None:
        super().__init__()
        self.config = config
        width = config.width
        hidden = config.raw_patch_hidden_dim
        self.teacher_projection: nn.Module
        if config.input_dim == width:
            self.teacher_projection = nn.Identity()
        else:
            self.teacher_projection = nn.Sequential(
                nn.LayerNorm(config.input_dim),
                nn.Linear(config.input_dim, width),
            )
        self.raw_patch_projection = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
        )
        # This task-agnostic route preserves the row-major 8x8 RGB layout all
        # the way to fusion.  It complements (rather than replaces) the
        # ResNet/Perceiver route: small spatial markers cannot disappear only
        # because cross-attention and latent-token averaging are lossy.
        flattened_rgb_dim = config.raw_patch_grid * config.raw_patch_grid * 3
        self.raw_shortcut_projection = nn.Sequential(
            nn.LayerNorm(flattened_rgb_dim),
            nn.Linear(flattened_rgb_dim, config.raw_shortcut_hidden_dim),
            nn.GELU(),
            nn.Linear(config.raw_shortcut_hidden_dim, width),
            nn.LayerNorm(width),
        )
        self.shortcut_frame_score = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 1, bias=False),
        )
        self.coordinate_projection = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
        )
        self.time_embedding = nn.Embedding(config.max_visual_history, width)
        self.camera_embedding = nn.Embedding(config.max_visual_cameras, width)
        self.teacher_type_embedding = nn.Parameter(torch.empty(width))
        self.raw_type_embedding = nn.Parameter(torch.empty(width))
        self.output_norm = nn.LayerNorm(width)
        nn.init.normal_(self.teacher_type_embedding, std=1.0 / math.sqrt(width))
        nn.init.normal_(self.raw_type_embedding, std=1.0 / math.sqrt(width))
        nn.init.normal_(self.time_embedding.weight, std=1.0 / math.sqrt(width))
        nn.init.normal_(self.camera_embedding.weight, std=1.0 / math.sqrt(width))

    def forward(
        self,
        images: Tensor,
        teacher_spatial_tokens: Tensor,
        frame_valid_mask: Tensor,
    ) -> VisualAdapterOutput:
        """Build context without reading future frames or privileged labels."""

        if images.ndim != 6 or images.shape[-3] != 3:
            raise ValueError("images must have shape [B,T,Cam,3,H,W]")
        batch_size, history, cameras = images.shape[:3]
        if history > self.config.max_visual_history:
            raise ValueError(
                "visual history exceeds configured positional embedding capacity"
            )
        if cameras > self.config.max_visual_cameras:
            raise ValueError(
                "camera count exceeds configured positional embedding capacity"
            )
        if frame_valid_mask.shape != (batch_size, history, cameras):
            raise ValueError("frame_valid_mask must have shape [B,T,Cam]")
        if frame_valid_mask.dtype != torch.bool:
            raise TypeError("frame_valid_mask must be boolean")
        if frame_valid_mask.device != images.device:
            raise TypeError("images and frame_valid_mask must share a device")
        if not torch.all(frame_valid_mask.flatten(1).any(dim=1)):
            raise ValueError("each sample requires at least one valid RGB frame")
        if teacher_spatial_tokens.ndim != 5:
            raise ValueError("teacher_spatial_tokens must have shape [B,T,Cam,P,D]")
        if teacher_spatial_tokens.shape[:3] != (batch_size, history, cameras):
            raise ValueError("teacher tokens and images have different frame axes")
        teacher_patch_count = int(teacher_spatial_tokens.shape[-2])
        if teacher_spatial_tokens.shape[-1] != self.config.input_dim:
            raise ValueError("teacher token width differs from adapter input_dim")
        teacher_grid = math.isqrt(teacher_patch_count)
        if teacher_grid * teacher_grid != teacher_patch_count:
            raise ValueError("teacher spatial tokens must form a square 2-D grid")

        raw = self._raw_rgb(images)
        flattened_raw = raw.reshape(-1, 3, *raw.shape[-2:])
        grid = self.config.raw_patch_grid
        raw_patches = F.adaptive_avg_pool2d(flattened_raw, (grid, grid))
        raw_patches = (
            raw_patches.flatten(2)
            .transpose(1, 2)
            .reshape(batch_size, history, cameras, grid * grid, 3)
        )
        raw_tokens = self.raw_patch_projection(raw_patches)
        raw_shortcut_frames = self.raw_shortcut_projection(
            raw_patches.reshape(batch_size, history, cameras, grid * grid * 3)
        )

        target_dtype = raw_tokens.dtype
        teacher_tokens = self.teacher_projection(
            teacher_spatial_tokens.to(dtype=target_dtype)
        )
        teacher_position = self._position_features(
            teacher_grid,
            teacher_grid,
            device=images.device,
            dtype=target_dtype,
        )
        raw_position = self._position_features(
            grid,
            grid,
            device=images.device,
            dtype=target_dtype,
        )
        frame_position = self._frame_features(
            history,
            cameras,
            device=images.device,
            dtype=target_dtype,
        )
        teacher_tokens = (
            teacher_tokens
            + teacher_position.view(1, 1, 1, teacher_patch_count, -1)
            + frame_position[:, :, :, None, :]
            + self.teacher_type_embedding.to(dtype=target_dtype)
        )
        raw_tokens = (
            raw_tokens
            + raw_position.view(1, 1, 1, grid * grid, -1)
            + frame_position[:, :, :, None, :]
            + self.raw_type_embedding.to(dtype=target_dtype)
        )
        teacher_tokens = self.output_norm(teacher_tokens)
        raw_tokens = self.output_norm(raw_tokens)
        raw_shortcut_frames = raw_shortcut_frames + frame_position
        shortcut_scores = self.shortcut_frame_score(raw_shortcut_frames).squeeze(-1)
        shortcut_scores = shortcut_scores.masked_fill(~frame_valid_mask, -torch.inf)
        shortcut_weights = torch.softmax(
            shortcut_scores.reshape(batch_size, history * cameras), dim=-1
        ).reshape(batch_size, history, cameras, 1)
        spatial_shortcut = (raw_shortcut_frames * shortcut_weights).sum(dim=(1, 2))
        context = torch.cat((teacher_tokens, raw_tokens), dim=-2).reshape(
            batch_size,
            history * cameras * (teacher_patch_count + grid * grid),
            self.config.width,
        )
        valid = frame_valid_mask.unsqueeze(-1).expand(
            batch_size,
            history,
            cameras,
            teacher_patch_count + grid * grid,
        )
        return VisualAdapterOutput(
            context=context,
            context_valid_mask=valid.reshape(batch_size, -1),
            spatial_shortcut=spatial_shortcut,
        )

    @staticmethod
    def _raw_rgb(images: Tensor) -> Tensor:
        if images.dtype == torch.uint8:
            return images.to(dtype=torch.float32).div(255.0)
        if not torch.is_floating_point(images):
            raise TypeError("RGB must be uint8 or floating point")
        raw = images.to(dtype=torch.float32)
        if not torch.isfinite(raw).all():
            raise ValueError("floating RGB contains NaN or Inf")
        if raw.numel() and (float(raw.amin()) < 0.0 or float(raw.amax()) > 1.0):
            raise ValueError("floating RGB must be scaled to [0,1]")
        return raw

    def _position_features(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        fixed = _two_dimensional_sincos(
            height,
            width,
            self.config.width,
            device=device,
            dtype=dtype,
        )
        rows, columns = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
            indexing="ij",
        )
        coordinates = torch.stack((rows, columns), dim=-1).reshape(-1, 2)
        return fixed + self.coordinate_projection(coordinates)

    def _frame_features(
        self,
        history: int,
        cameras: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        time_indices = torch.arange(history, device=device)
        camera_indices = torch.arange(cameras, device=device)
        temporal = self.time_embedding(time_indices).to(dtype=dtype)
        camera = self.camera_embedding(camera_indices).to(dtype=dtype)
        return temporal[None, :, None, :] + camera[None, None, :, :]


class _PerceiverLayer(nn.Module):
    def __init__(self, config: PerceiverResamplerConfig) -> None:
        super().__init__()
        width = config.width
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(
            width,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(
            width,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        hidden = config.mlp_ratio * width
        self.feed_forward_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, width),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        latents: Tensor,
        context: Tensor,
        *,
        context_valid_mask: Tensor,
    ) -> Tensor:
        normalized_latents = self.query_norm(latents)
        normalized_context = self.context_norm(context)
        cross, _ = self.cross_attention(
            normalized_latents,
            normalized_context,
            normalized_context,
            key_padding_mask=~context_valid_mask,
            need_weights=False,
        )
        latents = latents + cross
        normalized_latents = self.self_norm(latents)
        attended, _ = self.self_attention(
            normalized_latents,
            normalized_latents,
            normalized_latents,
            need_weights=False,
        )
        latents = latents + attended
        return latents + self.feed_forward(self.feed_forward_norm(latents))


class PerceiverResampler(nn.Module):
    """Compress an arbitrary valid context to a fixed latent-token set."""

    def __init__(self, config: PerceiverResamplerConfig) -> None:
        super().__init__()
        self.config = config
        # The adapter deliberately lives below ``resampler.*`` so the existing
        # M1 freeze curriculum treats both as one visual-adapter stage.
        self.visual_adapter = SpatialVisualTokenAdapter(config)
        # ``SpatialVisualTokenAdapter`` always projects teacher and raw tokens
        # into ``width`` before concatenation.  Keep this named identity module
        # for state-dict compatibility with the accepted ResNet checkpoints.
        self.context_projection: nn.Module = nn.Identity()
        self.latent_queries = nn.Parameter(
            torch.empty(config.num_latents, config.width)
        )
        nn.init.normal_(self.latent_queries, std=1.0 / math.sqrt(config.width))
        self.layers = nn.ModuleList(
            _PerceiverLayer(config) for _ in range(config.num_layers)
        )
        self.output_norm = nn.LayerNorm(config.width)
        self.summary_norm = nn.LayerNorm(config.width)
        # A scalar keeps the direct route auditable and lets training calibrate
        # its contribution without hiding it behind another deep bottleneck.
        self.spatial_shortcut_gain = nn.Parameter(torch.tensor(1.0))

    def summarize(self, latents: Tensor, spatial_shortcut: Tensor) -> Tensor:
        """Fuse resampled semantics with the generic flattened-RGB shortcut."""

        if latents.ndim != 3 or latents.shape[-1] != self.config.width:
            raise ValueError("latents must have shape [B,N,width]")
        if spatial_shortcut.shape != (latents.shape[0], self.config.width):
            raise ValueError("spatial_shortcut must have shape [B,width]")
        return self.summary_norm(
            latents.mean(dim=1) + self.spatial_shortcut_gain * spatial_shortcut
        )

    def forward(
        self,
        context: Tensor,
        *,
        context_valid_mask: Tensor | None = None,
    ) -> Tensor:
        if context.ndim != 3 or context.shape[-1] != self.config.width:
            raise ValueError(
                "context must have shape "
                f"[B,N,{self.config.width}], got {tuple(context.shape)}"
            )
        batch_size, token_count, _ = context.shape
        if token_count <= 0:
            raise ValueError("resampler context cannot be empty")
        if context_valid_mask is None:
            valid = torch.ones(
                batch_size,
                token_count,
                dtype=torch.bool,
                device=context.device,
            )
        else:
            if context_valid_mask.shape != (batch_size, token_count):
                raise ValueError("context_valid_mask must have shape [B,N]")
            if context_valid_mask.dtype != torch.bool:
                raise TypeError("context_valid_mask must be boolean")
            if context_valid_mask.device != context.device:
                raise TypeError("context and context_valid_mask must share a device")
            valid = context_valid_mask
        if not torch.all(valid.any(dim=1)):
            raise ValueError("each sample must contain at least one valid visual token")
        projected = self.context_projection(context)
        latents = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
        for layer in self.layers:
            latents = layer(
                latents,
                projected,
                context_valid_mask=valid,
            )
        return self.output_norm(latents)


__all__ = [
    "PerceiverResampler",
    "PerceiverResamplerConfig",
    "SpatialVisualTokenAdapter",
    "VisualAdapterOutput",
]
