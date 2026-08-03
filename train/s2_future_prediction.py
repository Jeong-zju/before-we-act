"""Shared S2-R3 PCA projection, target construction, and masked losses."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import io
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


S2_ARTIFACT_FORMAT = "wam.robofactory.s2_r3.future_artifacts/1"


def load_s2_artifact(path: str | Path, *, device: torch.device) -> dict[str, Any]:
    artifact = torch.load(
        Path(path).expanduser().resolve(strict=True),
        map_location=device,
        weights_only=False,
    )
    if not isinstance(artifact, dict) or artifact.get(
        "format_version"
    ) != S2_ARTIFACT_FORMAT:
        raise ValueError("artifact is not an S2-R3 PCA/statistics payload")
    required = {
        "pca_mean": (1024,),
        "pca_components": (1024, 256),
        "pca_projected_std": (256,),
        "state_delta_mean": (4, 18),
        "state_delta_std": (4, 18),
        "visual_delta_mean": (4, 256),
        "visual_delta_std": (4, 256),
    }
    for name, shape in required.items():
        value = artifact.get(name)
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"S2 artifact {name} must have shape {shape}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"S2 artifact {name} contains NaN or Inf")
    for name in ("pca_projected_std", "state_delta_std", "visual_delta_std"):
        if not bool(artifact[name].gt(0.0).all()):
            raise ValueError(f"S2 artifact {name} must be strictly positive")
    return artifact


def project_dino_grid(raw: Tensor, artifact: Mapping[str, Any]) -> Tensor:
    if raw.ndim < 2 or raw.shape[-1] != 1024:
        raise ValueError("raw DINO grid must end in 1024 features")
    mean = artifact["pca_mean"].to(raw)
    components = artifact["pca_components"].to(raw)
    std = artifact["pca_projected_std"].to(raw)
    return ((raw - mean) @ components) / std


def encode_local_visual_targets(
    vision: nn.Module,
    grouped: Mapping[str, Tensor],
    artifact: Mapping[str, Any],
    *,
    device: torch.device,
    grid_height: int,
    grid_width: int,
) -> tuple[Tensor, Tensor]:
    """Encode valid current/future local views and return normalized deltas."""

    valid_agents = grouped["valid_agent_mask"].to(device=device, dtype=torch.bool)
    future_valid = grouped["future_agent_visual_valid_mask"].to(
        device=device, dtype=torch.bool
    )
    batch_size, agents = valid_agents.shape
    futures = future_valid.shape[2]
    grid_tokens = grid_height * grid_width
    latent_dim = int(artifact["pca_components"].shape[1])
    current = torch.zeros(
        batch_size,
        agents,
        grid_tokens,
        latent_dim,
        device=device,
        dtype=torch.float32,
    )
    future = torch.zeros(
        batch_size,
        agents,
        futures,
        grid_tokens,
        latent_dim,
        device=device,
        dtype=torch.float32,
    )
    current_images = grouped["agent_observations"].to(
        device=device, non_blocking=True
    )
    future_images = grouped["future_agent_observations"].to(
        device=device, non_blocking=True
    )
    if bool(valid_agents.any()):
        output = vision.forward_spatial_grid(
            current_images[valid_agents],
            grid_height=grid_height,
            grid_width=grid_width,
        )
        current[valid_agents] = project_dino_grid(
            output.spatial_tokens.float(), artifact
        )
    if bool(future_valid.any()):
        output = vision.forward_spatial_grid(
            future_images[future_valid],
            grid_height=grid_height,
            grid_width=grid_width,
        )
        future[future_valid] = project_dino_grid(
            output.spatial_tokens.float(), artifact
        )
    delta = future - current[:, :, None]
    mean = artifact["visual_delta_mean"].to(delta)
    std = artifact["visual_delta_std"].to(delta)
    normalized = (delta - mean[None, None, :, None]) / std[
        None, None, :, None
    ]
    normalized = normalized.masked_fill(
        ~future_valid[:, :, :, None, None], 0.0
    )
    return current, normalized


def encode_current_visual_context(
    vision: nn.Module,
    grouped: Mapping[str, Tensor],
    artifact: Mapping[str, Any],
    *,
    device: torch.device,
    grid_height: int,
    grid_width: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Encode only the current local/global views needed by on-path S3.

    Returns raw local DINO tokens for the frozen Flow plus PCA-projected local
    and shared grids for the frozen future predictor. Future observations are
    deliberately not read by the deployed action path.
    """

    valid_agents = grouped["valid_agent_mask"].to(device=device, dtype=torch.bool)
    shared_valid = grouped["shared_observation_valid_mask"].to(
        device=device, dtype=torch.bool
    )
    batch_size, agents = valid_agents.shape
    grid_tokens = grid_height * grid_width
    latent_dim = int(artifact["pca_components"].shape[1])
    local_images = grouped["agent_observations"].to(device=device, non_blocking=True)
    shared_images = grouped["shared_observation"].to(device=device, non_blocking=True)
    raw_local = torch.zeros(
        batch_size,
        agents,
        1,
        1024,
        device=device,
        dtype=torch.float32,
    )
    projected_local = torch.zeros(
        batch_size,
        agents,
        grid_tokens,
        latent_dim,
        device=device,
        dtype=torch.float32,
    )
    projected_shared = torch.zeros(
        batch_size,
        grid_tokens,
        latent_dim,
        device=device,
        dtype=torch.float32,
    )
    local_count = int(valid_agents.sum().item())
    shared_count = int(shared_valid.sum().item())
    if local_count + shared_count:
        selected_images = torch.cat(
            (local_images[valid_agents], shared_images[shared_valid]), dim=0
        )
        raw = vision(selected_images).spatial_tokens.float()
        local_raw = raw[:local_count]
        raw_local = raw_local.expand(
            -1, -1, local_raw.shape[1], -1
        ).clone()
        raw_local[valid_agents] = local_raw
        patch_rows = int(vision.config.image_height) // int(vision.patch_size)
        patch_columns = int(vision.config.image_width) // int(vision.patch_size)
        grid = F.adaptive_avg_pool2d(
            raw.transpose(1, 2).reshape(
                raw.shape[0], raw.shape[-1], patch_rows, patch_columns
            ),
            (grid_height, grid_width),
        ).flatten(2).transpose(1, 2)
        projected = project_dino_grid(grid, artifact)
        projected_local[valid_agents] = projected[:local_count]
        projected_shared[shared_valid] = projected[local_count:]
    return raw_local, projected_local, projected_shared


def encode_future_visual_targets(
    vision: nn.Module,
    grouped: Mapping[str, Tensor],
    artifact: Mapping[str, Any],
    *,
    current_local: Tensor,
    current_shared: Tensor,
    device: torch.device,
    grid_height: int,
    grid_width: int,
) -> tuple[Tensor, Tensor]:
    """Encode local/shared future frames in one frozen-DINO call.

    The current projected grids are supplied by the deployed input path.  This
    avoids encoding the same current local and shared observations a second
    time solely to construct training targets, while retaining FP32 PCA target
    construction and the exact masking/statistics contract.
    """

    local_valid = grouped["future_agent_visual_valid_mask"].to(
        device=device, dtype=torch.bool
    )
    shared_valid = grouped["future_shared_visual_valid_mask"].to(
        device=device, dtype=torch.bool
    )
    batch_size, agents, futures = local_valid.shape
    grid_tokens = grid_height * grid_width
    latent_dim = int(artifact["pca_components"].shape[1])
    expected_local = (batch_size, agents, grid_tokens, latent_dim)
    expected_shared = (batch_size, grid_tokens, latent_dim)
    if tuple(current_local.shape) != expected_local:
        raise ValueError(f"current_local must have shape {expected_local}")
    if tuple(current_shared.shape) != expected_shared:
        raise ValueError(f"current_shared must have shape {expected_shared}")

    future_local = torch.zeros(
        batch_size,
        agents,
        futures,
        grid_tokens,
        latent_dim,
        device=device,
        dtype=torch.float32,
    )
    future_shared = torch.zeros(
        batch_size,
        futures,
        grid_tokens,
        latent_dim,
        device=device,
        dtype=torch.float32,
    )
    local_images = grouped["future_agent_observations"].to(
        device=device, non_blocking=True
    )
    shared_images = grouped["future_shared_observations"].to(
        device=device, non_blocking=True
    )
    local_count = int(local_valid.sum().item())
    shared_count = int(shared_valid.sum().item())
    if local_count + shared_count:
        selected_images = torch.cat(
            (local_images[local_valid], shared_images[shared_valid]), dim=0
        )
        grid = vision.forward_spatial_grid(
            selected_images,
            grid_height=grid_height,
            grid_width=grid_width,
        ).spatial_tokens.float()
        projected = project_dino_grid(grid, artifact)
        future_local[local_valid] = projected[:local_count]
        future_shared[shared_valid] = projected[local_count:]

    local_delta = future_local - current_local[:, :, None]
    local_mean = artifact["visual_delta_mean"].to(local_delta)
    local_std = artifact["visual_delta_std"].to(local_delta)
    normalized_local = (
        local_delta - local_mean[None, None, :, None]
    ) / local_std[None, None, :, None]
    normalized_local = normalized_local.masked_fill(
        ~local_valid[:, :, :, None, None], 0.0
    )

    shared_mean = artifact.get("shared_visual_delta_mean")
    shared_std = artifact.get("shared_visual_delta_std")
    if not isinstance(shared_mean, Tensor) or not isinstance(shared_std, Tensor):
        raise ValueError("S2-R4 artifact lacks shared-view target statistics")
    shared_delta = future_shared - current_shared[:, None]
    mean = shared_mean.to(shared_delta)
    std = shared_std.to(shared_delta)
    normalized_shared = (shared_delta - mean[None, :, None]) / std[
        None, :, None
    ]
    normalized_shared = normalized_shared.masked_fill(
        ~shared_valid[:, :, None, None], 0.0
    )
    return normalized_local, normalized_shared


def encode_shared_visual_targets(
    vision: nn.Module,
    grouped: Mapping[str, Tensor],
    artifact: Mapping[str, Any],
    *,
    device: torch.device,
    grid_height: int,
    grid_width: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Encode the global slot and its normalized future/persistence targets."""

    current_valid = grouped["shared_observation_valid_mask"].to(
        device=device, dtype=torch.bool
    )
    future_valid = grouped["future_shared_visual_valid_mask"].to(
        device=device, dtype=torch.bool
    )
    batch_size = current_valid.shape[0]
    futures = future_valid.shape[1]
    grid_tokens = grid_height * grid_width
    latent_dim = int(artifact["pca_components"].shape[1])
    current = torch.zeros(
        batch_size,
        grid_tokens,
        latent_dim,
        device=device,
        dtype=torch.float32,
    )
    future = torch.zeros(
        batch_size,
        futures,
        grid_tokens,
        latent_dim,
        device=device,
        dtype=torch.float32,
    )
    current_images = grouped["shared_observation"].to(
        device=device, non_blocking=True
    )
    future_images = grouped["future_shared_observations"].to(
        device=device, non_blocking=True
    )
    if bool(current_valid.any()):
        output = vision.forward_spatial_grid(
            current_images[current_valid],
            grid_height=grid_height,
            grid_width=grid_width,
        )
        current[current_valid] = project_dino_grid(
            output.spatial_tokens.float(), artifact
        )
    if bool(future_valid.any()):
        output = vision.forward_spatial_grid(
            future_images[future_valid],
            grid_height=grid_height,
            grid_width=grid_width,
        )
        future[future_valid] = project_dino_grid(
            output.spatial_tokens.float(), artifact
        )
    delta = future - current[:, None]
    shared_mean = artifact.get("shared_visual_delta_mean")
    shared_std = artifact.get("shared_visual_delta_std")
    if not isinstance(shared_mean, Tensor) or not isinstance(
        shared_std, Tensor
    ):
        raise ValueError("S2-R4 artifact lacks shared-view target statistics")
    mean = shared_mean.to(delta)
    std = shared_std.to(delta)
    normalized = (delta - mean[None, :, None]) / std[None, :, None]
    normalized = normalized.masked_fill(
        ~future_valid[:, :, None, None], 0.0
    )
    persistence = ((-mean) / std)[None, :, None].expand(
        batch_size,
        -1,
        grid_tokens,
        -1,
    )
    persistence = persistence.masked_fill(
        ~future_valid[:, :, None, None], 0.0
    )
    return current, normalized, persistence


def normalized_state_delta(
    grouped: Mapping[str, Tensor],
    artifact: Mapping[str, Any],
    *,
    device: torch.device,
) -> Tensor:
    delta = grouped["future_state_delta"].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    valid = grouped["future_state_valid_mask"].to(
        device=device, dtype=torch.bool
    )
    mean = artifact["state_delta_mean"].to(delta)
    std = artifact["state_delta_std"].to(delta)
    normalized = (delta - mean[None, None]) / std[None, None]
    return normalized.masked_fill(~valid[..., None], 0.0)


def normalized_persistence_state(
    artifact: Mapping[str, Any],
    *,
    batch_size: int,
    agents: int,
    device: torch.device,
) -> Tensor:
    """Return the normalized zero-delta persistence baseline."""

    mean = artifact["state_delta_mean"].to(device=device)
    std = artifact["state_delta_std"].to(device=device)
    return ((-mean) / std)[None, None].expand(
        batch_size,
        agents,
        -1,
        -1,
    )


def normalized_persistence_visual(
    artifact: Mapping[str, Any],
    *,
    batch_size: int,
    agents: int,
    grid_tokens: int,
    device: torch.device,
) -> Tensor:
    """Return normalized unchanged-view targets for every agent slot."""

    mean = artifact["visual_delta_mean"].to(device=device)
    std = artifact["visual_delta_std"].to(device=device)
    return ((-mean) / std)[None, None, :, None].expand(
        batch_size,
        agents,
        -1,
        grid_tokens,
        -1,
    )


def masked_future_prediction_losses(
    predicted_state: Tensor,
    target_state: Tensor,
    state_valid: Tensor,
    predicted_visual: Tensor,
    target_visual: Tensor,
    visual_valid: Tensor,
) -> dict[str, Tensor]:
    """Return equal-weight state/visual losses and per-trajectory composites."""

    if predicted_state.shape != target_state.shape or predicted_state.ndim != 4:
        raise ValueError("state predictions and targets must share [B,A,F,S]")
    if state_valid.shape != predicted_state.shape[:-1]:
        raise ValueError("state_valid must be [B,A,F]")
    if (
        predicted_visual.shape != target_visual.shape
        or predicted_visual.ndim != 5
    ):
        raise ValueError(
            "visual predictions and targets must share [B,A,F,G,D]"
        )
    if visual_valid.shape != predicted_visual.shape[:3]:
        raise ValueError("visual_valid must be [B,A,F]")
    state_values = F.smooth_l1_loss(
        predicted_state.float(),
        target_state.float(),
        reduction="none",
    ).mean(dim=-1)
    visual_values = 1.0 - F.cosine_similarity(
        predicted_visual.float(),
        target_visual.float(),
        dim=-1,
        eps=1e-6,
    )
    visual_values = visual_values.mean(dim=-1)
    state_per = _masked_per_trajectory(
        state_values,
        state_valid,
        allow_empty=False,
    )
    visual_per = _masked_per_trajectory(
        visual_values,
        visual_valid,
        allow_empty=True,
    )
    composite_per = state_per + visual_per
    return {
        "loss": composite_per.mean(),
        "state": state_per.mean(),
        "visual": visual_per.mean(),
        "per_trajectory": composite_per,
        "state_per_trajectory": state_per,
        "visual_per_trajectory": visual_per,
    }


def state_dict_sha256(module: nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(
        {
            name: value.detach().cpu()
            for name, value in sorted(module.state_dict().items())
        },
        buffer,
    )
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _masked_per_trajectory(
    values: Tensor,
    valid: Tensor,
    *,
    allow_empty: bool,
) -> Tensor:
    if values.shape != valid.shape or values.ndim != 3:
        raise ValueError("masked trajectory values must share [B,A,F]")
    weights = valid.to(values)
    denominator = weights.sum(dim=(1, 2))
    if not allow_empty and not bool(denominator.gt(0).all()):
        raise ValueError("every trajectory needs at least one valid future target")
    return (values * weights).sum(dim=(1, 2)) / denominator.clamp_min(1.0)


__all__ = [
    "S2_ARTIFACT_FORMAT",
    "encode_future_visual_targets",
    "encode_local_visual_targets",
    "encode_current_visual_context",
    "encode_shared_visual_targets",
    "file_sha256",
    "load_s2_artifact",
    "masked_future_prediction_losses",
    "normalized_persistence_state",
    "normalized_persistence_visual",
    "normalized_state_delta",
    "project_dino_grid",
    "state_dict_sha256",
]
