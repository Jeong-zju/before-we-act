"""Strict scratch construction for multimodal M1 without legacy checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
from typing import Any

import torch
from torch import nn

from models.wam import (
    AffineActionCodec,
    AffineActionCodecConfig,
    NormalizationStats,
    RWMARConfig,
    RWMARWorldModel,
    StatefulActionFlow,
    StatefulActionFlowConfig,
)
from models.wam_multimodal import LatentWAM, LatentWAMConfig


SCRATCH_INITIALIZATION_MODE = "scratch"


@dataclass(frozen=True)
class ScratchActionFlowConfig:
    """Action-flow architecture with feature width resolved from the world model."""

    action_dim: int = 16
    horizon: int = 8
    hidden_dim: int = 512
    hidden_layers: int = 4
    time_embedding_dim: int = 32
    anchor_hidden_dim: int = 256
    anchor_hidden_layers: int = 2
    anchor_min_log_std: float = -5.0
    anchor_max_log_std: float = 1.0
    anchor_mode: str = "none"

    def __post_init__(self) -> None:
        if self.anchor_mode != "none":
            raise ValueError("scratch action flow must use anchor_mode='none'")

    def resolve(self, *, feature_dim: int) -> StatefulActionFlowConfig:
        return StatefulActionFlowConfig(feature_dim=int(feature_dim), **asdict(self))


@dataclass(frozen=True)
class ScratchM1BuildConfig:
    """Complete random-initialization contract for task-side M1 modules."""

    seed: int
    world: RWMARConfig
    action_flow: ScratchActionFlowConfig
    latent_wam: LatentWAMConfig
    initialization_mode: str = SCRATCH_INITIALIZATION_MODE

    def __post_init__(self) -> None:
        if self.initialization_mode != SCRATCH_INITIALIZATION_MODE:
            raise ValueError("ScratchM1BuildConfig only supports scratch mode")
        if int(self.seed) < 0:
            raise ValueError("scratch initialization seed must be non-negative")
        dimensions = {
            self.world.action_dim,
            self.action_flow.action_dim,
            self.latent_wam.action_dim,
        }
        if len(dimensions) != 1:
            raise ValueError("scratch world/flow/latent action dimensions differ")
        if self.world.train_forecast_horizon < self.action_flow.horizon:
            raise ValueError("world train horizon is shorter than action-flow horizon")

    def to_dict(self) -> dict[str, Any]:
        return {
            "initialization_mode": self.initialization_mode,
            "seed": int(self.seed),
            "world": _plain(asdict(self.world)),
            "action_flow": _plain(asdict(self.action_flow)),
            "latent_wam": self.latent_wam.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScratchM1BuildConfig":
        raw = dict(payload)
        world = dict(_mapping(raw, "world"))
        for key in ("yaw_indices", "gripper_closed_indices"):
            if key in world:
                world[key] = tuple(world[key])
        return cls(
            initialization_mode=str(
                raw.get("initialization_mode", SCRATCH_INITIALIZATION_MODE)
            ),
            seed=int(raw["seed"]),
            world=RWMARConfig(**world),
            action_flow=ScratchActionFlowConfig(
                **dict(_mapping(raw, "action_flow"))
            ),
            latent_wam=LatentWAMConfig.from_dict(
                _mapping(raw, "latent_wam")
            ),
        )


@dataclass
class ScratchM1Bundle:
    """Task-side scratch modules plus immutable data/action contracts."""

    model: LatentWAM
    action_flow: StatefulActionFlow
    action_codec: AffineActionCodec
    normalization: NormalizationStats
    build_config: ScratchM1BuildConfig
    initialization_hashes: dict[str, str]
    vision_identity: dict[str, Any] | None

    def to(self, device: str | torch.device) -> "ScratchM1Bundle":
        self.model.to(device)
        self.action_flow.to(device)
        self.action_codec.to(device)
        return self


def build_scratch_m1(
    config: ScratchM1BuildConfig,
    normalization: NormalizationStats,
    action_codec: AffineActionCodec | AffineActionCodecConfig,
    *,
    vision_encoder: nn.Module | None,
) -> ScratchM1Bundle:
    """Randomize every task-side module once; never read a legacy checkpoint."""

    codec = (
        action_codec
        if isinstance(action_codec, AffineActionCodec)
        else AffineActionCodec(action_codec)
    )
    if codec.action_dim != config.world.action_dim:
        raise ValueError("action codec dimension differs from scratch model")
    if normalization.state_mean.shape != (config.world.state_dim,):
        raise ValueError("normalization state dimension differs from scratch model")
    if normalization.action_mean.shape != (config.world.action_dim,):
        raise ValueError("normalization action dimension differs from scratch model")
    if config.latent_wam.use_vision and vision_encoder is None:
        raise ValueError("vision-enabled scratch M1 requires a frozen encoder")
    if not config.latent_wam.use_vision and vision_encoder is not None:
        raise ValueError("state-only scratch M1 must not construct a vision encoder")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(config.seed))
        world = RWMARWorldModel(config.world, normalization)
        resolved_flow = config.action_flow.resolve(
            feature_dim=world.planning_feature_dim
        )
        flow = StatefulActionFlow(resolved_flow, normalization)
        model = LatentWAM(config.latent_wam, world, vision_encoder)

    if flow.has_anchor:
        raise RuntimeError("scratch action flow unexpectedly constructed an anchor")
    if model.planning_feature_dim != flow.config.feature_dim:
        raise RuntimeError("scratch latent/flow planning feature dimensions differ")
    if model.vision_encoder is not None:
        model.vision_encoder.eval()
        for parameter in model.vision_encoder.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in model.vision_encoder.parameters()):
            raise RuntimeError("scratch vision encoder must remain frozen")

    hashes = {
        "world_model_initial_sha256": module_state_sha256(world),
        "action_flow_initial_sha256": module_state_sha256(flow),
        "latent_task_modules_initial_sha256": module_state_sha256(
            model, excluded_prefixes=("vision_encoder.", "world_model.")
        ),
    }
    return ScratchM1Bundle(
        model=model,
        action_flow=flow,
        action_codec=codec,
        normalization=normalization,
        build_config=config,
        initialization_hashes=hashes,
        vision_identity=vision_encoder_identity(model.vision_encoder),
    )


def module_state_sha256(
    module: nn.Module,
    *,
    excluded_prefixes: tuple[str, ...] = (),
) -> str:
    """Hash tensor names, shapes, dtypes and bytes in stable key order."""

    digest = hashlib.sha256()
    included = 0
    for name, value in sorted(module.state_dict().items()):
        if name.startswith(excluded_prefixes):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        included += 1
    if included == 0:
        digest.update(b"empty-state")
    return digest.hexdigest()


def vision_encoder_identity(encoder: nn.Module | None) -> dict[str, Any] | None:
    if encoder is None:
        return None
    output_dim = int(getattr(encoder, "output_dim", -1))
    if output_dim <= 0:
        raise ValueError("vision encoder must expose a positive output_dim")
    artifact_sha256 = str(getattr(encoder, "artifact_sha256", ""))
    if len(artifact_sha256) != 64:
        artifact_sha256 = module_state_sha256(encoder)
    identity = {
        "family": type(encoder).__name__,
        "output_dim": output_dim,
        "artifact_sha256": artifact_sha256,
        "config_sha256": str(getattr(encoder, "config_sha256", "not_available")),
        "frozen": not any(parameter.requires_grad for parameter in encoder.parameters()),
    }
    if not identity["frozen"]:
        raise ValueError("vision encoder identity is not frozen")
    return identity


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ValueError(f"scratch config field {key!r} must be a mapping")
    return item


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


__all__ = [
    "SCRATCH_INITIALIZATION_MODE",
    "ScratchActionFlowConfig",
    "ScratchM1BuildConfig",
    "ScratchM1Bundle",
    "build_scratch_m1",
    "module_state_sha256",
    "vision_encoder_identity",
]
