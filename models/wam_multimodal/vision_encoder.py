"""Frozen visual teachers used by the Phase M1 multimodal WAM.

New Phase M1 runs default to a DINOv3 ViT encoder loaded from a verified local
Hugging Face artifact.  The legacy self-contained ResNet-18 implementation is
kept so already accepted v1 checkpoints remain explicitly reloadable.  Neither
implementation performs a network request or silently falls back to a
different backbone.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


OFFICIAL_RESNET18_FILENAME = "resnet18-f37072fd.pth"
OFFICIAL_RESNET18_SHA256 = (
    "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
)
IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)

DEFAULT_DINOV3_ENCODER = "dinov3_vitl16_lvd"
DEFAULT_DINOV3_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_DINOV3_REVISION = "dd0a398fa8e84f2a37179332f6c561d20276300b"
DEFAULT_DINOV3_WEIGHTS_SHA256 = (
    "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
)
DEFAULT_DINOV3_CONFIG_SHA256 = (
    "ce962b0c8ca4f2deb48c6fdfd6035257e3769f1d4d9154c92aba51991e46e290"
)
DINOV3_PREPROCESS_ID = "dinov3_imagenet_rgb_resize_square_antialias_v1"
DINOV3_RECTANGULAR_PREPROCESS_ID = (
    "dinov3_imagenet_rgb_resize_rectangular_antialias_v2"
)


@dataclass(frozen=True)
class DINOv3EncoderSpec:
    """Project alias expanded to an official immutable model identity."""

    name: str
    model_id: str
    output_dim: int
    patch_size: int = 16
    register_tokens: int = 4
    default_revision: str | None = None
    expected_weights_sha256: str | None = None


DINOV3_ENCODER_SPECS: dict[str, DINOv3EncoderSpec] = {
    "dinov3_vits16_lvd": DINOv3EncoderSpec(
        "dinov3_vits16_lvd",
        "facebook/dinov3-vits16-pretrain-lvd1689m",
        384,
    ),
    "dinov3_vits16plus_lvd": DINOv3EncoderSpec(
        "dinov3_vits16plus_lvd",
        "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        384,
    ),
    "dinov3_vitb16_lvd": DINOv3EncoderSpec(
        "dinov3_vitb16_lvd",
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        768,
    ),
    DEFAULT_DINOV3_ENCODER: DINOv3EncoderSpec(
        DEFAULT_DINOV3_ENCODER,
        DEFAULT_DINOV3_MODEL_ID,
        1024,
        default_revision=DEFAULT_DINOV3_REVISION,
        expected_weights_sha256=DEFAULT_DINOV3_WEIGHTS_SHA256,
    ),
    "dinov3_vith16plus_lvd": DINOv3EncoderSpec(
        "dinov3_vith16plus_lvd",
        "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        1280,
    ),
}


@dataclass(frozen=True)
class FrozenDINOv3Config:
    """Verified local artifact and preprocessing contract for DINOv3."""

    encoder_name: str
    model_id: str
    revision: str
    config_path: str | Path
    weights_path: str | Path
    expected_weights_sha256: str
    expected_config_sha256: str
    input_size: int | None = 256
    input_height: int | None = None
    input_width: int | None = None
    preprocess_id: str = DINOV3_PREPROCESS_ID
    inference_batch_size: int = 8

    def __post_init__(self) -> None:
        if self.encoder_name not in DINOV3_ENCODER_SPECS:
            raise ValueError(f"unknown DINOv3 encoder {self.encoder_name!r}")
        spec = DINOV3_ENCODER_SPECS[self.encoder_name]
        if self.model_id != spec.model_id:
            raise ValueError(
                f"{self.encoder_name} must use official model id {spec.model_id!r}"
            )
        revision = str(self.revision)
        if len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision
        ):
            raise ValueError("DINOv3 revision must be a full lowercase commit SHA")
        digest = str(self.expected_weights_sha256).lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("expected_weights_sha256 must be a lowercase SHA-256")
        config_digest = str(self.expected_config_sha256).lower()
        if len(config_digest) != 64 or any(
            character not in "0123456789abcdef" for character in config_digest
        ):
            raise ValueError("expected_config_sha256 must be a lowercase SHA-256")
        rectangular = self.input_height is not None or self.input_width is not None
        if rectangular:
            if self.input_height is None or self.input_width is None:
                raise ValueError(
                    "DINOv3 input_height and input_width must be configured together"
                )
            if self.input_size is not None:
                raise ValueError(
                    "DINOv3 rectangular input cannot also configure input_size"
                )
            dimensions = (int(self.input_height), int(self.input_width))
            if any(value <= 0 or value % spec.patch_size for value in dimensions):
                raise ValueError(
                    "DINOv3 input_height/input_width must be positive multiples "
                    f"of {spec.patch_size}"
                )
            if self.preprocess_id != DINOV3_RECTANGULAR_PREPROCESS_ID:
                raise ValueError(
                    "rectangular DINOv3 input requires the rectangular preprocess id"
                )
        else:
            if self.input_size is None:
                raise ValueError("DINOv3 input_size is required for square preprocessing")
            if int(self.input_size) <= 0 or int(self.input_size) % spec.patch_size:
                raise ValueError(
                    f"DINOv3 input_size must be a positive multiple of {spec.patch_size}"
                )
        if self.preprocess_id not in {
            DINOV3_PREPROCESS_ID,
            DINOV3_RECTANGULAR_PREPROCESS_ID,
        }:
            raise ValueError(f"unsupported DINOv3 preprocess {self.preprocess_id!r}")
        if int(self.inference_batch_size) <= 0:
            raise ValueError("inference_batch_size must be positive")
        object.__setattr__(self, "config_path", Path(self.config_path))
        object.__setattr__(self, "weights_path", Path(self.weights_path))
        object.__setattr__(self, "expected_weights_sha256", digest)
        object.__setattr__(self, "expected_config_sha256", config_digest)

    @property
    def image_height(self) -> int:
        return int(
            self.input_size if self.input_height is None else self.input_height
        )

    @property
    def image_width(self) -> int:
        return int(self.input_size if self.input_width is None else self.input_width)


@dataclass(frozen=True)
class FrozenResNet18Config:
    """Pinned artifact and preprocessing contract for the visual teacher."""

    weights_path: str | Path
    expected_sha256: str = OFFICIAL_RESNET18_SHA256
    resize_shorter_side: int = 256
    crop_size: int = 224

    def __post_init__(self) -> None:
        path = Path(self.weights_path)
        if not str(path):
            raise ValueError("weights_path cannot be empty")
        digest = str(self.expected_sha256).lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
        if int(self.resize_shorter_side) <= 0 or int(self.crop_size) <= 0:
            raise ValueError("ImageNet resize and crop dimensions must be positive")
        if int(self.resize_shorter_side) < int(self.crop_size):
            raise ValueError("resize_shorter_side cannot be smaller than crop_size")
        object.__setattr__(self, "weights_path", path)
        object.__setattr__(self, "expected_sha256", digest)


@dataclass(frozen=True)
class VisionEncoderOutput:
    """Spatial teacher tokens and their global average-pooled representation."""

    spatial_tokens: Tensor
    pooled_latent: Tensor


def default_resnet18_weights_path() -> Path:
    """Return the conventional torch-hub cache path without downloading data."""

    return Path(torch.hub.get_dir()) / "checkpoints" / OFFICIAL_RESNET18_FILENAME


def sha256_file(path: str | Path) -> str:
    """Hash an artifact using bounded memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def _conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, value: Tensor) -> Tensor:
        identity = value
        output = self.conv1(value)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        if self.downsample is not None:
            identity = self.downsample(value)
        output = output + identity
        return self.relu(output)


class _ResNet18(nn.Module):
    """Torchvision-key-compatible ResNet-18, including the unused classifier."""

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=2)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._initialize_parameters()

    def _make_layer(
        self, out_channels: int, *, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample: nn.Module | None = None
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )
        layers: list[nn.Module] = [
            _BasicBlock(
                self.inplanes,
                out_channels,
                stride=stride,
                downsample=downsample,
            )
        ]
        self.inplanes = out_channels
        layers.extend(
            _BasicBlock(self.inplanes, out_channels) for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def _initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def spatial_features(self, images: Tensor) -> Tensor:
        output = self.conv1(images)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.maxpool(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        return self.layer4(output)

    def forward(self, images: Tensor) -> Tensor:
        output = self.spatial_features(images)
        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        return self.fc(output)


def build_resnet18_classifier() -> nn.Module:
    """Build the exact classifier surface used for strict artifact loading."""

    return _ResNet18()


class FrozenResNet18Encoder(nn.Module):
    """Permanently frozen ResNet-18 returning 7x7 spatial patch tokens.

    Inputs are raw RGB tensors with arbitrary leading dimensions and trailing
    ``[3,H,W]``.  ``uint8`` inputs are scaled to ``[0,1]``; floating inputs are
    required to already be in that interval.  Official ImageNet resize,
    center-crop and channel normalization are applied internally.
    """

    output_dim = 512

    def __init__(self, config: FrozenResNet18Config) -> None:
        super().__init__()
        self.config = config
        actual_sha256 = sha256_file(config.weights_path)
        if actual_sha256 != config.expected_sha256:
            raise ValueError(
                "ResNet-18 artifact SHA-256 mismatch: "
                f"expected {config.expected_sha256}, got {actual_sha256}"
            )
        payload: Any = torch.load(
            config.weights_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(payload, Mapping) or not all(
            isinstance(name, str) and isinstance(value, Tensor)
            for name, value in payload.items()
        ):
            raise TypeError("ResNet-18 artifact must be a tensor state dictionary")
        backbone = _ResNet18()
        incompatible = backbone.load_state_dict(dict(payload), strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"strict ResNet-18 load failed: {incompatible}")
        self.backbone = backbone
        self.register_buffer(
            "imagenet_mean",
            torch.tensor(IMAGENET_RGB_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(IMAGENET_RGB_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=True,
        )
        self.artifact_sha256 = actual_sha256
        self._freeze()

    @property
    def patch_count(self) -> int:
        size = self.config.crop_size
        # ResNet reduces spatial resolution by 32 with ceil-like padded stages.
        spatial = (size + 31) // 32
        return spatial * spatial

    def _freeze(self) -> None:
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True) -> "FrozenResNet18Encoder":
        """Ignore parent training-mode changes and preserve frozen BN statistics."""

        del mode
        self._freeze()
        return self

    def preprocess(self, images: Tensor) -> tuple[Tensor, tuple[int, ...]]:
        if images.ndim < 4 or images.shape[-3] != 3:
            raise ValueError("images must have shape [...,3,H,W]")
        if images.shape[-2] <= 0 or images.shape[-1] <= 0:
            raise ValueError("image height and width must be positive")
        leading_shape = tuple(int(value) for value in images.shape[:-3])
        flattened = images.reshape(-1, *images.shape[-3:])
        if flattened.dtype == torch.uint8:
            normalized = flattened.to(dtype=torch.float32).div_(255.0)
        elif torch.is_floating_point(flattened):
            normalized = flattened.to(dtype=torch.float32)
            if not torch.isfinite(normalized).all():
                raise ValueError("floating RGB contains NaN or Inf")
            if normalized.numel() and (
                float(normalized.amin()) < 0.0 or float(normalized.amax()) > 1.0
            ):
                raise ValueError("floating RGB must be scaled to [0,1]")
        else:
            raise TypeError("RGB must be uint8 or floating point")

        height, width = normalized.shape[-2:]
        shorter = min(height, width)
        resize = self.config.resize_shorter_side
        if shorter != resize:
            scale = resize / shorter
            resized_height = int(round(height * scale))
            resized_width = int(round(width * scale))
            normalized = F.interpolate(
                normalized,
                size=(resized_height, resized_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        crop = self.config.crop_size
        top = (normalized.shape[-2] - crop) // 2
        left = (normalized.shape[-1] - crop) // 2
        normalized = normalized[..., top : top + crop, left : left + crop]
        if tuple(normalized.shape[-2:]) != (crop, crop):
            raise ValueError("ImageNet center crop could not be constructed")
        normalized = (normalized - self.imagenet_mean) / self.imagenet_std
        return normalized, leading_shape

    def forward(self, images: Tensor) -> VisionEncoderOutput:
        prepared, leading_shape = self.preprocess(images)
        with torch.no_grad():
            feature_map = self.backbone.spatial_features(prepared)
            spatial_tokens = feature_map.flatten(2).transpose(1, 2).contiguous()
            pooled_latent = spatial_tokens.mean(dim=1)
        token_count = int(spatial_tokens.shape[1])
        spatial_tokens = spatial_tokens.reshape(
            *leading_shape, token_count, self.output_dim
        ).detach()
        pooled_latent = pooled_latent.reshape(*leading_shape, self.output_dim).detach()
        return VisionEncoderOutput(
            spatial_tokens=spatial_tokens,
            pooled_latent=pooled_latent,
        )

class FrozenDINOv3Encoder(nn.Module):
    """Permanently frozen DINOv3 ViT returning patch tokens and CLS latent.

    The project alias (for example ``dinov3_vitl16_lvd``) is expanded to an
    official Hugging Face model id, but construction only reads the two local
    files named by :class:`FrozenDINOv3Config`.  The first output token is used
    as the global teacher target; the class token and all register tokens are
    excluded from the spatial token sequence.
    """

    family = "dinov3"

    def __init__(self, config: FrozenDINOv3Config) -> None:
        super().__init__()
        self.config = config
        if not config.config_path.is_file():
            raise FileNotFoundError(config.config_path)
        if not config.weights_path.is_file():
            raise FileNotFoundError(config.weights_path)
        actual_sha256 = sha256_file(config.weights_path)
        if actual_sha256 != config.expected_weights_sha256:
            raise ValueError(
                "DINOv3 artifact SHA-256 mismatch: "
                f"expected {config.expected_weights_sha256}, got {actual_sha256}"
            )
        try:
            raw_config = json.loads(config.config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("DINOv3 config.json is not valid UTF-8 JSON") from exc
        if not isinstance(raw_config, Mapping):
            raise ValueError("DINOv3 config.json root must be an object")
        config_sha256 = canonical_json_sha256(raw_config)
        if config_sha256 != config.expected_config_sha256:
            raise ValueError(
                "DINOv3 config identity mismatch: "
                f"expected {config.expected_config_sha256}, got {config_sha256}"
            )
        spec = DINOV3_ENCODER_SPECS[config.encoder_name]
        _validate_dinov3_architecture(raw_config, spec)

        try:
            from safetensors.torch import load_file
            from transformers import (
                DINOv3ViTConfig,
                DINOv3ViTModel,
                __version__ as transformers_version,
            )
        except ImportError as exc:  # pragma: no cover - dependency error surface
            raise RuntimeError(
                "DINOv3 requires transformers>=4.56 and safetensors"
            ) from exc
        backbone_config = DINOv3ViTConfig.from_dict(dict(raw_config))
        backbone = DINOv3ViTModel(backbone_config)
        payload = load_file(config.weights_path, device="cpu")
        payload = _normalize_dinov3_checkpoint_keys(backbone, payload)
        incompatible = backbone.load_state_dict(payload, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"strict DINOv3 load failed: {incompatible}")
        del payload
        self.backbone = backbone
        self.output_dim = spec.output_dim
        self.patch_size = spec.patch_size
        self.register_tokens = spec.register_tokens
        self.implementation = f"transformers.DINOv3ViTModel/{transformers_version}"
        self.artifact_sha256 = actual_sha256
        self.config_sha256 = config_sha256
        self.register_buffer(
            "imagenet_mean",
            torch.tensor(IMAGENET_RGB_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(IMAGENET_RGB_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=True,
        )
        self._freeze()

    @property
    def encoder_name(self) -> str:
        return self.config.encoder_name

    @property
    def patch_count(self) -> int:
        rows = self.config.image_height // self.patch_size
        columns = self.config.image_width // self.patch_size
        return rows * columns

    def _freeze(self) -> None:
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True) -> "FrozenDINOv3Encoder":
        del mode
        self._freeze()
        return self

    def preprocess(self, images: Tensor) -> tuple[Tensor, tuple[int, ...]]:
        if images.ndim < 4 or images.shape[-3] != 3:
            raise ValueError("images must have shape [...,3,H,W]")
        if images.shape[-2] <= 0 or images.shape[-1] <= 0:
            raise ValueError("image height and width must be positive")
        leading_shape = tuple(int(value) for value in images.shape[:-3])
        flattened = images.reshape(-1, *images.shape[-3:])
        if flattened.dtype == torch.uint8:
            normalized = flattened.to(dtype=torch.float32).div_(255.0)
        elif torch.is_floating_point(flattened):
            normalized = flattened.to(dtype=torch.float32)
            if not torch.isfinite(normalized).all():
                raise ValueError("floating RGB contains NaN or Inf")
            if normalized.numel() and (
                float(normalized.amin()) < 0.0 or float(normalized.amax()) > 1.0
            ):
                raise ValueError("floating RGB must be scaled to [0,1]")
        else:
            raise TypeError("RGB must be uint8 or floating point")
        size = (self.config.image_height, self.config.image_width)
        if tuple(normalized.shape[-2:]) != size:
            normalized = F.interpolate(
                normalized,
                size=size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        normalized = (normalized - self.imagenet_mean) / self.imagenet_std
        return normalized, leading_shape

    def forward(self, images: Tensor) -> VisionEncoderOutput:
        if images.ndim < 4 or images.shape[-3] != 3:
            raise ValueError("images must have shape [...,3,H,W]")
        leading_shape = tuple(int(value) for value in images.shape[:-3])
        flattened = images.reshape(-1, *images.shape[-3:])
        if flattened.shape[0] == 0:
            raise ValueError("DINOv3 requires at least one RGB frame")
        spatial_batches: list[Tensor] = []
        pooled_batches: list[Tensor] = []
        step = self.config.inference_batch_size
        with torch.inference_mode():
            for start in range(0, flattened.shape[0], step):
                prepared, _ = self.preprocess(flattened[start : start + step])
                output = self.backbone(pixel_values=prepared)
                hidden = output.last_hidden_state
                if hidden.ndim != 3 or hidden.shape[-1] != self.output_dim:
                    raise RuntimeError("DINOv3 returned an invalid hidden-state shape")
                special_tokens = 1 + self.register_tokens
                spatial = hidden[:, special_tokens:, :]
                if spatial.shape[1] != self.patch_count:
                    raise RuntimeError(
                        "DINOv3 patch-token count differs from the preprocessing contract"
                    )
                spatial_batches.append(spatial)
                pooled_batches.append(hidden[:, 0, :])
        spatial_tokens = torch.cat(spatial_batches, dim=0).contiguous()
        pooled_latent = torch.cat(pooled_batches, dim=0).contiguous()
        spatial_tokens = spatial_tokens.reshape(
            *leading_shape, self.patch_count, self.output_dim
        ).detach()
        pooled_latent = pooled_latent.reshape(*leading_shape, self.output_dim).detach()
        return VisionEncoderOutput(
            spatial_tokens=spatial_tokens,
            pooled_latent=pooled_latent,
        )

    def forward_pooled(self, images: Tensor) -> Tensor:
        """Encode only CLS latents with bounded resize/backbone memory.

        This remains the bounded CLS path for legacy one-token contexts and
        future-visual targets.  Spatial M2 contexts use ``forward_spatial_grid``
        so they retain local evidence without materializing all 1,200 patches.
        """

        if images.ndim < 4 or images.shape[-3] != 3:
            raise ValueError("images must have shape [...,3,H,W]")
        leading_shape = tuple(int(value) for value in images.shape[:-3])
        flattened = images.reshape(-1, *images.shape[-3:])
        if flattened.shape[0] == 0:
            raise ValueError("DINOv3 requires at least one RGB frame")
        pooled_batches: list[Tensor] = []
        step = self.config.inference_batch_size
        expected_tokens = 1 + self.register_tokens + self.patch_count
        with torch.inference_mode():
            for start in range(0, flattened.shape[0], step):
                prepared, _ = self.preprocess(flattened[start : start + step])
                hidden = self.backbone(pixel_values=prepared).last_hidden_state
                if hidden.ndim != 3 or hidden.shape[1:] != (
                    expected_tokens,
                    self.output_dim,
                ):
                    raise RuntimeError("DINOv3 returned an invalid hidden-state shape")
                pooled_batches.append(hidden[:, 0, :])
        pooled = torch.cat(pooled_batches, dim=0).contiguous()
        return pooled.reshape(*leading_shape, self.output_dim).detach()

    def forward_spatial_grid(
        self,
        images: Tensor,
        *,
        grid_height: int,
        grid_width: int,
    ) -> VisionEncoderOutput:
        """Encode a bounded spatial grid plus CLS without retaining 1,200 tokens.

        M2 needs spatial evidence for precise gripper/object alignment, but
        feeding every DINO patch token into the causal Transformer is
        prohibitively expensive.  Adaptive pooling preserves a small ordered
        grid for each named camera while CLS remains the future-visual target.
        """

        if images.ndim < 4 or images.shape[-3] != 3:
            raise ValueError("images must have shape [...,3,H,W]")
        if grid_height <= 0 or grid_width <= 0:
            raise ValueError("spatial grid dimensions must be positive")
        patch_rows = self.config.image_height // self.patch_size
        patch_columns = self.config.image_width // self.patch_size
        if grid_height > patch_rows or grid_width > patch_columns:
            raise ValueError("spatial grid exceeds the DINO patch grid")
        leading_shape = tuple(int(value) for value in images.shape[:-3])
        flattened = images.reshape(-1, *images.shape[-3:])
        if flattened.shape[0] == 0:
            raise ValueError("DINOv3 requires at least one RGB frame")
        spatial_batches: list[Tensor] = []
        pooled_batches: list[Tensor] = []
        step = self.config.inference_batch_size
        expected_tokens = 1 + self.register_tokens + self.patch_count
        with torch.inference_mode():
            for start in range(0, flattened.shape[0], step):
                prepared, _ = self.preprocess(flattened[start : start + step])
                hidden = self.backbone(pixel_values=prepared).last_hidden_state
                if hidden.ndim != 3 or hidden.shape[1:] != (
                    expected_tokens,
                    self.output_dim,
                ):
                    raise RuntimeError("DINOv3 returned an invalid hidden-state shape")
                special_tokens = 1 + self.register_tokens
                patches = hidden[:, special_tokens:, :].transpose(1, 2).reshape(
                    hidden.shape[0],
                    self.output_dim,
                    patch_rows,
                    patch_columns,
                )
                spatial = F.adaptive_avg_pool2d(
                    patches,
                    (grid_height, grid_width),
                ).flatten(2).transpose(1, 2)
                spatial_batches.append(spatial)
                pooled_batches.append(hidden[:, 0, :])
        spatial_tokens = torch.cat(spatial_batches, dim=0).contiguous().reshape(
            *leading_shape,
            grid_height * grid_width,
            self.output_dim,
        )
        pooled_latent = torch.cat(pooled_batches, dim=0).contiguous().reshape(
            *leading_shape,
            self.output_dim,
        )
        return VisionEncoderOutput(
            spatial_tokens=spatial_tokens.detach(),
            pooled_latent=pooled_latent.detach(),
        )


def _validate_dinov3_architecture(
    payload: Mapping[str, Any], spec: DINOv3EncoderSpec
) -> None:
    expected = {
        "model_type": "dinov3_vit",
        "hidden_size": spec.output_dim,
        "patch_size": spec.patch_size,
        "num_register_tokens": spec.register_tokens,
        "num_channels": 3,
    }
    mismatched = {
        name: {"expected": value, "observed": payload.get(name)}
        for name, value in expected.items()
        if payload.get(name) != value
    }
    architectures = payload.get("architectures")
    if architectures != ["DINOv3ViTModel"]:
        mismatched["architectures"] = {
            "expected": ["DINOv3ViTModel"],
            "observed": architectures,
        }
    if mismatched:
        raise ValueError(f"DINOv3 architecture identity mismatch: {mismatched}")


def _normalize_dinov3_checkpoint_keys(
    backbone: nn.Module,
    payload: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Expand the HF base-model prefix without weakening strict loading.

    Official ``DINOv3ViTModel`` safetensors are saved through Transformers'
    base-model convention: encoder layers use ``layer.*`` while a directly
    constructed wrapper exposes those same parameters as ``model.layer.*``.
    Embedding and final-norm keys already match.  ``from_pretrained`` normally
    reconciles this difference; Phase M1 loads verified local tensors itself,
    so it performs the same unambiguous mapping before a strict state load.
    """

    expected = set(backbone.state_dict())
    prefix = str(getattr(backbone, "base_model_prefix", ""))
    if not prefix:
        raise RuntimeError("DINOv3 wrapper has no Transformers base_model_prefix")
    normalized: dict[str, Tensor] = {}
    sources: dict[str, str] = {}
    for source_name, value in payload.items():
        prefixed = f"{prefix}.{source_name}"
        target_name = (
            source_name
            if source_name in expected
            else prefixed
            if prefixed in expected
            else source_name
        )
        if target_name in normalized:
            raise RuntimeError(
                "DINOv3 checkpoint key normalization collision: "
                f"{sources[target_name]!r} and {source_name!r} both map to "
                f"{target_name!r}"
            )
        normalized[target_name] = value
        sources[target_name] = source_name
    return normalized


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    """Hash JSON semantics independently of whitespace and key ordering."""

    serialized = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


__all__ = [
    "DEFAULT_DINOV3_ENCODER",
    "DEFAULT_DINOV3_CONFIG_SHA256",
    "DEFAULT_DINOV3_MODEL_ID",
    "DEFAULT_DINOV3_REVISION",
    "DEFAULT_DINOV3_WEIGHTS_SHA256",
    "DINOV3_ENCODER_SPECS",
    "DINOV3_PREPROCESS_ID",
    "DINOV3_RECTANGULAR_PREPROCESS_ID",
    "DINOv3EncoderSpec",
    "FrozenDINOv3Config",
    "FrozenDINOv3Encoder",
    "FrozenResNet18Config",
    "FrozenResNet18Encoder",
    "IMAGENET_RGB_MEAN",
    "IMAGENET_RGB_STD",
    "OFFICIAL_RESNET18_FILENAME",
    "OFFICIAL_RESNET18_SHA256",
    "VisionEncoderOutput",
    "build_resnet18_classifier",
    "canonical_json_sha256",
    "default_resnet18_weights_path",
    "sha256_file",
]
