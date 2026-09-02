"""One image/DINO preprocessing contract shared by training and rollout.

The released LeRobot videos are already 224x224 and therefore pass through
unchanged.  Native simulator RGB is resized with the same torchvision v2
uint8 kernel used by RCS' ``JointDatasetConverter``: bilinear interpolation,
``align_corners=False`` internally, and antialiasing enabled.  Keeping uint8
until after this operation matters: float interpolation followed by rounding
differs from torchvision's native uint8 CPU kernel by one intensity level for
many pixels.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from typing import Any

import torch


IMAGE_PREPROCESS_ID = "rcs_lerobot_v2_resize_uint8_bilinear_antialias_v1"
DINO_NORMALIZATION_ID = "dinov3_imagenet_rgb_mean_std_rescale_1_over_255_v1"
DINO_IMAGE_MEAN = (0.485, 0.456, 0.406)
DINO_IMAGE_STD = (0.229, 0.224, 0.225)
DINO_RESCALE_FACTOR = 1.0 / 255.0


def resize_rgb_batch(
    images: Any,
    height: int = 224,
    width: int = 224,
) -> torch.Tensor:
    """Apply the exact RCS converter resize to uint8 ``[N,H,W,3]`` RGB."""

    if height <= 0 or width <= 0 or height % 16 or width % 16:
        raise ValueError("DINO image height/width must be positive multiples of 16")
    value = torch.as_tensor(images)
    single = value.ndim == 3
    if single:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[-1] != 3:
        raise ValueError(f"expected RGB [N,H,W,3], got {tuple(value.shape)}")
    if value.dtype != torch.uint8:
        raise ValueError(f"expected uint8 RGB, got {value.dtype}")
    if value.device.type != "cpu":
        raise ValueError(
            "RCS converter-equivalent resize must run on CPU uint8 before GPU transfer"
        )
    try:
        # Keep torchvision optional at import time so metadata-only validators
        # remain usable in lightweight supervisor processes.  The actual image
        # path fails closed when the frozen converter kernel is unavailable.
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import functional as tv_functional
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "torchvision.transforms.v2 is required by the frozen Duo RGB contract"
        ) from error
    chw = value.permute(0, 3, 1, 2).contiguous()
    if tuple(chw.shape[-2:]) != (height, width):
        chw = tv_functional.resize(
            chw,
            size=[height, width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
    if chw.dtype != torch.uint8 or tuple(chw.shape[-2:]) != (height, width):
        raise RuntimeError("torchvision changed the frozen uint8 resize contract")
    chw = chw.contiguous()
    return chw[0] if single else chw


def _processor_size(processor: Any) -> tuple[int | None, int | None]:
    size = getattr(processor, "size", None)
    if isinstance(size, Mapping):
        return size.get("height"), size.get("width")
    return getattr(size, "height", None), getattr(size, "width", None)


def validate_dino_processor_contract(processor: Any) -> dict[str, Any]:
    """Fail closed unless the local DINO artifact has official normalization."""

    mean = tuple(float(value) for value in getattr(processor, "image_mean", ()))
    std = tuple(float(value) for value in getattr(processor, "image_std", ()))
    height, width = _processor_size(processor)
    observed = {
        "image_mean": mean,
        "image_std": std,
        "do_normalize": getattr(processor, "do_normalize", None),
        "do_rescale": getattr(processor, "do_rescale", None),
        "rescale_factor": float(getattr(processor, "rescale_factor", float("nan"))),
        "do_resize": getattr(processor, "do_resize", None),
        "default_to_square": getattr(processor, "default_to_square", None),
        "processor_height": height,
        "processor_width": width,
        "resample": int(getattr(processor, "resample", -1)),
    }
    valid = (
        mean == DINO_IMAGE_MEAN
        and std == DINO_IMAGE_STD
        and observed["do_normalize"] is True
        and observed["do_rescale"] is True
        and abs(float(observed["rescale_factor"]) - DINO_RESCALE_FACTOR) <= 1e-15
        and observed["do_resize"] is True
        and observed["default_to_square"] is True
        and (height, width) == (224, 224)
        and observed["resample"] == 2  # PIL/torchvision bilinear identifier
    )
    if not valid:
        raise ValueError(f"DINOv3 image processor contract differs: {observed}")
    contract = {
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "dino_image_mean": list(DINO_IMAGE_MEAN),
        "dino_image_std": list(DINO_IMAGE_STD),
        "dino_rescale_factor": DINO_RESCALE_FACTOR,
        "dino_processor_resize": [224, 224],
        "dino_processor_resample": "bilinear",
    }
    contract["dino_processor_contract_sha256"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return contract


def validate_dino_model_contract(model: Any) -> dict[str, Any]:
    """Reject a local artifact that is not the required DINOv3 ViT-B/16."""

    config = getattr(model, "config", None)
    observed = {
        "model_type": getattr(config, "model_type", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "patch_size": getattr(config, "patch_size", None),
        "num_register_tokens": getattr(config, "num_register_tokens", None),
        "image_size": getattr(config, "image_size", None),
    }
    expected = {
        "model_type": "dinov3_vit",
        "hidden_size": 768,
        "patch_size": 16,
        "num_register_tokens": 4,
        "image_size": 224,
    }
    if observed != expected:
        raise ValueError(f"DINOv3 ViT-B/16 model contract differs: {observed}")
    return {
        **expected,
        "dino_model_contract_sha256": hashlib.sha256(
            json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def validate_dino_normalization_buffers(mean: torch.Tensor, std: torch.Tensor) -> None:
    """Prove the policy and cache normalize pixels identically."""

    expected_mean = torch.tensor(DINO_IMAGE_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    expected_std = torch.tensor(DINO_IMAGE_STD, dtype=torch.float32).view(1, 3, 1, 1)
    if not torch.equal(mean.detach().float().cpu(), expected_mean):
        raise ValueError("policy DINO mean differs from the frozen cache contract")
    if not torch.equal(std.detach().float().cpu(), expected_std):
        raise ValueError("policy DINO std differs from the frozen cache contract")


__all__ = [
    "DINO_IMAGE_MEAN",
    "DINO_IMAGE_STD",
    "DINO_NORMALIZATION_ID",
    "DINO_RESCALE_FACTOR",
    "IMAGE_PREPROCESS_ID",
    "resize_rgb_batch",
    "validate_dino_model_contract",
    "validate_dino_normalization_buffers",
    "validate_dino_processor_contract",
]
