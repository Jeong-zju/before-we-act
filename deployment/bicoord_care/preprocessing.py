"""Frozen BiCoord JPEG and DINOv3 preprocessing contracts.

BiCoord stores each camera frame as a fixed-width JPEG byte string.  The
upstream writer passes the simulator's RGB array directly through OpenCV's
JPEG encoder, and the upstream policy decodes it with OpenCV without a channel
swap.  We reproduce that byte-to-array path exactly: the returned channel
positions have the same numerical meaning as the original simulator RGB
array, even though OpenCV calls its array layout BGR.

Decoding is deliberately a single-frame operation.  Callers must not expand a
whole HDF5 camera dataset into RGB in memory.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from typing import Any

import cv2
import numpy as np
import torch


IMAGE_PREPROCESS_ID = (
    "bicoord_opencv_jpeg_source_order_torchvision_v2_uint8_bilinear_"
    "antialias_v1"
)
DINO_NORMALIZATION_ID = "dinov3_imagenet_rgb_mean_std_rescale_1_over_255_v1"
DINO_IMAGE_MEAN = (0.485, 0.456, 0.406)
DINO_IMAGE_STD = (0.229, 0.224, 0.225)
DINO_RESCALE_FACTOR = 1.0 / 255.0
DEFAULT_IMAGE_HEIGHT = 224
DEFAULT_IMAGE_WIDTH = 224


def coerce_rgb_frame(value: Any) -> np.ndarray:
    """Convert a decoded/runtime frame to contiguous ``uint8`` HWC.

    Runtime evaluators sometimes provide a singleton batch or a float image in
    ``[0, 1]``.  Accepting those representations here is useful at the I/O
    boundary; the frozen resize and DINO normalization below still receive
    only uint8 data.  Integer values are never rescaled or clipped silently.
    """

    if isinstance(value, Mapping):
        if "rgb" in value:
            value = value["rgb"]
        elif "data" in value:
            value = value["data"]
    array = np.asarray(value)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"expected one RGB frame, got batch shape {array.shape}")
        array = array[0]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected RGB HWC frame, got {array.shape}")
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("RGB frame contains non-finite values")
        maximum = float(array.max()) if array.size else 0.0
        minimum = float(array.min()) if array.size else 0.0
        if 0.0 <= minimum and maximum <= 1.0:
            array = np.rint(array * 255.0)
        elif minimum < 0.0 or maximum > 255.0:
            raise ValueError("float RGB frame must be in [0,1] or [0,255]")
    elif np.issubdtype(array.dtype, np.integer):
        if array.size and (int(array.min()) < 0 or int(array.max()) > 255):
            raise ValueError("integer RGB frame outside [0,255]")
    else:
        raise TypeError(f"unsupported RGB frame dtype: {array.dtype}")
    return np.ascontiguousarray(array.astype(np.uint8, copy=False))


def dino_normalize(images: torch.Tensor) -> torch.Tensor:
    """Apply the frozen DINO ImageNet normalization to uint8 CHW/NCHW data."""

    value = torch.as_tensor(images)
    single = value.ndim == 3
    if single:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[1] != 3:
        raise ValueError(f"expected uint8 NCHW image, got {tuple(value.shape)}")
    if value.dtype != torch.uint8:
        raise ValueError("DINO normalization expects uint8 input")
    mean = torch.tensor(DINO_IMAGE_MEAN, dtype=torch.float32, device=value.device).view(1, 3, 1, 1)
    std = torch.tensor(DINO_IMAGE_STD, dtype=torch.float32, device=value.device).view(1, 3, 1, 1)
    result = value.float().mul(DINO_RESCALE_FACTOR).sub(mean).div(std)
    return result[0] if single else result


def encode_jpeg_rgb(frame: Any, *, quality: int = 95) -> bytes:
    """Encode one uint8 frame using the benchmark's OpenCV JPEG convention."""

    image = coerce_rgb_frame(frame)
    quality = int(quality)
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be in [1,100]")
    ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok:
        raise ValueError("OpenCV failed to encode JPEG")
    return bytes(encoded)


def normalize_vector(
    value: Any,
    mean: Any,
    std: Any,
    *,
    dimension: int = 7,
) -> torch.Tensor:
    """Normalize a local state/action vector without clipping its range."""

    vector = torch.as_tensor(value, dtype=torch.float32)
    center = torch.as_tensor(mean, dtype=torch.float32, device=vector.device)
    scale = torch.as_tensor(std, dtype=torch.float32, device=vector.device)
    if vector.shape[-1:] != (dimension,) or center.shape != (dimension,) or scale.shape != (dimension,):
        raise ValueError(f"expected trailing vector dimension {dimension}")
    if not torch.isfinite(vector).all() or not torch.isfinite(center).all() or not torch.isfinite(scale).all():
        raise ValueError("normalization inputs must be finite")
    if torch.any(scale <= 0):
        raise ValueError("normalization standard deviations must be positive")
    return (vector - center) / scale


def denormalize_vector(
    value: Any,
    mean: Any,
    std: Any,
    *,
    dimension: int = 7,
) -> torch.Tensor:
    """Invert :func:`normalize_vector` exactly up to float32 arithmetic."""

    vector = torch.as_tensor(value, dtype=torch.float32)
    center = torch.as_tensor(mean, dtype=torch.float32, device=vector.device)
    scale = torch.as_tensor(std, dtype=torch.float32, device=vector.device)
    if vector.shape[-1:] != (dimension,) or center.shape != (dimension,) or scale.shape != (dimension,):
        raise ValueError(f"expected trailing vector dimension {dimension}")
    if not torch.isfinite(vector).all() or not torch.isfinite(center).all() or not torch.isfinite(scale).all():
        raise ValueError("normalization inputs must be finite")
    if torch.any(scale <= 0):
        raise ValueError("normalization standard deviations must be positive")
    return vector * scale + center


def _jpeg_payload(value: Any) -> bytes:
    """Return one JPEG payload without accepting an RGB frame by mistake."""

    if isinstance(value, np.ndarray):
        if value.dtype != np.uint8 or value.ndim != 1:
            raise TypeError(
                "BiCoord JPEG must be a one-dimensional uint8 byte buffer"
            )
        payload = value.tobytes()
    elif isinstance(value, (bytes, bytearray, memoryview, np.bytes_)):
        payload = bytes(value)
    else:
        raise TypeError(f"unsupported BiCoord JPEG value: {type(value)!r}")

    # HDF5's fixed-width S dtype may expose the writer's zero padding.  A
    # valid JPEG terminates in FFD9, so removing only trailing NUL bytes does
    # not alter its compressed content.
    payload = payload.rstrip(b"\0")
    if len(payload) < 4 or not payload.startswith(b"\xff\xd8"):
        raise ValueError("BiCoord camera value lacks a JPEG SOI marker")
    if not payload.endswith(b"\xff\xd9"):
        raise ValueError("BiCoord camera value lacks a JPEG EOI marker")
    return payload


def decode_jpeg_rgb(value: Any) -> np.ndarray:
    """Decode exactly one upstream BiCoord JPEG to contiguous uint8 HWC RGB.

    No ``cvtColor`` is applied.  That is intentional and matches
    ``envs/utils/parse_hdf5.py`` and every shipped BiCoord baseline converter.
    """

    payload = _jpeg_payload(value)
    encoded = np.frombuffer(payload, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("OpenCV could not decode the BiCoord JPEG frame")
    if decoded.ndim != 3 or decoded.shape[-1] != 3 or decoded.dtype != np.uint8:
        raise ValueError(
            f"BiCoord JPEG decoded to an invalid RGB array: "
            f"{decoded.shape}/{decoded.dtype}"
        )
    return np.ascontiguousarray(decoded)


# An explicit benchmark-qualified alias makes instrumentation/monkeypatching
# in data-path audits unambiguous while retaining a concise public function.
decode_bicoord_jpeg_rgb = decode_jpeg_rgb


def resize_rgb_batch(
    images: Any,
    height: int = DEFAULT_IMAGE_HEIGHT,
    width: int = DEFAULT_IMAGE_WIDTH,
) -> torch.Tensor:
    """Resize uint8 HWC RGB with the frozen torchvision-v2 CPU kernel."""

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
        raise ValueError("frozen BiCoord resize runs on CPU uint8 before transfer")
    try:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import functional as tv_functional
    except (ImportError, AttributeError) as error:  # pragma: no cover - env guard
        raise RuntimeError(
            "torchvision.transforms.v2 is required by the BiCoord RGB contract"
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
        raise RuntimeError("torchvision changed the frozen BiCoord resize contract")
    chw = chw.contiguous()
    return chw[0] if single else chw


def _processor_size(processor: Any) -> tuple[int | None, int | None]:
    size = getattr(processor, "size", None)
    if isinstance(size, Mapping):
        return size.get("height"), size.get("width")
    return getattr(size, "height", None), getattr(size, "width", None)


def validate_dino_processor_contract(processor: Any) -> dict[str, Any]:
    """Fail closed unless the artifact uses official DINOv3 normalization."""

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
        and abs(observed["rescale_factor"] - DINO_RESCALE_FACTOR) <= 1e-15
        and observed["do_resize"] is True
        and observed["default_to_square"] is True
        and (height, width) == (DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)
        and observed["resample"] == 2
    )
    if not valid:
        raise ValueError(f"DINOv3 image processor contract differs: {observed}")
    contract = {
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "dino_image_mean": list(DINO_IMAGE_MEAN),
        "dino_image_std": list(DINO_IMAGE_STD),
        "dino_rescale_factor": DINO_RESCALE_FACTOR,
        "dino_processor_resize": [DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH],
        "dino_processor_resample": "bilinear",
    }
    contract["dino_processor_contract_sha256"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return contract


def validate_dino_model_contract(model: Any) -> dict[str, Any]:
    """Reject any visual artifact other than upstream DINOv3 ViT-B/16."""

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
    expected_mean = torch.tensor(DINO_IMAGE_MEAN, dtype=torch.float32).view(
        1, 3, 1, 1
    )
    expected_std = torch.tensor(DINO_IMAGE_STD, dtype=torch.float32).view(1, 3, 1, 1)
    if not torch.equal(mean.detach().float().cpu(), expected_mean):
        raise ValueError("policy DINO mean differs from the frozen BiCoord contract")
    if not torch.equal(std.detach().float().cpu(), expected_std):
        raise ValueError("policy DINO std differs from the frozen BiCoord contract")


__all__ = [
    "coerce_rgb_frame",
    "DEFAULT_IMAGE_HEIGHT",
    "DEFAULT_IMAGE_WIDTH",
    "DINO_IMAGE_MEAN",
    "DINO_IMAGE_STD",
    "DINO_NORMALIZATION_ID",
    "DINO_RESCALE_FACTOR",
    "IMAGE_PREPROCESS_ID",
    "decode_bicoord_jpeg_rgb",
    "decode_jpeg_rgb",
    "denormalize_vector",
    "dino_normalize",
    "encode_jpeg_rgb",
    "normalize_vector",
    "resize_rgb_batch",
    "validate_dino_model_contract",
    "validate_dino_normalization_buffers",
    "validate_dino_processor_contract",
]
