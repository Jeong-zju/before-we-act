"""Pi0/π0.5 transforms for the decentralized RoboFactory contract.

The policy is intentionally fed one agent's camera and proprioception only.
Missing wrist cameras are represented by masked zero images.  RoboFactory
actions are absolute PD joint-position targets; Pi0's base checkpoints use
delta joint actions, so the data config adds the corresponding delta transform
for the seven arm joints while leaving the gripper target absolute.
"""
from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0 if image.max(initial=0) <= 1.0 else image, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RoboFactoryInputs(transforms.DataTransformFn):
    """Map local RGB/qpos/action fields to Pi0's canonical model inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base = _parse_image(data["image"])
        # Keep the interface shape expected by the Pi0 vision tower, but mask
        # both wrist slots so no peer or synthetic wrist signal is consumed.
        zero = np.zeros_like(base)
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape[-1] != 9:
            raise ValueError(f"RoboFactory local qpos must be 9-D, got {state.shape}")
        out = {
            "state": state,
            "image": {"base_0_rgb": base, "left_wrist_0_rgb": zero, "right_wrist_0_rgb": zero},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.False_, "right_wrist_0_rgb": np.False_},
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != 8:
                raise ValueError(f"RoboFactory local commanded action must be 8-D, got {actions.shape}")
            out["actions"] = actions
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        return out


@dataclasses.dataclass(frozen=True)
class RoboFactoryOutputs(transforms.DataTransformFn):
    """Remove Pi0's padding and return one local 8-D commanded action chunk."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :8]}


@dataclasses.dataclass(frozen=True)
class CastFloat32(transforms.DataTransformFn):
    """Keep JAX inputs in the model's declared float32 dtype after JSON stats."""

    def __call__(self, data: dict) -> dict:
        for key in ("state", "actions"):
            if key in data:
                data[key] = np.asarray(data[key], dtype=np.float32)
        return data


def make_example() -> dict:
    return {
        "image": np.zeros((480, 640, 3), np.uint8),
        "state": np.zeros(9, np.float32),
        "actions": np.zeros((16, 8), np.float32),
        "prompt": "perform the task",
    }
