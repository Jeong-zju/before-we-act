from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _image(value) -> np.ndarray:
    value = np.asarray(value)
    if np.issubdtype(value.dtype, np.floating):
        value = np.clip(value * 255.0 if value.max(initial=0) <= 1 else value, 0, 255).astype(np.uint8)
    if value.ndim == 3 and value.shape[0] == 3: value = einops.rearrange(value, "c h w -> h w c")
    if value.shape != (224, 224, 3) or value.dtype != np.uint8:
        raise ValueError(f"DuoBench image contract requires uint8 224x224x3, got {value.shape}/{value.dtype}")
    return value


@dataclasses.dataclass(frozen=True)
class DuoBenchInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        head, wrist = _image(data["head"]), _image(data["wrist"])
        state = np.asarray(data["state"], np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError(f"DuoBench local state must be finite 8-D, got {state.shape}")
        if state[7] not in (0.0, 1.0): raise ValueError("DuoBench state gripper must be binary")
        # Every arm maps its own wrist to the same upstream slot.  The peer slot
        # remains masked, so the shared policy cannot learn an arm identity from slots.
        out = {
            "state": state,
            "image": {
                "base_0_rgb": head,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(wrist),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], np.float32)
            if actions.shape != (16, 8) or not np.isfinite(actions).all():
                raise ValueError(f"DuoBench local action chunk must be finite 16x8, got {actions.shape}")
            out["actions"] = actions
        if "prompt" in data: out["prompt"] = data["prompt"]
        return out


@dataclasses.dataclass(frozen=True)
class DuoBenchOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"], np.float32)[..., :8]}


@dataclasses.dataclass(frozen=True)
class CastFloat32(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        for key in ("state", "actions"):
            if key in data: data[key] = np.asarray(data[key], np.float32)
        return data
