"""Legal raw-RGB spatial conditioning for the R12-R3 action generators.

The frozen DINOv3 backbone is deliberately external to action checkpoints.  A
hash-locked cache is used during training; deployment recomputes the same 4x4
ordered grid from the current fixed views.  Future frames, task identity and
simulator state never enter this module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from models.wam_multimodal.vision_encoder import (
    DINOV3_PREPROCESS_ID,
    DINOV3_RECTANGULAR_PREPROCESS_ID,
    FrozenDINOv3Config,
    FrozenDINOv3Encoder,
)


def locked_r12_spatial_observation() -> dict[str, object]:
    """Return the preregistered deployment and cache identity."""

    return {
        "mode": "w11_plus_current_dinov3_spatial_v1",
        "encoder_name": "dinov3_vitb16_lvd",
        "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "weights_sha256": "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b",
        "config_sha256": "69256c4c142d59b0c0ccf5746542d9f2415f6c7db03bd7835a1f7b3afedb77fe",
        "preprocess": DINOV3_PREPROCESS_ID,
        "input_size": 224,
        "feature_dim": 768,
        "spatial_grid": [4, 4],
        "max_views": 5,
        "history_frames": 1,
        "fusion": "zero_gated_cross_attention_into_w11_tokens",
        "fusion_heads": 4,
    }


def locked_r12_full_episode_observation() -> dict[str, object]:
    """Full-data contract with native RGB encoding before bounded compression.

    The source image stays at the environment's native 480x640 resolution.
    DINO therefore sees and emits a 30x40 patch grid.  Adaptive pooling to 6x8
    happens only after the frozen backbone; the image encoder never receives a
    lower-resolution surrogate.  TeamBeliefState is supplemental context.
    """

    return {
        "mode": "native_480x640_dinov3_30x40_to_6x8_plus_w11_v2",
        "encoder_name": "dinov3_vitb16_lvd",
        "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "weights_sha256": "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b",
        "config_sha256": "69256c4c142d59b0c0ccf5746542d9f2415f6c7db03bd7835a1f7b3afedb77fe",
        "preprocess": DINOV3_RECTANGULAR_PREPROCESS_ID,
        "input_height": 480,
        "input_width": 640,
        "require_native_input_shape": True,
        "feature_dim": 768,
        "encoder_patch_grid": [30, 40],
        "spatial_grid": [6, 8],
        "max_views": 5,
        "history_frames": 1,
        "compression_stage": "adaptive_average_after_full_resolution_dinov3",
        "primary_input": "all_current_fixed_view_rgb_uint8_480x640",
        "supplemental_input": "frozen_team_belief_state",
        "training_cache_equivalence": "frozen_dinov3_post_encoder_6x8_tokens",
        "fusion": "direct_candidate_conditioning_no_scalar_gate",
        "fusion_heads": 4,
    }


class R12SpatialObservationEncoder(nn.Module):
    """Frozen DINOv3-B/16 with bounded per-view spatial pooling."""

    def __init__(
        self,
        observation: Mapping[str, object],
        artifact_root: str | Path,
        *,
        inference_batch_size: int = 32,
    ) -> None:
        super().__init__()
        root = Path(artifact_root).resolve()
        # The S10 transfer did not retain the gated Hub snapshot revision.  The
        # two content hashes below are therefore the authoritative identity;
        # FrozenDINOv3Encoder never performs a network lookup or consumes this
        # syntactic placeholder.
        revision_placeholder = "0" * 40
        rectangular = "input_height" in observation or "input_width" in observation
        self.encoder = FrozenDINOv3Encoder(
            FrozenDINOv3Config(
                encoder_name=str(observation["encoder_name"]),
                model_id=str(observation["model_id"]),
                revision=revision_placeholder,
                config_path=root / "config.json",
                weights_path=root / "model.safetensors",
                expected_weights_sha256=str(observation["weights_sha256"]),
                expected_config_sha256=str(observation["config_sha256"]),
                input_size=(None if rectangular else int(observation["input_size"])),
                input_height=(
                    int(observation["input_height"]) if rectangular else None
                ),
                input_width=(
                    int(observation["input_width"]) if rectangular else None
                ),
                preprocess_id=str(observation["preprocess"]),
                inference_batch_size=int(inference_batch_size),
            )
        )
        self.grid_height, self.grid_width = map(
            int, observation["spatial_grid"]
        )
        self.max_views = int(observation["max_views"])
        self.feature_dim = int(observation["feature_dim"])
        self.require_native_input_shape = bool(
            observation.get("require_native_input_shape", False)
        )
        self.input_height = int(
            observation.get("input_height", self.encoder.config.image_height)
        )
        self.input_width = int(
            observation.get("input_width", self.encoder.config.image_width)
        )

    def train(self, mode: bool = True) -> "R12SpatialObservationEncoder":
        del mode
        super().train(False)
        self.encoder.train(False)
        return self

    @torch.inference_mode()
    def forward(
        self,
        images: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 5 or images.shape[1] != self.max_views or images.shape[2] != 3:
            raise ValueError("R12 raw fixed-view RGB must be [batch,view,3,H,W]")
        if self.require_native_input_shape and tuple(images.shape[-2:]) != (
            self.input_height,
            self.input_width,
        ):
            raise ValueError(
                "R12-R4 requires native full-resolution fixed-view RGB before "
                "visual encoding"
            )
        if tuple(view_mask.shape) != tuple(images.shape[:2]):
            raise ValueError("R12 raw fixed-view mask shape differs")
        mask = view_mask.bool()
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every R12 sample requires at least one fixed view")
        batch = len(images)
        flat = images.reshape(-1, *images.shape[-3:])
        indices = mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        selected = flat.index_select(0, indices)
        encoded = self.encoder.forward_spatial_grid(
            selected,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
        ).spatial_tokens
        output = encoded.new_zeros(
            (
                batch * self.max_views,
                self.grid_height * self.grid_width,
                self.feature_dim,
            )
        )
        output.index_copy_(0, indices, encoded)
        output = output.reshape(
            batch,
            self.max_views,
            self.grid_height * self.grid_width,
            self.feature_dim,
        )
        return output.detach(), mask
