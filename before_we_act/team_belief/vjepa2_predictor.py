from __future__ import annotations

import torch
from torch import nn

from before_we_act.upstream_components.vjepa2.src.models.predictor import (
    VisionTransformerPredictor,
)


class VJEPA2BeliefEncoder(nn.Module):
    """Thin legal-input adapter around the exact upstream masked predictor."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        dim = int(config["embed_dim"])
        self.predictor = VisionTransformerPredictor(
            img_size=(4, 4), patch_size=1, num_frames=4, tubelet_size=1,
            embed_dim=dim, predictor_embed_dim=dim, out_embed_dim=dim,
            depth=int(config["depth"]), num_heads=int(config["num_heads"]),
            mlp_ratio=float(config["mlp_ratio"]), drop_rate=0.0,
            attn_drop_rate=0.0, drop_path_rate=0.0, use_mask_tokens=True,
            num_mask_tokens=2, zero_init_mask_tokens=True,
        )

    def forward(self, *, tokens, actions, qpos, agent_mask):
        del actions, qpos, agent_mask
        batch = tokens.shape[0]
        context = tokens.flatten(1, 2)
        masks_x = torch.arange(48, device=tokens.device).unsqueeze(0).expand(batch, -1)
        masks_y = torch.arange(48, 64, device=tokens.device).unsqueeze(0).expand(batch, -1)
        return self.predictor(context, masks_x, masks_y)


def build_encoder(config: dict) -> nn.Module:
    return VJEPA2BeliefEncoder(config)
