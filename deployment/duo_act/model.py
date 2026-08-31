from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ACT(nn.Module):
    """The same ResNet18 CVAE-ACT used by the local RoboFactory baseline."""

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 8,
        horizon: int = 100,
        d_model: int = 384,
        encoder_layers: int = 4,
        decoder_layers: int = 7,
        latent_dim: int = 32,
        tasks: int = 11,
    ):
        super().__init__()
        backbone = resnet18(weights=None)
        self.vision = nn.Sequential(*list(backbone.children())[:-2])
        self.vision_proj = nn.Conv2d(512, d_model, 1)
        self.state = nn.Sequential(nn.Linear(state_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        # ACT has no language encoder.  A learned embedding is the compact
        # equivalent of the shared task instruction and is identical for both arms.
        self.task = nn.Embedding(tasks, d_model)
        self.action = nn.Linear(action_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        self.query = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model, 8, d_model * 4, dropout=0.1, batch_first=True, norm_first=True, activation="gelu"
        )
        dec = nn.TransformerDecoderLayer(
            d_model, 8, d_model * 4, dropout=0.1, batch_first=True, norm_first=True, activation="gelu"
        )
        self.posterior = nn.TransformerEncoder(enc, num_layers=encoder_layers)
        self.decoder = nn.TransformerDecoder(dec, num_layers=decoder_layers)
        self.latent = nn.Linear(d_model, latent_dim * 2)
        self.z_proj = nn.Linear(latent_dim, d_model)
        self.out = nn.Linear(d_model, action_dim)
        self.config = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "horizon": horizon,
            "d_model": d_model,
            "encoder_layers": encoder_layers,
            "decoder_layers": decoder_layers,
            "latent_dim": latent_dim,
            "tasks": tasks,
            "vision_backbone": "resnet18_scratch",
        }

    def forward(self, image, qpos, task_id, actions=None):
        # image is head RGB and the arm-local wrist RGB side by side.  No peer
        # wrist or peer proprioception enters this shared-weight policy.
        image = F.interpolate(image, size=(256, 256), mode="bilinear", align_corners=False)
        vision = self.vision_proj(self.vision(image)).flatten(2).transpose(1, 2)
        state = self.state(qpos).unsqueeze(1)
        task = self.task(task_id).unsqueeze(1)
        if actions is not None:
            hidden = self.posterior(self.action(actions) + self.pos)
            mu, logvar = self.latent(hidden.mean(1)).chunk(2, -1)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            mu = logvar = None
            z = torch.zeros((image.shape[0], self.z_proj.in_features), device=image.device)
        memory = torch.cat((state, task, self.z_proj(z).unsqueeze(1), vision), dim=1)
        pred = self.out(self.decoder(self.query.expand(image.shape[0], -1, -1), memory))
        return pred, mu, logvar

