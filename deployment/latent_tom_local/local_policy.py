from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.common.lr_scheduler import get_scheduler  # noqa: F401
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


class LocalLatentEncoder(nn.Module):
    """Private visual/proprio feature plus a learned local ToM latent.

    The latent is computed exclusively from the actor's own observation.  No
    information from another actor, shared camera, global state, or joint
    action enters forward.
    """

    def __init__(self, task_dim: int = 6, latent_dim: int = 128):
        super().__init__()
        from torchvision.models import resnet18
        backbone = resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.private = nn.Sequential(nn.Linear(512 + 9 + task_dim, 512), nn.LayerNorm(512), nn.SiLU())
        self.latent = nn.Sequential(nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, latent_dim), nn.Tanh())
        self.private_dim = 512
        self.latent_dim = latent_dim

    def forward(self, image: torch.Tensor, qpos: torch.Tensor, task: torch.Tensor):
        visual = self.backbone(image)
        private = self.private(torch.cat([visual, qpos, task], dim=-1))
        latent = self.latent(private)
        return private, latent


class LocalLatentToMPolicy(nn.Module):
    def __init__(self, horizon=40, obs_steps=2, action_dim=8, task_dim=6,
                 down_dims=(256, 512, 1024), diffusion_steps=100):
        super().__init__()
        self.horizon, self.obs_steps, self.action_dim = int(horizon), int(obs_steps), int(action_dim)
        self.encoder = LocalLatentEncoder(task_dim=task_dim)
        # Strict-local analogue of LatentToM's private-feature predictor: infer
        # the next local private state from the actor's current latent only.
        self.tom_predictor = nn.Sequential(
            nn.Linear(self.encoder.latent_dim, 256), nn.SiLU(),
            nn.Linear(256, self.encoder.private_dim),
        )
        self.obs_dim = self.encoder.private_dim + self.encoder.latent_dim
        self.scheduler = DDIMScheduler(num_train_timesteps=diffusion_steps, beta_start=1e-4,
                                       beta_end=0.02, beta_schedule="squaredcos_cap_v2",
                                       # Actions are standardized with dataset mean/std, not
                                       # min-max encoded to [-1, 1].  Clipping the denoising
                                       # sample would make values beyond one standard deviation
                                       # unreachable (most visibly, a closed gripper target of
                                       # -1 standardizes to roughly -2.08).  DDIM clipping is
                                       # therefore invalid for this action representation.
                                       clip_sample=False, set_alpha_to_one=True,
                                       steps_offset=0, prediction_type="epsilon")
        self.model = ConditionalUnet1D(input_dim=action_dim, local_cond_dim=None,
                                       global_cond_dim=self.obs_dim * obs_steps,
                                       diffusion_step_embed_dim=128, down_dims=down_dims,
                                       kernel_size=5, n_groups=8, cond_predict_scale=True)
        self.register_buffer("q_mean", torch.zeros(9), persistent=True)
        self.register_buffer("q_std", torch.ones(9), persistent=True)
        self.register_buffer("a_mean", torch.zeros(action_dim), persistent=True)
        self.register_buffer("a_std", torch.ones(action_dim), persistent=True)
        self.register_buffer("inference_steps", torch.tensor(20), persistent=False)

    def set_stats(self, stats: dict):
        device = self.q_mean.device
        for name, key in (("q_mean", "qpos"), ("q_std", "qpos"), ("a_mean", "action"), ("a_std", "action")):
            value = torch.as_tensor(stats[key]["mean" if "mean" in name else "std"], device=device, dtype=torch.float32)
            getattr(self, name).copy_(value.clamp_min(1e-6) if "std" in name else value)

    def _encode(self, obs: dict[str, torch.Tensor], return_sequence: bool = False):
        image = obs["image"]
        if image.dtype == torch.uint8:
            image = image.float().div_(255.0)
        # HF trajectories are 640x480; the official LatentToM ResNet recipe
        # consumes 320x240 crops.  Resize before the backbone to cap VRAM.
        if image.shape[-2:] != (240, 320):
            b0, t0 = image.shape[:2]
            image = torch.nn.functional.interpolate(
                image.reshape(b0 * t0, *image.shape[2:]), size=(240, 320),
                mode="bilinear", align_corners=False
            ).reshape(b0, t0, 3, 240, 320)
        qpos = (obs["qpos"] - self.q_mean) / self.q_std.clamp_min(1e-6)
        task = obs["task"]
        b, t = image.shape[:2]
        private, latent = self.encoder(image.reshape(b * t, *image.shape[2:]),
                                       qpos.reshape(b * t, -1), task[:, None].expand(b, t, -1).reshape(b * t, -1))
        private = private.reshape(b, t, -1)
        latent = latent.reshape(b, t, -1)
        feat = torch.cat([private, latent], dim=-1)
        return (feat.reshape(b, -1), private, latent) if return_sequence else feat.reshape(b, -1)

    def loss(self, obs: dict[str, torch.Tensor], action: torch.Tensor):
        naction = (action - self.a_mean) / self.a_std.clamp_min(1e-6)
        cond, private, latent = self._encode(obs, return_sequence=True)
        noise = torch.randn_like(naction)
        ts = torch.randint(0, self.scheduler.config.num_train_timesteps, (naction.shape[0],), device=naction.device).long()
        noisy = self.scheduler.add_noise(naction, noise, ts)
        pred = self.model(noisy, ts, global_cond=cond)
        target = noise if self.scheduler.config.prediction_type == "epsilon" else naction
        diffusion_loss = F.mse_loss(pred, target)
        # No other actor is opened or passed: this auxiliary target is strictly
        # the same local stream at t+1, detached from the target encoder path.
        tom_loss = F.mse_loss(self.tom_predictor(latent[:, 0]), private[:, 1].detach())
        return diffusion_loss + 0.1 * tom_loss

    @torch.no_grad()
    def predict_chunk(self, obs: dict[str, torch.Tensor], steps: int | None = None):
        cond = self._encode(obs)
        b = cond.shape[0]
        sample = torch.randn((b, self.horizon, self.action_dim), device=cond.device, dtype=cond.dtype)
        self.scheduler.set_timesteps(int(steps or self.inference_steps.item()), device=cond.device)
        for t in self.scheduler.timesteps:
            sample = self.scheduler.step(self.model(sample, t, global_cond=cond), t, sample).prev_sample
        return sample * self.a_std + self.a_mean
