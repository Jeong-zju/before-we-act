from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


class LocalLatentEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128, private_dim: int = 512,
                 latent_hidden_dim: int = 256, qpos_dim: int = 9):
        super().__init__()
        from torchvision.models import resnet18
        backbone = resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.private = nn.Sequential(nn.Linear(512 + qpos_dim, private_dim), nn.LayerNorm(private_dim), nn.SiLU())
        self.latent = nn.Sequential(nn.Linear(private_dim, latent_hidden_dim), nn.SiLU(), nn.Linear(latent_hidden_dim, latent_dim), nn.Tanh())
        self.private_dim, self.latent_dim = int(private_dim), int(latent_dim)

    def forward(self, image, qpos):
        visual = self.backbone(image)
        private = self.private(torch.cat([visual, qpos], dim=-1))
        return private, self.latent(private)


class LocalLatentToMPolicy(nn.Module):
    """Shared policy whose forward graph contains only one actor's streams."""
    def __init__(self, horizon=40, obs_steps=2, action_dim=8, qpos_dim=9,
                 latent_dim=128, private_dim=512, down_dims=(256, 512, 1024),
                 diffusion_steps=100, beta_start=1e-4, beta_end=0.02,
                 beta_schedule="squaredcos_cap_v2", clip_sample=False,
                 diffusion_step_embed_dim=128, kernel_size=5, n_groups=8,
                 cond_predict_scale=True, tom_weight=0.1, latent_hidden_dim=256,
                 tom_hidden_dim=256, image_resize_hw=(240, 320),
                 prediction_type="epsilon", set_alpha_to_one=True, steps_offset=0,
                 inference_steps=20, trained_betas=None, thresholding=False,
                 dynamic_thresholding_ratio=0.995, clip_sample_range=1.0,
                 sample_max_value=1.0, timestep_spacing="leading",
                 rescale_betas_zero_snr=False):
        super().__init__()
        self.horizon, self.obs_steps, self.action_dim = int(horizon), int(obs_steps), int(action_dim)
        self.qpos_dim, self.tom_weight = int(qpos_dim), float(tom_weight)
        self.image_resize_hw = tuple(int(x) for x in image_resize_hw)
        self.encoder = LocalLatentEncoder(latent_dim=latent_dim, private_dim=private_dim, latent_hidden_dim=latent_hidden_dim, qpos_dim=qpos_dim)
        self.tom_predictor = nn.Sequential(nn.Linear(latent_dim, tom_hidden_dim), nn.SiLU(), nn.Linear(tom_hidden_dim, private_dim))
        self.obs_dim = private_dim + latent_dim
        self.scheduler = DDIMScheduler(num_train_timesteps=diffusion_steps, beta_start=beta_start, beta_end=beta_end,
                                       beta_schedule=beta_schedule, clip_sample=clip_sample,
                                       trained_betas=trained_betas, set_alpha_to_one=set_alpha_to_one,
                                       steps_offset=steps_offset, prediction_type=prediction_type,
                                       thresholding=thresholding,
                                       dynamic_thresholding_ratio=dynamic_thresholding_ratio,
                                       clip_sample_range=clip_sample_range, sample_max_value=sample_max_value,
                                       timestep_spacing=timestep_spacing,
                                       rescale_betas_zero_snr=rescale_betas_zero_snr)
        self.model = ConditionalUnet1D(input_dim=action_dim, local_cond_dim=None,
                                        global_cond_dim=self.obs_dim * obs_steps,
                                        diffusion_step_embed_dim=diffusion_step_embed_dim, down_dims=down_dims,
                                        kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale)
        self.register_buffer("q_mean", torch.zeros(qpos_dim)); self.register_buffer("q_std", torch.ones(qpos_dim))
        self.register_buffer("a_mean", torch.zeros(action_dim)); self.register_buffer("a_std", torch.ones(action_dim))
        self.register_buffer("inference_steps", torch.tensor(inference_steps), persistent=False)

    @classmethod
    def from_frozen_config(cls, frozen):
        model = frozen["model"]; denoiser = model["denoiser"]; diffusion = model["diffusion_scheduler"]
        return cls(
            horizon=model["action_horizon"], obs_steps=model["observation_steps"],
            action_dim=model["action_dim"], qpos_dim=frozen["data"]["qpos_dim"],
            latent_dim=model["latent_dim"], private_dim=model["private_dim"],
            latent_hidden_dim=model["latent_hidden_dim"], tom_hidden_dim=model["tom_hidden_dim"],
            image_resize_hw=tuple(model["image_resize_hw"]),
            down_dims=tuple(denoiser["down_dims"]), diffusion_steps=diffusion["num_train_timesteps"],
            beta_start=diffusion["beta_start"], beta_end=diffusion["beta_end"],
            beta_schedule=diffusion["beta_schedule"], clip_sample=diffusion["clip_sample"],
            prediction_type=diffusion["prediction_type"], set_alpha_to_one=diffusion["set_alpha_to_one"],
            steps_offset=diffusion["steps_offset"], inference_steps=frozen["validation20"]["diffusion_steps"],
            trained_betas=diffusion["trained_betas"], thresholding=diffusion["thresholding"],
            dynamic_thresholding_ratio=diffusion["dynamic_thresholding_ratio"],
            clip_sample_range=diffusion["clip_sample_range"], sample_max_value=diffusion["sample_max_value"],
            timestep_spacing=diffusion["timestep_spacing"],
            rescale_betas_zero_snr=diffusion["rescale_betas_zero_snr"],
            diffusion_step_embed_dim=denoiser["diffusion_step_embed_dim"],
            kernel_size=denoiser["kernel_size"], n_groups=denoiser["n_groups"],
            cond_predict_scale=denoiser["cond_predict_scale"], tom_weight=model["objective"]["tom_weight"],
        )

    def set_stats(self, stats):
        for name, key in (("q_mean", "qpos"), ("q_std", "qpos"), ("a_mean", "action"), ("a_std", "action")):
            value = torch.as_tensor(stats[key]["mean" if name.endswith("mean") else "std"], device=self.q_mean.device)
            getattr(self, name).copy_(value.clamp_min(1e-6) if name.endswith("std") else value)

    def _encode(self, obs, sequence=False):
        image = obs["image"]
        if image.dtype == torch.uint8: image = image.float().div_(255)
        if image.shape[-2:] != self.image_resize_hw:
            b, t = image.shape[:2]
            image = F.interpolate(image.reshape(b * t, *image.shape[2:]), self.image_resize_hw, mode="bilinear", align_corners=False).reshape(b, t, 3, *self.image_resize_hw)
        qpos = (obs["qpos"] - self.q_mean) / self.q_std.clamp_min(1e-6)
        b, t = image.shape[:2]
        private, latent = self.encoder(image.reshape(b * t, 3, *self.image_resize_hw), qpos.reshape(b * t, self.qpos_dim))
        private, latent = private.reshape(b, t, -1), latent.reshape(b, t, -1)
        feature = torch.cat([private, latent], dim=-1)
        return (feature.reshape(b, -1), private, latent) if sequence else feature.reshape(b, -1)

    def loss(self, obs, action, mask):
        normalized = (action - self.a_mean) / self.a_std.clamp_min(1e-6)
        condition, private, latent = self._encode(obs, sequence=True)
        noise = torch.randn_like(normalized)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (normalized.shape[0],), device=normalized.device).long()
        noisy = self.scheduler.add_noise(normalized, noise, timesteps)
        predicted = self.model(noisy, timesteps, global_cond=condition)
        diffusion = ((predicted - noise).square().mean(-1) * mask).sum() / mask.sum().clamp_min(1)
        tom = F.mse_loss(self.tom_predictor(latent[:, 0]), private[:, -1].detach())
        return diffusion + self.tom_weight * tom, {"diffusion": diffusion.detach(), "tom": tom.detach()}

    @torch.no_grad()
    def predict_chunk(self, obs, steps=20):
        condition = self._encode(obs)
        sample = torch.randn((condition.shape[0], self.horizon, self.action_dim), device=condition.device, dtype=condition.dtype)
        self.scheduler.set_timesteps(int(steps), device=condition.device)
        for timestep in self.scheduler.timesteps:
            sample = self.scheduler.step(self.model(sample, timestep, global_cond=condition), timestep, sample).prev_sample
        return sample * self.a_std + self.a_mean
