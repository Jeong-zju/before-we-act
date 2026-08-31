from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


class LocalLatentEncoder(nn.Module):
    def __init__(self, qpos_dim: int, task_dim: int, latent_dim: int,
                 private_dim: int, latent_hidden_dim: int):
        super().__init__()
        from torchvision.models import resnet18
        backbone = resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.private = nn.Sequential(
            nn.Linear(512 + qpos_dim + task_dim, private_dim),
            nn.LayerNorm(private_dim), nn.SiLU(),
        )
        self.latent = nn.Sequential(
            nn.Linear(private_dim, latent_hidden_dim), nn.SiLU(),
            nn.Linear(latent_hidden_dim, latent_dim), nn.Tanh(),
        )
        self.private_dim = int(private_dim)
        self.latent_dim = int(latent_dim)

    def forward(self, image: torch.Tensor, qpos: torch.Tensor, task: torch.Tensor):
        private = self.private(torch.cat((self.backbone(image), qpos, task), dim=-1))
        return private, self.latent(private)


class LocalLatentToMPolicy(nn.Module):
    """One-row graph: shared head, own wrist/qpos, fixed task id -> own action."""

    def __init__(self, *, horizon: int, obs_steps: int, action_dim: int,
                 qpos_dim: int, task_dim: int, latent_dim: int, private_dim: int,
                 latent_hidden_dim: int, tom_hidden_dim: int,
                 image_resize_hw: tuple[int, int], down_dims: tuple[int, ...],
                 diffusion_steps: int, beta_start: float, beta_end: float,
                 beta_schedule: str, prediction_type: str, clip_sample: bool,
                 tom_weight: float, inference_steps: int):
        super().__init__()
        self.horizon = int(horizon)
        self.obs_steps = int(obs_steps)
        self.action_dim = int(action_dim)
        self.qpos_dim = int(qpos_dim)
        self.task_dim = int(task_dim)
        self.image_resize_hw = tuple(int(x) for x in image_resize_hw)
        self.tom_weight = float(tom_weight)
        self.encoder = LocalLatentEncoder(
            qpos_dim, task_dim, latent_dim, private_dim, latent_hidden_dim
        )
        self.tom_predictor = nn.Sequential(
            nn.Linear(latent_dim, tom_hidden_dim), nn.SiLU(),
            nn.Linear(tom_hidden_dim, private_dim),
        )
        condition_dim = (private_dim + latent_dim) * self.obs_steps
        self.scheduler = DDIMScheduler(
            num_train_timesteps=diffusion_steps, beta_start=beta_start,
            beta_end=beta_end, beta_schedule=beta_schedule,
            prediction_type=prediction_type, clip_sample=clip_sample,
            set_alpha_to_one=True, steps_offset=0,
        )
        if bool(self.scheduler.config.clip_sample):
            raise ValueError("standardized DuoBench actions require DDIM clip_sample=False")
        self.model = ConditionalUnet1D(
            input_dim=action_dim, local_cond_dim=None, global_cond_dim=condition_dim,
            diffusion_step_embed_dim=128, down_dims=list(down_dims), kernel_size=5,
            n_groups=8, cond_predict_scale=True,
        )
        self.register_buffer("q_mean", torch.zeros(qpos_dim))
        self.register_buffer("q_std", torch.ones(qpos_dim))
        self.register_buffer("a_mean", torch.zeros(action_dim))
        self.register_buffer("a_std", torch.ones(action_dim))
        self.register_buffer("inference_steps", torch.tensor(inference_steps), persistent=False)

    @classmethod
    def from_config(cls, config: dict) -> "LocalLatentToMPolicy":
        model = config["model"]
        return cls(
            horizon=model["action_horizon"], obs_steps=model["observation_steps"],
            action_dim=model["action_dim"], qpos_dim=config["data"]["state_dim"],
            task_dim=model["task_dim"], latent_dim=model["latent_dim"],
            private_dim=model["private_dim"], latent_hidden_dim=model["latent_hidden_dim"],
            tom_hidden_dim=model["tom_hidden_dim"], image_resize_hw=tuple(model["image_resize_hw"]),
            down_dims=tuple(model["down_dims"]), diffusion_steps=model["diffusion_steps"],
            beta_start=model["diffusion_beta_start"], beta_end=model["diffusion_beta_end"],
            beta_schedule=model["diffusion_beta_schedule"], prediction_type=model["prediction_type"],
            clip_sample=model["clip_sample"], tom_weight=model["tom_loss_weight"],
            inference_steps=config["validation20"]["diffusion_steps"],
        )

    def set_stats(self, stats: dict) -> None:
        for name, group, field in (
            ("q_mean", "qpos", "mean"), ("q_std", "qpos", "std"),
            ("a_mean", "action", "mean"), ("a_std", "action", "std"),
        ):
            value = torch.as_tensor(stats[group][field], dtype=torch.float32, device=self.q_mean.device)
            if value.shape != getattr(self, name).shape or not torch.isfinite(value).all():
                raise ValueError(f"invalid normalization vector {name}")
            getattr(self, name).copy_(value.clamp_min(1e-6) if name.endswith("std") else value)

    def _encode(self, obs: dict[str, torch.Tensor], *, sequence: bool = False):
        image = obs["image"]
        if image.dtype == torch.uint8:
            image = image.float().div_(255.0)
        b, t = image.shape[:2]
        if t != self.obs_steps:
            raise ValueError(f"expected {self.obs_steps} image frames, got {t}")
        if image.shape[-2:] != self.image_resize_hw:
            image = F.interpolate(
                image.reshape(b * t, *image.shape[2:]), self.image_resize_hw,
                mode="bilinear", align_corners=False,
            ).reshape(b, t, 3, *self.image_resize_hw)
        qpos = (obs["qpos"] - self.q_mean) / self.q_std.clamp_min(1e-6)
        task = obs["task"][:, None].expand(b, t, self.task_dim)
        private, latent = self.encoder(
            image.reshape(b * t, 3, *self.image_resize_hw),
            qpos.reshape(b * t, self.qpos_dim),
            task.reshape(b * t, self.task_dim),
        )
        private = private.reshape(b, t, -1)
        latent = latent.reshape(b, t, -1)
        condition = torch.cat((private, latent), dim=-1).reshape(b, -1)
        return (condition, private, latent) if sequence else condition

    def loss(self, obs: dict[str, torch.Tensor], action: torch.Tensor,
             mask: torch.Tensor):
        normalized = (action - self.a_mean) / self.a_std.clamp_min(1e-6)
        condition, private, latent = self._encode(obs, sequence=True)
        noise = torch.randn_like(normalized)
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (normalized.shape[0],), device=normalized.device,
        ).long()
        noisy = self.scheduler.add_noise(normalized, noise, timesteps)
        predicted = self.model(noisy, timesteps, global_cond=condition)
        diffusion = ((predicted - noise).square().mean(-1) * mask).sum() / mask.sum().clamp_min(1)
        tom = F.mse_loss(self.tom_predictor(latent[:, 0]), private[:, -1].detach())
        return diffusion + self.tom_weight * tom, {"diffusion": diffusion.detach(), "tom": tom.detach()}

    @torch.no_grad()
    def predict_chunk(self, obs: dict[str, torch.Tensor], *, steps: int | None = None):
        condition = self._encode(obs)
        sample = torch.randn(
            (condition.shape[0], self.horizon, self.action_dim),
            device=condition.device, dtype=condition.dtype,
        )
        self.scheduler.set_timesteps(int(steps or self.inference_steps.item()), device=condition.device)
        for timestep in self.scheduler.timesteps:
            predicted = self.model(sample, timestep, global_cond=condition)
            sample = self.scheduler.step(predicted, timestep, sample).prev_sample
        return sample * self.a_std + self.a_mean
