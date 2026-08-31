from __future__ import annotations

"""Strict-local DuoBench adaptation of the upstream LatentToM sheaf policy."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


def _groupnorm_resnet18() -> nn.Module:
    from torchvision.models import resnet18
    backbone = resnet18(weights=None)

    def replace(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                module._modules[name] = nn.GroupNorm(max(1, child.num_features // 16), child.num_features)
            else:
                replace(child)
    replace(backbone)
    backbone.fc = nn.Identity()
    return backbone


class DuoSheafEncoder(nn.Module):
    """Encode public head and actor-private wrist/proprioception separately."""

    def __init__(self, qpos_dim: int, task_dim: int, arm_id_dim: int, private_dim: int,
                 latent_dim: int, latent_hidden_dim: int, crop_ratio: float):
        super().__init__()
        self.shared_backbone = _groupnorm_resnet18()
        self.private_backbone = _groupnorm_resnet18()
        self.private = nn.Sequential(
            nn.Linear(512 + qpos_dim + task_dim + arm_id_dim, private_dim), nn.LayerNorm(private_dim), nn.SiLU()
        )
        self.latent = nn.Sequential(
            nn.Linear(private_dim, latent_hidden_dim), nn.SiLU(),
            nn.Linear(latent_hidden_dim, latent_dim), nn.Tanh()
        )
        self.private_dim = int(private_dim)
        self.latent_dim = int(latent_dim)
        self.crop_ratio = float(crop_ratio)

    @staticmethod
    def _imagenet(x: torch.Tensor) -> torch.Tensor:
        mean = x.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = x.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return (x - mean) / std

    def _crop(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        crop_h = max(1, int(round(height * self.crop_ratio)))
        crop_w = max(1, int(round(width * self.crop_ratio)))
        if crop_h == height and crop_w == width:
            return x
        if self.training:
            top = int(torch.randint(0, height - crop_h + 1, (), device=x.device))
            left = int(torch.randint(0, width - crop_w + 1, (), device=x.device))
            return x[:, :, top:top + crop_h, left:left + crop_w]
        top, left = (height - crop_h) // 2, (width - crop_w) // 2
        return x[:, :, top:top + crop_h, left:left + crop_w]

    def forward(self, image: torch.Tensor, qpos: torch.Tensor, task: torch.Tensor,
                arm_id: torch.Tensor):
        if image.ndim != 4 or image.shape[1] != 3 or image.shape[-1] != image.shape[-2] * 2:
            raise ValueError(f"expected [B,3,H,2H] image, got {tuple(image.shape)}")
        image = image.float()
        if image.detach().amax() > 1.5:
            image = image / 255.0
        head, wrist = image.chunk(2, dim=-1)
        head, wrist = self._crop(head), self._crop(wrist)
        shared = self.shared_backbone(self._imagenet(head))
        wrist_feature = self.private_backbone(self._imagenet(wrist))
        private = self.private(torch.cat((wrist_feature, qpos, task, arm_id), dim=-1))
        return private, self.latent(private), shared


class ToMCrossPredictor(nn.Module):
    """The upstream public-to-private cross-actor predictor."""

    def __init__(self, shared_dim: int, private_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.query_fc = nn.Linear(shared_dim, hidden_dim)
        self.key_fc = nn.Linear(private_dim, hidden_dim)
        self.value_fc = nn.Linear(private_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.out_fc = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, private_dim))

    def forward(self, shared: torch.Tensor, candidate_private: torch.Tensor):
        query = self.query_fc(shared).unsqueeze(1)
        key = self.key_fc(candidate_private).unsqueeze(1)
        value = self.value_fc(candidate_private).unsqueeze(1)
        attended, _ = self.attn(query, key, value)
        return self.out_fc(attended.squeeze(1))


class LocalLatentToMPolicy(nn.Module):
    """One shared checkpoint; inference has no peer input."""

    def __init__(self, *, horizon: int, obs_steps: int, action_dim: int,
                 qpos_dim: int, task_dim: int, latent_dim: int, private_dim: int,
                 latent_hidden_dim: int, tom_hidden_dim: int,
                 image_resize_hw: tuple[int, int], down_dims: tuple[int, ...],
                 diffusion_steps: int, beta_start: float, beta_end: float,
                 beta_schedule: str, prediction_type: str, clip_sample: bool,
                 tom_weight: float, inference_steps: int,
                 normalization_mode: str = "limits", vision_crop_ratio: float = 0.9):
        super().__init__()
        self.horizon, self.obs_steps = int(horizon), int(obs_steps)
        self.action_dim, self.qpos_dim, self.task_dim = int(action_dim), int(qpos_dim), int(task_dim)
        self.image_resize_hw = tuple(int(x) for x in image_resize_hw)
        self.tom_weight = float(tom_weight)
        self.normalization_mode = str(normalization_mode)
        self.arm_id_dim = 2
        self.encoder = DuoSheafEncoder(qpos_dim, task_dim, self.arm_id_dim, private_dim, latent_dim,
                                       latent_hidden_dim, vision_crop_ratio)
        self.tom_predictor = ToMCrossPredictor(512, private_dim, tom_hidden_dim)
        condition_dim = (private_dim + latent_dim + 512) * self.obs_steps
        self.scheduler = DDIMScheduler(
            num_train_timesteps=diffusion_steps, beta_start=beta_start, beta_end=beta_end,
            beta_schedule=beta_schedule, prediction_type=prediction_type,
            clip_sample=clip_sample, set_alpha_to_one=True, steps_offset=0,
        )
        self.model = ConditionalUnet1D(
            input_dim=action_dim, local_cond_dim=None, global_cond_dim=condition_dim,
            diffusion_step_embed_dim=128, down_dims=list(down_dims), kernel_size=5,
            n_groups=8, cond_predict_scale=True,
        )
        for name, dim in (("q_mean", qpos_dim), ("q_std", qpos_dim), ("q_min", qpos_dim),
                          ("q_max", qpos_dim), ("a_mean", action_dim), ("a_std", action_dim),
                          ("a_min", action_dim), ("a_max", action_dim)):
            self.register_buffer(name, torch.zeros(dim) if "mean" in name or "min" in name else torch.ones(dim))
        self.register_buffer("inference_steps", torch.tensor(inference_steps), persistent=False)

    @classmethod
    def from_config(cls, config: dict) -> "LocalLatentToMPolicy":
        model = config["model"]
        return cls(
            horizon=model["action_horizon"], obs_steps=model["observation_steps"], action_dim=model["action_dim"],
            qpos_dim=config["data"]["state_dim"], task_dim=model["task_dim"], latent_dim=model["latent_dim"],
            private_dim=model["private_dim"], latent_hidden_dim=model["latent_hidden_dim"],
            tom_hidden_dim=model["tom_hidden_dim"], image_resize_hw=tuple(model["image_resize_hw"]),
            down_dims=tuple(model["down_dims"]), diffusion_steps=model["diffusion_steps"],
            beta_start=model["diffusion_beta_start"], beta_end=model["diffusion_beta_end"],
            beta_schedule=model["diffusion_beta_schedule"], prediction_type=model["prediction_type"],
            clip_sample=model["clip_sample"], tom_weight=model["tom_loss_weight"],
            inference_steps=config["validation20"]["diffusion_steps"],
            normalization_mode=config["data"].get("normalization_mode", "limits"),
            vision_crop_ratio=model.get("vision_crop_ratio", 0.9),
        )

    def set_stats(self, stats: dict) -> None:
        for name, group, field in (
            ("q_mean", "qpos", "mean"), ("q_std", "qpos", "std"), ("q_min", "qpos", "min"), ("q_max", "qpos", "max"),
            ("a_mean", "action", "mean"), ("a_std", "action", "std"), ("a_min", "action", "min"), ("a_max", "action", "max"),
        ):
            value = torch.as_tensor(stats[group][field], dtype=torch.float32, device=self.q_mean.device)
            target = getattr(self, name)
            if value.shape != target.shape or not torch.isfinite(value).all():
                raise ValueError(f"invalid normalization vector {name}")
            target.copy_(value)
        self.q_std.clamp_min_(1e-6); self.a_std.clamp_min_(1e-6)
        self.q_max.copy_(torch.maximum(self.q_max, self.q_min + 1e-6))
        self.a_max.copy_(torch.maximum(self.a_max, self.a_min + 1e-6))

    def _norm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
              minimum: torch.Tensor, maximum: torch.Tensor) -> torch.Tensor:
        if self.normalization_mode == "limits":
            return 2 * (x - minimum) / (maximum - minimum).clamp_min(1e-6) - 1
        return (x - mean) / std.clamp_min(1e-6)

    def _norm_q(self, x): return self._norm(x, self.q_mean, self.q_std, self.q_min, self.q_max)
    def _norm_a(self, x): return self._norm(x, self.a_mean, self.a_std, self.a_min, self.a_max)

    def _unnorm_a(self, x):
        if self.normalization_mode == "limits":
            return self.a_min + (x + 1) * 0.5 * (self.a_max - self.a_min)
        return x * self.a_std + self.a_mean

    def _encode(self, obs: dict[str, torch.Tensor], *, sequence: bool = False):
        image = obs["image"]
        b, t = image.shape[:2]
        if t != self.obs_steps:
            raise ValueError(f"expected {self.obs_steps} image frames, got {t}")
        if tuple(image.shape[-2:]) != self.image_resize_hw:
            image = F.interpolate(image.reshape(b * t, *image.shape[2:]), self.image_resize_hw,
                                  mode="bilinear", align_corners=False).reshape(b, t, 3, *self.image_resize_hw)
        qpos = self._norm_q(obs["qpos"])
        task = obs["task"][:, None].expand(b, t, self.task_dim)
        if "arm_id" not in obs:
            raise ValueError("strict-local DuoBench policy requires local arm_id")
        arm_id = obs["arm_id"][:, None].expand(b, t, self.arm_id_dim)
        private, latent, shared = self.encoder(
            image.reshape(b * t, 3, *self.image_resize_hw), qpos.reshape(b * t, self.qpos_dim),
            task.reshape(b * t, self.task_dim), arm_id.reshape(b * t, self.arm_id_dim))
        private = private.reshape(b, t, -1); latent = latent.reshape(b, t, -1); shared = shared.reshape(b, t, -1)
        condition = torch.cat((private, latent, shared), dim=-1).reshape(b, -1)
        return (condition, private, latent, shared) if sequence else condition

    def loss(self, obs: dict[str, torch.Tensor], action: torch.Tensor, mask: torch.Tensor):
        normalized = self._norm_a(action)
        condition, private, _latent, shared = self._encode(obs, sequence=True)
        noise = torch.randn_like(normalized)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps,
                                  (normalized.shape[0],), device=normalized.device).long()
        noisy = self.scheduler.add_noise(normalized, noise, timesteps)
        predicted = self.model(noisy, timesteps, global_cond=condition)
        diffusion = ((predicted - noise).square().mean(-1) * mask).sum() / mask.sum().clamp_min(1)
        tom = normalized.new_zeros(())
        if "peer_image" in obs:
            peer = {"image": obs["peer_image"], "qpos": obs["peer_qpos"], "task": obs["task"],
                    "arm_id": obs["peer_arm_id"]}
            _pc, peer_private, _pl, peer_shared = self._encode(peer, sequence=True)
            cross_a = self.tom_predictor(shared[:, -1], peer_private[:, -1].detach())
            cross_b = self.tom_predictor(peer_shared[:, -1], private[:, -1].detach())
            tom = 0.5 * (F.mse_loss(cross_a, peer_private[:, -1].detach()) +
                         F.mse_loss(cross_b, private[:, -1].detach()))
        return diffusion + self.tom_weight * tom, {"diffusion": diffusion.detach(), "tom": tom.detach()}

    @torch.no_grad()
    def predict_chunk(self, obs: dict[str, torch.Tensor], *, steps: int | None = None):
        condition = self._encode(obs)
        sample = torch.randn((condition.shape[0], self.horizon, self.action_dim),
                             device=condition.device, dtype=condition.dtype)
        self.scheduler.set_timesteps(int(steps or self.inference_steps.item()), device=condition.device)
        for timestep in self.scheduler.timesteps:
            sample = self.scheduler.step(self.model(sample, timestep, global_cond=condition), timestep, sample).prev_sample
        return self._unnorm_a(sample)
