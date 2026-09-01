from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy

from .common import HORIZON, IMAGE_SIZE, OBS_STEPS, TASKS, policy_contract


def _limits(low, high):
    low = np.asarray(low, np.float32)
    high = np.asarray(high, np.float32)
    span = np.maximum(high - low, np.float32(1e-4)).astype(np.float32)
    scale = (np.float32(2.0) / span).astype(np.float32)
    offset = (np.float32(-1.0) - scale * low).astype(np.float32)
    return SingleFieldLinearNormalizer.create_manual(
        scale,
        offset,
        {
            "min": low,
            "max": high,
            "mean": ((low + high) / np.float32(2.0)).astype(np.float32),
            "std": (span / np.float32(np.sqrt(12.0))).astype(np.float32),
        },
    )


class TransitionAwareDiffusionUnetImagePolicy(DiffusionUnetImagePolicy):
    """DP epsilon loss with an explicit weight on the binary gripper channel."""

    def __init__(self, *args, gripper_loss_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gripper_loss_weight = float(gripper_loss_weight)
        if self.gripper_loss_weight < 1.0:
            raise ValueError("gripper_loss_weight must be >= 1")

    def compute_loss(self, batch, **kwargs):
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
            )
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()
        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        pred = self.model(
            noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond
        )
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"unsupported prediction type: {pred_type}")
        loss = F.mse_loss(pred, target, reduction="none")
        weights = torch.ones(self.action_dim, dtype=loss.dtype, device=loss.device)
        weights[-1] = self.gripper_loss_weight
        loss = loss * weights * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean").mean()
        if "output_pred" in kwargs:
            return loss, pred
        return loss


def build_policy(
    stats: dict,
    device: str | torch.device = "cpu",
    inference_steps: int = 100,
    *,
    task_conditioning: bool = False,
    gripper_loss_weight: float = 1.0,
):
    state_dim = 8 + (len(TASKS) if task_conditioning else 0)
    shape_meta = {
        "obs": {
            "head_wrist": {"shape": [3, IMAGE_SIZE, IMAGE_SIZE * 2], "type": "rgb"},
            "agent_pos": {"shape": [state_dim], "type": "low_dim"},
        },
        "action": {"shape": [8]},
    }
    encoder = MultiImageObsEncoder(
        shape_meta,
        get_resnet("resnet18", weights=None),
        resize_shape=None,
        crop_shape=None,
        random_crop=False,
        use_group_norm=True,
        share_rgb_model=False,
        imagenet_norm=True,
    )
    scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        variance_type="fixed_small",
        clip_sample=True,
        prediction_type="epsilon",
    )
    policy_class = (
        TransitionAwareDiffusionUnetImagePolicy
        if gripper_loss_weight != 1.0
        else DiffusionUnetImagePolicy
    )
    policy_kwargs = {}
    if policy_class is TransitionAwareDiffusionUnetImagePolicy:
        policy_kwargs["gripper_loss_weight"] = gripper_loss_weight
    policy = policy_class(
        shape_meta,
        scheduler,
        encoder,
        HORIZON,
        HORIZON,
        OBS_STEPS,
        num_inference_steps=int(inference_steps),
        obs_as_global_cond=True,
        diffusion_step_embed_dim=128,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        **policy_kwargs,
    )
    normalizer = LinearNormalizer()
    normalizer["head_wrist"] = SingleFieldLinearNormalizer.create_identity()
    if task_conditioning:
        state_low = np.concatenate(
            (np.asarray(stats["q_min"], np.float32), np.zeros(len(TASKS), np.float32))
        )
        state_high = np.concatenate(
            (np.asarray(stats["q_max"], np.float32), np.ones(len(TASKS), np.float32))
        )
        normalizer["agent_pos"] = _limits(state_low, state_high)
    else:
        normalizer["agent_pos"] = _limits(stats["q_min"], stats["q_max"])
    normalizer["action"] = _limits(stats["a_min"], stats["a_max"])
    policy.set_normalizer(normalizer)
    policy = policy.to(device)
    policy.normalizer = policy.normalizer.to(device)
    return policy


def load_policy(checkpoint, device="cuda:0", inference_steps=20, weights="ema"):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    task_conditioning = bool(payload.get("config", {}).get("task_conditioning", False))
    if payload.get("policy_contract") != policy_contract(task_conditioning):
        raise RuntimeError("checkpoint policy contract mismatch")
    key = {"ema": "ema_model", "online": "model"}.get(weights)
    if key is None or key not in payload:
        raise ValueError(f"checkpoint has no requested weights: {weights}")
    policy = build_policy(
        payload["stats"], device, inference_steps, task_conditioning=task_conditioning
    )
    policy.load_state_dict(payload[key], strict=True)
    policy.normalizer = policy.normalizer.to(device)
    policy.eval()
    return policy, payload
