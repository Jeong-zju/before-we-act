from __future__ import annotations

import copy, numpy as np, torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy

# Official RoboFactory DP temporal contract. predict_action() exposes indices
# 2..7, so the v2 evaluator executes six actions before replanning.
OBS_STEPS, HORIZON, ACTION_STEPS = 3, 8, 8

def _limits(low, high):
    low, high = np.asarray(low, np.float32), np.asarray(high, np.float32); span = np.maximum(high - low, 1e-4)
    scale, offset = 2.0 / span, -1.0 - 2.0 * low / span
    return SingleFieldLinearNormalizer.create_manual(scale, offset, {"min": low, "max": high, "mean": (low + high) / 2, "std": span / np.sqrt(12.0)})

def build_policy(stats, device="cpu"):
    shape = {"obs": {"head_cam": {"shape": [3, 240, 320], "type": "rgb"}, "agent_pos": {"shape": [9], "type": "low_dim"}}, "action": {"shape": [8]}}
    encoder = MultiImageObsEncoder(shape, get_resnet("resnet18", weights=None), use_group_norm=True, imagenet_norm=True)
    policy = DiffusionUnetImagePolicy(shape, DDPMScheduler(num_train_timesteps=100, beta_start=1e-4, beta_end=.02, beta_schedule="squaredcos_cap_v2", variance_type="fixed_small", clip_sample=True, prediction_type="epsilon"), encoder, HORIZON, ACTION_STEPS, OBS_STEPS, num_inference_steps=100, obs_as_global_cond=True, diffusion_step_embed_dim=128, down_dims=(256, 512, 1024), kernel_size=5, n_groups=8, cond_predict_scale=True)
    normalizer = LinearNormalizer(); normalizer["head_cam"] = SingleFieldLinearNormalizer.create_identity(); normalizer["agent_pos"] = _limits(stats["q_min"], stats["q_max"]); normalizer["action"] = _limits(stats["a_min"], stats["a_max"]); policy.set_normalizer(normalizer)
    # LinearNormalizer is a registered module; move its frozen scale/offset
    # tensors together with the network so CUDA inference never falls back to
    # a CPU tensor during normalization.
    policy = policy.to(device)
    policy.normalizer = policy.normalizer.to(device)
    return policy

def load_policy(checkpoint, device="cuda:0", inference_steps=20):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("contract") != "shared_weights_strict_local_rgb_qpos_to_absolute_action8": raise RuntimeError("checkpoint contract mismatch")
    policy = build_policy(payload["stats"], device); policy.load_state_dict(payload["ema_model"], strict=True); policy.normalizer = policy.normalizer.to(device); policy.num_inference_steps = int(inference_steps); policy.eval()
    return policy, payload
