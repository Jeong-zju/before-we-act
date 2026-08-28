from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

from maniflow.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from maniflow.model.vision_2d.timm_obs_encoder import TimmObsEncoder
from maniflow.policy.maniflow_image_policy import ManiFlowTransformerImagePolicy
from .common import POLICY_CONTRACT, UPSTREAM_COMMIT, UPSTREAM_REPO

OBS_STEPS = 2
HORIZON = 16
ACTION_STEPS = 15  # Upstream starts at obs_steps-1 inside a length-16 trajectory.
IMAGE_SIZE = 224


def model_config():
    return {
        "upstream_repo": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT,
        "policy": "ManiFlowTransformerImagePolicy",
        "obs_steps": OBS_STEPS, "horizon": HORIZON, "action_steps": ACTION_STEPS,
        "image_size": IMAGE_SIZE, "action_dim": 8, "qpos_dim": 9,
        "encoder": "resnet34.a1_in1k", "n_layer": 12, "n_head": 8, "n_emb": 768,
        "flow_batch_ratio": 0.75, "consistency_batch_ratio": 0.25,
        "normalization": "global_all-data_qpos/action_limits_to_minus1_plus1",
        "rgb_preprocessing": "uint8_div_255_then_bilinear_resize_224",
        "policy_contract": POLICY_CONTRACT,
    }


def build_policy(device: str | torch.device = "cpu"):
    shape_meta = OmegaConf.create({
        "obs": {
            "head_cam": {"shape": [3, IMAGE_SIZE, IMAGE_SIZE], "type": "rgb", "horizon": OBS_STEPS},
            "agent_pos": {"shape": [9], "type": "low_dim", "horizon": OBS_STEPS},
        },
        "action": {"shape": [8], "horizon": HORIZON},
    })
    encoder = TimmObsEncoder(
        shape_meta=shape_meta, model_name="resnet34.a1_in1k", pretrained=False,
        frozen=False, global_pool="", transforms=None, use_group_norm=True,
        share_rgb_model=False, imagenet_norm=False, feature_aggregation="avg",
        downsample_ratio=32, position_encording="sinusoidal",
    )
    policy = ManiFlowTransformerImagePolicy(
        shape_meta=shape_meta, horizon=HORIZON, n_action_steps=ACTION_STEPS,
        n_obs_steps=OBS_STEPS, num_inference_steps=10, obs_as_global_cond=True,
        diffusion_timestep_embed_dim=128, diffusion_target_t_embed_dim=128,
        visual_cond_len=1, n_layer=12, n_head=8, n_emb=768,
        qkv_bias=True, qk_norm=True, block_type="DiTX", language_conditioned=False,
        flow_batch_ratio=0.75, consistency_batch_ratio=0.25, denoise_timesteps=10,
        sample_t_mode_flow="beta", sample_t_mode_consistency="discrete",
        sample_dt_mode_consistency="uniform", sample_target_t_mode="relative",
        obs_encoder=encoder,
    )
    normalizer = LinearNormalizer()
    normalizer["head_cam"] = SingleFieldLinearNormalizer.create_identity()
    normalizer["agent_pos"] = SingleFieldLinearNormalizer.create_identity()
    # The MARS loader maps full-corpus qpos/action limits to [-1,1].  Identity
    # here prevents a second normalization; evaluation performs the exact
    # same encode/decode once using checkpoint-embedded statistics.
    scale = np.ones(8, np.float32); offset = np.zeros(8, np.float32)
    stats = {"min": -scale, "max": scale, "mean": offset, "std": scale}
    normalizer["action"] = SingleFieldLinearNormalizer.create_manual(scale, offset, stats)
    policy.set_normalizer(normalizer)
    return policy.to(device)


def load_policy(checkpoint: str, device: str | torch.device = "cuda:0"):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = model_config()
    if payload.get("config") != expected or payload.get("contract") != expected["policy_contract"]:
        raise RuntimeError("ManiFlow checkpoint/config contract mismatch")
    policy = build_policy(device)
    policy.load_state_dict(payload.get("ema_model", payload["model"]), strict=True)
    policy.eval()
    return policy, payload
