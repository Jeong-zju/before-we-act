from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

from maniflow.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from maniflow.model.vision_2d.timm_obs_encoder import TimmObsEncoder
from maniflow.policy.maniflow_image_policy import ManiFlowTransformerImagePolicy
from .common import POLICY_CONTRACT, UPSTREAM_COMMIT, UPSTREAM_REPO, load_frozen_config

OBS_STEPS = 2
HORIZON = 16
ACTION_STEPS = 15  # Upstream starts at obs_steps-1 inside a length-16 trajectory.
IMAGE_SIZE = 224


def model_config():
    frozen = load_frozen_config()["model"]
    vision = frozen["vision"]
    transformer = frozen["transformer"]
    return {
        "upstream_repo": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT,
        "policy": frozen["class"],
        "obs_steps": frozen["n_obs_steps"], "horizon": frozen["horizon"],
        "action_steps": frozen["n_action_steps"],
        "image_size": vision["input_size"][1], "action_dim": frozen["action_dim"],
        "qpos_dim": frozen["qpos_dim"], "encoder": vision["model"],
        "n_layer": transformer["n_layer"], "n_head": transformer["n_head"],
        "n_emb": transformer["n_emb"], "flow_batch_ratio": frozen["flow_batch_ratio"],
        "consistency_batch_ratio": frozen["consistency_batch_ratio"],
        "normalization": "global_all-data_qpos/action_limits_to_minus1_plus1",
        "temporal_contract": "obs_tminus1_t_action_tminus1_through_tplus14_execute_from_index1",
        "action_clip": "robofactory_pd_joint_pos_before_stats_and_targets",
        "rgb_preprocessing": "uint8_div_255_then_bilinear_resize_224",
        "policy_contract": POLICY_CONTRACT,
    }


def build_policy(device: str | torch.device = "cpu"):
    frozen = load_frozen_config()["model"]
    vision = frozen["vision"]
    transformer = frozen["transformer"]
    image_size = vision["input_size"][1]
    shape_meta = OmegaConf.create({
        "obs": {
            "head_cam": {"shape": vision["input_size"], "type": "rgb", "horizon": frozen["n_obs_steps"]},
            "agent_pos": {"shape": [frozen["qpos_dim"]], "type": "low_dim", "horizon": frozen["n_obs_steps"]},
        },
        "action": {"shape": [frozen["action_dim"]], "horizon": frozen["horizon"]},
    })
    encoder = TimmObsEncoder(
        shape_meta=shape_meta, model_name=vision["model"], pretrained=vision["pretrained"],
        frozen=vision["frozen"], global_pool=vision["global_pool"], transforms=None,
        use_group_norm=vision["use_group_norm"], share_rgb_model=vision["share_rgb_model"],
        imagenet_norm=vision["imagenet_norm"], feature_aggregation=vision["feature_aggregation"],
        downsample_ratio=vision["downsample_ratio"], position_encording=vision["position_encoding"],
    )
    policy = ManiFlowTransformerImagePolicy(
        shape_meta=shape_meta, horizon=frozen["horizon"], n_action_steps=frozen["n_action_steps"],
        n_obs_steps=frozen["n_obs_steps"], num_inference_steps=frozen["denoise_timesteps"],
        obs_as_global_cond=transformer["obs_as_global_cond"],
        diffusion_timestep_embed_dim=frozen["diffusion_timestep_embed_dim"],
        diffusion_target_t_embed_dim=frozen["diffusion_target_t_embed_dim"],
        visual_cond_len=transformer["visual_cond_len"], n_layer=transformer["n_layer"],
        n_head=transformer["n_head"], n_emb=transformer["n_emb"],
        qkv_bias=transformer["qkv_bias"], qk_norm=transformer["qk_norm"],
        block_type=transformer["block_type"], language_conditioned=transformer["language_conditioned"],
        flow_batch_ratio=frozen["flow_batch_ratio"], consistency_batch_ratio=frozen["consistency_batch_ratio"],
        denoise_timesteps=frozen["denoise_timesteps"], sample_t_mode_flow=frozen["sample_t_mode_flow"],
        sample_t_mode_consistency=frozen["sample_t_mode_consistency"],
        sample_dt_mode_consistency=frozen["sample_dt_mode_consistency"],
        sample_target_t_mode=frozen["sample_target_t_mode"],
        obs_encoder=encoder,
    )
    normalizer = LinearNormalizer()
    normalizer["head_cam"] = SingleFieldLinearNormalizer.create_identity()
    normalizer["agent_pos"] = SingleFieldLinearNormalizer.create_identity()
    # The MARS loader maps full-corpus qpos/action limits to [-1,1].  Identity
    # here prevents a second normalization; evaluation performs the exact
    # same encode/decode once using checkpoint-embedded statistics.
    scale = np.ones(frozen["action_dim"], np.float32); offset = np.zeros(frozen["action_dim"], np.float32)
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
