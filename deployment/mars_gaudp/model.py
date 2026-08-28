from __future__ import annotations
import numpy as np, torch
from omegaconf import OmegaConf
from model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from model.vision.model_getter import get_resnet
from model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from policy.gaudp import GauDP
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

def _limits(low, high):
    low, high=np.asarray(low,np.float32),np.asarray(high,np.float32); span=np.maximum(high-low,1e-4)
    return SingleFieldLinearNormalizer.create_manual(2/span,-1-2*low/span,{"min":low,"max":high,"mean":(low+high)/2,"std":span/np.sqrt(12)})

def build_model(stats, device="cpu", inference_steps=100):
    shape={"obs":{"head_cam_0":{"shape":[3,120,160],"type":"rgb"},"state":{"shape":[9],"type":"low_dim"}},"action":{"shape":[8]}}
    encoder=MultiImageObsEncoder(shape,get_resnet("resnet18",weights=None),resize_shape=None,crop_shape=None,random_crop=False,use_group_norm=True,share_rgb_model=False,imagenet_norm=True)
    sched=DDPMScheduler(num_train_timesteps=100,beta_start=1e-4,beta_end=.02,beta_schedule="squaredcos_cap_v2",variance_type="fixed_small",clip_sample=True,prediction_type="epsilon")
    policy=GauDP(shape_meta=shape,noise_scheduler=sched,obs_encoder=encoder,optimazer_cfg=OmegaConf.create({"lr":1e-4,"betas":[.95,.999],"eps":1e-8,"weight_decay":1e-6}),scheduler_cfg=OmegaConf.create({"scheduler":"cosine","warmup_steps":500}),horizon=8,n_action_steps=6,n_obs_steps=3,num_inference_steps=inference_steps,obs_as_global_cond=True,diffusion_step_embed_dim=128,down_dims=(256,512,1024),kernel_size=5,n_groups=8,cond_predict_scale=True,gau_encoder=None,pre_fuse=True,skip_gaussian_encoder=True)
    norm=LinearNormalizer(); norm["head_cam_0"]=SingleFieldLinearNormalizer.create_identity(); norm["gaussian_0"]=SingleFieldLinearNormalizer.create_identity(); norm["state"]=_limits(stats["q_min"],stats["q_max"]); norm["action"]=_limits(stats["a_min"],stats["a_max"]); policy.set_normalizer(norm); policy=policy.to(device); policy.normalizer=policy.normalizer.to(device)
    for p in policy.normalizer.parameters(): p.data=p.data.to(device)
    return policy

def load_model(checkpoint,device="cuda:0",inference_steps=20):
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if payload.get("contract") != "mars-control.gaudp.shared_weights_decentralized_local_rgb_gaussian_qpos_to_absolute_action8": raise RuntimeError("checkpoint contract mismatch")
    policy=build_model(payload["stats"],device,inference_steps); policy.load_state_dict(payload["ema_model"],strict=True); policy=policy.to(device); policy.normalizer=policy.normalizer.to(device)
    for p in policy.normalizer.parameters(): p.data=p.data.to(device)
    policy.eval(); return policy,payload
