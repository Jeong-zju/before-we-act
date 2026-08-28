from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from collections import deque
from pathlib import Path
import gymnasium as gym, numpy as np, torch
import torch.nn.functional as F
from .common import ENVS, ARMS, atomic_json
from .model import load_model
from .precompute import load_encoder

REVISION="mars-gaudp-local-noposplat-self-obs3-horizon8-exec6-v1"
def scalar(x): return bool(np.asarray(x).reshape(-1)[0])
def local_obs(obs,arm):
    im=np.asarray(obs["sensor_data"][f"head_camera_agent{arm}"]["rgb"]); im=im[0] if im.ndim==4 else im
    q=np.asarray(obs["agent"][f"panda-{arm}"]["qpos"]); q=q[0] if q.ndim==2 else q
    if im.shape!=(240,320,3) or im.dtype!=np.uint8 or q.shape!=(9,): raise RuntimeError(f"observation contract drift {im.shape}/{im.dtype}/{q.shape}")
    return im,q.astype(np.float32)

@torch.no_grad()
def episode(policy,gaussian,task,rf_root,seed,device,max_steps):
    os.chdir(rf_root); sys.path.insert(0,str(rf_root)); import tasks  # noqa
    env_id,cfg_name,_,_=ENVS[task]; cfg=str(Path(rf_root)/"configs/table"/cfg_name)
    env=gym.make(env_id,config=cfg,obs_mode="rgb",control_mode="pd_joint_pos",render_mode="sensors",reward_mode="dense",sim_backend="cpu",sensor_configs={"shader_pack":"default"},human_render_camera_configs={"shader_pack":"default"},viewer_camera_configs={"shader_pack":"default"}); obs,_=env.reset(seed=int(seed)); torch.manual_seed(int(seed)); arms=range(ARMS[task]); hist=[deque(maxlen=3) for _ in arms]; pending=[deque() for _ in arms]; trace=hashlib.sha256(); times=[]; success=False
    try:
        for step in range(max_steps):
            current=[local_obs(obs,a) for a in arms]; x=torch.from_numpy(np.stack([r[0] for r in current])).permute(0,3,1,2).float().div(255).to(device); large=F.interpolate(x,size=(256,256),mode="bilinear",align_corners=False)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16): gau=gaussian({"image":large.mul(2).sub(1)[:,None]})[:,0].float()
            gau=F.interpolate(gau,size=(120,160),mode="bilinear",align_corners=False); small=F.interpolate(x,size=(120,160),mode="bilinear",align_corners=False)
            for arm,(im,q) in enumerate(zip(small, [r[1] for r in current])):
                row=(im,gau[arm],torch.from_numpy(q).to(device));
                if not hist[arm]: hist[arm].extend([row,row])
                hist[arm].append(row)
            if step%6==0 or any(not q for q in pending):
                torch.manual_seed(int(seed)+step//6); torch.cuda.manual_seed_all(int(seed)+step//6); started=time.perf_counter(); batch={"head_cam_0":torch.stack([torch.stack([r[0] for r in h]) for h in hist]),"gaussian_0":torch.stack([torch.stack([r[1] for r in h]) for h in hist]),"state":torch.stack([torch.stack([r[2] for r in h]) for h in hist])}; chunks=policy.predict_action(batch)["action"].float().cpu().numpy(); times.append(time.perf_counter()-started)
                for arm,row in enumerate(chunks): pending[arm]=deque(row[:6])
            actions={}
            for arm in arms:
                a=np.asarray(pending[arm].popleft(),np.float32); space=env.action_space.spaces[f"panda-{arm}"]; a=np.clip(a,space.low,space.high).astype(np.float32); trace.update(a.tobytes()); actions[f"panda-{arm}"]=a
            obs,_,terminated,truncated,info=env.step(actions); success=scalar(info.get("success",False))
            if success or scalar(terminated) or scalar(truncated): break
    finally: env.close()
    return {"seed":int(seed),"success":bool(success),"steps":step+1,"mean_inference_seconds":float(np.mean(times)) if times else None,"action_trace_sha256":trace.hexdigest(),"evaluator_revision":REVISION}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--noposplat-weight",required=True); p.add_argument("--task",choices=ENVS,required=True); p.add_argument("--robofactory-root",required=True); p.add_argument("--output",required=True); p.add_argument("--episodes",type=int,default=20); p.add_argument("--seed-start",type=int,required=True); p.add_argument("--max-steps",type=int,required=True); p.add_argument("--smoke",action="store_true"); a=p.parse_args(); device=torch.device("cuda:0"); policy,_=load_model(a.checkpoint,device,20); gaussian=load_encoder(Path(a.noposplat_weight),device); out=Path(a.output); rows=[]
    for i in range(a.episodes): row=episode(policy,gaussian,a.task,a.robofactory_root,a.seed_start+i,device,a.max_steps); rows.append(row); print(json.dumps(row),flush=True)
    result={"schema":"mars-control.gaudp.smoke.v1" if a.smoke else "mars-control.gaudp.validation20.task.v1","status":"complete","task":a.task,"episodes":len(rows),"successes":sum(int(x["success"]) for x in rows),"success_rate":sum(int(x["success"]) for x in rows)/len(rows),"rows":rows,"checkpoint":a.checkpoint,"checkpoint_sha256":hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),"evaluator_revision":REVISION,"rgb_preprocessing":"uint8_to_unit_float_then_imagenet_in_policy","state_action_codec":"corpus_minmax_to_minus1_plus1_and_inverse_once","policy_contract":"shared_weights_decentralized_local_rgb_gaussian_qpos_to_absolute_action8","max_steps":a.max_steps,"replan_interval":6}; atomic_json(out,result)
if __name__=="__main__": main()
