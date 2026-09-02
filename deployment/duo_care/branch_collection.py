"""Deterministic, strict-local CARE branch-family collection for DuoBench."""
from __future__ import annotations
import argparse, hashlib, json
from collections import deque
from pathlib import Path
from typing import Any
import cv2, numpy as np, torch
from deployment.mars_care.model import CAREPolicy, ModelConfig
from .evaluate import env_for
from .prepare import TASKS, JOINT_LOW, JOINT_HIGH

HORIZONS=(8,16,32,64); CANDIDATES=6; REPEATS=(0,1); HISTORY=16; ACTION_HORIZON=100

def _load(path:Path,device):
    saved=torch.load(path,map_location='cpu',weights_only=False); cfg=ModelConfig(**saved['model_config']); model=CAREPolicy(cfg).to(device); model.load_state_dict(saved['model'],strict=True); model.eval(); n=saved['normalization']
    return model,cfg,tuple(np.asarray(n[k],np.float32) for k in ('qpos_mean','qpos_std','action_mean','action_std'))
def _q(obs,arm):
    key=('left','right')[arm]; return np.concatenate((np.asarray(obs[key]['joints'],np.float32),np.asarray(obs[key]['gripper'],np.float32).reshape(-1)))
def _image(obs,arm,size):
    wrist=('left_wrist','right_wrist')[arm]; head=cv2.resize(np.asarray(obs['frames']['head']['rgb']['data'],np.uint8),(size,size)); wi=cv2.resize(np.asarray(obs['frames'][wrist]['rgb']['data'],np.uint8),(size,size)); return torch.from_numpy(np.concatenate((head,wi),1).copy()).permute(2,0,1).float().div_(255)
def _action(env,obs,encoded):
    out={}
    for arm,key in enumerate(('left','right')):
        x=np.asarray(encoded[arm],np.float32).copy(); x[:7]=np.clip(_q(obs,arm)[:7]+x[:7],JOINT_LOW,JOINT_HIGH); x[7]=float(x[7]>=.5); out[key]={'joints':x[:7].astype(np.float32),'gripper':np.asarray([x[7]],np.float32)}
    return out
def _step(model,cfg,stats,obs,hist,prev,tid,device):
    qm,qs,am,ass=stats; images=[]; qn=[]
    for arm in range(2):
        q=_q(obs,arm); tok=np.concatenate(((q-qm)/qs,np.zeros(8,np.float32) if prev[arm] is None else (prev[arm]-am)/ass)).astype(np.float32); hist[arm].append(tok); images.append(_image(obs,arm,cfg.image_size)); qn.append((q-qm)/qs)
    h=np.zeros((2,cfg.history,cfg.qpos_dim+cfg.action_dim),np.float32); mask=np.zeros((2,cfg.history),np.float32)
    for arm in range(2):
        vals=np.asarray(hist[arm],np.float32); h[arm,-len(vals):]=vals; mask[arm,-len(vals):]=1
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
        out=model(torch.stack(images).to(device),torch.from_numpy(np.asarray(qn)).to(device),torch.full((2,),tid,dtype=torch.long,device=device),torch.from_numpy(h).to(device),torch.from_numpy(mask).to(device))
    return (out['candidates'][:,0].float().cpu().numpy()*ass+am).astype(np.float32), np.asarray(hist[0],np.float32), h, mask
def _delta(candidate,step,ass):
    d=np.zeros(8,np.float32)
    if candidate and step<16:
        phase=(candidate-3)/3.; d[:7]=phase*.12*ass[:7]*np.sin((step+1)/4.)
        if candidate in (2,5): d[7]=1. if candidate==5 else -1.
    return d
def _utility(rows,h,start,success,unsafe):
    p=float(rows[min(h,len(rows))-1]['progress']) if rows else start; r=float(np.mean([x['reward'] for x in rows[:h]])) if rows else 0.
    v=np.clip(np.asarray((p-start,r,p,float(success),-float(unsafe),0,0,0),np.float32),-1,1)
    return {'bounded_utility_vector':v.tolist(),'hard_safety_violation':bool(unsafe),'progress':p}

def _rollout(task,seed,anchor,focal,candidate,regime,repeat,model,cfg,stats,device,teammate_trace=None):
    env=env_for(task); obs,info=env.reset(seed=int(seed)); hist=[deque(maxlen=cfg.history),deque(maxlen=cfg.history)]; prev=[None,None]
    for _ in range(anchor):
        ref,_m,_h,_mask=_step(model,cfg,stats,obs,hist,prev,TASKS.index(task),device); encoded=ref[:,0]; obs,_r,term,trunc,info=env.step(_action(env,obs,encoded)); prev=[encoded[0].copy(),encoded[1].copy()]
        if bool(info.get('success',False)) or bool(np.asarray(term).all()) or bool(np.asarray(trunc).all()): env.close(); raise RuntimeError(f'terminated before anchor: {task}:{seed}')
    start=float(info.get('stage',0))/max(float(info.get('max_stage',1)),1); rows=[]; executed=[]; trace=hashlib.sha256(); success=False; unsafe=False
    for branch_step in range(max(HORIZONS)):
        ref,_m,_h,_mask=_step(model,cfg,stats,obs,hist,prev,TASKS.index(task),device); encoded=ref[:,0].copy()
        teammate=1-focal
        if regime=='replay' and teammate_trace is not None and branch_step<len(teammate_trace): encoded[teammate]=teammate_trace[branch_step][teammate]
        encoded[focal]+=_delta(candidate,branch_step,stats[3]); unsafe|=not bool(np.isfinite(encoded).all())
        obs,reward,term,trunc,info=env.step(_action(env,obs,encoded)); prev=[encoded[0].copy(),encoded[1].copy()]; executed.append(encoded.copy()); trace.update(encoded.astype(np.float32).tobytes())
        progress=float(info.get('stage',0))/max(float(info.get('max_stage',1)),1); rows.append({'reward':float(np.asarray(reward).mean()),'progress':progress}); success=bool(info.get('success',False))
        if success or bool(np.asarray(term).all()) or bool(np.asarray(trunc).all()): break
    env.close(); outcomes={str(h):_utility(rows,h,start,success,unsafe) for h in HORIZONS}
    return {'candidate_id':candidate,'regime':regime,'repeat_id':repeat,'branch_seed':int(seed),'status':'SUCCESS' if success else 'VALID','candidate_valid':True,'steps':len(executed),'outcomes':outcomes,'success':success,'action_trace_sha256':trace.hexdigest()},executed

def collect_family(task,seed,anchor,focal,model,cfg,stats,device):
    # Reference trace defines candidate zero and the teammate replay branch.
    reference_row,reference_trace=_rollout(task,seed,anchor,focal,0,'reactive',0,model,cfg,stats,device)
    # Memory is constructed from only the focal arm's normalized qpos/previous-action history.
    env=env_for(task); obs,_=env.reset(seed=int(seed)); hist=[deque(maxlen=cfg.history),deque(maxlen=cfg.history)]; prev=[None,None]
    for _ in range(anchor+1):
        ref,local_memory,_h,_mask=_step(model,cfg,stats,obs,hist,prev,TASKS.index(task),device); encoded=ref[:,0]; obs,_r,_t,_tr,_i=env.step(_action(env,obs,encoded)); prev=[encoded[0].copy(),encoded[1].copy()]
    env.close(); rng=np.random.default_rng(731947); projection=rng.normal(0,.08,(cfg.qpos_dim+cfg.action_dim,384)).astype(np.float32); memory=np.zeros((HISTORY,384),np.float32); memory_mask=np.zeros(HISTORY,bool)
    vals=np.asarray(hist[focal],np.float32)[-HISTORY:]; memory[-len(vals):]=vals@projection; memory_mask[-len(vals):]=True
    branches=[]; chunks=[]
    for repeat in REPEATS:
        for candidate in range(CANDIDATES):
            if candidate==0 and repeat==0: reactive,executed=reference_row,reference_trace
            else: reactive,executed=_rollout(task,seed,anchor,focal,candidate,'reactive',repeat,model,cfg,stats,device)
            replay,_=_rollout(task,seed,anchor,focal,candidate,'replay',repeat,model,cfg,stats,device,reference_trace)
            branches.extend((reactive,replay))
            if repeat==0:
                local=np.asarray(executed,np.float32)[:,focal] if executed else np.zeros((1,8),np.float32); local=np.pad(local[:ACTION_HORIZON],((0,max(0,ACTION_HORIZON-len(local))),(0,0)),mode='edge'); chunks.append(local[:ACTION_HORIZON])
    family={'format_version':'before-we-act.care-duobench-branch-family/1','snapshot_id':f'{task}-{seed}-{anchor}-arm{focal}','task':task,'episode_seed':int(seed),'anchor_step':int(anchor),'focal_agent':int(focal),'branch_count':len(branches),'branches':branches,'candidate_legality':[{'candidate_id':i,'valid':True} for i in range(CANDIDATES)]}
    return family,{'memory':memory,'memory_mask':memory_mask,'candidate_chunks':np.stack(chunks).astype(np.float32)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--families-per-task',type=int,default=30); p.add_argument('--seed-start',type=int,default=20261001); p.add_argument('--task',action='append',choices=TASKS); p.add_argument('--device',default='cuda:0'); a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True); device=torch.device(a.device); model,cfg,stats=_load(a.checkpoint,device); manifest_path=a.output/'manifest.json'; manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'format_version':'before-we-act.care-duobench-family-manifest/1','tasks':list(TASKS),'branches_per_family':24,'families':[]}; known={x['snapshot_id'] for x in manifest['families']}
    for task in tuple(a.task or TASKS):
        for ordinal in range(a.families_per_task):
            seed=a.seed_start+TASKS.index(task)*10000+ordinal; anchor=ordinal; focal=ordinal%2; sid=f'{task}-{seed}-{anchor}-arm{focal}'; path=a.output/task/f'{sid}.json'; npz=path.with_suffix('.npz')
            if sid in known and path.exists() and npz.exists(): continue
            family,arrays=collect_family(task,seed,anchor,focal,model,cfg,stats,device); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(family,indent=2)+'\n'); np.savez_compressed(npz,**arrays); manifest['families'].append({'snapshot_id':sid,'task':task,'path':str(path),'npz':str(npz),'episode_seed':seed,'anchor_step':anchor,'focal_agent':focal}); known.add(sid); manifest_path.write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps({'event':'family','task':task,'ordinal':ordinal,'focal_agent':focal}),flush=True)
    manifest['family_count']=len(manifest['families']); manifest['status']='COMPLETE'; manifest_path.write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps({'status':'complete','families':len(manifest['families'])}))
if __name__=='__main__': main()
