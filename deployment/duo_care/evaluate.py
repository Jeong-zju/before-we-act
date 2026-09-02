from __future__ import annotations
import argparse,json,time,hashlib
from collections import deque
from pathlib import Path
import cv2, numpy as np, torch, gymnasium as gym
from rcs._core.sim import SimConfig
from rcs.envs.base import ControlMode, RelativeTo
from deployment.mars_care.model import CAREPolicy,ModelConfig
from .action_contract import current_relative_temporal_ensemble
from .prepare import TASKS

def env_for(task):
    mod=__import__(f'duobench.tasks.{task}',fromlist=['*']); cfg=getattr(mod,''.join(x.title() for x in task.split('_'))+'EnvConfig')().config(); cfg.headless=True; cfg.control_mode=ControlMode.JOINTS; cfg.relative_to=RelativeTo.NONE; cfg.sim_cfg=SimConfig(async_control=True,realtime=False,frequency=30); cfg.wrapper_cfg.binary_gripper=True
    for camera in cfg.camera_cfgs.values(): camera.resolution_width=224; camera.resolution_height=224
    return gym.make('duobench/'+task,cfg=cfg)
def image(obs,arm,size):
    head=cv2.resize(np.asarray(obs['frames']['head']['rgb']['data'],np.uint8),(size,size),interpolation=cv2.INTER_LINEAR); wrist=cv2.resize(np.asarray(obs['frames']['left_wrist' if arm==0 else 'right_wrist']['rgb']['data'],np.uint8),(size,size),interpolation=cv2.INTER_LINEAR); a=np.concatenate([head,wrist],1); return torch.from_numpy(a.copy()).permute(2,0,1).float().div_(255)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--episodes',type=int,default=20); p.add_argument('--seed-start',type=int,default=20260820); p.add_argument('--device',default='cuda:0'); p.add_argument('--task',choices=TASKS); p.add_argument('--max-steps',type=int); p.add_argument('--smoke',action='store_true'); p.add_argument('--candidate-zero',action='store_true',help='freeze the reference generator to candidate zero'); a=p.parse_args(); dev=torch.device(a.device)
 saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False); cfg=ModelConfig(**saved['model_config']); m=CAREPolicy(cfg).to(dev); m.load_state_dict(saved['model']); m.eval(); n=saved['normalization']; qm=np.asarray(n['qpos_mean'],np.float32); qs=np.asarray(n['qpos_std'],np.float32); am=np.asarray(n['action_mean'],np.float32); ass=np.asarray(n['action_std'],np.float32); manifest=json.loads((a.data/'manifest.json').read_text())
 rows=[]; a.output.parent.mkdir(parents=True,exist_ok=True); recovered={}
 if a.output.with_suffix('.jsonl').is_file():
  for line in a.output.with_suffix('.jsonl').read_text().splitlines():
   try:
    row=json.loads(line); recovered[(row['task'],int(row['seed']))]=row
   except Exception: pass
 selected=(a.task,) if a.task else TASKS
 for task in selected:
  tid=TASKS.index(task); env=env_for(task); max_steps=a.max_steps or int(manifest['tasks'][task]['validation_max_steps'])
  for ep in range(a.episodes):
   seed=a.seed_start+tid*1000+ep
   if (task,seed) in recovered: rows.append(recovered[(task,seed)]); continue
   obs,info=env.reset(seed=seed); hist=[deque(maxlen=cfg.history),deque(maxlen=cfg.history)]; prev=[None,None]; chunks=[]; trace=hashlib.sha256(); success=False; stages=[]; t0=time.perf_counter()
   for step in range(max_steps):
    imgs=[]; qn=[]
    for arm,key in enumerate(('left','right')):
      raw=np.concatenate([np.asarray(obs[key]['joints'],np.float32),np.asarray(obs[key]['gripper'],np.float32).reshape(-1)])
      imgs.append(image(obs,arm,cfg.image_size)); qn.append((raw-qm)/qs); tok=np.zeros(cfg.qpos_dim+cfg.action_dim,np.float32); tok[:8]=(raw-qm)/qs
      if prev[arm] is not None: tok[8:]=(prev[arm]-am)/ass
      hist[arm].append(tok)
    h=np.zeros((2,cfg.history,cfg.qpos_dim+cfg.action_dim),np.float32); mask=np.zeros((2,cfg.history),np.float32)
    for j in range(2):
      values=np.asarray(hist[j],np.float32); h[j,-len(values):]=values; mask[j,-len(values):]=1
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
     if a.candidate_zero:
      model_out=m(torch.stack(imgs).to(dev),torch.from_numpy(np.asarray(qn)).to(dev),torch.full((2,),tid,dtype=torch.long,device=dev),torch.from_numpy(h).to(dev),torch.from_numpy(mask).to(dev)); chunks_t=model_out['candidates'][:,0]
     else: chunks_t,_=m.act(torch.stack(imgs).to(dev),torch.from_numpy(np.asarray(qn)).to(dev),torch.full((2,),tid,dtype=torch.long,device=dev),torch.from_numpy(h).to(dev),torch.from_numpy(mask).to(dev))
    pred=chunks_t.float().cpu().numpy()*ass+am; current_q=np.asarray([np.concatenate([np.asarray(obs[key]['joints'],np.float32),np.asarray(obs[key]['gripper'],np.float32).reshape(-1)]) for key in ('left','right')],np.float32); chunks.append((step,pred,current_q.copy())); chunks=[x for x in chunks if step-x[0]<cfg.horizon]; encoded=current_relative_temporal_ensemble(chunks,step,current_q); action={}
    for arm,key in enumerate(('left','right')):
      local=encoded[arm].copy(); local[:7]+=np.asarray(obs[key]['joints'],np.float32); local[:7]=np.clip(local[:7],env.action_space[key]['joints'].low,env.action_space[key]['joints'].high); local[7]=float(np.clip(local[7],0,1)); action[key]={'joints':local[:7], 'gripper':np.asarray([local[7]],np.float32)}; prev[arm]=encoded[arm]; trace.update(local.tobytes())
    obs,reward,term,trunc,info=env.step(action); stages.append(float(reward)); success=bool(info.get('success',False));
    if success or bool(np.asarray(term).all()) or bool(np.asarray(trunc).all()): break
   rows.append({'task':task,'seed':seed,'success':success,'steps':step+1,'max_steps':max_steps,'final_stage_progress':float(stages[-1] if stages else 0),'max_stage_progress':float(max(stages) if stages else 0),'action_trace_sha256':trace.hexdigest(),'wall_seconds':time.perf_counter()-t0}); a.output.with_suffix('.jsonl').open('a').write(json.dumps(rows[-1])+'\n'); print(json.dumps(rows[-1]),flush=True)
  env.close()
 result={'status':'complete','schema':'duobench-care-validation20-v1','episodes_per_task':a.episodes,'total_episodes':len(rows),'successes':sum(int(x['success']) for x in rows),'macro_success_rate':float(np.mean([np.mean([x['success'] for x in rows if x['task']==t]) for t in selected])),'tasks':list(selected),'rows':rows,'policy_contract':saved['policy_contract'],'action_encoding':saved['action_encoding']}; a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
