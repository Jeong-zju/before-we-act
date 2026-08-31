from __future__ import annotations
import argparse, json, time
from pathlib import Path
import gymnasium as gym
import numpy as np, torch
from rcs._core.sim import SimConfig
from rcs.envs.base import ControlMode, RelativeTo
from .dataset import TASKS
from .model import ACT
from .evaluate import policy_image, policy_state
from .action_target import canonicalize_controller_action

def make_env(task):
    module=__import__(f"duobench.tasks.{task}",fromlist=["*"])
    cls=getattr(module,"".join(x.title() for x in task.split("_"))+"EnvConfig")
    c=cls().config(); c.headless=True; c.control_mode=ControlMode.JOINTS; c.relative_to=RelativeTo.NONE
    c.sim_cfg=SimConfig(async_control=True,realtime=False,frequency=30); c.wrapper_cfg.binary_gripper=True
    return gym.make(f"duobench/{task}",cfg=c)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--data',type=Path,required=True); p.add_argument('--task',required=True,choices=TASKS); p.add_argument('--mode',choices=['first','open30','ensemble','first1','open30s1','ensembles1'],default='first'); p.add_argument('--episodes',type=int,default=3); p.add_argument('--episode-start',type=int,default=0); p.add_argument('--max-steps',type=int); p.add_argument('--output',type=Path,required=True); p.add_argument('--gym-box-clip',dest='clip',action='store_true',help='Reproduce the legacy faulty API-Box clipping'); p.add_argument('--no-clip',dest='clip',action='store_false',help=argparse.SUPPRESS); p.add_argument('--continuous-state-gripper',action='store_true',help='Reproduce the legacy train/deploy state mismatch'); p.set_defaults(clip=False); a=p.parse_args()
    s=torch.load(a.checkpoint,map_location='cpu',weights_only=False); dev=torch.device('cuda:0'); m=ACT(**{k:v for k,v in s['model_config'].items() if k!='vision_backbone'}).to(dev); m.load_state_dict(s['model']); m.eval(); n=s['normalization']; qm=np.asarray(n['qpos_mean'],np.float32); qs=np.asarray(n['qpos_std'],np.float32); am=np.asarray(n['action_mean'],np.float32); ass=np.asarray(n['action_std'],np.float32)
    manifest=json.loads((a.data/'manifest.json').read_text()); mx=a.max_steps or int(manifest['tasks'][a.task]['validation_max_steps']); tid=TASKS.index(a.task); env=make_env(a.task); rows=[]
    for ep in range(a.episode_start,a.episode_start+a.episodes):
      obs,_=env.reset(seed=20260820+tid*1000+ep); chunks=[[],[]]; starts=[-1,-1]; trace=[]; st=time.perf_counter(); success=False; maxprog=0; transitional_widths=0
      for step in range(mx):
        # Prompt on every step for first/ensemble; only at 30-step boundaries for open30.
        open_loop=a.mode in {'open30','open30s1'}
        shift=int(a.mode in {'first1','open30s1','ensembles1'})
        refresh=(not open_loop or step%30==0 or not chunks[0])
        if refresh:
          ims=[]; qps=[]
          for arm in (0,1):
            key=('left','right')[arm]
            width=np.asarray(obs[key]['gripper'],np.float32).reshape(-1)
            transitional_widths += int(0.0 < width[0] <= 0.9)
            q=(np.concatenate([np.asarray(obs[key]['joints'],np.float32),width]) if a.continuous_state_gripper else policy_state(obs,key))
            ims.append(policy_image(obs,arm)); qps.append((q-qm)/qs)
          with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16): pred,_,_=m(torch.stack(ims).to(dev).float().div_(255),torch.from_numpy(np.asarray(qps)).to(dev),torch.full((2,),tid,dtype=torch.long,device=dev))
          pred=pred.float().cpu().numpy()*ass+am
          for arm in (0,1):
            # Keep every prompt for temporal aggregation; open30 only uses the
            # current chunk until the next prompt.
            chunks[arm].append(pred[arm]); starts[arm]=step
        action={}
        for arm,key in enumerate(('left','right')):
          if a.mode in {'first','first1'}: local=chunks[arm][-1][shift]
          elif open_loop:
            current=chunks[arm][-1]; offset=step-starts[arm]+shift; local=current[offset] if offset<len(current) else current[-1]
          else:
            # Exact counterpart of evaluate.py: each prompt contributes one
            # horizon-100 chunk, and age-weighted predictions are averaged.
            candidates=[]
            for prompt,chunk in enumerate(chunks[arm]):
              offset=step-prompt+shift
              if 0 <= offset < len(chunk): candidates.append(chunk[offset])
            weights=np.exp(-0.01*np.arange(len(candidates)-1,-1,-1)); weights/=weights.sum()
            local=np.sum(np.asarray(candidates)*weights[:,None],axis=0)
          local=np.asarray(local).copy()
          if getattr(a,'clip',True):
            # Legacy switch retained only to reproduce the original faulty
            # Gym-Box baseline. New experiments pass --no-clip and use the
            # controller-equivalent saturation below.
            local[:7]=np.clip(local[:7],env.action_space[key]['joints'].low,env.action_space[key]['joints'].high)
          local=canonicalize_controller_action(local); action[key]={'joints':local[:7].astype(np.float32),'gripper':np.asarray([local[7]],np.float32)}; trace.append(local.astype(np.float32))
        obs,reward,term,trunc,info=env.step(action); maxprog=max(maxprog,float(reward)); success=bool(info.get('success',False))
        if success or bool(np.asarray(term).all()) or bool(np.asarray(trunc).all()): break
      rows.append({'task':a.task,'mode':a.mode,'seed':20260820+tid*1000+ep,'success':success,'steps':step+1,'max_stage_progress':maxprog,'state_gripper_encoding':('legacy_continuous_physical_width' if a.continuous_state_gripper else 'physical_width_gt_0.9_to_binary'),'gym_box_clip':bool(a.clip),'transitional_gripper_observations':transitional_widths,'seconds':time.perf_counter()-st}); print(json.dumps(rows[-1]),flush=True)
    env.close(); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(rows,indent=2))
if __name__=='__main__': main()
