from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,torch
from .prepare import TASKS
from .evaluate import env_for,image

def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); m=json.loads((a.data/'manifest.json').read_text()); n=m['normalization']; qm,qs=np.asarray(n['qpos_mean']),np.asarray(n['qpos_std']); am,ass=np.asarray(n['action_mean']),np.asarray(n['action_std']); checks={'four_5090s':torch.cuda.device_count()==4 and all('5090' in torch.cuda.get_device_name(i) for i in range(4)),'normalization_roundtrip':True,'reset_normalized_finite':True,'camera_contract':True,'absolute_joint_binary_gripper_step':True,'task_specific_max_steps':True}; rows={}
 for tid,task in enumerate(TASKS):
  env=env_for(task); obs,_=env.reset(seed=20260820+tid*1000); action={}
  for arm,key in enumerate(('left','right')):
   raw=np.concatenate([np.asarray(obs[key]['joints'],np.float32),np.asarray(obs[key]['gripper'],np.float32).reshape(-1)]); checks['normalization_roundtrip'] &= bool(np.allclose((raw-qm)/qs*qs+qm,raw,atol=1e-6)); checks['reset_normalized_finite'] &= bool(np.isfinite((raw-qm)/qs).all()); frame=image(obs,arm,224); checks['camera_contract'] &= tuple(frame.shape)==(3,224,448) and bool(torch.isfinite(frame).all()) and 0<=float(frame.min())<=float(frame.max())<=1; low=env.action_space[key]['joints'].low; high=env.action_space[key]['joints'].high; action[key]={'joints':np.clip(raw[:7],low,high).astype(np.float32),'gripper':np.asarray([float(raw[7]>=.5)],np.float32)}
  _,_,term,trunc,_=env.step(action); checks['absolute_joint_binary_gripper_step'] &= isinstance(bool(np.asarray(term).all() or np.asarray(trunc).all()),bool); checks['task_specific_max_steps'] &= int(m['tasks'][task]['validation_max_steps'])>0; rows[task]={'validation_max_steps':int(m['tasks'][task]['validation_max_steps'])}; env.close()
 report={'schema':'duobench-care-preflight-v2','passed':all(checks.values()),'checks':checks,'tasks':rows}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report)); raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__': main()
