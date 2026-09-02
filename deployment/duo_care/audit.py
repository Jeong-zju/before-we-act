from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from .prepare import TASKS,JOINT_LOW,JOINT_HIGH

def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); m=json.loads((a.data/'manifest.json').read_text()); checks={'eleven_tasks':tuple(m['tasks'])==TASKS,'all_550_episodes':m['total_episodes']==550,'dimensions':True,'episode_contiguous':True,'finite_nonzero_normalization':True,'action_runtime_range':True,'binary_gripper':True,'image_contract':True}; details={}
 n=m['normalization']
 for key in ('qpos_mean','qpos_std','action_mean','action_std'): checks['finite_nonzero_normalization'] &= bool(np.isfinite(n[key]).all())
 checks['finite_nonzero_normalization'] &= bool(np.min(n['qpos_std'])>=1e-4 and np.min(n['action_std'])>=1e-4 and n['action_encoding']=='anchor_joint_residual_gripper_absolute' and int(n.get('action_chunk_horizon',-1))==16)
 for task in TASKS:
  arrays={x:np.load(a.data/task/f'{x}.npy',mmap_mode='r') for x in ('state','action','head','left','right','episodes')}; count=len(arrays['state']); checks['dimensions'] &= arrays['state'].shape==(count,16) and arrays['action'].shape==(count,16) and all(len(arrays[x])==count for x in ('head','left','right','episodes')); unique,first=np.unique(arrays['episodes'],return_index=True); checks['episode_contiguous'] &= len(unique)==50 and bool(np.all(np.diff(first)>0)); actions=arrays['action'].reshape(-1,2,8); checks['action_runtime_range'] &= bool(np.all(actions[:,:,:7]>=JOINT_LOW-1e-5) and np.all(actions[:,:,:7]<=JOINT_HIGH+1e-5)); checks['binary_gripper'] &= bool(np.isin(actions[:,:,7],(0,1)).all()); checks['image_contract'] &= all(arrays[x].dtype==np.uint8 and arrays[x].shape[1:]==(224,224,3) for x in ('head','left','right')); details[task]={'frames':count,'episodes':len(unique),'validation_max_steps':m['tasks'][task]['validation_max_steps']}
 report={'schema':'duobench-care-audit-v2','passed':all(checks.values()),'checks':checks,'tasks':details}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report)); raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__': main()
