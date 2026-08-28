#!/usr/bin/env python3
import glob,json,os
from pathlib import Path
import h5py,numpy as np
TASKS={'place_cube_in_cup':2,'strike_cube_hard':2,'three_robots_place_shoes':3,'four_robots_stack_cube':4}
root=Path(os.environ.get('MARS_ACT_DATA_ROOT','/workspace/datasets/mars_control')); report={'schema':'mars-control.act.audit.v1','status':'complete','tasks':{},'episodes':0,'local_streams':0,'policy_inputs':['head_camera_agent{i}/rgb','panda-{i}/qpos'],'policy_outputs':['panda-{i}/action8'],'forbidden_inputs':['global_rgb','peer_rgb','peer_qpos','joint_action']}
for task,arms in TASKS.items():
 paths=sorted(glob.glob(str(root/task/'motionplanning'/'*.shard*.h5'))); eps=succ=steps=0
 if len(paths)!=10: raise RuntimeError(f'{task}: {len(paths)} shards')
 for path in paths:
  with h5py.File(path,'r') as f:
   for key in f:
    if not key.startswith('traj_'): continue
    g=f[key]; eps+=1; succ+=int(bool(np.asarray(g['success'])[-1])); n=len(g['actions/panda-0']); steps+=n
    for arm in range(arms):
     if g[f'actions/panda-{arm}'].shape[-1]!=8 or g[f'obs/agent/panda-{arm}/qpos'].shape[-1]!=9 or g[f'obs/sensor_data/head_camera_agent{arm}/rgb'].shape[-1]!=3: raise RuntimeError(f'{task}:{key}:arm{arm} contract')
 if eps!=150 or succ!=150: raise RuntimeError(f'{task}: episodes/success {eps}/{succ}')
 report['tasks'][task]={'episodes':eps,'successful':succ,'arms':arms,'joint_steps':steps,'local_streams':eps*arms}; report['episodes']+=eps; report['local_streams']+=eps*arms
out=Path('/workspace/runs/mars_act/audit.json'); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report))
