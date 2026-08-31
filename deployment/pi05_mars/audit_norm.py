#!/usr/bin/env python3
import glob,json
from pathlib import Path
import h5py,numpy as np
from .common import TASKS,ARMS,LOW,HIGH,CONTRACT,atomic_json
root=Path("/workspace/datasets/mars_control"); states=[]; actions=[]; episodes=streams=timesteps=0; horizon=16; low=np.asarray(LOW,np.float32); high=np.asarray(HIGH,np.float32)
task_rows={}
for task in TASKS:
 te=ts=tt=0; paths=sorted(glob.glob(str(root/task/'motionplanning'/f'{task}.shard*.h5')))
 if len(paths)!=10: raise RuntimeError(f'{task}: shard count {len(paths)}')
 for path in paths:
  with h5py.File(path,'r') as f:
   for tr in sorted(f,key=lambda x:int(x.rsplit('_',1)[-1])):
    g=f[tr]
    if not bool(np.asarray(g['success']).reshape(-1)[-1]): raise RuntimeError(f'non-success {path}:{tr}')
    n=min(len(g[f'actions/panda-{a}']) for a in range(ARMS[task])); te+=1; episodes+=1
    for arm in range(ARMS[task]):
     q=np.asarray(g[f'obs/agent/panda-{arm}/qpos'][:n],np.float32); a=np.clip(np.asarray(g[f'actions/panda-{arm}'][:n],np.float32),low,high); idx=np.minimum(np.arange(n)[:,None]+np.arange(horizon)[None,:],n-1); chunk=a[idx]; chunk[...,:7]-=q[:,None,:7]
     states.append(q); actions.append(chunk.reshape(-1,8)); streams+=1; ts+=1; timesteps+=n; tt+=n
 if te!=150: raise RuntimeError(f'{task}: episodes {te}')
 task_rows[task]={"episodes":te,"local_streams":ts,"indexed_local_timesteps":tt,"arms":ARMS[task]}
if (episodes,streams,timesteps)!=(600,1650,1035318): raise RuntimeError(f'cardinality drift {(episodes,streams,timesteps)}')
s=np.concatenate(states); a=np.concatenate(actions)
def stats(x):
 q=np.quantile(x,[.01,.99],axis=0); return {"mean":x.mean(0).astype(float).tolist(),"std":x.std(0).astype(float).tolist(),"q01":q[0].astype(float).tolist(),"q99":q[1].astype(float).tolist()}
norm={"norm_stats":{"state":stats(s),"actions":stats(a)}}; out=Path('/workspace/runs/pi05_mars/assets/pi05_mars_control_lora/mars_control/norm_stats.json'); atomic_json(out,norm)
atomic_json('/workspace/runs/pi05_mars/audit.json',{"schema":"mars-control.pi05.audit.v1","status":"complete","episodes":episodes,"local_streams":streams,"indexed_local_timesteps":timesteps,"tasks":task_rows,"all_data_no_split":True,"policy_contract":CONTRACT,"forbidden_inputs":["peer_rgb","peer_qpos","global_rgb","joint_action","arm_id"],"state":"own qpos9","action":"own absolute pd_joint_pos action8; seven arm joints converted to delta for model normalization","action_bounds":{"low":list(LOW),"high":list(HIGH),"clipped_before_statistics":True},"normalization":"all-corpus 1st/99th quantiles after exact train action transform","image":{"source":"own head RGB","dtype":"uint8","shape":[240,320,3],"model_resize":[224,224]}})
print(json.dumps({"status":"complete","episodes":episodes,"streams":streams,"timesteps":timesteps,"norm":str(out)}))
