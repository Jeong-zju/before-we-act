#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,pickle,socket,struct
import gymnasium as gym,numpy as np,torch
os.chdir('/workspace/repos/RoboFactory')
import tasks
from deployment.pi05_mars.common import ARMS,CONTRACT,ENVS,HIGH,LOW,PROMPTS,atomic_json
def ex(c,n):
 b=[]
 while n:
  x=c.recv(n)
  if not x: raise EOFError
  b.append(x); n-=len(x)
 return b''.join(b)
def rpc(path,r):
 q=pickle.dumps(r,protocol=5)
 with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as c:
  c.settimeout(900); c.connect(path); c.sendall(struct.pack('!Q',len(q))+q); out=pickle.loads(ex(c,struct.unpack('!Q',ex(c,8))[0]))
 if not out.get('ok'): raise RuntimeError(out.get('error'))
 return out
def scalar(x):
 if torch.is_tensor(x): return bool(x.detach().cpu().reshape(-1)[0].item())
 return bool(np.asarray(x).reshape(-1)[0])
def main():
 p=argparse.ArgumentParser(); p.add_argument('--socket',required=True); p.add_argument('--task',choices=ARMS,required=True); p.add_argument('--output',required=True); p.add_argument('--episodes',type=int,default=20); p.add_argument('--max-steps',type=int); p.add_argument('--smoke',action='store_true'); a=p.parse_args(); env_id,formal_max,seed0=ENVS[a.task]; max_steps=a.max_steps or formal_max; rows=[]
 for ep in range(a.episodes):
  env=None; row={'episode':ep,'seed':seed0+ep,'success':False,'steps':0}
  try:
   env=gym.make(env_id,config=f'/workspace/repos/RoboFactory/configs/table/{a.task}.yaml',obs_mode='rgb',control_mode='pd_joint_pos',render_mode='rgb_array',reward_mode='dense',num_envs=1,sim_backend='cpu',render_backend='cpu',sensor_configs={'shader_pack':'minimal','width':320,'height':240},human_render_camera_configs={'shader_pack':'minimal'},viewer_camera_configs={'shader_pack':'minimal'})
   obs,_=env.reset(seed=seed0+ep); rpc(a.socket,{'op':'reset'}); histories=[[] for _ in range(ARMS[a.task])]
   for step in range(max_steps):
    chunks=[]
    for arm in range(ARMS[a.task]):
     im=np.asarray(obs['sensor_data'][f'head_camera_agent{arm}']['rgb']); im=im[0] if im.ndim==4 else im; q=np.asarray(obs['agent'][f'panda-{arm}']['qpos']); q=q[0] if q.ndim==2 else q
     chunks.append(rpc(a.socket,{'op':'infer','observation':{'image':im.astype(np.uint8,copy=False),'state':q[:9].astype(np.float32,copy=False),'prompt':PROMPTS[a.task]}})['chunk'])
    action={}
    for arm,ch in enumerate(chunks):
     histories[arm].append((step,np.asarray(ch,np.float32))); histories[arm][:]=[(born,x) for born,x in histories[arm] if step-born<len(x)]; cand=np.asarray([x[step-born] for born,x in histories[arm]]); w=np.exp(-.01*np.arange(len(cand)-1,-1,-1)); w/=w.sum(); action[f'panda-{arm}']=np.clip(np.sum(cand*w[:,None],axis=0),LOW,HIGH).astype(np.float32)
    obs,_,term,trunc,info=env.step(action); row['steps']=step+1; row['success']=scalar(info.get('success',False))
    if row['success'] or scalar(term) or scalar(trunc): break
  except Exception as e: row['error']=f'{type(e).__name__}: {e}'
  finally:
   if env is not None: env.close()
  rows.append(row); report={'schema':'mars-control.pi05.smoke.v1' if a.smoke else 'mars-control.pi05.validation20.task.v1','status':'failed' if any('error' in x for x in rows) else ('complete' if len(rows)==a.episodes else 'running'),'task':a.task,'episodes':len(rows),'target_episodes':a.episodes,'successes':sum(int(x['success']) for x in rows),'max_steps':max_steps,'seed_start':seed0,'rows':rows,'policy_contract':CONTRACT,'action_decode':'OpenPI inverse quantile and inverse DeltaActions exactly once, then environment bounds','image_contract':'uint8 HWC to OpenPI model preprocessing','temporal_ensemble_decay':.01}; atomic_json(a.output,report); print(json.dumps(row),flush=True)
 if any('error' in x for x in rows): raise RuntimeError('validation errors')
if __name__=='__main__': main()
