#!/usr/bin/env python3
"""Strict MARS-Control Validation20 for the shared local ManiFlow policy."""
from __future__ import annotations
import argparse,json,os
from collections import deque
from pathlib import Path
import gymnasium as gym, numpy as np, torch, yaml
import tasks  # noqa: F401
from .common import TASKS, POLICY_CONTRACT, atomic_json
from .modeling import load_policy

def scalar(x): return bool(x.detach().cpu().reshape(-1)[0]) if torch.is_tensor(x) else bool(np.asarray(x).reshape(-1)[0])
def env_for(root, spec):
    cfg=Path(root)/'configs/table'/spec.config; name=yaml.safe_load(cfg.read_text())['task_name']
    return gym.make(name+'-rf',config=str(cfg),obs_mode='rgb',control_mode='pd_joint_pos',render_mode='sensors',reward_mode='dense',sim_backend='cpu',render_backend='cpu',sensor_configs={'shader_pack':'default'},human_render_camera_configs={'shader_pack':'default'},viewer_camera_configs={'shader_pack':'default'})
def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--robofactory-root',default=os.getenv('ROBOFACTORY_ROOT','/workspace/repos/RoboFactory')); p.add_argument('--output',required=True); p.add_argument('--episodes',type=int,default=20); p.add_argument('--device',default='cuda:0'); p.add_argument('--replan-interval',type=int,default=8); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    policy,payload=load_policy(a.checkpoint,a.device)
    if payload.get('contract')!=POLICY_CONTRACT: raise RuntimeError('checkpoint policy contract mismatch')
    stats=payload['stats']; qmin=np.asarray(stats['qpos']['min'],np.float32); qmax=np.asarray(stats['qpos']['max'],np.float32); amin=np.asarray(stats['action']['min'],np.float32); amax=np.asarray(stats['action']['max'],np.float32)
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); reports={}; episodes=1 if a.smoke else a.episodes; os.chdir(a.robofactory_root)
    for tid,spec in enumerate(TASKS):
        rows=[]
        for ep in range(episodes):
            seed=(990000+tid*1000+ep) if a.smoke else spec.seed_start+ep; row={'episode':ep,'seed':seed,'success':False,'steps':0}; env=None
            try:
                env=env_for(a.robofactory_root,spec); obs,_=env.reset(seed=seed); ih=[deque(maxlen=2) for _ in range(spec.arms)]; qh=[deque(maxlen=2) for _ in range(spec.arms)]; chunks=[None]*spec.arms; offsets=[a.replan_interval]*spec.arms
                for step in range(spec.max_steps):
                    local=[]
                    for arm in range(spec.arms):
                        image=np.asarray(obs['sensor_data'][f'head_camera_agent{arm}']['rgb'])[0]; q=np.asarray(obs['agent'][f'panda-{arm}']['qpos'])[0,:9]; ih[arm].append(image); qh[arm].append(q)
                        while len(ih[arm])<2: ih[arm].appendleft(ih[arm][0]); qh[arm].appendleft(qh[arm][0])
                        local.append({'head_cam':np.asarray(ih[arm]),'agent_pos':np.clip(2*(np.asarray(qh[arm],np.float32)-qmin)/(qmax-qmin+1e-6)-1,-1,1)})
                    if any(chunks[i] is None or offsets[i]>=a.replan_interval for i in range(spec.arms)):
                        batch={'head_cam':torch.from_numpy(np.stack([x['head_cam'] for x in local])).to(a.device),'agent_pos':torch.from_numpy(np.stack([x['agent_pos'] for x in local])).to(a.device)}; pred=policy.predict_action(batch)['action'].float().cpu().numpy(); chunks=[pred[i] for i in range(spec.arms)]; offsets=[0]*spec.arms
                    action={}
                    for arm in range(spec.arms):
                        encoded=np.clip(chunks[arm][offsets[arm]],-1,1); raw=amin+0.5*(encoded+1)*(amax-amin); space=env.action_space.spaces[f'panda-{arm}']; action[f'panda-{arm}']=np.clip(raw,space.low,space.high).astype(np.float32); offsets[arm]+=1
                    obs,_,term,trunc,info=env.step(action); row['success']=scalar(info.get('success',False)); row['steps']=step+1
                    if row['success'] or scalar(term) or scalar(trunc): break
            except Exception as exc: row['error']=f'{type(exc).__name__}: {exc}'
            finally:
                if env is not None: env.close()
            rows.append(row); print(json.dumps({'task':spec.name,**row}),flush=True)
        report={'schema':'mars-control.maniflow.validation20.task.v1','status':'failed' if any('error' in x for x in rows) else 'complete','task':spec.name,'episodes':len(rows),'target_episodes':episodes,'successes':sum(int(x['success']) for x in rows),'success_rate':sum(int(x['success']) for x in rows)/len(rows),'max_steps':spec.max_steps,'seed_start':spec.seed_start,'policy_contract':POLICY_CONTRACT,'preprocessing':{'rgb':'uint8_div_255_then_model_resize','qpos':'global_corpus_minmax_to_minus1_plus1','action':'global_corpus_minmax_inverse_then_env_clip'},'episodes_detail':rows}; atomic_json(out/f'{spec.name}.json',report); reports[spec.name]=report
    errors=[x for r in reports.values() for x in r['episodes_detail'] if 'error' in x]; summary={'schema':'mars-control.maniflow.validation20.v1','status':'failed' if errors else 'complete','baseline':'ManiFlow','benchmark':'MARS-Control','episodes_per_task':episodes,'total_episodes':episodes*len(TASKS),'successes':sum(r['successes'] for r in reports.values()),'macro_success_rate':float(np.mean([r['success_rate'] for r in reports.values()])),'tasks':reports,'policy_contract':POLICY_CONTRACT}; atomic_json(out/'summary.json',summary)
    if errors: raise RuntimeError(f'{len(errors)} validation episodes failed')
if __name__=='__main__': main()
