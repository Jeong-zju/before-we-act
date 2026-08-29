#!/usr/bin/env python3
"""Strict MARS-Control Validation20 for the shared local ManiFlow policy."""
from __future__ import annotations
import argparse,json,os
from collections import deque
from pathlib import Path
import gymnasium as gym, numpy as np, torch, yaml
import tasks  # noqa: F401
from .common import ACTION_HIGH, ACTION_LOW, EVALUATOR_REVISION, MAX_STEPS_BY_TASK, POLICY_CONTRACT, REPLAN_INTERVAL, TASKS, TEMPORAL_ENSEMBLE_DECAY, TemporalEnsemble, atomic_json, load_frozen_config, sha256
from .modeling import load_policy

def scalar(x): return bool(x.detach().cpu().reshape(-1)[0]) if torch.is_tensor(x) else bool(np.asarray(x).reshape(-1)[0])
def env_for(root, spec):
    cfg=Path(root)/'configs/table'/spec.config; name=yaml.safe_load(cfg.read_text())['task_name']
    if name+'-rf' != spec.env_id: raise RuntimeError(f'environment ID drift for {spec.name}: {name}-rf')
    return gym.make(name+'-rf',config=str(cfg),obs_mode='rgb',control_mode='pd_joint_pos',render_mode='sensors',reward_mode='dense',sim_backend='cpu',render_backend='cpu',sensor_configs={'shader_pack':'default'},human_render_camera_configs={'shader_pack':'default'},viewer_camera_configs={'shader_pack':'default'})
@torch.no_grad()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--robofactory-root',default=os.getenv('ROBOFACTORY_ROOT','/workspace/repos/RoboFactory')); p.add_argument('--output',required=True); p.add_argument('--episodes',type=int,default=20); p.add_argument('--device',default='cuda:0'); p.add_argument('--replan-interval',type=int,default=REPLAN_INTERVAL); p.add_argument('--temporal-ensemble-decay',type=float,default=TEMPORAL_ENSEMBLE_DECAY); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    frozen=load_frozen_config(); validation=frozen['validation20']
    if not a.smoke and a.episodes != validation['episodes_per_task']: raise RuntimeError('formal Validation20 requires exactly 20 episodes per task')
    if a.replan_interval != REPLAN_INTERVAL: raise RuntimeError(f'replan interval must remain frozen at {REPLAN_INTERVAL}')
    if a.temporal_ensemble_decay != TEMPORAL_ENSEMBLE_DECAY: raise RuntimeError(f'temporal ensemble decay must remain frozen at {TEMPORAL_ENSEMBLE_DECAY}')
    if {spec.name: spec.max_steps for spec in TASKS} != MAX_STEPS_BY_TASK: raise RuntimeError('TaskSpec maximum-step contract mismatch')
    policy,payload=load_policy(a.checkpoint,a.device)
    if payload.get('contract')!=POLICY_CONTRACT: raise RuntimeError('checkpoint policy contract mismatch')
    stats=payload['stats']; qmin=np.asarray(stats['qpos']['min'],np.float32); qmax=np.asarray(stats['qpos']['max'],np.float32); amin=np.asarray(stats['action']['min'],np.float32); amax=np.asarray(stats['action']['max'],np.float32)
    if not stats['action'].get('clipped_before_stats') or not np.allclose(stats['action'].get('clip_low'),ACTION_LOW) or not np.allclose(stats['action'].get('clip_high'),ACTION_HIGH): raise RuntimeError('checkpoint action clipping contract mismatch')
    checkpoint_sha256=sha256(a.checkpoint); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); reports={}; episodes=1 if a.smoke else a.episodes; os.chdir(a.robofactory_root)
    for tid,spec in enumerate(TASKS):
        rows=[]
        for ep in range(episodes):
            seed=(990000+tid*1000+ep) if a.smoke else spec.seed_start+ep; row={'episode':ep,'seed':seed,'success':False,'steps':0}; env=None
            try:
                env=env_for(a.robofactory_root,spec); obs,_=env.reset(seed=seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); ih=[deque(maxlen=2) for _ in range(spec.arms)]; qh=[deque(maxlen=2) for _ in range(spec.arms)]; ensemble=TemporalEnsemble(spec.arms,a.temporal_ensemble_decay)
                for step in range(spec.max_steps):
                    local=[]
                    for arm in range(spec.arms):
                        image=np.asarray(obs['sensor_data'][f'head_camera_agent{arm}']['rgb'])[0]; q=np.asarray(obs['agent'][f'panda-{arm}']['qpos'])[0,:9]; ih[arm].append(image); qh[arm].append(q)
                        while len(ih[arm])<2: ih[arm].appendleft(ih[arm][0]); qh[arm].appendleft(qh[arm][0])
                        local.append({'head_cam':np.asarray(ih[arm]),'agent_pos':np.clip(2*(np.asarray(qh[arm],np.float32)-qmin)/(qmax-qmin+1e-6)-1,-1,1)})
                    if step % a.replan_interval == 0:
                        sample_seed=seed+step//max(a.replan_interval,1); torch.manual_seed(sample_seed); torch.cuda.manual_seed_all(sample_seed)
                        batch={'head_cam':torch.from_numpy(np.stack([x['head_cam'] for x in local])).to(a.device),'agent_pos':torch.from_numpy(np.stack([x['agent_pos'] for x in local])).to(a.device)}; pred=policy.predict_action(batch)['action'].float().cpu().numpy(); ensemble.add(step,pred)
                    encoded_actions=ensemble.select(step)
                    action={}
                    for arm in range(spec.arms):
                        encoded=np.clip(encoded_actions[arm],-1,1); raw=amin+0.5*(encoded+1)*(amax-amin); space=env.action_space.spaces[f'panda-{arm}']; action[f'panda-{arm}']=np.clip(raw,space.low,space.high).astype(np.float32)
                    obs,_,term,trunc,info=env.step(action); row['success']=scalar(info.get('success',False)); row['steps']=step+1
                    if row['success'] or scalar(term) or scalar(trunc): break
            except Exception as exc: row['error']=f'{type(exc).__name__}: {exc}'
            finally:
                if env is not None: env.close()
            rows.append(row); print(json.dumps({'task':spec.name,**row}),flush=True)
        report={'schema':'mars-control.maniflow.validation20.task.v3','status':'failed' if any('error' in x for x in rows) else 'complete','task':spec.name,'episodes':len(rows),'target_episodes':episodes,'successes':sum(int(x['success']) for x in rows),'success_rate':sum(int(x['success']) for x in rows)/len(rows),'max_steps':spec.max_steps,'seed_start':spec.seed_start,'policy_contract':POLICY_CONTRACT,'evaluator_revision':EVALUATOR_REVISION,'checkpoint_sha256':checkpoint_sha256,'replan_interval':a.replan_interval,'chunk_aggregation':'temporal_ensemble','temporal_ensemble_decay':a.temporal_ensemble_decay,'preprocessing':{'rgb':'uint8_div_255_then_model_resize','qpos':'global_corpus_minmax_to_minus1_plus1','action':'env_clipped_before_training_minmax_inverse_once_then_env_clip'},'episodes_detail':rows}; atomic_json(out/f'{spec.name}.json',report); reports[spec.name]=report
    errors=[x for r in reports.values() for x in r['episodes_detail'] if 'error' in x]; summary={'schema':'mars-control.maniflow.validation20.v3','status':'failed' if errors else 'complete','baseline':'ManiFlow','benchmark':'MARS-Control','episodes_per_task':episodes,'total_episodes':episodes*len(TASKS),'successes':sum(r['successes'] for r in reports.values()),'macro_success_rate':float(np.mean([r['success_rate'] for r in reports.values()])),'tasks':reports,'max_steps':MAX_STEPS_BY_TASK,'policy_contract':POLICY_CONTRACT,'evaluator_revision':EVALUATOR_REVISION,'checkpoint_sha256':checkpoint_sha256,'replan_interval':a.replan_interval,'chunk_aggregation':'temporal_ensemble','temporal_ensemble_decay':a.temporal_ensemble_decay}; atomic_json(out/'summary.json',summary)
    if errors: raise RuntimeError(f'{len(errors)} validation episodes failed')
if __name__=='__main__': main()
