"""Paired selector-off/CARE closed-loop validation for one DuoBench task."""
from __future__ import annotations
import argparse, hashlib, json, time
from collections import deque
from pathlib import Path
import numpy as np, torch
from before_we_act.care_belief import CAREBeliefConfig, CAREBeliefHead, CAREBeliefOutput, CARECalibration, select_care_candidate
from deployment.mars_care.model import CAREPolicy, ModelConfig
from .action_contract import current_relative_temporal_ensemble
from .branch_collection import _image, _q, _action
from .evaluate import env_for
from .prepare import TASKS

def load(reference_path,care_path,device):
    ref=torch.load(reference_path,map_location='cpu',weights_only=False); cfg=ModelConfig(**ref['model_config']); model=CAREPolicy(cfg).to(device); model.load_state_dict(ref['model'],strict=True); model.eval(); n=ref['normalization']; stats=tuple(np.asarray(n[k],np.float32) for k in ('qpos_mean','qpos_std','action_mean','action_std'))
    saved=torch.load(care_path,map_location='cpu',weights_only=False); head=CAREBeliefHead(CAREBeliefConfig.from_mapping(saved['config'])).to(device); head.load_state_dict(saved['model'],strict=True); head.eval(); calibration=CARECalibration.from_mapping(saved['calibration']); return model,cfg,stats,head,calibration,saved
def run(task,seed,mode,max_steps,model,cfg,stats,head,calibration,device):
    env=env_for(task); obs,info=env.reset(seed=int(seed)); histories=[deque(maxlen=cfg.history),deque(maxlen=cfg.history)]; previous=[None,None]; active=[]; trace=hashlib.sha256(); selected_rows=[]; success=False; stages=[]; t0=time.perf_counter(); rng=np.random.default_rng(731947); projection=torch.from_numpy(rng.normal(0,.08,(cfg.qpos_dim+cfg.action_dim,384)).astype(np.float32)).to(device)
    for step in range(max_steps):
        qm,qs,am,ass=stats; images=[]; qnorm=[]; memories=[]; masks=[]
        for arm in range(2):
            q=_q(obs,arm); tok=np.concatenate(((q-qm)/qs,np.zeros(8,np.float32) if previous[arm] is None else (previous[arm]-am)/ass)).astype(np.float32); histories[arm].append(tok); images.append(_image(obs,arm,cfg.image_size)); qnorm.append((q-qm)/qs); mem=np.zeros((16,cfg.qpos_dim+cfg.action_dim),np.float32); vals=np.asarray(histories[arm],np.float32); mem[-len(vals):]=vals; memories.append(mem); masks.append(np.arange(16)>=16-len(vals))
        h=np.asarray(memories,np.float32); mask=np.asarray(masks,bool)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
            out=model(torch.stack(images).to(device),torch.from_numpy(np.asarray(qnorm)).to(device),torch.full((2,),TASKS.index(task),dtype=torch.long,device=device),torch.from_numpy(h).to(device),torch.from_numpy(mask.astype(np.float32)).to(device)); raw=out['candidates'].float()*torch.from_numpy(ass).to(device)+torch.from_numpy(am).to(device); chunks=torch.cat((raw,raw[:,:,-1:].expand(-1,-1,100-cfg.horizon,-1)),2)
            if mode=='care':
                belief=head(torch.from_numpy(h).to(device)@projection,torch.from_numpy(mask).to(device),chunks,torch.full((2,),head.config.horizons.index(calibration.primary_horizon),dtype=torch.long,device=device)); selected,lower,unsafe=select_care_candidate(belief,calibration)
            else: selected=torch.zeros(2,dtype=torch.long,device=device); lower=torch.zeros(2,device=device); unsafe=torch.zeros((2,6),dtype=torch.bool,device=device)
        chosen=chunks[torch.arange(2,device=device),selected].cpu().numpy(); current_q=np.asarray([_q(obs,arm) for arm in range(2)],np.float32); active.append((step,chosen,current_q.copy())); active=[x for x in active if step-x[0]<cfg.horizon]; encoded=current_relative_temporal_ensemble(active,step,current_q); action=_action(env,obs,encoded)
        for arm in range(2): previous[arm]=encoded[arm].copy()
        trace.update(encoded.astype(np.float32).tobytes()); selected_rows.append({'step':step,'selected':selected.cpu().tolist(),'lower_bound':lower.float().cpu().tolist(),'unsafe_count':unsafe.sum(1).cpu().tolist()}); obs,reward,term,trunc,info=env.step(action); stage=float(info.get('stage',0))/max(float(info.get('max_stage',1)),1); stages.append(stage); success=bool(info.get('success',False))
        if success or bool(np.asarray(term).all()) or bool(np.asarray(trunc).all()): break
    env.close(); overrides=sum(sum(x!=0 for x in row['selected']) for row in selected_rows); decisions=max(1,2*len(selected_rows)); return {'task':task,'seed':int(seed),'mode':mode,'success':success,'steps':len(selected_rows),'max_steps':max_steps,'final_stage_progress':stages[-1] if stages else 0.,'max_stage_progress':max(stages) if stages else 0.,'override_rate':overrides/decisions,'selected_candidates':selected_rows,'action_trace_sha256':trace.hexdigest(),'wall_seconds':time.perf_counter()-t0}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--reference-checkpoint',type=Path,required=True); p.add_argument('--care-checkpoint',type=Path,required=True); p.add_argument('--data',type=Path,required=True); p.add_argument('--task',choices=TASKS,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--episodes',type=int,default=20); p.add_argument('--seed-start',type=int,default=20260820); p.add_argument('--device',default='cuda:0'); p.add_argument('--max-steps',type=int); a=p.parse_args(); device=torch.device(a.device); model,cfg,stats,head,calibration,saved=load(a.reference_checkpoint,a.care_checkpoint,device); manifest=json.loads((a.data/'manifest.json').read_text()); maximum=a.max_steps or int(manifest['tasks'][a.task]['validation_max_steps']); rows=[]; recovered={}
    jsonl=a.output.with_suffix('.jsonl'); a.output.parent.mkdir(parents=True,exist_ok=True)
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            try:r=json.loads(line); recovered[(r['mode'],int(r['seed']))]=r
            except Exception:pass
    for ep in range(a.episodes):
        seed=a.seed_start+TASKS.index(a.task)*1000+ep
        for mode in ('selector_off','care'):
            row=recovered.get((mode,seed)) or run(a.task,seed,mode,maximum,model,cfg,stats,head,calibration,device); rows.append(row)
            if (mode,seed) not in recovered: jsonl.open('a').write(json.dumps(row)+'\n'); print(json.dumps({'task':a.task,'seed':seed,'mode':mode,'success':row['success']}),flush=True)
    pairs=[]
    for ep in range(a.episodes):
        seed=a.seed_start+TASKS.index(a.task)*1000+ep; off=next(r for r in rows if r['seed']==seed and r['mode']=='selector_off'); care=next(r for r in rows if r['seed']==seed and r['mode']=='care'); pairs.append({'seed':seed,'selector_off_success':off['success'],'care_success':care['success'],'success_delta':int(care['success'])-int(off['success']),'progress_delta':care['final_stage_progress']-off['final_stage_progress'],'harmful_override':care['final_stage_progress']<off['final_stage_progress'] and care['override_rate']>0})
    result={'status':'complete','format_version':'before-we-act.care-duobench-paired-validation/1','task':a.task,'episodes':a.episodes,'max_steps':maximum,'selector_off_success_rate':float(np.mean([r['success'] for r in rows if r['mode']=='selector_off'])),'care_success_rate':float(np.mean([r['success'] for r in rows if r['mode']=='care'])),'paired_success_improvement':float(np.mean([r['success_delta'] for r in pairs])),'override_rate':float(np.mean([r['override_rate'] for r in rows if r['mode']=='care'])),'harmful_override_rate':float(np.mean([r['harmful_override'] for r in pairs])),'rows':rows,'pairs':pairs}; a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__': main()
