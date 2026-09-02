"""Restartable full DuoBench CARE supervisor with bounded four-GPU scheduling."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,time,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(os.environ.get('DUO_CARE_REPO','/workspace/repos/care-official')); RUN=Path(os.environ.get('DUO_CARE_RUN','/workspace/runs/duobench-care-full')); DATASET=Path(os.environ.get('DUO_CARE_DATASET','/workspace/datasets/duobench')); DATA=RUN/'base_data'; FAMILY=RUN/'families'; PREP=RUN/'care_prepared.pt'; CARE=RUN/'offline/care_deployment_checkpoint.pt'; PY='/venv/main/bin/python'; active=[]
def now():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def status(stage,state='running',**kw):
 RUN.mkdir(parents=True,exist_ok=True); val={'schema':'before-we-act.duobench-care-supervisor/1','stage':stage,'state':state,'updated_at':now(),'pid':os.getpid(),'gpu_schedule':'base DDP 0-3; branch/validation waves one isolated GPU; belief variants/seed waves max 4 concurrent',**kw}; t=RUN/'status.json.tmp'; t.write_text(json.dumps(val,indent=2)+'\n'); os.replace(t,RUN/'status.json')
def env(gpus='0,1,2,3'):
 e=os.environ.copy(); e.update({'PYTHONPATH':f'{ROOT}:/workspace/repos/duobench/src','MUJOCO_GL':'egl','DUOBENCH_PREFIX':'/workspace/repos/duobench','CUDA_VISIBLE_DEVICES':gpus,'HF_HOME':'/workspace/.hf_home','WANDB_MODE':'disabled','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8'}); return e
def run(stage,cmd,gpus='0,1,2,3',retries=1):
 log=RUN/'logs'/f'{stage}.log'; log.parent.mkdir(parents=True,exist_ok=True)
 for attempt in range(1,retries+1):
  status(stage,attempt=attempt,command=cmd,gpus=gpus)
  with log.open('a') as out:
   p=subprocess.Popen(cmd,cwd=ROOT,env=env(gpus),stdout=out,stderr=subprocess.STDOUT,start_new_session=True); active.append(p); code=p.wait(); active.remove(p)
  if code==0:return
  if attempt<retries:time.sleep(5*attempt)
 raise RuntimeError(f'{stage} failed code={code}')
def stop(sig,frame):
 status('stopping','stopping')
 for p in list(active):
  try:os.killpg(p.pid,signal.SIGTERM)
  except ProcessLookupError:pass
def main():
 signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop); RUN.mkdir(parents=True,exist_ok=True)
 try:
  run('preflight',[PY,'-m','deployment.duo_care.preflight','--data',str(DATA),'--output',str(RUN/'preflight.json')],retries=2) if (DATA/'manifest.json').exists() else run('gpu_preflight',[PY,'-c','import torch; assert torch.cuda.device_count()==4; assert all("5090" in torch.cuda.get_device_name(i) for i in range(4)); print([torch.cuda.get_device_name(i) for i in range(4)])'])
  if not (DATA/'manifest.json').exists(): run('base_data_prepare',[PY,'-m','deployment.duo_care.prepare','--dataset',str(DATASET),'--output',str(DATA),'--image-size','224','--jobs','8'],'0',2)
  if not (RUN/'audit.json').exists(): run('base_data_audit',[PY,'-m','deployment.duo_care.audit','--data',str(DATA),'--output',str(RUN/'audit.json')],'0')
  if not (RUN/'preflight.json').exists(): run('interface_preflight',[PY,'-m','deployment.duo_care.preflight','--data',str(DATA),'--output',str(RUN/'preflight.json')],'0',2)
  smoke=RUN/'base_smoke/final.pt';
  if not smoke.exists(): run('base_policy_smoke_train',[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m','deployment.duo_care.train','--data',str(DATA),'--output',str(smoke.parent),'--steps','5','--batch-size','16','--workers','4','--save-every','5','--smoke'],'0,1,2,3',2)
  if not (RUN/'base_smoke/validation/summary.json').exists(): run('base_policy_smoke_validation',[PY,'-m','deployment.duo_care.validation_launcher','--checkpoint',str(smoke),'--data',str(DATA),'--output',str((RUN/'base_smoke/validation').resolve()),'--episodes','1','--max-steps','2','--workers','4'],'0,1,2,3',2)
  formal=RUN/'base_formal/final.pt';
  if not formal.exists(): run('base_policy_formal_train',[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m','deployment.duo_care.train','--data',str(DATA),'--output',str(formal.parent),'--steps',os.environ.get('DUO_CARE_STEPS','20000'),'--batch-size','64','--workers','8','--save-every','1000','--init-checkpoint',str(smoke)],'0,1,2,3',3)
  if not (RUN/'reference_validation20.json').exists(): run('reference_validation20',[PY,'-m','deployment.duo_care.validation_launcher','--checkpoint',str(formal),'--data',str(DATA),'--output',str((RUN/'reference_validation').resolve()),'--episodes','20','--workers','4','--candidate-zero'],'0,1,2,3',3); os.replace(RUN/'reference_validation/summary.json',RUN/'reference_validation20.json')
  smoke_root=RUN/'care_smoke'; smoke_family=smoke_root/'family'
  if not (smoke_family/'manifest.json').exists(): run('branch_collection_smoke',[PY,'-m','deployment.duo_care.branch_collection','--checkpoint',str(formal),'--output',str(smoke_family),'--families-per-task','1','--task','ball_maze'],'0',2)
  smoke_prepared=smoke_root/'prepared.pt'
  if not smoke_prepared.exists(): run('care_prepared_smoke',[PY,'-m','scripts.before_we_act.prepare_duo_care_training','--family-root',str(smoke_family),'--reference-checkpoint',str(formal),'--output',str(smoke_prepared),'--manifest',str(smoke_root/'prepared_manifest.json'),'--expected-families','1'],'0')
  smoke_belief=smoke_root/'belief/checkpoint_latest.pt'
  if not smoke_belief.exists(): run('care_belief_smoke',[PY,'-m','before_we_act.train_mars_care_belief','--prepared-data',str(smoke_prepared),'--output',str(smoke_belief.parent),'--seed','20260818','--variant','care','--stage','smoke','--updates','2','--batch-size','2','--eval-every','1','--save-every','1','--device','cuda:0','--benchmark-adapter','DuoBench'],'0')
  smoke_deploy=smoke_root/'care_smoke.pt'
  if not smoke_deploy.exists(): run('care_smoke_deployment',[PY,'-m','scripts.before_we_act.build_duo_care_smoke_deployment','--belief-checkpoint',str(smoke_belief),'--reference-checkpoint',str(formal),'--prepared-data',str(smoke_prepared),'--output',str(smoke_deploy)],'0')
  if not (smoke_root/'paired.json').exists(): run('paired_validation_smoke',[PY,'-m','deployment.duo_care.paired_evaluate','--reference-checkpoint',str(formal),'--care-checkpoint',str(smoke_deploy),'--data',str(DATA),'--task','ball_maze','--output',str(smoke_root/'paired.json'),'--episodes','1','--max-steps','2'],'0',2)
  if not (FAMILY/'families/manifest.json').exists() or json.loads((FAMILY/'families/manifest.json').read_text()).get('family_count',0)<int(os.environ.get('DUO_CARE_FAMILIES_PER_TASK','30'))*11: run('branch_collection_formal',[PY,'-m','deployment.duo_care.branch_launcher','--checkpoint',str(formal),'--output',str(FAMILY),'--families-per-task',os.environ.get('DUO_CARE_FAMILIES_PER_TASK','30'),'--workers','4'],'0,1,2,3',2)
  family_root=FAMILY/'families'
  if not PREP.exists(): run('care_prepared_data',[PY,'-m','scripts.before_we_act.prepare_duo_care_training','--family-root',str(family_root),'--reference-checkpoint',str(formal),'--output',str(PREP),'--manifest',str(RUN/'care_prepared_manifest.json'),'--expected-families',str(int(os.environ.get('DUO_CARE_FAMILIES_PER_TASK','30'))*11)],'0')
  train_root=RUN/'belief_training'; variants=('care','reactive_only','replay_only','capacity'); seeds=(20260818,20260819,20260820); jobs=[]
  for variant in variants:
   for seed in seeds:
    out=train_root/variant/f'seed_{seed}'
    if (out/'status.json').exists() and json.loads((out/'status.json').read_text()).get('status')=='COMPLETED': continue
    gpu=str(len(jobs)%4); jobs.append((variant,seed,out,gpu))
    if len(jobs)==4:
     procs=[]
     for v,s,o,g in jobs:
      cmd=[PY,'-m','before_we_act.train_mars_care_belief','--prepared-data',str(PREP),'--output',str(o),'--seed',str(s),'--variant',v,'--stage','formal','--updates','4000','--batch-size','48','--eval-every','200','--learning-rate','3e-4','--weight-decay','1e-4','--device','cuda:0','--benchmark-adapter','DuoBench']; f=(RUN/'logs'/f'belief_{v}_{s}.log').open('a'); procs.append(subprocess.Popen(cmd,cwd=ROOT,env=env(g),stdout=f,stderr=subprocess.STDOUT,start_new_session=True))
     for p in procs:
      if p.wait()!=0: raise RuntimeError('belief training wave failed')
     jobs=[]
  if jobs:
   for v,s,o,g in jobs: run(f'belief_{v}_{s}',[PY,'-m','before_we_act.train_mars_care_belief','--prepared-data',str(PREP),'--output',str(o),'--seed',str(s),'--variant',v,'--stage','formal','--updates','4000','--batch-size','48','--eval-every','200','--device','cuda:0','--benchmark-adapter','DuoBench'],g,2)
  if not CARE.exists(): run('offline_selection_calibration',[PY,'-m','scripts.before_we_act.select_calibrate_duo_care','--prepared-data',str(PREP),'--training-root',str(train_root),'--reference-checkpoint',str(formal),'--output-root',str(CARE.parent)],'0')
  paired=RUN/'paired_validation20';
  if not (paired/'summary.json').exists(): run('paired_validation20',[PY,'-m','deployment.duo_care.paired_launcher','--reference-checkpoint',str(formal),'--care-checkpoint',str(CARE),'--data',str(DATA),'--output',str(paired),'--episodes','20','--workers','4'],'0,1,2,3',3)
  status('complete','complete',reference_checkpoint=str(formal),care_checkpoint=str(CARE),paired_summary=str(paired/'summary.json'))
 except Exception as exc:
  status('failed','failed',error=repr(exc),traceback=traceback.format_exc()); raise
if __name__=='__main__':main()
