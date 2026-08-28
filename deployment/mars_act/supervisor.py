#!/usr/bin/env python3
"""Crash-resumable MARS-Control ACT pipeline supervisor (one GPU)."""
import json,os,signal,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
RUN=Path(os.environ.get('MARS_ACT_RUN_ROOT','/workspace/runs/mars_act')); ROOT=Path('/workspace/repos/before-we-act'); DATA=Path(os.environ.get('MARS_ACT_DATA_ROOT','/workspace/datasets/mars_control')); RF=Path('/workspace/repos/RoboFactory'); PY='/venv/main/bin/python'; EVALUATOR_REVISION='rgb-uint8-to-unit-float-v2'; stop=False; active=None
def now(): return datetime.now(timezone.utc).isoformat()
def write(p,v): p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix('.tmp'); t.write_text(json.dumps(v,indent=2)+'\n'); os.replace(t,p)
def state(stage,status='running',**kw): write(RUN/'state.json',{'schema':'mars-control.act.supervisor.v1','stage':stage,'status':status,'updated_at':now(),**kw})
def done(name):
 p=RUN/'receipts'/f'{name}.json'
 try:
  if json.loads(p.read_text()).get('status')!='complete': return False
  if name=='validation20': return json.loads((RUN/'validation20/summary.json').read_text()).get('evaluator_revision')==EVALUATOR_REVISION
  if name=='finalize': return json.loads((RUN/'final_report.json').read_text()).get('validation20',{}).get('evaluator_revision')==EVALUATOR_REVISION
  return True
 except:return False
def run(name,cmd,env=None):
 global active
 log=RUN/'logs'/f'{name}.log'; log.parent.mkdir(parents=True,exist_ok=True); e=os.environ.copy(); e.update({'PYTHONPATH':f'{ROOT}:{RF}','CUDA_VISIBLE_DEVICES':'0','HF_HOME':'/workspace/.hf_home','TOKENIZERS_PARALLELISM':'false'}); e.update(env or {})
 state(name,command=cmd,log=str(log),gpu=[0]);
 with log.open('ab',buffering=0) as f:
  active=subprocess.Popen(cmd,cwd=ROOT,env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); code=active.wait()
 active=None
 if code: raise RuntimeError(f'{name} exited {code}')
 write(RUN/'receipts'/f'{name}.json',{'schema':'mars-control.act.stage.v1','stage':name,'status':'complete','completed_at':now(),'log':str(log)})
def sig(_n,_f):
 global stop; stop=True
 if active:
  try: os.killpg(active.pid,signal.SIGTERM)
  except ProcessLookupError: pass
def main():
 global stop
 RUN.mkdir(parents=True,exist_ok=True); signal.signal(signal.SIGTERM,sig); signal.signal(signal.SIGINT,sig); stages=[
  ('preflight',[PY,'-c','import torch,h5py; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(torch.__version__,torch.cuda.get_device_name(0))']),
  ('audit',[PY,str(ROOT/'deployment/mars_act/audit.py')]),
  ('assets',[str(ROOT/'deployment/mars_act/run_assets.sh')]),
  ('smoke_train',[PY,'-m','deployment.mars_act.train','--data-root',str(DATA),'--output',str(RUN/'smoke'),'--smoke']),
  ('smoke_eval',[PY,str(ROOT/'deployment/mars_act/run_validation.py'),'--checkpoint',str(RUN/'smoke/final.pt'),'--output-root',str(RUN/'smoke/validation'),'--robofactory-root',str(RF),'--smoke']),
  ('formal_train',[PY,'-m','deployment.mars_act.train','--data-root',str(DATA),'--output',str(RUN/'formal'),'--resume']),
  ('validation20',[PY,str(ROOT/'deployment/mars_act/run_validation.py'),'--checkpoint',str(RUN/'formal/final.pt'),'--output-root',str(RUN/'validation20'),'--robofactory-root',str(RF)]),
  ('finalize',[PY,str(ROOT/'deployment/mars_act/finalize.py')])]
 while not stop:
  progressed=False
  for name,cmd in stages:
   if stop: break
   if done(name): continue
   progressed=True
   while not stop:
    try: run(name,cmd); break
    except Exception as exc:
     state(name,'retrying',error=repr(exc)); time.sleep(15)
  if not progressed and not stop: state('complete','complete'); time.sleep(60)
 state('stopped','stopped')
if __name__=='__main__': main()
