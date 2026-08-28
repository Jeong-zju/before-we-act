#!/usr/bin/env python3
"""Crash-resumable one-GPU supervisor for the complete ManiFlow pipeline."""
from __future__ import annotations
import json, os, signal, subprocess
from datetime import datetime, timezone
from pathlib import Path
from .common import atomic_json
ROOT=Path(__file__).resolve().parents[2]; RUN=Path(os.getenv('MARS_MANIFLOW_RUN_ROOT','/workspace/runs/mars_maniflow')); DATA=Path(os.getenv('MARS_MANIFLOW_DATA_ROOT','/workspace/datasets/mars_control')); RF=Path(os.getenv('ROBOFACTORY_ROOT','/workspace/repos/RoboFactory')); PY=os.getenv('MARS_MANIFLOW_PYTHON','/venv/main/bin/python'); active=None; stop=False
def now(): return datetime.now(timezone.utc).isoformat()
def state(stage,status='running',**kw): atomic_json(RUN/'state.json',{'schema':'mars-control.maniflow.supervisor.v1','stage':stage,'status':status,'updated_at':now(),'gpu':[0],**kw})
def done(name):
 try:return json.loads((RUN/'receipts'/f'{name}.json').read_text()).get('status')=='complete'
 except:return False
def run(name,cmd):
 global active; log=RUN/'logs'/f'{name}.log'; log.parent.mkdir(parents=True,exist_ok=True); env=os.environ.copy(); env.update({'PYTHONPATH':f'{ROOT}:{RF}:/workspace/repos/ManiFlow_Policy/ManiFlow','CUDA_VISIBLE_DEVICES':'0','HF_HOME':'/workspace/.hf_home','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'8','VK_DRIVER_FILES':'/workspace/nvidia-580.159.03/root/usr/share/vulkan/icd.d/nvidia_icd.json','VK_ICD_FILENAMES':'/workspace/nvidia-580.159.03/root/usr/share/vulkan/icd.d/nvidia_icd.json','XDG_RUNTIME_DIR':'/tmp/bwa-xdg-runtime','LD_LIBRARY_PATH':'/workspace/nvidia-580.159.03/root/usr/lib/x86_64-linux-gnu:'+os.environ.get('LD_LIBRARY_PATH','')}); Path('/tmp/bwa-xdg-runtime').mkdir(mode=0o700,exist_ok=True); state(name,command=cmd,log=str(log))
 with log.open('ab',buffering=0) as f: active=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); code=active.wait()
 active=None
 if code: raise RuntimeError(f'{name} exited {code}; see {log}')
 atomic_json(RUN/'receipts'/f'{name}.json',{'schema':'mars-control.maniflow.stage.v1','status':'complete','stage':name,'completed_at':now(),'log':str(log)})
def handle(*_):
 global stop; stop=True; state('stopping','stopping')
 if active:
  try: os.killpg(active.pid,signal.SIGTERM)
  except ProcessLookupError: pass
def main():
 global stop; RUN.mkdir(parents=True,exist_ok=True); signal.signal(signal.SIGINT,handle); signal.signal(signal.SIGTERM,handle); stats=RUN/'normalization.json'; formal=RUN/'formal'; smoke=RUN/'smoke'
 stages=[('preflight',[PY,'-c','import torch,h5py; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(torch.__version__,torch.cuda.get_device_name(0))']),('download',[PY,'-m','deployment.mars_maniflow.download','--data-root',str(DATA)]),('audit',[PY,'-m','deployment.mars_maniflow.audit','--data-root',str(DATA),'--output',str(RUN/'audit.json'),'--stats',str(stats)]),('smoke_train',[PY,'-m','deployment.mars_maniflow.train','--dataset-root',str(DATA),'--stats',str(stats),'--output',str(smoke),'--steps','2','--batch-size','16','--workers','0','--save-every','2','--log-every','1','--smoke']),('smoke_validation',[PY,'-m','deployment.mars_maniflow.validate','--checkpoint',str(smoke/'last.pt'),'--output',str(smoke/'validation20'),'--robofactory-root',str(RF),'--episodes','1','--device','cuda:0','--replan-interval','8','--smoke']),('formal_train',[PY,'-m','deployment.mars_maniflow.train','--dataset-root',str(DATA),'--stats',str(stats),'--output',str(formal),'--steps',os.getenv('MANIFLOW_STEPS','60000'),'--batch-size',os.getenv('MANIFLOW_BATCH','128'),'--workers',os.getenv('MANIFLOW_WORKERS','16'),'--save-every','5000','--log-every','50','--resume']),('validation20',[PY,'-m','deployment.mars_maniflow.validate','--checkpoint',str(formal/'last.pt'),'--output',str(formal/'validation20'),'--robofactory-root',str(RF),'--episodes','20','--device','cuda:0','--replan-interval','8'])]
 for name,cmd in stages:
  if stop: break
  if done(name): continue
  try: run(name,cmd)
  except Exception as exc: state(name,'failed',error=repr(exc)); raise
 if not stop: state('complete','complete')
if __name__=='__main__': main()
