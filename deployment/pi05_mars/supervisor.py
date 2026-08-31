#!/usr/bin/env python3
"""Crash-resumable MARS π0.5 supervisor with four-GPU validation waves."""
from __future__ import annotations
import json,os,signal,subprocess,time
from pathlib import Path
from .common import TASKS,CONTRACT,atomic_json
RUN=Path(os.environ.get('PI05_MARS_RUN_ROOT','/workspace/runs/pi05_mars')); ROOT=Path('/workspace/repos/before-we-act'); OPENPI=Path('/workspace/repos/openpi'); PY='/workspace/venvs/openpi/bin/python'; SIM='/venv/main/bin/python'; active=[]; stop=False
def now(): return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def stage(name,cmd,gpus=(),env=None):
 global active
 log=RUN/'logs'/f'{name}.log'; log.parent.mkdir(parents=True,exist_ok=True); e=os.environ.copy(); e.update(env or {}); e['PATH']=f'/workspace/venvs/openpi/bin:{e.get("PATH","")}'; e['PYTHONPATH']=f'{ROOT}:/workspace/repos/RoboFactory'; e['TOKENIZERS_PARALLELISM']='false'; e['HF_HOME']='/workspace/.hf_home'; e['XLA_PYTHON_CLIENT_PREALLOCATE']='false'; e['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'; e['BWA_RENDER_ICD']='cpu'; e['VK_DRIVER_FILES']='/etc/vulkan/icd.d/nvidia_icd.json'; e['VK_ICD_FILENAMES']=e['VK_DRIVER_FILES']; e['LD_LIBRARY_PATH']='/opt/nvidia-drivers/lib64:'+e.get('LD_LIBRARY_PATH',''); Path('/tmp/bwa-xdg-runtime').mkdir(mode=0o700,exist_ok=True); e['XDG_RUNTIME_DIR']='/tmp/bwa-xdg-runtime';
 if gpus: e['CUDA_VISIBLE_DEVICES']=','.join(map(str,gpus))
 atomic_json(RUN/'state.json',{'status':'running','stage':name,'gpus':list(gpus),'started_at':now(),'log':str(log)})
 with log.open('ab',buffering=0) as f:
  p=subprocess.Popen(cmd,cwd=str(ROOT),env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); active=[p]; code=p.wait()
 active=[]
 if code: atomic_json(RUN/'state.json',{'status':'failed','stage':name,'returncode':code,'log':str(log)}); raise RuntimeError(f'{name} failed ({code})')
 atomic_json(RUN/'receipts'/f'{name}.json',{'status':'complete','stage':name,'completed_at':now(),'log':str(log)})
def done(name):
 try:return json.loads((RUN/'receipts'/f'{name}.json').read_text()).get('status')=='complete'
 except:return False
def handler(*_):
 global stop; stop=True
 for p in active:
  try: os.killpg(p.pid,signal.SIGTERM)
  except ProcessLookupError: pass
def main():
 RUN.mkdir(parents=True,exist_ok=True); signal.signal(signal.SIGTERM,handler); signal.signal(signal.SIGINT,handler)
 # The formal run is a single global-batch experiment sharded over all four
 # GPUs.  Keep its own directory so a pre-existing single-GPU run can never be
 # accidentally resumed with an incompatible mesh.
 formal_exp='all600_4gpu_dp_b128'; formal_dir=RUN/f'checkpoints/pi05_mars_control_lora/{formal_exp}'
 formal_resume=['--resume'] if any(x.is_dir() and x.name.isdigit() for x in formal_dir.glob('*')) else (['--overwrite'] if formal_dir.is_dir() else [])
 collective_check="import importlib.metadata as m,jax,jax.numpy as j; from packaging.version import Version as V; assert V(m.version('jax'))>=V('0.6.2'); assert V(m.version('nvidia-nccl-cu12'))>=V('2.31.2'); assert len(jax.devices())==4; f=jax.pmap(lambda x:jax.lax.psum(x,'i'),axis_name='i'); y=f(j.arange(4)).block_until_ready(); assert (y==6).all(); print(jax.devices(),m.version('nvidia-nccl-cu12'),y)"
 stages=[('preflight_4gpu_blackwell',[PY,'-c',collective_check],(0,1,2,3)),('download',[PY,'-m','deployment.pi05_mars.download'],()),('audit',[PY,'-m','deployment.pi05_mars.audit_norm'],()),('smoke_train_4gpu_dp_b128',[PY,str(OPENPI/'scripts/train.py'),'pi05_mars_control_lora','--checkpoint-base-dir=/workspace/runs/pi05_mars/smoke128_dp_checkpoints','--exp-name=smoke128dp','--batch-size=128','--num-workers=8','--num-train-steps=2','--save-interval=2','--keep-period=2','--fsdp-devices=1','--no-wandb-enabled','--overwrite'],(0,1,2,3)),('smoke_validation_4gpu_dp_b128',[PY,'-m','deployment.pi05_mars.validation_launcher','--checkpoint','/workspace/runs/pi05_mars/smoke128_dp_checkpoints/pi05_mars_control_lora/smoke128dp/1','--output','/workspace/runs/pi05_mars/smoke128_dp_validation','--smoke'],(0,1,2,3)),('formal_train',[PY,str(OPENPI/'scripts/train.py'),'pi05_mars_control_lora','--checkpoint-base-dir=/workspace/runs/pi05_mars/checkpoints','--exp-name',formal_exp,'--batch-size=128','--num-workers=8','--num-train-steps=30000','--save-interval=1000','--keep-period=10000','--fsdp-devices=1','--no-wandb-enabled']+formal_resume,(0,1,2,3))]
 for name,cmd,gpu in stages:
  if stop: return
  if not done(name): stage(name,cmd,gpu)
 ck=f'/workspace/runs/pi05_mars/checkpoints/pi05_mars_control_lora/{formal_exp}/29999'; final=ck
 if not Path(final).is_dir(): raise FileNotFoundError(f'Final checkpoint missing: {final}')
 if not done('validation20'): stage('validation20',[PY,'-m','deployment.pi05_mars.validation_launcher','--checkpoint',final,'--output',str(RUN/'validation20')],(0,1,2,3))
 atomic_json(RUN/'state.json',{'status':'complete','stage':'validation20','updated_at':now()})
if __name__=='__main__': main()
