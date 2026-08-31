#!/usr/bin/env python3
"""Run four independent arm-local policy/simulator pairs, one per GPU."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,time
from pathlib import Path
from .common import TASKS,CONTRACT,atomic_json
children=[]
def stop():
 for p in reversed(children):
  if p.poll() is None:
   try: os.killpg(p.pid,signal.SIGTERM)
   except ProcessLookupError: pass
 for p in reversed(children):
  try:p.wait(timeout=60)
  except subprocess.TimeoutExpired:
   try:os.killpg(p.pid,signal.SIGKILL)
   except ProcessLookupError:pass
def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True); p.add_argument('--smoke',action='store_true'); a=p.parse_args(); root=Path(a.output); root.mkdir(parents=True,exist_ok=True); repo=Path('/workspace/repos/before-we-act'); policy_py=os.environ.get('PI05_MARS_POLICY_PYTHON','/workspace/venvs/openpi/bin/python'); sim_py=os.environ.get('PI05_MARS_SIM_PYTHON','/workspace/venvs/robofactory/bin/python'); pairs=[]
 try:
  for gpu,task in enumerate(TASKS):
   sock=f'/tmp/pi05-mars-{task}-{os.getpid()}.sock'; env=os.environ.copy(); env.update({'PYTHONPATH':f'{repo}:/workspace/repos/RoboFactory','CUDA_VISIBLE_DEVICES':str(gpu),'HF_HOME':'/workspace/.hf_home','XLA_PYTHON_CLIENT_PREALLOCATE':'false','XLA_PYTHON_CLIENT_ALLOCATOR':'platform','XDG_RUNTIME_DIR':'/tmp/bwa-xdg-runtime','VK_DRIVER_FILES':'/etc/vulkan/icd.d/nvidia_icd.json','VK_ICD_FILENAMES':'/etc/vulkan/icd.d/nvidia_icd.json','LD_LIBRARY_PATH':'/opt/nvidia-drivers/lib64:'+env.get('LD_LIBRARY_PATH','')}); Path('/tmp/bwa-xdg-runtime').mkdir(mode=0o700,exist_ok=True)
   wf=(root/'logs'/f'{task}.worker.log'); wf.parent.mkdir(parents=True,exist_ok=True); w=subprocess.Popen([policy_py,str(repo/'deployment/pi05_mars/rpc_server.py'),'--checkpoint',a.checkpoint,'--socket',sock],env=env,stdout=wf.open('ab'),stderr=subprocess.STDOUT,start_new_session=True); children.append(w); pairs.append((gpu,task,sock,env,w))
  deadline=time.time()+1800
  while any(not Path(x[2]).exists() for x in pairs):
   if any(x[4].poll() is not None for x in pairs): raise RuntimeError('policy worker exited before ready')
   if time.time()>deadline: raise TimeoutError('policy worker startup timeout')
   time.sleep(1)
  evals=[]
  for gpu,task,sock,env,_ in pairs:
   cmd=[sim_py,str(repo/'deployment/pi05_mars/task_validate.py'),'--socket',sock,'--task',task,'--output',str(root/f'{task}.json'),'--episodes','1' if a.smoke else '20']
   if a.smoke: cmd+=['--max-steps','2','--smoke']
   log=(root/'logs'/f'{task}.eval.log').open('ab'); proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); children.append(proc); evals.append((task,proc,log))
  failed=[]
  for task,proc,log in evals:
   code=proc.wait(); log.close()
   if code: failed.append((task,code))
  if failed: raise RuntimeError(f'evaluator failures: {failed}')
  reports={task:json.loads((root/f'{task}.json').read_text()) for task in TASKS}; summary={'schema':'mars-control.pi05.smoke.v1' if a.smoke else 'mars-control.pi05.validation20.v1','status':'complete','benchmark':'MARS-Control','policy':'pi0.5_lora','policy_contract':CONTRACT,'episodes_per_task':1 if a.smoke else 20,'total_episodes':4 if a.smoke else 80,'successes':sum(r['successes'] for r in reports.values()),'tasks':reports,'max_steps':{t:r['max_steps'] for t,r in reports.items()},'checkpoint':a.checkpoint}; atomic_json(root/'summary.json',summary)
 finally: stop()
if __name__=='__main__': main()
