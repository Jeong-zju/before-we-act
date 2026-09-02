#!/usr/bin/env python3
"""Crash-resumable RDT-1B MARS-Control supervisor.

Stages own their process groups and GPU masks.  A failed stage is retried with
the same immutable command; later stages cannot run until artifact checks pass.
"""
from __future__ import annotations
import hashlib,json,os,signal,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
RUN=Path(os.environ.get("RDT_MARS_RUN_ROOT","/workspace/runs/rdt_mars")); ROOT=Path("/workspace/repos/before-we-act"); PY="/workspace/venvs/rdt/bin/python"; stop=False; active=None
def now(): return datetime.now(timezone.utc).isoformat()
def write(p,v): p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(".tmp"); t.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n"); os.replace(t,p)
def digest(cmd,env=None,gpus=()): return hashlib.sha256(json.dumps({"cmd":cmd,"env":env or {},"gpus":list(gpus)},sort_keys=True).encode()).hexdigest()
def done(name, d):
 p=RUN/"receipts"/(name+".json")
 try: return json.loads(p.read_text()).get("status")=="complete" and json.loads(p.read_text()).get("command_sha256")==d
 except: return False
def sig(_n,_f):
 global stop; stop=True
 if active:
  try: os.killpg(active.pid,signal.SIGTERM)
  except ProcessLookupError: pass
def stage(name,cmd,env=None,gpus=()):
 global active
 d=digest(cmd,env,gpus)
 if done(name,d): print(json.dumps({"event":"skip_complete","stage":name}),flush=True); return True
 e=os.environ.copy(); e.update({str(k):str(v) for k,v in (env or {}).items()});
 if gpus: e["CUDA_VISIBLE_DEVICES"]=",".join(map(str,gpus))
 log=RUN/"logs"/(name+".log"); log.parent.mkdir(parents=True,exist_ok=True); write(RUN/"state.json",{"stage":name,"status":"running","gpus":list(gpus),"log":str(log),"updated_at":now()})
 if name == "robofactory_assets": cwd = "/workspace/repos/RoboFactory"
 elif name in {"audit","statistics","language","smoke_train","formal_train","smoke_checkpoint_audit","formal_checkpoint_audit"}: cwd = "/workspace/repos/rdt-1b"
 else: cwd = str(ROOT)
 with log.open("ab",buffering=0) as f:
  active=subprocess.Popen(cmd,cwd=cwd,env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  code=active.wait()
 active=None
 if stop: return False
 if code: write(RUN/"state.json",{"stage":name,"status":"retrying","returncode":code,"updated_at":now()}); return False
 write(RUN/"receipts"/(name+".json"),{"schema":"mars-control.rdt.stage.v1","stage":name,"status":"complete","command_sha256":d,"completed_at":now(),"log":str(log)})
 print(json.dumps({"event":"stage_complete","stage":name}),flush=True); return True
def main():
 global stop
 RUN.mkdir(parents=True,exist_ok=True); signal.signal(signal.SIGTERM,sig); signal.signal(signal.SIGINT,sig)
 # Keep the official RDT checkout first: the benchmark repository also has a
 # top-level ``train`` package, which must never shadow RDT's trainer during
 # DeepSpeed launch.  The benchmark path remains available for deployment.*
 common={"HF_HOME":"/workspace/.hf_home","HUGGINGFACE_HUB_TOKEN":Path("/workspace/.secrets/hf_token").read_text().strip(),"RDT_MARS_DATASET":"/workspace/datasets/mars_control","PYTHONPATH":"/workspace/repos/rdt-1b:/workspace/repos/before-we-act"}
 stages=[
  ("download",[PY,str(ROOT/"deployment/rdt_mars/download.py")],{},()),
  ("configure",[PY,str(ROOT/"deployment/rdt_mars/configure.py")],{},()),
  ("audit",[PY,str(ROOT/"deployment/rdt_mars/audit.py")],common,()),
  ("statistics",[PY,"-m","data.compute_dataset_stat_hdf5","--save_path","configs/dataset_stat.json"],common,()),
  ("language",[PY,str(ROOT/"deployment/rdt_mars/prepare_lang_embeds.py")],common,(0,)),
  ("robofactory_assets",[PY,"script/download_assets.py"],dict(common,HF_TOKEN=common["HUGGINGFACE_HUB_TOKEN"]),()),
  ("smoke_train",["bash",str(ROOT/"deployment/rdt_mars/run_train.sh")],dict(common,RDT_RUN_NAME="smoke",RDT_MAX_TRAIN_STEPS="2",RDT_MICRO_BATCH="1",RDT_NUM_WORKERS="1",RDT_CHECKPOINT_PERIOD="1"),(0,1,2,3)),
  ("smoke_checkpoint_audit",[PY,str(ROOT/"deployment/rdt_mars/checkpoint_audit.py")],dict(common,RDT_AUDIT_CHECKPOINT="/workspace/runs/rdt_mars/smoke/checkpoints",RDT_AUDIT_OUTPUT="/workspace/runs/rdt_mars/smoke/checkpoint_audit.json"),(0,)),
  ("smoke_validation20",[PY,"-m","deployment.rdt_mars.run_validation","--checkpoint","/workspace/runs/rdt_mars/smoke/checkpoints","--output-root","/workspace/runs/rdt_mars/smoke/validation","--smoke"],common,(0,1,2,3)),
  ("formal_train",["bash",str(ROOT/"deployment/rdt_mars/run_train.sh")],dict(common,RDT_RUN_NAME="formal",RDT_MAX_TRAIN_STEPS=os.environ.get("RDT_MAX_TRAIN_STEPS","300000"),RDT_MICRO_BATCH=os.environ.get("RDT_MICRO_BATCH","4"),RDT_NUM_WORKERS=os.environ.get("RDT_NUM_WORKERS","4"),RDT_CHECKPOINT_PERIOD=os.environ.get("RDT_CHECKPOINT_PERIOD","2500")),(0,1,2,3)),
  ("formal_checkpoint_audit",[PY,str(ROOT/"deployment/rdt_mars/checkpoint_audit.py")],dict(common,RDT_AUDIT_CHECKPOINT="/workspace/runs/rdt_mars/formal/checkpoints",RDT_AUDIT_OUTPUT="/workspace/runs/rdt_mars/formal/checkpoint_audit.json"),(0,)),
  ("validation20",[PY,"-m","deployment.rdt_mars.run_validation","--checkpoint","/workspace/runs/rdt_mars/formal/checkpoints","--output-root","/workspace/runs/rdt_mars/formal/validation20"],common,(0,1,2,3)),
 ]
 while not stop:
  pending=False
  for name,cmd,env,gpus in stages:
   if stop: break
   if not stage(name,cmd,env,gpus): pending=True; break
  if stop: break
  if not pending: write(RUN/"state.json",{"stage":"complete","status":"complete","updated_at":now()}); time.sleep(60)
  else: time.sleep(15)
 write(RUN/"state.json",{"stage":"stopped" if stop else "complete","status":"stopped" if stop else "complete","updated_at":now()})
if __name__=="__main__": main()
