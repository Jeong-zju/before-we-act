from __future__ import annotations
import json, os, signal, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
from .common import atomic_json, TASKS
ROOT=Path(os.environ.get("MARS_DP_REPO","/workspace/repos/before-we-act")); RF=Path(os.environ.get("MARS_DP_ROBOFACTORY","/workspace/repos/RoboFactory")); DATA=Path(os.environ.get("MARS_DP_DATA_ROOT","/workspace/datasets/mars_control")); RUN=Path(os.environ.get("MARS_DP_RUN_ROOT","/workspace/runs/mars_dp_v2")); PY=os.environ.get("MARS_DP_PYTHON","/venv/main/bin/python"); active=None; stopping=False
def now(): return datetime.now(timezone.utc).isoformat()
def state(stage,status="running",**kw): atomic_json(RUN/"state.json",{"schema":"mars-control.dp.supervisor.v2","stage":stage,"status":status,"updated_at":now(),**kw})
def finished(name):
    p=RUN/"receipts"/f"{name}.json"
    if not p.is_file(): return False
    try: return json.loads(p.read_text()).get("status")=="complete"
    except Exception:return False
def run(name,cmd):
    global active; log=RUN/"logs"/f"{name}.log"; log.parent.mkdir(parents=True,exist_ok=True); env=os.environ.copy(); env.update({"PYTHONPATH":f"{ROOT}:{RF}/robofactory/policy/Diffusion-Policy", "MARS_DP_DATA_ROOT":str(DATA),"MARS_DP_RUN_ROOT":str(RUN),"CUDA_VISIBLE_DEVICES":"0","TOKENIZERS_PARALLELISM":"false","WANDB_MODE":"offline"}); state(name,command=cmd,log=str(log),gpu=[0]);
    with log.open("ab",buffering=0) as stream: active=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True); code=active.wait()
    active=None
    if code: raise RuntimeError(f"{name} exited {code}")
    atomic_json(RUN/"receipts"/f"{name}.json",{"schema":"mars-control.dp.stage.v1","stage":name,"status":"complete","completed_at":now(),"log":str(log)})
def stop(_s,_f):
    global stopping; stopping=True
    if active:
        try: os.killpg(active.pid,signal.SIGTERM)
        except ProcessLookupError: pass
def main():
    global stopping; RUN.mkdir(parents=True,exist_ok=True); signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    stages=[("preflight",[PY,"-c","import torch,h5py,diffusers; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(torch.cuda.get_device_name(0))"]),("audit",[PY,"-m","deployment.mars_dp.audit"]),("contract_test",[PY,"-m","deployment.mars_dp.preflight_v2"]),("smoke_train",[PY,"-m","deployment.mars_dp.train","--data-root",str(DATA),"--output",str(RUN/"smoke"),"--smoke","--workers","0"]),("smoke_eval",[PY,"-m","deployment.mars_dp.run_validation","--checkpoint",str(RUN/"smoke/last.pt"),"--output-root",str(RUN/"smoke/validation"),"--robofactory-root",str(RF),"--smoke"]),("formal_train",[PY,"-m","deployment.mars_dp.train","--data-root",str(DATA),"--output",str(RUN/"formal"),"--steps","60000","--batch-size","64","--workers","16","--resume"]),("validation20",[PY,"-m","deployment.mars_dp.run_validation","--checkpoint",str(RUN/"formal/last.pt"),"--output-root",str(RUN/"validation20"),"--robofactory-root",str(RF)]),("finalize",[PY,"-m","deployment.mars_dp.finalize"])]
    for name,cmd in stages:
        if stopping: break
        if finished(name): continue
        while not stopping:
            try: run(name,cmd); break
            except Exception as exc: state(name,"retrying",error=repr(exc)); time.sleep(15)
    state("stopped" if stopping else "complete","stopped" if stopping else "complete")
if __name__=="__main__": main()
