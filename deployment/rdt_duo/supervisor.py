"""Crash-resumable DuoBench RDT-1B supervisor.

Every stage has an immutable command receipt and owns a process group.  The
formal trainer is started only after data, normalization, model wiring, and
four-GPU smoke gates pass; validation starts only after a reload/forward audit.
"""
from __future__ import annotations
import hashlib, json, os, signal, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
from .protocol import FORMAL_STEPS, RDT_UPSTREAM_COMMIT, TASKS

ROOT=Path(os.environ.get("RDT_DUO_REPO","/workspace/repos/before-we-act")); RDT=Path(os.environ.get("RDT_DUO_UPSTREAM","/workspace/repos/rdt-1b")); DUO=Path(os.environ.get("RDT_DUO_DUOBENCH","/workspace/repos/duobench")); RCS=Path(os.environ.get("RDT_DUO_RCS","/workspace/repos/robot-control-stack")); DATASET=Path(os.environ.get("RDT_DUO_DATASET","/workspace/datasets/duobench")); RUN=Path(os.environ.get("RDT_DUO_RUN","/workspace/runs/rdt_duo")); DATA=RUN/"data"; LOG=RUN/"logs"; RECEIPTS=RUN/"receipts"; PY="/venv/main/bin/python"; active=None; stopping=False
def now(): return datetime.now(timezone.utc).isoformat()
def atomic(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)
def status(stage,state="running",**extra): atomic(RUN/"status.json",{"schema":"duobench.rdt.supervisor.v1","stage":stage,"state":state,"updated_at":now(),"gpu_schedule":"GPU0-3 exclusive for smoke/formal RDT; validation wave uses one isolated GPU per task","tasks":list(TASKS),**extra})
def command_hash(cmd,env): return hashlib.sha256(json.dumps({"cmd":cmd,"env":env},sort_keys=True).encode()).hexdigest()
def run(stage,cmd,env=None,gpus=(),retries=2,cwd=None):
    global active
    # Put upstream RDT first: before-we-act also contains a top-level `models`
    # package, while the untouched RDT loader imports `models.rdt_runner`.
    env=dict(env or {}); env.update({"PYTHONUNBUFFERED":"1","PYTHONPATH":f"{RDT}:{ROOT}:{DUO/'src'}:"+env.get("PYTHONPATH","")});
    if gpus: env["CUDA_VISIBLE_DEVICES"]=",".join(map(str,gpus))
    digest=command_hash(cmd,env); receipt=RECEIPTS/(stage+".json")
    if receipt.is_file():
        try:
            saved=json.loads(receipt.read_text())
            if saved.get("status")=="complete" and saved.get("command_sha256")==digest: return
        except Exception: pass
    LOG.mkdir(parents=True,exist_ok=True); RECEIPTS.mkdir(parents=True,exist_ok=True); workdir=str(cwd or ROOT)
    for attempt in range(1,retries+1):
        status(stage,attempt=attempt,command=cmd,gpus=list(gpus),log=str(LOG/(stage+".log")))
        with (LOG/(stage+".log")).open("ab",buffering=0) as out:
            out.write((json.dumps({"event":"launch","attempt":attempt,"time":now(),"command":cmd})+"\n").encode()); active=subprocess.Popen(cmd,cwd=workdir,env={**os.environ,**env},stdout=out,stderr=subprocess.STDOUT,start_new_session=True); rc=active.wait()
        active=None
        if rc==0:
            atomic(receipt,{"schema":"duobench.rdt.stage.v1","stage":stage,"status":"complete","command_sha256":digest,"command":cmd,"gpus":list(gpus),"completed_at":now()}); return
        status(stage,"retrying",attempt=attempt,returncode=rc)
        if attempt<retries: time.sleep(10*attempt)
    raise RuntimeError(f"{stage} failed after {retries} attempts")
def stop(_sig,_frame):
    global stopping; stopping=True
    if active:
        try: os.killpg(active.pid,signal.SIGTERM)
        except ProcessLookupError: pass
def exists(path): return Path(path).is_file()
def main():
    global stopping
    RUN.mkdir(parents=True,exist_ok=True); signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    token=Path("/workspace/.secrets/hf_token")
    # Keep benchmark and robot assets in distinct prefixes: merging them can
    # overwrite same-named MuJoCo materials. rcs.get_prefix uses its global
    # empty-world sentinel for both packages, so add only that sentinel to the
    # downloaded DuoBench asset tree.
    duo_assets=Path("/workspace/datasets/duobench_assets")
    common={"HF_HOME":"/workspace/.hf_home","HF_TOKEN":token.read_text().strip() if token.is_file() else "","HUGGINGFACE_HUB_TOKEN":token.read_text().strip() if token.is_file() else "","DUOBENCH_PREFIX":str(duo_assets),"RCS_PREFIX":"/root/.rcs","MUJOCO_GL":"egl","WANDB_MODE":"disabled","TOKENIZERS_PARALLELISM":"false"}
    try:
        status("bootstrap")
        for path in (RDT, DUO, RCS, DATASET): path.parent.mkdir(parents=True, exist_ok=True)
        # RDT's upstream hub mixin still uses the huggingface-hub 0.23 API.
        # Pin the upstream training stack and the NumPy-1 compatible image
        # stack so a resumed supervisor recreates the smoke-tested environment.
        run("deps",[PY,"-m","pip","install","-q","--no-deps","packaging==24.0","huggingface_hub==0.23.0","accelerate==0.30.1","diffusers==0.27.2","transformers==4.41.0","sentencepiece==0.2.0","deepspeed==0.19.6","timm==1.0.3","numpy==1.26.4","scipy==1.16.3","h5py==3.11.0","opencv-python-headless==4.10.0.84","imgaug==0.4.0","rcs-core==0.7.2"],env=common,gpus=(),retries=3,cwd=ROOT)
        if not (RDT/".git").is_dir(): run("rdt_clone",["git","clone", "https://github.com/thu-ml/RoboticsDiffusionTransformer.git",str(RDT)],env=common,cwd=ROOT)
        run("rdt_checkout",["git","-C",str(RDT),"checkout",RDT_UPSTREAM_COMMIT],env=common,cwd=ROOT)
        if not (DUO/".git").is_dir(): run("duobench_clone",["git","clone","https://github.com/RobotControlStack/duobench.git",str(DUO)],env=common,cwd=ROOT)
        run("duobench_checkout",["git","-C",str(DUO),"checkout","082a57cdafea9db115029e6fe9e03691e755f93f"],env=common,cwd=ROOT)
        if not (RCS/".git").is_dir(): run("rcs_clone",["git","clone","https://github.com/RobotControlStack/robot-control-stack.git",str(RCS)],env=common,cwd=ROOT)
        run("rcs_checkout",["git","-C",str(RCS),"checkout","4f78aeffae3bc4d0c02e7beab993e5406261dcf6"],env=common,cwd=ROOT)
        # Its runtime dependencies are installed above; do not let an editable
        # reinstall silently upgrade the pinned NumPy/RDT compatibility stack.
        run("duobench_install",[PY,"-m","pip","install","-q","--no-deps","-e",str(DUO)],env=common,cwd=ROOT,retries=3)
        run("assets",["bash","-lc",f"mkdir -p {duo_assets}/assets/scenes/empty_world; cp -a {DUO}/assets/. {duo_assets}/assets/; cp {RCS}/assets/scenes/empty_world/scene.xml {duo_assets}/assets/scenes/empty_world/scene.xml"],env=common,cwd=ROOT)
        if not (DATASET/"download_receipt.json").is_file(): run("download",[PY,"-m","deployment.rdt_duo.download","--output",str(DATASET),"--workers","16"],env=common,cwd=ROOT,retries=3)
        if not (DATA/"manifest.json").is_file(): run("prepare",[PY,"-m","deployment.rdt_duo.prepare","--dataset",str(DATASET),"--output",str(DATA),"--image-size","224","--jobs","6"],env=common,cwd=ROOT,retries=2)
        run("configure",[PY,"-m","deployment.rdt_duo.configure"],env=common,cwd=ROOT)
        run("audit",[PY,"-m","deployment.rdt_duo.audit","--data",str(DATA),"--rdt",str(RDT),"--output",str(RUN/"audit.json")],env={**common,"RDT_DUOBENCH_DATA":str(DATA)},cwd=ROOT)
        run("statistics",[PY,"-m","data.compute_dataset_stat_hdf5","--save_path","configs/dataset_stat.json"],env={**common,"RDT_DUOBENCH_DATA":str(DATA)},cwd=RDT)
        run("language",[PY,"-m","deployment.rdt_duo.prepare_lang_embeds","--data",str(DATA),"--device","cuda:0"],env={**common,"PYTHONPATH":f"{RDT}:{ROOT}:{DUO/'src'}"},gpus=(0,),cwd=RDT)
        smoke=RUN/"smoke"; run("smoke_train",["bash",str(ROOT/"deployment/rdt_duo/run_train.sh")],env={**common,"RDT_DUO_RUN_NAME":"smoke","RDT_DUO_MAX_TRAIN_STEPS":"2","RDT_DUO_CHECKPOINT_PERIOD":"1","RDT_DUO_MICRO_BATCH":"1","RDT_DUO_NUM_WORKERS":"1"},gpus=(0,1,2,3),cwd=ROOT)
        run("smoke_checkpoint_audit",[PY,"-m","deployment.rdt_duo.checkpoint_audit","--checkpoint",str(smoke/"checkpoints"),"--output",str(smoke/"checkpoint_audit.json")],env=common,gpus=(0,),cwd=RDT)
        # Smoke each task serially on a single isolated GPU. This exercises all
        # task XMLs without concurrent MuJoCo composer state while keeping the
        # formal validation launcher fully wave-parallel.
        run("smoke_validation",[PY,"-m","deployment.rdt_duo.validation_launcher","--checkpoint",str(smoke/"checkpoints"),"--data",str(DATA),"--output",str(smoke/"validation"),"--episodes","1","--workers","1","--max-steps","2","--smoke"],env={**common,"PYTHONPATH":f"{RDT}:{ROOT}:{DUO/'src'}"},gpus=(0,),cwd=RDT)
        run("formal_train",["bash",str(ROOT/"deployment/rdt_duo/run_train.sh")],env={**common,"RDT_DUO_RUN_NAME":"formal","RDT_DUO_MAX_TRAIN_STEPS":os.environ.get("RDT_DUO_MAX_TRAIN_STEPS",str(FORMAL_STEPS)),"RDT_DUO_MICRO_BATCH":os.environ.get("RDT_DUO_MICRO_BATCH","4"),"RDT_DUO_NUM_WORKERS":os.environ.get("RDT_DUO_NUM_WORKERS","8"),"RDT_DUO_CHECKPOINT_PERIOD":os.environ.get("RDT_DUO_CHECKPOINT_PERIOD","2500")},gpus=(0,1,2,3),cwd=ROOT,retries=3)
        formal=RUN/"formal"; run("formal_checkpoint_audit",[PY,"-m","deployment.rdt_duo.checkpoint_audit","--checkpoint",str(formal/"checkpoints"),"--output",str(formal/"checkpoint_audit.json")],env=common,gpus=(0,),cwd=RDT)
        run("validation20",[PY,"-m","deployment.rdt_duo.validation_launcher","--checkpoint",str(formal/"checkpoints"),"--data",str(DATA),"--output",str(formal/"validation20"),"--episodes","20","--workers","4"],env={**common,"PYTHONPATH":f"{RDT}:{ROOT}:{DUO/'src'}"},gpus=(0,1,2,3),cwd=RDT,retries=3)
        status("complete","complete",checkpoint=str(formal/"checkpoints"),summary=str(formal/"validation20/summary.json"))
    except Exception as error:
        status("failed","failed",error=repr(error),traceback=traceback.format_exc()); raise
if __name__=="__main__": main()
