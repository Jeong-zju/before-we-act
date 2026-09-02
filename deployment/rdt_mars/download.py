#!/usr/bin/env python3
import concurrent.futures,hashlib,json,os,tempfile
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
ROOT=Path(os.environ.get("RDT_MARS_DATASET","/workspace/datasets/mars_control")); TOKEN=Path("/workspace/.secrets/hf_token")
REPOS={"place_cube_in_cup":("Jeong-zju/mars-control-place-cube-in-cup-rf","3878150bec8f4830e1a57a01a13762a10abc8d52"),"strike_cube_hard":("Jeong-zju/mars-control-strike-cube-hard-rf","bc7051cb0560058bf426e792871faa1ca8a4f78f"),"three_robots_place_shoes":("Jeong-zju/mars-control-three-robots-place-shoes-rf","ad231c7eff530f71f0c5302b6c03c7164bbcc896"),"four_robots_stack_cube":("Jeong-zju/mars-control-four-robots-stack-cube-rf","3fa4833f5e34c3565da04af99c62d516e048fcfc")}
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=p.parent)
 with os.fdopen(fd,"w") as f: json.dump(v,f,indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
token=TOKEN.read_text().strip(); api=HfApi(token=token); plans={}; jobs=[]
for task,(repo,rev) in REPOS.items():
 out=ROOT/task; out.mkdir(parents=True,exist_ok=True); info=api.dataset_info(repo,revision=rev,files_metadata=True)
 if info.sha!=rev: raise RuntimeError(f"{task}: revision drift {info.sha}")
 files={s.rfilename:s for s in info.siblings if s.rfilename.startswith("motionplanning/") and "/." not in s.rfilename and s.rfilename.endswith(".h5")}
 if len(files)!=10: raise RuntimeError(f"{task}: expected 10 shards, got {len(files)}")
 plans[task]=(repo,rev,out,files); jobs.extend((task,name) for name in sorted(files))
def fetch(job):
 task,name=job; repo,rev,out,_=plans[task]; return hf_hub_download(repo,name,repo_type="dataset",revision=rev,local_dir=str(out),token=token)
with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get("RDT_HF_WORKERS","8"))) as pool:
 for path in pool.map(fetch,jobs): print(json.dumps({"downloaded":path}),flush=True)
total=0
for task,(repo,rev,out,files) in plans.items():
 rows=[]
 for name in sorted(files):
  p=out/name; h=hashlib.sha256()
  with p.open("rb") as f:
   for b in iter(lambda:f.read(16*1024*1024),b""): h.update(b)
  meta=files[name]; expected=meta.lfs.sha256 if meta.lfs else None
  if expected and h.hexdigest()!=expected: raise RuntimeError(f"{task}: sha mismatch {name}")
  rows.append({"path":name,"size_bytes":p.stat().st_size,"sha256":h.hexdigest()}); total+=p.stat().st_size
 atomic(out/"download_receipt.json",{"schema":"mars-control.rdt.dataset.v1","status":"complete","task":task,"repo_id":repo,"revision":rev,"shards":10,"formal_episodes":150,"formal_shards":rows,"training_policy":"all_data_no_split"}); print(json.dumps({"task":task,"status":"complete"}),flush=True)
print(json.dumps({"status":"complete","bytes":total}))
