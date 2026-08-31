#!/usr/bin/env python3
import concurrent.futures,hashlib,json,os,time
from pathlib import Path
from huggingface_hub import hf_hub_download
from .common import REPOS,atomic_json
ROOT=Path(os.environ.get("OPENPI_MARS_CONTROL_ROOT","/workspace/datasets/mars_control")); TOKEN=Path("/workspace/.secrets/hf_token").read_text().strip()
plans={}; jobs=[]
for task,(repo,rev) in REPOS.items():
 out=ROOT/task; out.mkdir(parents=True,exist_ok=True)
 files={f"motionplanning/{task}.shard{i:02d}.h5":None for i in range(10)}
 plans[task]=(repo,rev,out,files); jobs.extend((task,n) for n in sorted(files))
def fetch(job):
 task,name=job; repo,rev,out,_=plans[task]
 for attempt in range(8):
  try: return hf_hub_download(repo,name,repo_type="dataset",revision=rev,local_dir=str(out),token=TOKEN,force_download=False)
  except Exception:
   if attempt==7: raise
   time.sleep(min(60,2**attempt))
with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get("HF_WORKERS","12"))) as pool:
 for path in pool.map(fetch,jobs): print(json.dumps({"downloaded":path}),flush=True)
for task,(repo,rev,out,files) in plans.items():
 rows=[]
 for name,meta in sorted(files.items()):
  p=out/name; h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(16*1024*1024),b''): h.update(b)
  if p.stat().st_size < 1024 or h.hexdigest() == "0"*64: raise RuntimeError(f"{task}: invalid object {name}")
  rows.append({"path":name,"bytes":p.stat().st_size,"sha256":h.hexdigest()})
 atomic_json(out/"download_receipt.json",{"schema":"mars-control.pi05.dataset.v1","status":"complete","task":task,"repo_id":repo,"revision":rev,"formal_shards":rows,"formal_episodes":150,"training_policy":"all_data_no_split"})
