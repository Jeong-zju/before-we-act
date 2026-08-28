#!/usr/bin/env python3
"""Download and verify the four pinned MARS-Control HF corpora."""
import hashlib, json, os, tempfile
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

ROOT=Path(os.environ.get("MARS_ACT_DATA_ROOT","/workspace/datasets/mars_control"))
TOKEN=Path(os.environ.get("HF_TOKEN_FILE","/workspace/.secrets/hf_token"))
REPOS={
 "place_cube_in_cup":("Jeong-zju/mars-control-place-cube-in-cup-rf","3878150bec8f4830e1a57a01a13762a10abc8d52"),
 "strike_cube_hard":("Jeong-zju/mars-control-strike-cube-hard-rf","bc7051cb0560058bf426e792871faa1ca8a4f78f"),
 "three_robots_place_shoes":("Jeong-zju/mars-control-three-robots-place-shoes-rf","ad231c7eff530f71f0c5302b6c03c7164bbcc896"),
 "four_robots_stack_cube":("Jeong-zju/mars-control-four-robots-stack-cube-rf","3fa4833f5e34c3565da04af99c62d516e048fcfc"),
}
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=p.parent,prefix=p.name+'.');
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def main():
 token=TOKEN.read_text().strip(); api=HfApi(token=token); total=0
 for task,(repo,rev) in REPOS.items():
  out=ROOT/task; out.mkdir(parents=True,exist_ok=True); info=api.dataset_info(repo,revision=rev,files_metadata=True)
  if info.sha!=rev: raise RuntimeError(f'{task}: revision drift {info.sha}')
  files={s.rfilename:{'size':int(s.size or 0),'sha256':s.lfs.sha256 if s.lfs else None} for s in info.siblings if s.rfilename.startswith('motionplanning/') and '/.' not in s.rfilename and s.rfilename.endswith('.h5')}
  if len(files)!=10: raise RuntimeError(f'{task}: expected 10 formal shards, got {len(files)}')
  # Download only the ten promoted shard objects.  A revision also contains
  # hundreds of failed/timeout `.parts` fragments; a broad glob would fetch
  # those and silently contaminate the training pool.
  for name in sorted(files):
   hf_hub_download(repo,name,repo_type='dataset',revision=rev,local_dir=str(out),token=token)
  for name in sorted(files):
   sidecar=name[:-3]+'json'
   if any(s.rfilename==sidecar for s in info.siblings):
    hf_hub_download(repo,sidecar,repo_type='dataset',revision=rev,local_dir=str(out),token=token)
  rows=[]
  for name,meta in sorted(files.items()):
   p=out/name
   if p.stat().st_size!=meta['size']: raise RuntimeError(f'{task}: size mismatch {name}')
   h=hashlib.sha256();
   with p.open('rb') as f:
    for b in iter(lambda:f.read(16*1024*1024),b''): h.update(b)
   if meta['sha256'] and h.hexdigest()!=meta['sha256']: raise RuntimeError(f'{task}: sha mismatch {name}')
   rows.append({'path':name,'size_bytes':meta['size'],'sha256':h.hexdigest()}); total+=meta['size']
  atomic(out/'download_receipt.json',{'schema':'mars-control.act.dataset.v1','task':task,'repo_id':repo,'revision':rev,'formal_shards':rows,'formal_episodes':150,'status':'complete','training_policy':'all_data_no_split'})
  print(json.dumps({'task':task,'status':'complete','shards':10,'bytes':sum(x['size_bytes'] for x in rows)}),flush=True)
 print(json.dumps({'four_task_bytes':total,'status':'complete'}))
if __name__=='__main__': main()
