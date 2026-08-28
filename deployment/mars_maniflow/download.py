#!/usr/bin/env python3
"""Download the four pinned MARS-Control corpora with resumable HF transfers."""
from __future__ import annotations
import argparse, os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from .common import DATASET_REPOS, atomic_json, sha256

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,default=Path('/workspace/datasets/mars_control')); p.add_argument('--token',default=os.getenv('HF_TOKEN')); p.add_argument('--token-file',type=Path,default=Path('/workspace/.secrets/hf_token')); a=p.parse_args()
    token=(a.token or (a.token_file.read_text().strip() if a.token_file.exists() else ''))
    if not token: raise RuntimeError('HF token missing; set HF_TOKEN or --token-file')
    api=HfApi(token=token)
    for task,(repo,rev) in DATASET_REPOS.items():
        out=a.data_root/task; out.mkdir(parents=True,exist_ok=True); info=api.dataset_info(repo,revision=rev,files_metadata=True)
        if info.sha!=rev: raise RuntimeError(f'{task}: revision drift {info.sha}')
        siblings={x.rfilename:x for x in info.siblings}; names=sorted(n for n in siblings if n.startswith('motionplanning/') and n.endswith('.h5') and '/.' not in n)
        if len(names)!=10: raise RuntimeError(f'{task}: expected 10 HDF5 shards, found {len(names)}')
        rows=[]
        for name in names:
            item=siblings[name]; local=Path(hf_hub_download(repo,name,repo_type='dataset',revision=rev,local_dir=str(out),token=token))
            if local.stat().st_size!=int(item.size or 0): raise RuntimeError(f'{task}: size mismatch {name}')
            digest=sha256(local)
            if item.lfs and item.lfs.sha256 and digest!=item.lfs.sha256: raise RuntimeError(f'{task}: sha256 mismatch {name}')
            side=name[:-3]+'json'
            if side in siblings: hf_hub_download(repo,side,repo_type='dataset',revision=rev,local_dir=str(out),token=token)
            rows.append({'path':name,'bytes':local.stat().st_size,'sha256':digest})
        atomic_json(out/'download_receipt.json',{'schema':'mars-control.maniflow.dataset.v1','status':'complete','task':task,'repo_id':repo,'revision':rev,'formal_shards':rows,'formal_episodes':150,'bytes_total':sum(x['bytes'] for x in rows),'training_policy':'all_data_no_split'})
        print({'task':task,'shards':len(rows),'bytes':sum(x['bytes'] for x in rows)},flush=True)
if __name__=='__main__': main()
