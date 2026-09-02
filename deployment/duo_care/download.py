from __future__ import annotations
import argparse,json,os
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--revision',default='b741bc915d942ecadaefb4e3de6bbd716c1b8b1b'); a=p.parse_args(); a.output.parent.mkdir(parents=True,exist_ok=True); token=os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
 if not token: raise RuntimeError('HF_TOKEN is not configured in /workspace/.env')
 path=snapshot_download(repo_id='RobotControlStack/duobench',repo_type='dataset',revision=a.revision,local_dir=a.output,token=token,max_workers=12); print(json.dumps({'status':'complete','dataset':path,'revision':a.revision}))
if __name__=='__main__':main()
