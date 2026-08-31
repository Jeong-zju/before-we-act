#!/usr/bin/env python3
from __future__ import annotations
import argparse,os,pickle,socket,struct,sys
from pathlib import Path
import numpy as np
def ex(c,n):
 b=[]
 while n:
  x=c.recv(n)
  if not x: raise EOFError
  b.append(x); n-=len(x)
 return b''.join(b)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--socket',required=True); a=p.parse_args(); sys.path.insert(0,'/workspace/repos/openpi'); os.chdir('/workspace/repos/openpi')
 from openpi.policies import policy_config
 from openpi.training import config
 policy=policy_config.create_trained_policy(config.get_config('pi05_mars_control_lora'),a.checkpoint)
 path=Path(a.socket); path.unlink(missing_ok=True); s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); s.bind(str(path)); os.chmod(path,0o600); s.listen(4); running=True
 while running:
  c,_=s.accept()
  with c:
   try:
    n=struct.unpack('!Q',ex(c,8))[0]; r=pickle.loads(ex(c,n)); op=r.get('op')
    if op=='ping': out={'ok':True}
    elif op=='reset': out={'ok':True}
    elif op=='shutdown': out={'ok':True}; running=False
    elif op=='infer':
     o=r['observation']; image=np.asarray(o['image']); state=np.asarray(o['state'],np.float32)
     if set(o)!={'image','state','prompt'} or image.ndim!=3 or image.shape[-1]!=3 or image.dtype!=np.uint8 or state.shape!=(9,) or not np.isfinite(state).all(): raise ValueError('strict local observation contract failed')
     chunk=np.asarray(policy.infer({'image':image,'state':state,'prompt':o['prompt']})['actions'],np.float32)
     if chunk.ndim!=2 or chunk.shape[1]!=8 or not np.isfinite(chunk).all(): raise ValueError('action chunk contract failed')
     out={'ok':True,'chunk':chunk}
    else: raise ValueError(f'unknown op {op}')
   except Exception as e: out={'ok':False,'error':f'{type(e).__name__}: {e}'}
   q=pickle.dumps(out,protocol=5); c.sendall(struct.pack('!Q',len(q))+q)
 s.close(); path.unlink(missing_ok=True)
if __name__=='__main__': main()
