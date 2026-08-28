from __future__ import annotations
import bisect, glob, json
from pathlib import Path
import h5py, numpy as np, torch
from torch.utils.data import Dataset, Sampler
from .common import TASKS, atomic_json

def index_corpus(root, stats_path=None):
    root=Path(root); streams=[[] for _ in TASKS]; qmin=qmax=amin=amax=None; qsum=asum=qsq=asq=None; episodes=local_streams=timesteps=0
    for tid,spec in enumerate(TASKS):
        paths=sorted(glob.glob(str(root/spec.name/'motionplanning'/'*.shard*.h5')))
        if len(paths)!=10: raise RuntimeError(f'{spec.name}: expected 10 shards, found {len(paths)}')
        task_eps=0
        for path in paths:
            with h5py.File(path,'r') as f:
                for tr in sorted(f,key=lambda n:int(n.rsplit('_',1)[-1])):
                    g=f[tr]
                    if not bool(np.asarray(g['success'])[-1]): raise RuntimeError(f'non-success trajectory {path}:{tr}')
                    n=min(len(g[f'actions/panda-{a}']) for a in range(spec.arms)); task_eps+=1; episodes+=1
                    for arm in range(spec.arms):
                        q=np.asarray(g[f'obs/agent/panda-{arm}/qpos'][:n],np.float64); x=np.asarray(g[f'actions/panda-{arm}'][:n],np.float64); im=g[f'obs/sensor_data/head_camera_agent{arm}/rgb']
                        if q.shape!=(n,9) or x.shape!=(n,8) or im.shape[0]<n or tuple(im.shape[1:])!=(240,320,3) or im.dtype!=np.uint8: raise RuntimeError(f'local contract mismatch {path}:{tr}:arm{arm}')
                        streams[tid].append((path,tr,arm,n)); local_streams+=1; timesteps+=n
                        qsum=q.sum(0) if qsum is None else qsum+q.sum(0); qsq=np.square(q).sum(0) if qsq is None else qsq+np.square(q).sum(0); asum=x.sum(0) if asum is None else asum+x.sum(0); asq=np.square(x).sum(0) if asq is None else asq+np.square(x).sum(0)
                        qmin=q.min(0) if qmin is None else np.minimum(qmin,q.min(0)); qmax=q.max(0) if qmax is None else np.maximum(qmax,q.max(0)); amin=x.min(0) if amin is None else np.minimum(amin,x.min(0)); amax=x.max(0) if amax is None else np.maximum(amax,x.max(0))
        if task_eps!=150: raise RuntimeError(f'{spec.name}: expected 150 episodes, found {task_eps}')
    if episodes!=600 or local_streams!=1650: raise RuntimeError(f'corpus count drift episodes={episodes}, streams={local_streams}')
    qm=qsum/timesteps; am=asum/timesteps; qs=np.sqrt(np.maximum(qsq/timesteps-qm**2,0)).clip(1e-4); ats=np.sqrt(np.maximum(asq/timesteps-am**2,0)).clip(1e-4)
    stats={'schema':'mars-control.maniflow.normalization.v1','status':'complete','episodes':episodes,'local_streams':local_streams,'indexed_local_timesteps':timesteps,'all_data_no_split':True,'qpos':{'mean':qm.tolist(),'std':qs.tolist(),'min':qmin.tolist(),'max':qmax.tolist()},'action':{'mean':am.tolist(),'std':ats.tolist(),'min':amin.tolist(),'max':amax.tolist()},'rgb':{'dtype':'uint8','shape_hwc':[240,320,3],'transform':'float32_div_255_then_resize_224'}}
    if stats_path: atomic_json(stats_path,stats)
    return streams,stats

class MarsManiFlowDataset(Dataset):
    def __init__(self,root,stats_path,obs_steps=2,horizon=16,image_size=224):
        self.obs_steps=int(obs_steps); self.horizon=int(horizon); self.image_size=int(image_size); self.handles={}; self.streams,self.stats=index_corpus(root,stats_path); self.entries=[]; self.task_indices=[[] for _ in TASKS]
        for tid,rows in enumerate(self.streams):
            for path,tr,arm,n in rows:
                for t in range(n): self.task_indices[tid].append(len(self.entries)); self.entries.append((tid,path,tr,arm,n,t))
        self.qmin=np.asarray(self.stats['qpos']['min'],np.float32); self.qmax=np.asarray(self.stats['qpos']['max'],np.float32); self.amin=np.asarray(self.stats['action']['min'],np.float32); self.amax=np.asarray(self.stats['action']['max'],np.float32)
    def __getstate__(self): s=dict(self.__dict__); s['handles']={}; return s
    def __len__(self): return len(self.entries)
    def _handle(self,path):
        if path not in self.handles:self.handles[path]=h5py.File(path,'r',libver='latest',swmr=True)
        return self.handles[path]
    def __getitem__(self,i):
        tid,path,tr,arm,n,t=self.entries[i]; g=self._handle(path)[tr]; obs0=max(0,t-self.obs_steps+1); obs1=t+1
        ims=np.asarray(g[f'obs/sensor_data/head_camera_agent{arm}/rgb'][obs0:obs1],np.uint8); q=np.asarray(g[f'obs/agent/panda-{arm}/qpos'][obs0:obs1],np.float32); ims=np.concatenate([np.repeat(ims[:1],self.obs_steps-len(ims),0),ims],0) if len(ims)<self.obs_steps else ims; q=np.concatenate([np.repeat(q[:1],self.obs_steps-len(q),0),q],0) if len(q)<self.obs_steps else q
        valid=min(self.horizon,n-t); x=np.asarray(g[f'actions/panda-{arm}'][t:t+valid],np.float32); x=np.concatenate([x,np.repeat(x[-1:],self.horizon-len(x),0)],0) if len(x)<self.horizon else x
        def norm(v,lo,hi): return np.clip(2*(v-lo)/(hi-lo+1e-6)-1,-1,1)
        return {'obs':{'head_cam':torch.from_numpy(ims.copy()),'agent_pos':torch.from_numpy(norm(q,self.qmin,self.qmax))},'action':torch.from_numpy(norm(x,self.amin,self.amax)),'task':TASKS[tid].name}

class TaskBalancedBatchSampler(Sampler):
    def __init__(self,indices,batch_size,updates,seed=20260828):
        if batch_size%len(indices): raise ValueError('batch size must divide four tasks')
        self.indices=indices; self.batch_size=batch_size; self.updates=updates; self.seed=seed; self.epoch=0
    def __len__(self): return self.updates
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch); self.epoch+=1; k=self.batch_size//len(self.indices)
        for _ in range(self.updates):
            b=np.concatenate([rng.choice(rows,k,replace=True) for rows in self.indices]); rng.shuffle(b); yield b.tolist()
