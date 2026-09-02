from __future__ import annotations
import json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import Dataset, Sampler
from .action_contract import encode_anchor_relative_chunk
from .prepare import TASKS

class DuoLocalDataset(Dataset):
    def __init__(self, root:Path, normalization:Path, horizon=16, history=16, image_size=224):
        self.root=Path(root); self.norm=json.loads(Path(normalization).read_text()); self.norm=self.norm.get('normalization',self.norm); self.horizon=horizon; self.history=history; self.image_size=image_size
        self.qm=np.asarray(self.norm['qpos_mean'],np.float32); self.qs=np.asarray(self.norm['qpos_std'],np.float32); self.am=np.asarray(self.norm['action_mean'],np.float32); self.ass=np.asarray(self.norm['action_std'],np.float32)
        self.task_data=[]; self.rows=[]; self.task_rows=[[] for _ in TASKS]
        for tid,t in enumerate(TASKS):
            d={name:np.load(self.root/t/f'{name}.npy',mmap_mode='r') for name in ('state','action','head','left','right','episodes')}; episodes=np.asarray(d['episodes']); changes=np.r_[True,episodes[1:]!=episodes[:-1]]; starts=np.flatnonzero(changes); ends=np.r_[starts[1:],len(episodes)]; d['episode_start']=np.repeat(starts,ends-starts); d['episode_end']=np.repeat(ends,ends-starts); self.task_data.append(d); n=len(d['state'])
            for step in range(n): self.rows.extend([(tid,arm,step) for arm in (0,1)]); self.task_rows[tid].extend(range(len(self.rows)-2,len(self.rows)))
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        tid,arm,step=self.rows[i]; d=self.task_data[tid]; q=d['state'].reshape(-1,2,8)[:,arm]; u=d['action'].reshape(-1,2,8)[:,arm]; episode_start=int(d['episode_start'][step]); episode_end=int(d['episode_end'][step]); end=min(episode_end,step+self.horizon); act=encode_anchor_relative_chunk(u[step:end],q[step])
        if len(act)<self.horizon: act=np.concatenate([act,np.repeat(act[-1:],self.horizon-len(act),0)])
        # head and the arm-local wrist camera, packed side by side into RGB.
        image=np.concatenate([d['head'][step], d['left' if arm==0 else 'right'][step]],axis=1)
        # Each view was independently resized during preparation.  Keep the
        # [head|own-wrist] geometry instead of squeezing two views to a square.
        image=torch.from_numpy(image.copy()).permute(2,0,1).float().div_(255)
        hist=np.zeros((self.history,16),np.float32); mask=np.zeros(self.history,np.float32); begin=max(episode_start,step-self.history+1); states=q[begin:step+1]
        hist[-len(states):,:8]=(states-self.qm)/self.qs; mask[-len(states):]=1
        for j,s in enumerate(range(begin,step+1)):
            if s>episode_start: prev=u[s-1].copy(); prev[:7]-=q[s-1,:7]; hist[-len(states)+j,8:]=(prev-self.am)/self.ass
        future=q[min(step+self.horizon,episode_end-1)]
        return {'image':image,'qpos':torch.from_numpy((q[step]-self.qm)/self.qs),'actions':torch.from_numpy((act-self.am)/self.ass),'future_delta':torch.from_numpy((future-q[step])/self.qs),'history':torch.from_numpy(hist),'history_mask':torch.from_numpy(mask),'task_id':torch.tensor(tid)}

class TaskSampler(Sampler[int]):
    def __init__(self,ds,rank=0,replicas=1,seed=20260829): self.ds=ds; self.seed=seed; self.rank=rank; self.replicas=replicas; self.epoch=0; self.n=max(1,len(ds)//replicas)
    def __len__(self): return self.n
    def set_epoch(self,e): self.epoch=e
    def __iter__(self):
        g=torch.Generator().manual_seed(self.seed+self.epoch*self.replicas+self.rank); tasks=torch.randint(len(self.ds.task_rows),(self.n,),generator=g)
        return iter([self.ds.task_rows[t][int(torch.randint(len(self.ds.task_rows[t]),(1,),generator=g))] for t in tasks.tolist()])
