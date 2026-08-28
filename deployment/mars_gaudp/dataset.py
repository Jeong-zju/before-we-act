from __future__ import annotations
import glob, json, os
from pathlib import Path
import h5py, numpy as np, torch
from torch.utils.data import Dataset, Sampler
from .common import TASKS, ARMS, ACTION_LOW, ACTION_HIGH

OBS_STEPS, HORIZON = 3, 8

def _limits(low, high):
    from model.common.normalizer import SingleFieldLinearNormalizer
    low, high = np.asarray(low, np.float32), np.asarray(high, np.float32)
    span = np.maximum(high - low, 1e-4)
    return SingleFieldLinearNormalizer.create_manual(2.0 / span, -1.0 - 2.0 * low / span,
        {"min": low, "max": high, "mean": (low + high) / 2, "std": span / np.sqrt(12.0)})

class MarsGauDPDataset(Dataset):
    """All successful demonstrations, indexed as independent local arm streams."""
    def __init__(self, root, cache_root, stats_path, obs_steps=OBS_STEPS, horizon=HORIZON, gaussian_hw=(30,40)):
        self.root, self.cache_root = Path(root), Path(cache_root); self.obs_steps, self.horizon = int(obs_steps), int(horizon)
        self.gaussian_hw = tuple(gaussian_hw); self.handles = {}; self.cache_handles = {}
        self.entries, self.task_indices, self.stats = [], [[] for _ in TASKS], {"q_min":None,"q_max":None,"a_min":None,"a_max":None}
        self.streams = []
        meta = json.loads((self.cache_root / "metadata.json").read_text())
        for tid, task in enumerate(TASKS):
            paths = sorted(glob.glob(str(self.root / task / "motionplanning" / "*.shard*.h5")))
            if len(paths) != 10: raise RuntimeError(f"{task}: expected 10 shards, got {len(paths)}")
            task_rows = [r for r in meta["tasks"][task]]
            if len(task_rows) != 150 and os.environ.get("MARS_GAUDP_ALLOW_PARTIAL_CACHE") != "1": raise RuntimeError(f"{task}: cache metadata episode count drift")
            for path in paths:
                with h5py.File(path, "r") as f:
                    for traj in sorted(f, key=lambda x:int(x.rsplit("_",1)[-1])):
                        g=f[traj]; n=min(len(g[f"actions/panda-{a}"]) for a in range(ARMS[task]))
                        if not bool(np.asarray(g["success"])[-1]): raise RuntimeError(f"failed trajectory {path}:{traj}")
                        row = next((x for x in task_rows if x["shard"] == Path(path).name and x["trajectory"] == traj), None)
                        if row is None:
                            if os.environ.get("MARS_GAUDP_ALLOW_PARTIAL_CACHE") == "1":
                                continue
                            raise RuntimeError(f"missing cache row {task}:{path}:{traj}")
                        for arm in range(ARMS[task]):
                            q=np.asarray(g[f"obs/agent/panda-{arm}/qpos"][:n],np.float32); a=np.clip(np.asarray(g[f"actions/panda-{arm}"][:n],np.float32),ACTION_LOW,ACTION_HIGH)
                            self._update_stats(q,a); stream=(path,traj,arm,n,int(row["offsets"][str(arm)]),tid)
                            self.streams.append(stream)
                            for t in range(n):
                                idx=len(self.entries); self.entries.append((*stream,t)); self.task_indices[tid].append(idx)
        if len(self.entries) != meta["indexed_local_timesteps"]: raise RuntimeError("indexed sample count drift")
        self.stats = {k:v.tolist() for k,v in self.stats.items()} | {"episodes":sum(len(meta["tasks"][t]) for t in TASKS),"local_streams":len(self.streams),"indexed_local_timesteps":len(self.entries),"all_data":True}
        Path(stats_path).parent.mkdir(parents=True,exist_ok=True); Path(stats_path).write_text(json.dumps(self.stats,indent=2,sort_keys=True)+"\n")
    def _update_stats(self,q,a):
        for key,val in (("q_min",q.min(0)),("q_max",q.max(0)),("a_min",a.min(0)),("a_max",a.max(0))):
            self.stats[key]=val if self.stats[key] is None else (np.minimum(self.stats[key],val) if key.endswith("min") else np.maximum(self.stats[key],val))
    def __len__(self): return len(self.entries)
    def __getstate__(self): d=dict(self.__dict__); d["handles"]={}; d["cache_handles"]={}; return d
    def _open(self,path):
        if path not in self.handles: self.handles[path]=h5py.File(path,"r",swmr=True)
        return self.handles[path]
    def _cache(self,task):
        key=str(self.cache_root/task/f"{task}.h5")
        if key not in self.cache_handles: self.cache_handles[key]=h5py.File(key,"r",swmr=True)
        return self.cache_handles[key]
    def __getitem__(self,index):
        path,traj,arm,n,offset,tid,t=self.entries[index]; task=TASKS[tid]; g=self._open(path)[traj]; start=t-(self.obs_steps-1)
        pos=np.clip(np.arange(start,start+self.horizon),0,n-1); obs_pos=pos[:self.obs_steps]
        gather=lambda ds,ix: np.stack([np.asarray(ds[int(i)]) for i in ix])
        images=gather(g[f"obs/sensor_data/head_camera_agent{arm}/rgb"],obs_pos)
        images=torch.nn.functional.interpolate(torch.from_numpy(images).permute(0,3,1,2).float(),size=(120,160),mode="bilinear",align_corners=False).numpy()
        q=gather(g[f"obs/agent/panda-{arm}/qpos"],obs_pos).astype(np.float32)
        actions=np.clip(gather(g[f"actions/panda-{arm}"],pos).astype(np.float32),ACTION_LOW,ACTION_HIGH)
        cache=self._cache(task); gaussian=np.asarray(cache[f"gaussian_arm{arm}"][offset+obs_pos[0]:offset+obs_pos[-1]+1],np.float32)
        if len(gaussian)<self.obs_steps: gaussian=np.concatenate([gaussian,np.repeat(gaussian[-1:],self.obs_steps-len(gaussian),axis=0)])
        # Cache is low resolution to bound disk use; GauDP's CNN receives it at RGB resolution.
        gaussian=torch.from_numpy(gaussian); gaussian=torch.nn.functional.interpolate(gaussian,size=(120,160),mode="bilinear",align_corners=False)
        return {"obs":{"head_cam_0":torch.from_numpy(images).float().div(255),"gaussian_0":gaussian,"state":torch.from_numpy(q)},"action":torch.from_numpy(actions)}
    def _normalizer(self):
        from model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
        n=LinearNormalizer(); n["head_cam_0"]=SingleFieldLinearNormalizer.create_identity(); n["gaussian_0"]=SingleFieldLinearNormalizer.create_identity(); n["state"]=_limits(self.stats["q_min"],self.stats["q_max"]); n["action"]=_limits(self.stats["a_min"],self.stats["a_max"]); return n
    def get_normalizer(self): return self._normalizer()

class TaskBalancedBatchSampler(Sampler):
    def __init__(self,task_indices,batch_size,updates,seed):
        if batch_size%len(task_indices): raise ValueError("batch size must divide evenly across four tasks")
        self.rows,self.batch_size,self.updates,self.seed=task_indices,batch_size,updates,seed
    def __len__(self): return self.updates
    def __iter__(self):
        import random
        rng=random.Random(self.seed); each=self.batch_size//len(self.rows)
        for _ in range(self.updates):
            b=[]
            for rows in self.rows: b.extend(rng.choices(rows,k=each))
            rng.shuffle(b); yield b
