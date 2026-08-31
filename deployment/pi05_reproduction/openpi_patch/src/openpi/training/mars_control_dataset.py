"""Streaming decentralized MARS-Control HDF5 dataset for OpenPI."""
from __future__ import annotations
import bisect, glob, os
from pathlib import Path
from typing import SupportsIndex
import h5py, numpy as np

TASKS=("place_cube_in_cup","strike_cube_hard","three_robots_place_shoes","four_robots_stack_cube")
ARMS={"place_cube_in_cup":2,"strike_cube_hard":2,"three_robots_place_shoes":3,"four_robots_stack_cube":4}
PROMPTS={"place_cube_in_cup":"Place the cube in the cup","strike_cube_hard":"Strike the cube hard","three_robots_place_shoes":"Three robots place shoes","four_robots_stack_cube":"Four robots stack the cube"}

class MarsControlDataset:
    def __init__(self, root: str|Path="/workspace/datasets/mars_control", action_horizon:int=16):
        self.root=Path(os.environ.get("OPENPI_MARS_CONTROL_ROOT",root)); self.action_horizon=int(action_horizon)
        self.streams=[]; self.cumulative=[]; self._cache={}; self.task_streams=[[] for _ in TASKS]; self.task_cumulative=[[] for _ in TASKS]; episodes=0
        for task_id,task in enumerate(TASKS):
            paths=sorted(glob.glob(str(self.root/task/"motionplanning"/f"{task}.shard*.h5")))
            if len(paths)!=10: raise RuntimeError(f"{task}: expected 10 HDF5 shards, found {len(paths)}")
            for path in paths:
                with h5py.File(path,"r") as f:
                    for tr in sorted(f,key=lambda x:int(x.rsplit("_",1)[-1]) if "_" in x else x):
                        g=f[tr]; success=np.asarray(g["success"])
                        if not bool(success.reshape(-1)[-1]): raise RuntimeError(f"non-success trajectory {path}:{tr}")
                        arms=sorted(int(k.rsplit("-",1)[-1]) for k in g["obs/agent"] if k.startswith("panda-"))
                        if arms!=list(range(ARMS[task])): raise RuntimeError(f"{task}: arm layout drift {arms}")
                        n=min(len(g[f"actions/panda-{a}"]) for a in arms)
                        if "done" in g:
                            done=np.flatnonzero(np.asarray(g["done"][:n],bool)); n=int(done[0]+1) if len(done) else n
                        for arm in arms:
                            q=g[f"obs/agent/panda-{arm}/qpos"]; a=g[f"actions/panda-{arm}"]; im=g[f"obs/sensor_data/head_camera_agent{arm}/rgb"]
                            if tuple(q.shape[1:])!=(9,) or tuple(a.shape[1:])!=(8,) or im.dtype!=np.uint8: raise RuntimeError(f"local schema drift {path}:{tr}:arm{arm}")
                            self.streams.append((path,tr,arm,n)); self.cumulative.append((self.cumulative[-1] if self.cumulative else 0)+n)
                            self.task_streams[task_id].append((path,tr,arm,n)); self.task_cumulative[task_id].append((self.task_cumulative[task_id][-1] if self.task_cumulative[task_id] else 0)+n)
                        episodes+=1
        if episodes!=600 or len(self.streams)!=1650: raise RuntimeError(f"MARS corpus drift episodes={episodes} streams={len(self.streams)}")
        self.balanced_task_length=max(x[-1] for x in self.task_cumulative)
    def __len__(self): return self.balanced_task_length*len(TASKS)
    def _file(self,path):
        key=(path,os.getpid())
        if key not in self._cache: self._cache[key]=h5py.File(path,"r")
        return self._cache[key]
    def __getitem__(self,index:SupportsIndex):
        flat=index.__index__(); flat=flat+len(self) if flat<0 else flat
        if flat<0 or flat>=len(self): raise IndexError(flat)
        # Interleave four equal-probability virtual task lanes. Shorter tasks
        # wrap, so every source timestep remains eligible while the long
        # four-arm task cannot dominate solely through cardinality.
        task_id=flat%len(TASKS); local=(flat//len(TASKS))%self.task_cumulative[task_id][-1]; cumulative=self.task_cumulative[task_id]; i=bisect.bisect_right(cumulative,local); start=cumulative[i-1] if i else 0; path,tr,arm,n=self.task_streams[task_id][i]; t=local-start; g=self._file(path)[tr]
        image=np.asarray(g[f"obs/sensor_data/head_camera_agent{arm}/rgb"][t],np.uint8); state=np.asarray(g[f"obs/agent/panda-{arm}/qpos"][t],np.float32); actions=np.asarray(g[f"actions/panda-{arm}"][t:min(t+self.action_horizon,n)],np.float32)
        if len(actions)<self.action_horizon: actions=np.concatenate([actions,np.repeat(actions[-1:],self.action_horizon-len(actions),axis=0)],axis=0)
        return {"observation/image":image,"observation/state":state,"actions":actions,"prompt":PROMPTS[Path(path).parts[-3]]}
    def close(self):
        for h in self._cache.values(): h.close()
        self._cache.clear()
