"""RDT-1B adapter for the four MARS-Control HDF5 corpora.

Each trajectory/arm is exposed as an independent stream.  No peer state,
global image, arm id, or split label reaches the policy.  The RDT unified
128-D state contains the local 7 joints and gripper only.
"""
from __future__ import annotations
import glob, json, os
from pathlib import Path
import h5py, numpy as np, yaml
from configs.state_vec import STATE_VEC_IDX_MAPPING

TASKS = ("place_cube_in_cup", "strike_cube_hard", "three_robots_place_shoes", "four_robots_stack_cube")
ARMS = {"place_cube_in_cup": 2, "strike_cube_hard": 2, "three_robots_place_shoes": 3, "four_robots_stack_cube": 4}

class HDF5VLADataset:
    def __init__(self) -> None:
        root = Path(os.environ.get("RDT_MARS_DATASET", "/workspace/datasets/mars_control"))
        with open("configs/base.yaml", encoding="utf-8") as f: cfg = yaml.safe_load(f)
        self.DATASET_NAME = "mars_control"; self.CHUNK_SIZE = int(cfg["common"]["action_chunk_size"])
        self.IMG_HISTORY_SIZE = int(cfg["common"]["img_history_size"]); self.STATE_DIM = int(cfg["common"]["state_dim"])
        self.file_paths = []; self._episode_lengths = {}; expected = found = 0
        for task in TASKS:
            paths = sorted(glob.glob(str(root / task / "motionplanning" / "*.shard*.h5")))
            if len(paths) != 10: raise RuntimeError(f"{task}: expected 10 shards, found {len(paths)}")
            for path in paths:
                with h5py.File(path, "r") as f:
                    for key in sorted((k for k in f if k.startswith("traj_")), key=lambda x: int(x.rsplit("_",1)[-1])):
                        expected += 1; tr = f[key]; n = min(len(tr[f"actions/panda-{a}"]) for a in range(ARMS[task]))
                        if not bool(np.asarray(tr["success"])[-1]): raise RuntimeError(f"unsuccessful trajectory: {path}:{key}")
                        if n < 1: raise RuntimeError(f"empty trajectory: {path}:{key}")
                        found += 1; ident = f"{path}::{key}"; self._episode_lengths[ident] = n
                        self.file_paths.extend((path, key, arm, task) for arm in range(ARMS[task]))
        if os.environ.get("RDT_ALLOW_INCOMPLETE_DATASET") != "1" and (expected != 600 or found != 600):
            raise RuntimeError(f"formal MARS RDT requires 600 successful episodes; found {found}/{expected}")
        if not self.file_paths: raise RuntimeError(f"no MARS HDF5 under {root}")
        self._episodes = sorted(self._episode_lengths); self._tasks = list(TASKS)
        self._task_episodes = {t: [x for x in self._episodes if f"/{t}/" in x] for t in TASKS}
        self._episode_to_items = {x: [i for i,r in enumerate(self.file_paths) if f"{r[0]}::{r[1]}" == x] for x in self._episodes}
        self._task_weights = {t: np.asarray([self._episode_lengths[x] for x in xs], float) / sum(self._episode_lengths[x] for x in xs) for t,xs in self._task_episodes.items()}
    def __len__(self): return len(self.file_paths)
    def get_dataset_name(self): return self.DATASET_NAME
    def _indices(self): return [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)] + [STATE_VEC_IDX_MAPPING["right_gripper_open"]]
    def _unified(self, values, action=False):
        out = np.zeros(values.shape[:-1] + (self.STATE_DIM,), np.float32); ids = self._indices(); out[..., ids[:7]] = values[..., :7]
        out[..., ids[7]] = (values[..., 7] + 1.0) / 2.0 if action else values[..., 7:9].mean(-1) / 0.04
        return out
    def _trajectory(self, path, key, arm):
        with h5py.File(path, "r") as f:
            tr=f[key]; q=np.asarray(tr[f"obs/agent/panda-{arm}/qpos"], np.float32); a=np.asarray(tr[f"actions/panda-{arm}"], np.float32)
            # RoboFactory records the post-action observation, hence qpos has
            # one extra row in most shards.  RDT samples aligned (q_t, a_t)
            # pairs and must never index a state without a target action.
            n=min(len(q),len(a)); q=q[:n]; a=a[:n]
            text=f"MARS-Control task {Path(path).parent.parent.name.replace('_',' ')}"
        return self._unified(q), self._unified(a, action=True), text
    def get_item(self, index=None, state_only=False):
        if index is None:
            task = str(np.random.choice(self._tasks)); ep = str(np.random.choice(self._task_episodes[task], p=self._task_weights[task])); index = int(np.random.choice(self._episode_to_items[ep]))
        path,key,arm,task=self.file_paths[int(index)]; state,action,text=self._trajectory(path,key,arm); n=len(state); t=int(np.random.randint(0,n)); acts=action[t:t+self.CHUNK_SIZE]
        if len(acts)<self.CHUNK_SIZE: acts=np.concatenate([acts,np.repeat(acts[-1:],self.CHUNK_SIZE-len(acts),0)],0)
        with h5py.File(path,"r") as f: frames=np.asarray(f[key][f"obs/sensor_data/head_camera_agent{arm}/rgb"][max(0,t-self.IMG_HISTORY_SIZE+1):t+1],np.uint8)
        valid=len(frames)
        if valid<self.IMG_HISTORY_SIZE: frames=np.concatenate([np.repeat(frames[:1],self.IMG_HISTORY_SIZE-valid,0),frames],0)
        mask=np.zeros(self.IMG_HISTORY_SIZE,bool); mask[-valid:]=True; ids=self._indices(); ind=np.zeros(self.STATE_DIM,np.float32); ind[ids]=1
        embed=Path(path).parents[1]/"lang_embed.pt"; instruction=str(embed) if embed.is_file() else text
        if state_only:return {"state":state,"action":action}
        empty=np.zeros((self.IMG_HISTORY_SIZE,0,0,0),np.uint8)
        return {"meta":{"dataset_name":self.DATASET_NAME,"#steps":n,"step_id":t,"instruction":instruction,"task":task,"agent":arm},"state":state[t:t+1],"state_std":state.std(0).clip(1e-4),"state_mean":state.mean(0),"state_norm":np.sqrt(np.mean(state**2,0)),"actions":acts,"state_indicator":ind,"cam_high":frames,"cam_high_mask":mask,"cam_right_wrist":empty,"cam_right_wrist_mask":np.zeros(self.IMG_HISTORY_SIZE,bool),"cam_left_wrist":empty,"cam_left_wrist_mask":np.zeros(self.IMG_HISTORY_SIZE,bool)}

if __name__ == "__main__":
    d=HDF5VLADataset(); s=d.get_item(0); print(json.dumps({"streams":len(d),"episodes":len(d._episodes),"tasks":d._tasks,"state":s["state"].shape,"actions":s["actions"].shape,"image":s["cam_high"].shape}, default=list))
