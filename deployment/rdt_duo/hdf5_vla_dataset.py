"""All-data, local-arm DuoBench adapter implementing RDT's HDF5 contract.

The upstream trainer calls this class ``HDF5VLADataset``.  DuoBench is released
as LeRobot parquet/video, so the audited numpy preparation is the backing store;
the public class deliberately retains the upstream interface and semantics.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
from configs.state_vec import STATE_VEC_IDX_MAPPING
from deployment.rdt_duo.protocol import TASKS, ACTION_CHUNK_SIZE, IMAGE_HISTORY_SIZE, STATE_DIM

class HDF5VLADataset:
    DATASET_NAME = "duobench"
    def __init__(self) -> None:
        root = Path(os.environ.get("RDT_DUOBENCH_DATA", "/workspace/runs/rdt_duo/data")); self.root = root
        manifest = json.loads((root / "manifest.json").read_text())
        if tuple(manifest.get("rdt_tasks", TASKS)) != TASKS: raise RuntimeError("DuoBench task contract drift")
        self.CHUNK_SIZE, self.IMG_HISTORY_SIZE, self.STATE_DIM = ACTION_CHUNK_SIZE, IMAGE_HISTORY_SIZE, STATE_DIM
        self._data, self._streams, self._tasks = [], [], list(TASKS); self._task_streams = {t: [] for t in TASKS}; self._task_weights = {}
        for tid, task in enumerate(TASKS):
            d = {k: np.load(root / task / f"{k}.npy", mmap_mode="r") for k in ("state", "action", "head", "left", "right", "episodes")}
            if d["state"].shape != d["action"].shape or d["state"].shape[1:] != (16,): raise RuntimeError(f"{task}: bad state/action shape")
            episodes = np.asarray(d["episodes"]); starts = np.flatnonzero(np.r_[True, episodes[1:] != episodes[:-1]]); ends = np.r_[starts[1:], len(episodes)]
            if len(starts) != 50: raise RuntimeError(f"{task}: expected 50 episodes, found {len(starts)}")
            self._data.append(d); lengths = []
            for start, end in zip(starts, ends, strict=True):
                # Last observation has no next action under the pinned lag-1 contract.
                if int(end - start) < 2: raise RuntimeError(f"{task}: episode too short")
                for arm in (0, 1):
                    stream = (tid, arm, int(start), int(end)); self._streams.append(stream); self._task_streams[task].append(stream); lengths.append(int(end - start - 1))
            weights = np.asarray(lengths, dtype=np.float64); self._task_weights[task] = weights / weights.sum()
        self._episodes = [(t, s, e) for t in TASKS for _, _, s, e in self._task_streams[t]]
    def __len__(self): return len(self._streams)
    def get_dataset_name(self): return self.DATASET_NAME
    @staticmethod
    def _ids(): return [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)] + [STATE_VEC_IDX_MAPPING["right_gripper_open"]]
    def _unified(self, local):
        local = np.asarray(local, np.float32); out = np.zeros(local.shape[:-1] + (self.STATE_DIM,), np.float32); out[..., self._ids()] = local[..., :8]; return out
    def _stream(self, stream):
        tid, arm, start, end = stream; task = TASKS[tid]; d = self._data[tid]; state = d["state"].reshape(-1,2,8)[start:end, arm]; action = d["action"].reshape(-1,2,8)[start:end, arm]
        # Causal rows: observation at i supervises the next executed action i+1.
        return self._unified(state[:-1]), self._unified(action[1:]), task, arm, d
    def get_item(self, index=None, state_only=False):
        if index is None:
            task = str(np.random.choice(TASKS)); stream = self._task_streams[task][np.random.choice(len(self._task_streams[task]), p=self._task_weights[task])]
        else: stream = self._streams[int(index)]
        states, actions, task, arm, d = self._stream(stream); n = len(states)
        if state_only: return {"state": states, "action": actions}
        t = int(np.random.randint(0, n)); end = min(t + self.CHUNK_SIZE, n); targets = actions[t:end]
        if len(targets) < self.CHUNK_SIZE: targets = np.concatenate((targets, np.repeat(targets[-1:], self.CHUNK_SIZE-len(targets), axis=0)), axis=0)
        head = np.asarray(d["head"][stream[2]+t- min(t, self.IMG_HISTORY_SIZE-1):stream[2]+t+1])
        wrist_key = "left" if arm == 0 else "right"; wrist = np.asarray(d[wrist_key][stream[2]+t- min(t, self.IMG_HISTORY_SIZE-1):stream[2]+t+1])
        valid = len(head)
        if valid < self.IMG_HISTORY_SIZE:
            head = np.concatenate((np.repeat(head[:1], self.IMG_HISTORY_SIZE-valid, axis=0), head)); wrist = np.concatenate((np.repeat(wrist[:1], self.IMG_HISTORY_SIZE-valid, axis=0), wrist))
        mask = np.asarray([False] * (self.IMG_HISTORY_SIZE-valid) + [True] * valid)
        empty = np.zeros((self.IMG_HISTORY_SIZE, 0, 0, 0), np.uint8); indicator = np.zeros(self.STATE_DIM, np.float32); indicator[self._ids()] = 1
        full_state = self._unified(d["state"].reshape(-1,2,8)[stream[2]:stream[3], arm]); std = full_state.std(0).clip(1e-4); mean = full_state.mean(0); norm = np.sqrt(np.mean(full_state**2, axis=0))
        embed = self.root / task / "lang_embed.pt"; instruction = str(embed) if embed.is_file() else f"DuoBench task {task.replace('_', ' ')}"
        return {"meta":{"dataset_name":self.DATASET_NAME,"#steps":n,"step_id":t,"instruction":instruction,"task":task,"agent":arm}, "state":states[t:t+1], "state_std":std, "state_mean":mean, "state_norm":norm, "actions":targets, "state_indicator":indicator, "cam_high":head, "cam_high_mask":mask, "cam_right_wrist":wrist, "cam_right_wrist_mask":mask.copy(), "cam_left_wrist":empty, "cam_left_wrist_mask":np.zeros(self.IMG_HISTORY_SIZE, bool)}

if __name__ == "__main__":
    d = HDF5VLADataset(); s = d.get_item(0); print(json.dumps({"streams":len(d), "episodes":len(d._episodes), "tasks":d._tasks, "state":list(s["state"].shape), "actions":list(s["actions"].shape), "cam_high":list(s["cam_high"].shape)}))
