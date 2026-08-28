from __future__ import annotations
import glob, json, os
from pathlib import Path
import h5py, numpy as np
from .common import TASKS, ARMS, atomic_json
def main():
    root=Path(os.environ.get("MARS_DP_DATA_ROOT","/workspace/datasets/mars_control")); report={"schema":"mars-control.dp.audit.v1","status":"complete","tasks":{},"episodes":0,"local_streams":0,"policy_contract":"shared_weights_decentralized_local_rgb_qpos_to_absolute_action8","forbidden_inputs":["peer_rgb","peer_qpos","global_rgb","joint_action","task_id","arm_id"]}
    for task in TASKS:
        eps=streams=steps=0; paths=sorted(glob.glob(str(root/task/"motionplanning"/"*.shard*.h5")))
        if len(paths)!=10: raise RuntimeError(f"{task}: expected 10 formal shards, got {len(paths)}")
        for path in paths:
            with h5py.File(path,"r") as f:
                for tr in f:
                    g=f[tr]; eps+=1; streams+=ARMS[task]; n=min(len(g[f"actions/panda-{a}"]) for a in range(ARMS[task])); steps+=n
                    if not bool(np.asarray(g["success"])[-1]): raise RuntimeError(f"failed trajectory {path}:{tr}")
                    for arm in range(ARMS[task]):
                        if g[f"obs/sensor_data/head_camera_agent{arm}/rgb"].shape[-1] != 3 or g[f"obs/agent/panda-{arm}/qpos"].shape[-1] != 9 or g[f"actions/panda-{arm}"].shape[-1] != 8: raise RuntimeError(f"schema drift {task}:{tr}:{arm}")
        if eps!=150: raise RuntimeError(f"{task}: expected 150 episodes, got {eps}")
        report["tasks"][task]={"episodes":eps,"local_streams":streams,"joint_steps":steps}; report["episodes"]+=eps; report["local_streams"]+=streams
    atomic_json(Path(os.environ.get("MARS_DP_RUN_ROOT","/workspace/runs/mars_dp"))/"audit.json",report); print(json.dumps(report))
if __name__=="__main__": main()
