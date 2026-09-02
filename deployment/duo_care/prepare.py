from __future__ import annotations
import argparse, json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import av, cv2, numpy as np, pyarrow.parquet as pq

JOINT_LOW=np.asarray([-2.3093,-1.5133,-2.4937,-2.7478,-2.48,.8521,-2.6895],np.float32)
JOINT_HIGH=np.asarray([2.3093,1.5133,2.4937,-.4461,2.48,4.2094,2.6895],np.float32)

TASKS=("ball_maze","bin_sort","block_balance","carry_pot","hinge_chest","join_blocks","pour_marbles","spring_door","transfer_cube","transfer_gate","transfer_reorient")
TRAINING_HORIZON=16

def decode_video(path:Path, size:int)->np.ndarray:
    c=av.open(str(path)); frames=[]
    for f in c.decode(video=0):
        a=f.to_ndarray(format="rgb24")
        frames.append(cv2.resize(a,(size,size),interpolation=cv2.INTER_LINEAR))
    c.close(); return np.asarray(frames,dtype=np.uint8)

def prepare_task(dataset:Path, output:Path, size:int, tid:int, task:str):
    root=dataset/task/"sim"; parquet=sorted((root/"data").glob("**/*.parquet"))
    if len(parquet)!=1: raise RuntimeError(f"{task}: expected one parquet, found {len(parquet)}")
    table=pq.read_table(parquet[0]).to_pydict(); episodes=np.asarray(table["episode_index"]); states=np.asarray(table["observation.state"],dtype=np.float32); raw_actions=np.asarray(table["action"],dtype=np.float32); actions=raw_actions.copy().reshape(-1,2,8); actions[:,:,:7]=np.clip(actions[:,:,:7],JOINT_LOW,JOINT_HIGH); actions[:,:,7]=(actions[:,:,7]>=.5).astype(np.float32); clipped=int(np.count_nonzero(actions.reshape(-1,16)!=raw_actions)); actions=actions.reshape(-1,16)
    if states.ndim!=2 or states.shape[1]!=16 or actions.shape!=states.shape: raise RuntimeError(f"{task}: bad state/action shapes {states.shape}/{actions.shape}")
    out=output/task; out.mkdir(exist_ok=True); unique=np.unique(episodes); rows=[int(np.sum(episodes==ep)) for ep in unique]
    complete=all((out/name).is_file() for name in ('head.npy','left.npy','right.npy'))
    if not complete:
        head=decode_video(next((root/"videos/observation.images.head").glob("**/*.mp4")),size); left=decode_video(next((root/"videos/observation.images.left_wrist").glob("**/*.mp4")),size); right=decode_video(next((root/"videos/observation.images.right_wrist").glob("**/*.mp4")),size)
        if len(head)!=len(states) or len(left)!=len(states) or len(right)!=len(states): raise RuntimeError(f"{task}: video/table length mismatch {len(head)}/{len(states)} vs {len(states)}")
        np.save(out/'head.npy',head); np.save(out/'left.npy',left); np.save(out/'right.npy',right)
    np.save(out/'state.npy',states); np.save(out/'action.npy',actions); np.save(out/'episodes.npy',episodes)
    if len(unique)!=50: raise RuntimeError(f"{task}: expected all 50 demos, found {len(unique)}")
    return task,{"episodes":len(rows),"frames":int(sum(rows)),"min_demo_steps":int(min(rows)),"max_demo_steps":int(max(rows)),"validation_max_steps":int(np.ceil(np.quantile(rows,.99))),"mean_steps":float(np.mean(rows)),"action_dim":16,"local_action_dim":8,"state_min":states.min(0).tolist(),"state_max":states.max(0).tolist(),"action_min":actions.min(0).tolist(),"action_max":actions.max(0).tolist(),"raw_action_min":raw_actions.min(0).tolist(),"raw_action_max":raw_actions.max(0).tolist(),"clipped_or_binarized_action_values":clipped}

def chunk_action_statistics(output:Path,horizon:int=TRAINING_HORIZON):
    """Statistics over the exact padded, anchor-relative training targets."""
    total=0; summed=np.zeros(8,np.float64); squared=np.zeros(8,np.float64)
    for task in TASKS:
      q=np.load(output/task/'state.npy',mmap_mode='r').reshape(-1,2,8); u=np.load(output/task/'action.npy',mmap_mode='r').reshape(-1,2,8); episodes=np.load(output/task/'episodes.npy',mmap_mode='r'); starts=np.r_[0,np.flatnonzero(episodes[1:]!=episodes[:-1])+1]; ends=np.r_[starts[1:],len(episodes)]
      for start,end in zip(starts,ends):
        anchor=q[start:end]
        local_steps=np.arange(end-start)
        for offset in range(horizon):
          target=u[start+np.minimum(local_steps+offset,end-start-1)].copy(); target[:,:,:7]-=anchor[:,:,:7]; values=target.reshape(-1,8).astype(np.float64); total+=len(values); summed+=values.sum(0); squared+=np.square(values).sum(0)
    mean=summed/total; variance=np.maximum(squared/total-np.square(mean),0); return mean,np.sqrt(variance),total

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--image-size",type=int,default=96); p.add_argument("--jobs",type=int,default=5)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); manifest={"schema":"duobench-care-prepared-v1","tasks":{},"image_size":a.image_size}
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        futures=[pool.submit(prepare_task,a.dataset,a.output,a.image_size,tid,task) for tid,task in enumerate(TASKS)]
        for future in as_completed(futures):
            task,row=future.result(); manifest["tasks"][task]=row; print(json.dumps({"event":"task_prepared","task":task,**row}),flush=True)
    manifest["tasks"]={task:manifest["tasks"][task] for task in TASKS}
    manifest["total_episodes"]=sum(x["episodes"] for x in manifest["tasks"].values()); manifest["total_frames"]=sum(x["frames"] for x in manifest["tasks"].values()); manifest["normalization"]={"state_mean":[]}
    # Qpos uses every frame and both local arms.  Action statistics use every
    # element of the same padded, anchor-relative chunks consumed by training.
    qs=[]
    for task in TASKS:
      qs.append(np.load(a.output/task/'state.npy',mmap_mode='r').reshape(-1,2,8))
    q=np.concatenate(qs); action_mean,action_std,action_count=chunk_action_statistics(a.output); manifest["normalization"]={"qpos_mean":q.mean((0,1)).tolist(),"qpos_std":np.maximum(q.std((0,1)),1e-4).tolist(),"action_mean":action_mean.tolist(),"action_std":np.maximum(action_std,1e-4).tolist(),"action_encoding":"anchor_joint_residual_gripper_absolute","action_chunk_horizon":TRAINING_HORIZON,"action_target_count":action_count}
    manifest["normalization"]["population"]="all_padded_horizon16_targets_all_50_demos_all_11_tasks_both_local_arms"; manifest["dataset_revision"]="b741bc915d942ecadaefb4e3de6bbd716c1b8b1b"
    (a.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n"); print(json.dumps(manifest,sort_keys=True))
if __name__=="__main__": main()
