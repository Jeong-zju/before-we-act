"""Full-timestep held-out validation for the R13N B6 baseline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from before_we_act.action_generator.r13n_baseline import R13NActionGenerator, load_r13n_config
from before_we_act.data.full_episode_windows import FullEpisodeActionWindows, SequentialFullEpisodeSampler
from before_we_act.r13n import FULL_CACHE_PROTOCOL, TASKS, sha256
from before_we_act.train_action_generator_r4 import atomic_json
from before_we_act.train_r13n_baseline import validate_index


@torch.inference_mode()
def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True); parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--full-index",required=True); parser.add_argument("--output",required=True)
    parser.add_argument("--heartbeat",default=""); parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--batch-size",type=int,default=10); parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    if args.batch_size<1 or args.workers<0: raise ValueError("R13N offline batch/workers differ")
    device=torch.device(args.device)
    config=load_r13n_config(args.config)
    checkpoint_path=Path(args.checkpoint).resolve(strict=True)
    checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    if checkpoint.get("round")!="R13N" or checkpoint.get("model_id")!="b6_act_six_task" or int(checkpoint.get("update",-1))!=int(config.training["updates"]):
        raise ValueError("R13N offline checkpoint identity differs")
    model=R13NActionGenerator(config).to(device); model.load_state_dict(checkpoint["model"],strict=True); model.eval()
    index_path=Path(args.full_index).resolve(strict=True); index=json.loads(index_path.read_text()); validate_index(index)
    if index.get("protocol_variant")!=FULL_CACHE_PROTOCOL or checkpoint.get("full_index_sha256")!=sha256(index_path):
        raise ValueError("R13N offline index/checkpoint identity differs")
    stats={key:torch.as_tensor(checkpoint["stats"][key],dtype=torch.float32) for key in ("a_mean","a_std")}
    dataset=FullEpisodeActionWindows(index["episodes"],stats,split="validation",cache_episodes=8,tasks=TASKS)
    expected=sum(int(value) for value in index["step_counts"]["validation"].values())
    if len(dataset)!=expected: raise ValueError("R13N validation row count differs")
    loader=DataLoader(dataset,batch_sampler=SequentialFullEpisodeSampler(dataset,args.batch_size),num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0,prefetch_factor=2 if args.workers>0 else None)
    totals={task:{"rows":0,"first_error":0.0,"first_elements":0,"full_error":0.0,"full_elements":0} for task in TASKS}
    heartbeat=Path(args.heartbeat).resolve() if args.heartbeat else None
    observed=0; finite=True; started=time.monotonic()
    for batch_index,cpu_batch in enumerate(loader):
        batch={key:value.to(device,non_blocking=True) for key,value in cpu_batch.items()}
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
            proposals=model.sample(batch,spatial_tokens=batch["spatial_tokens"],spatial_view_mask=batch["spatial_view_mask"],task_index=batch["task_index"])
        predicted=proposals.actions[:,0].permute(0,2,1,3).float(); target=batch["joint_actions"].float()
        active=batch["agent_mask"][:,None,:,None]; valid=active & batch["action_step_mask"][:,:,None,None]
        finite=finite and bool(torch.isfinite(predicted[valid.expand_as(predicted)]).all())
        error=(predicted-target).square()
        for task_index,task in enumerate(TASKS):
            selected=batch["task_index"].eq(task_index)
            if not bool(selected.any()): continue
            selected_error=error[selected]; first_mask=active[selected,0].expand_as(selected_error[:,0]); full_mask=valid[selected].expand_as(selected_error)
            row=totals[task]; row["rows"]+=int(selected.sum()); row["first_error"]+=float(selected_error[:,0][first_mask].sum().cpu()); row["first_elements"]+=int(first_mask.sum()); row["full_error"]+=float(selected_error[full_mask].sum().cpu()); row["full_elements"]+=int(full_mask.sum())
        observed+=len(target)
        if heartbeat and (batch_index==0 or batch_index%50==0):
            elapsed=time.monotonic()-started
            atomic_json(heartbeat,{"producer":"evaluate_r13n_offline","pid":os.getpid(),"rows":observed,"total_rows":expected,"eta_seconds":(expected-observed)*elapsed/max(observed,1),"updated_at_epoch":time.time()})
    per_task={task:{"rows":row["rows"],"first_step_normalized_mse":row["first_error"]/row["first_elements"],"full_chunk_normalized_mse":row["full_error"]/row["full_elements"]} for task,row in totals.items()}
    first_error=sum(row["first_error"] for row in totals.values()); first_elements=sum(row["first_elements"] for row in totals.values()); full_error=sum(row["full_error"] for row in totals.values()); full_elements=sum(row["full_elements"] for row in totals.values())
    result={"schema_version":1,"round":"R13N","model_id":"b6_act_six_task","checkpoint":str(checkpoint_path),"checkpoint_sha256":sha256(checkpoint_path),"update":int(checkpoint["update"]),"full_index":str(index_path),"full_index_sha256":sha256(index_path),"validation_rows":observed,"expected_validation_rows":expected,"all_outputs_finite":finite,"first_step_normalized_mse":first_error/first_elements,"full_chunk_normalized_mse":full_error/full_elements,"per_task":per_task,"candidate_native":True,"elapsed_seconds":time.monotonic()-started,"created_at_epoch":time.time()}
    if not finite or observed!=expected: raise ValueError("R13N offline validation failed")
    atomic_json(Path(args.output),result)
    if heartbeat: atomic_json(heartbeat,{"producer":"evaluate_r13n_offline","status":"PASSED","rows":observed,"updated_at_epoch":time.time()})
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
