"""Train the official DINOv3 temporal reference policy on all MARS data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from before_we_act.mars_temporal_data import (
    EFFECTIVE_BATCH, MARS_TASKS, MarsBalancedDistributedBatchSampler,
    MarsTemporalDataset, load_mars_episodes,
)
from before_we_act.temporal_history_policy import TemporalHistoryPolicy


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n"); os.replace(tmp,path)


def atomic_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(".tmp")
    torch.save(value,tmp); os.replace(tmp,path)


def reduce_sum(value: torch.Tensor, world: int) -> torch.Tensor:
    result=value.detach().clone()
    if world>1: dist.all_reduce(result)
    return result


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root",type=Path,required=True); p.add_argument("--normalization",type=Path,required=True)
    p.add_argument("--visual-cache",type=Path,required=True); p.add_argument("--dino-model",required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--updates",type=int,required=True)
    p.add_argument("--workers",type=int,default=8); p.add_argument("--seed",type=int,default=20260826)
    p.add_argument("--save-every",type=int,default=5000); p.add_argument("--log-every",type=int,default=20)
    p.add_argument("--lr",type=float,default=2e-4); p.add_argument("--warmup",type=int,default=500)
    p.add_argument("--resume",type=Path); p.add_argument("--stage",choices=("smoke","formal"),required=True)
    args=p.parse_args()
    if args.stage=="formal" and args.updates != 120_000: raise ValueError("formal CARE reference budget is 120000")
    world=int(os.environ.get("WORLD_SIZE","1")); rank=int(os.environ.get("RANK","0")); local=int(os.environ.get("LOCAL_RANK","0"))
    if EFFECTIVE_BATCH % world: raise ValueError("world size must divide effective batch")
    device=torch.device("cuda",local); torch.cuda.set_device(device)
    if world>1: dist.init_process_group("nccl")
    torch.manual_seed(args.seed+rank); np.random.seed((args.seed+rank)%(2**32))
    episodes=load_mars_episodes(args.raw_root); stats=json.loads(args.normalization.read_text())
    receipt=json.loads((args.visual_cache/"cache_receipt.json").read_text())
    if receipt.get("status")!="PASSED" or receipt.get("episodes")!=600: raise RuntimeError("DINO cache incomplete")
    dataset=MarsTemporalDataset(episodes,stats,args.visual_cache)
    saved=torch.load(args.resume,map_location="cpu",weights_only=False) if args.resume else None
    start=int(saved["update"]) if saved else 0
    sampler=MarsBalancedDistributedBatchSampler(episodes,args.updates,args.seed,rank,world,start)
    loader=DataLoader(dataset,batch_sampler=sampler,num_workers=args.workers,pin_memory=True,
                      persistent_workers=args.workers>0,prefetch_factor=2 if args.workers>0 else None)
    model=TemporalHistoryPolicy(variant="hidden_residual",dino_model=args.dino_model,
                                image_height=240,image_width=320).to(device)
    trainable=[x for x in model.parameters() if x.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=args.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1.,(step+1)/max(args.warmup,1))*0.5*(1+math.cos(math.pi*(step+1)/args.updates)))
    if saved: model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
    wrapped=DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False) if world>1 else model
    args.output.mkdir(parents=True,exist_ok=True); started=time.time()
    if rank==0: atomic_json(args.output/"status.json",{"status":"TRAINING","stage":args.stage,"update":start,"target_updates":args.updates,"started_at":datetime.now(timezone.utc).isoformat()})
    for update,batch in enumerate(loader,start=start+1):
        inputs={k:batch[k].to(device,non_blocking=True) for k in MarsTemporalDataset.MODEL_INPUT_FIELDS}
        inputs["global_rgb"]=inputs["global_rgb"].float().div_(255); inputs["local_rgb"]=inputs["local_rgb"].float().div_(255)
        actions=batch["action"].to(device,non_blocking=True); mask=batch["action_mask"].to(device,non_blocking=True)
        optimizer.zero_grad(set_to_none=True); counterfactual=update%4==0
        with torch.autocast("cuda",dtype=torch.bfloat16):
            prediction,mu,logvar,routes,cf,cf_target,base,residual,_=wrapped(**inputs,actions=actions,return_routing=True,counterfactual=counterfactual)
            numerator=((prediction-actions).square().mean(-1)*mask).sum(); denominator=reduce_sum(mask.sum().float(),world).clamp_min(1)
            action_loss=numerator*world/denominator; kl=-.5*(1+logvar-mu.square()-logvar.exp()).sum(-1).mean()
            coupling=prediction.new_zeros(())
            if counterfactual:
                errors=(cf-cf_target.unsqueeze(2)).square().mean(-1); target=(-errors.detach()/errors.detach().std(-1,keepdim=True).clamp_min(1e-3)).softmax(-1)
                coupling=F.kl_div(routes[:1].clamp_min(1e-8).log(),target,reduction="batchmean")
            loss=action_loss+1e-3*kl+0.05*coupling
        if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at {update}")
        loss.backward(); grad=torch.nn.utils.clip_grad_norm_(trainable,1.0)
        if not torch.isfinite(grad): raise FloatingPointError(f"non-finite gradient at {update}")
        optimizer.step(); scheduler.step()
        if rank==0 and (update==start+1 or update%args.log_every==0 or update==args.updates):
            row={"status":"TRAINING","stage":args.stage,"update":update,"target_updates":args.updates,"loss":float(loss),"action":float(action_loss),"kl":float(kl),"lr":scheduler.get_last_lr()[0],"elapsed_seconds":time.time()-started,"strict_local":True}
            print(json.dumps(row),flush=True); atomic_json(args.output/"status.json",row)
        if rank==0 and (update%args.save_every==0 or update==args.updates):
            atomic_save({"format":"before-we-act.mars.temporal/3-residual-own-base","update":update,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"stats":stats,"config":{"image_height":240,"image_width":320,"tasks":list(MARS_TASKS),"strict_local":True,"vision":"dinov3_vitb16_frozen","action_encoding":"joint_residual_gripper_absolute","role_context":"own_base_xy_in_task_context"}},args.output/"checkpoint_latest.pt")
    if rank==0: atomic_json(args.output/"status.json",{"status":"PASSED","stage":args.stage,"update":args.updates,"target_updates":args.updates})
    if world>1: dist.destroy_process_group()


if __name__=="__main__": main()
