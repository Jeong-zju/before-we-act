"""Train the official DINOv3 temporal reference policy on all MARS data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from before_we_act.mars_temporal_data import (
    EFFECTIVE_BATCH, MARS_TASKS, MarsBalancedDistributedBatchSampler,
    MarsTemporalDataset, load_mars_episodes, validate_mars_normalization,
)
from before_we_act.mars_action_contract import (
    ACTION_CONTRACT_VERSION,
    action_contract_hash,
    checkpoint_action_contract,
    normalization_stats_hash,
    validate_checkpoint_action_contract,
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
    p.add_argument("--protocol-updates",type=int,default=120_000)
    p.add_argument("--workers",type=int,default=8); p.add_argument("--seed",type=int,default=20260826)
    p.add_argument("--save-every",type=int,default=5000); p.add_argument("--log-every",type=int,default=20)
    p.add_argument("--lr",type=float,default=2e-4); p.add_argument("--router-lr",type=float,default=3e-4)
    p.add_argument("--warmup",type=int,default=500)
    p.add_argument("--resume",type=Path); p.add_argument("--stage",choices=("f1","smoke","formal"),required=True)
    args=p.parse_args()
    if not 1 <= args.updates <= args.protocol_updates: raise ValueError("invalid update target")
    if args.protocol_updates != 120_000: raise ValueError("official B0-H protocol is fixed at 120000 updates")
    if args.stage=="formal" and args.updates != 120_000: raise ValueError("formal CARE reference budget is 120000")
    world=int(os.environ.get("WORLD_SIZE","1")); rank=int(os.environ.get("RANK","0")); local=int(os.environ.get("LOCAL_RANK","0"))
    if EFFECTIVE_BATCH % world: raise ValueError("world size must divide effective batch")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device("cuda",local); torch.cuda.set_device(device)
    if world>1: dist.init_process_group("nccl")
    random.seed(args.seed); np.random.seed(args.seed%(2**32)); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    episodes=load_mars_episodes(args.raw_root); stats=json.loads(args.normalization.read_text())
    # The temporal reference and every downstream CARE artifact must use the
    # same physical action projection as ACT.  Stale pre-contract statistics
    # are rejected before any GPU work starts; silently reusing them would
    # invalidate paired comparisons and branch labels.
    validate_mars_normalization(stats)
    receipt=json.loads((args.visual_cache/"cache_receipt.json").read_text())
    if receipt.get("status")!="PASSED" or receipt.get("episodes")!=600: raise RuntimeError("DINO cache incomplete")
    dataset=MarsTemporalDataset(episodes,stats,args.visual_cache)
    saved=torch.load(args.resume,map_location="cpu",weights_only=False) if args.resume else None
    if saved is not None:
        saved_contract = validate_checkpoint_action_contract(saved)
        saved_annotations = saved_contract.get("annotations", {})
        if saved_annotations.get("normalization_sha256") != normalization_stats_hash(stats):
            raise ValueError("resume checkpoint normalization/action contract differs")
    start=int(saved["update"]) if saved else 0
    sampler=MarsBalancedDistributedBatchSampler(episodes,args.protocol_updates,args.seed,rank,world,start)
    if saved and "sample_cursor" in saved: sampler.validate_cursor(saved["sample_cursor"])
    loader=DataLoader(dataset,batch_sampler=sampler,num_workers=args.workers,pin_memory=True,
                      persistent_workers=args.workers>0,prefetch_factor=2 if args.workers>0 else None)
    model=TemporalHistoryPolicy(variant="hidden_residual",dino_model=args.dino_model,
                                image_height=240,image_width=320).to(device)
    router_prefix=("compatibility","role_prototypes","route_state","route_observation","route_mlp")
    router=[]; body=[]
    for name,parameter in model.named_parameters():
        if parameter.requires_grad: (router if name.startswith(router_prefix) else body).append(parameter)
    trainable=body+router
    optimizer=torch.optim.AdamW([
        {"params":body,"lr":args.lr},{"params":router,"lr":args.router_lr},
    ],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step:min(1.,(step+1)/max(args.warmup,1))*0.5*
        (1+math.cos(math.pi*min(1.,(step+1)/args.protocol_updates))),
    )
    if saved: model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
    wrapped=DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False) if world>1 else model
    args.output.mkdir(parents=True,exist_ok=True); started=time.time()
    if rank==0: atomic_json(args.output/"status.json",{"status":"TRAINING","stage":args.stage,"update":start,"target_updates":args.updates,"started_at":datetime.now(timezone.utc).isoformat()})
    for update,batch in enumerate(loader,start=start+1):
        if update > args.updates: break
        step_seed=args.seed+10_000_019*update+100_003*rank
        random.seed(step_seed); np.random.seed(step_seed%(2**32)); torch.manual_seed(step_seed)
        torch.cuda.manual_seed_all(step_seed)
        inputs={k:batch[k].to(device,non_blocking=True) for k in MarsTemporalDataset.MODEL_INPUT_FIELDS}
        inputs["global_rgb"]=inputs["global_rgb"].float().div_(255); inputs["local_rgb"]=inputs["local_rgb"].float().div_(255)
        actions=batch["action"].to(device,non_blocking=True); mask=batch["action_mask"].to(device,non_blocking=True)
        optimizer.zero_grad(set_to_none=True); counterfactual=update%4==0
        with torch.autocast("cuda",dtype=torch.bfloat16):
            prediction,mu,logvar,routes,cf,cf_target,base,residual,_=wrapped(**inputs,actions=actions,return_routing=True,counterfactual=counterfactual)
            numerator=((prediction-actions).square().mean(-1)*mask).sum(); denominator=reduce_sum(mask.sum().float(),world).clamp_min(1)
            action_loss=numerator*world/denominator
            local_kl=-.5*(1+logvar-mu.square()-logvar.exp()).sum(-1).sum()
            kl=local_kl*world/EFFECTIVE_BATCH
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
            row={"status":"TRAINING","stage":args.stage,"update":update,"target_updates":args.updates,"loss":float(loss),"action":float(action_loss),"kl":float(kl),"lr":scheduler.get_last_lr()[0],"router_lr":scheduler.get_last_lr()[1],"elapsed_seconds":time.time()-started,"strict_local":True}
            print(json.dumps(row),flush=True); atomic_json(args.output/"status.json",row)
        if rank==0 and (update%args.save_every==0 or update==args.updates):
            payload={"format":"before-we-act.mars.temporal/5-action-contract","update":update,
                "model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
                "sample_cursor":sampler.cursor_receipt(update),"stats":stats,
                "action_contract":checkpoint_action_contract(
                    normalization_sha256=normalization_stats_hash(stats)
                ),
                "config":{"image_height":240,"image_width":320,"tasks":list(MARS_TASKS),
                "strict_local":True,"vision":"dinov3_vitb16_frozen","action_encoding":"absolute_pd_joint_pos",
                "role_context":"own_base_xy_in_task_context","recipe":"robofactory_b0h_bench_port",
                "protocol_updates":args.protocol_updates,"seed":args.seed,"dino_model":args.dino_model,
                "state_dim":9,"action_dim":8,"horizon":100,"d_model":384,"enc_layers":4,
                "dec_layers":7,"roles":4,"role_rank":32,"history_layers":2,
                "action_contract_version":ACTION_CONTRACT_VERSION,
                "action_contract_sha256":action_contract_hash(),
                "normalization_sha256":normalization_stats_hash(stats)}}
            atomic_save(payload,args.output/f"checkpoint_{update:06d}.pt")
            atomic_save(payload,args.output/"checkpoint_latest.pt")
    if rank==0: atomic_json(args.output/"status.json",{"status":"PASSED","stage":args.stage,"update":args.updates,"target_updates":args.updates})
    if world>1: dist.destroy_process_group()


if __name__=="__main__": main()
