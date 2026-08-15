"""Train one R1-4 omniscient teacher seed and its zero-privilege control."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from before_we_act.b3_n1_model import MatchedActionProbe, masked_mse
from before_we_act.b3_n1_r1 import (
    FrozenR1Backbones,
    R1BalancedBatchSampler,
    R1OracleDataset,
    R1_DATA_SEED,
    R1_EARLIEST_PLATFORM,
    R1_EVAL_EVERY,
    R1_LR_DROP,
    R1_MAX_UPDATES,
    R1_SEEDS,
    action_sample_mse,
    deterministic_permutations,
    fixed_requests,
    load_split,
    split_by_episode_key,
)
from before_we_act.b3_n1_r1_teacher_student import (
    PrivilegedBeliefTeacher,
    gaussian_nll,
    oracle_fields,
)
from before_we_act.step2_temporal_data import SIX_TASKS, sha256_file
from before_we_act.train_b3_n1_r1_fair_probe import atomic_json, atomic_save, device_batch


CONDITIONS=("h","h_teacher","h_teacher_shuffle","h_matched_capacity")


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,required=True); parser.add_argument("--parent-contract",type=Path,required=True); parser.add_argument("--teacher-contract",type=Path,required=True); parser.add_argument("--scenario-split",type=Path,required=True)
    parser.add_argument("--fair-run-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True); parser.add_argument("--seed",type=int,required=True); parser.add_argument("--workers",type=int,default=2); parser.add_argument("--lr",type=float,default=2e-4)
    return parser.parse_args()


def utc_now(): return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")


def load_base_probe(fair_root: Path, seed: int, device: torch.device):
    status=json.loads((fair_root/f"seed_{seed}"/"status.json").read_text()); update=int(status["selected_update"])
    payload=torch.load(fair_root/f"seed_{seed}"/f"checkpoint_{update:06d}.pt",map_location="cpu",weights_only=False)
    prefix="h."
    state={key[len(prefix):]:value for key,value in payload["probes"].items() if key.startswith(prefix)}
    probe=MatchedActionProbe().to(device); probe.load_state_dict(state,strict=True); probe.eval().requires_grad_(False)
    return probe,update,sha256_file(fair_root/f"seed_{seed}"/f"checkpoint_{update:06d}.pt")


def zero_oracle(oracle):
    return {key:(value if "mask" in key else torch.zeros_like(value)) for key,value in oracle.items()}


def permute_oracle(oracle, permutation): return {key:value[permutation] for key,value in oracle.items()}


def fixed_loader(dataset,split,name):
    requests=fixed_requests(dataset.episodes,split,name); batches=[requests[i:i+192] for i in range(0,len(requests),192)]
    return DataLoader(dataset,batch_sampler=batches,num_workers=0,pin_memory=True)


@torch.no_grad()
def evaluate_teacher(backbones,base_probe,teacher,matched,loader,device):
    teacher.eval(); matched.eval(); values={name:[] for name in CONDITIONS}; tasks={name:{task:[] for task in range(6)} for name in CONDITIONS}; aux=[]
    for raw in loader:
        batch=device_batch(raw,device)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            frozen=backbones(batch); base=base_probe(frozen.h); oracle=oracle_fields(batch); permutation=deterministic_permutations(batch)["shuffle"]
            real=teacher(base,frozen.h,oracle); shuffled=teacher(base,frozen.h,permute_oracle(oracle,permutation)); control=matched(base,frozen.h,zero_oracle(oracle))
        output={"h":base,"h_teacher":real.action,"h_teacher_shuffle":shuffled.action,"h_matched_capacity":control.action}
        scores={name:action_sample_mse(pred.float(),batch) for name,pred in output.items()}
        for name in CONDITIONS:
            rows=scores[name].cpu().tolist(); values[name].extend(rows)
            for task in range(6): tasks[name][task].extend(scores[name][batch["task_index"]==task].cpu().tolist())
        aux.append((float(gaussian_nll(real.teammate_action_mean.float(),real.teammate_action_logvar.float(),batch["oracle_teammate_action"],batch["oracle_teammate_action_mask"])),float(masked_mse(real.teammate_delta.float(),batch["teammate_delta"],batch["future_mask"]))))
    return {"macro":{name:float(np.mean(rows)) for name,rows in values.items()},"per_task":{name:{str(task):float(np.mean(rows)) for task,rows in by_task.items()} for name,by_task in tasks.items()},"teammate_action_nll":float(np.mean([row[0] for row in aux])),"teammate_delta_mse":float(np.mean([row[1] for row in aux])),"rows":len(values["h"])}


def platform(metrics):
    if len(metrics)<4 or metrics[-1]["update"]<R1_EARLIEST_PLATFORM:return False
    keys=[("macro","h_teacher"),("macro","h_matched_capacity"),(None,"teammate_action_nll"),(None,"teammate_delta_mse")]
    for section,key in keys:
        scores=[float(row["validation"][section][key] if section else row["validation"][key]) for row in metrics[-4:]]
        if any((a-b)/max(abs(a),1e-12)>=.01 for a,b in zip(scores,scores[1:])):return False
    return True


def main():
    args=parse_args(); parent=json.loads(args.parent_contract.read_text()); contract=json.loads(args.teacher_contract.read_text())
    if contract.get("status")!="FROZEN_BEFORE_F0_F1" or args.seed not in R1_SEEDS:raise RuntimeError("invalid teacher contract")
    if sha256_file(args.parent_contract)!=contract["parent_contract_sha256"]:raise RuntimeError("teacher prerequisite hash differs")
    if contract.get("format_version")!="before-we-act.b3-n1-r1-teacher-contract/2":raise RuntimeError("teacher requires the owner-revised contract")
    split_payload=load_split(args.scenario_split);split=split_by_episode_key(split_payload);device=torch.device("cuda:0");torch.cuda.set_device(device)
    random.seed(args.seed);np.random.seed(args.seed%2**32);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    dataset=R1OracleDataset(args.cache);n1=parent["old_n1_read_only"]["representation_checkpoints"][str(args.seed)]
    backbones=FrozenR1Backbones(b0h_checkpoint=Path(parent["b0h"]["checkpoint"]),n1_checkpoint=Path(n1["path"]),visual_mean=dataset.visual_mean,visual_std=dataset.visual_std).to(device)
    base_probe,base_update,base_sha=load_base_probe(args.fair_run_root,args.seed,device);teacher=PrivilegedBeliefTeacher().to(device);matched=PrivilegedBeliefTeacher().to(device)
    parameters=list(teacher.parameters())+list(matched.parameters());optimizer=torch.optim.AdamW(parameters,lr=args.lr,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.0 if step<R1_LR_DROP else .1)
    latest=args.output/"checkpoint_latest.pt";saved=torch.load(latest,map_location="cpu",weights_only=False) if latest.is_file() else None;start=0;metrics=[]
    provenance={"seed":args.seed,"teacher_contract_sha256":sha256_file(args.teacher_contract),"scenario_split_sha256":sha256_file(args.scenario_split),"base_h_checkpoint_sha256":base_sha}
    if saved:
        if saved["provenance"]!=provenance:raise RuntimeError("teacher resume provenance differs")
        teacher.load_state_dict(saved["teacher"]);matched.load_state_dict(saved["matched"]);optimizer.load_state_dict(saved["optimizer"]);scheduler.load_state_dict(saved["scheduler"]);start=int(saved["update"]);metrics=list(saved["evaluations"])
    sampler=R1BalancedBatchSampler(dataset.episodes,split,updates=R1_MAX_UPDATES,data_seed=R1_DATA_SEED,start_update=start);loader=DataLoader(dataset,batch_sampler=sampler,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0,prefetch_factor=2 if args.workers>0 else None);validation=fixed_loader(dataset,split,"validation")
    args.output.mkdir(parents=True,exist_ok=True);atomic_json(args.output/"status.json",{"status":"TRAINING","seed":args.seed,"update":start,"started_at_utc":utc_now()});started=time.time();completion=None
    weights=contract["objectives"]
    for update,raw in enumerate(loader,start=start+1):
        step_seed=args.seed+10_000_019*update;random.seed(step_seed);np.random.seed(step_seed%2**32);torch.manual_seed(step_seed);torch.cuda.manual_seed_all(step_seed);batch=device_batch(raw,device);optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):frozen=backbones(batch);base=base_probe(frozen.h)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            oracle=oracle_fields(batch);real=teacher(base,frozen.h,oracle);control=matched(base,frozen.h,zero_oracle(oracle));action=action_sample_mse(real.action.float(),batch).mean();matched_action=action_sample_mse(control.action.float(),batch).mean();teammate=gaussian_nll(real.teammate_action_mean.float(),real.teammate_action_logvar.float(),batch["oracle_teammate_action"],batch["oracle_teammate_action_mask"]);delta=masked_mse(real.teammate_delta.float(),batch["teammate_delta"],batch["future_mask"]);control_teammate=gaussian_nll(control.teammate_action_mean.float(),control.teammate_action_logvar.float(),batch["oracle_teammate_action"],batch["oracle_teammate_action_mask"]);control_delta=masked_mse(control.teammate_delta.float(),batch["teammate_delta"],batch["future_mask"]);loss=weights["ego_action"]*(action+matched_action)/2+weights["teammate_action_gaussian_nll"]*(teammate+control_teammate)/2+weights["teammate_delta"]*(delta+control_delta)/2
        if not torch.isfinite(loss):raise FloatingPointError(f"non-finite teacher loss at {update}")
        loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.0);optimizer.step();scheduler.step()
        if update%100==0:atomic_json(args.output/"heartbeat.json",{"status":"TRAINING","seed":args.seed,"update":update,"loss":float(loss),"updated_at_epoch":time.time()})
        if update%R1_EVAL_EVERY:continue
        val=evaluate_teacher(backbones,base_probe,teacher,matched,validation,device);row={"update":update,"train":{"total":float(loss.detach()),"action":float(action.detach()),"matched_action":float(matched_action.detach()),"teammate_nll":float(teammate.detach()),"delta":float(delta.detach())},"validation":val,"learning_rate":scheduler.get_last_lr()[0]};metrics.append(row)
        with (args.output/"evaluations.jsonl").open("a") as stream:stream.write(json.dumps(row,sort_keys=True)+"\n")
        payload={"format_version":"before-we-act.b3-n1-r1-teacher-checkpoint/1","teacher":teacher.state_dict(),"matched":matched.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"update":update,"evaluations":metrics,"sample_cursor":sampler.cursor_receipt(update),"provenance":provenance,"base_h_selected_update":base_update};atomic_save(payload,latest);atomic_save(payload,args.output/f"checkpoint_{update:06d}.pt");print(json.dumps(row,sort_keys=True),flush=True)
        if platform(metrics):completion="PLATFORM_REACHED";break
        if update>=R1_MAX_UPDATES:completion="INCONCLUSIVE_TRAINING_NOT_CONVERGED";break
    completion=completion or "INCONCLUSIVE_TRAINING_NOT_CONVERGED";selected=min(metrics,key=lambda row:row["validation"]["macro"]["h_teacher"]);atomic_json(args.output/"status.json",{"status":completion,"seed":args.seed,"update":int(metrics[-1]["update"]),"selected_update":int(selected["update"]),"selected_validation":selected["validation"],"elapsed_hours":(time.time()-started)/3600,"completed_at_utc":utc_now()})


if __name__=="__main__":main()
