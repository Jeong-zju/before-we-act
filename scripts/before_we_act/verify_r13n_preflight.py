#!/usr/bin/env python3
"""Strict restore and causal action-effect preflight for R13N."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data._utils.collate import default_collate

from before_we_act.action_generator.r13n_baseline import R13NActionGenerator, load_r13n_config
from before_we_act.data.full_episode_windows import FullEpisodeActionWindows
from before_we_act.r13n import TASKS, sha256
from before_we_act.train_action_generator_r4 import atomic_json
from before_we_act.train_r13n_baseline import validate_index


@torch.inference_mode()
def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--checkpoint",required=True); parser.add_argument("--full-index",required=True); parser.add_argument("--device",default="cuda:0"); parser.add_argument("--output",required=True); args=parser.parse_args()
    device=torch.device(args.device); config=load_r13n_config(args.config); checkpoint_path=Path(args.checkpoint).resolve(strict=True); saved=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    if saved.get("round")!="R13N" or saved.get("model_id")!="b6_act_six_task" or int(saved.get("update",-1))!=2: raise ValueError("R13N preflight checkpoint identity differs")
    model=R13NActionGenerator(config).to(device); incompatible=model.load_state_dict(saved["model"],strict=True); model.eval()
    index_path=Path(args.full_index).resolve(strict=True); index=json.loads(index_path.read_text()); validate_index(index)
    stats={key:torch.as_tensor(saved["stats"][key],dtype=torch.float32) for key in ("a_mean","a_std")}; dataset=FullEpisodeActionWindows(index["episodes"],stats,split="train",cache_episodes=2,tasks=TASKS)
    requests=[dataset.requests_by_task[0][0],dataset.requests_by_task[len(TASKS)-1][0]]; cpu=default_collate([dataset[request] for request in requests]); batch={key:value.to(device) for key,value in cpu.items()}
    with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
        base=model.sample(batch,spatial_tokens=batch["spatial_tokens"],spatial_view_mask=batch["spatial_view_mask"],task_index=batch["task_index"])
        shuffled=model.sample(batch,spatial_tokens=batch["spatial_tokens"].flip(2),spatial_view_mask=batch["spatial_view_mask"],task_index=batch["task_index"])
        changed_task=model.sample(batch,spatial_tokens=batch["spatial_tokens"],spatial_view_mask=batch["spatial_view_mask"],task_index=(batch["task_index"]+1)%len(TASKS))
    spatial_effect=float((base.actions-shuffled.actions).abs().mean().cpu()); task_effect=float((base.actions-changed_task.actions).abs().mean().cpu())
    result={"schema_version":1,"round":"R13N","checkpoint":str(checkpoint_path),"checkpoint_sha256":sha256(checkpoint_path),"strict_restore":not incompatible.missing_keys and not incompatible.unexpected_keys,"all_finite":bool(torch.isfinite(base.actions).all()),"absent_agents_zero":bool((base.actions[:,:,2:]==0).all()),"spatial_action_effect_l1":spatial_effect,"task_action_effect_l1":task_effect,"spatial_action_effect":spatial_effect>0,"task_action_effect":task_effect>0,"historical_checkpoint_loaded":False,"passed":False,"created_at_epoch":time.time()}
    result["passed"]=all((result["strict_restore"],result["all_finite"],result["absent_agents_zero"],result["spatial_action_effect"],result["task_action_effect"],not result["historical_checkpoint_loaded"]))
    atomic_json(Path(args.output),result); print(json.dumps(result,sort_keys=True)); raise SystemExit(0 if result["passed"] else 10)


if __name__=="__main__": main()
