"""Strict-local MARS closed-loop evaluator for the official CARE belief policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from before_we_act.mars_action_contract import (
    ACTION_CONTRACT_VERSION,
    canonicalize_action,
    normalization_stats_hash,
    validate_action_space_bounds,
    validate_checkpoint_action_contract,
)
from before_we_act.mars_temporal_data import validate_mars_normalization
from before_we_act.evaluate_mars_temporal_policy import ChunkEnsembler, LocalHistory, scalar
from before_we_act.predictive_team_belief_policy import PredictiveTeamBeliefPolicy
from before_we_act.team_belief.predictive_core import TeamBeliefConfig
from deployment.mars_care.common import TASK_BY_NAME, local_observation, make_env


def load_policy(checkpoint:Path,device:torch.device):
    saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if saved.get("format_version")!="before-we-act.mars-n2-deployment-checkpoint/1": raise ValueError("wrong MARS N2 checkpoint")
    contract=validate_checkpoint_action_contract(saved)
    validate_mars_normalization(saved.get("stats"))
    expected_normalization=normalization_stats_hash(saved["stats"])
    if contract.get("annotations",{}).get("normalization_sha256")!=expected_normalization:
        raise ValueError("MARS N2 checkpoint action statistics hash differs")
    config=saved["config"]; values=dict(config["n2_config"])
    if config.get("action_contract_version")!=ACTION_CONTRACT_VERSION:
        raise ValueError("MARS N2 checkpoint config predates the shared action contract")
    for key in ("future_offsets_steps","future_offsets_seconds"):
        if key in values: values[key]=tuple(values[key])
    model=PredictiveTeamBeliefPolicy(
        TeamBeliefConfig(**values),state_dim=9,action_dim=8,horizon=100,d_model=384,
        enc_layers=4,dec_layers=7,roles=4,role_rank=32,history_layers=2,
        dino_model=str(config["dino_model"]),image_height=int(config["image_height"]),
        image_width=int(config["image_width"]),include_teacher=False,
        residual_safety=config.get("residual_safety"),
    ).to(device)
    model.load_state_dict(saved["model"],strict=True); model.eval()
    stats={key:torch.as_tensor(saved["stats"][key],device=device,dtype=torch.float32) for key in ("q_mean","q_std","a_mean","a_std")}
    return model,stats,config


@torch.inference_mode()
def run_episode(model,stats,task,root,seed,device,max_steps,decay):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    path=str(root.resolve());
    if path not in sys.path: sys.path.insert(0,path)
    env=make_env(task,root); observation,_=env.reset(seed=seed); arms=tuple(range(task.arms))
    history=LocalHistory(arms); ensemble=ChunkEnsembler(arms,decay); success=False; trace=hashlib.sha256()
    movement=np.zeros(len(arms)); inference=[]; diagnostics=[]
    try:
        for step in range(max_steps):
            images=[]; qposes=[]
            for arm in arms:
                image,qpos=local_observation(observation,arm); images.append(image); qposes.append(qpos.reshape(-1))
            rgb=torch.as_tensor(np.stack(images),device=device).permute(0,3,1,2).float().div_(255)
            qpos=torch.as_tensor(np.stack(qposes),device=device).float(); qnorm=(qpos-stats["q_mean"])/stats["q_std"]
            temporal=history.batch(qnorm,task.name,device,True); started=time.perf_counter()
            with torch.autocast("cuda",dtype=torch.bfloat16): output=model(rgb,rgb,**temporal)
            inference.append(time.perf_counter()-started); history.append_observation(output.current_visual_raw,qnorm)
            chunks=(output.prediction*stats["a_std"]+stats["a_mean"]).float().cpu().numpy()
            selected=ensemble.select(step,chunks); action={}; normalized={}
            for arm,row in selected.items():
                space=env.action_space.spaces[f"panda-{arm}"]; validate_action_space_bounds(space); row=canonicalize_action(row)
                action[f"panda-{arm}"]=row; trace.update(row.tobytes()); movement[arm]+=float(np.linalg.norm(row[:7]-qposes[arm][:7]))
                normalized[arm]=(torch.as_tensor(row,device=device)-stats["a_mean"])/stats["a_std"]
            history.append_action(normalized)
            diagnostics.append({"gate":float(output.residual_gate.float().mean()),
                "residual_norm":float(output.belief_residual.float().norm(dim=-1).mean()),
                "reliability":float(output.belief.reliability.float().mean()),"sigma":float(output.belief.sigma.float().mean())})
            observation,_reward,terminated,truncated,info=env.step(action); success=scalar(info.get("success",False))
            if success or scalar(terminated) or scalar(truncated): break
    finally: env.close()
    return {"task":task.name,"seed":seed,"success":success,"steps":step+1,
        "mean_inference_seconds":float(np.mean(inference)),"action_trace_sha256":trace.hexdigest(),
        "mean_command_delta_by_arm":(movement/(step+1)).tolist(),
        "belief_diagnostics":{key:float(np.mean([row[key] for row in diagnostics])) for key in diagnostics[0]}}


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--task",choices=tuple(TASK_BY_NAME),required=True); parser.add_argument("--robofactory-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True); parser.add_argument("--episodes",type=int,default=20)
    parser.add_argument("--seed-start",type=int,default=20260827); parser.add_argument("--max-steps",type=int)
    parser.add_argument("--ensemble-decay",type=float,default=.01); parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args(); device=torch.device(args.device); model,stats,config=load_policy(args.checkpoint,device); task=TASK_BY_NAME[args.task]
    log=args.output.with_suffix(".jsonl"); recovered={}
    if log.is_file():
        for line in log.read_text().splitlines():
            try: row=json.loads(line); recovered[int(row["seed"])]=row
            except Exception: pass
    rows=[]
    for seed in range(args.seed_start,args.seed_start+args.episodes):
        row=recovered.get(seed) or run_episode(model,stats,task,args.robofactory_root,seed,device,args.max_steps or task.max_steps,args.ensemble_decay)
        rows.append(row)
        if seed not in recovered:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            with log.open("a") as stream: stream.write(json.dumps(row)+"\n")
        print(json.dumps(row),flush=True)
    result={"status":"complete","policy":"official_care_predictive_team_belief","strict_local":True,
        "per_robot_independent_inputs":True,"task":task.name,"episodes":len(rows),
        "successes":sum(int(row["success"]) for row in rows),"success_rate":float(np.mean([row["success"] for row in rows])),
        "rows":rows,"config":config}
    tmp=args.output.with_suffix(".tmp"); tmp.write_text(json.dumps(result,indent=2)+"\n"); os.replace(tmp,args.output)


if __name__=="__main__": main()
